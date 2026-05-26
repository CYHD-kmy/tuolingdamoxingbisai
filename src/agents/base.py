"""
分析师基础模块 — 统一的报告模型和分析器基类。

设计模式: 策略模式
- 每个分析师实现相同的 analyze() 接口
- BaseAnalyst 提供 Tool Calling 循环骨架
- 子类只需定义 system_prompt 和工具列表
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..llm.schema import Message, Tool, ToolResult
from ..llm.client import LLMClient
from ..utils.validators import extract_json

logger = logging.getLogger(__name__)


# ── 分析报告 (统一输出格式) ──────────────────

@dataclass
class AnalystReport:
    """四位分析师统一输出的分析报告"""

    analyst_type: str          # "technical" / "fundamentals" / "fund_flow" / "news"
    code: str
    name: str                  # 股票名称
    signal: str                # "bullish" / "bearish" / "neutral"
    confidence: float          # 0.0 ~ 1.0
    reasoning: str             # 核心推理逻辑 (200字以内)
    key_points: list[str] = field(default_factory=list)   # 关键发现
    risks: list[str] = field(default_factory=list)         # 风险提示
    raw_response: str = ""     # LLM 原始输出 (调试用)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> AnalystReport:
        return cls(
            analyst_type=data.get("analyst_type", ""),
            code=data.get("code", ""),
            name=data.get("name", ""),
            signal=data.get("signal", "neutral"),
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
            key_points=data.get("key_points", []),
            risks=data.get("risks", []),
            raw_response=json.dumps(data, ensure_ascii=False),
        )


# ── 分析器基类 ────────────────────────────────

class BaseAnalyst(ABC):
    """
    分析师基类。

    子类需要:
    1. 设置 analyst_type 类属性
    2. 实现 system_prompt 属性 (返回角色描述)
    3. 实现 tools 属性 (返回可用工具列表)
    4. 实现 build_context(code, data) → str (构建分析上下文)

    使用方式:
        analyst = TechnicalAnalyst(llm_client, data_interface)
        report = analyst.analyze("600519")
    """

    analyst_type: str = ""

    def __init__(self, llm: LLMClient, data: Any) -> None:
        """
        llm:  quick LLM 客户端
        data: UnifiedDataInterface 实例
        """
        self._llm = llm
        self._data = data

    # ── 子类必须实现 ──────────────────────────

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """角色定义 + 分析框架 + 输出格式要求"""
        ...

    @property
    @abstractmethod
    def tools(self) -> list[Tool]:
        """分析师可用的工具列表"""
        ...

    @abstractmethod
    def build_context(self, code: str) -> str:
        """构建分析所需的上下文数据 (股票代码 → 文本描述)"""
        ...

    # ── 分析流程 ──────────────────────────────

    def analyze(self, code: str) -> AnalystReport:
        """
        执行分析: 上下文 → LLM(Tool Calling) → 解析 → 报告。
        先使用系统提示词和数据上下文请求 LLM 进行分析。
        如果模型判断需要更多数据，它会通过 Tool Calling 请求。
        """
        name = self._data.get_stock_name(code)

        # 构建上下文
        context = self.build_context(code)

        # 构建消息
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=f"请分析以下股票：\n\n股票代码: {code}\n股票名称: {name}\n\n{context}"),
        ]

        # 多轮工具调用循环
        all_tool_results: list[ToolResult] = []
        for round_idx in range(3):
            resp = self._llm.chat(messages, tools=self.tools if self.tools else None)

            if not resp.has_tool_calls:
                # 无工具调用 → 解析最终报告
                report = self._parse_report(resp.content, code, name)
                logger.info(
                    "%s: %s %s → %s (置信度:%.2f)",
                    self.analyst_type, code, name, report.signal, report.confidence,
                )
                return report

            # 执行工具调用
            round_results: list[ToolResult] = []
            for tc in resp.tool_calls:
                result = self._execute_tool(tc.name, tc.arguments, tc.id)
                round_results.append(result)
                all_tool_results.append(result)

            # 将工具结果反馈给 LLM
            messages.append(Message(
                role="assistant", content=resp.content or "",
                tool_calls=resp.tool_calls,
            ))
            for i, tc in enumerate(resp.tool_calls):
                if i < len(round_results):
                    tr = round_results[i]
                    messages.append(Message(
                        role="tool", content=tr.result, tool_call_id=tc.id,
                    ))
                else:
                    logger.warning("%s: tool_call[%d] 无对应执行结果，跳过", self.analyst_type, i)

            logger.debug("%s: 第%d轮工具调用完成", self.analyst_type, round_idx + 1)

        # 最后再请求一次 LLM 给出最终报告
        final_resp = self._llm.chat(messages, tools=None)
        return self._parse_report(final_resp.content, code, name)

    # ── 工具执行 ──────────────────────────────

    def _execute_tool(self, name: str, args: dict[str, Any], call_id: str = "") -> ToolResult:
        """执行数据查询工具，返回 ToolResult"""
        code = args.get("code", "")
        # 各工具声明的默认 days 值不同
        _DEFAULT_DAYS: dict[str, int] = {
            "get_daily_data": 30,
            "get_fund_flow": 5,
            "get_news": 3,
            "get_announcements": 7,
        }
        days = int(args.get("days", _DEFAULT_DAYS.get(name, 30)))

        try:
            if name == "get_daily_data":
                data = self._data.get_daily_data(code, days=days)
                limit = min(days, len(data))
                return ToolResult(
                    call_id=call_id, name=name,
                    result=json.dumps([_daily_to_dict(d) for d in data[-limit:]], ensure_ascii=False),
                )

            elif name == "get_realtime_quote":
                q = self._data.get_realtime_quote(code)
                if q is None:
                    return ToolResult(call_id=call_id, name=name, result="{}", success=False, error="无数据")
                return ToolResult(call_id=call_id, name=name, result=json.dumps({
                    "code": q.code, "name": q.name, "price": q.price,
                    "open": q.open, "high": q.high, "low": q.low,
                    "pre_close": q.pre_close, "pct_chg": q.pct_chg,
                    "volume": q.volume, "amount": q.amount,
                    "turnover": q.turnover, "volume_ratio": q.volume_ratio,
                    "pe": q.pe, "pb": q.pb, "total_mv": q.total_mv,
                }, ensure_ascii=False))

            elif name == "get_fund_flow":
                flows = self._data.get_fund_flow(code, days=days)
                result = [{"date": f.date, "main_net_inflow": f.main_net_inflow,
                           "main_pct": f.main_pct, "super_large_net": f.super_large_net,
                           "large_net": f.large_net, "medium_net": f.medium_net,
                           "small_net": f.small_net} for f in flows]
                return ToolResult(call_id=call_id, name=name, result=json.dumps(result, ensure_ascii=False))

            elif name == "get_stock_info":
                info = self._data.get_stock_info(code)
                return ToolResult(call_id=call_id, name=name, result=json.dumps(info, ensure_ascii=False))

            elif name == "get_news":
                keyword = args.get("keyword", code)
                news = self._data.get_news(keyword, days=days)
                return ToolResult(call_id=call_id, name=name, result=json.dumps(news[:10], ensure_ascii=False))

            elif name == "get_announcements":
                announcements = self._data.get_announcements(code, days=days)
                return ToolResult(call_id=call_id, name=name, result=json.dumps(announcements[:5], ensure_ascii=False))

            else:
                return ToolResult(call_id=call_id, name=name, result="{}", success=False, error=f"未知工具: {name}")

        except Exception as e:
            logger.warning("%s: 工具执行失败 %s: %s", self.analyst_type, name, e)
            return ToolResult(call_id=call_id, name=name, result="{}", success=False, error=str(e))

    # ── 报告解析 ──────────────────────────────

    def _parse_report(self, content: str, code: str, name: str) -> AnalystReport:
        """从 LLM 输出中提取 JSON 报告"""
        try:
            json_str = extract_json(content)
            data = json.loads(json_str)
            data.setdefault("analyst_type", self.analyst_type)
            data.setdefault("code", code)
            data.setdefault("name", name)
            return AnalystReport.from_json(data)
        except json.JSONDecodeError as e:
            logger.warning("%s: JSON 解析失败, 使用文本兜底: %s", self.analyst_type, e)
            return AnalystReport(
                analyst_type=self.analyst_type,
                code=code,
                name=name,
                signal=_guess_signal(content),
                confidence=0.5,
                reasoning=content[:300],
                raw_response=content,
            )


# ── 辅助函数 ──────────────────────────────────

def _daily_to_dict(d: Any) -> dict[str, Any]:
    """StockDaily → dict (挑选关键字段)"""
    return {
        "date": d.date, "open": d.open, "high": d.high, "low": d.low,
        "close": d.close, "volume": d.volume, "amount": d.amount,
        "pct_chg": d.pct_chg, "turnover": d.turnover,
        "ma5": d.ma5, "ma10": d.ma10, "ma20": d.ma20,
        "macd_dif": d.macd_dif, "macd_dea": d.macd_dea, "macd_bar": d.macd_bar,
        "rsi_6": d.rsi_6, "rsi_14": d.rsi_14,
    }


def _guess_signal(content: str) -> str:
    """从文本中推测多空信号"""
    lower = content.lower()
    if any(w in lower for w in ["看多", "买入", "bullish", "金叉", "突破", "看涨", "利好", "增持"]):
        return "bullish"
    if any(w in lower for w in ["看空", "卖出", "bearish", "死叉", "破位", "看跌", "利空", "减持", "回调"]):
        return "bearish"
    return "neutral"
