"""
组合主管 — deep LLM，做出最终买卖决策。

输入: 研究结论 + 风控约束 + 当前持仓 + 可用资金
输出: [{symbol, symbol_name, volume}] 赛道标准 JSON 格式
"""

from __future__ import annotations

import json
import logging

from ..models import FinalDecision, PortfolioResult, ResearchVerdict, PositionLimit
from ...llm.client import LLMClient
from ...llm.schema import Message
from ...utils.validators import validate_and_clip, get_latest_price, extract_json

logger = logging.getLogger(__name__)

PORTFOLIO_MANAGER_PROMPT = """你是一位 A 股投资组合主管，负责做出最终买卖决策。

## 背景
- 初始资金 50 万元人民币
- A 股 T+1 交易规则 (今日买入次日才能卖出)
- 最小交易单位 100 股 (1手)
- 你的目标是在控制风险的前提下追求绝对收益

## 决策原则
1. **集中投资**: 3-5 只股票即可，过度分散会稀释收益
2. **仓位管理**: 单票不超过风控给你设定的上限
3. **现金保留**: 至少保留 10% 现金应对机会或风险
4. **空仓也是一种策略**: 如果所有标的置信度都很低，输出空数组 []
5. **保守优于激进**: 不确定时宁可少买或不买

## 输出格式
请严格按照以下 JSON 数组格式输出 (无其他文字):
```json
[
  {
    "symbol": "600519",
    "symbol_name": "贵州茅台",
    "volume": 200
  },
  {
    "symbol": "000858",
    "symbol_name": "五粮液",
    "volume": 500
  }
]
```

注意事项:
- volume 必须是 100 的整数倍
- 总买入金额不得超过可用资金
- 按优先级排序: 置信度高的排在前面
- 如果当日不适合买入任何标的，输出空数组 []
- 不要在 JSON 外添加任何解释文字
"""


class PortfolioManager:
    """组合主管 — deep LLM"""

    def __init__(self, deep_llm: LLMClient) -> None:
        self._llm = deep_llm

    def construct_etf(
        self,
        verdicts: list[ResearchVerdict],
        limits: dict[str, PositionLimit],
        daily_data: dict[str, list],
        cash_available: float,
        total_capital: float = 500_000.0,
    ) -> PortfolioResult:
        """
        ETF 组合构建 — 确定性规则，不消耗 LLM Token。

        按置信度降序分配，每只 ETF 最多占 etf_max_single_position。
        """
        decisions: list[FinalDecision] = []
        remaining = cash_available

        buy_candidates = [
            v for v in verdicts
            if v.direction == "buy" and v.code in limits and limits[v.code].max_shares > 0
        ]
        buy_candidates.sort(key=lambda x: x.confidence, reverse=True)

        for v in buy_candidates:
            if remaining <= 0:
                break

            limit = limits[v.code]
            price = self._get_price(v.code, daily_data)
            if price <= 0:
                continue

            affordable = int(remaining / price / 100) * 100
            shares = min(affordable, limit.max_shares)
            if shares < 100:
                continue

            cost = shares * price
            decisions.append(FinalDecision(
                symbol=v.code,
                symbol_name=v.name,
                volume=shares,
                entry_price=price,
                asset_type="etf",
            ))
            remaining -= cost

        cash_used = sum(d.volume * d.entry_price for d in decisions)
        logger.info(
            "PortfolioManager(ETF): %d 笔决策, 使用资金 ¥%.0f/¥%.0f",
            len(decisions), cash_used, cash_available,
        )

        return PortfolioResult(
            decisions=decisions,
            cash_used=round(cash_used, 2),
            cash_remaining=round(cash_available - cash_used, 2),
            total_positions=len(decisions),
        )

    def construct(
        self,
        verdicts: list[ResearchVerdict],
        limits: dict[str, PositionLimit],
        daily_data: dict[str, list],
        cash_available: float,
        total_capital: float = 500_000.0,
    ) -> PortfolioResult:
        """
        构建最终投资组合。

        verdicts: 研究主管对每只股票的研判
        limits: 风控主管给出的仓位上限
        daily_data: 日线数据 (用于获取最新价格)
        cash_available: 当前可用资金
        total_capital: 总资金 (用于计算比例)

        LLM 不可用时自动降级为确定性规则。
        """
        try:
            return self._construct_with_llm(
                verdicts, limits, daily_data, cash_available, total_capital,
            )
        except Exception as e:
            logger.warning("PortfolioManager: LLM 调用失败 (%s)，降级为确定性规则", e)
            from ..fallback import fallback_portfolio
            return fallback_portfolio(verdicts, limits, daily_data, cash_available, total_capital)

    def _construct_with_llm(
        self,
        verdicts: list[ResearchVerdict],
        limits: dict[str, PositionLimit],
        daily_data: dict[str, list],
        cash_available: float,
        total_capital: float,
    ) -> PortfolioResult:
        """LLM 驱动的组合构建"""
        # 筛选有买入信号的标的
        buy_candidates = [
            (v, limits[v.code])
            for v in verdicts
            if v.direction == "buy" and v.code in limits and limits[v.code].max_shares > 0
        ]

        if not buy_candidates:
            logger.info("PortfolioManager: 无可买入标的，输出空仓")
            return PortfolioResult(decisions=[])

        # 按置信度降序
        buy_candidates.sort(key=lambda x: x[0].confidence, reverse=True)

        # 构建上下文
        context = self._build_context(
            buy_candidates, daily_data, cash_available, total_capital,
        )

        messages = [
            Message(role="system", content=PORTFOLIO_MANAGER_PROMPT),
            Message(role="user", content=context),
        ]

        resp = self._llm.chat(messages)
        decisions = self._parse_decisions(resp.content, daily_data)

        # 校验和裁剪 (传入 verdicts 供置信度排序)
        verdict_map = {v.code: v for v in verdicts}
        decisions = self._validate(decisions, limits, daily_data, cash_available, total_capital, verdict_map)

        cash_used = sum(
            d.volume * self._get_price(d.symbol, daily_data)
            for d in decisions
        )
        cash_remaining = cash_available - cash_used

        logger.info(
            "PortfolioManager: 最终决策 %d 笔, 使用资金 %.0f/%.0f",
            len(decisions), cash_used, cash_available,
        )

        return PortfolioResult(
            decisions=decisions,
            cash_used=round(cash_used, 2),
            cash_remaining=round(cash_remaining, 2),
            total_positions=len(decisions),
            risk_summary=f"min_cash_ratio: {cash_remaining/total_capital:.1%}",
        )

    # ── 内部 ──────────────────────────────────

    def _build_context(
        self,
        candidates: list[tuple[ResearchVerdict, PositionLimit]],
        daily_data: dict[str, list],
        cash_available: float,
        total_capital: float,
    ) -> str:
        lines = [
            f"## 账户状态",
            f"总资金: ¥{total_capital:,.0f}  可用资金: ¥{cash_available:,.0f}",
            f"最低现金保留: ¥{total_capital * 0.10:,.0f} (10%)",
            "",
            f"## 候选标的 (共 {len(candidates)} 只，按置信度降序)",
        ]

        for v, limit in candidates:
            price = self._get_price(v.code, daily_data)
            max_value_desc = f"¥{limit.max_value:,.0f}" if limit.max_value > 0 else "不可买入"
            lines.append(f"""
### {v.name} ({v.code})
- 最新价: ¥{price:.2f}  置信度: {v.confidence:.0%}  风险: {v.risk_level}
- 仓位上限: {limit.max_position_pct:.0%} ({max_value_desc})  最大{limit.max_shares}股
- 波动率: {limit.volatility:.1f}%  风控标记: {', '.join(limit.risk_flags) if limit.risk_flags else '无'}
- 核心理由: {v.core_reasoning}
- 关键风险: {'; '.join(v.key_risks[:3])}
""")

        lines.append("请给出最终买入决策 (JSON 数组格式)。如果都不适合买入，输出 []。")
        return "\n".join(lines)

    @staticmethod
    def _get_price(code: str, daily_data: dict[str, list]) -> float:
        return get_latest_price(code, daily_data)

    @staticmethod
    def _parse_decisions(raw: str, daily_data: dict[str, list] | None = None) -> list[FinalDecision]:
        """解析 LLM 输出的 JSON 决策，并填充入场价格"""
        try:
            data = json.loads(extract_json(raw))
            if not isinstance(data, list):
                return []
            result = []
            for d in data:
                code = str(d.get("symbol", ""))
                entry_price = get_latest_price(code, daily_data or {})
                result.append(FinalDecision(
                    symbol=code,
                    symbol_name=str(d.get("symbol_name", "")),
                    volume=int(d.get("volume", 0)),
                    entry_price=entry_price,
                ))
            return result
        except (json.JSONDecodeError, ValueError, KeyError):
            logger.warning("PortfolioManager: JSON 解析失败")
            return []

    @staticmethod
    def _validate(
        decisions: list[FinalDecision],
        limits: dict[str, PositionLimit],
        daily_data: dict[str, list],
        cash_available: float,
        total_capital: float,
        verdicts: dict[str, ResearchVerdict] | None = None,
    ) -> list[FinalDecision]:
        """校验并裁剪决策，委托到共享校验模块"""
        return validate_and_clip(
            decisions, limits, daily_data,
            cash_available=cash_available,
            total_capital=total_capital,
            min_cash_reserve=0.10,
            verdicts=verdicts,
        )
