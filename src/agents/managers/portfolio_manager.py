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
        """
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
        decisions = self._parse_decisions(resp.content)

        # 校验和裁剪
        decisions = self._validate(decisions, limits, daily_data, cash_available, total_capital)

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
        records = daily_data.get(code, [])
        if not records:
            return 0.0
        return records[-1].close

    @staticmethod
    def _parse_decisions(raw: str) -> list[FinalDecision]:
        """解析 LLM 输出的 JSON 决策"""
        try:
            if "```json" in raw:
                start = raw.index("```json") + 7
                end = raw.index("```", start)
                raw = raw[start:end]
            elif "```" in raw:
                start = raw.index("```") + 3
                end = raw.index("```", start)
                raw = raw[start:end]
            if "[" in raw and "]" in raw:
                raw = raw[raw.index("["):raw.rindex("]") + 1]
            data = json.loads(raw)
            if not isinstance(data, list):
                return []
            return [
                FinalDecision(
                    symbol=str(d.get("symbol", "")),
                    symbol_name=str(d.get("symbol_name", "")),
                    volume=int(d.get("volume", 0)),
                )
                for d in data
            ]
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
    ) -> list[FinalDecision]:
        """校验并裁剪决策"""
        valid = []
        total_cost = 0.0
        min_cash = total_capital * 0.10

        for d in decisions:
            price = PortfolioManager._get_price(d.symbol, daily_data)
            if price <= 0:
                continue

            # volume 向下取整到 100 的倍数
            d.volume = d.volume // 100 * 100
            if d.volume <= 0:
                continue

            # 不超过风控上限
            limit = limits.get(d.symbol)
            if limit and d.volume > limit.max_shares:
                logger.info("PortfolioManager: %s 裁剪 %d→%d (风控上限)", d.symbol, d.volume, limit.max_shares)
                d.volume = limit.max_shares

            cost = d.volume * price
            if total_cost + cost > cash_available - min_cash:
                # 超预算，尝试缩量
                remaining = cash_available - min_cash - total_cost
                new_volume = int(remaining / price / 100) * 100
                if new_volume >= 100:
                    logger.info("PortfolioManager: %s 裁剪 %d→%d (超预算)", d.symbol, d.volume, new_volume)
                    d.volume = new_volume
                    total_cost += d.volume * price
                    valid.append(d)
                else:
                    logger.info("PortfolioManager: %s 跳过 (预算不足)", d.symbol)
                break
            else:
                total_cost += cost
                valid.append(d)

        return valid
