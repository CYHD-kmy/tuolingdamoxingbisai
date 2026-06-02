"""
组合主管 — deep LLM，做出最终买卖决策。

输入: 研究结论 + 风控约束 + 当前持仓 + 可用资金
输出: [{symbol, symbol_name, volume}] 赛道标准 JSON 格式
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..models import FinalDecision, PortfolioResult, ResearchVerdict, PositionLimit, TargetAllocation
from ...llm.client import LLMClient
from ...llm.schema import Message
from ...utils.validators import validate_and_clip, get_latest_price, extract_json
from ...utils.config import get_config

logger = logging.getLogger(__name__)

PORTFOLIO_MANAGER_PROMPT = """你是一位 A 股投资组合主管，负责做出最终买卖决策。

## 背景
- 初始资金 50 万元人民币
- A 股 T+1 交易规则 (今日买入次日才能卖出，当天不能卖当天买的)
- 最小交易单位 100 股 (1手)
- 你的目标是在控制风险的前提下追求绝对收益
- 你需要同时管理现有持仓和新候选标的

## 决策原则
1. **审视持仓**: 首先检查当前持仓，判断哪些该持有/减仓/清仓
2. **集中投资**: 3-5 只股票即可，过度分散会稀释收益
3. **仓位管理**: 单票不超过风控给你设定的上限
4. **现金保留**: 至少保留 10% 现金应对机会或风险
5. **空仓也是一种策略**: 如果所有标的置信度都很低，输出空数组 []
6. **优胜劣汰**: 如果候选标的明显优于现有持仓，应卖出弱股换强股

## 卖出规则 (重要)
对已持仓股票，满足以下条件应考虑卖出:
- 持有天数过长(>8天)且仍然亏损或微利(<3%)
- 触发止损信号 (系统会在上下文中标注 ⚠️)
- 基本面恶化或置信度大幅下降
- 发现更好的投资机会需要释放资金

## 输出格式
请严格按照以下 JSON 数组格式输出 (无其他文字):
```json
[
  {
    "symbol": "600519",
    "symbol_name": "贵州茅台",
    "volume": 200,
    "direction": "buy"
  },
  {
    "symbol": "000858",
    "symbol_name": "五粮液",
    "volume": 300,
    "direction": "sell"
  }
]
```

注意事项:
- direction 为 "buy" 表示买入, "sell" 表示卖出
- volume 必须是 100 的整数倍
- 卖出时 volume 不能超过当前持有股数
- 总买入金额不得超过可用资金
- 按优先级排序: 先卖出(释放资金)再买入(使用资金)
- 如果当日不适合买入任何标的，输出空数组 []
- 不要在 JSON 外添加任何解释文字
"""

PORTFOLIO_ALLOCATION_PROMPT = """你是一位 A 股投资组合主管，负责分配目标仓位权重。

## 背景
- 初始资金 50 万元人民币
- A 股 T+1 交易规则
- 你的目标是在控制风险的前提下追求绝对收益

## 决策原则
1. 为每只标的(现有持仓 + 候选)分配一个目标权重 (0.0 ~ 0.30)
2. 0% 表示清仓该标的
3. 总权重之和不超过 90% (保留 10% 现金)
4. 对已持仓的股票: 根据当前盈亏、持有天数、最新研判决定增减
5. 对候选股票: 仅对高置信度的分配权重
6. 优先保留盈利中的强势股，果断替换弱势持仓

## 输出格式
请严格按照以下 JSON 数组格式输出 (无其他文字):
```json
[
  {
    "code": "600519",
    "name": "贵州茅台",
    "target_weight": 0.20,
    "reasoning": "强势整理，继续持有"
  },
  {
    "code": "300750",
    "name": "宁德时代",
    "target_weight": 0.0,
    "reasoning": "趋势走弱，建议清仓"
  }
]
```

注意事项:
- target_weight 是占总权益的百分比 (0.0=清仓, 0.20=20%仓位)
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
        portfolio_context: dict[str, Any] | None = None,
        market_regime: str = "neutral",
    ) -> PortfolioResult:
        """
        构建最终投资组合。

        verdicts: 研究主管对每只股票的研判
        limits: 风控主管给出的仓位上限
        daily_data: 日线数据
        cash_available: 当前可用资金
        total_capital: 总资金
        portfolio_context: 当前持仓详解 (由 PortfolioTracker.build_context() 生成)
        market_regime: 市场环境 "bull"/"neutral"/"bear"

        LLM 不可用时自动降级为确定性规则。
        """
        try:
            cfg = get_config()
            if cfg.use_target_allocation:
                return self._construct_via_allocation(
                    verdicts, limits, daily_data, cash_available, total_capital,
                    portfolio_context, market_regime,
                )
            return self._construct_with_llm(
                verdicts, limits, daily_data, cash_available, total_capital,
                portfolio_context, market_regime,
            )
        except Exception as e:
            logger.warning("PortfolioManager: LLM 调用失败 (%s)，降级为确定性规则", e)
            from ..fallback import fallback_portfolio
            return fallback_portfolio(
                verdicts, limits, daily_data, cash_available, total_capital,
                portfolio_context, market_regime,
            )

    def _construct_with_llm(
        self,
        verdicts: list[ResearchVerdict],
        limits: dict[str, PositionLimit],
        daily_data: dict[str, list],
        cash_available: float,
        total_capital: float,
        portfolio_context: dict[str, Any] | None = None,
        market_regime: str = "neutral",
    ) -> PortfolioResult:
        """LLM 驱动的组合构建 (含买卖双向决策)"""
        # 筛选可操作标的: buy 信号 + 有风控上界
        buy_candidates = [
            (v, limits[v.code])
            for v in verdicts
            if v.direction == "buy" and v.code in limits and limits[v.code].max_shares > 0
        ]

        # 构建上下文 (含持仓信息)
        context = self._build_context(
            buy_candidates, daily_data, cash_available, total_capital,
            portfolio_context, market_regime,
        )

        # 如果既无候选又无持仓，直接返回空仓
        has_holdings = portfolio_context and portfolio_context.get("position_count", 0) > 0
        if not buy_candidates and not has_holdings:
            logger.info("PortfolioManager: 无可买入标的且无持仓，输出空仓")
            return PortfolioResult(decisions=[])

        buy_candidates.sort(key=lambda x: x[0].confidence, reverse=True)

        messages = [
            Message(role="system", content=PORTFOLIO_MANAGER_PROMPT),
            Message(role="user", content=context),
        ]

        resp = self._llm.chat(messages)
        decisions = self._parse_decisions(resp.content, daily_data)

        # 分离买卖决策
        buy_decisions = [d for d in decisions if d.direction == "buy"]
        sell_decisions = [d for d in decisions if d.direction == "sell"]

        # 校验买入决策
        verdict_map = {v.code: v for v in verdicts}
        if buy_decisions:
            buy_decisions = self._validate(
                buy_decisions, limits, daily_data, cash_available, total_capital, verdict_map,
            )

        # 校验卖出决策 (不消耗现金，检查是否在持仓中)
        if sell_decisions and portfolio_context:
            sell_decisions = self._validate_sells(sell_decisions, portfolio_context)

        all_decisions = sell_decisions + buy_decisions
        buy_cash_used = sum(
            d.volume * self._get_price(d.symbol, daily_data)
            for d in buy_decisions
        )
        sell_proceeds = sum(
            d.volume * self._get_price(d.symbol, daily_data)
            for d in sell_decisions
        )
        cash_remaining = cash_available + sell_proceeds - buy_cash_used

        logger.info(
            "PortfolioManager: 最终决策 %d 笔 (买%d卖%d), 买入用 ¥%.0f, 卖出收 ¥%.0f",
            len(all_decisions), len(buy_decisions), len(sell_decisions),
            buy_cash_used, sell_proceeds,
        )

        return PortfolioResult(
            decisions=all_decisions,
            cash_used=round(buy_cash_used, 2),
            cash_remaining=round(cash_remaining, 2),
            sell_proceeds=round(sell_proceeds, 2),
            total_positions=len(all_decisions),
            risk_summary=f"market: {market_regime}",
        )

    # ── 内部 ──────────────────────────────────

    def _build_context(
        self,
        candidates: list[tuple[ResearchVerdict, PositionLimit]],
        daily_data: dict[str, list],
        cash_available: float,
        total_capital: float,
        portfolio_context: dict[str, Any] | None = None,
        market_regime: str = "neutral",
    ) -> str:
        regime_labels = {"bull": "牛市(强势)", "neutral": "震荡市(中性)", "bear": "熊市(弱势)"}
        regime_label = regime_labels.get(market_regime, "未知")

        total_equity = total_capital
        if portfolio_context:
            total_equity = portfolio_context.get("total_equity", total_capital)

        lines = [
            f"## 账户状态",
            f"总权益: ¥{total_equity:,.0f}  可用资金: ¥{cash_available:,.0f}",
            f"市场环境: {regime_label}",
            f"最低现金保留: ¥{total_capital * 0.10:,.0f} (10%)",
        ]

        # 当前持仓详情
        if portfolio_context and portfolio_context.get("position_count", 0) > 0:
            lines.append("")
            lines.append(f"## 当前持仓 (共 {portfolio_context['position_count']} 只)")
            lines.append(f"持仓总市值: ¥{portfolio_context.get('market_value', 0):,.0f}  累计收益: {portfolio_context.get('total_return_pct', 0):+.1f}%")
            lines.append("")

            for pos in portfolio_context.get("positions", []):
                flags = []
                if pos.get("stop_triggered"):
                    flags.append("止损触发!")
                if pos.get("is_stale"):
                    flags.append(f"建议{pos.get('stale_action', '处理')}")
                status = " ⚠️ " + " ".join(flags) if flags else ""

                lines.append(
                    f"- {pos['name']} ({pos['code']}): "
                    f"{pos['shares']}股 @ 成本¥{pos['avg_cost']:.2f} / 现价¥{pos['last_price']:.2f} | "
                    f"市值¥{pos['market_value']:,.0f} ({pos.get('weight_in_equity', 0):.1f}%) | "
                    f"盈亏{pos['pnl_pct']:+.1f}% | "
                    f"持有{pos['holding_days']}天"
                    f"{status}"
                )
            lines.append("")

        # Stale positions alerts
        if portfolio_context:
            stale = portfolio_context.get("stale_positions", [])
            if stale:
                lines.append("## ⚠️ 需要处理的持仓 (持有过久或收益不达标)")
                for s in stale:
                    lines.append(f"- {s['code']} {s['name']}: {s['reason']} → 建议{s['action']}")
                lines.append("")

        # 候选标的
        lines.append(f"## 候选标的 (共 {len(candidates)} 只，按置信度降序)")
        if not candidates:
            lines.append("(无新的买入候选)")

        for v, limit in candidates:
            price = self._get_price(v.code, daily_data)
            held_info = ""
            if portfolio_context:
                held = next((p for p in portfolio_context.get("positions", []) if p["code"] == v.code), None)
                if held:
                    held_info = f"[已持有 {held['shares']}股, 盈亏{held['pnl_pct']:+.1f}%]"

            max_value_desc = f"¥{limit.max_value:,.0f}" if limit.max_value > 0 else "不可买入"
            lines.append(f"""
### {v.name} ({v.code}) {held_info}
- 最新价: ¥{price:.2f}  置信度: {v.confidence:.0%}  风险: {v.risk_level}
- 仓位上限: {limit.max_position_pct:.0%} ({max_value_desc})
- 核心理由: {v.core_reasoning}
""")

        lines.append("请给出买卖决策 (JSON 数组格式，direction 为 buy 或 sell)。如果不操作，输出 []。")
        return "\n".join(lines)

    @staticmethod
    def _get_price(code: str, daily_data: dict[str, list]) -> float:
        return get_latest_price(code, daily_data)

    @staticmethod
    def _parse_decisions(raw: str, daily_data: dict[str, list] | None = None) -> list[FinalDecision]:
        """解析 LLM 输出的 JSON 决策，并填充入场价格和方向"""
        try:
            data = json.loads(extract_json(raw))
            if not isinstance(data, list):
                return []
            result = []
            for d in data:
                code = str(d.get("symbol", ""))
                entry_price = get_latest_price(code, daily_data or {})
                direction = str(d.get("direction", "buy")).lower()
                if direction not in ("buy", "sell"):
                    direction = "buy"
                result.append(FinalDecision(
                    symbol=code,
                    symbol_name=str(d.get("symbol_name", "")),
                    volume=int(d.get("volume", 0)),
                    entry_price=entry_price,
                    direction=direction,
                ))
            return result
        except (json.JSONDecodeError, ValueError, KeyError, TypeError, AttributeError):
            logger.warning("PortfolioManager: JSON 解析失败")
            return []

    @staticmethod
    def _parse_allocations(raw: str) -> list[TargetAllocation]:
        """解析 LLM 输出的目标仓位 JSON"""
        try:
            data = json.loads(extract_json(raw))
            if not isinstance(data, list):
                return []
            return [
                TargetAllocation(
                    code=str(d.get("code", "")),
                    name=str(d.get("name", "")),
                    target_weight=float(d.get("target_weight", 0)),
                    confidence=float(d.get("confidence", 0)),
                    reasoning=str(d.get("reasoning", "")),
                )
                for d in data
            ]
        except (json.JSONDecodeError, ValueError, KeyError, TypeError, AttributeError):
            logger.warning("PortfolioManager: 目标仓位 JSON 解析失败")
            return []

    @staticmethod
    def _validate_sells(
        decisions: list[FinalDecision],
        portfolio_context: dict[str, Any],
    ) -> list[FinalDecision]:
        """校验卖出决策：是否在持仓中、是否超过持有股数"""
        positions = {
            p["code"]: p["shares"]
            for p in portfolio_context.get("positions", [])
        }
        valid = []
        for d in decisions:
            if d.code not in positions:
                logger.warning("PortfolioManager: 卖出 %s 不在持仓中，跳过", d.code)
                continue
            max_sell = positions[d.code]
            if d.volume > max_sell:
                logger.warning(
                    "PortfolioManager: 卖出 %s %d > 持有 %d，裁剪至 %d",
                    d.code, d.volume, max_sell, max_sell,
                )
                d.volume = (max_sell // 100) * 100
            if d.volume >= 100:
                valid.append(d)
        return valid

    def _construct_via_allocation(
        self,
        verdicts: list[ResearchVerdict],
        limits: dict[str, PositionLimit],
        daily_data: dict[str, list],
        cash_available: float,
        total_capital: float,
        portfolio_context: dict[str, Any] | None = None,
        market_regime: str = "neutral",
    ) -> PortfolioResult:
        """
        目标仓位模式: LLM 输出每只标的的目标权重，系统自动计算买卖量。
        """
        # 构建上下文
        buy_candidates = [
            (v, limits[v.code])
            for v in verdicts
            if v.direction == "buy" and v.code in limits and limits[v.code].max_shares > 0
        ]
        buy_candidates.sort(key=lambda x: x[0].confidence, reverse=True)

        has_holdings = portfolio_context and portfolio_context.get("position_count", 0) > 0
        if not buy_candidates and not has_holdings:
            return PortfolioResult(decisions=[])

        context = self._build_context(
            buy_candidates, daily_data, cash_available, total_capital,
            portfolio_context, market_regime,
        )

        # 告诉 LLM 这是目标仓位模式
        context += "\n\n⚠️ 当前使用目标仓位模式：请为每只标的分配目标权重 (0.0~0.30)，而非具体的买卖股数。"

        messages = [
            Message(role="system", content=PORTFOLIO_ALLOCATION_PROMPT),
            Message(role="user", content=context),
        ]

        resp = self._llm.chat(messages)
        allocations = self._parse_allocations(resp.content)

        if not allocations:
            logger.info("PortfolioManager(Alloc): LLM 未返回有效目标仓位")
            return PortfolioResult(decisions=[])

        # 用 AllocationCalculator 将目标权重转为买卖决策
        from ..allocation_calculator import AllocationCalculator

        equity = portfolio_context.get("total_equity", total_capital) if portfolio_context else total_capital
        current_positions = {}
        if portfolio_context:
            for pos in portfolio_context.get("positions", []):
                from ..portfolio_tracker import Position
                current_positions[pos["code"]] = Position(
                    code=pos["code"], name=pos["name"],
                    shares=pos["shares"], avg_cost=pos["avg_cost"],
                    entry_date=pos.get("entry_date", ""),
                    last_price=pos["last_price"],
                    industry=pos.get("industry", ""),
                    asset_type=pos.get("asset_type", "stock"),
                )

        decisions = AllocationCalculator.compute_trades(
            allocations, current_positions, cash_available,
            daily_data, equity, limits,
        )

        buy_decisions = [d for d in decisions if d.direction == "buy"]
        buy_cash = sum(d.volume * get_latest_price(d.symbol, daily_data) for d in buy_decisions)
        sell_decisions = [d for d in decisions if d.direction == "sell"]
        sell_cash = sum(d.volume * get_latest_price(d.symbol, daily_data) for d in sell_decisions)

        logger.info(
            "PortfolioManager(Alloc): %d 笔决策 (目标%d, 买%d卖%d), 买¥%.0f 卖¥%.0f",
            len(decisions), len(allocations) - len(decisions), len(buy_decisions), len(sell_decisions),
            buy_cash, sell_cash,
        )

        return PortfolioResult(
            decisions=decisions,
            target_allocations=allocations,
            cash_used=round(buy_cash, 2),
            cash_remaining=round(cash_available + sell_cash - buy_cash, 2),
            sell_proceeds=round(sell_cash, 2),
            total_positions=len(decisions),
            risk_summary=f"allocation: {len(allocations)} targets, market: {market_regime}",
        )

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
