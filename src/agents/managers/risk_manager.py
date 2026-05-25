"""
风控主管 — 确定性规则计算，不消耗 LLM Token。

职责:
- 计算每只股票的最大可买仓位
- 波动率调整 + 行业集中度检查
- 日内熔断检测
"""

from __future__ import annotations

import logging
from typing import Any

from ..models import PositionLimit, ResearchVerdict

logger = logging.getLogger(__name__)

# ── 风控参数 ──────────────────────────────────

MAX_SINGLE_POSITION = 0.20       # 基准: 单票 ≤ 20%
MAX_INDUSTRY_EXPOSURE = 0.40     # 同行业 ≤ 40%
MAX_DRAWDOWN_DAILY = 0.05        # 日内熔断 5%
MIN_CASH_RESERVE = 0.10           # 保留 ≥ 10% 现金
LOT_SIZE = 100                    # A股最小交易单位


class RiskManager:
    """
    风控主管 — 所有计算均为确定性规则。

    使用方式:
        rm = RiskManager(total_capital=500_000)
        limits = rm.compute_limits(verdicts, daily_data, current_portfolio)
    """

    def __init__(self, total_capital: float = 500_000.0) -> None:
        self._capital = total_capital

    # ── 主入口 ────────────────────────────────

    def compute_limits(
        self,
        verdicts: list[ResearchVerdict],
        daily_data: dict[str, list[Any]],
        current_positions: dict[str, int],
        industry_map: dict[str, str] | None = None,
    ) -> dict[str, PositionLimit]:
        """
        为每个候选股计算仓位上限。

        verdicts: 研究主管的研判结论
        daily_data: {code: [StockDaily, ...]} 用于计算波动率
        current_positions: 当前持仓 {code: shares}
        industry_map: {code: industry_name} 可选，用于行业集中度

        返回: {code: PositionLimit}
        """
        limits: dict[str, PositionLimit] = {}

        for v in verdicts:
            # 1. 基础仓位比例
            base_pct = MAX_SINGLE_POSITION

            # 2. 波动率调整
            volatility = self._calc_volatility(v.code, daily_data)
            if volatility < 2.0:
                vol_mult = 1.25    # 低波动 → 可加仓
            elif volatility < 4.0:
                vol_mult = 1.00    # 中波动 → 标准
            else:
                vol_mult = 0.50    # 高波动 → 减仓

            # 3. 置信度调整
            conf_mult = 0.5 + v.confidence * 0.5  # 0.5 ~ 1.0

            # 4. 风险等级调整
            risk_mult = {"low": 1.0, "medium": 0.8, "high": 0.5}.get(v.risk_level, 0.8)

            # 5. 方向过滤: 非 buy 的标的仓位上限为 0
            if v.direction != "buy":
                limits[v.code] = PositionLimit(
                    code=v.code, name=v.name,
                    max_position_pct=0, max_shares=0, max_value=0,
                    volatility=volatility,
                    risk_flags=[f"方向不是buy: {v.direction}"],
                )
                continue

            # 综合计算
            final_pct = base_pct * vol_mult * conf_mult * risk_mult
            final_pct = min(final_pct, MAX_SINGLE_POSITION)  # 硬上限 20%

            # 行业集中度检查
            risk_flags = []
            if industry_map and v.code in industry_map:
                industry = industry_map[v.code]
                industry_codes = [c for c, ind in industry_map.items() if ind == industry]
                industry_current = sum(
                    self._position_value(c, s, daily_data)
                    for c, s in current_positions.items()
                    if c in industry_codes
                )
                if industry_current / self._capital > MAX_INDUSTRY_EXPOSURE:
                    final_pct *= 0.5
                    risk_flags.append(f"行业 {industry} 集中度超标")

            # 计算最大股数 (向下取 100 的整数倍)
            max_value = self._capital * final_pct
            price = self._get_latest_price(v.code, daily_data)
            if price > 0:
                max_shares = int(max_value / price / LOT_SIZE) * LOT_SIZE
            else:
                max_shares = 0

            limits[v.code] = PositionLimit(
                code=v.code, name=v.name,
                max_position_pct=round(final_pct, 4),
                max_shares=max_shares,
                max_value=round(max_value, 2),
                volatility=volatility,
                risk_flags=risk_flags,
            )

        logger.info(
            "RiskManager: %d 个标的计算完毕, %d 可买入",
            len(verdicts),
            sum(1 for l in limits.values() if l.max_shares > 0),
        )
        return limits

    # ── 日内熔断 ──────────────────────────────

    def check_drawdown(self, current_value: float) -> bool:
        """检查是否触发日内熔断"""
        drawdown = 1 - current_value / self._capital
        if drawdown >= MAX_DRAWDOWN_DAILY:
            logger.warning("触发日内熔断! 回撤: %.2f%%", drawdown * 100)
            return True
        return False

    # ── 辅助 ──────────────────────────────────

    @staticmethod
    def _calc_volatility(code: str, daily_data: dict[str, list[Any]]) -> float:
        """计算近10日平均绝对涨跌幅 (波动率代理)"""
        records = daily_data.get(code, [])
        if len(records) < 10:
            return 3.0  # 数据不足时默认中等波动
        pcts = [abs(r.pct_chg) for r in records[-10:]]
        return sum(pcts) / len(pcts)

    @staticmethod
    def _get_latest_price(code: str, daily_data: dict[str, list[Any]]) -> float:
        records = daily_data.get(code, [])
        if not records:
            return 0.0
        return records[-1].close

    @staticmethod
    def _position_value(code: str, shares: int, daily_data: dict[str, list[Any]]) -> float:
        price = RiskManager._get_latest_price(code, daily_data)
        return price * shares
