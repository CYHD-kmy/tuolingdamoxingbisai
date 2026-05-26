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
from ...utils.config import get_config

logger = logging.getLogger(__name__)

LOT_SIZE = 100  # A股最小交易单位


class RiskManager:
    """
    风控主管 — 所有计算均为确定性规则。

    使用方式:
        rm = RiskManager(total_capital=500_000)
        limits = rm.compute_limits(verdicts, daily_data, current_portfolio)
    """

    def __init__(self, total_capital: float = 500_000.0) -> None:
        self._capital = total_capital
        self._cfg = get_config()

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
            base_pct = self._cfg.max_single_position

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
            final_pct = min(final_pct, self._cfg.max_single_position)  # 硬上限 20%

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
                if industry_current / self._capital > self._cfg.max_industry_exposure:
                    final_pct *= 0.5
                    risk_flags.append(f"行业 {industry} 集中度超标")

            # 相关性惩罚: 与已持仓股票高相关 (ρ > 0.7) → ×0.7
            if current_positions:
                for held_code, held_shares in current_positions.items():
                    if held_shares <= 0:
                        continue
                    corr = self._calc_correlation(v.code, held_code, daily_data)
                    if corr > 0.70:
                        final_pct *= 0.7
                        risk_flags.append(f"与 {held_code} 高相关 (ρ={corr:.2f})")
                        break  # 只惩罚一次，取最高相关性

            # 日换手率检查
            turnover_warning = self._check_turnover(v.code, daily_data)
            if turnover_warning:
                risk_flags.append(turnover_warning)

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
        if drawdown >= self._cfg.max_drawdown_daily:
            logger.warning("触发日内熔断! 回撤: %.2f%%", drawdown * 100)
            return True
        return False

    # ── 辅助 ──────────────────────────────────

    @staticmethod
    def _calc_correlation(code_a: str, code_b: str, daily_data: dict[str, list[Any]]) -> float:
        """计算两只股票的 Pearson 相关系数 (基于近20日收盘价涨跌幅)"""
        records_a = daily_data.get(code_a, [])
        records_b = daily_data.get(code_b, [])
        if len(records_a) < 10 or len(records_b) < 10:
            return 0.0
        # 取两者共有的最短长度
        n = min(len(records_a), len(records_b), 20)
        pcts_a = [records_a[-(n - i)].pct_chg for i in range(n)]
        pcts_b = [records_b[-(n - i)].pct_chg for i in range(n)]
        return _pearson(pcts_a, pcts_b)

    @staticmethod
    def _check_turnover(code: str, daily_data: dict[str, list[Any]]) -> str:
        """检查日换手率是否超限，返回警告信息或空字符串"""
        records = daily_data.get(code, [])
        if not records:
            return ""
        latest = records[-1]
        turnover = getattr(latest, "turnover", 0.0)
        if turnover > 50.0:
            return f"日换手率 {turnover:.1f}% 过高，流动性风险"
        return ""

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


def _pearson(x: list[float], y: list[float]) -> float:
    """计算 Pearson 相关系数"""
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_x = (sum((xi - mean_x) ** 2 for xi in x) ** 0.5)
    std_y = (sum((yi - mean_y) ** 2 for yi in y) ** 0.5)
    if std_x < 1e-9 or std_y < 1e-9:
        return 0.0
    return round(cov / (std_x * std_y), 4)
