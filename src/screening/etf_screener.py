"""
ETF 筛选器 — 从全市场 ETF 中选出具备交易价值的标的。

筛选逻辑:
- 按成交额排除流动性不足的 ETF (> 5000万)
- 折溢价检查 (折价 ETF 有安全边际)
- 趋势评分
- 基金规模阈值

使用方式:
    from src.screening.etf_screener import ETFScreener
    screener = ETFScreener(data)
    candidates = screener.screen(top_n=10)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..data.interface import UnifiedDataInterface
from ..data.fetchers.akshare_fetcher import ETFSpot, StockDaily

logger = logging.getLogger(__name__)

# ETF 筛选参数
MIN_DAILY_AMOUNT = 50_000_000   # 最小成交额 5000万
MIN_FUND_SIZE = 100_000_000     # 最小基金规模 1亿
MAX_DISCOUNT = -5.0             # 折价 > 5% 可能有坑
MAX_PREMIUM = 3.0               # 溢价 > 3% 追高风险


@dataclass
class ETFCandidate:
    """ETF 候选标的"""
    code: str
    name: str
    price: float
    pct_chg: float
    amount: float             # 成交额 (元)
    fund_size: float          # 基金规模 (亿)
    discount: float           # 折溢价 (%)
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)


class ETFScreener:
    """ETF 海选筛选器"""

    def __init__(self, data: UnifiedDataInterface) -> None:
        self._data = data

    def screen(self, top_n: int = 10) -> list[ETFCandidate]:
        """
        执行 ETF 筛选。

        返回: 按综合得分降序的候选列表
        """
        etf_list = self._data.get_etf_spot()
        if not etf_list:
            logger.warning("ETF 行情获取失败")
            return []

        candidates: list[ETFCandidate] = []

        for e in etf_list:
            # 1. 流动性过滤
            if e.amount < MIN_DAILY_AMOUNT:
                continue

            # 2. 规模过滤
            if e.fund_size > 0 and e.fund_size < MIN_FUND_SIZE / 1e8:
                continue

            # 3. 价格有效性
            if e.price <= 0.01:
                continue

            # 4. 评分
            score = 50.0
            reasons = []

            # 成交额
            amount_yi = e.amount / 1e8
            if amount_yi >= 5:
                score += 15
            elif amount_yi >= 1:
                score += 8
                reasons.append(f"成交额 {amount_yi:.1f}亿")

            # 折溢价 (考虑 IOPV 估算)
            discount = 0.0  # ETFSpot 通常不含 IOPV，以后续数据增强
            if hasattr(e, "discount") and e.discount:
                discount = e.discount
                if -2.0 <= discount <= 0.5:
                    score += 12
                    reasons.append(f"小幅折价 {discount:+.2f}%")
                elif discount < MAX_DISCOUNT:
                    score -= 15
                    reasons.append(f"深度折价 {discount:.1f}%")

            # 近期趋势
            if e.pct_chg > 0:
                score += min(10, e.pct_chg * 2)
            elif e.pct_chg < -1:
                score += max(-15, e.pct_chg * 3)

            # 换手率适中
            if e.turnover > 0:
                if 2 <= e.turnover <= 10:
                    score += 8
                elif e.turnover > 20:
                    score -= 5

            # 规模加分
            if e.fund_size > 5:
                score += 8

            candidates.append(ETFCandidate(
                code=e.code,
                name=e.name,
                price=e.price,
                pct_chg=e.pct_chg,
                amount=e.amount,
                fund_size=e.fund_size,
                discount=discount,
                score=min(100, max(0, score)),
                reasons=reasons,
            ))

        candidates.sort(key=lambda x: x.score, reverse=True)
        top = candidates[:top_n]

        logger.info("ETF筛选: %d → %d", len(etf_list), len(top))
        for i, c in enumerate(top, 1):
            logger.info("  %2d. %s %s 评分:%.0f", i, c.code, c.name, c.score)

        return top
