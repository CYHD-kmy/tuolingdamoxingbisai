"""
ETF 筛选器 — 从全市场 ETF 中选出具备交易价值的标的。

筛选逻辑:
- 按成交额排除流动性不足的 ETF
- 折溢价检查 (折价 ETF 有安全边际)
- 趋势评分 (基于日线数据)
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
from ..utils.config import get_config

logger = logging.getLogger(__name__)


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
        self._config = get_config()

    def screen(self, top_n: int | None = None) -> list[ETFCandidate]:
        """
        执行 ETF 筛选。

        返回: 按综合得分降序的候选列表
        """
        if top_n is None:
            top_n = self._config.etf_max_candidates

        cfg = self._config
        min_amount = cfg.etf_min_daily_amount
        min_fund_size = cfg.etf_min_fund_size

        etf_list = self._data.get_etf_spot()
        if not etf_list:
            logger.warning("ETF 行情获取失败")
            return []

        # 批量获取日线数据用于趋势评分
        codes = [e.code for e in etf_list if e.amount >= min_amount]
        daily_map: dict[str, list[StockDaily]] = {}
        if codes:
            daily_map = self._data.batch_etf_daily(codes[:200], days=20)

        candidates: list[ETFCandidate] = []

        for e in etf_list:
            # 1. 流动性过滤
            if e.amount < min_amount:
                continue

            # 2. 规模过滤
            if e.fund_size > 0 and e.fund_size < min_fund_size / 1e8:
                continue

            # 3. 价格有效性
            if e.price <= 0.01:
                continue

            # 4. 评分
            score = 50.0
            reasons: list[str] = []

            # 成交额
            amount_yi = e.amount / 1e8
            if amount_yi >= 5:
                score += 15
            elif amount_yi >= 1:
                score += 8
                reasons.append(f"成交额 {amount_yi:.1f}亿")

            # 折溢价
            discount = getattr(e, "discount", 0.0) or 0.0
            if -2.0 <= discount <= 0.5:
                score += 12
                reasons.append(f"小幅折价 {discount:+.2f}%")
            elif discount < -5.0:
                score -= 15
                reasons.append(f"深度折价 {discount:.1f}%")

            # 近期趋势 (基于日线数据)
            records = daily_map.get(e.code, [])
            if records and len(records) >= 5:
                trend_score = _compute_trend_score(records)
                score += trend_score
                if trend_score > 5:
                    reasons.append(f"趋势向好 ({_trend_label(records)})")
            else:
                # 降级: 仅用当日涨跌幅
                if e.pct_chg > 0:
                    score += min(8, e.pct_chg * 2)
                elif e.pct_chg < -1:
                    score += max(-12, e.pct_chg * 3)

            # 换手率适中
            if e.turnover > 0:
                if 2 <= e.turnover <= 15:
                    score += 6
                elif e.turnover > 25:
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


def _compute_trend_score(records: list[StockDaily]) -> float:
    """基于日线数据计算趋势评分 (-20 ~ +20)"""
    if len(records) < 5:
        return 0.0

    closes = [r.close for r in records]
    volumes = [r.volume for r in records]

    score = 0.0

    # 5 日涨跌幅
    pct_5d = (closes[-1] / closes[-5] - 1) * 100 if closes[-5] > 0 else 0
    if pct_5d > 3:
        score += 8
    elif pct_5d > 0:
        score += 4
    elif pct_5d < -5:
        score -= 8

    # 均线排列 (MA5 > MA10 > MA20)
    ma5 = sum(closes[-5:]) / 5
    ma10 = sum(closes[-10:]) / min(10, len(closes)) if len(closes) >= 10 else ma5
    ma20 = sum(closes[-20:]) / min(20, len(closes)) if len(closes) >= 20 else ma10
    if ma5 > ma10 > ma20:
        score += 6
    elif ma5 < ma10 < ma20:
        score -= 4

    # 量价配合 (最近 3 天量增价升)
    if len(records) >= 3:
        vol_avg = sum(volumes[-6:-3]) / 3 if len(volumes) >= 6 else volumes[-3]
        if volumes[-1] > vol_avg * 1.2 and closes[-1] > closes[-2]:
            score += 5

    return max(-20, min(20, score))


def _trend_label(records: list[StockDaily]) -> str:
    s = _compute_trend_score(records)
    if s >= 8:
        return "强上升"
    elif s >= 3:
        return "温和上升"
    elif s <= -8:
        return "明显下跌"
    elif s <= -3:
        return "偏弱"
    return "震荡"
