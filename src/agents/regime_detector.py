"""
市场环境检测器 — 判断当前处于牛市/熊市/震荡市。

使用沪深300指数 (000300) 的均线排列 + 趋势 + 波动率来判定。
纯确定性规则，不消耗 LLM Token。
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class MarketRegime(StrEnum):
    BULL = "bull"       # 牛市/强势
    NEUTRAL = "neutral"  # 震荡/中性
    BEAR = "bear"       # 熊市/弱势


class RegimeDetector:
    """
    市场环境检测器。

    使用方式:
        detector = RegimeDetector()
        regime = detector.detect(index_daily_data)
    """

    def __init__(self, lookback: int = 20, index_code: str = "000300") -> None:
        self._lookback = lookback
        self._index_code = index_code

    def detect(self, index_data: list[Any] | None = None) -> MarketRegime:
        """
        检测市场环境。

        index_data: 指数日线数据 (需要 close, pct_chg 字段, 至少 60 天)
        如果为 None，默认返回 NEUTRAL。
        """
        if not index_data or len(index_data) < 60:
            logger.info("RegimeDetector: 指数数据不足 (需>=60天), 默认中性")
            return MarketRegime.NEUTRAL

        # 1. 均线排列: MA20 vs MA60
        close_prices = [r.close for r in index_data[-60:]]
        ma20 = sum(close_prices[-20:]) / 20
        ma60 = sum(close_prices[-60:]) / 60 if len(close_prices) >= 60 else ma20
        ma_divergence = (ma20 / ma60 - 1) * 100  # MA20偏离MA60的百分比

        # 2. 近期趋势: 5日涨跌幅
        pct5 = sum(r.pct_chg for r in index_data[-5:]) if len(index_data) >= 5 else 0

        # 3. 波动率: 近10日涨跌幅标准差
        pcts10 = [r.pct_chg for r in index_data[-10:]]
        vol10 = _std(pcts10) if len(pcts10) >= 5 else 0

        # 4. 判断
        if ma_divergence > 1.0 and pct5 > 0:
            regime = MarketRegime.BULL
        elif ma_divergence < -1.0 and pct5 < 0:
            regime = MarketRegime.BEAR
        else:
            regime = MarketRegime.NEUTRAL

        # 波动率修正: 波动极高时下调一级 (不确定性大)
        if vol10 > 3.0:
            if regime == MarketRegime.BULL:
                regime = MarketRegime.NEUTRAL
            elif regime == MarketRegime.NEUTRAL:
                regime = MarketRegime.BEAR

        logger.info(
            "RegimeDetector: %s (MA20/MA60=%+.1f%%, 5日=%+.1f%%, 波动=%.1f%%)",
            regime.value, ma_divergence, pct5, vol10,
        )
        return regime


def _std(values: list[float]) -> float:
    """计算标准差"""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return variance ** 0.5
