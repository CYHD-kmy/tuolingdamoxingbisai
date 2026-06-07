"""
集合竞价分析模块 — 盘前 9:15-9:25 关键信号分析。

集合竞价是 A 股短线交易最重要的盘前信号:
- 竞价量比: 衡量盘前资金关注度
- 竞价价格偏离: 判断开盘方向
- 订单流不平衡: 多空力量对比

比赛场景: 早盘采集建议前，优先分析竞价信号锁定强势标的。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── 信号模型 ──────────────────────────────────


@dataclass
class AuctionSignal:
    """单只股票的集合竞价信号"""
    code: str
    name: str
    auction_price: float          # 集合竞价价格 (元)
    prev_close: float             # 昨日收盘价 (元)
    auction_volume: int           # 集合竞价成交量 (股)
    avg_daily_volume_20: int      # 近20日均量 (股)
    _volume_ratio_direct: float = 0.0  # 直接从数据源获取的量比 (替代计算)

    @property
    def price_deviation_pct(self) -> float:
        """竞价价格偏离 (%)"""
        if self.prev_close <= 0:
            return 0.0
        return (self.auction_price - self.prev_close) / self.prev_close * 100

    @property
    def volume_ratio(self) -> float:
        """竞价量比 (优先使用直接值, 否则计算)"""
        if self._volume_ratio_direct > 0:
            return self._volume_ratio_direct
        if self.avg_daily_volume_20 <= 0:
            return 0.0
        return (self.auction_volume / self.avg_daily_volume_20) * 100

    @classmethod
    def from_dict(cls, data: dict) -> "AuctionSignal":
        """从 tushare daily_basic 等数据源字典创建"""
        return cls(
            code=data.get("code", ""),
            name=data.get("name", ""),
            auction_price=float(data.get("open", data.get("auction_price", 0)) or 0),
            prev_close=float(data.get("pre_close", data.get("prev_close", 0)) or 0),
            auction_volume=int(data.get("vol", data.get("auction_volume", 0)) or 0),
            avg_daily_volume_20=0,
            _volume_ratio_direct=float(data.get("volume_ratio", 0) or 0),
        )

    @property
    def is_gap_up(self) -> bool:
        """是否跳空高开"""
        return self.price_deviation_pct > 2.0

    @property
    def is_gap_down(self) -> bool:
        """是否跳空低开"""
        return self.price_deviation_pct < -2.0

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "name": self.name,
            "auction_price": self.auction_price,
            "prev_close": self.prev_close,
            "price_deviation_pct": round(self.price_deviation_pct, 2),
            "volume_ratio": round(self.volume_ratio, 2),
            "is_gap_up": self.is_gap_up,
        }


# ── 分析器 ────────────────────────────────────


class AuctionAnalyzer:
    """
    集合竞价信号分析器。

    分析维度:
    1. 竞价量比 — 量越大，开盘后越可能延续方向
    2. 竞价价格偏离 — 偏离越大，开盘动能越强
    3. 综合评分 — 量价配合的信号更有价值
    """

    # 评分配置
    _VOL_RATIO_THRESHOLD_HIGH = 8.0    # 竞价量比 > 8% → 极度活跃
    _VOL_RATIO_THRESHOLD_MID = 4.0     # 竞价量比 > 4% → 活跃
    _VOL_RATIO_THRESHOLD_LOW = 2.0     # 竞价量比 > 2% → 一般
    _PRICE_DEV_HIGH = 5.0              # 竞价涨幅 > 5% → 强势高开
    _PRICE_DEV_MID = 2.0               # 竞价涨幅 > 2% → 温和高开
    _PRICE_DEV_LOW = 0.5               # 竞价涨幅 > 0.5% → 小幅高开

    def analyze(
        self, signals: list[AuctionSignal | dict]
    ) -> list[AuctionSignal]:
        """
        批量分析竞价信号 (支持 dict → AuctionSignal 自动转换)，
        返回按综合评分降序排序的信号列表。
        """
        converted = [
            AuctionSignal.from_dict(s) if isinstance(s, dict) else s
            for s in signals
        ]
        for s in converted:
            s._score = self._score_signal(s)
        converted.sort(key=lambda x: getattr(x, '_score', 0), reverse=True)
        return converted

    def _score_signal(self, s: AuctionSignal) -> float:
        """单信号综合评分 0-100"""
        score = 50.0

        # 竞价量比 (权重 0.5)
        vr = s.volume_ratio
        if vr >= self._VOL_RATIO_THRESHOLD_HIGH:
            score += 25
        elif vr >= self._VOL_RATIO_THRESHOLD_MID:
            score += 15
        elif vr >= self._VOL_RATIO_THRESHOLD_LOW:
            score += 5
        elif vr < 1.0:
            score -= 10

        # 竞价价格偏离 (权重 0.5)
        dev = s.price_deviation_pct
        if 2.0 <= dev <= 5.0:
            score += 25  # 最佳开盘点: 温和高开
        elif 0.5 <= dev < 2.0:
            score += 15  # 小幅高开
        elif 5.0 < dev <= 8.0:
            score += 8   # 偏高开 (有追高风险)
        elif -2.0 <= dev < 0.5:
            score -= 5   # 弱势开盘
        elif dev < -2.0:
            score -= 15  # 跳空低开
        else:
            score -= 20  # 异常高开 > 8% (追板风险)

        return max(0, min(100, score))

    def get_top_signals(
        self, signals: list[AuctionSignal], top_n: int = 30
    ) -> list[AuctionSignal]:
        """获取竞价评分最高的 Top-N"""
        analyzed = self.analyze(signals)
        return analyzed[:top_n]

    def filter_strong_signals(
        self, signals: list[AuctionSignal],
        min_volume_ratio: float = 3.0,
        min_price_dev: float = 0.5,
        max_price_dev: float = 8.0,
    ) -> list[AuctionSignal]:
        """过滤出强势竞价信号"""
        return [
            s for s in signals
            if (s.volume_ratio >= min_volume_ratio and
                min_price_dev <= s.price_deviation_pct <= max_price_dev)
        ]

    @staticmethod
    def build_auction_context(signals: list[AuctionSignal], top_n: int = 10) -> str:
        """构建竞价信号文本上下文 (供 LLM 分析使用)"""
        if not signals:
            return "暂无集合竞价数据"

        lines = ["## 集合竞价信号 (Top {})".format(min(top_n, len(signals)))]
        for i, s in enumerate(signals[:top_n]):
            direction = "🔺高开" if s.price_deviation_pct > 0 else "🔻低开"
            lines.append(
                f"{i+1}. {s.name}({s.code}) {direction} "
                f"{s.price_deviation_pct:+.2f}% | "
                f"竞价量比 {s.volume_ratio:.1f} | "
                f"竞价价 {s.auction_price:.2f}"
            )
        return "\n".join(lines)
