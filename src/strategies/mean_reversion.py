"""
均值回归策略 — 寻找超卖反弹机会。

筛选逻辑:
- RSI(6) < 30 (超卖)
- 价格接近布林带下轨 (距下轨 < 10% 带宽)
- 得分: 超卖程度 + 距均值距离
"""

from __future__ import annotations

from .base import BaseStrategy, StrategyResult
from ..screening.scorer import FactorScore


class MeanReversionStrategy(BaseStrategy):
    """均值回归策略"""

    name = "mean_reversion"
    description = "均值回归策略 — 寻找 RSI 超卖、触及布林下轨的反弹机会"

    def run(
        self,
        snapshots: list,
        daily_data: dict[str, list],
        fund_flows: dict[str, list],
    ) -> StrategyResult:
        candidates = []

        for snap in snapshots:
            code = snap.code
            records = daily_data.get(code, [])
            if len(records) < 20:
                continue

            latest = records[-1]
            closes = [r.close for r in records[-20:]]

            # 1. RSI 超卖检查
            rsi = latest.rsi_6 if latest.rsi_6 > 0 else 50
            if rsi > 40:
                continue  # 不超卖 → 不参与

            # 2. 布林带计算
            ma20 = sum(closes) / len(closes)
            variance = sum((c - ma20) ** 2 for c in closes) / len(closes)
            std20 = variance ** 0.5
            lower_band = ma20 - 2 * std20
            band_width = 4 * std20  # 上轨 - 下轨

            if band_width < 0.01:
                continue

            # 距下轨距离 (百分比)
            dist_from_lower = (latest.close - lower_band) / band_width if band_width > 0 else 0.5

            # 3. 超卖得分
            if rsi < 20:
                rsi_score = 90
            elif rsi < 25:
                rsi_score = 80
            elif rsi < 30:
                rsi_score = 70
            else:
                rsi_score = 55

            # 4. 距均值距离得分 (越远越好，反映回归空间)
            dist_from_mean = (latest.close / ma20 - 1) * 100  # 负值=低于均值
            if dist_from_mean < -5:
                mean_score = 85
            elif dist_from_mean < -3:
                mean_score = 70
            elif dist_from_mean < -1:
                mean_score = 55
            else:
                mean_score = 40

            # 5. 距下轨得分 (越靠近下轨越好)
            if dist_from_lower < 0.05:
                band_score = 90
            elif dist_from_lower < 0.15:
                band_score = 70
            else:
                band_score = 45

            composite = rsi_score * 0.40 + mean_score * 0.30 + band_score * 0.30
            composite = min(95, max(5, composite))

            candidates.append(FactorScore(
                code=code, name=getattr(snap, "name", code),
                composite=round(composite, 1),
                scores={
                    "trend": round(rsi_score, 1),
                    "momentum": round(mean_score, 1),
                    "volume_price": round(band_score, 1),
                },
            ))

        candidates.sort(key=lambda x: x.composite, reverse=True)
        return StrategyResult(
            name=self.name,
            candidates=candidates[:20],
            metadata={"total_evaluated": len(snapshots)},
        )
