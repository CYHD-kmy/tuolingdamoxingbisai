"""
趋势动量策略 — 追涨强势股。

筛选逻辑:
- 均线多头排列: MA5 > MA10 > MA20
- 近期收益在 TOP 四分位
- 量比 > 1.2 (放量确认)
- 得分: 趋势强度 + 动量 + 量能确认
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import BaseStrategy, StrategyResult
from ..screening.scorer import FactorScore


class MomentumStrategy(BaseStrategy):
    """趋势动量策略"""

    name = "momentum"
    description = "趋势动量策略 — 追涨均线多头排列、放量上涨的强势股"

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
            if len(records) < 10:
                continue

            latest = records[-1]

            # 1. 均线多头排列检查
            if not (latest.ma5 > latest.ma10 > latest.ma20 and latest.ma5 > 0):
                continue

            # 2. 趋势强度得分
            ma_spread = (latest.ma5 / latest.ma20 - 1) * 100
            trend_score = min(95, max(20, 50 + ma_spread * 5))

            # 3. 动量得分 (只奖励正向涨幅，温和上涨 2-5% 最优)
            pct = latest.pct_chg
            if pct >= 2 and pct <= 5:
                momentum_score = 75 + pct * 4
            elif pct >= 0.5 and pct <= 8:
                momentum_score = 50 + pct * 2
            else:
                momentum_score = 30  # 下跌或涨幅 > 8% 都不适合追涨

            # 4. 量比确认 (量比 = latest.volume / avg volume)
            avg_vol = sum(r.volume for r in records[-10:]) / 10
            vol_ratio = latest.volume / avg_vol if avg_vol > 0 else 1.0
            if vol_ratio > 1.5:
                volume_score = 80 + min(15, (vol_ratio - 1.5) * 20)
            elif vol_ratio > 1.0:
                volume_score = 55 + (vol_ratio - 1.0) * 50
            else:
                volume_score = 35

            # 综合得分
            composite = trend_score * 0.35 + momentum_score * 0.35 + volume_score * 0.30
            composite = min(95, max(5, composite))

            candidates.append(FactorScore(
                code=code, name=getattr(snap, "name", code),
                composite=round(composite, 1),
                scores={
                    "trend": round(trend_score, 1),
                    "momentum": round(momentum_score, 1),
                    "volume_price": round(volume_score, 1),
                },
            ))

        candidates.sort(key=lambda x: x.composite, reverse=True)
        return StrategyResult(
            name=self.name,
            candidates=candidates[:20],
            metadata={"total_evaluated": len(snapshots)},
        )
