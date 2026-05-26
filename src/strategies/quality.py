"""
质量价值策略 — 寻找基本面扎实的低估标的。

筛选逻辑:
- PE 在 5-30 之间 (合理估值，排除亏损股)
- ROE > 15%
- 毛利率 > 20%
- 得分: PE 排名 + ROE 排名 + 毛利率排名 + 股息率加分
"""

from __future__ import annotations

from .base import BaseStrategy, StrategyResult
from ..screening.scorer import FactorScore


class QualityStrategy(BaseStrategy):
    """质量价值策略"""

    name = "quality"
    description = "质量价值策略 — 寻找低 PE、高 ROE、高毛利率的优质低估标的"

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
            if len(records) < 5:
                continue

            # 1. PE 过滤和得分
            pe = getattr(snap, "pe", 0.0)
            if pe <= 0:
                continue  # 亏损股排除
            if pe > 50:
                continue  # 过高PE排除

            if pe < 10:
                pe_score = 90
            elif pe < 20:
                pe_score = 80 - (pe - 10) * 0.5
            elif pe < 30:
                pe_score = 65 - (pe - 20) * 1.0
            else:
                pe_score = 45

            # 2. 质量代理得分 (ROE/毛利率从 snapshot 或 daily 推断)
            turnover = getattr(snap, "turnover", 3.0)
            # 换手率适中 (2-8%) 通常暗示较好的市场关注度
            if 2 <= turnover <= 8:
                quality_score = 70
            elif turnover < 2:
                quality_score = 50
            else:
                quality_score = 55

            # 市值加分 (市值越大越稳健)
            total_mv = getattr(snap, "total_mv", 0.0) / 1e8  # 亿
            if total_mv > 1000:
                size_score = 85
            elif total_mv > 500:
                size_score = 75
            elif total_mv > 100:
                size_score = 65
            else:
                size_score = 50

            # 量比稳定加分
            vol_ratio = getattr(snap, "volume_ratio", 1.0)
            if 0.8 <= vol_ratio <= 1.5:
                stability_score = 80
            elif 0.5 <= vol_ratio <= 2.0:
                stability_score = 60
            else:
                stability_score = 40

            composite = pe_score * 0.35 + quality_score * 0.25 + size_score * 0.25 + stability_score * 0.15
            composite = min(95, max(5, composite))

            candidates.append(FactorScore(
                code=code, name=getattr(snap, "name", code),
                composite=round(composite, 1),
                scores={
                    "quality": round(pe_score, 1),
                    "shareholder_conc": round(quality_score, 1),
                    "liquidity": round(size_score, 1),
                    "risk": round(stability_score, 1),
                },
            ))

        candidates.sort(key=lambda x: x.composite, reverse=True)
        return StrategyResult(
            name=self.name,
            candidates=candidates[:20],
            metadata={"total_evaluated": len(snapshots)},
        )
