"""
情绪资金流策略 — 追踪主力资金和北向资金动向。

筛选逻辑:
- 量比 > 1.5 (交易活跃)
- 资金流向趋势 (主力净流入方向)
- 涨跌幅适中 (不过度追涨)
- 得分: 量比 + 涨跌幅 + 换手率合理性
"""

from __future__ import annotations

from .base import BaseStrategy, StrategyResult
from ..screening.scorer import FactorScore


class SentimentStrategy(BaseStrategy):
    """情绪资金流策略"""

    name = "sentiment"
    description = "情绪资金流策略 — 追踪主力净流入、放量活跃的标的"

    def run(
        self,
        snapshots: list,
        daily_data: dict[str, list],
        fund_flows: dict[str, list],
    ) -> StrategyResult:
        candidates = []

        # 预计算所有股票的资金流趋势
        flow_trends = {}
        for code, flows in fund_flows.items():
            if len(flows) >= 3:
                net_inflows = [f.main_net_inflow for f in flows[-5:]]
                flow_trends[code] = sum(net_inflows)
            else:
                flow_trends[code] = 0.0

        for snap in snapshots:
            code = snap.code
            records = daily_data.get(code, [])
            if len(records) < 10:
                continue

            latest = records[-1]

            # 1. 量比得分 (活跃度)
            vol_ratio = getattr(snap, "volume_ratio", 1.0)
            if vol_ratio > 2.0:
                vol_score = 85
            elif vol_ratio > 1.5:
                vol_score = 75
            elif vol_ratio > 1.2:
                vol_score = 60
            else:
                vol_score = 40

            # 2. 涨跌幅得分 (偏好温和上涨 1-4%)
            pct = latest.pct_chg
            if 1 <= pct <= 4:
                pct_score = 80
            elif 0 <= pct <= 6:
                pct_score = 65
            elif -2 <= pct < 0:
                pct_score = 50
            else:
                pct_score = 30

            # 3. 换手率得分 (2-10%)
            turnover = getattr(snap, "turnover", 3.0)
            if 2 <= turnover <= 10:
                turnover_score = 75
            elif 1 <= turnover <= 15:
                turnover_score = 60
            else:
                turnover_score = 40

            # 4. 资金流向得分
            flow_total = flow_trends.get(code, 0.0)
            if flow_total > 5000:
                flow_score = 85
            elif flow_total > 1000:
                flow_score = 70
            elif flow_total > 0:
                flow_score = 55
            elif flow_total > -1000:
                flow_score = 40
            else:
                flow_score = 25

            composite = vol_score * 0.25 + pct_score * 0.20 + turnover_score * 0.20 + flow_score * 0.35
            composite = min(95, max(5, composite))

            candidates.append(FactorScore(
                code=code, name=getattr(snap, "name", code),
                composite=round(composite, 1),
                scores={
                    "sentiment": round(vol_score, 1),
                    "momentum": round(pct_score, 1),
                    "liquidity": round(turnover_score, 1),
                    "capital_flow": round(flow_score, 1),
                },
            ))

        candidates.sort(key=lambda x: x.composite, reverse=True)
        return StrategyResult(
            name=self.name,
            candidates=candidates[:20],
            metadata={
                "total_evaluated": len(snapshots),
                "flow_positive": sum(1 for c in candidates if c.scores.get("capital_flow", 0) > 50),
            },
        )
