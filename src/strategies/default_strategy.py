"""
默认策略 — 委托给现有 10 因子筛选流水线。
"""

from __future__ import annotations

from .base import BaseStrategy, StrategyResult


class DefaultStrategy(BaseStrategy):
    """默认 10 因子模型策略 — 委托 ScreeningPipeline"""

    name = "default"
    description = "默认10因子模型 — 全市场筛选 + 多因子打分"

    def run(
        self,
        snapshots: list,
        daily_data: dict[str, list],
        fund_flows: dict[str, list],
    ) -> StrategyResult:
        """委托给现有筛选流水线"""
        from ..screening.pipeline import ScreeningPipeline
        from ..data.interface import UnifiedDataInterface

        data = UnifiedDataInterface()
        pipeline = ScreeningPipeline(data)
        result = pipeline.run()

        return StrategyResult(
            name=self.name,
            candidates=list(result.candidates),
            metadata={
                "total_screened": result.total_screened,
                "after_filters": result.after_filters,
            },
        )
