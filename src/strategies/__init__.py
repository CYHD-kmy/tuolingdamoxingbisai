"""
多策略竞争框架 — 不同 Alpha 模型并行运行并竞争资金分配。

策略:
- default: 默认 10 因子模型 (委托 ScreeningPipeline)
- momentum: 趋势动量策略 (MA排列 + 近期收益 + 量比)
- mean_reversion: 均值回归策略 (RSI + 布林带位置)
- quality: 质量价值策略 (PE/ROE/毛利率)
- sentiment: 情绪资金流策略 (主力净流入 + 北向 + 量比)

使用方式:
    from src.strategies import CompetitionEngine
    engine = CompetitionEngine()
    result = engine.run(daily_data, fund_flows)
"""

from .base import BaseStrategy, StrategyResult
from .registry import StrategyRegistry, StrategyPerformance
from .engine import CompetitionEngine, CompetitionResult
