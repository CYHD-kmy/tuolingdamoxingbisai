"""
海选筛选模块 — 从全市场 5000+ 股票中筛选 Top-N 候选。

流水线:
1. filters:  确定性规则过滤 (ST/停牌/新股/流动性)
2. scorer:   多因子加权打分 (8 因子)
3. pipeline: 串联数据层 → 过滤 → 打分 → Top-N
"""

from .scorer import ScreeningScorer, FactorScore
from .filters import (
    filter_tradable,
    filter_liquidity,
    filter_volatility,
)
from .pipeline import ScreeningPipeline, ScreeningResult
