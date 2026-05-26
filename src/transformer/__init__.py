"""轻量级 Transformer 时序编码器 — Phase 4

纯 Python 实现，零外部 ML 依赖。
从 K 线序列中学习股票的时序表示，增强手工因子评分。
"""

from .model import StockTransformer
from .scorer import TransformerScorer
from .training import generate_training_data, train_transformer

__all__ = [
    "StockTransformer",
    "TransformerScorer",
    "generate_training_data",
    "train_transformer",
]
