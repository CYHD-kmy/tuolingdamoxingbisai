"""
组合优化模块 — 风险平价/最小方差/最大分散化等权重分配方法。

提供:
- RiskParityOptimizer: 风险贡献均衡优化器
- OptimizationMethod: 优化方法枚举 (ERC/MinVar/MaxDiv)
- OptimizationResult: 优化结果数据类

全部纯 Python 实现，无 numpy 依赖。
"""

from .risk_parity import RiskParityOptimizer, OptimizationResult, OptimizationMethod
