"""
风险平价优化测试
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_record(close: float, pct_chg: float, date: str = "2026-05-01"):
    return type("Record", (), {"close": close, "pct_chg": pct_chg, "date": date})()


def test_covariance_manual():
    """验证手动协方差矩阵计算"""
    from src.optimization.risk_parity import RiskParityOptimizer
    returns = [[1.0, -1.0, 2.0, -2.0], [2.0, -2.0, 1.0, -1.0]]
    cov = RiskParityOptimizer._covariance_matrix(returns)
    assert len(cov) == 2
    assert len(cov[0]) == 2
    # 协方差应非零
    assert abs(cov[0][0]) > 0, f"Variance should be non-zero: {cov[0][0]}"


def test_equal_weight_fallback():
    """空数据返回等权重"""
    from src.optimization.risk_parity import RiskParityOptimizer, OptimizationMethod
    optimizer = RiskParityOptimizer(method=OptimizationMethod.ERC)
    result = optimizer.optimize(["600519", "000858"], {})
    assert result.method == "equal_fallback"
    assert len(result.weights) == 2


def test_erc_basic():
    """两资产 ERC: 高波动资产权重应更低"""
    from src.optimization.risk_parity import RiskParityOptimizer, OptimizationMethod
    from src.data.fetchers.akshare_fetcher import StockDaily

    daily_data = {}
    for code, volatility in [("A", 1.0), ("B", 5.0)]:
        days = []
        for i in range(20):
            days.append(_make_record(
                close=100 + i,
                pct_chg=volatility * (1 if i % 2 == 0 else -1),
                date=f"2026-05-{i+1:02d}",
            ))
        daily_data[code] = days

    optimizer = RiskParityOptimizer(method=OptimizationMethod.ERC)
    result = optimizer.optimize(["A", "B"], daily_data)
    assert result.converged
    assert result.weights["A"] > result.weights["B"], \
        f"Low-vol asset should have higher weight: {result.weights}"


def test_optimization_result():
    """验证 OptimizationResult 数据结构"""
    from src.optimization.risk_parity import OptimizationResult, OptimizationMethod
    r = OptimizationResult(
        weights={"600519": 0.6, "000858": 0.4},
        method=OptimizationMethod.ERC.value,
        converged=True,
        expected_volatility=0.015,
    )
    assert r.weights["600519"] == 0.6
    assert r.method == "erc"
    assert r.converged is True
    assert r.expected_volatility == 0.015


def test_weight_clipping():
    """权重裁剪到风控上限"""
    from src.optimization.risk_parity import RiskParityOptimizer

    class FakeLimit:
        def __init__(self, pct):
            self.max_position_pct = pct

    weights = {"A": 0.5, "B": 0.5}
    limits = {"A": FakeLimit(0.3), "B": FakeLimit(0.3)}
    clipped = RiskParityOptimizer._clip_to_limits(weights, limits)
    assert clipped["A"] == 0.3
    assert clipped["B"] == 0.3


def test_min_variance():
    """最小方差优化基本验证"""
    from src.optimization.risk_parity import RiskParityOptimizer, OptimizationMethod
    from src.data.fetchers.akshare_fetcher import StockDaily

    daily_data = {}
    for code, vol in [("A", 1.0), ("B", 3.0)]:
        days = []
        for i in range(20):
            days.append(_make_record(
                close=100 + i,
                pct_chg=vol * (1 if i % 3 == 0 else -1),
                date=f"2026-05-{i+1:02d}",
            ))
        daily_data[code] = days

    optimizer = RiskParityOptimizer(method=OptimizationMethod.MIN_VARIANCE)
    result = optimizer.optimize(["A", "B"], daily_data)
    assert result.converged
    assert result.method == "min_var"


def test_max_div():
    """最大分散化优化基本验证"""
    from src.optimization.risk_parity import RiskParityOptimizer, OptimizationMethod
    from src.data.fetchers.akshare_fetcher import StockDaily

    daily_data = {}
    for code, vol in [("A", 1.0), ("B", 2.0)]:
        days = []
        for i in range(20):
            days.append(_make_record(
                close=100 + i,
                pct_chg=vol * (1 if i % 2 == 0 else -1),
                date=f"2026-05-{i+1:02d}",
            ))
        daily_data[code] = days

    optimizer = RiskParityOptimizer(method=OptimizationMethod.MAX_DIVERSIFICATION)
    result = optimizer.optimize(["A", "B"], daily_data)
    assert result.converged
    assert result.method == "max_div"


if __name__ == "__main__":
    test_covariance_manual()
    test_equal_weight_fallback()
    test_erc_basic()
    test_optimization_result()
    test_weight_clipping()
    test_min_variance()
    test_max_div()
    print("All optimization tests passed!")
