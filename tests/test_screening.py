"""
测试筛选模块 — 过滤器。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.screening.filters import filter_tradable, filter_liquidity
from src.data.fetchers.akshare_fetcher import MarketSnapshot


def _make_snapshot(code: str, name: str, price: float = 10.0,
                   turnover: float = 3.0, amount: float = 100_000_000):
    return MarketSnapshot(
        code=code, name=name, price=price,
        pct_chg=2.0, volume_ratio=1.5, amount=amount,
        turnover=turnover, pe=20.0, total_mv=1e11,
    )


def test_filter_st():
    """ST 股票应被过滤"""
    snapshots = [
        _make_snapshot("000001", "*ST测试"),
        _make_snapshot("600519", "贵州茅台"),
    ]
    result = filter_tradable(snapshots)
    codes = [s.code for s in result]
    assert "000001" not in codes
    assert "600519" in codes


def test_filter_suspended():
    """停牌股 (价格为0) 应被过滤"""
    snapshots = [
        _make_snapshot("000001", "停牌股", price=0.0),
        _make_snapshot("600519", "贵州茅台"),
    ]
    result = filter_tradable(snapshots)
    codes = [s.code for s in result]
    assert "000001" not in codes
    assert "600519" in codes


def test_filter_low_turnover():
    """换手率极低的股票视为停牌"""
    snapshots = [
        _make_snapshot("000001", "僵尸股", turnover=0.0),
        _make_snapshot("600519", "贵州茅台"),
    ]
    result = filter_tradable(snapshots)
    codes = [s.code for s in result]
    assert "000001" not in codes


def test_filter_liquidity():
    """低流动性股票应被过滤"""
    snapshots = [
        _make_snapshot("000001", "低成交", amount=1_000_000),       # 100万
        _make_snapshot("600519", "贵州茅台", amount=1_000_000_000),  # 10亿
    ]
    result = filter_liquidity(snapshots, min_amount=50_000_000)
    codes = [s.code for s in result]
    assert "000001" not in codes
    assert "600519" in codes


def test_filter_pass():
    """正常股票应通过所有过滤"""
    snapshots = [
        _make_snapshot("600519", "贵州茅台"),
        _make_snapshot("000858", "五粮液"),
    ]
    result = filter_tradable(snapshots)
    assert len(result) == 2


if __name__ == "__main__":
    test_filter_st()
    test_filter_suspended()
    test_filter_low_turnover()
    test_filter_liquidity()
    test_filter_pass()
    print("filters: 全部通过")
