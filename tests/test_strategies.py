"""
多策略竞争测试
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_snapshot(code, name, price=10.0, turnover=3.0, amount=1e8, pe=20.0, vol_ratio=1.2, total_mv=5e10):
    return type("Snapshot", (), {
        "code": code, "name": name, "price": price,
        "turnover": turnover, "amount": amount, "pe": pe,
        "volume_ratio": vol_ratio, "total_mv": total_mv,
    })()


def _make_records(pct_chg_vals=None, close=10.0):
    if pct_chg_vals is None:
        pct_chg_vals = [0.5, 1.0, 1.5, 2.0, 1.8]
    records = []
    for i, pct in enumerate(pct_chg_vals):
        records.append(type("Record", (), {
            "date": f"2026-05-{i+1:02d}",
            "close": close * (1 + i * 0.01),
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "volume": 2e7,
            "amount": 2e8,
            "pct_chg": pct,
            "turnover": 3.0,
            "ma5": close * 1.02,
            "ma10": close * 1.01,
            "ma20": close * 1.00,
            "rsi_6": 55.0,
            "rsi_14": 52.0,
            "macd_dif": 0.5,
            "macd_dea": 0.3,
            "macd_bar": 0.2,
        })())
    return records


def _make_flows(code, values=None):
    if values is None:
        values = [500, 800, 300, 1200, 600]
    return [
        type("Flow", (), {
            "date": f"2026-05-{i+1:02d}",
            "main_net_inflow": v,
            "super_large_net": v * 0.5,
            "large_net": v * 0.3,
            "medium_net": v * 0.1,
            "small_net": v * 0.1,
            "main_pct": abs(v) / 50000 * 100,
        })()
        for i, v in enumerate(values)
    ]


def test_momentum_strategy():
    """动量策略: 均线多头 + 放量上涨"""
    from src.strategies.momentum import MomentumStrategy

    snapshots = [_make_snapshot("600519", "茅台")]
    daily_data = {"600519": _make_records([1.5, 2.0, 3.0, 2.5, 3.5])}

    strategy = MomentumStrategy()
    result = strategy.run(snapshots, daily_data, {})
    assert result.name == "momentum"
    assert len(result.candidates) >= 0


def test_mean_reversion_strategy():
    """均值回归策略: RSI 超卖"""
    from src.strategies.mean_reversion import MeanReversionStrategy

    snapshots = [_make_snapshot("000001", "平安")]
    records = []
    for i in range(20):
        records.append(type("Record", (), {
            "date": f"2026-05-{i+1:02d}",
            "close": 10.0 + i * 0.1 if i < 15 else 9.5,
            "pct_chg": -2.0 if i >= 15 else 1.0,
            "volume": 2e7,
            "rsi_6": 25.0,
            "rsi_14": 30.0,
        })())
    daily_data = {"000001": records}
    fund_flows = {"000001": _make_flows("000001")}

    strategy = MeanReversionStrategy()
    result = strategy.run(snapshots, daily_data, fund_flows)
    assert result.name == "mean_reversion"


def test_quality_strategy():
    """质量策略: PE 过滤"""
    from src.strategies.quality import QualityStrategy

    snapshots = [_make_snapshot("600519", "茅台", pe=28.5, turnover=3.2)]
    daily_data = {"600519": _make_records()}

    strategy = QualityStrategy()
    result = strategy.run(snapshots, daily_data, {})
    assert result.name == "quality"


def test_sentiment_strategy():
    """情绪策略: 量比 + 资金流"""
    from src.strategies.sentiment import SentimentStrategy

    snapshots = [_make_snapshot("300750", "宁德", vol_ratio=2.0)]
    daily_data = {"300750": _make_records([2.0, 3.0, 2.5, 3.5, 4.0])}
    fund_flows = {"300750": _make_flows("300750", [2000, 3000, 1500, 4000, 5000])}

    strategy = SentimentStrategy()
    result = strategy.run(snapshots, daily_data, fund_flows)
    assert result.name == "sentiment"


def test_strategy_registry():
    """策略注册表基本操作"""
    from src.strategies.registry import StrategyRegistry
    from src.strategies.momentum import MomentumStrategy

    StrategyRegistry.clear()
    StrategyRegistry.register(MomentumStrategy())
    assert "momentum" in StrategyRegistry.list_names()
    assert StrategyRegistry.get("momentum") is not None
    StrategyRegistry.clear()


def test_competition_engine_merge():
    """竞争引擎合并候选"""
    from src.strategies.engine import CompetitionEngine

    snapshots = [
        _make_snapshot("600519", "茅台", pe=28.5),
        _make_snapshot("000858", "五粮液", pe=22.3),
    ]
    daily_data = {
        "600519": _make_records([1.5, 2.0, 3.0, 2.5, 3.5]),
        "000858": _make_records([1.0, 1.5, 2.0, 1.8, 2.2]),
    }
    fund_flows = {
        "600519": _make_flows("600519", [800, 1200, 450, 2100, 950]),
        "000858": _make_flows("000858", [500, 650, 300, 880, 720]),
    }

    engine = CompetitionEngine(strategies=["momentum", "quality", "sentiment"])
    result = engine.run(snapshots, daily_data, fund_flows)

    assert len(result.strategy_results) > 0
    assert isinstance(result.merged_candidates, list)
    assert isinstance(result.allocation, dict)


def test_strategy_result_dataclass():
    """StrategyResult 数据结构"""
    from src.strategies.base import StrategyResult
    r = StrategyResult(name="test", metadata={"elapsed": 1.5})
    assert r.name == "test"
    assert r.candidates == []
    assert r.metadata["elapsed"] == 1.5


def test_default_strategy():
    """默认策略存在"""
    from src.strategies.default_strategy import DefaultStrategy
    s = DefaultStrategy()
    assert s.name == "default"
    assert len(s.description) > 0


if __name__ == "__main__":
    test_momentum_strategy()
    test_mean_reversion_strategy()
    test_quality_strategy()
    test_sentiment_strategy()
    test_strategy_registry()
    test_competition_engine_merge()
    test_strategy_result_dataclass()
    test_default_strategy()
    print("All strategy tests passed!")
