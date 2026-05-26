"""
回测框架测试
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_compute_sharpe():
    """验证 Sharpe 公式: 已知正收益应产生正 Sharpe"""
    from src.backtesting.metrics import compute_metrics
    # 每天约 +0.5%, 有微小波动 → 高 Sharpe
    returns = [0.005 + (i % 3 - 1) * 0.001 for i in range(252)]
    m = compute_metrics(returns)
    assert m.sharpe_ratio > 1.0, f"Expected high Sharpe, got {m.sharpe_ratio}"
    assert m.total_return > 0, f"Expected positive total return"


def test_compute_max_drawdown():
    """验证最大回撤: 峰-谷计算"""
    from src.backtesting.metrics import compute_metrics
    # 先涨 10% 再跌 5%
    returns = [0.02, 0.03, 0.05, -0.03, -0.02, 0.01]
    m = compute_metrics(returns)
    assert m.max_drawdown < 0, f"Expected negative drawdown, got {m.max_drawdown}"
    assert m.max_drawdown > -0.10, f"Drawdown too large: {m.max_drawdown}"


def test_compute_win_rate():
    """验证胜率计算"""
    from src.backtesting.metrics import compute_metrics
    returns = [0.01, -0.01, 0.02, -0.005, 0.005]
    m = compute_metrics(returns)
    assert m.win_rate == 0.6, f"Expected 0.6 win rate, got {m.win_rate}"


def test_empty_returns():
    """空收益率序列返回零指标"""
    from src.backtesting.metrics import compute_metrics, MetricsReport
    m = compute_metrics([])
    assert m.sharpe_ratio == 0.0
    assert m.max_drawdown == 0.0
    assert m.win_rate == 0.0
    assert m.annualized_return == 0.0


def test_metrics_report_dataclass():
    """验证 MetricsReport 数据结构"""
    from src.backtesting.metrics import MetricsReport
    m = MetricsReport(
        sharpe_ratio=1.5,
        max_drawdown=-0.15,
        win_rate=0.65,
        annualized_return=0.25,
    )
    assert m.sharpe_ratio == 1.5
    assert m.max_drawdown == -0.15
    assert m.win_rate == 0.65
    assert m.annualized_return == 0.25
    assert m.calmar_ratio == 0.0  # default


def test_backtest_config():
    """验证 BacktestConfig 数据结构"""
    from src.backtesting.engine import BacktestConfig
    c = BacktestConfig(start_date="20260501", end_date="20260526")
    assert c.start_date == "20260501"
    assert c.end_date == "20260526"
    assert c.initial_capital == 500_000.0


def test_backtest_date_range():
    """验证日期范围生成 (跳过周末)"""
    from src.backtesting.engine import BacktestEngine, BacktestConfig
    config = BacktestConfig(start_date="20260525", end_date="20260526")
    engine = BacktestEngine(config)
    dates = engine._date_range("20260525", "20260526")
    # 2026-05-25 is Monday, 2026-05-26 is Tuesday
    assert len(dates) == 2, f"Expected 2 weekdays, got {len(dates)}: {dates}"


def test_backtest_engine_run():
    """验证回测引擎完整运行"""
    from src.backtesting.engine import BacktestEngine, BacktestConfig
    config = BacktestConfig(start_date="20260525", end_date="20260526")
    engine = BacktestEngine(config)
    result = engine.run()
    assert result.config == config
    assert len(result.daily_results) == 2
    assert result.final_equity > 0
    assert result.metrics.total_trades == 2
    for dr in result.daily_results:
        assert dr.date in ("20260525", "20260526")
        assert dr.equity > 0


def test_generate_report():
    """验证回测报告生成"""
    import tempfile
    from src.backtesting.engine import BacktestEngine, BacktestConfig
    from src.backtesting.report import generate_backtest_report

    config = BacktestConfig(start_date="20260525", end_date="20260526")
    engine = BacktestEngine(config)
    result = engine.run()

    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = generate_backtest_report(result, tmpdir)
        assert os.path.exists(os.path.join(tmpdir, f"{prefix}.json"))
        assert os.path.exists(os.path.join(tmpdir, f"{prefix}.md"))


if __name__ == "__main__":
    test_compute_sharpe()
    test_compute_max_drawdown()
    test_compute_win_rate()
    test_empty_returns()
    test_metrics_report_dataclass()
    test_backtest_config()
    test_backtest_date_range()
    test_backtest_engine_run()
    test_generate_report()
    print("All backtesting tests passed!")
