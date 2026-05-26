"""
回测框架 — 在历史日期上重放完整流水线并计算绩效指标。

提供:
- BacktestEngine: 逐日重放流水线
- compute_metrics: 绩效指标计算 (Sharpe/MaxDD/WinRate/Calmar/ProfitFactor)
- generate_backtest_report: JSON + Markdown 双格式报告
"""

from .engine import BacktestEngine, BacktestConfig, BacktestResult, DailyBacktestResult
from .metrics import compute_metrics, MetricsReport
from .report import generate_backtest_report
