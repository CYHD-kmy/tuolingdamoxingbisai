"""
回测绩效指标 — 纯手动计算，无 numpy/pandas 依赖。

指标:
- Sharpe Ratio (年化)
- Max Drawdown (最大回撤)
- Win Rate (胜率)
- Annualized Return (年化收益)
- Calmar Ratio (卡玛比率)
- Profit Factor (盈亏比)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MetricsReport:
    """回测绩效报告"""
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    annualized_return: float = 0.0
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_return: float = 0.0
    volatility: float = 0.0
    total_return: float = 0.0


def compute_metrics(
    daily_returns: list[float],
    benchmark_returns: list[float] | None = None,
    risk_free_rate: float = 0.02,
    trading_days_per_year: int = 252,
) -> MetricsReport:
    """
    从日收益率序列计算绩效指标。

    daily_returns: 每日收益率 (如 0.01 = 1%)
    benchmark_returns: 基准日收益率 (可选，暂用于信息比率占位)
    risk_free_rate: 年化无风险利率 (默认 2%)
    trading_days_per_year: 年交易日数 (A股约 242-252)
    """
    n = len(daily_returns)
    if n == 0:
        return MetricsReport()

    rf_daily = risk_free_rate / trading_days_per_year

    # 总收益
    cumulative = 1.0
    for r in daily_returns:
        cumulative *= (1 + r)
    total_return = cumulative - 1.0

    # 年化收益
    annualized_return = (cumulative ** (trading_days_per_year / n)) - 1.0 if n > 0 else 0.0

    # 平均日收益
    avg_return = sum(daily_returns) / n

    # 波动率 (年化)
    variance = sum((r - avg_return) ** 2 for r in daily_returns) / (n - 1) if n > 1 else 0.0
    daily_vol = variance ** 0.5
    volatility = daily_vol * (trading_days_per_year ** 0.5)

    # Sharpe Ratio
    excess_returns = [r - rf_daily for r in daily_returns]
    avg_excess = sum(excess_returns) / n
    std_excess = (sum((e - avg_excess) ** 2 for e in excess_returns) / (n - 1)) ** 0.5 if n > 1 else 0.0
    sharpe_ratio = (avg_excess / std_excess) * (trading_days_per_year ** 0.5) if std_excess > 1e-10 else 0.0

    # Max Drawdown (基于累计权益曲线)
    cum = 1.0 + daily_returns[0]
    peak = cum  # 累计权益峰值
    max_drawdown = 0.0
    for r in daily_returns[1:]:
        cum *= (1 + r)
        if cum > peak:
            peak = cum
        dd = (cum / peak - 1) if peak > 0 else 0.0
        if dd < max_drawdown:
            max_drawdown = dd

    # Win Rate
    wins = sum(1 for r in daily_returns if r > 0)
    win_rate = wins / n if n > 0 else 0.0

    # Calmar Ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if abs(max_drawdown) > 1e-10 else 0.0

    # Profit Factor
    gains = sum(r for r in daily_returns if r > 0)
    losses = abs(sum(r for r in daily_returns if r < 0))
    profit_factor = gains / losses if losses > 1e-10 else float("inf") if gains > 0 else 0.0

    return MetricsReport(
        sharpe_ratio=round(sharpe_ratio, 4),
        max_drawdown=round(max_drawdown, 4),
        win_rate=round(win_rate, 4),
        annualized_return=round(annualized_return, 4),
        calmar_ratio=round(calmar_ratio, 4),
        profit_factor=round(profit_factor, 4) if profit_factor != float("inf") else 999.0,
        total_trades=n,
        avg_return=round(avg_return, 6),
        volatility=round(volatility, 4),
        total_return=round(total_return, 4),
    )
