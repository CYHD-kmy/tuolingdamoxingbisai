"""
回测报告生成 — JSON + Markdown 双格式输出。
"""

from __future__ import annotations

import json
import os
from datetime import datetime

from .engine import BacktestResult


def generate_backtest_report(result: BacktestResult, output_dir: str) -> str:
    """
    生成回测报告 (JSON + Markdown)。

    返回: 报告文件路径前缀 (不含扩展名)
    """
    os.makedirs(output_dir, exist_ok=True)

    prefix = f"backtest_{result.config.start_date}_{result.config.end_date}"

    # JSON 报告
    json_path = os.path.join(output_dir, f"{prefix}.json")
    _write_json(result, json_path)

    # Markdown 报告
    md_path = os.path.join(output_dir, f"{prefix}.md")
    _write_markdown(result, md_path)

    return prefix


def _write_json(result: BacktestResult, path: str) -> None:
    """写入 JSON 回测报告"""
    m = result.metrics
    report = {
        "config": {
            "start_date": result.config.start_date,
            "end_date": result.config.end_date,
            "initial_capital": result.config.initial_capital,
            "benchmark": result.config.benchmark,
        },
        "metrics": {
            "sharpe_ratio": m.sharpe_ratio,
            "max_drawdown": m.max_drawdown,
            "win_rate": m.win_rate,
            "annualized_return": m.annualized_return,
            "calmar_ratio": m.calmar_ratio,
            "profit_factor": m.profit_factor,
            "total_trades": m.total_trades,
            "avg_return": m.avg_return,
            "volatility": m.volatility,
            "total_return": m.total_return,
            "final_equity": result.final_equity,
        },
        "daily": [
            {
                "date": dr.date,
                "equity": dr.equity,
                "daily_return": dr.daily_return,
                "benchmark_return": dr.benchmark_return,
                "candidates_count": dr.candidates_count,
                "buy_count": dr.buy_count,
                "decisions": [
                    {"symbol": d.symbol, "name": d.symbol_name, "volume": d.volume}
                    for d in dr.decisions
                ],
                "errors": dr.errors,
            }
            for dr in result.daily_results
        ],
        "generated_at": datetime.now().isoformat(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def _write_markdown(result: BacktestResult, path: str) -> None:
    """写入 Markdown 回测报告"""
    m = result.metrics

    lines = [
        f"# 智投未来 回测报告",
        "",
        f"**回测区间**: {result.config.start_date} ~ {result.config.end_date}",
        f"**初始资金**: ¥{result.config.initial_capital:,.0f}",
        f"**最终权益**: ¥{result.final_equity:,.0f}",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 绩效指标",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| 总收益率 | {m.total_return * 100:+.2f}% |",
        f"| 年化收益率 | {m.annualized_return * 100:+.2f}% |",
        f"| Sharpe 比率 | {m.sharpe_ratio:.2f} |",
        f"| 最大回撤 | {m.max_drawdown * 100:.2f}% |",
        f"| Calmar 比率 | {m.calmar_ratio:.2f} |",
        f"| 胜率 | {m.win_rate * 100:.1f}% |",
        f"| 盈亏比 | {m.profit_factor:.2f} |",
        f"| 年化波动率 | {m.volatility * 100:.2f}% |",
        f"| 交易日数 | {m.total_trades} |",
        "",
        "## 每日明细",
        "",
        "| 日期 | 权益 | 日收益 | 候选数 | 买入数 |",
        "|------|------|--------|--------|--------|",
    ]

    for dr in result.daily_results:
        lines.append(
            f"| {dr.date} | ¥{dr.equity:,.0f} | {dr.daily_return * 100:+.2f}% | "
            f"{dr.candidates_count} | {dr.buy_count} |"
        )

    # 最佳/最差日
    if result.daily_results:
        sorted_days = sorted(result.daily_results, key=lambda x: x.daily_return, reverse=True)
        best = sorted_days[0]
        worst = sorted_days[-1]
        lines.extend([
            "",
            "## 极值日",
            "",
            f"- **最佳日**: {best.date} ({best.daily_return * 100:+.2f}%)",
            f"- **最差日**: {worst.date} ({worst.daily_return * 100:+.2f}%)",
        ])

    # 错误汇总
    all_errors = [e for dr in result.daily_results for e in dr.errors]
    if all_errors:
        lines.extend([
            "",
            "## 错误",
            "",
        ])
        for e in all_errors[:10]:
            lines.append(f"- {e}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
