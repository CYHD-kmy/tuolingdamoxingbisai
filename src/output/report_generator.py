"""
日报生成器 — 将流水线结果渲染为人类可读的 Markdown 报告。

内容包括:
  1. 今日操作摘要 (买入/卖出/持仓变动)
  2. 决策推理链 (数据→信号→辩论→决策 全链路回顾)
  3. 持仓快照 (当前持仓 + 成本 + 浮动盈亏)
  4. 明日关注 (明日需重点关注的候选股)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..graph.state import PipelineState
from ..agents.models import FinalDecision, ResearchVerdict, PositionLimit


def generate_daily_report(state: PipelineState) -> str:
    """基于流水线状态生成完整的 Markdown 日报"""
    today = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"# 智投未来 — 日内投资日报",
        f"**日期**: {today}",
        f"**总资金**: ¥{state.total_capital:,.0f}",
        "",
        "---",
        "",
    ]

    # ── 1. 操作摘要 ──
    lines.extend(_section_summary(state))

    # ── 2. 决策推理链 ──
    lines.extend(_section_reasoning(state))

    # ── 3. 持仓快照 ──
    lines.extend(_section_portfolio(state))

    # ── 4. 明日关注 ──
    lines.extend(_section_watchlist(state))

    # ── 5. 附录: 耗时 ──
    lines.extend(_section_elapsed(state))

    return "\n".join(lines)


def _section_summary(state: PipelineState) -> list[str]:
    lines = ["## 一、今日操作摘要", ""]

    if not state.final_result or not state.final_result.decisions:
        lines.append("**操作**: 今日无买入操作 (空仓)")
        lines.append(f"**原因**: 候选股票 {len(state.candidates)} 只, 有效研判 {len(state.verdicts)} 只")
        if state.errors:
            lines.append("**警告**:")
            for e in state.errors[:3]:
                lines.append(f"  - {e}")
        lines.append("")
        return lines

    decisions = state.final_result.decisions
    lines.append(f"共 **{len(decisions)}** 笔买入操作:")
    lines.append("")
    lines.append("| 代码 | 名称 | 买入股数 |")
    lines.append("|------|------|----------|")
    for d in decisions:
        lines.append(f"| {d.symbol} | {d.symbol_name} | {d.volume} |")
    lines.append("")

    if state.final_result.cash_used > 0:
        lines.append(f"- 使用资金: ¥{state.final_result.cash_used:,.0f}")
        lines.append(f"- 剩余资金: ¥{state.final_result.cash_remaining:,.0f}")
        lines.append(f"- 仓位比例: {state.final_result.cash_used / state.total_capital:.1%}")
    lines.append("")
    return lines


def _section_reasoning(state: PipelineState) -> list[str]:
    lines = ["## 二、决策推理链", ""]

    # 汇总所有 buy 信号的股票
    verdicts = state.verdicts
    buy_verdicts = {code: v for code, v in verdicts.items() if v.direction == "buy"}

    if not buy_verdicts:
        lines.append("今日无买入信号的股票。")
        # 展示部分 hold/sell 的原因
        for code, v in list(verdicts.items())[:5]:
            lines.append(f"- **{v.name}**({code}): {v.direction} | 置信度 {v.confidence:.0%} | {v.core_reasoning[:100]}")
        lines.append("")
        return lines

    # 按置信度降序
    sorted_verdicts = sorted(buy_verdicts.values(), key=lambda x: x.confidence, reverse=True)

    for v in sorted_verdicts[:10]:
        lines.append(f"### {v.name} ({v.code})")
        lines.append(f"- **方向**: {_direction_label(v.direction)} | **置信度**: {v.confidence:.0%} | **风险**: {v.risk_level}")
        if v.target_price:
            lines.append(f"- **目标价**: ¥{v.target_price:.2f}")
        lines.append(f"- **核心理由**: {v.core_reasoning}")
        if v.key_risks:
            lines.append(f"- **关键风险**: {'; '.join(v.key_risks[:3])}")
        lines.append("")

        # 分析师报告摘要
        reports = state.analyst_reports.get(v.code, [])
        if reports:
            lines.append("**四维分析信号**:")
            for r in reports:
                sig = _signal_icon(r.signal)
                lines.append(f"- {r.analyst_type}: {sig} ({r.confidence:.0%}) — {r.reasoning[:80]}...")
            lines.append("")

    return lines


def _section_portfolio(state: PipelineState) -> list[str]:
    lines = ["## 三、持仓快照", ""]

    if not state.final_result or not state.final_result.decisions:
        lines.append("当前无持仓。")
        lines.append("")
        return lines

    lines.append("| 代码 | 名称 | 股数 | 入场价 | 现价 | 浮动盈亏 | 仓位上限 |")
    lines.append("|------|------|------|--------|------|----------|----------|")
    for d in state.final_result.decisions:
        limit = state.position_limits.get(d.symbol)
        max_pct = f"{limit.max_position_pct:.0%}" if limit else "-"

        # 获取入场价和当前价
        entry = d.entry_price if d.entry_price > 0 else _lookup_price(d.symbol, state.daily_data)
        current = _lookup_price(d.symbol, state.daily_data)

        if entry > 0 and current > 0:
            pnl = (current - entry) * d.volume
            pnl_pct = (current / entry - 1) * 100
            pnl_str = f"¥{pnl:+,.0f} ({pnl_pct:+.1f}%)"
        else:
            pnl_str = "-"

        entry_str = f"¥{entry:.2f}" if entry > 0 else "-"
        current_str = f"¥{current:.2f}" if current > 0 else "-"

        lines.append(f"| {d.symbol} | {d.symbol_name} | {d.volume} | {entry_str} | {current_str} | {pnl_str} | {max_pct} |")
    lines.append("")
    return lines


def _lookup_price(code: str, daily_data: dict[str, list]) -> float:
    """从日线数据中查找最新价格"""
    records = daily_data.get(code, [])
    if not records:
        return 0.0
    return records[-1].close


def _section_watchlist(state: PipelineState) -> list[str]:
    lines = ["## 四、明日关注", ""]

    # 筛选高置信度但未买入的标的
    verdicts = state.verdicts
    bought = {d.symbol for d in (state.final_result.decisions if state.final_result else [])}

    watch = [
        v for code, v in verdicts.items()
        if v.direction == "buy" and code not in bought and v.confidence >= 0.5
    ]
    watch.sort(key=lambda x: x.confidence, reverse=True)

    if not watch:
        lines.append("暂无明确明日关注标的。")
    else:
        for v in watch[:8]:
            lines.append(f"- **{v.name}**({v.code}): 置信度 {v.confidence:.0%} | {v.core_reasoning[:80]}")
    lines.append("")
    return lines


def _section_elapsed(state: PipelineState) -> list[str]:
    lines = ["## 五、流水线耗时", ""]
    for stage, secs in state.elapsed.items():
        lines.append(f"- {stage}: {secs:.1f}s")
    total = sum(state.elapsed.values())
    lines.append(f"- **总计**: {total:.1f}s")
    lines.append("")
    return lines


def _direction_label(d: str) -> str:
    return {"buy": "买入", "sell": "卖出", "hold": "持有"}.get(d, d)


def _signal_icon(s: str) -> str:
    return {"bullish": "看多", "bearish": "看空", "neutral": "中性"}.get(s, s)
