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


def generate_daily_report(state: PipelineState, tracker=None, review_result=None) -> str:
    """基于流水线状态生成完整的 Markdown 日报。

    tracker: 可选的 PortfolioTracker 实例，用于持仓快照和累计收益
    review_result: 可选的 ReviewResult 实例，用于持仓复盘章节
    """
    today = datetime.now().strftime("%Y-%m-%d")

    # 从 state 提取所有字段 (兼容 dict 和 PipelineState)
    total_capital = getattr(state, "total_capital", 0)
    candidates = getattr(state, "candidates", [])
    verdicts = getattr(state, "verdicts", {})
    analyst_reports = getattr(state, "analyst_reports", {})
    final_result = getattr(state, "final_result", None)
    position_limits = getattr(state, "position_limits", {})
    daily_data = getattr(state, "daily_data", {})
    elapsed = getattr(state, "elapsed", {})
    errors = getattr(state, "errors", [])
    etf_candidates = getattr(state, "etf_candidates", [])
    etf_verdicts = getattr(state, "etf_verdicts", {})

    # 账户概览
    equity = total_capital
    total_return = 0.0
    cum_pnl = 0.0
    if tracker:
        equity = tracker.total_equity()
        total_return = tracker.total_return()
        cum_pnl = tracker.cumulative_pnl

    lines = [
        f"# 智投未来 — 日内投资日报",
        f"**日期**: {today}",
        f"**总资金**: ¥{total_capital:,.0f}    "
        f"**当前权益**: ¥{equity:,.0f}    "
        f"**累计收益**: ¥{cum_pnl:+,.0f} ({total_return:+.2f}%)",
        "",
        "---",
        "",
    ]

    # ── 1. 操作摘要 ──
    lines.extend(_section_summary(candidates, verdicts, final_result, errors, total_capital))

    # ── ETF 摘要 ──
    if etf_candidates or etf_verdicts:
        lines.extend(_section_etf(etf_candidates, etf_verdicts, final_result))

    # ── 2. 决策推理链 ──
    lines.extend(_section_reasoning(verdicts, analyst_reports))

    # ── 3. 持仓快照 ──
    lines.extend(_section_portfolio(final_result, position_limits, daily_data, tracker))

    # ── 4. 明日关注 ──
    lines.extend(_section_watchlist(verdicts, final_result))

    # ── 4.5 持仓复盘 ──
    if review_result:
        lines.extend(_section_review(review_result))

    # ── 5. 附录: 耗时 ──
    lines.extend(_section_elapsed(elapsed))

    # ── 6. 历史收益曲线 (如有) ──
    if tracker and tracker.history:
        lines.extend(_section_history(tracker))

    return "\n".join(lines)


def _section_summary(candidates, verdicts, final_result, errors, total_capital: float = 0) -> list[str]:
    lines = ["## 一、今日操作摘要", ""]

    if not final_result or not final_result.decisions:
        lines.append("**操作**: 今日无买入操作 (空仓)")
        lines.append(f"**原因**: 候选股票 {len(candidates)} 只, 有效研判 {len(verdicts)} 只")
        if errors:
            lines.append("**警告**:")
            for e in errors[:3]:
                lines.append(f"  - {e}")
        lines.append("")
        return lines

    decisions = final_result.decisions
    lines.append(f"共 **{len(decisions)}** 笔买入操作:")
    lines.append("")
    lines.append("| 代码 | 名称 | 买入股数 |")
    lines.append("|------|------|----------|")
    for d in decisions:
        lines.append(f"| {d.symbol} | {d.symbol_name} | {d.volume} |")
    lines.append("")

    if final_result.cash_used > 0:
        lines.append(f"- 使用资金: ¥{final_result.cash_used:,.0f}")
        lines.append(f"- 剩余资金: ¥{final_result.cash_remaining:,.0f}")
        lines.append(f"- 仓位比例: {final_result.cash_used / total_capital:.1%}")
    lines.append("")
    return lines


def _section_etf(etf_candidates, etf_verdicts, final_result) -> list[str]:
    lines = ["## ETF 操作摘要", ""]

    if not etf_candidates:
        lines.append("今日 ETF 筛选无候选。")
        lines.append("")
        return lines

    lines.append(f"ETF 筛选候选: {len(etf_candidates)} 只")
    lines.append("")
    lines.append("| 代码 | 名称 | 评分 | 方向 | 置信度 | 成交额(亿) |")
    lines.append("|------|------|------|------|--------|------------|")
    for c in etf_candidates:
        v = etf_verdicts.get(c.code)
        direction = getattr(v, "direction", "-") if v else "-"
        conf = f"{v.confidence:.0%}" if v and v.confidence else "-"
        amount_yi = c.amount / 1e8 if c.amount else 0
        lines.append(f"| {c.code} | {c.name} | {c.score:.0f} | {direction} | {conf} | {amount_yi:.1f} |")
    lines.append("")

    # ETF 买入决策
    if final_result:
        etf_decisions = [d for d in final_result.decisions if getattr(d, "asset_type", "stock") == "etf"]
        if etf_decisions:
            lines.append("**ETF 买入决策**:")
            for d in etf_decisions:
                lines.append(f"- {d.symbol} {d.symbol_name}: {d.volume} 份 @ ¥{d.entry_price:.3f}")
            lines.append("")
        else:
            lines.append("今日无 ETF 买入操作。")
            lines.append("")

    return lines


def _section_reasoning(verdicts, analyst_reports) -> list[str]:
    lines = ["## 二、决策推理链", ""]

    buy_verdicts = {code: v for code, v in verdicts.items() if v.direction == "buy"}

    if not buy_verdicts:
        lines.append("今日无买入信号的股票。")
        for code, v in list(verdicts.items())[:5]:
            lines.append(f"- **{v.name}**({code}): {v.direction} | 置信度 {v.confidence:.0%} | {v.core_reasoning[:100]}")
        lines.append("")
        return lines

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

        reports = analyst_reports.get(v.code, [])
        if reports:
            lines.append("**四维分析信号**:")
            for r in reports:
                sig = _signal_icon(r.signal)
                lines.append(f"- {r.analyst_type}: {sig} ({r.confidence:.0%}) — {r.reasoning[:80]}...")
            lines.append("")

    return lines


def _section_portfolio(final_result, position_limits, daily_data, tracker=None) -> list[str]:
    lines = ["## 三、持仓快照", ""]

    if tracker and tracker.positions:
        summary = tracker.to_summary()
        lines.append(f"现金: ¥{summary['cash']:,.0f}    "
                     f"持仓市值: ¥{summary['market_value']:,.0f}    "
                     f"累计收益: ¥{summary['cumulative_pnl']:+,.0f}")
        lines.append("")
        lines.append("| 代码 | 名称 | 类型 | 股数 | 成本 | 现价 | 浮动盈亏 | 行业 |")
        lines.append("|------|------|------|------|------|------|----------|------|")
        for p in summary["positions"]:
            pnl = f"¥{p['pnl']:+,.0f} ({p['pnl_pct']:+.1f}%)" if p["shares"] > 0 else "-"
            atype = "ETF" if p.get("asset_type") == "etf" else "股票"
            lines.append(
                f"| {p['code']} | {p['name']} | {atype} | {p['shares']} | "
                f"¥{p['avg_cost']:.2f} | ¥{p['last_price']:.2f} | "
                f"{pnl} | {p.get('industry', '-')} |"
            )
        lines.append("")

        if summary.get("industry_exposure"):
            lines.append("**行业分布**:")
            for ind, pct in summary["industry_exposure"].items():
                lines.append(f"- {ind}: {pct:.1f}%")
            lines.append("")
        return lines

    if not final_result or not final_result.decisions:
        lines.append("当前无持仓。")
        lines.append("")
        return lines

    lines.append("| 代码 | 名称 | 股数 | 入场价 | 现价 | 浮动盈亏 | 仓位上限 |")
    lines.append("|------|------|------|--------|------|----------|----------|")
    for d in final_result.decisions:
        limit = position_limits.get(d.symbol)
        max_pct = f"{limit.max_position_pct:.0%}" if limit else "-"

        entry = d.entry_price if d.entry_price > 0 else _lookup_price(d.symbol, daily_data)
        current = _lookup_price(d.symbol, daily_data)

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


def _section_history(tracker) -> list[str]:
    lines = ["## 六、历史收益", ""]
    if not tracker.history:
        lines.append("暂无历史数据。")
        lines.append("")
        return lines

    lines.append("| 日期 | 总权益 | 日收益 | 日收益率 |")
    lines.append("|------|--------|--------|----------|")
    for h in tracker.history[-10:]:
        lines.append(
            f"| {h['date']} | ¥{h['total_value']:,.0f} | "
            f"¥{h['daily_pnl']:+,.0f} | {h['daily_return']:+.2f}% |"
        )
    lines.append("")
    return lines


def _lookup_price(code: str, daily_data: dict[str, list]) -> float:
    """从日线数据中查找最新价格"""
    records = daily_data.get(code, [])
    if not records:
        return 0.0
    return records[-1].close


def _section_watchlist(verdicts, final_result) -> list[str]:
    lines = ["## 四、明日关注", ""]

    bought = {d.symbol for d in (final_result.decisions if final_result else [])}

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


def _section_elapsed(elapsed) -> list[str]:
    lines = ["## 五、流水线耗时", ""]
    for stage, secs in elapsed.items():
        lines.append(f"- {stage}: {secs:.1f}s")
    total = sum(elapsed.values())
    lines.append(f"- **总计**: {total:.1f}s")
    lines.append("")
    return lines


def _section_review(review_result) -> list[str]:
    """持仓复盘章节 — 风控红线检查 + 合理性评分 + 卖飞统计 + 绩效归因"""
    lines = ["## 持仓复盘", ""]

    # ── P0: 风控红线 ──
    if review_result.risk_checks:
        lines.append(f"### 风控红线 | {review_result.risk_summary}")
        lines.append("")
        lines.append("| 代码 | 名称 | R1仓位 | R5止损 | R6止损价 | R7持时 |")
        lines.append("|------|------|--------|--------|----------|--------|")
        for code, check_lines in review_result.risk_checks.items():
            if code == "_portfolio":
                continue
            name = ""
            # 从 position_scores 获取名称
            if review_result.position_scores and code in review_result.position_scores:
                name = review_result.position_scores[code].name
            # 构建各规则状态
            status_map = {rl.rule_id: rl for rl in check_lines}
            r1 = status_map.get("R1")
            r5 = status_map.get("R5")
            r6 = status_map.get("R6")
            r7 = status_map.get("R7")
            def _icon(s: str) -> str:
                return {"pass": "通过", "warn": "预警", "violation": "违规"}.get(s, s)
            lines.append(
                f"| {code} | {name} | {_icon(r1.status) if r1 else '-'} | "
                f"{_icon(r5.status) if r5 else '-'} | "
                f"{_icon(r6.status) if r6 else '-'} | "
                f"{_icon(r7.status) if r7 else '-'} |"
            )
        # 组合级红线
        combo = review_result.risk_checks.get("_portfolio", [])
        if combo:
            lines.append("")
            for rl in combo:
                icon = {"pass": "通过", "warn": "预警", "violation": "违规"}.get(rl.status, rl.status)
                lines.append(f"- **{rl.rule_name}** ({rl.rule_id}): {icon} — {rl.message}")
        lines.append("")

    # ── P1: 持仓合理性评分 ──
    if review_result.position_scores:
        lines.append(f"### 持仓合理性评分 | {review_result.score_summary}")
        lines.append("")
        lines.append("| 代码 | 名称 | 综合得分 | 得分变化 | 盈亏 | 建议 | 理由 |")
        lines.append("|------|------|----------|----------|------|------|------|")
        for code, pr in review_result.position_scores.items():
            score_str = f"{pr.current_score:.0f}"
            if pr.score_change < -5:
                score_str += f" ({pr.score_change:+.0f})"
            elif pr.score_change > 5:
                score_str += f" (+{pr.score_change:.0f})"
            rec_icon = {"hold": "持有", "reduce": "减仓", "clear": "清仓"}.get(pr.recommendation, pr.recommendation)
            lines.append(
                f"| {code} | {pr.name} | {score_str} | {pr.score_change:+.0f} | "
                f"{pr.pnl_pct:+.1f}% | {rec_icon} | {pr.reasoning[:60]} |"
            )
        lines.append("")

    # ── P2: 卖飞检测 ──
    pm = review_result.post_mortem
    if pm and pm.total_sells > 0:
        lines.append("### 历史卖出复盘")
        lines.append("")
        lines.append(f"- 共卖出 **{pm.total_sells}** 笔")
        lines.append(f"- 卖飞 **{pm.sell_too_early}** 笔 (平均卖飞 {pm.avg_missed_gain_pct:+.1f}%)")
        lines.append(f"- 有效止损 **{pm.correct_stops}** 笔 (平均避免损失 {pm.avg_avoided_loss_pct:+.1f}%)")
        if pm.sell_too_early > pm.correct_stops:
            lines.append("- 卖出倾向: 偏早 (建议放宽止损阈值或延长持有)")
        elif pm.correct_stops > pm.sell_too_early:
            lines.append("- 卖出倾向: 偏合理")
        lines.append("")

    # ── P3: 绩效归因 ──
    attr = review_result.attribution
    if attr and attr.total_return_pct != 0:
        lines.append("### 绩效归因")
        lines.append("")
        lines.append(f"- 总收益: **{attr.total_return_pct:+.2f}%**")
        lines.append(f"- 选股贡献: {attr.selection_contribution:+.2f}%")
        lines.append(f"- 择时贡献: {attr.timing_contribution:+.2f}%")
        if attr.selection_contribution > attr.timing_contribution:
            lines.append("- 结论: 选股能力强于择时能力")
        else:
            lines.append("- 结论: 择时能力强于选股能力")
        lines.append("")

    return lines


def _direction_label(d: str) -> str:
    return {"buy": "买入", "sell": "卖出", "hold": "持有"}.get(d, d)


def _signal_icon(s: str) -> str:
    return {"bullish": "看多", "bearish": "看空", "neutral": "中性"}.get(s, s)
