"""
智投未来 — Streamlit Web 仪表板

数据源:
  - Demo 模式: 使用 src/demo.py 生成的样本数据, 零依赖0秒加载
  - Trace 模式: 从 results/trace_*.json 加载历史运行记录
  - Live 模式: 实时运行流水线后加载结果

使用方式:
  streamlit run src/web/app.py
  或: python manage.py web
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ── 项目路径 ──────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# ── 页面配置 ──────────────────────────────────
st.set_page_config(
    page_title="智投未来 — A股日内投资AI系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS 美化 ──────────────────────────────────
st.markdown(
    """
<style>
    .stMetric { text-align: center; }
    .stProgress > div > div > div { border-radius: 4px; }
    .signal-bullish { color: #22c55e; font-weight: 600; }
    .signal-bearish { color: #ef4444; font-weight: 600; }
    .signal-neutral { color: #f59e0b; font-weight: 600; }
    .direction-buy { background: #dcfce7; color: #166534; padding: 2px 10px; border-radius: 12px; font-weight: 600; }
    .direction-hold { background: #fef3c7; color: #92400e; padding: 2px 10px; border-radius: 12px; font-weight: 600; }
    .direction-sell { background: #fecaca; color: #991b1b; padding: 2px 10px; border-radius: 12px; font-weight: 600; }
    .debate-bull { background: #f0fdf4; border-left: 4px solid #22c55e; padding: 12px; border-radius: 6px; margin: 8px 0; }
    .debate-bear { background: #fef2f2; border-left: 4px solid #ef4444; padding: 12px; border-radius: 6px; margin: 8px 0; }
    .debate-summary { background: #f8fafc; border-left: 4px solid #6366f1; padding: 12px; border-radius: 6px; margin: 8px 0; }
    .risk-flag { display: inline-block; background: #fef2f2; color: #dc2626; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; margin: 2px; }
    .risk-ok { display: inline-block; background: #f0fdf4; color: #16a34a; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; margin: 2px; }
</style>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════

def load_demo_data() -> dict[str, Any]:
    """调用 demo.py 生成 PipelineState 并转换为 dict"""
    from src.demo import generate_demo_state

    state = generate_demo_state()
    return _state_to_dict(state)


def _state_to_dict(state) -> dict[str, Any]:
    """将 PipelineState 转换为与 trace JSON 一致的 dict 格式"""
    # 候选池
    candidates = []
    for c in getattr(state, "candidates", []) or []:
        candidates.append({
            "code": c.code,
            "name": c.name,
            "score": c.composite,
            **{k: c.scores.get(k, 0) for k in [
                "trend", "momentum", "volume_price", "capital_flow",
                "northbound", "sentiment", "quality", "risk", "liquidity",
                "shareholder_conc",
            ]},
        })

    # 分析师报告
    analyst_reports = {}
    for code, reports in (getattr(state, "analyst_reports", {}) or {}).items():
        analyst_reports[code] = {}
        for r in reports:
            analyst_reports[code][r.analyst_type] = {
                "signal": r.signal,
                "confidence": r.confidence,
                "reasoning": r.reasoning,
                "key_points": r.key_points,
                "risks": r.risks,
            }

    # 辩论
    debates = {}
    for code, d in (getattr(state, "debates", {}) or {}).items():
        debates[code] = {
            "name": d.name,
            "total_rounds": d.total_rounds,
            "rounds": [
                {
                    "round": r.round_num,
                    "bull_argument": r.bull_argument,
                    "bear_argument": r.bear_argument,
                    "bull_rebuttal": r.bull_rebuttal,
                    "bear_summary": r.bear_summary,
                }
                for r in d.rounds
            ],
        }

    # 研判结论
    verdicts = {}
    for code, v in (getattr(state, "verdicts", {}) or {}).items():
        verdicts[code] = {
            "name": v.name,
            "direction": v.direction,
            "confidence": v.confidence,
            "target_price": v.target_price,
            "risk_level": v.risk_level,
            "core_reasoning": v.core_reasoning,
            "key_risks": v.key_risks,
        }

    # 风控
    position_limits = {}
    for code, pl in (getattr(state, "position_limits", {}) or {}).items():
        position_limits[code] = {
            "name": pl.name,
            "max_position_pct": pl.max_position_pct,
            "max_shares": pl.max_shares,
            "max_value": pl.max_value,
            "volatility": pl.volatility,
            "risk_flags": pl.risk_flags,
        }

    # 组合
    fr = getattr(state, "final_result", None)
    decisions = []
    portfolio = {}
    if fr:
        decisions = [
            {
                "symbol": d.symbol,
                "symbol_name": d.symbol_name,
                "volume": d.volume,
                "entry_price": d.entry_price,
                "asset_type": d.asset_type,
                "direction": d.direction,
            }
            for d in fr.decisions
        ]
        portfolio = {
            "cash_used": fr.cash_used,
            "cash_remaining": fr.cash_remaining,
            "total_positions": fr.total_positions,
            "risk_summary": getattr(fr, "risk_summary", ""),
        }

    # 日线数据
    daily_data = {}
    for code, days in (getattr(state, "daily_data", {}) or {}).items():
        daily_data[code] = [
            {
                "date": d.date, "open": d.open, "high": d.high,
                "low": d.low, "close": d.close, "volume": d.volume,
                "ma5": d.ma5, "ma10": d.ma10, "ma20": d.ma20,
                "macd_dif": d.macd_dif, "macd_dea": d.macd_dea,
                "macd_bar": d.macd_bar, "rsi_6": d.rsi_6, "rsi_14": d.rsi_14,
            }
            for d in days
        ]

    # 资金流向
    fund_flows = {}
    for code, flows in (getattr(state, "fund_flows", {}) or {}).items():
        fund_flows[code] = [
            {
                "date": f.date, "main_net_inflow": f.main_net_inflow,
                "super_large_net": f.super_large_net, "large_net": f.large_net,
                "medium_net": f.medium_net, "small_net": f.small_net,
                "main_pct": f.main_pct,
            }
            for f in flows
        ]

    return {
        "date": getattr(state, "date", datetime.now().strftime("%Y-%m-%d")),
        "stage": getattr(state, "stage", "done"),
        "total_capital": getattr(state, "total_capital", 500_000),
        "elapsed": {
            "total": sum((getattr(state, "elapsed", {}) or {}).values()),
            "stages": getattr(state, "elapsed", {}) or {},
        },
        "screening": {"total_candidates": len(candidates), "candidates": candidates},
        "analysis": analyst_reports,
        "debates": debates,
        "verdicts": verdicts,
        "risk": position_limits,
        "decisions": decisions,
        "portfolio": portfolio,
        "daily_data": daily_data,
        "fund_flows": fund_flows,
        "errors": getattr(state, "errors", []) or [],
        "data_quality": getattr(state, "data_quality", {}) or {},
        "etf_candidates": getattr(state, "etf_candidates", []) or [],
        "market_regime": getattr(state, "market_regime", "neutral"),
    }


def list_trace_files() -> dict[str, str]:
    """扫描 results/ 目录返回 {label: filepath}"""
    results_dir = PROJECT_DIR / "results"
    if not results_dir.exists():
        return {}
    files = sorted(results_dir.glob("trace_*.json"), reverse=True)
    out = {}
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
            ts = data.get("timestamp", fp.stem)
            date_str = data.get("date", fp.stem.replace("trace_", ""))
            decisions = data.get("decisions", [])
            n_buy = len([d for d in decisions if d.get("direction", "buy") == "buy"])
            label = f"{date_str} — {n_buy}笔买入 — {data.get('screening', {}).get('total_candidates', 0)}候选"
            out[label] = str(fp)
        except (json.JSONDecodeError, OSError):
            out[fp.stem] = str(fp)
    return out


def load_trace_file(filepath: str) -> dict[str, Any]:
    """加载 trace JSON 文件"""
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════

def _d(data: dict, key: str, default: Any = None) -> Any:
    return data.get(key, default)

def _get_candidates(data: dict) -> list[dict]:
    return _d(_d(data, "screening", {}), "candidates", [])

def _get_analyst_reports(data: dict) -> dict:
    return _d(data, "analysis", {})

def _get_debates(data: dict) -> dict:
    return _d(data, "debates", {})

def _get_verdicts(data: dict) -> dict:
    return _d(data, "verdicts", {})

def _get_position_limits(data: dict) -> dict:
    return _d(data, "risk", {})

def _get_decisions(data: dict) -> list[dict]:
    return _d(data, "decisions", [])

def _get_portfolio(data: dict) -> dict:
    return _d(data, "portfolio", {})

def _get_daily_data(data: dict) -> dict:
    return _d(data, "daily_data", {})

def _get_fund_flows(data: dict) -> dict:
    return _d(data, "fund_flows", {})

def _get_errors(data: dict) -> list[str]:
    return _d(data, "errors", [])

def _get_elapsed(data: dict) -> dict:
    return _d(_d(data, "elapsed", {}), "stages", {})

def _get_data_quality(data: dict) -> dict:
    return _d(data, "data_quality", {})


FACTOR_LABELS = {
    "trend": "趋势", "momentum": "动量", "volume_price": "量价",
    "capital_flow": "资金流", "northbound": "北向资金", "sentiment": "情绪",
    "quality": "质量", "risk": "风险", "liquidity": "流动性",
    "shareholder_conc": "筹码集中",
}


def _score_bar(value: float, max_v: float = 100) -> str:
    """生成带颜色的分数进度条 HTML"""
    if value >= 80:
        color = "#22c55e"
    elif value >= 60:
        color = "#f59e0b"
    else:
        color = "#ef4444"
    pct = min(100, max(0, value / max_v * 100))
    return (
        f'<div style="background:#f1f5f9;border-radius:4px;height:18px;width:100%;">'
        f'<div style="background:{color};border-radius:4px;height:18px;width:{pct:.0f}%;'
        f'display:flex;align-items:center;justify-content:center;font-size:11px;color:#fff;">'
        f'{value:.0f}</div></div>'
    )


def _conf_bar(value: float) -> str:
    """置信度进度条"""
    if value >= 0.7:
        color = "#22c55e"
    elif value >= 0.5:
        color = "#f59e0b"
    else:
        color = "#ef4444"
    pct = value * 100
    return (
        f'<div style="background:#f1f5f9;border-radius:4px;height:20px;width:100%;">'
        f'<div style="background:{color};border-radius:4px;height:20px;width:{pct:.0f}%;'
        f'display:flex;align-items:center;justify-content:center;font-size:12px;color:#fff;font-weight:600;">'
        f'{value:.0%}</div></div>'
    )


def _direction_html(d: str) -> str:
    cls = {"buy": "direction-buy", "sell": "direction-sell"}.get(d, "direction-hold")
    label = {"buy": "买入", "sell": "卖出", "hold": "观望"}.get(d, d)
    return f'<span class="{cls}">{label}</span>'


# ═══════════════════════════════════════════════
#  Tab 渲染
# ═══════════════════════════════════════════════

def render_overview(data: dict):
    """流水线总览"""
    candidates = _get_candidates(data)
    decisions = _get_decisions(data)
    verdicts = _get_verdicts(data)
    reports = _get_analyst_reports(data)
    elapsed = _get_elapsed(data)
    errors = _get_errors(data)
    portfolio = _get_portfolio(data)

    buy_decisions = [d for d in decisions if d.get("direction", "buy") == "buy"]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("候选股票", len(candidates))
    with c2:
        st.metric("深度分析", len(reports))
    with c3:
        n_buy = sum(1 for v in verdicts.values() if v.get("direction") == "buy")
        st.metric("买入研判", n_buy)
    with c4:
        st.metric("最终决策", len(buy_decisions))
    with c5:
        total_s = _d(_d(data, "elapsed", {}), "total", sum(elapsed.values()))
        st.metric("总耗时", f"{total_s:.1f}s")

    st.divider()

    # 流水线阶段时间线
    st.subheader("流水线阶段")
    stages = ["screening", "analysis", "risk", "portfolio"]
    stage_labels = {"screening": "海选筛选", "analysis": "深度分析", "risk": "风控约束", "portfolio": "组合构建"}
    cols = st.columns(len(stages))
    for i, s in enumerate(stages):
        t = elapsed.get(s, 0)
        with cols[i]:
            st.markdown(f"**{stage_labels[s]}**")
            st.caption(f"{t:.1f} 秒")

    # 耗时柱状图
    if elapsed:
        st.bar_chart(
            pd.DataFrame(
                {"阶段": [stage_labels[k] for k in stages if k in elapsed],
                 "耗时(秒)": [elapsed[k] for k in stages if k in elapsed]}
            ).set_index("阶段")
        )

    # 市场环境
    regime = _d(data, "market_regime", "neutral")
    regime_label = {"bull": "🟢 牛市", "neutral": "🟡 震荡", "bear": "🔴 熊市"}.get(regime, regime)
    st.info(f"市场环境: **{regime_label}**  |  日期: **{_d(data, 'date', '-')}**  |  总资金: **¥{_d(data, 'total_capital', 0):,.0f}**")

    # 错误信息
    if errors:
        with st.expander(f"⚠️ 错误信息 ({len(errors)})"):
            for e in errors:
                st.warning(e)


def render_screening(data: dict):
    """市场筛选 — 候选池多因子打分"""
    candidates = _get_candidates(data)
    if not candidates:
        st.info("暂无候选股票数据")
        return

    st.subheader(f"候选池 — {len(candidates)} 只股票")

    # 表格
    df = pd.DataFrame(candidates)
    sort_cols = ["code", "name", "score"] + list(FACTOR_LABELS.keys())
    cols = [c for c in sort_cols if c in df.columns]
    df = df[cols]

    # 用中文列名
    rename = {"code": "代码", "name": "名称", "score": "综合分"}
    rename.update({k: v for k, v in FACTOR_LABELS.items()})
    df_display = df.rename(columns=rename)
    df_display = df_display.sort_values("综合分" if "综合分" in df_display.columns else df_display.columns[2], ascending=False)

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "综合分": st.column_config.ProgressColumn("综合分", min_value=0, max_value=100, format="%.0f"),
            **{rename.get(k, k): st.column_config.ProgressColumn(rename.get(k, k), min_value=0, max_value=100, format="%.0f")
               for k in FACTOR_LABELS.keys() if k in df.columns},
        },
    )

    st.divider()

    # Top-5 因子雷达对比
    st.subheader("Top-5 因子分解对比")
    top5 = sorted(candidates, key=lambda x: x.get("score", x.get("composite", 0)), reverse=True)[:5]
    if top5:
        chart_data = []
        for c in top5:
            row = {"股票": f"{c.get('name','')}({c.get('code','')})"}
            for fk, fl in FACTOR_LABELS.items():
                row[fl] = c.get(fk, 0)
            chart_data.append(row)
        chart_df = pd.DataFrame(chart_data).set_index("股票")
        st.bar_chart(chart_df.T, height=400)


def render_analyst_reports(data: dict):
    """分析师报告"""
    reports = _get_analyst_reports(data)
    verdicts = _get_verdicts(data)
    if not reports:
        st.info("暂无分析师报告数据")
        return

    stock_codes = list(reports.keys())
    default_idx = 0
    selected_code = st.selectbox(
        "选择股票",
        stock_codes,
        index=default_idx,
        format_func=lambda c: f"{verdicts.get(c, {}).get('name', c)} ({c})",
    )

    stock_reports = reports.get(selected_code, {})
    if not stock_reports:
        st.info(f"{selected_code} 无分析师报告")
        return

    # 分析师类型
    analyst_types = ["technical", "fundamentals", "fund_flow", "news"]
    analyst_labels = {"technical": "技术面", "fundamentals": "基本面", "fund_flow": "资金流", "news": "消息面"}

    cols = st.columns(4)
    for i, atype in enumerate(analyst_types):
        r = stock_reports.get(atype)
        if not r:
            with cols[i]:
                st.caption(f"{analyst_labels.get(atype, atype)}: 无数据")
            continue

        signal = r.get("signal", "neutral")
        emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(signal, "⚪")
        signal_cn = {"bullish": "看多", "bearish": "看空", "neutral": "中性"}.get(signal, signal)

        with cols[i]:
            with st.container(border=True):
                st.markdown(f"### {emoji} {analyst_labels.get(atype, atype)}")
                st.caption(f"信号: **{signal_cn}**")
                st.write(f"置信度: **{r.get('confidence', 0):.0%}**")
                st.progress(r.get("confidence", 0))
                st.write(r.get("reasoning", "")[:200])

                if r.get("key_points"):
                    with st.expander("要点"):
                        for p in r["key_points"]:
                            st.write(f"- {p}")
                if r.get("risks"):
                    with st.expander("风险"):
                        for risk in r["risks"]:
                            st.write(f"- ⚠️ {risk}")


def render_debates(data: dict):
    """多空辩论"""
    debates = _get_debates(data)
    verdicts = _get_verdicts(data)
    if not debates:
        st.info("暂无辩论数据")
        return

    # 仅显示有辩论记录的股票
    debated_stocks = {c: d for c, d in debates.items() if d.get("total_rounds", 0) > 0}
    all_stocks = list(debates.keys())

    if not all_stocks:
        st.info("无股票参与辩论")
        return

    selected_code = st.selectbox(
        "选择股票",
        all_stocks,
        index=0,
        format_func=lambda c: f"{verdicts.get(c, {}).get('name', c)} ({c}){' 🔥' if c in debated_stocks else ''}",
    )

    debate = debates.get(selected_code, {})
    rounds = debate.get("rounds", [])
    if not rounds:
        st.info(f"{debate.get('name', selected_code)} 无辩论记录 (可能未通过内部竞赛筛选)")
        return

    st.subheader(f"{debate.get('name', selected_code)} — {debate.get('total_rounds', 0)} 轮辩论")

    for rnd in rounds:
        round_num = rnd.get("round", 0)
        st.markdown(f"#### 第 {round_num} 轮")

        c1, c2 = st.columns(2)
        with c1:
            if rnd.get("bull_argument"):
                st.markdown(
                    f'<div class="debate-bull"><strong>🐂 多头观点</strong><br>{rnd["bull_argument"]}</div>',
                    unsafe_allow_html=True,
                )
        with c2:
            if rnd.get("bear_argument"):
                st.markdown(
                    f'<div class="debate-bear"><strong>🐻 空头观点</strong><br>{rnd["bear_argument"]}</div>',
                    unsafe_allow_html=True,
                )

        if rnd.get("bull_rebuttal"):
            st.markdown(
                f'<div class="debate-bull"><strong>🐂 多头反驳</strong><br>{rnd["bull_rebuttal"]}</div>',
                unsafe_allow_html=True,
            )

        if rnd.get("bear_summary"):
            st.markdown(
                f'<div class="debate-summary"><strong>📋 空头总结</strong><br>{rnd["bear_summary"]}</div>',
                unsafe_allow_html=True,
            )


def render_verdicts(data: dict):
    """研究主管研判结论"""
    verdicts = _get_verdicts(data)
    if not verdicts:
        st.info("暂无研判数据")
        return

    st.subheader(f"研究主管研判 — {len(verdicts)} 只股票")

    # 构建表格数据
    rows = []
    for code, v in verdicts.items():
        rows.append({
            "代码": code,
            "名称": v.get("name", ""),
            "方向": v.get("direction", "hold"),
            "置信度": v.get("confidence", 0),
            "目标价": v.get("target_price", 0),
            "风险等级": v.get("risk_level", "low"),
            "核心逻辑": v.get("core_reasoning", ""),
            "关键风险": ", ".join(v.get("key_risks", [])),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["方向", "置信度"], ascending=[True, False])

    # 方向色标注
    def _dir_color(val):
        return {
            "buy": "background-color: #dcfce7; color: #166534",
            "hold": "background-color: #fef3c7; color: #92400e",
            "sell": "background-color: #fecaca; color: #991b1b",
        }.get(val, "")

    def _risk_color(val):
        return {
            "low": "color: #16a34a",
            "medium": "color: #f59e0b",
            "high": "color: #dc2626",
        }.get(val, "")

    styled = df.style.map(_dir_color, subset=["方向"]).map(_risk_color, subset=["风险等级"])

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        column_config={
            "置信度": st.column_config.ProgressColumn("置信度", min_value=0, max_value=1, format=".0%"),
            "目标价": st.column_config.NumberColumn("目标价 ¥", format="¥%.2f"),
            "方向": st.column_config.Column("方向", width="small"),
        },
    )

    # 详情
    st.divider()
    st.subheader("详细研判")
    codes = list(verdicts.keys())
    selected = st.selectbox(
        "选择股票查看详情",
        codes,
        format_func=lambda c: f"{verdicts[c]['name']} ({c}) — {verdicts[c]['direction']}",
        key="verdict_detail",
    )

    v = verdicts[selected]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("方向", _direction_html(v.get("direction", "")), )
        st.markdown(_direction_html(v.get("direction", "")), unsafe_allow_html=True)
    with c2:
        st.metric("置信度", f"{v.get('confidence', 0):.0%}")
    with c3:
        st.metric("目标价", f"¥{v.get('target_price', 0):.2f}")
    with c4:
        risk = v.get("risk_level", "low")
        risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(risk, "⚪")
        st.metric("风险等级", f"{risk_emoji} {risk}")

    st.markdown(f"**核心逻辑:** {v.get('core_reasoning', '-')}")
    st.markdown("**关键风险:**")
    for risk in v.get("key_risks", []):
        st.markdown(f"- ⚠️ {risk}")


def render_risk(data: dict):
    """风控约束"""
    position_limits = _get_position_limits(data)
    if not position_limits:
        st.info("暂无风控数据")
        return

    st.subheader(f"风控仓位限制 — {len(position_limits)} 只股票")

    rows = []
    for code, pl in position_limits.items():
        rows.append({
            "代码": code,
            "名称": pl.get("name", ""),
            "最大仓位%": pl.get("max_position_pct", 0) * 100,
            "最大股数": pl.get("max_shares", 0),
            "最大市值": pl.get("max_value", 0),
            "波动率%": pl.get("volatility", 0),
            "风险标识": pl.get("risk_flags", []),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("最大仓位%", ascending=False)

    # 波动率色
    def _vol_color(val):
        if val < 2:
            return "color: #16a34a"
        elif val < 4:
            return "color: #f59e0b"
        return "color: #dc2626"

    styled = df.style.map(_vol_color, subset=["波动率%"])

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        column_config={
            "最大仓位%": st.column_config.ProgressColumn("最大仓位%", min_value=0, max_value=50, format="%.1f%%"),
            "最大市值": st.column_config.NumberColumn("最大市值", format="¥%.0f"),
            "波动率%": st.column_config.NumberColumn("波动率%", format="%.1f%%"),
            "风险标识": st.column_config.ListColumn("风险标识"),
        },
    )

    # 风险标识摘要
    all_flags: dict[str, int] = {}
    for pl in position_limits.values():
        for f in pl.get("risk_flags", []):
            if f != "无":
                all_flags[f] = all_flags.get(f, 0) + 1

    if all_flags:
        st.divider()
        st.caption("风险标识汇总:")
        for flag, count in all_flags.items():
            st.markdown(f'<span class="risk-flag">{flag} ×{count}</span>', unsafe_allow_html=True)


def render_portfolio(data: dict):
    """组合持仓"""
    decisions = _get_decisions(data)
    portfolio = _get_portfolio(data)
    verdicts = _get_verdicts(data)

    if not decisions:
        st.info("暂无持仓决策")
        return

    buy_decisions = [d for d in decisions if d.get("direction", "buy") == "buy"]
    sell_decisions = [d for d in decisions if d.get("direction", "sell") == "sell"]

    # 关键指标
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("总资金", f"¥{_d(data, 'total_capital', 0):,.0f}")
    with c2:
        st.metric("已使用", f"¥{portfolio.get('cash_used', 0):,.0f}")
    with c3:
        st.metric("剩余现金", f"¥{portfolio.get('cash_remaining', 0):,.0f}")
    with c4:
        st.metric("持仓数", portfolio.get("total_positions", len(buy_decisions)))

    # 资金占用饼图
    cash_used = portfolio.get("cash_used", 0)
    cash_remaining = portfolio.get("cash_remaining", _d(data, "total_capital", 0) - cash_used)
    if cash_used > 0:
        pie_data = pd.DataFrame({
            "类别": ["已分配", "剩余现金"],
            "金额": [cash_used, cash_remaining],
        }).set_index("类别")
        c_left, c_right = st.columns([1, 2])
        with c_left:
            st.bar_chart(pie_data, height=250)

    st.divider()

    # 买入决策
    if buy_decisions:
        st.subheader(f"🟢 买入决策 ({len(buy_decisions)})")
        buy_rows = []
        for d in buy_decisions:
            code = d.get("symbol", "")
            v = verdicts.get(code, {})
            buy_rows.append({
                "代码": code,
                "名称": d.get("symbol_name", ""),
                "股数": d.get("volume", 0),
                "买入价": d.get("entry_price", 0),
                "预计金额": d.get("volume", 0) * d.get("entry_price", 0),
                "类型": d.get("asset_type", "stock"),
                "置信度": v.get("confidence", 0),
            })
        df = pd.DataFrame(buy_rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "买入价": st.column_config.NumberColumn("买入价", format="¥%.2f"),
                "预计金额": st.column_config.NumberColumn("预计金额", format="¥%.0f"),
                "置信度": st.column_config.ProgressColumn("置信度", min_value=0, max_value=1, format=".0%"),
            },
        )

    # 卖出决策
    if sell_decisions:
        st.subheader(f"🔴 卖出决策 ({len(sell_decisions)})")
        sell_rows = [{
            "代码": d.get("symbol", ""),
            "名称": d.get("symbol_name", ""),
            "股数": d.get("volume", 0),
            "卖出价": d.get("entry_price", 0),
            "类型": d.get("asset_type", "stock"),
        } for d in sell_decisions]
        st.dataframe(pd.DataFrame(sell_rows), use_container_width=True, hide_index=True)

    # 风险摘要
    if portfolio.get("risk_summary"):
        st.divider()
        st.info(f"**风险摘要:** {portfolio['risk_summary']}")


def render_technical(data: dict):
    """技术数据 — K线 + 指标 + 资金流"""
    daily_data = _get_daily_data(data)
    fund_flows = _get_fund_flows(data)
    verdicts = _get_verdicts(data)
    candidates = _get_candidates(data)

    all_codes = list(daily_data.keys())
    if not all_codes:
        all_codes = [c.get("code") for c in candidates]

    if not all_codes:
        st.info("暂无技术数据")
        return

    selected_code = st.selectbox(
        "选择股票",
        all_codes,
        format_func=lambda c: f"{verdicts.get(c, {}).get('name', '') or {}.get('name', c)} ({c})",
        key="tech_stock",
    )

    days = daily_data.get(selected_code, [])
    if not days:
        st.info(f"{selected_code} 无日线数据")
        return

    df = pd.DataFrame(days)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

    # K线图 (close + MA)
    st.subheader("价格 & 均线")
    if all(c in df.columns for c in ["close", "ma5", "ma10", "ma20"]):
        price_df = df.set_index("date")[["close", "ma5", "ma10", "ma20"]]
        price_df.columns = ["收盘价", "MA5", "MA10", "MA20"]
        st.line_chart(price_df, height=300)

    # 成交量
    if "volume" in df.columns:
        st.subheader("成交量")
        vol_df = df.set_index("date")[["volume"]]
        vol_df.columns = ["成交量"]
        st.bar_chart(vol_df, height=200)

    # MACD
    c1, c2 = st.columns(2)
    if all(c in df.columns for c in ["macd_dif", "macd_dea", "macd_bar"]):
        with c1:
            st.subheader("MACD")
            macd_df = df.set_index("date")[["macd_dif", "macd_dea", "macd_bar"]]
            macd_df.columns = ["DIF", "DEA", "BAR"]
            st.line_chart(macd_df, height=200)

    # RSI
    if all(c in df.columns for c in ["rsi_6", "rsi_14"]):
        with c2:
            st.subheader("RSI")
            rsi_df = df.set_index("date")[["rsi_6", "rsi_14"]]
            rsi_df.columns = ["RSI(6)", "RSI(14)"]
            st.line_chart(rsi_df, height=200)

    # 资金流向
    flows = fund_flows.get(selected_code, [])
    if flows:
        st.divider()
        st.subheader("近5日主力资金流向 (万元)")
        flow_df = pd.DataFrame(flows)
        if "date" in flow_df.columns:
            flow_df = flow_df.set_index("date")

        cols = ["main_net_inflow", "super_large_net", "large_net", "medium_net", "small_net"]
        labels = {"main_net_inflow": "主力净流入", "super_large_net": "超大单", "large_net": "大单",
                   "medium_net": "中单", "small_net": "小单"}
        available = [c for c in cols if c in flow_df.columns]
        if available:
            chart_df = flow_df[available].rename(columns=labels)
            st.bar_chart(chart_df, height=250)

        if "main_pct" in flow_df.columns:
            st.caption(f"主力占比: {flow_df['main_pct'].iloc[-1]:.1f}%")


def render_system_info(data: dict):
    """系统信息"""
    elapsed = _get_elapsed(data)
    data_quality = _get_data_quality(data)
    errors = _get_errors(data)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("流水线耗时")
        if elapsed:
            st.bar_chart(
                pd.DataFrame({
                    "阶段": list(elapsed.keys()),
                    "秒": list(elapsed.values()),
                }).set_index("阶段")
            )
        else:
            st.caption("无耗时数据")

    with c2:
        st.subheader("数据质量")
        if data_quality:
            qdf = pd.DataFrame([
                {"来源": k, "质量": v} for k, v in data_quality.items()
            ])
            st.dataframe(qdf, use_container_width=True, hide_index=True)
        else:
            st.caption("无数据质量记录")

    st.divider()

    # 配置摘要
    st.subheader("运行配置")
    try:
        from src.utils.config import get_config
        cfg = get_config()
        c3, c4, c5 = st.columns(3)
        with c3:
            st.caption(f"初始资金: ¥{cfg.initial_capital:,.0f}")
            st.caption(f"风控: 核心{cfg.core_single_pct:.0%} / 卫星{satellite_single_pct:.0%}" if hasattr(cfg, 'satellite_single_pct') else "")
            st.caption(f"最低现金: {cfg.min_cash_reserve:.0%}")
        with c4:
            st.caption(f"最大候选: {cfg.max_candidates}")
            st.caption(f"最小成交额: ¥{cfg.min_daily_amount/1e8:.1f}亿")
            st.caption(f"最大波动: {cfg.max_volatility_pct}%")
        with c5:
            st.caption(f"快模型: {cfg.llm_quick}")
            st.caption(f"深模型: {cfg.llm_deep}")
            st.caption(f"辩论轮数: {cfg.max_debate_rounds}")
    except Exception:
        st.caption("无法加载配置")

    # 错误
    if errors:
        st.divider()
        st.subheader(f"错误日志 ({len(errors)})")
        for e in errors:
            st.code(e, language=None)


# ═══════════════════════════════════════════════
#  AI 对话
# ═══════════════════════════════════════════════

SYSTEM_PROMPT = """你是「智投未来」的AI投资助手，一个面向A股日内投资的智能体系统。

## 你的能力
你可以回答用户关于以下方面的问题：
- 当前分析的股票及其研判结论（看多/看空/观望）
- 投资策略解读（多因子筛选、六维分析师、多空辩论等流程）
- 市场行情与数据分析
- 风控体系与仓位管理逻辑
- 具体的投资建议和风险提示

## 回答原则
1. 如果有当前数据上下文，优先基于数据回答，引用具体的置信度、目标价、风险等级等
2. 投资建议需附带风险提示，不要给出绝对化的买卖建议
3. 回答简洁、有条理，适当使用分点
4. 如果用户问的是系统功能问题，请介绍本系统的核心流程和优势
5. 始终保持专业、理性的语气

## 免责声明
所有回答仅供参考和学习交流，不构成实际投资建议。投资有风险，入市需谨慎。"""


def _build_chat_context(data: dict) -> str:
    """从当前数据构建对话上下文"""
    parts = []

    candidates = _get_candidates(data)
    verdicts = _get_verdicts(data)
    decisions = _get_decisions(data)
    portfolio = _get_portfolio(data)

    if candidates:
        parts.append(f"## 当前候选池 ({len(candidates)}只)")
        for c in sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:5]:
            parts.append(f"- {c.get('code','')} {c.get('name','')} 综合分:{c.get('score',0):.0f}")

    if verdicts:
        parts.append(f"\n## 研究主管研判")
        for code, v in verdicts.items():
            d = v.get("direction", "hold")
            dir_cn = {"buy": "买入", "hold": "观望", "sell": "卖出"}.get(d, d)
            parts.append(
                f"- {code} {v.get('name','')}: **{dir_cn}** "
                f"(置信度:{v.get('confidence',0):.0%}, 目标价:¥{v.get('target_price',0):.2f}, "
                f"风险:{v.get('risk_level','low')})"
            )
            if v.get("core_reasoning"):
                parts.append(f"  逻辑: {v['core_reasoning'][:120]}")
            if v.get("key_risks"):
                parts.append(f"  风险: {', '.join(v['key_risks'][:3])}")

    if decisions:
        buy_decisions = [d for d in decisions if d.get("direction", "buy") == "buy"]
        if buy_decisions:
            parts.append(f"\n## 最终买入决策 ({len(buy_decisions)}笔)")
            for d in buy_decisions:
                parts.append(f"- {d.get('symbol','')} {d.get('symbol_name','')} {d.get('volume',0)}股 @ ¥{d.get('entry_price',0):.2f}")
            parts.append(f"资金使用: ¥{portfolio.get('cash_used',0):,.0f} / ¥{_d(data, 'total_capital', 0):,.0f}")
        else:
            parts.append("\n## 最终决策: 空仓 ([])，今日无符合条件的买入标的")

    elapsed = _get_elapsed(data)
    if elapsed:
        total = sum(elapsed.values())
        parts.append(f"\n## 运行信息: 总耗时{total:.0f}s, 日期{_d(data, 'date', '-')}")

    return "\n".join(parts)


def render_chat(data: dict):
    """AI 对话助手"""
    st.subheader("💬 AI 投资助手")

    # 初始化 chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_context" not in st.session_state:
        st.session_state.chat_context = _build_chat_context(data)

    # 快捷提问
    st.caption("快捷提问:")
    cols = st.columns(4)
    quick_prompts = {
        "今天推荐什么？": "基于当前分析数据，今天推荐买入哪些股票？为什么？",
        "风控怎么样？": "当前的风控约束和仓位管理方案是怎样的？",
        "解释一下流程": "请介绍智投未来系统的核心分析流程和优势。",
        "风险最大的标的": "当前候选池中风险最大的标的是什么？具体风险有哪些？",
    }
    triggered = None
    for i, (label, full) in enumerate(quick_prompts.items()):
        with cols[i]:
            if st.button(label, key=f"quick_{i}", use_container_width=True):
                triggered = full

    st.divider()

    # 渲染历史消息
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 处理输入
    if triggered:
        user_input = triggered
    else:
        user_input = st.chat_input("向AI助手提问...", key="chat_input")

    if user_input:
        # 添加用户消息
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 调用 LLM
        with st.chat_message("assistant"):
            with st.spinner("AI正在思考..."):
                response = _call_chat_llm(
                    user_input,
                    st.session_state.chat_messages,
                    st.session_state.chat_context,
                )
            st.markdown(response)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})

    # 清除对话按钮
    if st.session_state.chat_messages:
        st.divider()
        if st.button("🗑️ 清除对话", key="clear_chat"):
            st.session_state.chat_messages = []
            st.rerun()


def _call_chat_llm(
    user_input: str,
    history: list[dict],
    context: str,
) -> str:
    """调用 DeepSeek API 进行对话"""
    try:
        import os as _os
        import json as _json
        from urllib import request, error as urllib_error

        api_key = _os.getenv("LLM_API_KEY", "").strip()
        base_url = _os.getenv("LLM_BASE_URL", "https://api.deepseek.com").strip().rstrip("/")

        if not api_key:
            return (
                "⚠️ 未检测到 LLM API Key 配置。\n\n"
                "请运行 `python manage.py setup` 配置 DeepSeek API Key，"
                "或创建 `.env` 文件设置 `LLM_API_KEY=你的Key`。\n\n"
                "如果暂时没有 API Key，可以使用演示模式体验系统功能。"
            )

        # 构建消息
        messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + context}]

        # 添加最近10轮历史(避免token过长)
        recent = history[-10:] if len(history) > 10 else history
        messages.extend(recent)

        body = _json.dumps({
            "model": _os.getenv("LLM_QUICK_MODEL", "deepseek-chat"),
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048,
        }).encode("utf-8")

        req = request.Request(
            f"{base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )

        resp = request.urlopen(req, timeout=30)
        data = _json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    except urllib_error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")[:500]
        if e.code == 401:
            return f"⚠️ API Key 无效或已过期，请运行 `python manage.py setup` 重新配置。"
        elif e.code == 429:
            return f"⚠️ API 请求频率过高，请稍后再试。"
        return f"⚠️ API 调用失败 (HTTP {e.code}): {err_body}"
    except Exception as e:
        msg = str(e)
        if "timed out" in msg.lower() or "timeout" in msg.lower():
            return f"⚠️ API 请求超时，请稍后重试。"
        return f"⚠️ 对话服务暂时不可用: {msg[:200]}\n\n可以切换到演示模式使用仪表板功能。"


# ═══════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════

def main():
    # ── Sidebar ────────────────────────────
    st.sidebar.title("智投未来 📊")
    st.sidebar.caption("A股日内投资AI系统 — 可视化仪表板")

    mode = st.sidebar.radio(
        "数据源",
        ["🎮 演示模式", "📁 历史追踪", "🔄 实时运行"],
        help="演示模式: 预生成样本数据 | 历史追踪: 加载 trace JSON | 实时运行: 执行全流水线",
    )

    st.sidebar.divider()
    st.sidebar.caption("""
    **核心流程:**
    海选筛选 → AI深度分析 → 多空辩论 → 风控决策 → 组合构建
    """)

    # ── 数据加载 ──────────────────────────
    data: dict[str, Any] = {}
    source_label = ""

    if mode == "🎮 演示模式":
        data = load_demo_data()
        source_label = "演示数据 (6只样本股)"

    elif mode == "📁 历史追踪":
        trace_files = list_trace_files()
        if not trace_files:
            st.sidebar.warning("results/ 目录中无 trace 文件")
            st.warning("暂无历史追踪数据。请先运行 `python manage.py run` 生成追踪数据。")
            st.caption("或切换到「演示模式」查看样本数据。")
            return

        selected_file = st.sidebar.selectbox("选择追踪文件", list(trace_files.keys()))
        if selected_file:
            filepath = trace_files[selected_file]
            data = load_trace_file(filepath)
            source_label = f"追踪文件: {Path(filepath).name}"
            st.sidebar.caption(f"文件: {Path(filepath).name}")
            if data:
                st.sidebar.caption(f"日期: {_d(data, 'date', '-')}")
                st.sidebar.caption(f"时间: {_d(data, 'timestamp', '-')}")

    elif mode == "🔄 实时运行":
        if st.sidebar.button("开始运行", type="primary", use_container_width=True):
            with st.spinner("正在运行全流水线..."):
                import subprocess

                python = sys.executable
                log_path = PROJECT_DIR / "logs" / "live_run.log"
                log_path.parent.mkdir(exist_ok=True)

                start = time.time()
                result = subprocess.run(
                    [python, "-m", "src.main"],
                    cwd=str(PROJECT_DIR),
                    capture_output=True, text=True,
                    timeout=600,
                )
                elapsed = time.time() - start

                if result.returncode != 0:
                    st.error(f"流水线执行失败 (exit={result.returncode})")
                    st.code(result.stderr[:3000], language=None)
                else:
                    st.success(f"流水线运行完成, 耗时 {elapsed:.1f}s")
                    st.code(result.stdout, language="json")

                    # 尝试加载刚生成的 trace
                    today = datetime.now().strftime("%Y%m%d")
                    trace_path = PROJECT_DIR / "results" / f"trace_{today}.json"
                    if trace_path.exists():
                        data = load_trace_file(str(trace_path))
                        source_label = f"实时运行: {today}"
                        st.rerun()
                    else:
                        st.warning(f"未找到 trace 文件: {trace_path}")
            return
        else:
            st.info("点击「开始运行」执行全流水线，运行完毕后自动加载结果。")
            st.caption("注意: 需要 API Key 配置和网络连接，运行时间约 15-60 秒。")
            return

    if not data:
        st.info("请选择数据源并加载数据")
        return

    # ── 标题 ───────────────────────────────
    st.title("智投未来 — A股日内投资AI系统")
    st.caption(source_label)

    # ── Tabs ───────────────────────────────
    tabs = st.tabs([
        "📋 流水线总览",
        "🔍 市场筛选",
        "🧠 分析师报告",
        "⚔️ 多空辩论",
        "📝 研究结论",
        "🛡️ 风控约束",
        "💰 组合持仓",
        "📈 技术数据",
        "ℹ️ 系统信息",
        "💬 AI对话",
    ])

    with tabs[0]:
        render_overview(data)
    with tabs[1]:
        render_screening(data)
    with tabs[2]:
        render_analyst_reports(data)
    with tabs[3]:
        render_debates(data)
    with tabs[4]:
        render_verdicts(data)
    with tabs[5]:
        render_risk(data)
    with tabs[6]:
        render_portfolio(data)
    with tabs[7]:
        render_technical(data)
    with tabs[8]:
        render_system_info(data)
    with tabs[9]:
        render_chat(data)


if __name__ == "__main__":
    main()
