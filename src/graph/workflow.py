"""
LangGraph 工作流 — 基于 StateGraph 的日内投资决策流水线。

节点:
  screening → analysis → risk → portfolio → END

条件路由:
  - screening 无候选 → 跳过后续阶段
  - analysis 无有效研判 → 跳过风控和组合
  - risk 无仓位限制 → 跳过组合构建

使用方式:
    from src.graph.workflow import run_pipeline

    result = run_pipeline()
    print(result.final_result)
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langgraph.graph import StateGraph, END

from .state import PipelineState
from ..agents.models import DebateResult, PortfolioResult, ResearchVerdict
from ..data.interface import UnifiedDataInterface
from ..screening.pipeline import ScreeningPipeline
from ..llm.factory import get_quick_llm, get_deep_llm
from ..agents.analysts.technical import TechnicalAnalyst
from ..agents.analysts.fundamentals import FundamentalsAnalyst
from ..agents.analysts.fund_flow import FundFlowAnalyst
from ..agents.analysts.news_sentiment import NewsSentimentAnalyst
from ..agents.analysts.etf import ETFAnalyst
from ..agents.researchers.engine import DebateEngine
from ..agents.managers.research_manager import ResearchManager
from ..agents.managers.risk_manager import RiskManager
from ..agents.managers.portfolio_manager import PortfolioManager
from ..screening.etf_screener import ETFScreener
from ..utils.config import get_config

logger = logging.getLogger(__name__)


# ── 工作流节点 ─────────────────────────────────

def run_screening(state: PipelineState) -> dict[str, Any]:
    """阶段 1: 海选筛选 — 全市场 → Top-N 候选"""
    t0 = time.monotonic()
    logger.info("===== 阶段 1/4: 海选筛选 =====")

    data = UnifiedDataInterface()
    pipeline = ScreeningPipeline(data)

    try:
        cfg = get_config()
        strategy_name = cfg.active_strategies

        if strategy_name and strategy_name != "default":
            # 多策略竞争模式
            from ..strategies.engine import CompetitionEngine
            from ..strategies.registry import StrategyRegistry
            from ..strategies.base import StrategyResult

            snapshots = data.get_market_snapshot()
            if not snapshots:
                result = pipeline.run()
            else:
                engine = CompetitionEngine(
                    strategies=None if strategy_name == "all" else strategy_name.split(","),
                )
                daily_all = data.batch_daily_data(
                    [s.code for s in snapshots[:100]], days=30, max_workers=8,
                )
                flows_all = data.batch_fund_flows(
                    [s.code for s in snapshots[:100]], days=5, max_workers=8,
                )
                comp_result = engine.run(snapshots, daily_all, flows_all)
                from ..screening.pipeline import ScreeningResult
                result = ScreeningResult(
                    candidates=comp_result.merged_candidates[:cfg.max_candidates],
                    total_screened=len(snapshots),
                    after_filters=len(comp_result.merged_candidates),
                    elapsed_filter=0,
                    elapsed_score=comp_result.strategy_results.get("momentum", StrategyResult(name="")).metadata.get("elapsed", 0),
                )
        else:
            result = pipeline.run()

        # Transformer 评分增强
        if cfg.transformer_enabled and cfg.transformer_model_path:
            try:
                from ..transformer import TransformerScorer, StockTransformer

                tf_model = StockTransformer.load(cfg.transformer_model_path)
                tf_scorer = TransformerScorer(tf_model, score_weight=cfg.transformer_scorer_weight)
                tf_scores = tf_scorer.score_all(
                    [c.code for c in result.candidates],
                    data.batch_daily_data([c.code for c in result.candidates], days=30, max_workers=6),
                )
                tf_map = {fs.code: fs.composite for fs in tf_scores}
                for c in result.candidates:
                    if c.code in tf_map:
                        tf_score = tf_map[c.code]
                        c.scores["transformer"] = round(tf_score, 1)
                        c.composite = round(
                            c.composite * (1 - cfg.transformer_scorer_weight)
                            + tf_score * cfg.transformer_scorer_weight,
                            1,
                        )
                logger.info("Transformer 评分已融合: %d 只 (权重=%.0f%%)",
                            len(tf_map), cfg.transformer_scorer_weight * 100)
            except Exception:
                logger.debug("Transformer 评分增强失败", exc_info=True)

        updates: dict[str, Any] = {
            "candidates": result.candidates,
            "errors": state.errors + result.errors,
        }

        codes = [c.code for c in result.candidates]
        if codes:
            updates["daily_data"] = data.batch_daily_data(codes, days=30, max_workers=6)
            updates["fund_flows"] = data.batch_fund_flows(codes, days=5, max_workers=6)
            # 捕获数据质量标记
            quality: dict[str, str] = {}
            for c in codes:
                for dt in ("daily", "fund_flow"):
                    q = data.get_data_quality(c, dt)
                    if q:
                        quality[f"{c}:{dt}"] = q
            updates["data_quality"] = quality

        updates["stage"] = "screening_done"
        logger.info("筛选完成: %d 只候选", len(result.candidates))
    except Exception as e:
        logger.exception("筛选阶段异常")
        updates = {
            "errors": state.errors + [f"筛选失败: {e}"],
            "stage": "screening_failed",
        }

    elapsed = dict(state.elapsed)
    elapsed["screening"] = time.monotonic() - t0
    updates["elapsed"] = elapsed
    return updates


def run_etf_screening(state: PipelineState) -> dict[str, Any]:
    """ETF 筛选 — 全市场 ETF → Top-N 候选"""
    t0 = time.monotonic()
    cfg = get_config()
    logger.info("===== ETF 筛选 =====")

    if not cfg.etf_enabled:
        logger.info("ETF 流水线已禁用，跳过")
        elapsed = dict(state.elapsed)
        elapsed["etf_screening"] = 0
        return {"stage": "etf_screening_skipped", "elapsed": elapsed}

    try:
        data = UnifiedDataInterface()
        screener = ETFScreener(data)
        etf_candidates = screener.screen()

        updates: dict[str, Any] = {
            "etf_candidates": etf_candidates,
            "stage": "etf_screening_done",
        }

        logger.info("ETF 筛选完成: %d 只候选", len(etf_candidates))
    except Exception as e:
        logger.exception("ETF 筛选异常")
        updates = {
            "etf_candidates": [],
            "errors": state.errors + [f"ETF 筛选失败: {e}"],
            "stage": "etf_screening_failed",
        }

    elapsed = dict(state.elapsed)
    elapsed["etf_screening"] = time.monotonic() - t0
    updates["elapsed"] = elapsed
    return updates


def run_etf_analysis(state: PipelineState) -> dict[str, Any]:
    """ETF 分析 — 对 ETF 候选运行 ETF 分析师"""
    t0 = time.monotonic()
    logger.info("===== ETF 分析 (%d 只) =====", len(state.etf_candidates))

    if not state.etf_candidates:
        elapsed = dict(state.elapsed)
        elapsed["etf_analysis"] = 0
        return {"stage": "etf_analysis_skipped", "elapsed": elapsed}

    quick_llm = get_quick_llm()
    data = UnifiedDataInterface()
    analyst = ETFAnalyst(quick_llm, data)

    etf_verdicts: dict[str, ResearchVerdict] = {}
    errors: list[str] = list(state.errors)

    for c in state.etf_candidates:
        try:
            report = analyst.analyze(c.code)
            direction = getattr(report, "signal", "neutral")
            confidence = getattr(report, "confidence", 0.0)
            reasoning = getattr(report, "reasoning", "")[:200]

            etf_verdicts[c.code] = ResearchVerdict(
                code=c.code,
                name=c.name,
                direction=_map_etf_signal(direction),
                confidence=confidence,
                core_reasoning=reasoning,
                asset_type="etf",
            )
        except Exception as e:
            logger.warning("ETF 分析 %s 失败: %s", c.code, e)
            errors.append(f"ETF 分析 {c.code} 失败: {e}")

    elapsed = dict(state.elapsed)
    elapsed["etf_analysis"] = time.monotonic() - t0
    logger.info("ETF 分析完成: %d 只研判", len(etf_verdicts))

    return {
        "etf_verdicts": etf_verdicts,
        "errors": errors,
        "stage": "etf_analysis_done",
        "elapsed": elapsed,
    }


def run_etf_risk(state: PipelineState) -> dict[str, Any]:
    """ETF 风控 — ETF 专用风控规则"""
    t0 = time.monotonic()
    logger.info("===== ETF 风控 (%d 只) =====", len(state.etf_verdicts))

    if not state.etf_verdicts:
        elapsed = dict(state.elapsed)
        elapsed["etf_risk"] = 0
        return {"stage": "etf_risk_skipped", "elapsed": elapsed}

    cfg = get_config()
    risk_mgr = RiskManager(total_capital=state.total_capital)
    etf_limits = risk_mgr.compute_etf_limits(
        list(state.etf_verdicts.values()),
        state.daily_data,
    )

    elapsed = dict(state.elapsed)
    elapsed["etf_risk"] = time.monotonic() - t0
    logger.info("ETF 风控完成: %d 只可买入", sum(1 for l in etf_limits.values() if l.max_shares > 0))

    return {
        "etf_position_limits": etf_limits,
        "stage": "etf_risk_done",
        "elapsed": elapsed,
    }


def _map_etf_signal(signal: str) -> str:
    """将 ETF 分析信号映射为方向"""
    if signal in ("bullish", "buy"):
        return "buy"
    elif signal in ("bearish", "sell"):
        return "sell"
    return "hold"


def run_analysis(state: PipelineState) -> dict[str, Any]:
    """阶段 2: 深度分析 — 四分析师 + 辩论 + 研究主管研判"""
    t0 = time.monotonic()
    logger.info("===== 阶段 2/4: 深度分析 (%d 只) =====", len(state.candidates))

    if not state.candidates:
        return {"stage": "analysis_skipped"}

    quick_llm = get_quick_llm()
    deep_llm = get_deep_llm()
    data = UnifiedDataInterface()
    engine = DebateEngine(quick_llm)
    research_mgr = ResearchManager(deep_llm)

    analysts = [
        TechnicalAnalyst(quick_llm, data),
        FundamentalsAnalyst(quick_llm, data),
        FundFlowAnalyst(quick_llm, data),
        NewsSentimentAnalyst(quick_llm, data),
    ]

    def analyze_single(candidate) -> tuple[str, list, DebateResult, Any]:
        code = candidate.code
        name = candidate.name

        reports = []
        for a in analysts:
            try:
                report = a.analyze(code)
                reports.append(report)
            except Exception as e:
                logger.warning("%s 分析师 %s 失败: %s", code, a.analyst_type, e)

        if len(reports) < 2:
            return code, reports, DebateResult(code=code, name=name), ResearchVerdict(
                code=code, name=name, direction="hold", confidence=0.0,
                core_reasoning="分析报告不足",
            )

        try:
            debate = engine.debate(code, name, reports)
        except Exception as e:
            logger.warning("%s 辩论失败: %s", code, e)
            debate = DebateResult(code=code, name=name)

        try:
            records = state.daily_data.get(code, [])
            price = records[-1].close if records else 0
            verdict = research_mgr.decide(code, name, reports, debate, price)
        except Exception as e:
            logger.warning("%s 研究主管研判失败: %s", code, e)
            verdict = ResearchVerdict(code=code, name=name, direction="hold", confidence=0.0, core_reasoning=str(e))

        return code, reports, debate, verdict

    analyst_reports: dict = {}
    debates: dict = {}
    verdicts: dict = {}
    errors: list[str] = list(state.errors)

    max_workers = max(1, min(len(state.candidates), 4))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(analyze_single, c): c for c in state.candidates}
        for future in as_completed(futures):
            try:
                code, reports, debate, verdict = future.result()
                analyst_reports[code] = reports
                debates[code] = debate
                verdicts[code] = verdict
            except Exception as e:
                c = futures[future]
                logger.exception("%s 分析全链路失败: %s", c.code, e)
                errors.append(f"{c.code} 分析失败: {e}")
        del futures

    elapsed = dict(state.elapsed)
    elapsed["analysis"] = time.monotonic() - t0
    logger.info("分析完成: %d 只有效研判", len(verdicts))

    return {
        "analyst_reports": analyst_reports,
        "debates": debates,
        "verdicts": verdicts,
        "errors": errors,
        "stage": "analysis_done",
        "elapsed": elapsed,
    }


def run_risk(state: PipelineState) -> dict[str, Any]:
    """阶段 3: 风控 — 计算仓位上限"""
    t0 = time.monotonic()
    logger.info("===== 阶段 3/4: 风控计算 =====")

    cfg = get_config()

    verdicts = list(state.verdicts.values())
    if not verdicts:
        return {"stage": "risk_skipped"}

    data = UnifiedDataInterface()
    codes = [v.code for v in verdicts]
    stock_infos = data.batch_stock_info(codes)
    industry_map: dict[str, str] = {}
    for code, info in stock_infos.items():
        industry = info.get("industry", "")
        if industry:
            industry_map[code] = industry
    if industry_map:
        logger.info("行业映射: %d 只", len(industry_map))

    risk_mgr = RiskManager(total_capital=state.total_capital)
    # 传入当前已买入股票作为 current_positions，使行业集中度检查生效
    current_positions = _build_current_positions(state)

    # 获取限售解禁数据
    unlock_map: dict[str, float] = {}
    try:
        unlocks = data.get_unlock_shares(days_ahead=30)
        for u in unlocks:
            if u.unlock_ratio > 0.005:
                unlock_map[u.code] = u.unlock_ratio
        if unlock_map:
            logger.info("限售解禁: %d 只股票有近期解禁风险", len(unlock_map))
    except Exception:
        logger.debug("限售解禁数据获取失败，跳过")

    # 可选: 加载 RL 模型生成交易信号
    rl_signals = None
    if cfg.rl_enabled or cfg.rl_model_path:
        try:
            from ..rl.agent import DQNAgent
            agent = DQNAgent()
            model_path = cfg.rl_model_path or os.path.join(cfg.results_dir, "rl_model.json")
            if os.path.exists(model_path):
                agent.load(model_path)
                rl_signals = {}
                for v in verdicts:
                    records = state.daily_data.get(v.code, [])
                    if len(records) >= 20:
                        signal = agent.infer(records)
                        conf = agent.get_q_confidence(records)
                        rl_signals[v.code] = (signal, conf)
                logger.info("RL 信号已生成: %d 只股票", len(rl_signals))
            else:
                logger.debug("RL 模型文件 %s 不存在，跳过", model_path)
        except Exception:
            logger.debug("RL 信号生成失败", exc_info=True)

    limits = risk_mgr.compute_limits(
        verdicts, state.daily_data, current_positions,
        industry_map=industry_map or None,
        unlock_shares=unlock_map or None,
        rl_signals=rl_signals,
    )

    elapsed = dict(state.elapsed)
    elapsed["risk"] = time.monotonic() - t0

    buy_count = sum(1 for l in limits.values() if l.max_shares > 0)
    logger.info("风控完成: %d 只可买入", buy_count)

    return {
        "position_limits": limits,
        "stage": "risk_done",
        "elapsed": elapsed,
    }


def run_portfolio(state: PipelineState) -> dict[str, Any]:
    """阶段 4: 组合构建 — 最终买卖决策 (股票 + ETF 混合)"""
    t0 = time.monotonic()
    cfg = get_config()
    logger.info("===== 阶段 4/4: 组合构建 =====")

    stock_verdicts = list(state.verdicts.values())
    etf_verdicts = list(state.etf_verdicts.values())
    has_stock = bool(stock_verdicts and state.position_limits)
    has_etf = bool(etf_verdicts and state.etf_position_limits)

    if not has_stock and not has_etf:
        return {"stage": "portfolio_skipped", "final_result": PortfolioResult(decisions=[])}

    deep_llm = get_deep_llm()
    portfolio_mgr = PortfolioManager(deep_llm)

    # 使用实际可用现金 (由 main.py 从 tracker 加载, 跨日后不再是 50 万)
    current_positions = _build_current_positions(state)
    cash_available = max(0.0, state.available_cash)

    # ETF 专用资金比例 (基于实际可用现金)
    etf_budget = cash_available * cfg.etf_max_allocation if has_etf else 0.0
    stock_budget = cash_available - etf_budget

    all_decisions: list = []
    total_cash_used = 0.0

    # ETF 分配
    if has_etf:
        etf_result = portfolio_mgr.construct_etf(
            etf_verdicts, state.etf_position_limits, state.daily_data,
            cash_available=min(etf_budget, cash_available),
            total_capital=state.total_capital,
        )
        all_decisions.extend(etf_result.decisions)
        total_cash_used += etf_result.cash_used

    # 股票分配
    if has_stock:
        stock_cash = cash_available - total_cash_used
        stock_result = portfolio_mgr.construct(
            stock_verdicts, state.position_limits, state.daily_data,
            cash_available=max(0, stock_cash),
            total_capital=state.total_capital,
        )
        all_decisions.extend(stock_result.decisions)
        total_cash_used += stock_result.cash_used

    final = PortfolioResult(
        decisions=all_decisions,
        cash_used=total_cash_used,
        cash_remaining=cash_available - total_cash_used,
        total_positions=len(all_decisions),
    )

    elapsed = dict(state.elapsed)
    elapsed["portfolio"] = time.monotonic() - t0

    logger.info("组合构建完成: %d 笔决策 (股票 %d + ETF %d), 使用资金 ¥%.0f",
                len(all_decisions),
                sum(1 for d in all_decisions if d.asset_type == "stock"),
                sum(1 for d in all_decisions if d.asset_type == "etf"),
                total_cash_used)

    return {
        "final_result": final,
        "stage": "done",
        "elapsed": elapsed,
    }


# ── 条件路由 ─────────────────────────────────

def _after_screening(state: PipelineState) -> str:
    """有候选 → ETF 筛选; 否则 → 结束"""
    if state.candidates and state.stage != "screening_failed":
        cfg = get_config()
        if cfg.etf_enabled:
            return "etf_screening"
        return "analysis"
    logger.warning("筛选无候选，流水线终止")
    return END


def _after_etf_screening(state: PipelineState) -> str:
    """ETF 筛选完成 → ETF 分析 或跳过"""
    if state.etf_candidates:
        return "etf_analysis"
    return "analysis"


def _after_etf_analysis(state: PipelineState) -> str:
    """ETF 分析完成 → ETF 风控 或跳过"""
    if state.etf_verdicts:
        return "etf_risk"
    return "analysis"


def _after_etf_risk(state: PipelineState) -> str:
    """ETF 风控完成 → 股票分析"""
    return "analysis"


def _after_analysis(state: PipelineState) -> str:
    """有研判 → 风控; 否则 → 结束"""
    if state.verdicts or state.etf_verdicts:
        return "risk"
    logger.warning("分析无有效研判，流水线终止")
    return END


def _after_risk(state: PipelineState) -> str:
    """有仓位限制 → 组合; 否则 → 结束"""
    if state.position_limits:
        return "portfolio"
    logger.warning("风控无有效仓位限制，流水线终止")
    return END


# ── 图构建 ────────────────────────────────────

def _build_graph() -> StateGraph:
    """构建 LangGraph StateGraph (含 ETF 并行流水线)"""
    workflow = StateGraph(PipelineState)

    # 股票流水线节点
    workflow.add_node("screening", run_screening)
    workflow.add_node("analysis", run_analysis)
    workflow.add_node("risk", run_risk)
    workflow.add_node("portfolio", run_portfolio)

    # ETF 流水线节点
    workflow.add_node("etf_screening", run_etf_screening)
    workflow.add_node("etf_analysis", run_etf_analysis)
    workflow.add_node("etf_risk", run_etf_risk)

    workflow.set_entry_point("screening")

    # screening → etf_screening (or analysis)
    workflow.add_conditional_edges("screening", _after_screening, {
        "etf_screening": "etf_screening",
        "analysis": "analysis",
        END: END,
    })

    # etf_screening → etf_analysis (or skip to analysis)
    workflow.add_conditional_edges("etf_screening", _after_etf_screening, {
        "etf_analysis": "etf_analysis",
        "analysis": "analysis",
    })

    # etf_analysis → etf_risk (or skip to analysis)
    workflow.add_conditional_edges("etf_analysis", _after_etf_analysis, {
        "etf_risk": "etf_risk",
        "analysis": "analysis",
    })

    # etf_risk → analysis
    workflow.add_conditional_edges("etf_risk", _after_etf_risk, {
        "analysis": "analysis",
    })

    # analysis → risk (or END)
    workflow.add_conditional_edges("analysis", _after_analysis, {
        "risk": "risk",
        END: END,
    })

    # risk → portfolio (or END)
    workflow.add_conditional_edges("risk", _after_risk, {
        "portfolio": "portfolio",
        END: END,
    })
    workflow.add_edge("portfolio", END)

    return workflow.compile()


# 模块级编译缓存
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


# ── 流水线入口 ─────────────────────────────────

def run_pipeline(
    total_capital: float = 500_000.0,
    available_cash: float = 500_000.0,
    current_holdings: dict[str, int] | None = None,
) -> PipelineState:
    """
    执行完整的日内投资决策流水线。

    基于 LangGraph StateGraph 编排，支持条件路由:
      screening → analysis → risk → portfolio → END
                          ↑ 无候选/无研判/无仓位时提前终止

    参数:
        total_capital: 初始总资金 (用于报告展示)
        available_cash: 当前实际可用现金 (跨日后可能 < total_capital)
        current_holdings: 当前持仓 {code: shares}

    返回: 包含所有阶段结果的 PipelineState
    """
    initial_state = PipelineState(
        total_capital=total_capital,
        available_cash=available_cash,
        current_holdings=current_holdings or {},
    )
    app = _get_graph()
    result = app.invoke(initial_state)

    # LangGraph 可能返回 dict，统一转回 PipelineState
    if isinstance(result, dict):
        from dataclasses import fields
        field_names = {f.name for f in fields(PipelineState)}
        result = PipelineState(**{k: v for k, v in result.items() if k in field_names})

    total_elapsed = sum(result.elapsed.values())
    logger.info("===== 流水线完成 (%.1fs) =====", total_elapsed)
    _print_summary(result)

    return result


# ── 辅助 ──────────────────────────────────────

def _build_current_positions(state: PipelineState) -> dict[str, int]:
    """加载当前持仓: 优先使用 state.current_holdings (由 main.py 传入), 否则从文件读取"""
    if state.current_holdings:
        logger.info("已加载持仓 (state): %d 只", len(state.current_holdings))
        return state.current_holdings

    from ..agents.portfolio_tracker import PortfolioTracker
    from ..utils.config import get_config

    config = get_config()
    tracker = PortfolioTracker(
        total_capital=state.total_capital,
        results_dir=config.results_dir,
    )
    tracker.load()
    positions = tracker.current_positions_dict()
    if positions:
        logger.info("已加载持仓 (文件): %d 只, 现金 %.0f", len(positions), tracker.cash)
    return positions


def _print_summary(state) -> None:
    """打印流水线摘要"""
    final = getattr(state, "final_result", None)
    if final:
        decisions = final.decisions
        if decisions:
            logger.info("最终决策:")
            for d in decisions:
                logger.info("  %s %s %d股", d.symbol, d.symbol_name, d.volume)
        else:
            logger.info("最终决策: 空仓 (无可买入标的)")
