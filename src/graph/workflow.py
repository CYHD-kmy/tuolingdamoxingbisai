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
from ..agents.researchers.engine import DebateEngine
from ..agents.managers.research_manager import ResearchManager
from ..agents.managers.risk_manager import RiskManager
from ..agents.managers.portfolio_manager import PortfolioManager
from ..utils.validators import get_latest_price

logger = logging.getLogger(__name__)


# ── 工作流节点 ─────────────────────────────────

def run_screening(state: PipelineState) -> dict[str, Any]:
    """阶段 1: 海选筛选 — 全市场 → Top-N 候选"""
    t0 = time.monotonic()
    logger.info("===== 阶段 1/4: 海选筛选 =====")

    data = UnifiedDataInterface()
    pipeline = ScreeningPipeline(data)

    try:
        result = pipeline.run()
        updates: dict[str, Any] = {
            "candidates": result.candidates,
            "errors": state.errors + result.errors,
        }

        codes = [c.code for c in result.candidates]
        if codes:
            updates["daily_data"] = data.batch_daily_data(codes, days=30, max_workers=6)
            updates["fund_flows"] = data.batch_fund_flows(codes, days=5, max_workers=6)

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
    limits = risk_mgr.compute_limits(
        verdicts, state.daily_data, current_positions,
        industry_map=industry_map if industry_map else None,
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
    """阶段 4: 组合构建 — 最终买卖决策"""
    t0 = time.monotonic()
    logger.info("===== 阶段 4/4: 组合构建 =====")

    verdicts = list(state.verdicts.values())
    if not verdicts or not state.position_limits:
        return {"stage": "portfolio_skipped", "final_result": PortfolioResult(decisions=[])}

    deep_llm = get_deep_llm()
    portfolio_mgr = PortfolioManager(deep_llm)

    result = portfolio_mgr.construct(
        verdicts, state.position_limits, state.daily_data,
        cash_available=state.total_capital, total_capital=state.total_capital,
    )

    elapsed = dict(state.elapsed)
    elapsed["portfolio"] = time.monotonic() - t0

    return {
        "final_result": result,
        "stage": "done",
        "elapsed": elapsed,
    }


# ── 条件路由 ─────────────────────────────────

def _after_screening(state: PipelineState) -> str:
    """有候选 → 分析; 否则 → 结束"""
    if state.candidates and state.stage != "screening_failed":
        return "analysis"
    logger.warning("筛选无候选，流水线终止")
    return END


def _after_analysis(state: PipelineState) -> str:
    """有研判 → 风控; 否则 → 结束"""
    if state.verdicts:
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
    """构建 LangGraph StateGraph"""
    workflow = StateGraph(PipelineState)

    workflow.add_node("screening", run_screening)
    workflow.add_node("analysis", run_analysis)
    workflow.add_node("risk", run_risk)
    workflow.add_node("portfolio", run_portfolio)

    workflow.set_entry_point("screening")

    workflow.add_conditional_edges("screening", _after_screening, {
        "analysis": "analysis",
        END: END,
    })
    workflow.add_conditional_edges("analysis", _after_analysis, {
        "risk": "risk",
        END: END,
    })
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

def run_pipeline(total_capital: float = 500_000.0) -> PipelineState:
    """
    执行完整的日内投资决策流水线。

    基于 LangGraph StateGraph 编排，支持条件路由:
      screening → analysis → risk → portfolio → END
                          ↑ 无候选/无研判/无仓位时提前终止

    返回: 包含所有阶段结果的 PipelineState
    """
    initial_state = PipelineState(total_capital=total_capital)
    app = _get_graph()
    result = app.invoke(initial_state)

    total_elapsed = sum(result.elapsed.values()) if hasattr(result, "elapsed") else 0
    logger.info("===== 流水线完成 (%.1fs) =====", total_elapsed)
    _print_summary(result)

    return result


# ── 辅助 ──────────────────────────────────────

def _build_current_positions(state: PipelineState) -> dict[str, int]:
    """从已有最终结果中提取当前持仓 {code: shares}"""
    if state.final_result and state.final_result.decisions:
        return {d.symbol: d.volume for d in state.final_result.decisions}
    return {}


def _print_summary(state: PipelineState) -> None:
    """打印流水线摘要"""
    if state.final_result:
        decisions = state.final_result.decisions
        if decisions:
            logger.info("最终决策:")
            for d in decisions:
                logger.info("  %s %s %d股", d.symbol, d.symbol_name, d.volume)
        else:
            logger.info("最终决策: 空仓 (无可买入标的)")
