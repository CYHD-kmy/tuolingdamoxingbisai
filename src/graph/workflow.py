"""
LangGraph 工作流 — 将数据/筛选/分析/风控/决策串联为完整的日内流水线。

节点:
  screening → analyze_all → risk → portfolio → finalize

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
        state.candidates = result.candidates
        state.errors.extend(result.errors)

        # 批量获取候选股的日线和资金数据 (后续阶段复用)
        codes = [c.code for c in result.candidates]
        if codes:
            state.daily_data = data.batch_daily_data(codes, days=30, max_workers=6)
            state.fund_flows = data.batch_fund_flows(codes, days=5, max_workers=6)

        state.stage = "screening_done"
        logger.info("筛选完成: %d 只候选", len(result.candidates))
    except Exception as e:
        logger.exception("筛选阶段异常")
        state.errors.append(f"筛选失败: {e}")
        state.stage = "screening_failed"

    state.elapsed["screening"] = time.monotonic() - t0
    return {
        "candidates": state.candidates,
        "daily_data": state.daily_data,
        "fund_flows": state.fund_flows,
        "errors": state.errors,
        "stage": state.stage,
        "elapsed": state.elapsed,
    }


def run_analysis(state: PipelineState) -> dict[str, Any]:
    """阶段 2: 深度分析 — 四分析师 + 辩论 + 研究主管研判"""
    t0 = time.monotonic()
    logger.info("===== 阶段 2/4: 深度分析 (%d 只) =====", len(state.candidates))

    if not state.candidates:
        state.stage = "analysis_skipped"
        return {"stage": state.stage}

    quick_llm = get_quick_llm()
    deep_llm = get_deep_llm()
    data = UnifiedDataInterface()
    engine = DebateEngine(quick_llm)
    research_mgr = ResearchManager(deep_llm)

    # 创建分析师实例
    analysts = [
        TechnicalAnalyst(quick_llm, data),
        FundamentalsAnalyst(quick_llm, data),
        FundFlowAnalyst(quick_llm, data),
        NewsSentimentAnalyst(quick_llm, data),
    ]

    def analyze_single(candidate) -> tuple[str, list, DebateResult, Any]:
        """对单只股票执行完整分析链"""
        code = candidate.code
        name = candidate.name

        # 1. 四维分析师并行分析
        reports = []
        for a in analysts:
            try:
                report = a.analyze(code)
                reports.append(report)
            except Exception as e:
                logger.warning("%s 分析师 %s 失败: %s", code, a.analyst_type, e)

        if len(reports) < 2:
            logger.warning("%s: 有效分析报告不足 2 份，跳过", code)
            return code, reports, DebateResult(code=code, name=name), ResearchVerdict(
                code=code, name=name, direction="hold", confidence=0.0,
                core_reasoning="分析报告不足",
            )

        # 2. 辩论
        try:
            debate = engine.debate(code, name, reports, max_rounds=2)
        except Exception as e:
            logger.warning("%s 辩论失败: %s", code, e)
            debate = DebateResult(code=code, name=name)

        # 3. 研究主管研判
        try:
            price = state.daily_data.get(code, [None])[-1].close if state.daily_data.get(code) else 0
            verdict = research_mgr.decide(code, name, reports, debate, price)
        except Exception as e:
            logger.warning("%s 研究主管研判失败: %s", code, e)
            verdict = ResearchVerdict(code=code, name=name, direction="hold", confidence=0.0, core_reasoning=str(e))

        return code, reports, debate, verdict

    # 并发分析所有候选 (每只股票内部的分析师和辩论串行，股票之间并行)
    max_workers = min(len(state.candidates), 4)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(analyze_single, c): c for c in state.candidates}
        for future in as_completed(futures):
            try:
                code, reports, debate, verdict = future.result()
                state.analyst_reports[code] = reports
                state.debates[code] = debate
                state.verdicts[code] = verdict
            except Exception as e:
                c = futures[future]
                logger.exception("%s 分析全链路失败: %s", c.code, e)
                state.errors.append(f"{c.code} 分析失败: {e}")

    state.stage = "analysis_done"
    state.elapsed["analysis"] = time.monotonic() - t0
    logger.info("分析完成: %d 只有效研判", len(state.verdicts))

    return {
        "analyst_reports": state.analyst_reports,
        "debates": state.debates,
        "verdicts": state.verdicts,
        "errors": state.errors,
        "stage": state.stage,
        "elapsed": state.elapsed,
    }


def run_risk(state: PipelineState) -> dict[str, Any]:
    """阶段 3: 风控 — 计算仓位上限"""
    t0 = time.monotonic()
    logger.info("===== 阶段 3/4: 风控计算 =====")

    verdicts = list(state.verdicts.values())
    if not verdicts:
        state.stage = "risk_skipped"
        return {"stage": state.stage}

    risk_mgr = RiskManager(total_capital=state.total_capital)
    limits = risk_mgr.compute_limits(verdicts, state.daily_data, {})

    state.position_limits = limits
    state.stage = "risk_done"
    state.elapsed["risk"] = time.monotonic() - t0

    buy_count = sum(1 for l in limits.values() if l.max_shares > 0)
    logger.info("风控完成: %d 只可买入", buy_count)

    return {
        "position_limits": state.position_limits,
        "stage": state.stage,
        "elapsed": state.elapsed,
    }


def run_portfolio(state: PipelineState) -> dict[str, Any]:
    """阶段 4: 组合构建 — 最终买卖决策"""
    t0 = time.monotonic()
    logger.info("===== 阶段 4/4: 组合构建 =====")

    verdicts = list(state.verdicts.values())
    if not verdicts or not state.position_limits:
        state.stage = "portfolio_skipped"
        return {"stage": state.stage, "final_result": PortfolioResult(decisions=[])}

    deep_llm = get_deep_llm()
    portfolio_mgr = PortfolioManager(deep_llm)

    cash = state.total_capital  # 简化: MVP 阶段默认全仓可用

    result = portfolio_mgr.construct(
        verdicts, state.position_limits, state.daily_data,
        cash_available=cash, total_capital=state.total_capital,
    )

    state.final_result = result
    state.stage = "done"
    state.elapsed["portfolio"] = time.monotonic() - t0

    return {
        "final_result": state.final_result,
        "stage": state.stage,
        "elapsed": state.elapsed,
    }


# ── 流水线入口 ─────────────────────────────────

def run_pipeline(total_capital: float = 500_000.0) -> PipelineState:
    """
    执行完整的日内投资决策流水线。

    返回: 包含所有阶段结果的 PipelineState
    """
    state = PipelineState(total_capital=total_capital)

    # 顺序执行各阶段 (MVP 阶段不使用 LangGraph 图结构，保持简单可调试)
    state_dict = run_screening(state)
    _apply(state, state_dict)

    state_dict = run_analysis(state)
    _apply(state, state_dict)

    state_dict = run_risk(state)
    _apply(state, state_dict)

    state_dict = run_portfolio(state)
    _apply(state, state_dict)

    total_elapsed = sum(state.elapsed.values())
    logger.info("===== 流水线完成 (%.1fs) =====", total_elapsed)
    _print_summary(state)

    return state


def _apply(state: PipelineState, updates: dict[str, Any]) -> None:
    """将更新字典应用到 state 对象"""
    for key, value in updates.items():
        if hasattr(state, key):
            setattr(state, key, value)


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
