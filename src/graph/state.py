"""
工作流共享状态 — 贯穿整个 LangGraph 流水线。

每个节点读取并更新此状态，实现模块间的数据传递。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..screening.scorer import FactorScore
from ..agents.base import AnalystReport
from ..agents.models import (
    DebateResult, ResearchVerdict, PositionLimit, PortfolioResult,
)


@dataclass
class PipelineState:
    """
    流水线全局状态。

    LangGraph 通过此 dataclass 在节点间传递数据。
    每个字段对应流水线的一个阶段的输出。
    """

    # ── 配置 ──────────────────────────────────
    date: str = ""
    total_capital: float = 500_000.0
    available_cash: float = 500_000.0       # 实际可用现金 (跨日后可能 < total_capital)
    current_holdings: dict[str, int] = field(default_factory=dict)  # {code: shares} 当前持仓

    # ── 阶段 1: 海选筛选 ─────────────────────
    candidates: list[FactorScore] = field(default_factory=list)
    daily_data: dict[str, list[Any]] = field(default_factory=dict)
    fund_flows: dict[str, list[Any]] = field(default_factory=dict)

    # ── 阶段 2: 深度分析 ─────────────────────
    # {code: [technical_report, fundamentals_report, fund_flow_report, news_report]}
    analyst_reports: dict[str, list[AnalystReport]] = field(default_factory=dict)
    # {code: DebateResult}
    debates: dict[str, DebateResult] = field(default_factory=dict)
    # {code: ResearchVerdict}
    verdicts: dict[str, ResearchVerdict] = field(default_factory=dict)

    # ── ETF 并行流水线 ───────────────────────
    etf_candidates: list[Any] = field(default_factory=list)
    etf_analyst_reports: dict[str, list[AnalystReport]] = field(default_factory=dict)
    etf_verdicts: dict[str, ResearchVerdict] = field(default_factory=dict)
    etf_position_limits: dict[str, PositionLimit] = field(default_factory=dict)

    # ── 阶段 3: 风控与决策 ──────────────────
    position_limits: dict[str, PositionLimit] = field(default_factory=dict)
    final_result: PortfolioResult | None = None

    # ── 市场环境 (增强版) ──────────────────────
    market_regime: str = "neutral"  # "bull" / "neutral" / "bear"
    market_sentiment: Any = None    # MarketSentimentResult
    sector_heats: list[Any] = field(default_factory=list)  # 板块热度排行
    auction_signals: list[Any] = field(default_factory=list)  # 集合竞价强势信号
    limit_up_signals: list[Any] = field(default_factory=list)  # 涨停信号
    dragon_tiger_signals: list[Any] = field(default_factory=list)  # 龙虎榜信号

    # ── 竞赛评分 ──────────────────────────────
    # {code: {"analyst_votes": int, "consensus_score": float, "passed": bool}}
    competition_scores: dict[str, dict] = field(default_factory=dict)

    # ── 元信息 ────────────────────────────────
    stage: str = "init"
    errors: list[str] = field(default_factory=list)
    elapsed: dict[str, float] = field(default_factory=dict)  # {stage: seconds}
    data_quality: dict[str, str] = field(default_factory=dict)  # {code:data_type → quality}
