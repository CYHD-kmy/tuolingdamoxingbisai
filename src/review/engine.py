"""
持仓复盘引擎 — 编排四个子模块执行完整复盘。

数据流:
    PortfolioReviewer.review()
        ├─→ RiskChecker.check()        # P0: 风控红线
        ├─→ PositionScorer.score()      # P1: 合理性评分
        ├─→ PostMortem.analyze()        # P2: 卖飞检测
        └─→ Attribution.decompose()     # P3: 绩效归因
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


# ── 数据模型 ───────────────────────────────

@dataclass
class RiskLine:
    """单条风控红线检查结果 (借鉴金策智算门下省)"""
    rule_id: str           # 规则编号 (R1-R7)
    rule_name: str          # 规则名称
    value: float            # 当前值
    threshold: float        # 阈值
    status: str             # "pass" / "warn" / "violation"
    message: str            # 人类可读的描述


@dataclass
class PositionReview:
    """单只持仓的复盘结果"""
    code: str
    name: str
    # 评分
    current_score: float            # 当前综合得分 (0-100)
    score_change: float             # 相对建仓时的得分变化
    factor_detail: dict[str, float] = field(default_factory=dict)  # 各因子得分
    # LLM 研判
    recommendation: str = ""        # "hold" / "reduce" / "clear"
    confidence: float = 0.0         # LLM 置信度
    reasoning: str = ""             # LLM 推理
    # 持仓状态
    pnl_pct: float = 0.0
    holding_days: int = 0
    position_pct: float = 0.0       # 仓位占比


@dataclass
class PostMortemSummary:
    """已平仓事后验证汇总"""
    total_sells: int = 0
    sell_too_early: int = 0         # 卖出后上涨 > 3%
    correct_stops: int = 0          # 卖出后继续下跌 > 3%
    neutral: int = 0                # 卖出后波动在 ±3% 内
    avg_missed_gain_pct: float = 0.0   # 平均卖飞幅度
    avg_avoided_loss_pct: float = 0.0  # 平均避免的损失
    details: list[dict] = field(default_factory=list)


@dataclass
class AttributionResult:
    """绩效归因分解"""
    total_return_pct: float = 0.0
    selection_contribution: float = 0.0     # 选股贡献
    timing_contribution: float = 0.0        # 择时贡献
    industry_contribution: float = 0.0      # 行业配置贡献
    residual: float = 0.0                   # 残差


@dataclass
class ReviewResult:
    """完整复盘结果"""
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d"))
    # P0: 风控红线
    risk_checks: dict[str, list[RiskLine]] = field(default_factory=dict)
    risk_summary: str = ""
    risk_violation_count: int = 0
    risk_warning_count: int = 0
    # P1: 持仓评分
    position_scores: dict[str, PositionReview] = field(default_factory=dict)
    score_summary: str = ""
    # P2: 卖飞检测
    post_mortem: PostMortemSummary | None = None
    # P3: 绩效归因
    attribution: AttributionResult | None = None

    def to_dict(self) -> dict:
        """序列化为 API/JSON 兼容的字典"""
        d: dict[str, Any] = {
            "date": self.date,
            "risk_summary": self.risk_summary,
            "risk_violation_count": self.risk_violation_count,
            "risk_warning_count": self.risk_warning_count,
            "risk_checks": {},
            "position_scores": {},
            "score_summary": self.score_summary,
        }
        for code, lines in self.risk_checks.items():
            d["risk_checks"][code] = [
                {"rule_id": rl.rule_id, "rule_name": rl.rule_name,
                 "value": rl.value, "threshold": rl.threshold,
                 "status": rl.status, "message": rl.message}
                for rl in lines
            ]
        for code, pr in self.position_scores.items():
            d["position_scores"][code] = {
                "name": pr.name,
                "current_score": pr.current_score,
                "score_change": pr.score_change,
                "factor_detail": pr.factor_detail,
                "recommendation": pr.recommendation,
                "confidence": pr.confidence,
                "reasoning": pr.reasoning,
                "pnl_pct": pr.pnl_pct,
                "holding_days": pr.holding_days,
                "position_pct": pr.position_pct,
            }
        if self.post_mortem:
            d["post_mortem"] = {
                "total_sells": self.post_mortem.total_sells,
                "sell_too_early": self.post_mortem.sell_too_early,
                "correct_stops": self.post_mortem.correct_stops,
                "neutral": self.post_mortem.neutral,
                "avg_missed_gain_pct": self.post_mortem.avg_missed_gain_pct,
                "avg_avoided_loss_pct": self.post_mortem.avg_avoided_loss_pct,
            }
        if self.attribution:
            d["attribution"] = {
                "total_return_pct": self.attribution.total_return_pct,
                "selection_contribution": self.attribution.selection_contribution,
                "timing_contribution": self.attribution.timing_contribution,
                "industry_contribution": self.attribution.industry_contribution,
                "residual": self.attribution.residual,
            }
        return d


# ── 核心引擎 ─────────────────────────────────

class PortfolioReviewer:
    """
    持仓复盘引擎 — 对现有持仓执行风控红线和合理性审查。

    使用方式:
        reviewer = PortfolioReviewer(tracker, data_interface)
        result = reviewer.review()
    """

    def __init__(self, tracker, data_interface=None, llm=None,
                 market_regime: str = "neutral"):
        """
        tracker: PortfolioTracker 实例 (持仓状态)
        data_interface: UnifiedDataInterface (可选, 用于获取最新数据)
        llm: LLMClient (可选, 用于 P1 合理性评分)
        market_regime: "bull" / "neutral" / "bear"
        """
        self._tracker = tracker
        self._data = data_interface
        self._llm = llm
        self._regime = market_regime

    def review(self) -> ReviewResult:
        """执行完整复盘: P1 → P0 (用P1评分做分层) → P2 → P3"""
        date = datetime.now().strftime("%Y%m%d")
        result = ReviewResult(date=date)
        codes = list(self._tracker.positions.keys())

        # 获取市场数据 (R8/R9 + 分层分类)
        market_data = None
        stock_infos: dict[str, dict] = {}
        if self._data:
            try:
                market_data = self._data.get_market_snapshot()
            except Exception:
                logger.debug("获取市场快照失败")
            if codes:
                try:
                    stock_infos = self._data.batch_stock_info(codes, max_workers=4)
                except Exception:
                    logger.debug("获取股票信息失败")

        # P1: 先评分 (为P0分层提供依据)
        scores_for_tiers: dict[str, float] = {}
        if self._data and self._tracker.positions:
            try:
                from .position_scorer import PositionScorer
                scorer = PositionScorer(self._data, self._tracker, self._llm)
                score_result = scorer.score_all()
                result.position_scores = score_result["positions"]
                result.score_summary = score_result["summary"]
                scores_for_tiers = {
                    code: pr.current_score
                    for code, pr in result.position_scores.items()
                }
                logger.info("P1 持仓评分: %d 只", len(result.position_scores))
            except Exception:
                logger.exception("P1 持仓评分失败")

        # P0: 风控红线（使用P1评分做分层分类）
        try:
            from .risk_checker import RiskChecker
            checker = RiskChecker(self._tracker, market_regime=self._regime)
            risk_result = checker.check(
                scores=scores_for_tiers,
                market_data=market_data,
                stock_infos=stock_infos,
            )
            result.risk_checks = risk_result["checks"]
            result.risk_summary = risk_result["summary"]
            result.risk_violation_count = risk_result["violation_count"]
            result.risk_warning_count = risk_result["warning_count"]
            logger.info("P0 风控红线: %d 违规, %d 警告",
                        result.risk_violation_count, result.risk_warning_count)
        except Exception:
            logger.exception("P0 风控红线检查失败")

        # P2: 卖飞检测（基于历史 trace）
        try:
            from .post_mortem import PostMortem
            pm = PostMortem(self._data, self._tracker)
            result.post_mortem = pm.analyze()
            if result.post_mortem and result.post_mortem.total_sells > 0:
                logger.info("P2 卖飞检测: %d 笔卖出, 卖飞 %d 笔",
                            result.post_mortem.total_sells,
                            result.post_mortem.sell_too_early)
        except Exception:
            logger.debug("P2 卖飞检测跳过", exc_info=True)

        # P3: 绩效归因
        try:
            from .attribution import Attribution
            attr = Attribution(self._tracker)
            result.attribution = attr.decompose()
            if result.attribution:
                logger.info("P3 绩效归因: 选股%+.2f%% 择时%+.2f%%",
                            result.attribution.selection_contribution,
                            result.attribution.timing_contribution)
        except Exception:
            logger.debug("P3 绩效归因跳过", exc_info=True)

        return result
