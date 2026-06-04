"""
持仓复盘模块 — 盘后审查现有持仓合理性、风控合规、事后验证与绩效归因。

使用方式:
    from src.review import PortfolioReviewer, run_review

    # 方式1: 类方式
    reviewer = PortfolioReviewer(tracker, data_interface, llm)
    result = reviewer.review()

    # 方式2: 便捷函数
    result = run_review(tracker)
"""

from .engine import PortfolioReviewer, ReviewResult, PositionReview, RiskLine, PostMortemSummary, AttributionResult

__all__ = [
    "PortfolioReviewer",
    "ReviewResult",
    "PositionReview",
    "RiskLine",
    "PostMortemSummary",
    "AttributionResult",
    "run_review",
]


def run_review(tracker, data_interface=None, llm=None) -> "ReviewResult":
    """便捷函数: 执行完整持仓复盘。"""
    reviewer = PortfolioReviewer(tracker, data_interface, llm)
    return reviewer.review()
