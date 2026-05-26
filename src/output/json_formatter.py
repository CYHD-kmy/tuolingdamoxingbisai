"""
赛道 JSON 格式化器 — 输出符合比赛要求的决策格式。

使用 src.utils.validators 中的共享校验逻辑。
"""

from __future__ import annotations

import logging
from typing import Any

from ..agents.models import FinalDecision, PositionLimit
from ..utils.validators import validate_and_clip, LOT_SIZE

logger = logging.getLogger(__name__)


def format_decisions(decisions: list[FinalDecision]) -> list[dict[str, Any]]:
    """将决策列表转为赛道标准 JSON 格式"""
    return [d.to_dict() for d in decisions]


def validate_decisions(
    decisions: list[FinalDecision],
    limits: dict[str, PositionLimit],
    daily_data: dict[str, list[Any]],
    cash_available: float,
    min_cash_reserve: float = 0.10,
    total_capital: float = 500_000.0,
    suspended_codes: set[str] | None = None,
) -> list[FinalDecision]:
    """硬约束校验并裁剪决策。委托到共享校验模块。"""
    return validate_and_clip(
        decisions, limits, daily_data,
        cash_available=cash_available,
        total_capital=total_capital,
        min_cash_reserve=min_cash_reserve,
        suspended_codes=suspended_codes,
    )
