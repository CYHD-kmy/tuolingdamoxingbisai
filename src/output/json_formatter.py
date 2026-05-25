"""
赛道 JSON 格式化器 — 输出符合比赛要求的决策格式 + 硬约束校验。

约束校验:
  - volume 必须是 100 的整数倍 → 自动向下取整
  - 总买入金额 ≤ 可用现金 → 按优先级截断
  - 单票不超风控上限 → 自动裁剪
  - 标的必须可交易 → 交叉验证停牌列表
"""

from __future__ import annotations

import logging
from typing import Any

from ..agents.models import FinalDecision, PositionLimit

logger = logging.getLogger(__name__)

LOT_SIZE = 100


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
    """
    硬约束校验并裁剪决策。

    返回: 通过校验的有效决策列表
    """
    if not decisions:
        return []

    valid: list[FinalDecision] = []
    total_cost = 0.0
    min_cash = total_capital * min_cash_reserve
    suspended = suspended_codes or set()

    for d in decisions:
        # 1. 停牌检查
        if d.symbol in suspended:
            logger.warning("json_formatter: %s 已停牌，跳过", d.symbol)
            continue

        # 2. volume 向下取整到 100 的倍数
        d.volume = d.volume // LOT_SIZE * LOT_SIZE
        if d.volume <= 0:
            logger.warning("json_formatter: %s volume=0，跳过", d.symbol)
            continue

        # 3. 获取最新价
        price = _get_price(d.symbol, daily_data)
        if price <= 0:
            logger.warning("json_formatter: %s 无有效价格，跳过", d.symbol)
            continue

        # 4. 不超过风控上限
        limit = limits.get(d.symbol)
        if limit and limit.max_shares > 0 and d.volume > limit.max_shares:
            logger.info("json_formatter: %s 裁剪 %d→%d (风控上限)", d.symbol, d.volume, limit.max_shares)
            d.volume = limit.max_shares

        # 5. 预算检查
        cost = d.volume * price
        remaining = cash_available - min_cash - total_cost

        if cost > remaining:
            new_volume = int(remaining / price / LOT_SIZE) * LOT_SIZE
            if new_volume >= LOT_SIZE:
                logger.info("json_formatter: %s 裁剪 %d→%d (超预算)", d.symbol, d.volume, new_volume)
                d.volume = new_volume
                total_cost += d.volume * price
                valid.append(d)
            else:
                logger.info("json_formatter: %s 跳过 (预算不足)", d.symbol)
            break  # 后续不再处理
        else:
            total_cost += cost
            valid.append(d)

    logger.info("json_formatter: 校验完成, %d→%d 笔有效决策, 总成本 ¥%.0f",
                len(decisions), len(valid), total_cost)
    return valid


def _get_price(code: str, daily_data: dict[str, list[Any]]) -> float:
    records = daily_data.get(code, [])
    if not records:
        return 0.0
    return records[-1].close
