"""
输出校验器 — 对赛道 JSON 决策进行硬约束检查。

约束规则:
  - volume 必须是 100 的整数倍
  - 单票不超过风控仓位上限
  - 总买入金额不超过可用资金
  - 标的代码格式校验 (6位数字)
  - 保留最低现金比例
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..agents.models import FinalDecision, PositionLimit

logger = logging.getLogger(__name__)

LOT_SIZE = 100


def _compile_symbol_pattern():
    import re
    return re.compile(r"^\d{6}$")


VALID_SYMBOL_PATTERN = _compile_symbol_pattern()


@dataclass
class ValidationResult:
    """校验结果"""
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    corrected_decisions: list[FinalDecision] = field(default_factory=list)


def validate_symbol(symbol: str) -> bool:
    """校验股票代码格式: 6位数字"""
    if not VALID_SYMBOL_PATTERN:
        return len(symbol) == 6 and symbol.isdigit()
    return bool(VALID_SYMBOL_PATTERN.match(symbol))


def validate_volume(volume: int) -> tuple[int, str | None]:
    """
    校验并修正股数。

    返回: (修正后的股数, 警告信息或 None)
    """
    if volume <= 0:
        return 0, "volume 必须 > 0"

    corrected = volume // LOT_SIZE * LOT_SIZE
    if corrected == 0:
        return 0, f"volume={volume} 取整后为 0，已剔除"

    warning = None
    if volume != corrected:
        warning = f"volume={volume} 已取整为 {corrected} (100的整数倍)"

    return corrected, warning


def validate_single_decision(
    decision: FinalDecision,
    limits: dict[str, PositionLimit],
    daily_data: dict[str, list[Any]],
    suspended_codes: set[str] | None = None,
) -> tuple[bool, list[str], list[str]]:
    """
    校验单条决策。

    返回: (是否通过, 错误列表, 警告列表)
    """
    errors: list[str] = []
    warnings: list[str] = []

    # 1. 代码格式
    if not validate_symbol(decision.symbol):
        errors.append(f"股票代码格式错误: {decision.symbol}")
        return False, errors, warnings

    # 2. 停牌检查
    if suspended_codes and decision.symbol in suspended_codes:
        errors.append(f"{decision.symbol} {decision.symbol_name} 已停牌")
        return False, errors, warnings

    # 3. volume 校验
    corrected, warning = validate_volume(decision.volume)
    if corrected == 0:
        errors.append(f"{decision.symbol} volume 无效")
        return False, errors, warnings
    if warning:
        warnings.append(f"{decision.symbol}: {warning}")
    decision.volume = corrected

    # 4. 风控上限
    limit = limits.get(decision.symbol)
    if limit and limit.max_shares > 0 and decision.volume > limit.max_shares:
        warnings.append(
            f"{decision.symbol}: volume {decision.volume} 超过风控上限 {limit.max_shares}"
        )

    # 5. 价格有效性
    price = _get_price(decision.symbol, daily_data)
    if price <= 0:
        errors.append(f"{decision.symbol} 无有效价格数据")
        return False, errors, warnings

    return True, errors, warnings


def validate_all_decisions(
    decisions: list[FinalDecision],
    limits: dict[str, PositionLimit],
    daily_data: dict[str, list[Any]],
    cash_available: float,
    total_capital: float = 500_000.0,
    min_cash_reserve: float = 0.10,
    suspended_codes: set[str] | None = None,
) -> ValidationResult:
    """
    对所有决策进行硬约束校验并裁剪。

    返回包含错误、警告和修正后决策列表的 ValidationResult。
    """
    result = ValidationResult()

    if not decisions:
        result.valid = True
        return result

    valid_decisions: list[FinalDecision] = []
    total_cost = 0.0
    min_cash = total_capital * min_cash_reserve

    for d in decisions:
        ok, errors, warnings = validate_single_decision(d, limits, daily_data, suspended_codes)
        result.errors.extend(errors)
        result.warnings.extend(warnings)

        if not ok:
            continue

        price = _get_price(d.symbol, daily_data)
        cost = d.volume * price
        remaining = cash_available - min_cash - total_cost

        # 预算检查
        if cost > remaining:
            new_volume = int(remaining / price / LOT_SIZE) * LOT_SIZE
            if new_volume >= LOT_SIZE:
                result.warnings.append(
                    f"{d.symbol}: 超预算裁剪 {d.volume}→{new_volume}"
                )
                d.volume = new_volume
                total_cost += d.volume * price
                valid_decisions.append(d)
            else:
                result.warnings.append(f"{d.symbol}: 预算不足，跳过")
            break  # 预算耗尽，后续不再处理
        else:
            total_cost += cost
            valid_decisions.append(d)

    result.corrected_decisions = valid_decisions
    result.valid = len(result.errors) == 0

    logger.info(
        "validators: %d→%d 笔有效, 总成本 ¥%.0f, %d 错误, %d 警告",
        len(decisions), len(valid_decisions), total_cost,
        len(result.errors), len(result.warnings),
    )

    return result


def _get_price(code: str, daily_data: dict[str, list[Any]]) -> float:
    records = daily_data.get(code, [])
    if not records:
        return 0.0
    return records[-1].close
