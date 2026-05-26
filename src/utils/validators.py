"""
输出校验器 — 共享的硬约束校验逻辑。

校验规则:
  - volume 必须是 100 的整数倍 → 自动向下取整
  - 总买入金额 ≤ 可用现金 - 最低现金保留 → 按优先级截断
  - 单票不超风控上限 → 自动裁剪
  - 标的必须可交易 → 交叉验证停牌列表
"""

from __future__ import annotations

import logging
from typing import Any

LOT_SIZE = 100

logger = logging.getLogger(__name__)


def extract_json(text: str) -> str:
    """从 LLM 原始输出中提取 JSON 字符串。

    依次尝试:
    1. ```json ... ``` 代码块
    2. ``` ... ``` 代码块
    3. { ... } 或 [ ... ] 直接匹配
    4. 返回原始文本 (解析失败时由调用方兜底)
    """
    if not text:
        return text

    # 1. JSON 代码块
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        return text[start:end].strip()

    # 2. 普通代码块
    if "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        return text[start:end].strip()

    # 3. 直接匹配 JSON 对象/数组 (先数组后对象, 因为数组可含对象)
    if "[" in text and "]" in text:
        start = text.index("[")
        end = text.rindex("]") + 1
        return text[start:end]

    if "{" in text and "}" in text:
        start = text.index("{")
        end = text.rindex("}") + 1
        return text[start:end]

    return text


def get_latest_price(code: str, daily_data: dict[str, list[Any]]) -> float:
    """从日线数据中获取最新收盘价"""
    records = daily_data.get(code, [])
    if not records:
        return 0.0
    return records[-1].close


def validate_and_clip(
    decisions: list[Any],
    limits: dict[str, Any],
    daily_data: dict[str, list[Any]],
    cash_available: float,
    total_capital: float = 500_000.0,
    min_cash_reserve: float = 0.10,
    suspended_codes: set[str] | None = None,
) -> list[Any]:
    """
    硬约束校验并裁剪决策。

    返回: 通过校验的有效决策列表 (保持原始决策对象类型)
    """
    if not decisions:
        return []

    valid: list[Any] = []
    total_cost = 0.0
    min_cash = total_capital * min_cash_reserve
    suspended = suspended_codes or set()

    for d in decisions:
        # 1. 停牌检查
        code = d.symbol if hasattr(d, "symbol") else d.get("symbol", "")
        if code in suspended:
            logger.warning("validators: %s 已停牌，跳过", code)
            continue

        # 2. volume 向下取整到 100 的倍数
        volume = d.volume if hasattr(d, "volume") else d.get("volume", 0)
        volume = volume // LOT_SIZE * LOT_SIZE
        if volume <= 0:
            logger.warning("validators: %s volume=%d，跳过", code, volume)
            continue

        # 3. 获取最新价
        price = get_latest_price(code, daily_data)
        if price <= 0:
            logger.warning("validators: %s 无有效价格，跳过", code)
            continue

        # 4. 不超过风控上限
        limit = limits.get(code)
        if limit is not None:
            max_shares = limit.max_shares if hasattr(limit, "max_shares") else limit.get("max_shares", 0)
            if max_shares > 0 and volume > max_shares:
                logger.info("validators: %s 裁剪 %d→%d (风控上限)", code, volume, max_shares)
                volume = max_shares

        # 5. 预算检查
        cost = volume * price
        remaining = cash_available - min_cash - total_cost

        if cost > remaining:
            new_volume = int(remaining / price / LOT_SIZE) * LOT_SIZE
            if new_volume >= LOT_SIZE:
                logger.info("validators: %s 裁剪 %d→%d (超预算)", code, volume, new_volume)
                volume = new_volume
                if hasattr(d, "volume"):
                    d.volume = volume
                else:
                    d["volume"] = volume
                total_cost += volume * price
                valid.append(d)
            else:
                logger.info("validators: %s 跳过 (预算不足)", code)
            break
        else:
            if hasattr(d, "volume"):
                d.volume = volume
            else:
                d["volume"] = volume
            total_cost += cost
            valid.append(d)

    logger.info(
        "validators: 校验完成, %d→%d 笔有效决策, 总成本 ¥%.0f",
        len(decisions), len(valid), total_cost,
    )
    return valid
