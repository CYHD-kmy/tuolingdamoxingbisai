"""
目标仓位计算器 — 将 LLM 输出的目标权重转为具体的买卖决策。

输入: TargetAllocation[] + 当前持仓 + 权益
输出: FinalDecision[] (含 buy/sell)
"""

from __future__ import annotations

import logging
from typing import Any

from .models import TargetAllocation, FinalDecision, PositionLimit

logger = logging.getLogger(__name__)

LOT_SIZE = 100
MIN_TRADE_VALUE = 5000.0  # 最小交易金额


class AllocationCalculator:
    """将目标权重分配转为买卖交易清单"""

    @staticmethod
    def compute_trades(
        allocations: list[TargetAllocation],
        current_positions: dict[str, Any],
        available_cash: float,
        daily_data: dict[str, list],
        equity: float,
        limits: dict[str, PositionLimit] | None = None,
        min_cash_reserve: float = 0.10,
    ) -> list[FinalDecision]:
        """
        核心计算: target_weight → buy/sell decisions

        allocations: LLM 输出的目标权重
        current_positions: {code: Position} 当前持仓
        available_cash: 可用现金
        daily_data: 行情数据
        equity: 总权益 (现金 + 市值)
        limits: 风控仓位上限 {code: PositionLimit}
        min_cash_reserve: 最低现金保留比例

        返回: FinalDecision[] (先 sell 后 buy)
        """
        decisions: list[FinalDecision] = []
        total_buy_value = 0.0
        total_buy_cost_cash = 0.0  # 实际消耗的现金 (扣去卖出现金后)
        sell_proceeds_total = 0.0
        stock_buy_count = 0

        for ta in allocations:
            if ta.code not in limits:
                continue

            price = _get_close(ta.code, daily_data)
            if price <= 0:
                continue

            target_value = ta.target_weight * equity
            current_pos = current_positions.get(ta.code)
            current_value = current_pos.market_value if current_pos else 0.0
            current_shares = current_pos.shares if current_pos else 0

            delta_value = target_value - current_value
            delta_shares = int(delta_value / price / LOT_SIZE) * LOT_SIZE

            if delta_shares >= LOT_SIZE:
                # 需要买入
                limit = limits.get(ta.code)
                if limit and limit.max_shares > 0:
                    max_buy = min(delta_shares, limit.max_shares)
                    # 检查已达仓位是否超上限
                    already_pct = current_value / equity if equity > 0 else 0
                    remaining_pct = limit.max_position_pct - already_pct
                    if remaining_pct <= 0:
                        continue
                    max_by_pct = int((remaining_pct * equity) / price / LOT_SIZE) * LOT_SIZE
                    shares = min(max_buy, max_by_pct)
                    if shares >= LOT_SIZE:
                        cost = shares * price
                        decisions.append(FinalDecision(
                            symbol=ta.code, symbol_name=ta.name,
                            volume=shares, entry_price=price,
                            direction="buy",
                        ))
                        total_buy_value += cost
                        stock_buy_count += 1

            elif delta_shares <= -LOT_SIZE and current_shares > 0:
                # 需要卖出
                sell_shares = min(abs(delta_shares), current_shares)
                sell_shares = (sell_shares // LOT_SIZE) * LOT_SIZE
                if sell_shares >= LOT_SIZE:
                    proceeds = sell_shares * price
                    decisions.append(FinalDecision(
                        symbol=ta.code, symbol_name=ta.name,
                        volume=sell_shares, entry_price=price,
                        direction="sell",
                    ))
                    sell_proceeds_total += proceeds

        # 最终现金检查: 买入总金额不能超过 可用现金 + 卖出回收 - 最低现金保留
        reserve = equity * min_cash_reserve
        max_buy_cash = available_cash + sell_proceeds_total - reserve

        if total_buy_value > max_buy_cash and stock_buy_count > 0:
            # 按比例缩减买入
            scale = max_buy_cash / total_buy_value if total_buy_value > 0 else 1.0
            logger.info(
                "AllocationCalculator: 买入金额 ¥%.0f 超限 ¥%.0f, 缩放至 %.0f%%",
                total_buy_value, max_buy_cash, scale * 100,
            )
            for d in decisions:
                if d.direction == "buy":
                    new_shares = int(d.volume * scale / LOT_SIZE) * LOT_SIZE
                    d.volume = max(LOT_SIZE, new_shares) if new_shares >= LOT_SIZE else 0

            # 移除被缩至 0 的决策
            decisions = [d for d in decisions if d.volume >= LOT_SIZE]

        logger.info(
            "AllocationCalculator: %d 个目标 → %d 笔交易 (买%d卖%d), 买入¥%.0f 卖出¥%.0f",
            len(allocations), len(decisions),
            sum(1 for d in decisions if d.direction == "buy"),
            sum(1 for d in decisions if d.direction == "sell"),
            sum(d.volume * _get_close(d.symbol, daily_data) for d in decisions if d.direction == "buy"),
            sell_proceeds_total,
        )
        return decisions


def _get_close(code: str, daily_data: dict[str, list]) -> float:
    records = daily_data.get(code, [])
    if not records:
        return 0.0
    return records[-1].close
