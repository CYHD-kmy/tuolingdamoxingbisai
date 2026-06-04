"""
绩效归因 (P3) — 将组合收益分解为选股能力和择时能力。

方法 (借鉴 jQuantStats 的 tilt/timing 分解理念):
    - 选股贡献 = 假设等权持有所有建仓股票的理论收益
    - 择时贡献 = 实际仓位权重偏离等权带来的超额收益
    - 残差 = 无法由选股/择时解释的部分

使用方式:
    from src.review.attribution import Attribution
    attr = Attribution(tracker)
    result = attr.decompose()
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)


class Attribution:
    """绩效归因分解器"""

    def __init__(self, tracker):
        self._tracker = tracker

    def decompose(self) -> "AttributionResult | None":
        """执行绩效归因分解。"""
        from .engine import AttributionResult

        history = self._tracker.history
        positions = self._tracker.positions

        if not history or len(history) < 2:
            return None

        # 总收益率
        initial_value = history[0].get("total_value", 0)
        final_value = history[-1].get("total_value", 0)
        if initial_value <= 0:
            return None

        total_return = (final_value / initial_value - 1) * 100

        # 从 trace 文件中重建建仓记录
        results_dir = getattr(self._tracker, "_results_dir", "./results")
        trade_records = self._load_trade_records(results_dir)

        if not trade_records:
            # 没有足够数据，返回仅含总收益的结果
            return AttributionResult(total_return_pct=round(total_return, 2))

        # 计算选股贡献: 等权持有所有买入股票
        selection_return = self._calc_selection_return(history, trade_records)
        # 择时贡献 = 总收益 - 选股收益
        timing_return = total_return - selection_return

        return AttributionResult(
            total_return_pct=round(total_return, 2),
            selection_contribution=round(selection_return, 2),
            timing_contribution=round(timing_return, 2),
            industry_contribution=0.0,  # 需要行业基准数据，暂不计算
            residual=round(total_return - selection_return - timing_return, 2),
        )

    def _load_trade_records(self, results_dir: str) -> list[dict]:
        """从历史 trace 中加载买入交易记录。"""
        records: list[dict] = []
        try:
            files = sorted([
                f for f in os.listdir(results_dir)
                if f.startswith("trace_") and f.endswith(".json")
            ])
        except OSError:
            return records

        for fname in files:
            fpath = os.path.join(results_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    trace = json.load(f)
                trace_date = trace.get("date", "")
                decisions = trace.get("decisions", [])
                for d in decisions:
                    if d.get("direction") == "buy" or d.get("action") == "buy":
                        records.append({
                            "date": trace_date,
                            "code": d.get("symbol", d.get("code", "")),
                            "name": d.get("symbol_name", d.get("name", "")),
                            "volume": d.get("volume", d.get("shares", 0)),
                            "price": d.get("entry_price", d.get("price", 0)),
                        })
            except (json.JSONDecodeError, OSError):
                continue

        records.sort(key=lambda r: r["date"])
        return records

    def _calc_selection_return(self, history: list[dict], trade_records: list[dict]) -> float:
        """
        计算选股贡献: 如果每只买入股票等权持有，理论收益是多少。

        简化方法: 计算所有买入过的股票的平均盈亏，作为"纯选股"收益。
        """
        positions = self._tracker.positions
        if not positions:
            return 0.0

        # 用当前持仓的浮动盈亏作为"纯选股"的近似
        total_pnl = sum(p.unrealized_pnl for p in positions.values())
        total_cost = sum(p.cost_value for p in positions.values())

        if total_cost <= 0:
            return 0.0

        # 选股收益 = 持仓加权平均盈亏率
        return (total_pnl / total_cost) * 100
