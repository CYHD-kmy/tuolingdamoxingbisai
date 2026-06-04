"""
已平仓事后验证 (P2) — 分析历史卖出决策是否正确。

从历史 trace 中找出所有卖出决策:
    1. 获取卖出后 N 天的价格走势
    2. 计算卖飞幅度 (卖出后上涨 > 3%) 和止损有效性 (卖出后继续下跌 > 3%)
    3. 统计汇总

使用方式:
    from src.review.post_mortem import PostMortem
    pm = PostMortem(data_interface, tracker)
    result = pm.analyze()
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

_SELL_THRESHOLD_PCT = 3.0   # ±3% 为判断卖飞/有效止损的阈值
_LOOKAHEAD_DAYS = 5          # 卖出后追踪天数


class PostMortem:
    """已平仓事后验证器"""

    def __init__(self, data_interface, tracker):
        self._data = data_interface
        self._tracker = tracker

    def analyze(self, lookahead_days: int = _LOOKAHEAD_DAYS) -> "PostMortemSummary":
        """
        分析历史卖出决策的事后表现。

        lookahead_days: 卖出后追踪的天数 (默认 5 天)
        """
        from .engine import PostMortemSummary

        results_dir = getattr(self._tracker, "_results_dir", "./results")

        # 1. 收集所有历史 trace 中的卖出决策
        sells = self._collect_sells(results_dir)
        if not sells:
            return PostMortemSummary(total_sells=0)

        # 2. 对每笔卖出做事后分析
        sell_too_early = 0
        correct_stops = 0
        neutral = 0
        missed_gains: list[float] = []
        avoided_losses: list[float] = []
        details: list[dict] = []

        for sell in sells:
            result = self._analyze_single_sell(sell, lookahead_days)
            if result is None:
                continue

            details.append(result)
            pct_change = result["post_sell_pct"]

            if pct_change > _SELL_THRESHOLD_PCT:
                sell_too_early += 1
                missed_gains.append(pct_change)
            elif pct_change < -_SELL_THRESHOLD_PCT:
                correct_stops += 1
                avoided_losses.append(abs(pct_change))
            else:
                neutral += 1

        return PostMortemSummary(
            total_sells=sell_too_early + correct_stops + neutral,
            sell_too_early=sell_too_early,
            correct_stops=correct_stops,
            neutral=neutral,
            avg_missed_gain_pct=round(sum(missed_gains) / len(missed_gains), 1) if missed_gains else 0.0,
            avg_avoided_loss_pct=round(sum(avoided_losses) / len(avoided_losses), 1) if avoided_losses else 0.0,
            details=details,
        )

    def _collect_sells(self, results_dir: str) -> list[dict]:
        """从历史 trace 文件中收集所有卖出决策。"""
        sells: list[dict] = []
        try:
            files = sorted([
                f for f in os.listdir(results_dir)
                if f.startswith("trace_") and f.endswith(".json")
            ])
        except OSError:
            return sells

        for fname in files:
            fpath = os.path.join(results_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    trace = json.load(f)
                decisions = trace.get("decisions", [])
                trace_date = trace.get("date", fname.replace("trace_", "").replace(".json", ""))
                for d in decisions:
                    if d.get("direction") == "sell" or d.get("action") == "sell":
                        sells.append({
                            "date": trace_date,
                            "code": d.get("symbol", d.get("code", "")),
                            "name": d.get("symbol_name", d.get("name", "")),
                            "volume": d.get("volume", d.get("shares", 0)),
                            "price": d.get("entry_price", d.get("price", 0)),
                        })
            except (json.JSONDecodeError, OSError, KeyError):
                continue

        # 按日期排序 (最新在前)
        sells.sort(key=lambda s: s["date"], reverse=True)
        return sells

    def _analyze_single_sell(self, sell: dict, lookahead_days: int) -> dict | None:
        """分析单笔卖出决策的事后表现。"""
        code = sell["code"]
        sell_date = sell["date"]
        sell_price = sell["price"]

        if not code or not sell_date or sell_price <= 0:
            return None

        # 获取卖出后的日线数据
        try:
            daily = self._data.get_daily_data(code, days=lookahead_days + 10)
        except Exception:
            logger.debug("无法获取 %s 的日线数据", code)
            return None

        if not daily or len(daily) < 2:
            return None

        # 找到卖出后 lookahead_days 的价格
        # daily 数据按日期排列，查找 >= sell_date 的下一个数据点
        post_prices = []
        for d in daily:
            d_str = getattr(d, "date", "")
            if isinstance(d_str, str) and d_str > sell_date:
                post_prices.append(d.close)
            elif hasattr(d, "date"):
                # 可能是 date 对象
                d_str = str(d.date)
                if d_str > sell_date:
                    post_prices.append(d.close)

        if not post_prices:
            return None

        # 取卖出后 N 日的价格
        future_price = post_prices[min(lookahead_days - 1, len(post_prices) - 1)]
        if future_price <= 0:
            return None

        pct_change = (future_price / sell_price - 1) * 100

        return {
            "code": code,
            "name": sell["name"],
            "sell_date": sell_date,
            "sell_price": sell_price,
            "post_sell_price": round(future_price, 2),
            "post_sell_pct": round(pct_change, 1),
            "verdict": "卖飞" if pct_change > _SELL_THRESHOLD_PCT
                       else "有效止损" if pct_change < -_SELL_THRESHOLD_PCT
                       else "中性",
        }
