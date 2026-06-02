"""
增量持仓追踪器 — 跨日持仓状态管理。

职责:
- 加载/保存持仓状态 (JSON 文件持久化)
- 计算持仓成本、浮动盈亏、收益率
- 行业分布统计
- 日度收益率追踪

使用方式:
    tracker = PortfolioTracker(total_capital=500_000)
    tracker.load()
    # ... 运行流水线 ...
    tracker.apply_decisions(state.final_result.decisions)
    tracker.save()
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .models import FinalDecision

logger = logging.getLogger(__name__)

_POSITIONS_FILE = "positions.json"


@dataclass
class Position:
    """单只股票/ETF 持仓"""
    code: str
    name: str
    shares: int
    avg_cost: float          # 平均成本价
    entry_date: str          # 首次建仓日期
    last_price: float = 0.0  # 最新市价
    industry: str = ""
    asset_type: str = "stock"  # "stock" / "etf"
    trailing_stop: float = 0.0  # 移动止损价 (0=未设置)

    @property
    def cost_value(self) -> float:
        return self.shares * self.avg_cost

    @property
    def market_value(self) -> float:
        return self.shares * self.last_price if self.last_price > 0 else self.cost_value

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_value

    @property
    def pnl_pct(self) -> float:
        return (self.market_value / self.cost_value - 1) * 100 if self.cost_value > 0 else 0.0

    @property
    def holding_days(self) -> int:
        """持有天数 (基于 entry_date)"""
        if not self.entry_date:
            return 0
        try:
            d1 = datetime.strptime(self.entry_date, "%Y%m%d")
            d2 = datetime.now()
            return (d2 - d1).days
        except (ValueError, TypeError):
            return 0

    def ratio_of_equity(self, equity: float) -> float:
        """仓位占比 (相对于总权益)"""
        if equity <= 0:
            return 0.0
        return self.market_value / equity


@dataclass
class PortfolioSnapshot:
    """某个时间点的组合快照"""
    date: str
    positions: dict[str, Position] = field(default_factory=dict)
    cash: float = 0.0
    total_value: float = 0.0  # 持仓市值 + 现金
    daily_pnl: float = 0.0
    daily_return: float = 0.0


class PortfolioTracker:
    """
    增量持仓追踪器。

    文件格式 (positions.json):
    {
      "date": "20260526",
      "total_capital": 500000.0,
      "cash": 118210.0,
      "cumulative_pnl": 0.0,
      "positions": {
        "600519": {
          "name": "贵州茅台", "shares": 200, "avg_cost": 1680.50,
          "entry_date": "20260526", "last_price": 1680.50, "industry": "白酒"
        }
      },
      "history": [
        {"date": "20260526", "total_value": 500000.0, "daily_pnl": 0.0, "daily_return": 0.0}
      ]
    }
    """

    def __init__(self, total_capital: float = 500_000.0, results_dir: str = "./results") -> None:
        self._capital = total_capital
        self._results_dir = results_dir
        self._filepath = os.path.join(results_dir, _POSITIONS_FILE)
        self.positions: dict[str, Position] = {}
        self.cash: float = total_capital
        self.cumulative_pnl: float = 0.0
        self.history: list[dict[str, Any]] = []
        self._date: str = datetime.now().strftime("%Y%m%d")
        self._tampered: bool = False
        self._pipeline_version: str = ""

    # ── 完整性校验 ──────────────────────────────

    @staticmethod
    def _compute_checksum(positions: dict, cash: float, pnl: float, history: list) -> str:
        """计算持仓数据的确定性哈希 (排除时间戳等非确定性字段)。"""
        raw = json.dumps({
            "positions": {k: {
                "code": v.get("code", k),
                "name": v.get("name", ""),
                "shares": v.get("shares", 0),
                "avg_cost": v.get("avg_cost", 0),
            } for k, v in sorted(positions.items())},
            "cash": round(cash, 2),
            "cumulative_pnl": round(pnl, 2),
            "history_dates": [h.get("date") for h in history[-30:]],
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _verify_integrity(self, data: dict) -> bool:
        """验证加载的数据与存储哈希是否一致。"""
        stored_hash = data.get("_integrity_hash", "")
        if not stored_hash:
            # 旧格式文件没有哈希, 不算篡改
            return True
        computed = self._compute_checksum(
            data.get("positions", {}),
            data.get("cash", 0),
            data.get("cumulative_pnl", 0),
            data.get("history", []),
        )
        return computed == stored_hash

    @staticmethod
    def _trace_decisions_hash(trace: dict) -> str:
        """提取 trace 中决策部分的确定性哈希。"""
        decisions = trace.get("decisions", [])
        raw = json.dumps([
            {
                "symbol": d.get("symbol", d.get("code", "")),
                "direction": d.get("direction", d.get("action", "")),
                "volume": d.get("volume", d.get("shares", 0)),
                "price": d.get("entry_price", d.get("price", 0)),
            }
            for d in decisions
        ], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _cross_validate_trace(self, data: dict) -> bool:
        """交叉验证: 持仓引用的 trace 文件是否与原始决策记录一致。"""
        ref_date = data.get("_last_trace_date", "")
        ref_hash = data.get("_last_trace_hash", "")
        if not ref_date or not ref_hash:
            # 旧格式没有引用, 标记为不确定但放行 (新管道写入后会补全)
            return True

        trace_path = os.path.join(self._results_dir, f"trace_{ref_date}.json")
        if not os.path.isfile(trace_path):
            logger.error("交叉验证: trace 文件 %s 不存在", os.path.basename(trace_path))
            return False

        try:
            with open(trace_path, "r", encoding="utf-8") as f:
                trace = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.error("交叉验证: trace 文件 %s 损坏", os.path.basename(trace_path))
            return False

        actual_hash = self._trace_decisions_hash(trace)
        if actual_hash != ref_hash:
            logger.error(
                "交叉验证: trace 决策哈希不匹配 (期望 %s, 实际 %s)",
                ref_hash[:8], actual_hash[:8],
            )
            return False

        return True

    @property
    def tampered(self) -> bool:
        """持仓文件是否被非管道程序篡改。"""
        return self._tampered

    @property
    def last_pipeline_date(self) -> str:
        """上次由管道程序写入的日期。"""
        return getattr(self, "_last_pipeline_date", "")

    def load(self) -> None:
        """从磁盘加载持仓状态, 含完整性校验和交叉验证。"""
        if not os.path.isfile(self._filepath):
            logger.info("PortfolioTracker: 无历史持仓文件，使用初始资金")
            self.cash = self._capital
            return

        try:
            with open(self._filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("PortfolioTracker: 持仓文件损坏，使用初始资金")
            self.cash = self._capital
            return

        prev_date = data.get("date", "")
        self.cash = data.get("cash", self._capital)
        self.cumulative_pnl = data.get("cumulative_pnl", 0.0)
        self.history = data.get("history", [])

        # 第1层: 完整性哈希校验 (防直接篡改)
        if not self._verify_integrity(data):
            self._tampered = True
            logger.error(
                "PortfolioTracker: 持仓文件完整性校验失败! "
                "positions.json 可能被手动修改或替换, 请检查数据来源"
            )

        # 第2层: 交叉验证 — 引用trace是否依然匹配 (防整体替换)
        if not self._cross_validate_trace(data):
            self._tampered = True
            logger.error(
                "PortfolioTracker: 交叉验证失败! "
                "positions.json 与管道决策记录不匹配, "
                "文件可能被其他来源的持仓数据替换"
            )

        for code, pos_data in data.get("positions", {}).items():
            self.positions[code] = Position(
                code=code,
                name=pos_data.get("name", code),
                shares=pos_data.get("shares", 0),
                avg_cost=pos_data.get("avg_cost", 0.0),
                entry_date=pos_data.get("entry_date", ""),
                last_price=pos_data.get("last_price", 0.0),
                industry=pos_data.get("industry", ""),
                asset_type=pos_data.get("asset_type", "stock"),
                trailing_stop=pos_data.get("trailing_stop", 0.0),
            )

        logger.info(
            "PortfolioTracker: 已加载持仓 (%s), %d 只, 现金 ¥%.0f",
            prev_date, len(self.positions), self.cash,
        )

    def save(self) -> None:
        """持久化当前持仓到磁盘, 附带完整性哈希和trace引用。"""
        os.makedirs(self._results_dir, exist_ok=True)

        # 先构建 positions 字典 (不含哈希), 再用它以计算哈希
        positions_dict = {
            code: {
                "name": p.name,
                "shares": p.shares,
                "avg_cost": p.avg_cost,
                "entry_date": p.entry_date,
                "last_price": p.last_price,
                "industry": p.industry,
                "asset_type": p.asset_type,
                "trailing_stop": p.trailing_stop,
            }
            for code, p in self.positions.items() if p.shares > 0
        }

        # 关联当日 trace 文件, 用于交叉验证
        trace_hash = ""
        trace_path = os.path.join(self._results_dir, f"trace_{self._date}.json")
        if os.path.isfile(trace_path):
            try:
                with open(trace_path, "r", encoding="utf-8") as f:
                    trace_data = json.load(f)
                trace_hash = self._trace_decisions_hash(trace_data)
            except (json.JSONDecodeError, OSError):
                logger.warning("PortfolioTracker: 无法读取 trace 文件用于引用")

        data = {
            "date": self._date,
            "total_capital": self._capital,
            "cash": round(self.cash, 2),
            "cumulative_pnl": round(self.cumulative_pnl, 2),
            "positions": positions_dict,
            "history": self.history[-90:],
            "_integrity_hash": self._compute_checksum(
                positions_dict, self.cash, self.cumulative_pnl, self.history
            ),
            "_pipeline_version": "1.1.0",
            "_last_pipeline_date": self._date,
            "_last_trace_date": self._date if trace_hash else "",
            "_last_trace_hash": trace_hash,
        }
        with open(self._filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 每日备份: 保存时间戳副本
        backup_dir = os.path.join(self._results_dir, "backups")
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, f"positions_{self._date}.json")
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(
            "PortfolioTracker: 持仓已保存 (%d 只) + 备份",
            len(self.positions),
        )

    # ── 交易操作 ──────────────────────────────

    def apply_decisions(
        self,
        decisions: list[FinalDecision],
        daily_data: dict[str, list[Any]],
        industry_map: dict[str, str] | None = None,
    ) -> None:
        """
        应用当日买入决策，更新持仓状态。

        decisions: 最终买入决策列表
        daily_data: 日线数据 (用于获取价格)
        industry_map: {code: industry} 行业映射
        """
        today = self._date
        industries = industry_map or {}

        for d in decisions:
            if d.volume <= 0:
                continue

            price = self._get_price(d.symbol, daily_data)
            if price <= 0:
                logger.warning("PortfolioTracker: %s 无有效价格，跳过", d.symbol)
                continue

            cost = d.volume * price

            if d.symbol in self.positions:
                old = self.positions[d.symbol]
                total_shares = old.shares + d.volume
                new_avg_cost = (
                    (old.cost_value + cost) / total_shares
                    if total_shares > 0 else 0.0
                )
                self.positions[d.symbol] = Position(
                    code=d.symbol,
                    name=d.symbol_name or old.name,
                    shares=total_shares,
                    avg_cost=round(new_avg_cost, 4),
                    entry_date=old.entry_date,
                    last_price=price,
                    industry=industries.get(d.symbol, old.industry),
                    asset_type=getattr(d, "asset_type", "stock") or old.asset_type,
                )
            else:
                self.positions[d.symbol] = Position(
                    code=d.symbol,
                    name=d.symbol_name,
                    shares=d.volume,
                    avg_cost=price,
                    entry_date=today,
                    last_price=price,
                    industry=industries.get(d.symbol, ""),
                    asset_type=getattr(d, "asset_type", "stock") or "stock",
                )

            self.cash -= cost

    def liquidate_all(self, daily_data: dict[str, list[Any]]) -> float:
        """
        卖出全部持仓 (清仓), 将市值转为现金。

        daily_data: 当日行情数据, 用于获取最新卖出价

        返回: 卖出总金额 (market value)
        """
        total_sold = 0.0
        sold_count = 0
        for code, pos in list(self.positions.items()):
            if pos.shares <= 0:
                continue
            price = self._get_price(code, daily_data)
            if price <= 0:
                price = pos.avg_cost  # 无市价时用成本价兜底
            proceeds = pos.shares * price
            pnl = proceeds - pos.cost_value
            self.cash += proceeds
            self.cumulative_pnl += pnl
            total_sold += proceeds
            sold_count += 1

        self.positions.clear()
        if sold_count > 0:
            logger.info(
                "PortfolioTracker: 清仓 %d 只, 回收 ¥%.0f, 现金余额 ¥%.0f",
                sold_count, total_sold, self.cash,
            )
        return total_sold

    def apply_sells(
        self,
        decisions: list[FinalDecision],
        daily_data: dict[str, list[Any]],
    ) -> float:
        """
        执行卖出决策，回收资金到现金账户。

        返回: 卖出总金额
        """
        total_proceeds = 0.0
        sell_count = 0
        for d in decisions:
            if d.direction != "sell" or d.volume <= 0:
                continue
            if d.symbol not in self.positions:
                logger.warning("PortfolioTracker: 卖出 %s 不在持仓中，跳过", d.symbol)
                continue

            pos = self.positions[d.symbol]
            sell_shares = min(d.volume, pos.shares)
            if sell_shares < 100:
                continue

            price = self._get_price(d.symbol, daily_data)
            if price <= 0:
                price = pos.last_price or d.entry_price or pos.avg_cost
            proceeds = sell_shares * price
            pnl = proceeds - sell_shares * pos.avg_cost
            self.cash += proceeds
            self.cumulative_pnl += pnl
            total_proceeds += proceeds
            sell_count += 1

            remaining = pos.shares - sell_shares
            if remaining < 100:  # 全部清仓
                del self.positions[d.symbol]
                logger.info(
                    "PortfolioTracker: 清仓 %s (%s) %d股 @ ¥%.2f, 盈亏 %+.0f",
                    d.symbol, pos.name, sell_shares, price, pnl,
                )
            else:
                pos.shares = remaining
                pos.last_price = price
                logger.info(
                    "PortfolioTracker: 减仓 %s (%s) %d→%d股, 盈亏 %+.0f",
                    d.symbol, pos.name, sell_shares + remaining, remaining, pnl,
                )

        if sell_count > 0:
            logger.info(
                "PortfolioTracker: 卖出 %d 笔, 回收 ¥%.0f, 现金余额 ¥%.0f",
                sell_count, total_proceeds, self.cash,
            )
        return total_proceeds

    def get_stale_positions(self) -> list[dict[str, Any]]:
        """
        检测持仓天数过长、收益不达标的持仓。

        规则:
        - 持有 > holding_clear_days 且收益 <= holding_clear_return → 清仓信号
        - 持有 > holding_reduce_days 且收益 < holding_reduce_return → 减仓信号

        返回: [{"code": ..., "action": "clear"/"reduce", "reason": ...}, ...]
        """
        from ..utils.config import get_config
        cfg = get_config()
        stale = []

        for code, pos in self.positions.items():
            if pos.shares <= 0:
                continue
            days = pos.holding_days
            pnl_pct = pos.pnl_pct

            if days > cfg.holding_clear_days and pnl_pct <= cfg.holding_clear_return:
                stale.append({
                    "code": code, "name": pos.name, "shares": pos.shares,
                    "holding_days": days, "pnl_pct": round(pnl_pct, 2),
                    "action": "clear",
                    "reason": f"持有{days}天(>{cfg.holding_clear_days})且收益{pnl_pct:+.1f}%≤{cfg.holding_clear_return}",
                })
            elif days > cfg.holding_reduce_days and pnl_pct < cfg.holding_reduce_return:
                stale.append({
                    "code": code, "name": pos.name, "shares": pos.shares,
                    "holding_days": days, "pnl_pct": round(pnl_pct, 2),
                    "action": "reduce",
                    "reason": f"持有{days}天(>{cfg.holding_reduce_days})且收益{pnl_pct:+.1f}%<{cfg.holding_reduce_return}",
                })

        return stale

    def check_stop_losses(self, daily_data: dict[str, list[Any]]) -> list[str]:
        """
        检查是否有持仓触发了止损线。

        返回: 触发止损的 code 列表
        """
        triggered = []
        for code, pos in self.positions.items():
            if pos.shares <= 0:
                continue
            stop_price = pos.trailing_stop or (pos.avg_cost * 0.93)
            price = self._get_price(code, daily_data) or pos.last_price
            if 0 < price <= stop_price:
                triggered.append(code)
                logger.warning(
                    "PortfolioTracker: %s (%s) 触发止损! 现价 ¥%.2f ≤ 止损 ¥%.2f",
                    code, pos.name, price, stop_price,
                )
        return triggered

    def update_trailing_stops(
        self,
        daily_data: dict[str, list[Any]],
        atr_stop_levels: dict[str, float] | None = None,
    ) -> None:
        """
        更新移动止损价 (价格上涨时抬高止损)。

        atr_stop_levels: {code: atr_stop_price} 由 RiskManager 计算
        """
        for code, pos in self.positions.items():
            if pos.shares <= 0:
                continue
            price = self._get_price(code, daily_data) or pos.last_price
            if price <= 0:
                continue

            # 如果有 ATR 止损价，使用它；否则用简单固定比例
            if atr_stop_levels and code in atr_stop_levels:
                new_stop = atr_stop_levels[code]
            else:
                new_stop = pos.avg_cost * 0.93  # 固定 7% 回落止损

            # 只在价格创新高时抬升止损 (移动止损)
            if new_stop > pos.trailing_stop:
                pos.trailing_stop = new_stop

    def build_context(
        self,
        daily_data: dict[str, list[Any]] | None = None,
    ) -> dict[str, Any]:
        """
        构建完整的组合上下文 (供 LLM 策略/分析使用)。

        返回一个 dict，包含:
        - cash, total_equity, total_return_pct
        - positions: 每只持仓的详细状态
        - stale_positions: 需要处理的持仓
        - industry_exposure: 行业分布
        """
        if daily_data:
            self.update_prices(daily_data)

        equity = self.total_equity()
        stale = self.get_stale_positions()
        stop_triggered = self.check_stop_losses(daily_data or {})
        stale_codes = {s["code"] for s in stale}

        positions_detail = []
        for code, pos in self.positions.items():
            if pos.shares <= 0:
                continue
            days = pos.holding_days
            positions_detail.append({
                "code": code,
                "name": pos.name,
                "shares": pos.shares,
                "avg_cost": pos.avg_cost,
                "last_price": pos.last_price,
                "market_value": round(pos.market_value, 2),
                "weight_in_equity": round(pos.ratio_of_equity(equity) * 100, 1),
                "pnl": round(pos.unrealized_pnl, 2),
                "pnl_pct": round(pos.pnl_pct, 2),
                "holding_days": days,
                "entry_date": pos.entry_date,
                "industry": pos.industry,
                "asset_type": pos.asset_type,
                "is_stale": code in stale_codes,
                "stale_action": next((s["action"] for s in stale if s["code"] == code), None),
                "stop_triggered": code in stop_triggered,
                "trailing_stop": pos.trailing_stop if pos.trailing_stop > 0 else None,
            })

        return {
            "date": self._date,
            "total_capital": self._capital,
            "cash": round(self.cash, 2),
            "market_value": round(self.total_market_value(), 2),
            "total_equity": round(equity, 2),
            "total_return_pct": round(self.total_return(), 2),
            "cumulative_pnl": round(self.cumulative_pnl, 2),
            "position_count": len(positions_detail),
            "positions": positions_detail,
            "stale_positions": stale,
            "stop_triggered_codes": stop_triggered,
            "industry_exposure": {
                k: round(v * 100, 1) for k, v in self.industry_exposure().items()
            },
        }

    def update_prices(self, daily_data: dict[str, list[Any]]) -> None:
        """用最新行情更新所有持仓的市价"""
        for code, pos in self.positions.items():
            price = self._get_price(code, daily_data)
            if price > 0:
                pos.last_price = price

    def record_daily(self) -> None:
        """记录当日组合快照"""
        total_value = self.cash + sum(p.market_value for p in self.positions.values())
        prev_total = self._capital
        if self.history:
            prev_total = self.history[-1].get("total_value", self._capital)

        daily_pnl = total_value - prev_total
        daily_return = (daily_pnl / prev_total * 100) if prev_total > 0 else 0.0
        self.cumulative_pnl += daily_pnl

        self.history.append({
            "date": self._date,
            "total_value": round(total_value, 2),
            "daily_pnl": round(daily_pnl, 2),
            "daily_return": round(daily_return, 4),
        })

    # ── 查询 ──────────────────────────────────

    def total_market_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    def total_equity(self) -> float:
        return self.cash + self.total_market_value()

    def total_return(self) -> float:
        return (self.total_equity() / self._capital - 1) * 100

    def industry_exposure(self) -> dict[str, float]:
        """各行业仓位占比 {industry: pct}"""
        total = self.total_equity()
        if total <= 0:
            return {}
        exposure: dict[str, float] = {}
        for p in self.positions.values():
            ind = p.industry or "未知"
            exposure[ind] = exposure.get(ind, 0) + p.market_value / total
        return exposure

    def current_positions_dict(self) -> dict[str, int]:
        """返回 {code: shares} 用于风控接口"""
        return {code: p.shares for code, p in self.positions.items() if p.shares > 0}

    def to_summary(self) -> dict[str, Any]:
        """生成组合摘要"""
        return {
            "date": self._date,
            "total_capital": self._capital,
            "cash": round(self.cash, 2),
            "market_value": round(self.total_market_value(), 2),
            "total_equity": round(self.total_equity(), 2),
            "total_return_pct": round(self.total_return(), 2),
            "cumulative_pnl": round(self.cumulative_pnl, 2),
            "position_count": sum(1 for p in self.positions.values() if p.shares > 0),
            "positions": [
                {
                    "code": p.code, "name": p.name, "shares": p.shares,
                    "avg_cost": p.avg_cost, "last_price": p.last_price,
                    "market_value": round(p.market_value, 2),
                    "pnl": round(p.unrealized_pnl, 2),
                    "pnl_pct": round(p.pnl_pct, 2),
                    "industry": p.industry,
                    "asset_type": p.asset_type,
                }
                for p in self.positions.values() if p.shares > 0
            ],
            "industry_exposure": {
                k: round(v * 100, 1) for k, v in self.industry_exposure().items()
            },
        }

    # ── 辅助 ──────────────────────────────────

    @staticmethod
    def _get_price(code: str, daily_data: dict[str, list[Any]]) -> float:
        records = daily_data.get(code, [])
        if not records:
            return 0.0
        return records[-1].close
