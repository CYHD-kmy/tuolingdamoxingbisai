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

    # ── 加载/保存 ──────────────────────────────

    def load(self) -> None:
        """从磁盘加载持仓状态"""
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
            )

        logger.info(
            "PortfolioTracker: 已加载持仓 (%s), %d 只, 现金 ¥%.0f",
            prev_date, len(self.positions), self.cash,
        )

    def save(self) -> None:
        """持久化当前持仓到磁盘"""
        os.makedirs(self._results_dir, exist_ok=True)

        data = {
            "date": self._date,
            "total_capital": self._capital,
            "cash": round(self.cash, 2),
            "cumulative_pnl": round(self.cumulative_pnl, 2),
            "positions": {
                code: {
                    "name": p.name,
                    "shares": p.shares,
                    "avg_cost": p.avg_cost,
                    "entry_date": p.entry_date,
                    "last_price": p.last_price,
                    "industry": p.industry,
                    "asset_type": p.asset_type,
                }
                for code, p in self.positions.items() if p.shares > 0
            },
            "history": self.history[-90:],  # 保留最近90天
        }
        with open(self._filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("PortfolioTracker: 持仓已保存 (%d 只)", len(self.positions))

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
