"""
策略注册表 — 管理所有 Alpha 策略的注册和查找。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class StrategyPerformance:
    """策略表现追踪"""
    name: str
    total_runs: int = 0
    win_count: int = 0
    cumulative_return: float = 0.0
    returns: list[float] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.win_count / self.total_runs if self.total_runs > 0 else 0.0

    @property
    def avg_return(self) -> float:
        return sum(self.returns) / len(self.returns) if self.returns else 0.0

    @property
    def sharpe_approx(self) -> float:
        """简化的 Sharpe 代理: avg/std * sqrt(n)"""
        if len(self.returns) < 2:
            return 0.0
        avg = self.avg_return
        var = sum((r - avg) ** 2 for r in self.returns) / (len(self.returns) - 1)
        std = var ** 0.5
        return avg / std * (len(self.returns) ** 0.5) if std > 1e-10 else 0.0


class StrategyRegistry:
    """策略注册表 (类级别单例)"""

    _strategies: dict[str, BaseStrategy] = {}
    _performance: dict[str, StrategyPerformance] = {}

    @classmethod
    def register(cls, strategy: BaseStrategy) -> None:
        cls._strategies[strategy.name] = strategy
        if strategy.name not in cls._performance:
            cls._performance[strategy.name] = StrategyPerformance(name=strategy.name)

    @classmethod
    def get(cls, name: str) -> BaseStrategy | None:
        return cls._strategies.get(name)

    @classmethod
    def get_all(cls) -> dict[str, BaseStrategy]:
        return dict(cls._strategies)

    @classmethod
    def list_names(cls) -> list[str]:
        return list(cls._strategies.keys())

    @classmethod
    def update_performance(cls, name: str, daily_return: float) -> None:
        perf = cls._performance.get(name)
        if perf:
            perf.total_runs += 1
            if daily_return > 0:
                perf.win_count += 1
            perf.cumulative_return += daily_return
            perf.returns.append(daily_return)

    @classmethod
    def get_performance(cls, name: str) -> StrategyPerformance | None:
        return cls._performance.get(name)

    @classmethod
    def get_all_performance(cls) -> dict[str, StrategyPerformance]:
        return dict(cls._performance)

    @classmethod
    def clear(cls) -> None:
        cls._strategies.clear()
        cls._performance.clear()
