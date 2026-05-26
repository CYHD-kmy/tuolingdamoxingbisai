"""
策略基类 — 所有 Alpha 策略的抽象基类。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class StrategyResult:
    """策略输出结果"""
    name: str
    candidates: list = field(default_factory=list)  # list[FactorScore]
    metadata: dict = field(default_factory=dict)


class BaseStrategy(ABC):
    """
    Alpha 策略抽象基类。

    子类必须实现:
    - name: 策略名称
    - description: 策略描述 (中文)
    - run(): 生成候选列表
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def run(
        self,
        snapshots: list,
        daily_data: dict[str, list],
        fund_flows: dict[str, list],
    ) -> StrategyResult: ...
