"""
Gym 风格交易环境 — 离散动作空间 (hold=0, buy=1, sell=2)。

状态: 7 维特征 + 仓位信息
动作: 0=hold, 1=buy, 2=sell
奖励: 持仓期间的日收益率 - 交易成本

使用方式:
    env = TradingEnvironment(daily_records)
    state = env.reset()
    for step in range(max_steps):
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
"""

from __future__ import annotations

from dataclasses import dataclass

from .features import compute_features

# 交易成本 (双边 0.1% 印花税 + 佣金)
TRANSACTION_COST = 0.001


@dataclass
class TradingEnvState:
    """环境状态"""
    position: int = 0       # 0=空仓, 1=持仓
    entry_price: float = 0.0
    current_step: int = 0
    cash: float = 100_000.0
    shares: int = 0


class TradingEnvironment:
    """
    离散动作交易环境。

    动作:
    - 0 (hold): 维持现状
    - 1 (buy):  全仓买入 (空仓时)
    - 2 (sell): 清仓卖出 (持仓时)

    奖励: 持仓期间的日收益率
    """

    def __init__(
        self,
        daily_data: list,
        initial_cash: float = 100_000.0,
        transaction_cost: float = TRANSACTION_COST,
    ) -> None:
        self._data = daily_data
        self._initial_cash = initial_cash
        self._cost = transaction_cost
        self._min_idx = 19  # 最小需要 20 条数据计算特征
        self._state = TradingEnvState()

    @property
    def max_steps(self) -> int:
        return len(self._data) - self._min_idx - 1

    def reset(self) -> list[float]:
        """重置环境，返回初始状态"""
        self._state = TradingEnvState(cash=self._initial_cash)
        return self._get_state(self._min_idx)

    def step(self, action: int) -> tuple[list[float], float, bool, dict]:
        """
        执行动作。

        返回: (next_state, reward, done, info)
        """
        s = self._state
        idx = s.current_step
        price = self._data[idx].close

        reward = 0.0
        info: dict = {"action": action}

        if action == 1 and s.position == 0:
            # 买入
            max_shares = int(s.cash / (price * (1 + self._cost)))
            if max_shares >= 100:
                shares = (max_shares // 100) * 100
                cost = shares * price * (1 + self._cost)
                s.shares = shares
                s.entry_price = price
                s.cash -= cost
                s.position = 1
                info["buy"] = shares

        elif action == 2 and s.position == 1:
            # 卖出
            revenue = s.shares * price * (1 - self._cost)
            pnl = revenue - s.shares * s.entry_price * (1 + self._cost)
            reward = pnl / (s.shares * s.entry_price)  # 归一化收益率
            s.cash += revenue
            s.position = 0
            s.shares = 0
            s.entry_price = 0.0
            info["sell"] = pnl

        # 持仓浮动盈亏 (作为奖励信号)
        if s.position == 1:
            reward = (price / s.entry_price - 1) if s.entry_price > 0 else 0.0
            # 持仓时的微小正奖励 (鼓励盈利持仓)
            if reward > 0:
                reward *= 1.05

        s.current_step = idx + 1
        done = s.current_step >= len(self._data) - 1

        # 强行在最后一天卖出
        if done and s.position == 1:
            final_price = self._data[-1].close
            revenue = s.shares * final_price * (1 - self._cost)
            pnl = revenue - s.shares * s.entry_price * (1 + self._cost)
            reward = pnl / (s.shares * s.entry_price) if s.entry_price > 0 else 0.0
            s.cash += revenue
            s.position = 0
            s.shares = 0

        next_state = self._get_state(s.current_step) if not done else [0.0] * 9
        return next_state, reward, done, info

    def _get_state(self, idx: int) -> list[float]:
        """构建状态向量: 7 维特征 + 仓位 + entry_price_ratio"""
        features = compute_features(self._data, min(idx, len(self._data) - 1))
        return features + [
            float(self._state.position),
            self._state.entry_price / self._data[idx].close if self._state.position == 1 and self._state.entry_price > 0 else 0.0,
        ]
