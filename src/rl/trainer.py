"""
RL 训练器 — 跨股票共享训练 DQN 智能体。
"""

from __future__ import annotations

import logging

from .agent import DQNAgent
from .environment import TradingEnvironment

logger = logging.getLogger(__name__)


def train_rl_agent(
    codes: list[str],
    daily_data: dict[str, list],
    episodes: int = 200,
    output_path: str = "",
) -> DQNAgent:
    """
    跨股票训练 DQN 智能体。

    codes: 股票代码列表
    daily_data: {code: [StockDaily, ...]}
    episodes: 总训练轮数 (平均分配各股票)
    output_path: 模型保存路径 (不为空时自动保存)

    返回: 训练完成的 DQNAgent
    """
    agent = DQNAgent()

    total_symbols = len(codes)
    if total_symbols == 0:
        logger.warning("无训练数据")
        return agent

    eps_per_stock = max(20, episodes // total_symbols)

    for code in codes:
        records = daily_data.get(code, [])
        if len(records) < 30:
            logger.warning("%s: 数据不足 (需要 ≥30, 实际 %d), 跳过", code, len(records))
            continue

        env = TradingEnvironment(records)
        logger.info("RL 训练 %s: %d episodes (数据 %d 条)", code, eps_per_stock, len(records))
        agent.train(env, episodes=eps_per_stock)

    if output_path:
        agent.save(output_path)
        logger.info("RL 模型已保存: %s", output_path)

    logger.info("RL 训练完成: %d 股票, epsilon=%.4f", total_symbols, agent.epsilon)
    return agent
