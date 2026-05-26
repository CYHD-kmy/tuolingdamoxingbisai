"""
强化学习模块 — DQN 智能体生成交易信号作为风控的附加输入。

提供:
- TradingEnvironment: Gym 风格交易环境 (3 离散动作: hold/buy/sell)
- DQNAgent: DQN 智能体 (手动神经网络, 无 PyTorch 依赖)
- compute_features: 7 维技术特征提取
- train_rl_agent: 跨股票训练入口

使用方式:
    # 训练
    python -m src.main --rl-train --rl-episodes 200

    # 推断
    from src.rl.agent import DQNAgent
    agent = DQNAgent()
    agent.load("results/rl_model.json")
    signal = agent.infer(daily_records)
"""

from .environment import TradingEnvironment, TradingEnvState
from .agent import DQNAgent, AgentState, SimpleNN
from .features import compute_features
from .trainer import train_rl_agent
