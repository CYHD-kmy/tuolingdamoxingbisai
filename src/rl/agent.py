"""
DQN 智能体 — 手动实现三层全连接网络 + 经验回放 + 目标网络。

无 PyTorch/TensorFlow 依赖，全部手动实现前向/反向传播。

网络结构: state_dim → 64 → 32 → action_dim (ReLU + Linear)
训练: MSE loss + SGD + 经验回放

使用方式:
    agent = DQNAgent(state_dim=9, action_dim=3)
    agent.train(env, episodes=200)
    agent.save("model.json")
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections import deque
from dataclasses import dataclass, field

from .features import FEATURE_DIM

logger = logging.getLogger(__name__)

STATE_DIM = FEATURE_DIM + 2  # 7 特征 + 仓位 + entry_ratio
ACTION_DIM = 3                # hold / buy / sell


@dataclass
class AgentState:
    """DQN 智能体训练状态"""
    epsilon: float = 1.0
    episodes_trained: int = 0
    total_reward: float = 0.0


class SimpleNN:
    """
    三层全连接神经网络，手动实现前向和反向传播。

    结构: input → 64 (ReLU) → 32 (ReLU) → output (Linear)
    损失: MSE
    优化: SGD + Xavier 初始化
    """

    def __init__(self, layer_sizes: list[int], lr: float = 0.001) -> None:
        self._lr = lr
        self._sizes = layer_sizes
        self._weights: list[list[list[float]]] = []
        self._biases: list[list[float]] = []
        # 缓存前向传播中间值用于反向传播
        self._cache: dict = {}

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # Xavier 初始化
            limit = (6.0 / (fan_in + fan_out)) ** 0.5
            w = [[random.uniform(-limit, limit) for _ in range(fan_in)] for _ in range(fan_out)]
            b = [0.0] * fan_out
            self._weights.append(w)
            self._biases.append(b)

    def forward(self, x: list[float]) -> list[float]:
        """前向传播，返回输出层激活值"""
        self._cache = {"a": [list(x)], "z": []}
        a = x

        for layer_idx, (w, b) in enumerate(zip(self._weights, self._biases)):
            # z = W @ a + b
            z = [sum(w[i][j] * a[j] for j in range(len(a))) + b[i] for i in range(len(b))]
            self._cache["z"].append(list(z))

            # 激活函数 (最后一层用 Linear，其余 ReLU)
            if layer_idx < len(self._weights) - 1:
                a = [max(0.0, zi) for zi in z]  # ReLU
            else:
                a = list(z)  # Linear
            self._cache["a"].append(list(a))

        return a

    def backward(self, x: list[float], y: list[float], y_pred: list[float]) -> float:
        """
        反向传播 + 权重更新。

        x: 输入
        y: 目标值 (one-hot like)
        y_pred: 预测值

        返回: MSE loss
        """
        n_layers = len(self._weights)

        # MSE loss 梯度: dL/dy_pred = 2 * (y_pred - y) / n
        n_out = len(y_pred)
        loss = sum((y_pred[i] - y[i]) ** 2 for i in range(n_out)) / n_out
        d_output = [(y_pred[i] - y[i]) * 2.0 / n_out for i in range(n_out)]

        # 反向传播
        delta = d_output

        for layer_idx in range(n_layers - 1, -1, -1):
            a_prev = self._cache["a"][layer_idx]  # 当前层的输入
            z = self._cache["z"][layer_idx]        # 当前层的线性输出

            # ReLU 反向 (除最后一层外)
            if layer_idx < n_layers - 1:
                delta = [delta[i] * (1.0 if z[i] > 0 else 0.0) for i in range(len(delta))]

            # 权重梯度: dW = delta ⊗ a_prev, db = delta
            w = self._weights[layer_idx]
            for i in range(len(w)):
                for j in range(len(w[i])):
                    w[i][j] -= self._lr * delta[i] * a_prev[j]
                self._biases[layer_idx][i] -= self._lr * delta[i]

            # 传播到前一层: delta_prev = W^T @ delta
            if layer_idx > 0:
                next_delta = [0.0] * len(w[0])
                for j in range(len(w[0])):
                    s = 0.0
                    for i in range(len(w)):
                        s += w[i][j] * delta[i]
                    next_delta[j] = s
                delta = next_delta

        return loss

    def clone(self) -> SimpleNN:
        """深拷贝网络 (用于目标网络)"""
        clone = SimpleNN.__new__(SimpleNN)
        clone._lr = self._lr
        clone._sizes = list(self._sizes)
        clone._weights = [[list(row) for row in w] for w in self._weights]
        clone._biases = [[b for b in layer] for layer in self._biases]
        clone._cache = {}
        return clone

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "sizes": self._sizes,
            "weights": [[[float(v) for v in row] for row in w] for w in self._weights],
            "biases": [[float(b) for b in layer] for layer in self._biases],
        }

    @classmethod
    def from_dict(cls, data: dict, lr: float = 0.001) -> SimpleNN:
        """从字典反序列化"""
        nn = SimpleNN.__new__(SimpleNN)
        nn._lr = lr
        nn._sizes = list(data["sizes"])
        nn._weights = data["weights"]
        nn._biases = data["biases"]
        nn._cache = {}
        return nn


class DQNAgent:
    """
    DQN 智能体 — epsilon-greedy 探索 + 经验回放 + 目标网络。

    使用方式:
        agent = DQNAgent()
        agent.train(env, episodes=200)
        signal = agent.infer(records)  # "buy" / "hold"
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 2000,
        batch_size: int = 32,
        target_update_freq: int = 10,
    ) -> None:
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._gamma = gamma
        self.epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay
        self._batch_size = batch_size
        self._target_update_freq = target_update_freq

        layer_sizes = [state_dim, 64, 32, action_dim]
        self._policy_net = SimpleNN(layer_sizes, lr=learning_rate)
        self._target_net = self._policy_net.clone()

        self._replay_buffer: deque = deque(maxlen=memory_size)
        self._step_counter = 0
        self._total_reward = 0.0

    def act(self, state: list[float]) -> int:
        """epsilon-greedy 动作选择"""
        if random.random() < self.epsilon:
            return random.randint(0, self._action_dim - 1)

        q_values = self._policy_net.forward(state)
        max_q = max(q_values)
        # 返回最大 Q 值对应的动作 (平局随机)
        best = [i for i, q in enumerate(q_values) if q == max_q]
        return random.choice(best)

    def remember(self, state: list[float], action: int, reward: float,
                 next_state: list[float], done: bool) -> None:
        """存储经验"""
        self._replay_buffer.append((state, action, reward, next_state, done))

    def replay(self) -> float:
        """从经验回放中采样并训练"""
        if len(self._replay_buffer) < self._batch_size:
            return 0.0

        batch = random.sample(self._replay_buffer, self._batch_size)
        total_loss = 0.0

        for state, action, reward, next_state, done in batch:
            # 目标 Q 值
            target_q = self._policy_net.forward(state)
            if done:
                target_q[action] = reward
            else:
                next_q = self._target_net.forward(next_state)
                target_q[action] = reward + self._gamma * max(next_q)

            # 训练
            pred_q = self._policy_net.forward(state)
            loss = self._policy_net.backward(state, target_q, pred_q)
            total_loss += loss

        return total_loss / self._batch_size

    def update_target(self) -> None:
        """更新目标网络"""
        self._target_net = self._policy_net.clone()

    def train(self, env, episodes: int = 100) -> AgentState:
        """
        训练 DQN 智能体。

        env: TradingEnvironment 实例
        episodes: 训练轮数

        返回: AgentState
        """
        for ep in range(episodes):
            state = env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                ep_reward += reward

                self._step_counter += 1
                self.replay()

                if self._step_counter % self._target_update_freq == 0:
                    self.update_target()

            self._total_reward += ep_reward
            self.epsilon = max(self._epsilon_min, self.epsilon * self._epsilon_decay)

            if (ep + 1) % 50 == 0:
                logger.debug("RL episode %d/%d: reward=%.3f epsilon=%.3f",
                             ep + 1, episodes, ep_reward, self.epsilon)

        return AgentState(
            epsilon=self.epsilon,
            episodes_trained=episodes,
            total_reward=self._total_reward,
        )

    def save(self, filepath: str) -> None:
        """保存模型权重到 JSON 文件"""
        import os
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        data = {
            "policy_net": self._policy_net.to_dict(),
            "epsilon": self.epsilon,
            "total_reward": self._total_reward,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, filepath: str) -> None:
        """从 JSON 文件加载模型权重"""
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        self._policy_net = SimpleNN.from_dict(data["policy_net"])
        self._target_net = self._policy_net.clone()
        self.epsilon = data.get("epsilon", 0.01)
        self._total_reward = data.get("total_reward", 0.0)

    def infer(self, records: list) -> str:
        """
        运行推断: 对每根 K 线做决策，返回汇总信号。

        records: 日线数据列表 (≥ 30 条)

        返回: "buy" / "hold"
        """
        if len(records) < 20:
            return "hold"

        from .environment import TradingEnvironment
        env = TradingEnvironment(records)

        state = env.reset()
        done = False
        buy_signals = 0
        sell_signals = 0
        total_steps = 0

        while not done:
            q_values = self._policy_net.forward(state)
            action = max(range(len(q_values)), key=lambda i: q_values[i])
            if action == 1:
                buy_signals += 1
            elif action == 2:
                sell_signals += 1
            state, _, done, _ = env.step(action)
            total_steps += 1

            if total_steps > 100:
                break

        if buy_signals > sell_signals and buy_signals > 0:
            return "buy"
        return "hold"

    def get_q_confidence(self, records: list) -> float:
        """
        获取 Q 值置信度 (用于风控权重调整)。

        max(Q) / sum(|Q|) → [0, 1] 之间的置信度代理
        """
        if len(records) < 20:
            return 0.5

        from .features import compute_features, FEATURE_DIM
        features = compute_features(records, len(records) - 1)
        state = features + [0.0, 0.0]  # 空仓状态
        q = self._policy_net.forward(state)
        abs_sum = sum(abs(qi) for qi in q)
        if abs_sum < 1e-10:
            return 0.5
        return min(1.0, max(abs(qi) for qi in q) / abs_sum)
