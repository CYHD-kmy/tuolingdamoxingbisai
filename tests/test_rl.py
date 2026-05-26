"""
强化学习模块测试
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_record(close, pct_chg, date="2026-05-01", ma5=100, ma20=98,
                 rsi_14=50, macd_bar=0.5, volume=2e7):
    return type("Record", (), {
        "date": date, "close": close, "pct_chg": pct_chg,
        "ma5": ma5, "ma20": ma20, "rsi_14": rsi_14,
        "macd_bar": macd_bar, "volume": volume,
        "open": close * 0.99, "high": close * 1.02, "low": close * 0.98,
        "amount": close * volume,
    })()


def _make_records(n=30, base_price=100):
    records = []
    for i in range(n):
        pct = 2.0 if i % 3 == 0 else -1.0
        price = base_price * (1 + i * 0.005)
        records.append(_make_record(
            close=price, pct_chg=pct,
            date=f"2026-05-{i+1:02d}",
            ma5=price * 0.99, ma20=price * 0.95,
            rsi_14=50 + pct * 2, macd_bar=0.3 + i * 0.02,
            volume=2e7 + i * 1e6,
        ))
    return records


def test_compute_features():
    """特征提取: 7 维向量 + 裁剪"""
    from src.rl.features import compute_features, FEATURE_DIM
    records = _make_records(30)
    features = compute_features(records, 25)
    assert len(features) == FEATURE_DIM
    for f in features:
        assert -5.0 <= f <= 5.0, f"Feature out of range: {f}"


def test_environment_reset():
    """环境重置: 初始状态正确"""
    from src.rl.environment import TradingEnvironment
    records = _make_records(30)
    env = TradingEnvironment(records)
    state = env.reset()
    assert len(state) == 9
    assert state[7] == 0.0  # position=0


def test_environment_step():
    """环境步进: 状态转移 + 奖励"""
    from src.rl.environment import TradingEnvironment
    records = _make_records(30)
    env = TradingEnvironment(records)
    state = env.reset()
    next_state, reward, done, info = env.step(0)  # hold
    assert len(next_state) == 9
    assert not done
    assert isinstance(info, dict)


def test_dqn_agent_act():
    """DQN 动作选择: 有效动作 (0/1/2)"""
    from src.rl.agent import DQNAgent
    agent = DQNAgent()
    state = [0.0] * 9
    action = agent.act(state)
    assert action in (0, 1, 2)


def test_dqn_agent_remember_replay():
    """DQN 经验回放: 存储和采样"""
    from src.rl.agent import DQNAgent
    agent = DQNAgent(batch_size=4)
    for i in range(10):
        agent.remember([0.0] * 9, i % 3, 0.1 * i, [0.0] * 9, False)
    loss = agent.replay()
    assert isinstance(loss, float)


def test_agent_save_load():
    """模型序列化往返"""
    import tempfile
    from src.rl.agent import DQNAgent

    agent = DQNAgent()
    agent.epsilon = 0.5
    agent._total_reward = 100.0

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmppath = f.name

    try:
        agent.save(tmppath)
        agent2 = DQNAgent()
        agent2.load(tmppath)
        assert agent2.epsilon == 0.5
        assert agent2._total_reward == 100.0
    finally:
        os.unlink(tmppath)


def test_agent_infer_signal():
    """推断信号: buy / hold"""
    from src.rl.agent import DQNAgent
    records = _make_records(30)
    agent = DQNAgent()
    signal = agent.infer(records)
    assert signal in ("buy", "hold")


def test_simple_nn_forward():
    """SimpleNN 前向传播"""
    from src.rl.agent import SimpleNN
    nn = SimpleNN([9, 16, 3], lr=0.01)
    output = nn.forward([0.1] * 9)
    assert len(output) == 3


def test_simple_nn_train():
    """SimpleNN 反向传播降 loss"""
    from src.rl.agent import SimpleNN
    nn = SimpleNN([3, 8, 2], lr=0.05)
    x = [0.5, -0.3, 0.8]
    y = [1.0, 0.0]
    losses = []
    for _ in range(50):
        pred = nn.forward(x)
        loss = nn.backward(x, y, pred)
        losses.append(loss)
    assert losses[-1] < losses[0], "Loss should decrease"


def test_train_rl_agent():
    """训练流程基本运行"""
    from src.rl.trainer import train_rl_agent
    daily_data = {"test": _make_records(30)}
    agent = train_rl_agent(["test"], daily_data, episodes=5)
    assert agent.epsilon > 0
    assert agent.epsilon <= 1.0


if __name__ == "__main__":
    test_compute_features()
    test_environment_reset()
    test_environment_step()
    test_dqn_agent_act()
    test_dqn_agent_remember_replay()
    test_agent_save_load()
    test_agent_infer_signal()
    test_simple_nn_forward()
    test_simple_nn_train()
    test_train_rl_agent()
    print("All RL tests passed!")
