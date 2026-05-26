"""训练数据生成 + 训练循环 (解析反向传播 + SGD)。

训练策略:
- 监督学习：输入 30 天 K 线序列 → 预测未来 5 日收益率
- 损失：MSE(pred, target)
- 优化器：SGD + 梯度裁剪
- 反向传播：输出头全解析梯度；编码器层采用逐步反向传播
"""

import math
import random
from typing import Optional

from .features import extract_features, pad_or_truncate, N_FEATURES
from .model import StockTransformer
from .embedding import Linear


# ── 梯度工具函数 ──


def _mse_loss_grad(pred: float, target: float) -> float:
    """MSE 损失对预测值的梯度: dL/dpred = 2*(pred - target)"""
    return 2.0 * (pred - target)


def _relu_grad(x: float, dout: float) -> float:
    """ReLU 梯度: dout if x > 0 else 0"""
    return dout if x > 0 else 0.0


def _layer_norm_backward(
    dout: list[float],
    x: list[float],
    mean: float,
    std: float,
    gamma: list[float],
    eps: float = 1e-5,
) -> tuple[list[float], list[float], list[float]]:
    """LayerNorm 反向传播。

    Args:
        dout: 上层传来的梯度 (d_model,)
        x: 原始输入 (d_model,)
        mean: 前向时计算的均值
        std: 前向时计算的标准差
        gamma: 缩放参数 (d_model,)
        eps: 防止除零

    Returns:
        (dx, dgamma, dbeta)
    """
    d_model = len(x)
    x_hat = [(x[i] - mean) / std for i in range(d_model)]

    # dbeta = dout
    dbeta = list(dout)

    # dgamma = dout * x_hat
    dgamma = [dout[i] * x_hat[i] for i in range(d_model)]

    # dx_hat = dout * gamma
    dx_hat = [dout[i] * gamma[i] for i in range(d_model)]

    # 通过 x_hat = (x - mean) / std 反向
    # dx = (1/std) * (dx_hat - mean(dx_hat) - x_hat * mean(dx_hat * x_hat))
    mean_dx_hat = sum(dx_hat) / d_model
    mean_dx_hat_x_hat = sum(dx_hat[i] * x_hat[i] for i in range(d_model)) / d_model

    dx = [0.0] * d_model
    for i in range(d_model):
        dx[i] = (dx_hat[i] - mean_dx_hat - x_hat[i] * mean_dx_hat_x_hat) / std

    return dx, dgamma, dbeta


def _linear_backward_full(
    dout: list[float],
    x: list[float],
    W: list[list[float]],
    lr: float,
    grad_clip: float = 5.0,
) -> tuple[list[float], list[list[float]], list[float]]:
    """线性层反向传播 (返回梯度，不更新参数)。

    Returns:
        (dx, dW, db): 各参数的梯度
    """
    in_features = len(x)
    out_features = len(dout)

    # dW = dout ⊗ x
    dW = [[0.0] * in_features for _ in range(out_features)]
    for i in range(out_features):
        di = dout[i]
        for j in range(in_features):
            dW[i][j] = di * x[j]

    # db = dout
    db = list(dout)

    # dx = W^T @ dout
    dx = [0.0] * in_features
    for j in range(in_features):
        s = 0.0
        for i in range(out_features):
            s += W[i][j] * dout[i]
        dx[j] = s

    # 梯度裁剪
    for i in range(out_features):
        for j in range(in_features):
            if dW[i][j] > grad_clip:
                dW[i][j] = grad_clip
            elif dW[i][j] < -grad_clip:
                dW[i][j] = -grad_clip
    for i in range(out_features):
        if db[i] > grad_clip:
            db[i] = grad_clip
        elif db[i] < -grad_clip:
            db[i] = -grad_clip

    return dx, dW, db


# ── 训练数据 ──


class TrainingSample:
    """一个训练样本。"""
    __slots__ = ("features", "mask", "target")

    def __init__(self, features: list[list[float]], mask: list[bool], target: float):
        self.features = features
        self.mask = mask
        self.target = target


def generate_training_data(
    daily_data: dict[str, list],
    seq_len: int = 30,
    forward_days: int = 5,
    max_seq_len: int = 60,
) -> list[TrainingSample]:
    """从 {code: [StockDaily]} 生成滑动窗口训练样本。

    对每只股票，取 seq_len + forward_days 天数据，滑动窗口:
    - 输入: 第 i 到 i+seq_len-1 天 → 预测第 i+seq_len+forward_days-1 天相对第 i+seq_len-1 天的收益率

    Args:
        daily_data: {code: [StockDaily 按日期升序]}
        seq_len: 输入序列长度
        forward_days: 预测未来天数
        max_seq_len: 最大序列长度 (用于 padding)

    Returns:
        TrainingSample 列表
    """
    samples: list[TrainingSample] = []

    for code, records in daily_data.items():
        if len(records) < seq_len + forward_days:
            continue

        # 提取全序列特征
        all_features = extract_features(records)

        for i in range(len(records) - seq_len - forward_days + 1):
            # 输入窗口
            window = all_features[i:i + seq_len]

            # Padding/truncation
            padded, mask = pad_or_truncate(window, max_seq_len)

            # 目标: forward_days 后的收益率
            current_close = records[i + seq_len - 1].close
            future_close = records[i + seq_len + forward_days - 1].close
            if current_close > 0:
                target = (future_close / current_close) - 1.0
            else:
                target = 0.0

            # 裁剪极端值
            target = max(-0.3, min(0.3, target))

            samples.append(TrainingSample(padded, mask, target))

    return samples


# ── 训练循环 ──


def _apply_gradients_to_linear(layer: Linear, dW: list[list[float]], db: list[float], lr: float):
    """将梯度应用到 Linear 层。"""
    for i in range(layer.out_features):
        for j in range(layer.in_features):
            layer.W[i][j] -= lr * dW[i][j]
    if layer.b is not None:
        for i in range(layer.out_features):
            layer.b[i] -= lr * db[i]


def _train_step(
    model: StockTransformer,
    sample: TrainingSample,
    lr: float,
    grad_clip: float,
) -> float:
    """单步训练：前向 + 解析反向传播。

    反向传播路径:
    MSE → output_head → pooled → encoder layers (逐层)
    → embedding.projection

    Returns:
        loss: 当前样本的损失值
    """
    features = sample.features
    mask = sample.mask
    target = sample.target

    # ── 前向传播 (全部手动做以便记录中间值) ──
    # 1. 嵌入
    embedded = model.embedding.forward(features, add_position=True)

    # 2. 编码器 (记录每层输出)
    encoder_inputs = [embedded]
    encoder_outputs = []
    for layer in model.encoder.layers:
        out = layer.forward(encoder_inputs[-1], mask)
        encoder_inputs.append(out)
        encoder_outputs.append(out)
    encoded = encoder_outputs[-1]  # (seq_len, d_model)

    # 3. 均值池化
    d_model = model.d_model
    if mask:
        valid_count = sum(1 for m in mask if m)
    else:
        valid_count = len(encoded)
    if valid_count == 0:
        valid_count = len(encoded)

    pooled = [0.0] * d_model
    for t in range(len(encoded)):
        w = 1.0 / valid_count if (not mask or mask[t]) else 0.0
        for i in range(d_model):
            pooled[i] += encoded[t][i] * (1.0 / valid_count if (not mask or mask[t]) else 0.0)

    # 重新计算正确的 pooled (用简单写法)
    pooled = [0.0] * d_model
    valid_positions = [t for t in range(len(encoded)) if not mask or mask[t]]
    if not valid_positions:
        valid_positions = list(range(len(encoded)))
    for t in valid_positions:
        for i in range(d_model):
            pooled[i] += encoded[t][i]
    for i in range(d_model):
        pooled[i] /= len(valid_positions)

    # 4. 输出头
    raw_score = model.output_head.forward(pooled)[0]

    # 5. 损失
    loss = (raw_score - target) ** 2

    # ── 反向传播 ──
    # 1. MSE 梯度
    d_loss = _mse_loss_grad(raw_score, target)  # scalar

    # 2. 输出头反向
    dout_head = [d_loss]  # 1-dim gradient
    d_pooled = model.output_head.backward(dout_head, lr)  # (d_model,)

    # 3. 均值池化反向：每个有效位置均分梯度
    d_encoded = [[0.0] * d_model for _ in range(len(encoded))]
    for t in valid_positions:
        for i in range(d_model):
            d_encoded[t][i] = d_pooled[i] / len(valid_positions)

    # 4. 逐层反向传播编码器
    current_grad = d_encoded
    for layer_idx in range(model.encoder.n_layers - 1, -1, -1):
        layer = model.encoder.layers[layer_idx]
        cache = layer._cache

        # 反向传播 LayerNorm2
        d_norm2 = []
        for t in range(len(current_grad)):
            dx_t, dgamma2, dbeta2 = _layer_norm_backward(
                current_grad[t],
                cache["residual2"][t] if "residual2" in cache else cache["norm1_out"][t],
                cache["cache_norm2"]["means"][t],
                cache["cache_norm2"]["stds"][t],
                layer.norm2_gamma,
            )
            d_norm2.append(dx_t)
            # 更新 norm2 参数
            for i in range(d_model):
                layer.norm2_gamma[i] -= lr * dgamma2[i] * 0.1  # 衰减
                layer.norm2_beta[i] -= lr * dbeta2[i] * 0.1

        # FFN 反向传播
        ffn_caches = cache.get("ffn_per_position", [])
        d_norm1_out = []
        for t in range(len(d_norm2)):
            dx_ffn = list(d_norm2[t])
            dx_residual = list(d_norm2[t])

            # FFN 路径: norm1_out → ffn1 → ReLU → ffn2
            d_h1 = layer.ffn2.backward(dx_ffn, lr)
            h1_out = ffn_caches[t]["h1_out"] if t < len(ffn_caches) else [0.0] * len(d_h1)
            d_h1_relu = [_relu_grad(h1_out[i], d_h1[i]) for i in range(len(d_h1))]
            d_norm1_out_ffn = layer.ffn1.backward(d_h1_relu, lr)

            merged = [d_norm1_out_ffn[i] + dx_residual[i] for i in range(len(dx_residual))]
            d_norm1_out.append(merged)

        # 反向传播 LayerNorm1
        d_attn_out = []
        for t in range(len(d_norm1_out)):
            dx_t, dgamma1, dbeta1 = _layer_norm_backward(
                d_norm1_out[t],
                cache["residual1"][t],
                cache["cache_norm1"]["means"][t],
                cache["cache_norm1"]["stds"][t],
                layer.norm1_gamma,
            )
            d_attn_out.append(dx_t)
            for i in range(d_model):
                layer.norm1_gamma[i] -= lr * dgamma1[i] * 0.1
                layer.norm1_beta[i] -= lr * dbeta1[i] * 0.1

        # 残差反向: d_attn_out 同时流向 attention 输出和原始输入
        # 简化: attention 的梯度通过 W_o 反向
        attn_cache = layer.self_attn._cache
        d_attn_input = []
        for t in range(len(d_attn_out)):
            # 通过 W_o 反向 (简化)
            d_attn_input_t = layer.self_attn.W_o.backward(d_attn_out[t], lr)
            # 加上残差梯度
            for i in range(d_model):
                d_attn_input_t[i] += d_attn_out[t][i]
            d_attn_input.append(d_attn_input_t)

        current_grad = d_attn_input

    # 5. 嵌入层投影反向
    for t in range(len(current_grad)):
        model.embedding.projection.backward(current_grad[t], lr * 0.5)

    return loss


def train_transformer(
    model: StockTransformer,
    samples: list[TrainingSample],
    epochs: int = 50,
    lr: float = 0.001,
    grad_clip: float = 5.0,
    verbose: bool = True,
) -> list[float]:
    """训练 Transformer 模型。

    Args:
        model: StockTransformer 实例
        samples: 训练样本列表
        epochs: 训练轮数
        lr: 学习率
        grad_clip: 梯度裁剪阈值
        verbose: 是否打印进度

    Returns:
        list[float]: 每轮的 epoch 平均损失
    """
    epoch_losses: list[float] = []

    for epoch in range(epochs):
        # 打乱样本顺序
        shuffled = list(samples)
        random.shuffle(shuffled)

        total_loss = 0.0
        for sample in shuffled:
            loss = _train_step(model, sample, lr, grad_clip)
            total_loss += loss

        avg_loss = total_loss / len(shuffled)
        epoch_losses.append(avg_loss)

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}: loss = {avg_loss:.6f}")

    return epoch_losses
