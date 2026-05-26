"""输入投影 + 正弦位置编码。"""

import math
from typing import Optional


def _init_xavier_uniform(fan_in: int, fan_out: int) -> list[list[float]]:
    """Xavier 均匀初始化：limit = sqrt(6 / (fan_in + fan_out))"""
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return [
        [_rand_uniform(-limit, limit) for _ in range(fan_in)]
        for _ in range(fan_out)
    ]


def _rand_uniform(lo: float, hi: float) -> float:
    """简易均匀随机数（0~1 的线性同余）。"""
    import random
    return random.uniform(lo, hi)


class Linear:
    """纯 Python 线性层 y = xW^T + b，含前向和反向传播。"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.W = _init_xavier_uniform(in_features, out_features)  # (out, in)
        self.b = [0.0] * out_features if bias else None
        self._cache: dict = {}  # 存储前向中间值供反向使用

    def forward(self, x: list[float]) -> list[float]:
        """y = xW^T + b，x: (in_features,), y: (out_features,)"""
        W, b = self.W, self.b
        out = [0.0] * self.out_features
        for i in range(self.out_features):
            s = 0.0
            row = W[i]
            for j in range(self.in_features):
                s += row[j] * x[j]
            if b is not None:
                s += b[i]
            out[i] = s
        self._cache = {"x": list(x)}
        return out

    def forward_matrix(self, X: list[list[float]]) -> list[list[float]]:
        """批量前向：X (batch, in_features) → Y (batch, out_features)"""
        return [self.forward(row) for row in X]

    def backward(self, dout: list[float], lr: float) -> list[float]:
        """反向传播，SGD 更新参数。返回对输入的梯度。"""
        x = self._cache["x"]
        # dW = dout ⊗ x  (outer product)
        for i in range(self.out_features):
            di = dout[i]
            row = self.W[i]
            for j in range(self.in_features):
                row[j] -= lr * di * x[j]
        # db = dout
        if self.b is not None:
            for i in range(self.out_features):
                self.b[i] -= lr * dout[i]
        # dx = dout · W  = W^T @ dout
        dx = [0.0] * self.in_features
        for j in range(self.in_features):
            s = 0.0
            for i in range(self.out_features):
                s += self.W[i][j] * dout[i]
            dx[j] = s
        return dx

    def to_dict(self) -> dict:
        return {
            "W": self.W,
            "b": self.b,
            "in_features": self.in_features,
            "out_features": self.out_features,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Linear":
        layer = cls(d["in_features"], d["out_features"], bias=d.get("b") is not None)
        layer.W = d["W"]
        layer.b = d["b"]
        return layer


class PositionalEncoding:
    """正弦/余弦位置编码 (不可训练)。"""

    def __init__(self, d_model: int, max_len: int = 60):
        self.d_model = d_model
        self.max_len = max_len
        self._pe: Optional[list[list[float]]] = None  # (max_len, d_model)
        self._init_pe()

    def _init_pe(self):
        pe: list[list[float]] = []
        for pos in range(self.max_len):
            row = [0.0] * self.d_model
            for i in range(0, self.d_model, 2):
                angle = pos / (10000.0 ** (i / self.d_model))
                row[i] = math.sin(angle)
                if i + 1 < self.d_model:
                    row[i + 1] = math.cos(angle)
            pe.append(row)
        self._pe = pe

    def forward(self, seq_len: int) -> list[list[float]]:
        """返回 (seq_len, d_model) 的位置编码矩阵。"""
        return [list(row) for row in self._pe[:seq_len]]

    def add_to(self, X: list[list[float]]) -> list[list[float]]:
        """将位置编码逐元素加到输入 X (seq_len, d_model) 上。"""
        seq_len = len(X)
        pe = self._pe[:seq_len]
        d_model = self.d_model
        result = []
        for t in range(seq_len):
            result.append([X[t][i] + pe[t][i] for i in range(d_model)])
        return result


class FeatureEmbedding:
    """特征嵌入: Linear(n_features → d_model) + 可选位置编码。"""

    def __init__(self, n_features: int, d_model: int, max_seq_len: int = 60):
        self.n_features = n_features
        self.d_model = d_model
        self.projection = Linear(n_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

    def forward(self, features: list[list[float]], add_position: bool = True) -> list[list[float]]:
        """输入 (seq_len, n_features) → 输出 (seq_len, d_model)。"""
        projected = self.projection.forward_matrix(features)
        if add_position:
            return self.pos_encoding.add_to(projected)
        return projected

    def to_dict(self) -> dict:
        return {
            "projection": self.projection.to_dict(),
            "n_features": self.n_features,
            "d_model": self.d_model,
        }

    @classmethod
    def from_dict(cls, d: dict, max_seq_len: int = 60) -> "FeatureEmbedding":
        emb = cls(d["n_features"], d["d_model"], max_seq_len)
        emb.projection = Linear.from_dict(d["projection"])
        return emb
