"""Transformer 编码器层 + 编码器堆栈。"""

from .attention import MultiHeadAttention
from .embedding import Linear


def _layer_norm(
    X: list[list[float]], gamma: list[float], beta: list[float], eps: float = 1e-5
) -> tuple[list[list[float]], dict]:
    """Layer Normalization 前向。

    Args:
        X: (seq_len, d_model)
        gamma, beta: (d_model,) 可训练参数
        eps: 防止除零

    Returns:
        (normalized_output, cache_dict)
    """
    d_model = len(gamma)
    result = []
    cache = {"x": X, "means": [], "stds": [], "x_hat": []}

    for row in X:
        mean = sum(row) / d_model
        variance = sum((x - mean) ** 2 for x in row) / d_model
        std = (variance + eps) ** 0.5

        x_hat = [(x - mean) / std for x in row]
        out = [x_hat[i] * gamma[i] + beta[i] for i in range(d_model)]

        result.append(out)
        cache["means"].append(mean)
        cache["stds"].append(std)
        cache["x_hat"].append(x_hat)

    return result, cache


class TransformerEncoderLayer:
    """单层 Transformer 编码器：Self-Attention + FFN，每子层含残差连接 + LayerNorm。"""

    def __init__(self, d_model: int = 32, n_heads: int = 4, d_ff: int = 64):
        self.d_model = d_model
        self.self_attn = MultiHeadAttention(d_model, n_heads)

        # FFN: Linear(d_model → d_ff) → ReLU → Linear(d_ff → d_model)
        self.ffn1 = Linear(d_model, d_ff)
        self.ffn2 = Linear(d_ff, d_model)

        # LayerNorm 参数 (每层两个)
        self.norm1_gamma = [1.0] * d_model
        self.norm1_beta = [0.0] * d_model
        self.norm2_gamma = [1.0] * d_model
        self.norm2_beta = [0.0] * d_model

        self._cache: dict = {}

    def forward(self, X: list[list[float]], mask: list[bool] | None = None) -> list[list[float]]:
        """前向传播。

        X → Self-Attention → Add&Norm → FFN → Add&Norm → output
        """
        # 子层 1: Self-Attention + 残差 + LayerNorm
        attn_out = self.self_attn.forward(X, mask)
        residual1 = [[X[t][i] + attn_out[t][i] for i in range(self.d_model)] for t in range(len(X))]
        norm1_out, cache_norm1 = _layer_norm(residual1, self.norm1_gamma, self.norm1_beta)

        # 子层 2: FFN + 残差 + LayerNorm
        ffn_out = self._ffn_forward(norm1_out)
        ffn_per_pos = self._cache.get("ffn_per_position", [])  # 保留 _ffn_forward 写入的缓存
        residual2 = [[norm1_out[t][i] + ffn_out[t][i] for i in range(self.d_model)] for t in range(len(X))]
        norm2_out, cache_norm2 = _layer_norm(residual2, self.norm2_gamma, self.norm2_beta)

        self._cache = {
            "X": X, "attn_out": attn_out, "residual1": residual1,
            "norm1_out": norm1_out, "cache_norm1": cache_norm1,
            "ffn_out": ffn_out, "ffn_per_position": ffn_per_pos,
            "residual2": residual2,
            "norm2_out": norm2_out, "cache_norm2": cache_norm2,
        }
        return norm2_out

    def _ffn_forward(self, X: list[list[float]]) -> list[list[float]]:
        """FFN: Linear → ReLU → Linear，保存每位置的缓存供反向传播"""
        out = []
        ffn_cache: list[dict] = []  # 每位置的中间值
        for row in X:
            h1 = self.ffn1.forward(row)
            h1_relu = [max(0.0, v) for v in h1]
            h2 = self.ffn2.forward(h1_relu)
            out.append(h2)
            ffn_cache.append({"h1_out": list(h1), "h1_relu": list(h1_relu)})
        self._cache["ffn_per_position"] = ffn_cache
        return out

    def to_dict(self) -> dict:
        return {
            "d_model": self.d_model,
            "n_heads": self.self_attn.n_heads,
            "d_ff": self.ffn1.out_features,
            "self_attn": self.self_attn.to_dict(),
            "ffn1": self.ffn1.to_dict(),
            "ffn2": self.ffn2.to_dict(),
            "norm1_gamma": self.norm1_gamma,
            "norm1_beta": self.norm1_beta,
            "norm2_gamma": self.norm2_gamma,
            "norm2_beta": self.norm2_beta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TransformerEncoderLayer":
        layer = cls(d["d_model"], d["n_heads"], d["d_ff"])
        layer.self_attn = MultiHeadAttention.from_dict(d["self_attn"])
        layer.ffn1 = Linear.from_dict(d["ffn1"])
        layer.ffn2 = Linear.from_dict(d["ffn2"])
        layer.norm1_gamma = d["norm1_gamma"]
        layer.norm1_beta = d["norm1_beta"]
        layer.norm2_gamma = d["norm2_gamma"]
        layer.norm2_beta = d["norm2_beta"]
        return layer


class TransformerEncoder:
    """Transformer 编码器堆栈：n_layers × TransformerEncoderLayer。"""

    def __init__(self, d_model: int = 32, n_heads: int = 4, d_ff: int = 64, n_layers: int = 2):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.layers = [
            TransformerEncoderLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ]

    def forward(self, X: list[list[float]], mask: list[bool] | None = None) -> list[list[float]]:
        """逐层前向传播。"""
        out = X
        for layer in self.layers:
            out = layer.forward(out, mask)
        return out

    def to_dict(self) -> dict:
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "n_layers": self.n_layers,
            "layers": [layer.to_dict() for layer in self.layers],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TransformerEncoder":
        encoder = cls(d["d_model"], d["n_heads"], d["d_ff"], d["n_layers"])
        encoder.layers = [
            TransformerEncoderLayer.from_dict(ld) for ld in d["layers"]
        ]
        return encoder
