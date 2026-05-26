"""StockTransformer — 完整 Transformer 评分模型 + JSON 序列化。

架构:
    Input (seq_len, 10) → FeatureEmbedding → (seq_len, 32)
    → PositionalEncoding → (seq_len, 32)
    → TransformerEncoder × 2 → (seq_len, 32)
    → Mean Pooling → (32,)
    → Output Head → (1,)  raw_score
"""

import json
from .embedding import FeatureEmbedding, Linear
from .encoder import TransformerEncoder


class StockTransformer:
    """完整的 Transformer 股票评分模型。

    forward(features) → raw_score (float)
    高分 → 预测正向收益 → 看多
    """

    def __init__(
        self,
        n_features: int = 10,
        d_model: int = 32,
        n_heads: int = 4,
        d_ff: int = 64,
        n_layers: int = 2,
        max_seq_len: int = 60,
    ):
        self.n_features = n_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        self.embedding = FeatureEmbedding(n_features, d_model, max_seq_len)
        self.encoder = TransformerEncoder(d_model, n_heads, d_ff, n_layers)
        self.output_head = Linear(d_model, 1)

    def forward(self, features: list[list[float]], mask: list[bool] | None = None) -> float:
        """前向传播：特征序列 → 原始评分。

        Args:
            features: (seq_len, n_features) 归一化特征矩阵
            mask: (seq_len,) padding mask

        Returns:
            float: 原始预测收益率（通常在 [-0.3, 0.3] 范围）
        """
        # 1. 嵌入 + 位置编码
        X = self.embedding.forward(features, add_position=True)

        # 2. Transformer 编码器
        encoded = self.encoder.forward(X, mask)  # (seq_len, d_model)

        # 3. 全局均值池化（仅对有效位置）
        if mask:
            valid_count = sum(1 for m in mask if m)
            if valid_count > 0:
                pooled = [0.0] * self.d_model
                for t in range(len(encoded)):
                    if mask[t]:
                        for i in range(self.d_model):
                            pooled[i] += encoded[t][i]
                pooled = [v / valid_count for v in pooled]
            else:
                # 全部 padding → 用全部位置
                pooled = [0.0] * self.d_model
                for t in range(len(encoded)):
                    for i in range(self.d_model):
                        pooled[i] += encoded[t][i]
                pooled = [v / len(encoded) for v in pooled]
        else:
            seq_len = len(encoded)
            pooled = [0.0] * self.d_model
            for t in range(seq_len):
                for i in range(self.d_model):
                    pooled[i] += encoded[t][i]
            pooled = [v / seq_len for v in pooled]

        # 4. 输出头
        raw = self.output_head.forward(pooled)  # [score]
        return raw[0]

    def encode(self, features: list[list[float]], mask: list[bool] | None = None) -> list[list[float]]:
        """获取编码器输出（不含池化和输出头），供 RL 等下游使用。

        Returns:
            (seq_len, d_model) 编码器最后层输出
        """
        X = self.embedding.forward(features, add_position=True)
        return self.encoder.forward(X, mask)

    def save(self, path: str) -> None:
        """保存模型为 JSON 文件。"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "StockTransformer":
        """从 JSON 文件加载模型。"""
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)

    def to_dict(self) -> dict:
        return {
            "n_features": self.n_features,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "n_layers": self.n_layers,
            "max_seq_len": self.max_seq_len,
            "embedding": self.embedding.to_dict(),
            "encoder": self.encoder.to_dict(),
            "output_head": self.output_head.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StockTransformer":
        model = cls(
            n_features=d["n_features"],
            d_model=d["d_model"],
            n_heads=d["n_heads"],
            d_ff=d["d_ff"],
            n_layers=d["n_layers"],
            max_seq_len=d.get("max_seq_len", 60),
        )
        model.embedding = FeatureEmbedding.from_dict(
            d["embedding"], d.get("max_seq_len", 60)
        )
        model.encoder = TransformerEncoder.from_dict(d["encoder"])
        model.output_head = Linear.from_dict(d["output_head"])
        return model
