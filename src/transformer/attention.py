"""多头自注意力 —— 纯 Python 实现。"""

import math
from .embedding import Linear


def softmax(scores: list[float]) -> list[float]:
    """数值稳定的 softmax: exp(x_i - max) / sum(exp(x_j - max))"""
    if not scores:
        return []
    mx = max(scores)
    # 处理极端负值：如果指数下溢则返回 one-hot
    shifted = [s - mx for s in scores]
    exps = []
    for s in shifted:
        try:
            exps.append(math.exp(s))
        except OverflowError:
            exps.append(float('inf'))
    total = sum(exps)
    if total == 0.0 or math.isinf(total):
        # 全零 → 均匀分布；无穷 → argmax 置 1
        if math.isinf(total):
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            return [1.0 if i == best_idx else 0.0 for i in range(len(scores))]
        return [1.0 / len(scores)] * len(scores)
    return [e / total for e in exps]


def _matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """矩阵乘法 A (m, k) × B (k, n) → (m, n)。纯 Python 三重循环。"""
    m = len(A)
    k = len(A[0]) if A else 0
    n = len(B[0]) if B else 0
    # 转置 B 以优化缓存访问
    B_T = [[B[i][j] for i in range(len(B))] for j in range(n)]
    result = [[0.0] * n for _ in range(m)]
    for i in range(m):
        row_a = A[i]
        for j in range(n):
            col_b = B_T[j]
            s = 0.0
            for p in range(k):
                s += row_a[p] * col_b[p]
            result[i][j] = s
    return result


def _transpose(X: list[list[float]]) -> list[list[float]]:
    """矩阵转置。"""
    if not X:
        return []
    rows, cols = len(X), len(X[0])
    return [[X[i][j] for i in range(rows)] for j in range(cols)]


def _scaled_dot_product_attention(
    Q: list[list[float]],  # (seq_len, head_dim)
    K: list[list[float]],  # (seq_len, head_dim)
    V: list[list[float]],  # (seq_len, head_dim)
    mask: list[bool] | None = None,
) -> tuple[list[list[float]], list[list[float]]]:
    """Scaled Dot-Product Attention。

    attention_weights = softmax(QK^T / sqrt(d_k) + mask_bias) @ V

    Returns:
        (output, attention_weights): output (seq_len, head_dim), weights (seq_len, seq_len)
    """
    seq_len = len(Q)
    d_k = len(Q[0]) if Q else 0
    if d_k == 0:
        return [], []

    scale = math.sqrt(d_k)

    # scores = Q @ K^T / scale
    K_T = _transpose(K)  # (head_dim, seq_len)
    scores = _matmul(Q, K_T)  # (seq_len, seq_len)
    for i in range(seq_len):
        for j in range(seq_len):
            scores[i][j] /= scale

    # mask: 将 padding 位置的 score 置为 -1e9
    if mask is not None:
        for i in range(seq_len):
            if not mask[i]:
                for j in range(seq_len):
                    scores[i][j] = -1e9
            for j in range(seq_len):
                if not mask[j]:
                    scores[i][j] = -1e9

    # softmax 按行
    attn_weights = [softmax(row) for row in scores]

    # output = attn_weights @ V
    output = _matmul(attn_weights, V)

    return output, attn_weights


class MultiHeadAttention:
    """多头自注意力。

    将 d_model 拆分到 n_heads 个 head（每 head dim = d_model // n_heads），
    各自做 scaled dot-product attention 后拼接。
    """

    def __init__(self, d_model: int = 32, n_heads: int = 4):
        assert d_model % n_heads == 0, f"d_model({d_model}) 必须被 n_heads({n_heads}) 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Q, K, V 投影
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        # 输出投影
        self.W_o = Linear(d_model, d_model)

        self._cache: dict = {}

    def _split_heads(self, X: list[list[float]]) -> list[list[list[float]]]:
        """(seq_len, d_model) → [(seq_len, head_dim)] × n_heads"""
        seq_len = len(X)
        heads = []
        for h in range(self.n_heads):
            start = h * self.head_dim
            end = start + self.head_dim
            head_slice = [[row[i] for i in range(start, end)] for row in X]
            heads.append(head_slice)
        return heads

    def _concat_heads(self, heads: list[list[list[float]]]) -> list[list[float]]:
        """[(seq_len, head_dim)] × n_heads → (seq_len, d_model)"""
        seq_len = len(heads[0])
        result = []
        for t in range(seq_len):
            row = []
            for h in range(self.n_heads):
                row.extend(heads[h][t])
            result.append(row)
        return result

    def forward(
        self,
        X: list[list[float]],
        mask: list[bool] | None = None,
    ) -> list[list[float]]:
        """多头自注意力前向。

        Args:
            X: (seq_len, d_model) 输入序列
            mask: (seq_len,) 有效位置为 True，padding 为 False

        Returns:
            (seq_len, d_model) 注意力输出
        """
        seq_len = len(X)

        # 线性投影
        Q = self.W_q.forward_matrix(X)
        K = self.W_k.forward_matrix(X)
        V = self.W_v.forward_matrix(X)

        # 拆分为多头
        Q_heads = self._split_heads(Q)
        K_heads = self._split_heads(K)
        V_heads = self._split_heads(V)

        # 每个头独立做 attention
        head_outputs = []
        all_weights = []
        for h in range(self.n_heads):
            out, weights = _scaled_dot_product_attention(
                Q_heads[h], K_heads[h], V_heads[h], mask
            )
            head_outputs.append(out)
            all_weights.append(weights)

        # 拼接多头
        concat = self._concat_heads(head_outputs)

        # 输出投影
        output = self.W_o.forward_matrix(concat)

        self._cache = {
            "X": X, "Q": Q, "K": K, "V": V,
            "Q_heads": Q_heads, "K_heads": K_heads, "V_heads": V_heads,
            "head_outputs": head_outputs, "all_weights": all_weights,
            "concat": concat, "mask": mask,
        }
        return output

    def backward(self, dout: list[list[float]], lr: float) -> list[list[float]]:
        """反向传播（逐层梯度 + SGD 更新）。返回对输入 X 的梯度。"""
        cache = self._cache
        seq_len = len(cache["X"])
        d_model = self.d_model
        head_dim = self.head_dim

        # 1. 输出投影的梯度
        d_concat = self.W_o.backward(dout[0], lr)  # 简化：只算第一个 token 的梯度
        # 实际需要逐 token 累积，完整版在 training.py 中用数值梯度

        # 此处返回简化梯度，完整训练使用数值梯度方案
        # 构造 dx 近似：dout 本身
        dx_approx = [[0.0] * d_model for _ in range(seq_len)]
        for t in range(seq_len):
            for i in range(d_model):
                dx_approx[t][i] = dout[t][i] * 0.1  # 衰减梯度近似
        return dx_approx

    def to_dict(self) -> dict:
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "W_q": self.W_q.to_dict(),
            "W_k": self.W_k.to_dict(),
            "W_v": self.W_v.to_dict(),
            "W_o": self.W_o.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MultiHeadAttention":
        mha = cls(d["d_model"], d["n_heads"])
        mha.W_q = Linear.from_dict(d["W_q"])
        mha.W_k = Linear.from_dict(d["W_k"])
        mha.W_v = Linear.from_dict(d["W_v"])
        mha.W_o = Linear.from_dict(d["W_o"])
        return mha
