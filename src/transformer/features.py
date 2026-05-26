"""StockDaily → 数值特征向量 (10 维)，序列内 Z-score 归一化。"""

from typing import Any

# 选用的 10 个特征字段
FEATURE_NAMES = [
    "open", "high", "low", "close",
    "volume", "amount",
    "pct_chg", "turnover",
    "macd_bar", "rsi_14",
]

N_FEATURES = len(FEATURE_NAMES)


def _get_attr(obj: Any, name: str) -> float:
    """安全取值，兼容 dict / dataclass / 普通对象。"""
    if isinstance(obj, dict):
        return float(obj.get(name, 0.0))
    return float(getattr(obj, name, 0.0))


def extract_features(daily_records: list[Any]) -> list[list[float]]:
    """将 StockDaily 列表转为 (seq_len, 10) 特征矩阵，并做 Z-score 归一化。

    Args:
        daily_records: StockDaily 对象列表，按日期升序。

    Returns:
        list[list[float]]: shape (seq_len, 10)，每个内部 list 为 10 维特征。
        若序列长度 < 2，不做归一化直接返回原始值。
    """
    if not daily_records:
        return []

    # 1. 提取原始值
    raw: list[list[float]] = []
    for r in daily_records:
        raw.append([_get_attr(r, name) for name in FEATURE_NAMES])

    seq_len = len(raw)
    if seq_len < 2:
        return raw

    # 2. 按列计算均值与标准差
    means: list[float] = []
    stds: list[float] = []
    for col_idx in range(N_FEATURES):
        col = [row[col_idx] for row in raw]
        m = sum(col) / seq_len
        variance = sum((x - m) ** 2 for x in col) / seq_len
        s = variance ** 0.5
        means.append(m)
        stds.append(max(s, 1e-8))

    # 3. Z-score 归一化
    normalized: list[list[float]] = []
    for row in raw:
        normalized.append([
            (row[i] - means[i]) / stds[i]
            for i in range(N_FEATURES)
        ])

    return normalized


def pad_or_truncate(
    features: list[list[float]],
    max_seq_len: int,
) -> tuple[list[list[float]], list[bool]]:
    """填充/截断到固定长度，返回 (特征矩阵, 有效位置 mask)。

    不足 max_seq_len 时前面填 0（早期日期不足）；
    超过 max_seq_len 时保留最近 max_seq_len 条。
    """
    seq_len = len(features)
    if seq_len == 0:
        return [[0.0] * N_FEATURES] * max_seq_len, [False] * max_seq_len

    if seq_len > max_seq_len:
        features = features[-max_seq_len:]
        return features, [True] * max_seq_len

    # 前面补零
    pad_count = max_seq_len - seq_len
    padded = [[0.0] * N_FEATURES for _ in range(pad_count)] + features
    mask = [False] * pad_count + [True] * seq_len
    return padded, mask
