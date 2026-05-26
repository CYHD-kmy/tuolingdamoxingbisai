"""Transformer 模块单元测试。"""

import json
import math
import tempfile
import os
import random

random.seed(42)


# ── 测试辅助 ──


def _make_stock_daily(close, pct_chg, date="2026-05-01", volume=2e7):
    """创建 StockDaily-like 对象。"""
    return type("Record", (), {
        "date": date, "open": close * 0.99, "high": close * 1.02,
        "low": close * 0.98, "close": close, "volume": volume,
        "amount": close * volume * 0.7, "pct_chg": pct_chg,
        "turnover": 3.0, "ma5": close * 0.99, "ma10": close * 0.97,
        "ma20": close * 0.95, "macd_dif": 0.5, "macd_dea": 0.3,
        "macd_bar": 0.2, "rsi_6": 55.0, "rsi_14": 52.0,
    })()


def _make_stock_data(n_days=35, start_price=100.0):
    """生成 n_days 的 StockDaily 序列，带上升趋势。"""
    records = []
    price = start_price * 0.82  # 30 天前的价格
    for i in range(n_days):
        pct = 0.6 + random.uniform(-2.0, 3.0)
        close = price * (1 + pct / 100)
        r = _make_stock_daily(close, pct, date=f"2026-05-{i+1:02d}")
        records.append(r)
        price = close
    return records


# ── 数学工具测试 ──


def test_softmax():
    from src.transformer.attention import softmax

    probs = softmax([1.0, 2.0, 3.0])
    assert len(probs) == 3
    assert abs(sum(probs) - 1.0) < 1e-6
    assert probs[2] > probs[1] > probs[0]

    # 全零
    probs = softmax([0.0, 0.0, 0.0])
    assert abs(sum(probs) - 1.0) < 1e-6

    # 大数值稳定性
    probs = softmax([1000.0, 1000.0])
    assert abs(sum(probs) - 1.0) < 1e-6
    assert abs(probs[0] - 0.5) < 0.01

    # 空输入
    assert softmax([]) == []


def test_linear_forward():
    from src.transformer.embedding import Linear

    layer = Linear(10, 32)
    x = [1.0] * 10
    out = layer.forward(x)
    assert len(out) == 32
    assert all(isinstance(v, float) for v in out)


def test_linear_forward_matrix():
    from src.transformer.embedding import Linear

    layer = Linear(10, 32)
    X = [[float(i + j) for j in range(10)] for i in range(5)]
    out = layer.forward_matrix(X)
    assert len(out) == 5
    assert len(out[0]) == 32


def test_linear_backward():
    from src.transformer.embedding import Linear

    layer = Linear(10, 32)
    x = [1.0] * 10
    _ = layer.forward(x)
    dout = [0.1] * 32
    dx = layer.backward(dout, lr=0.01)
    assert len(dx) == 10
    assert all(isinstance(v, float) for v in dx)


def test_linear_serialization():
    from src.transformer.embedding import Linear

    layer = Linear(10, 32)
    d = layer.to_dict()
    restored = Linear.from_dict(d)
    x = [1.0] * 10
    out1 = layer.forward(x)
    out2 = restored.forward(x)
    for a, b in zip(out1, out2):
        assert abs(a - b) < 1e-10


def test_layer_norm():
    from src.transformer.encoder import _layer_norm

    d_model = 32
    gamma = [1.0] * d_model
    beta = [0.0] * d_model
    X = [[float(i + j) for j in range(d_model)] for i in range(3)]

    result, cache = _layer_norm(X, gamma, beta)
    assert len(result) == 3
    assert len(result[0]) == d_model

    # 验证输出均值和方差近似 0 和 1
    for row in result:
        m = sum(row) / d_model
        var = sum((x - m) ** 2 for x in row) / d_model
        assert abs(m) < 0.1, f"mean {m} not near 0"
        assert abs(var - 1.0) < 0.2, f"var {var} not near 1"


# ── 特征提取测试 ──


def test_feature_extraction():
    from src.transformer.features import extract_features, FEATURE_NAMES, N_FEATURES

    records = _make_stock_data(35)
    features = extract_features(records)
    assert len(features) == 35
    assert len(features[0]) == N_FEATURES
    assert N_FEATURES == 10
    assert FEATURE_NAMES[0] == "open"
    assert FEATURE_NAMES[-1] == "rsi_14"


def test_feature_extraction_empty():
    from src.transformer.features import extract_features

    assert extract_features([]) == []


def test_feature_pad_truncate():
    from src.transformer.features import extract_features, pad_or_truncate, N_FEATURES

    records = _make_stock_data(25)
    features = extract_features(records)

    # 截断
    padded, mask = pad_or_truncate(features, max_seq_len=20)
    assert len(padded) == 20
    assert len(mask) == 20
    assert all(mask)

    # 填充
    padded, mask = pad_or_truncate(features, max_seq_len=60)
    assert len(padded) == 60
    assert len(mask) == 60
    assert mask[-1] is True  # 最后的元素是真实的
    assert mask[0] is False  # 前面填充的是假的


# ── 位置编码测试 ──


def test_positional_encoding():
    from src.transformer.embedding import PositionalEncoding

    pe = PositionalEncoding(d_model=32, max_len=60)
    out = pe.forward(seq_len=10)
    assert len(out) == 10
    assert len(out[0]) == 32

    # 不同位置不应完全相同
    assert out[0] != out[5]

    # 值应在 [-1, 1] 范围
    for row in out:
        for v in row:
            assert -1.1 <= v <= 1.1


def test_feature_embedding():
    from src.transformer.embedding import FeatureEmbedding

    emb = FeatureEmbedding(n_features=10, d_model=32, max_seq_len=60)
    features = [[float(i + j) / 10 for j in range(10)] for i in range(5)]
    out = emb.forward(features, add_position=True)
    assert len(out) == 5
    assert len(out[0]) == 32


# ── 注意力测试 ──


def test_scaled_dot_product_attention():
    from src.transformer.attention import _scaled_dot_product_attention

    seq_len, head_dim = 5, 8
    Q = [[0.1 * (i + j) for j in range(head_dim)] for i in range(seq_len)]
    K = [[0.1 * (i + j) for j in range(head_dim)] for i in range(seq_len)]
    V = [[0.1 * (i + j) for j in range(head_dim)] for i in range(seq_len)]

    output, weights = _scaled_dot_product_attention(Q, K, V)
    assert len(output) == seq_len
    assert len(output[0]) == head_dim
    assert len(weights) == seq_len
    # 权重每行和为 1
    for row in weights:
        assert abs(sum(row) - 1.0) < 1e-6


def test_multi_head_attention():
    from src.transformer.attention import MultiHeadAttention

    mha = MultiHeadAttention(d_model=32, n_heads=4)
    seq_len = 10
    X = [[float(i + j) / 32 for j in range(32)] for i in range(seq_len)]

    out = mha.forward(X)
    assert len(out) == seq_len
    assert len(out[0]) == 32


def test_multi_head_attention_with_mask():
    from src.transformer.attention import MultiHeadAttention

    mha = MultiHeadAttention(d_model=32, n_heads=4)
    seq_len = 10
    X = [[float(i + j) / 32 for j in range(32)] for i in range(seq_len)]
    mask = [True] * 5 + [False] * 5  # 后 5 个是 padding

    out = mha.forward(X, mask)
    assert len(out) == seq_len


# ── 编码器测试 ──


def test_transformer_encoder_layer():
    from src.transformer.encoder import TransformerEncoderLayer

    layer = TransformerEncoderLayer(d_model=32, n_heads=4, d_ff=64)
    seq_len = 10
    X = [[float(i + j) / 32 for j in range(32)] for i in range(seq_len)]

    out = layer.forward(X)
    assert len(out) == seq_len
    assert len(out[0]) == 32


def test_transformer_encoder():
    from src.transformer.encoder import TransformerEncoder

    encoder = TransformerEncoder(d_model=32, n_heads=4, d_ff=64, n_layers=2)
    seq_len = 10
    X = [[float(i + j) / 32 for j in range(32)] for i in range(seq_len)]

    out = encoder.forward(X)
    assert len(out) == seq_len
    assert len(out[0]) == 32


def test_encoder_serialization():
    from src.transformer.encoder import TransformerEncoder

    encoder = TransformerEncoder(d_model=32, n_heads=4, d_ff=64, n_layers=2)
    d = encoder.to_dict()
    restored = TransformerEncoder.from_dict(d)

    seq_len = 5
    X = [[float(i + j) / 32 for j in range(32)] for i in range(seq_len)]
    out1 = encoder.forward(X)
    out2 = restored.forward(X)
    for t in range(seq_len):
        for i in range(32):
            assert abs(out1[t][i] - out2[t][i]) < 1e-10


# ── 完整模型测试 ──


def test_stock_transformer_forward():
    from src.transformer.model import StockTransformer

    model = StockTransformer()
    features = [[float(i + j) / 20 for j in range(10)] for i in range(30)]

    score = model.forward(features)
    assert isinstance(score, float)
    assert -5.0 < score < 5.0  # 合理范围


def test_stock_transformer_with_mask():
    from src.transformer.model import StockTransformer

    model = StockTransformer()
    features = [[float(i + j) / 20 for j in range(10)] for i in range(30)]
    mask = [False] * 5 + [True] * 25

    score = model.forward(features, mask)
    assert isinstance(score, float)


def test_stock_transformer_encode():
    from src.transformer.model import StockTransformer

    model = StockTransformer()
    features = [[float(i + j) / 20 for j in range(10)] for i in range(20)]

    encoded = model.encode(features)
    assert len(encoded) == 20
    assert len(encoded[0]) == 32


def test_model_save_load_roundtrip():
    from src.transformer.model import StockTransformer

    model = StockTransformer()
    features = [[float(i + j) / 20 for j in range(10)] for i in range(30)]

    score_before = model.forward(features)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name
        model.save(path)

    try:
        loaded = StockTransformer.load(path)
        score_after = loaded.forward(features)
        assert abs(score_before - score_after) < 1e-10
    finally:
        os.unlink(path)


def test_model_json_size():
    from src.transformer.model import StockTransformer

    model = StockTransformer()
    d = model.to_dict()
    json_str = json.dumps(d)
    # 确保模型大小合理 (< 500KB)
    assert len(json_str) < 500_000


# ── 训练测试 ──


def test_generate_training_data():
    from src.transformer.training import generate_training_data

    records_list = {}
    for code in ["600519", "000858"]:
        records_list[code] = _make_stock_data(60)

    samples = generate_training_data(
        records_list, seq_len=30, forward_days=5, max_seq_len=60
    )
    assert len(samples) > 0
    assert all(isinstance(s.target, float) for s in samples)
    assert all(len(s.features) == 60 for s in samples)
    assert all(len(s.features[0]) == 10 for s in samples)


def test_training_loop_basic():
    from src.transformer.model import StockTransformer
    from src.transformer.training import generate_training_data, train_transformer

    records_list = {}
    for code in ["600519"]:
        records_list[code] = _make_stock_data(50)

    samples = generate_training_data(
        records_list, seq_len=30, forward_days=5, max_seq_len=30
    )
    assert len(samples) > 0

    model = StockTransformer(max_seq_len=30)
    losses = train_transformer(model, samples, epochs=10, lr=0.001, verbose=False)
    assert len(losses) == 10
    # 训练后 loss 应降到一个合理值
    assert losses[-1] < 1.0


def test_overfitting_capability():
    """训练模型过拟合 5 个相同样本，验证学习能力。"""
    from src.transformer.model import StockTransformer
    from src.transformer.training import TrainingSample, train_transformer

    # 创建 5 个相同样本，目标为 0.05
    samples = []
    for _ in range(5):
        features = [[float(i + j) / 20 for j in range(10)] for i in range(30)]
        mask = [True] * 30
        samples.append(TrainingSample(features, mask, 0.05))

    model = StockTransformer(max_seq_len=30)
    losses = train_transformer(model, samples, epochs=100, lr=0.01, verbose=False)
    # 过拟合后 loss 应很小
    assert losses[-1] < 0.01, f"Overfitting loss {losses[-1]} not small enough"


# ── Scorer 测试 ──


def test_transformer_scorer_single():
    from src.transformer.model import StockTransformer
    from src.transformer.scorer import TransformerScorer

    model = StockTransformer()
    scorer = TransformerScorer(model)

    records = _make_stock_data(35)
    fs = scorer.score_single("600519", records, "贵州茅台")
    assert fs is not None
    assert fs.code == "600519"
    assert fs.name == "贵州茅台"
    assert 5.0 <= fs.composite <= 95.0
    assert "transformer" in fs.scores


def test_transformer_scorer_all():
    from src.transformer.model import StockTransformer
    from src.transformer.scorer import TransformerScorer

    model = StockTransformer()
    scorer = TransformerScorer(model)

    daily_data = {
        "600519": _make_stock_data(35),
        "000858": _make_stock_data(35),
    }
    codes = ["600519", "000858"]

    results = scorer.score_all(codes, daily_data)
    assert len(results) == 2
    # 按 composite 降序
    assert results[0].composite >= results[1].composite


def test_transformer_scorer_insufficient_data():
    from src.transformer.model import StockTransformer
    from src.transformer.scorer import TransformerScorer

    model = StockTransformer()
    scorer = TransformerScorer(model)

    records = _make_stock_data(10)  # 不足 20 条
    fs = scorer.score_single("600519", records)
    assert fs is None


def test_raw_to_score_mapping():
    from src.transformer.scorer import TransformerScorer

    # 极端正值 → 接近 95
    assert TransformerScorer._raw_to_score(0.3) >= 90
    # 极端负值 → 接近 5
    assert TransformerScorer._raw_to_score(-0.3) <= 10
    # 零 → 50
    assert abs(TransformerScorer._raw_to_score(0.0) - 50.0) < 1e-6
    # 裁剪
    assert TransformerScorer._raw_to_score(10.0) == 95.0
    assert TransformerScorer._raw_to_score(-10.0) == 5.0


# ── 数学辅助测试 ──


def test_matmul():
    from src.transformer.attention import _matmul

    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[5.0, 6.0], [7.0, 8.0]]
    result = _matmul(A, B)
    assert len(result) == 2
    assert len(result[0]) == 2
    assert abs(result[0][0] - 19.0) < 1e-6  # 1*5 + 2*7
    assert abs(result[0][1] - 22.0) < 1e-6  # 1*6 + 2*8
    assert abs(result[1][0] - 43.0) < 1e-6  # 3*5 + 4*7


def test_transpose():
    from src.transformer.attention import _transpose

    X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    XT = _transpose(X)
    assert len(XT) == 3
    assert len(XT[0]) == 2
    assert XT[0][0] == 1.0
    assert XT[0][1] == 4.0
    assert XT[2][1] == 6.0
