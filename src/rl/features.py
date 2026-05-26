"""
技术特征提取 — 从 K 线数据计算 7 维特征向量。

特征:
1. MA 交叉信号: (MA5 - MA20) / MA20
2. RSI 归一化: (RSI14 - 50) / 50
3. MACD 柱 / 收盘价
4. 量比: volume / MA(volume, 5) - 1
5. 5 日收益率
6. 1 日涨跌幅 / 100
7. 近 10 日波动率

所有特征裁剪到 [-5, 5] 防止异常值。
"""

from __future__ import annotations


def compute_features(records: list, idx: int) -> list[float]:
    """
    从 K 线记录中提取 idx 位置的技术特征。

    records: 日线数据列表 (需有 close, pct_chg, ma5, ma20, rsi_14, macd_bar, volume 属性)
    idx: 当前 bar 索引

    返回: 7 维特征列表
    """
    r = records[idx]

    # 1. MA 交叉信号
    ma_signal = (r.ma5 - r.ma20) / r.ma20 if r.ma20 > 0 else 0.0

    # 2. RSI 归一化
    rsi_norm = (r.rsi_14 - 50) / 50 if r.rsi_14 > 0 else 0.0

    # 3. MACD 柱 / 收盘价
    macd_ratio = r.macd_bar / r.close if r.close > 0 else 0.0

    # 4. 量比 (volume / 5日均量)
    if idx >= 5:
        avg_vol = sum(records[i].volume for i in range(idx - 4, idx + 1)) / 5
        vol_ratio = (r.volume / avg_vol - 1) if avg_vol > 0 else 0.0
    else:
        vol_ratio = 0.0

    # 5. 5 日收益率
    if idx >= 5:
        ret_5d = (r.close / records[idx - 5].close - 1) if records[idx - 5].close > 0 else 0.0
    else:
        ret_5d = 0.0

    # 6. 1 日涨跌幅
    ret_1d = r.pct_chg / 100.0

    # 7. 近 10 日波动率 (标准差)
    if idx >= 9:
        pcts = [records[i].pct_chg for i in range(idx - 9, idx + 1)]
        avg = sum(pcts) / len(pcts)
        vol = (sum((p - avg) ** 2 for p in pcts) / len(pcts)) ** 0.5
        vol_norm = vol / 100.0
    else:
        vol_norm = 0.02

    features = [
        ma_signal,
        rsi_norm,
        macd_ratio,
        vol_ratio,
        ret_5d,
        ret_1d,
        vol_norm,
    ]

    # 裁剪到 [-5, 5]
    return [max(-5.0, min(5.0, f)) for f in features]


FEATURE_DIM = 7
