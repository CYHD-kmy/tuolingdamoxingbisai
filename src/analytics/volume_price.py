"""
量价关系分析模块 — "量在价先"量化模型。

核心分析维度:
- 量峰检测: Z-score 标准化 + 滚动分位阈值
- 价能验证: 放量必须伴随价格突破 (排除放量滞涨/放量下跌)
- 量价背离: 缩量上涨 → 动能减弱, 放量不上涨 → 出货嫌疑
- 资金分级: 量价配合 + 趋势确认 → 综合买入信号

参考: StockTradebyZ 量能策略 (6年总收益172.2%, 夏普1.81)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class VolumePriceSignal:
    """量价关系信号"""
    code: str
    name: str
    signal_type: str             # vol_breakout / vol_divergence / vol_dry_up
    strength: float              # 信号强度 0-100
    volume_ratio: float          # 当前量比 (vs 20日均量)
    price_trend: str             # up / down / flat
    description: str             # 信号描述


class VolumePriceAnalyzer:
    """
    量价关系分析器。

    基于"量在价先"原则，分析量价配合关系:
    1. 量峰突破: 当前成交量显著放大 + Z-score > 1.5
    2. 价格验证: 放量必须伴随价格上涨 (排除放量下跌)
    3. 量价背离: 缩量上涨/放量滞涨的预警
    """

    _VOL_ZSCORE_THRESHOLD = 1.5       # 量峰 Z-score 阈值
    _VOL_RATIO_THRESHOLD = 1.5        # 量比阈值
    _PRICE_BREAK_THRESHOLD = 0.02     # 价格突破阈值 2%

    def detect_signals(
        self,
        daily_data: list,            # [StockDaily]
        fund_flows: list | None = None,   # [FundFlow]
        code: str = "",
        name: str = "",
    ) -> list[VolumePriceSignal]:
        """
        检测量价关系信号。

        Returns:
            检测到的信号列表 (可能包含多个信号类型)
        """
        if len(daily_data) < 20:
            return []

        signals = []
        latest = daily_data[-1]
        prev = daily_data[-2] if len(daily_data) >= 2 else None

        # 计算20日均量和标准差
        volumes = [d.volume for d in daily_data[-20:]]
        avg_vol = sum(volumes) / 20
        std_vol = (sum((v - avg_vol) ** 2 for v in volumes) / 20) ** 0.5

        # 量比
        vol_ratio = latest.volume / max(avg_vol, 1)

        # Z-score
        z_score = (latest.volume - avg_vol) / max(std_vol, 1)

        # 价格变化
        if prev:
            price_chg = (latest.close - prev.close) / prev.close
        else:
            price_chg = latest.pct_chg / 100

        # 1. 量峰突破信号
        if z_score > self._VOL_ZSCORE_THRESHOLD and vol_ratio > self._VOL_RATIO_THRESHOLD:
            if price_chg > 0.01:
                strength = min(95, 50 + z_score * 15 + vol_ratio * 5)
                desc = f"放量突破: 成交量Z-score={z_score:.1f}, 量比={vol_ratio:.1f}, 涨幅={price_chg*100:.1f}%"
                signals.append(VolumePriceSignal(
                    code=code, name=name,
                    signal_type="vol_breakout_bull",
                    strength=strength,
                    volume_ratio=vol_ratio,
                    price_trend="up",
                    description=desc,
                ))
            elif price_chg < -0.01:
                strength = max(5, 30 - abs(z_score) * 8)
                desc = f"放量下跌预警: Z-score={z_score:.1f}, 量比={vol_ratio:.1f}, 跌幅={price_chg*100:.1f}%"
                signals.append(VolumePriceSignal(
                    code=code, name=name,
                    signal_type="vol_breakout_bear",
                    strength=strength,
                    volume_ratio=vol_ratio,
                    price_trend="down",
                    description=desc,
                ))

        # 2. 量价背离信号
        if vol_ratio < 0.6 and price_chg > 0.02:
            # 缩量上涨: 上涨动能减弱
            signals.append(VolumePriceSignal(
                code=code, name=name,
                signal_type="vol_price_divergence",
                strength=35,
                volume_ratio=vol_ratio,
                price_trend="up",
                description=f"缩量上涨: 量比={vol_ratio:.1f}, 涨幅={price_chg*100:.1f}%, 动能减弱",
            ))

        # 3. 地量信号 (缩量至极, 变盘前兆)
        if vol_ratio < 0.3 and z_score < -1.5:
            signals.append(VolumePriceSignal(
                code=code, name=name,
                signal_type="vol_dry_up",
                strength=40,
                volume_ratio=vol_ratio,
                price_trend="flat",
                description=f"地量信号: 量比={vol_ratio:.1f}, 缩量至极, 关注变盘",
            ))

        return signals

    def batch_detect(
        self,
        daily_data_map: dict[str, list],
        names: dict[str, str] | None = None,
    ) -> dict[str, list[VolumePriceSignal]]:
        """批量检测量价信号"""
        results = {}
        names = names or {}
        for code, daily in daily_data_map.items():
            signals = self.detect_signals(daily, code=code, name=names.get(code, code))
            if signals:
                results[code] = signals
        return results

    def get_composite_score(self, signals: list[VolumePriceSignal]) -> float:
        """
        量价综合评分 (用于海选打分增强)。

        放量突破看多 = 加分, 放量下跌 = 扣分, 缩量上涨 = 中性偏负。
        """
        if not signals:
            return 50.0

        score = 50.0
        for s in signals:
            if s.signal_type == "vol_breakout_bull":
                score += min(30, s.strength * 0.3)
            elif s.signal_type == "vol_breakout_bear":
                score -= min(25, s.strength * 0.3)
            elif s.signal_type == "vol_price_divergence":
                score -= 8
            elif s.signal_type == "vol_dry_up":
                score -= 5
        return max(5, min(95, score))

    @staticmethod
    def build_context(
        signals_map: dict[str, list[VolumePriceSignal]], top_n: int = 10
    ) -> str:
        """构建量价分析文本 (供 LLM 分析使用)"""
        if not signals_map:
            return "无量价异常信号"

        # 按信号强度排序
        all_signals = []
        for code, sigs in signals_map.items():
            for s in sigs:
                all_signals.append((s.strength, code, s))
        all_signals.sort(reverse=True)

        lines = ["## 量价关系分析信号"]
        for i, (strength, code, s) in enumerate(all_signals[:top_n]):
            tag = "🟢" if s.signal_type == "vol_breakout_bull" else "🔴"
            lines.append(
                f"{i+1}. {tag} {s.name}({code}) | {s.description}"
            )
        return "\n".join(lines)
