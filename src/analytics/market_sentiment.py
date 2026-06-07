"""
市场情绪与板块轮动分析模块。

分析维度:
- 市场广度: 涨跌比、涨停跌停比、成交额变化
- 市场状态: 高潮/发酵/启动/低迷/冰点 (DeepPulse 五阶段模型)
- 板块热度: 板块涨幅排名、涨停家数、资金流向
- 板块轮动: 热点切换检测、持续天数
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── 市场状态枚举 ──────────────────────────────

MARKET_REGIME_CLIMAX = "高潮"       # 赚钱效应极强，谨慎追高
MARKET_REGIME_FERMENT = "发酵"      # 赚钱效应扩散，积极参与
MARKET_REGIME_START = "启动"        # 市场回暖信号，试探性参与
MARKET_REGIME_DULL = "低迷"         # 震荡偏弱，控制仓位
MARKET_REGIME_FREEZE = "冰点"       # 恐慌/暴跌，空仓或极轻仓


@dataclass
class MarketBreadth:
    """市场广度数据"""
    up_count: int = 0            # 上涨家数
    down_count: int = 0          # 下跌家数
    flat_count: int = 0          # 平盘家数
    limit_up_count: int = 0      # 涨停家数 (含一字板)
    limit_down_count: int = 0    # 跌停家数
    total_volume_yi: float = 0.0 # 全市场成交额 (亿)
    prev_volume_yi: float = 0.0  # 前一日成交额 (亿)

    @property
    def up_ratio(self) -> float:
        total = self.up_count + self.down_count + self.flat_count
        return self.up_count / max(total, 1)

    @property
    def adv_decline_ratio(self) -> float:
        """涨跌比"""
        return self.up_count / max(self.down_count, 1)

    @property
    def volume_change_pct(self) -> float:
        """成交额变化百分比"""
        if self.prev_volume_yi <= 0:
            return 0.0
        return (self.total_volume_yi - self.prev_volume_yi) / self.prev_volume_yi * 100

    @property
    def limit_ratio(self) -> float:
        """涨停/跌停比"""
        return self.limit_up_count / max(self.limit_down_count, 1)


@dataclass
class SectorHeat:
    """板块热度"""
    sector_name: str
    pct_chg: float              # 板块涨幅
    limit_up_count: int = 0     # 板块内涨停家数
    net_inflow_yi: float = 0.0  # 板块资金净流入 (亿)
    consecutive_days: int = 0   # 连续走强天数
    leading_stocks: list[str] = field(default_factory=list)  # 领涨股代码


@dataclass
class MarketSentimentResult:
    """市场情绪分析结果"""
    breadth: MarketBreadth
    regime: str                         # 市场状态
    regime_score: float                 # 状态评分 0-100
    hot_sectors: list[SectorHeat]       # 热门板块 Top-5
    cooling_sectors: list[SectorHeat]   # 降温板块 Top-3
    participation_score: float          # 市场参与度评分 0-100
    risk_level: str                     # 市场风险等级 low/medium/high/extreme
    position_advice: float              # 建议仓位比例 0-1.0
    summary: str                        # 一句话总结


class MarketSentimentAnalyzer:
    """
    市场情绪分析器。

    分析全市场广度数据，判断市场状态和参与度，
    为仓位管理和选股提供宏观背景。
    """

    # 市场状态判定阈值
    _REGIME_THRESHOLDS = {
        MARKET_REGIME_CLIMAX:  {"up_ratio": 0.75, "limit_up": 80},
        MARKET_REGIME_FERMENT: {"up_ratio": 0.55, "limit_up": 40},
        MARKET_REGIME_START:   {"up_ratio": 0.45, "limit_up": 20},
        MARKET_REGIME_DULL:    {"up_ratio": 0.30, "limit_up": 10},
        MARKET_REGIME_FREEZE:  {"up_ratio": 0.15, "limit_up": 5},
    }

    # 各状态下的建议仓位
    _REGIME_POSITION = {
        MARKET_REGIME_CLIMAX:  0.50,  # 高潮期减仓
        MARKET_REGIME_FERMENT: 0.85,  # 发酵期积极参与
        MARKET_REGIME_START:   0.60,  # 启动期试探
        MARKET_REGIME_DULL:    0.35,  # 低迷期控仓
        MARKET_REGIME_FREEZE:  0.15,  # 冰点期极轻仓
    }

    def analyze(
        self,
        breadth: MarketBreadth | dict,
        hot_sectors: list[SectorHeat] | None = None,
        cooling_sectors: list[SectorHeat] | None = None,
    ) -> MarketSentimentResult:
        """综合市场情绪分析"""
        if isinstance(breadth, dict):
            breadth = MarketBreadth(
                up_count=breadth.get("up_count", 0),
                down_count=breadth.get("down_count", 0),
                flat_count=breadth.get("flat_count", 0),
                limit_up_count=breadth.get("limit_up_count", 0),
                limit_down_count=breadth.get("limit_down_count", 0),
                total_volume_yi=breadth.get("total_volume_yi", 0.0),
            )
        regime, regime_score = self._classify_regime(breadth)
        participation = self._calc_participation(breadth)
        risk = self._assess_risk(breadth, regime)
        position = self._REGIME_POSITION.get(regime, 0.5)

        # 根据成交额变化微调仓位
        if breadth.volume_change_pct > 20:
            position = min(1.0, position + 0.10)
        elif breadth.volume_change_pct < -30:
            position = max(0.10, position - 0.15)

        return MarketSentimentResult(
            breadth=breadth,
            regime=regime,
            regime_score=regime_score,
            hot_sectors=hot_sectors or [],
            cooling_sectors=cooling_sectors or [],
            participation_score=participation,
            risk_level=risk,
            position_advice=round(position, 2),
            summary=self._build_summary(breadth, regime, hot_sectors or []),
        )

    def _classify_regime(self, b: MarketBreadth) -> tuple[str, float]:
        """五阶段市场状态分类"""
        score = 50.0

        # 涨跌比贡献
        if b.up_ratio >= 0.75:
            score += 20
        elif b.up_ratio >= 0.55:
            score += 10
        elif b.up_ratio >= 0.40:
            score += 0
        elif b.up_ratio >= 0.25:
            score -= 10
        else:
            score -= 20

        # 涨停家数贡献
        if b.limit_up_count >= 80:
            score += 15
        elif b.limit_up_count >= 40:
            score += 8
        elif b.limit_up_count >= 15:
            score += 2
        elif b.limit_up_count < 5:
            score -= 12

        # 涨停跌停比
        if b.limit_ratio >= 5:
            score += 10
        elif b.limit_ratio >= 2:
            score += 5
        elif b.limit_ratio < 1:
            score -= 15

        # 成交额变化
        if b.volume_change_pct > 15:
            score += 8
        elif b.volume_change_pct > 5:
            score += 4
        elif b.volume_change_pct < -20:
            score -= 10

        # 状态判定
        score = max(0, min(100, score))
        if score >= 75:
            regime = MARKET_REGIME_CLIMAX
        elif score >= 55:
            regime = MARKET_REGIME_FERMENT
        elif score >= 40:
            regime = MARKET_REGIME_START
        elif score >= 20:
            regime = MARKET_REGIME_DULL
        else:
            regime = MARKET_REGIME_FREEZE

        return regime, score

    def _calc_participation(self, b: MarketBreadth) -> float:
        """市场参与度评分"""
        score = 50.0
        if b.up_ratio > 0.5:
            score += 20
        elif b.up_ratio > 0.35:
            score += 8
        if b.adv_decline_ratio > 2:
            score += 15
        if b.volume_change_pct > 0:
            score += min(15, b.volume_change_pct * 0.5)
        if b.limit_up_count > 20:
            score += 10
        return max(0, min(100, score))

    def _assess_risk(self, b: MarketBreadth, regime: str) -> str:
        """市场风险等级评估"""
        if regime == MARKET_REGIME_FREEZE:
            return "extreme"
        if regime == MARKET_REGIME_CLIMAX:
            return "high"  # 高潮期隐含反转风险
        if b.limit_down_count > 15:
            return "high"
        if b.volume_change_pct < -25:
            return "high"
        if regime == MARKET_REGIME_DULL:
            return "medium"
        return "low"

    def _build_summary(
        self, b: MarketBreadth, regime: str, hot: list[SectorHeat]
    ) -> str:
        parts = [f"市场状态: {regime}"]
        parts.append(f"涨{int(b.up_count)}跌{int(b.down_count)}")
        parts.append(f"涨停{int(b.limit_up_count)}跌停{int(b.limit_down_count)}")
        if hot:
            parts.append(f"热点: {', '.join(s.sector_name for s in hot[:3])}")
        return " | ".join(parts)

    @staticmethod
    def analyze_sector_rotation(
        sector_perf: dict[str, dict],  # {板块名: {pct_chg, limit_up_count, net_inflow}}
        prev_perf: dict[str, dict] | None = None,
    ) -> list[SectorHeat]:
        """
        板块轮动分析。

        从板块行情数据中识别:
        - 当前最热门板块 (涨幅 + 涨停家数 + 资金流入)
        - 板块持续性与轮动信号
        """
        heats = []
        for name, data in sector_perf.items():
            prev = (prev_perf or {}).get(name, {})
            consecutive = data.get("consecutive_days", 0)
            if prev:
                prev_pct = prev.get("pct_chg", 0)
                if data["pct_chg"] > 0 and prev_pct > 0:
                    consecutive = prev.get("consecutive_days", 0) + 1
                elif data["pct_chg"] <= 0:
                    consecutive = 0

            heats.append(SectorHeat(
                sector_name=name,
                pct_chg=data["pct_chg"],
                limit_up_count=data.get("limit_up_count", 0),
                net_inflow_yi=data.get("net_inflow", 0) / 1e8,
                consecutive_days=consecutive,
                leading_stocks=data.get("leading_stocks", []),
            ))

        # 综合评分排序
        for h in heats:
            h._sort_score = (
                h.pct_chg * 0.4 +
                h.limit_up_count * 2 +
                h.net_inflow_yi * 0.1 +
                h.consecutive_days * 3
            )
        heats.sort(key=lambda x: getattr(x, '_sort_score', 0), reverse=True)
        return heats
