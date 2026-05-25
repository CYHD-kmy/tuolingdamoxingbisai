"""
多因子打分模型 — 从候选池中选出 Top-N 进入深度分析。

8 因子加权打分:
  趋势(15%) + 动量(10%) + 量价(15%) + 资金(20%) + 情绪(10%)
  + 质量(5%) + 风险(10%) + 流动性(15%)

每只股票得分 0-100，综合加权后排序取 Top-N。
所有计算均为确定性规则，不消耗 LLM Token。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..data.fetchers.akshare_fetcher import MarketSnapshot, StockDaily, FundFlow

logger = logging.getLogger(__name__)

# ── 默认因子权重 ──────────────────────────────

DEFAULT_WEIGHTS = {
    "trend":      0.15,   # 趋势: 均线多头排列
    "momentum":   0.10,   # 动量: 近期涨幅
    "volume_price": 0.15, # 量价: 放量上涨
    "capital_flow": 0.20, # 资金: 主力净流入
    "sentiment":  0.10,   # 情绪: 新闻正面率 (MVP 阶段用替代)
    "quality":    0.05,   # 质量: PE/ROE 过滤
    "risk":       0.10,   # 风险: 波动率适中
    "liquidity":  0.15,   # 流动性: 日均成交额
}


# ── 输出模型 ──────────────────────────────────

@dataclass
class FactorScore:
    """单只股票的多因子得分"""
    code: str
    name: str
    scores: dict[str, float] = field(default_factory=dict)
    composite: float = 0.0


# ── 打分器 ────────────────────────────────────

class ScreeningScorer:
    """
    多因子加权打分器。

    使用方式:
        scorer = ScreeningScorer()
        scored = scorer.score_all(
            codes=filtered_codes,
            snapshots=market_data,       # {code: MarketSnapshot}
            daily_data=ohlcv_data,       # {code: [StockDaily]}
            fund_flows=flow_data,        # {code: [FundFlow]}
        )
        top20 = scorer.top_n(scored, n=20)
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self._weights = weights or dict(DEFAULT_WEIGHTS)

    # ── 主入口 ────────────────────────────────

    def score_all(
        self,
        codes: list[str],
        snapshots: dict[str, MarketSnapshot],
        daily_data: dict[str, list[StockDaily]],
        fund_flows: dict[str, list[FundFlow]],
    ) -> list[FactorScore]:
        """对候选池中所有股票进行多因子打分"""
        results = []
        available_factors = self._available_factors(fund_flows)

        for code in codes:
            snapshot = snapshots.get(code)
            if snapshot is None:
                continue

            daily = daily_data.get(code, [])
            flow = fund_flows.get(code, [])

            scores = {}
            weights_used = 0.0

            # 趋势
            if "trend" in available_factors:
                s = self._score_trend(daily)
                scores["trend"] = s
                weights_used += self._weights["trend"]

            # 动量
            if "momentum" in available_factors:
                s = self._score_momentum(daily, snapshot)
                scores["momentum"] = s
                weights_used += self._weights["momentum"]

            # 量价
            if "volume_price" in available_factors:
                s = self._score_volume_price(snapshot, daily)
                scores["volume_price"] = s
                weights_used += self._weights["volume_price"]

            # 资金 (如果数据不可用则跳过，权重重新分配)
            if "capital_flow" in available_factors and flow:
                s = self._score_capital_flow(flow)
                scores["capital_flow"] = s
                weights_used += self._weights["capital_flow"]

            # 情绪 (MVP阶段用交易活跃度替代)
            if "sentiment" in available_factors:
                s = self._score_sentiment_proxy(snapshot, daily)
                scores["sentiment"] = s
                weights_used += self._weights["sentiment"]

            # 质量
            if "quality" in available_factors:
                s = self._score_quality(snapshot)
                scores["quality"] = s
                weights_used += self._weights["quality"]

            # 风险
            if "risk" in available_factors:
                s = self._score_risk(daily)
                scores["risk"] = s
                weights_used += self._weights["risk"]

            # 流动性
            if "liquidity" in available_factors:
                s = self._score_liquidity(snapshot)
                scores["liquidity"] = s
                weights_used += self._weights["liquidity"]

            # 归一化权重
            composite = self._weighted_sum(scores, weights_used)
            results.append(FactorScore(code=code, name=snapshot.name, scores=scores, composite=round(composite, 1)))

        logger.info("score_all: %d 只股票打分完成", len(results))
        return results

    def top_n(self, scored: list[FactorScore], n: int = 20) -> list[FactorScore]:
        """按综合得分降序取 Top-N"""
        scored.sort(key=lambda x: x.composite, reverse=True)
        return scored[:n]

    # ── 因子可用性 ─────────────────────────────

    def _available_factors(self, fund_flows: dict[str, list]) -> set[str]:
        """检测哪些因子的数据可用"""
        factors = {"trend", "momentum", "volume_price", "sentiment", "quality", "risk", "liquidity"}
        # 检查是否有资金流向数据
        if any(v for v in fund_flows.values()):
            factors.add("capital_flow")
        else:
            logger.info("资金流向数据不可用，跳过 capital_flow 因子")
        return factors

    def _weighted_sum(self, scores: dict[str, float], weights_used: float) -> float:
        """加权求和，归一化到 0-100"""
        if weights_used == 0:
            return 50.0
        total = sum(
            scores[k] * self._weights.get(k, 0)
            for k in scores
        )
        return total / weights_used  # 归一化，补偿不可用因子

    # ── 各因子打分 (0-100) ────────────────────

    @staticmethod
    def _score_trend(daily: list[StockDaily]) -> float:
        """
        趋势因子: 均线多头排列程度。
        - MA5 > MA10 > MA20 → 满分
        - 部分满足 → 按满足条数给分
        """
        if len(daily) < 20:
            return 50.0
        latest = daily[-1]
        score = 50.0
        if latest.ma5 > latest.ma10:
            score += 15
        if latest.ma10 > latest.ma20:
            score += 15
        if latest.close > latest.ma5:
            score += 10
        if latest.close > latest.ma20:
            score += 10
        return min(score, 100.0)

    @staticmethod
    def _score_momentum(
        daily: list[StockDaily],
        snapshot: MarketSnapshot,
    ) -> float:
        """
        动量因子: 近期涨幅。
        - 涨幅 2%-5% → 最优 (70-90分)
        - 涨幅 0-2% → 中等 (50-70分)
        - 涨幅 >8% → 追高风险 (30分)
        - 下跌 → 低分
        """
        pct = snapshot.pct_chg
        if len(daily) >= 5:
            # 用5日累计平滑
            pct5 = (daily[-1].close / daily[-5].close - 1) * 100
            pct = pct * 0.4 + pct5 * 0.6

        if 2 <= pct <= 5:
            return 70 + (pct - 2) * 6.7
        elif 0 <= pct < 2:
            return 50 + pct * 10
        elif -2 <= pct < 0:
            return 40 + pct * 5
        elif pct > 5:
            return max(20, 80 - (pct - 5) * 8)
        else:
            return max(10, 30 + pct * 5)

    @staticmethod
    def _score_volume_price(
        snapshot: MarketSnapshot,
        daily: list[StockDaily],
    ) -> float:
        """
        量价因子: 放量上涨。
        - 量比 > 1.5 且涨幅 > 0 → 高分
        - 缩量上涨 → 中等
        - 放量下跌 → 低分
        """
        vol_ratio = snapshot.volume_ratio if snapshot.volume_ratio > 0 else 1.0
        pct = snapshot.pct_chg

        score = 50.0
        # 量比打分
        if vol_ratio >= 2.0:
            score += 20
        elif vol_ratio >= 1.5:
            score += 12
        elif vol_ratio >= 1.0:
            score += 5
        else:
            score -= 10

        # 涨幅调整
        if pct > 0:
            score += min(20, pct * 5)
        else:
            score += max(-20, pct * 3)

        return max(0, min(100, score))

    @staticmethod
    def _score_capital_flow(flows: list[FundFlow]) -> float:
        """
        资金因子: 主力资金净流入。
        - 连续正向流入 → 高分
        - 忽进忽出 → 中等
        - 持续流出 → 低分
        """
        if not flows:
            return 50.0

        score = 50.0
        recent = flows[-3:] if len(flows) >= 3 else flows

        for f in recent:
            if f.main_net_inflow > 0:
                score += 8
            else:
                score -= 6
            # 占比加分
            if f.main_pct > 5:
                score += 5
            elif f.main_pct > 2:
                score += 3

        return max(0, min(100, score))

    @staticmethod
    def _score_sentiment_proxy(
        snapshot: MarketSnapshot,
        daily: list[StockDaily],
    ) -> float:
        """
        情绪替代因子 (MVP阶段): 用交易活跃度和趋势强度替代。
        - 换手率适中 (2%-8%) + 量比温和 → 高分
        - 换手率过高/过低 → 低分
        """
        turnover = snapshot.turnover
        score = 50.0

        if 2 <= turnover <= 8:
            score += 20
        elif 1 <= turnover < 2 or 8 < turnover <= 15:
            score += 10
        elif turnover > 15:
            score -= 20
        else:
            score -= 10

        # 连续阳线加分
        if daily:
            consecutive = 0
            for d in reversed(daily):
                if d.pct_chg > 0:
                    consecutive += 1
                else:
                    break
            score += min(15, consecutive * 5)

        return max(0, min(100, score))

    @staticmethod
    def _score_quality(snapshot: MarketSnapshot) -> float:
        """
        质量因子: PE 合理性检测。
        - PE 10-40 → 合理区间
        - PE < 0 (亏损) → 扣分
        - PE > 100 → 高估值扣分
        """
        pe = snapshot.pe
        if pe <= 0:
            return 30.0
        elif pe <= 15:
            return 80.0
        elif pe <= 30:
            return 75.0
        elif pe <= 50:
            return 60.0
        elif pe <= 100:
            return 45.0
        else:
            return 25.0

    @staticmethod
    def _score_risk(daily: list[StockDaily]) -> float:
        """
        风险因子: 近期波动率适中为佳。
        - 标准差 1%-3% → 最优 (稳健但有波动)
        - 标准差 < 1% → 不活跃 (扣分)
        - 标准差 > 5% → 异常波动 (扣分)
        """
        if len(daily) < 10:
            return 50.0

        pcts = [abs(d.pct_chg) for d in daily[-10:]]
        avg_vol = sum(pcts) / len(pcts)

        if 1 <= avg_vol <= 3:
            return 80.0
        elif 0.5 <= avg_vol < 1:
            return 60.0
        elif 3 < avg_vol <= 5:
            return 50.0
        elif avg_vol < 0.5:
            return 35.0
        else:
            return max(10, 50 - (avg_vol - 5) * 8)

    @staticmethod
    def _score_liquidity(snapshot: MarketSnapshot) -> float:
        """
        流动性因子: 成交额越大越好 (A股特征)。
        - > 5亿 → 满分 (大盘股流动性)
        - 5000万-5亿 → 线性递增
        """
        amount = snapshot.amount  # 当日成交额 (元)
        amount_yi = amount / 1e8

        if amount_yi >= 10:
            return 95.0
        elif amount_yi >= 5:
            return 85 + (amount_yi - 5) * 2
        elif amount_yi >= 1:
            return 55 + (amount_yi - 1) * 7.5
        elif amount_yi >= 0.5:
            return 30 + (amount_yi - 0.5) * 50
        else:
            return max(5, amount_yi * 40)
