"""
多因子打分模型 — 从候选池中选出 Top-N 进入深度分析。

10 因子加权打分:
  趋势(12%) + 动量(10%) + 量价(12%) + 主力资金(15%) + 北向资金(10%)
  + 情绪(8%) + 质量(10%) + 风险(8%) + 流动性(10%) + 筹码集中度(5%)

每只股票得分 0-100，综合加权后排序取 Top-N。
所有计算均为确定性规则，不消耗 LLM Token。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..data.fetchers.akshare_fetcher import MarketSnapshot, StockDaily, FundFlow, FinancialIndicator

logger = logging.getLogger(__name__)

# ── 默认因子权重 ──────────────────────────────

DEFAULT_WEIGHTS = {
    "trend":          0.12,   # 趋势: 均线多头排列
    "momentum":       0.10,   # 动量: 近期涨幅
    "volume_price":   0.12,   # 量价: 放量上涨
    "capital_flow":   0.15,   # 资金: 主力净流入
    "northbound":     0.10,   # 北向: 外资持仓变化 (新增)
    "sentiment":      0.08,   # 情绪: 交易活跃度替代
    "quality":        0.10,   # 质量: PE合理性+ROE/毛利率+财务趋势 (增强)
    "risk":           0.08,   # 风险: 波动率适中
    "liquidity":      0.10,   # 流动性: 日均成交额
    "shareholder_conc": 0.05, # 筹码集中度: 股东人数变化 (新增)
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
        northbound_stocks: dict[str, list[dict]] | None = None,
        financials: dict[str, list[FinancialIndicator]] | None = None,
        shareholders: dict[str, list] | None = None,
    ) -> list[FactorScore]:
        """对候选池中所有股票进行多因子打分"""
        results = []
        available_factors = self._available_factors(fund_flows, northbound_stocks, financials, shareholders)

        for code in codes:
            snapshot = snapshots.get(code)
            if snapshot is None:
                logger.warning("scorer: %s 无快照数据，跳过打分", code)
                continue

            daily = daily_data.get(code, [])
            flow = fund_flows.get(code, [])
            nb_data = (northbound_stocks or {}).get(code, [])
            fin_data = (financials or {}).get(code, [])
            sh_data = (shareholders or {}).get(code, [])

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

            # 北向资金 (新增)
            if "northbound" in available_factors:
                s = self._score_northbound(nb_data)
                scores["northbound"] = s
                weights_used += self._weights["northbound"]

            # 情绪 (交易活跃度替代)
            if "sentiment" in available_factors:
                s = self._score_sentiment_proxy(snapshot, daily)
                scores["sentiment"] = s
                weights_used += self._weights["sentiment"]

            # 质量 (增强版：PE + 财务指标)
            if "quality" in available_factors:
                s = self._score_quality_enhanced(snapshot, fin_data)
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

            # 筹码集中度 (新增)
            if "shareholder_conc" in available_factors:
                s = self._score_shareholder_concentration(sh_data)
                scores["shareholder_conc"] = s
                weights_used += self._weights["shareholder_conc"]

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

    def _available_factors(
        self,
        fund_flows: dict[str, list],
        northbound_stocks: dict[str, list[dict]] | None = None,
        financials: dict[str, list] | None = None,
        shareholders: dict[str, list] | None = None,
    ) -> set[str]:
        """检测哪些因子的数据可用"""
        factors = {"trend", "momentum", "volume_price", "sentiment", "quality", "risk", "liquidity"}
        if any(v for v in fund_flows.values()):
            factors.add("capital_flow")
        else:
            logger.info("资金流向数据不可用，跳过 capital_flow 因子")
        if northbound_stocks and any(v for v in northbound_stocks.values()):
            factors.add("northbound")
        else:
            logger.info("北向资金数据不可用，跳过 northbound 因子")
        if shareholders and any(v for v in shareholders.values()):
            factors.add("shareholder_conc")
        else:
            logger.info("股东人数数据不可用，跳过 shareholder_conc 因子")
        if not (financials and any(v for v in financials.values())):
            logger.info("深度财务数据不可用，quality 因子使用基础 PE 评分")
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
        情绪替代因子: 用交易活跃度和趋势强度替代新闻情感分析。

        MVP 阶段使用换手率+连续阳线作为代理指标，而非真实新闻正负面率。
        第二阶段引入 ChromaDB + 新闻情感 LLM 分析后将替换为真实情感因子。

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
    def _score_quality_enhanced(snapshot: MarketSnapshot, financials: list[FinancialIndicator]) -> float:
        """
        质量因子 (增强): PE 合理性 + ROE/毛利率 + 财务趋势。
        - PE 10-30 → 最优
        - ROE > 15% + 趋势向上 → 加分
        - 毛利率稳定/上升 → 加分
        """
        score = 50.0
        pe = snapshot.pe
        if pe <= 0:
            score -= 20
        elif pe <= 15:
            score += 25
        elif pe <= 30:
            score += 20
        elif pe <= 50:
            score += 5
        elif pe <= 100:
            score -= 10
        else:
            score -= 20

        # 财务指标增强
        if financials:
            latest = financials[-1]
            if latest.roe > 20:
                score += 12
            elif latest.roe > 15:
                score += 8
            elif latest.roe > 10:
                score += 4
            elif latest.roe < 5:
                score -= 8
            if latest.gross_margin > 40:
                score += 8
            elif latest.gross_margin > 25:
                score += 4
            if latest.revenue_yoy > 15:
                score += 5
            elif latest.revenue_yoy < 0:
                score -= 5
            if latest.cf_operating > 0:
                score += 3

            # ROE 趋势 (连续上升加分)
            if len(financials) >= 3:
                last3 = [f.roe for f in financials[-3:]]
                if last3[0] < last3[1] < last3[2]:
                    score += 8
                elif last3[1] < last3[2]:
                    score += 4

        return max(5, min(100, score))

    @staticmethod
    def _score_northbound(nb_data: list[dict]) -> float:
        """
        北向因子: 外资持续增持信号。
        - 近10日连续增持 → 高分
        - 持股占比稳定上升 → 中等
        - 减持 → 低分
        """
        if not nb_data:
            return 50.0
        score = 50.0
        hold_pcts = [d.get("hold_pct", 0) for d in nb_data if d.get("hold_pct")]
        if len(hold_pcts) >= 3:
            recent = hold_pcts[-3:]
            if all(recent[i] < recent[i + 1] for i in range(len(recent) - 1)):
                score += 20  # 连续增持
            elif recent[-1] > recent[0]:
                score += 10  # 整体增持
            elif recent[-1] < recent[0]:
                score -= 15  # 减持
        latest_pct = hold_pcts[-1] if hold_pcts else 0
        if latest_pct > 5:
            score += 10
        elif latest_pct > 2:
            score += 5
        return max(0, min(100, score))

    @staticmethod
    def _score_shareholder_concentration(sh_data: list) -> float:
        """
        筹码集中度因子: 股东人数下降 = 筹码集中。
        - 连续3期下降 → 高分 (筹码集中)
        - 下降趋势 → 中等偏上
        - 上升趋势 → 低分 (筹码分散)
        """
        if not sh_data:
            return 50.0
        score = 50.0
        # 获取最近几期环比变化
        changes = [h.change_pct for h in sh_data if hasattr(h, "change_pct")]
        if len(changes) >= 3:
            recent = changes[-3:]
            if all(c < 0 for c in recent):
                score += 25  # 连续下降，筹码集中
            elif sum(recent) < 0:
                score += 12  # 整体下降
            elif sum(recent) > 0:
                score -= 15  # 分散
        if changes and changes[-1] < -5:
            score += 10  # 单期大幅下降
        return max(0, min(100, score))

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
