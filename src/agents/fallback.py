"""
LLM 降级策略 — LLM 不可用时用确定性规则生成报告和决策。

触发场景:
- LLM API 超时/网络错误
- API 返回非 200
- 重试耗尽

降级原则:
- 分析师: 用预加载的技术指标数据做规则引擎判断
- 研究主管: 简单多数投票聚合分析师信号
- 组合主管: 按置信度排序，在风控约束内买入 Top-N
"""

from __future__ import annotations

import logging
from typing import Any

from .base import AnalystReport
from .models import (
    FinalDecision, PortfolioResult, PositionLimit, ResearchVerdict,
)

logger = logging.getLogger(__name__)


# ── 分析师降级 ──────────────────────────────────

def fallback_technical_report(code: str, name: str, data: Any) -> AnalystReport:
    """用 MA 排列 + MACD + RSI 做确定性技术面判断"""
    daily = data.get_daily_data(code, days=30) if data else []
    if not daily:
        return AnalystReport(
            analyst_type="technical", code=code, name=name,
            signal="neutral", confidence=0.30,
            reasoning="[降级] 无数据可用，无法进行技术分析",
            key_points=["数据缺失"],
        )

    latest = daily[-1]
    bull_score = 0
    points = []
    risks = []

    # MA 多头排列
    if latest.ma5 > latest.ma10 > latest.ma20 > 0:
        bull_score += 30
        points.append("均线多头排列 (MA5>MA10>MA20)")
    elif latest.ma5 < latest.ma10 < latest.ma20 and latest.ma20 > 0:
        bull_score -= 25
        risks.append("均线空头排列")
    else:
        points.append("均线交织，方向不明确")

    # MACD
    if latest.macd_dif > latest.macd_dea and latest.macd_bar > 0:
        bull_score += 20
        points.append("MACD 金叉状态，红柱放大")
    elif latest.macd_dif < latest.macd_dea:
        bull_score -= 15
        risks.append("MACD 死叉或走弱")

    # RSI
    if 30 <= latest.rsi_14 <= 70:
        if 40 <= latest.rsi_14 <= 60:
            bull_score += 10
        else:
            bull_score += 5
    elif latest.rsi_14 > 80:
        bull_score -= 15
        risks.append(f"RSI(14)={latest.rsi_14:.1f} 超买")
    elif latest.rsi_14 < 20:
        bull_score += 15
        points.append(f"RSI(14)={latest.rsi_14:.1f} 超卖，反弹概率大")

    # 近期趋势
    if len(daily) >= 5:
        pct5 = (daily[-1].close / daily[-5].close - 1) * 100
        if pct5 > 3:
            bull_score += 15
            points.append(f"近5日涨 {pct5:.1f}%")
        elif pct5 < -5:
            bull_score -= 15
            risks.append(f"近5日跌 {abs(pct5):.1f}%")

    signal, confidence = _score_to_signal(bull_score, 50)
    return AnalystReport(
        analyst_type="technical", code=code, name=name,
        signal=signal, confidence=confidence,
        reasoning=f"[降级-规则引擎] 评分{bull_score:+d}。{'; '.join(points[:3])}",
        key_points=points[:4],
        risks=risks[:3],
    )


def fallback_fundamentals_report(code: str, name: str, data: Any) -> AnalystReport:
    """用 PE 范围 + 市值做确定性基本面判断"""
    quote = data.get_realtime_quote(code) if data else None
    if quote is None:
        return AnalystReport(
            analyst_type="fundamentals", code=code, name=name,
            signal="neutral", confidence=0.30,
            reasoning="[降级] 无估值数据可用",
        )

    pe = quote.pe
    points = []
    risks = []
    bull_score = 0

    if pe <= 0:
        bull_score -= 20
        risks.append(f"PE={pe:.1f} 亏损状态")
    elif pe <= 15:
        bull_score += 25
        points.append(f"PE={pe:.1f} 低估值区间")
    elif pe <= 30:
        bull_score += 15
        points.append(f"PE={pe:.1f} 合理估值")
    elif pe <= 50:
        bull_score += 5
        points.append(f"PE={pe:.1f} 略偏高")
    else:
        bull_score -= 10
        risks.append(f"PE={pe:.1f} 估值偏高")

    # 市值
    if quote.total_mv > 500:
        bull_score += 10
        points.append(f"大盘蓝筹 (市值{quote.total_mv:.0f}亿)")
    elif quote.total_mv < 50:
        bull_score -= 5
        risks.append("小盘股，流动性风险")

    signal, confidence = _score_to_signal(bull_score, 30)
    return AnalystReport(
        analyst_type="fundamentals", code=code, name=name,
        signal=signal, confidence=confidence,
        reasoning=f"[降级-规则引擎] PE={pe:.1f}，市值{quote.total_mv:.0f}亿",
        key_points=points[:3],
        risks=risks[:3],
    )


def fallback_fund_flow_report(code: str, name: str, data: Any) -> AnalystReport:
    """用资金净流入方向做确定性判断"""
    flows = data.get_fund_flow(code, days=5) if data else []
    if not flows:
        return AnalystReport(
            analyst_type="fund_flow", code=code, name=name,
            signal="neutral", confidence=0.30,
            reasoning="[降级] 无资金流向数据",
        )

    total_inflow = sum(f.main_net_inflow for f in flows)
    positive_days = sum(1 for f in flows if f.main_net_inflow > 0)
    points = []
    risks = []

    bull_score = 0
    if total_inflow > 5000:
        bull_score += 30
        points.append(f"近5日主力净流入 {total_inflow:.0f}万")
    elif total_inflow > 1000:
        bull_score += 15
        points.append(f"近5日主力小幅净流入 {total_inflow:.0f}万")
    elif total_inflow < -5000:
        bull_score -= 25
        risks.append(f"近5日主力净流出 {abs(total_inflow):.0f}万")
    elif total_inflow < -1000:
        bull_score -= 10
        risks.append(f"近5日主力小幅净流出 {abs(total_inflow):.0f}万")

    if positive_days >= 4:
        bull_score += 15
        points.append(f"主力持续流入 ({positive_days}/5天)")
    elif positive_days <= 1:
        bull_score -= 10
        risks.append(f"主力多数日期流出 ({5-positive_days}/5天)")

    signal, confidence = _score_to_signal(bull_score, 40)
    return AnalystReport(
        analyst_type="fund_flow", code=code, name=name,
        signal=signal, confidence=confidence,
        reasoning=f"[降级-规则引擎] 近5日主力{'流入' if total_inflow > 0 else '流出'} {abs(total_inflow):.0f}万",
        key_points=points[:3],
        risks=risks[:3],
    )


def fallback_news_report(code: str, name: str, data: Any) -> AnalystReport:
    """消息面降级: 无 LLM 时无法分析新闻情感，返回中性"""
    return AnalystReport(
        analyst_type="news", code=code, name=name,
        signal="neutral", confidence=0.35,
        reasoning="[降级] LLM 不可用，无法分析新闻情感，默认中性",
        key_points=["消息面分析需 LLM 支持"],
        risks=["缺少消息面信号"],
    )


# ── 研究主管降级 ────────────────────────────────

def fallback_verdict(code: str, name: str, reports: list[AnalystReport], price: float) -> ResearchVerdict:
    """简单多数投票聚合分析师信号"""
    if not reports:
        return ResearchVerdict(code=code, name=name, direction="hold", confidence=0.0,
                               core_reasoning="[降级] 无分析师报告")

    signal_votes = {"bullish": 0, "bearish": 0, "neutral": 0}
    confidences = []
    for r in reports:
        signal_votes[r.signal] = signal_votes.get(r.signal, 0) + 1
        confidences.append(r.confidence)

    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.3

    if signal_votes["bullish"] >= 3:
        direction = "buy"
        confidence = min(avg_confidence + 0.05, 0.85)
        reasoning = f"[降级-投票] 四维分析 {signal_votes['bullish']}/4 看多"
    elif signal_votes["bullish"] >= 2 and signal_votes["bearish"] == 0:
        direction = "buy"
        confidence = min(avg_confidence, 0.70)
        reasoning = f"[降级-投票] {signal_votes['bullish']}/4 看多，0 看空"
    elif signal_votes["bearish"] >= 3:
        direction = "sell"
        confidence = min(avg_confidence, 0.75)
        reasoning = f"[降级-投票] {signal_votes['bearish']}/4 看空"
    elif signal_votes["bearish"] >= 2:
        direction = "hold"
        confidence = 0.45
        reasoning = f"[降级-投票] 信号分歧 ({signal_votes['bullish']}多/{signal_votes['bearish']}空)"
    else:
        direction = "hold"
        confidence = 0.40
        reasoning = "[降级-投票] 信号不明确，建议观望"

    risk_level = "medium"
    if confidence >= 0.70:
        risk_level = "low"
    elif confidence < 0.45:
        risk_level = "high"

    return ResearchVerdict(
        code=code, name=name,
        direction=direction, confidence=round(confidence, 2),
        target_price=price * (1.05 if direction == "buy" else 0.95),
        risk_level=risk_level,
        core_reasoning=reasoning,
        key_risks=["[降级模式] LLM 不可用，信号基于确定性规则"],
    )


# ── 组合主管降级 ────────────────────────────────

def fallback_portfolio(
    verdicts: list[ResearchVerdict],
    limits: dict[str, PositionLimit],
    daily_data: dict[str, list],
    cash_available: float,
    total_capital: float = 500_000.0,
) -> PortfolioResult:
    """简单规则: 买入置信度最高的 buy 标的，在风控约束内分配"""
    from ..utils.validators import get_latest_price

    buy_candidates = [
        v for v in verdicts
        if v.direction == "buy" and v.code in limits and limits[v.code].max_shares > 0
    ]
    if not buy_candidates:
        return PortfolioResult(decisions=[])

    buy_candidates.sort(key=lambda x: x.confidence, reverse=True)
    decisions = []
    remaining = cash_available * 0.90  # 保留 10% 现金
    min_cash = total_capital * 0.10

    for v in buy_candidates[:5]:  # 最多 5 只
        if remaining <= min_cash:
            break

        limit = limits[v.code]
        price = get_latest_price(v.code, daily_data)
        if price <= 0:
            continue

        # 保守配置: 只使用风控上限的一半
        max_shares = min(limit.max_shares, int(remaining * 0.3 / price / 100) * 100)
        if max_shares < 100:
            continue

        cost = max_shares * price
        remaining -= cost

        decisions.append(FinalDecision(
            symbol=v.code, symbol_name=v.name,
            volume=max_shares, entry_price=price,
        ))

    cash_used = sum(d.volume * get_latest_price(d.symbol, daily_data) for d in decisions)
    return PortfolioResult(
        decisions=decisions,
        cash_used=round(cash_used, 2),
        cash_remaining=round(cash_available - cash_used, 2),
        total_positions=len(decisions),
        risk_summary="[降级模式] 使用确定性规则构建组合，保守配置",
    )


# ── 辅助 ──────────────────────────────────────

def _score_to_signal(score: float, threshold: float = 40) -> tuple[str, float]:
    """将评分转换为 signal 和 confidence"""
    if score >= threshold:
        confidence = min(0.85, 0.50 + score / 200)
        return ("bullish", round(confidence, 2))
    elif score <= -threshold:
        confidence = min(0.80, 0.50 + abs(score) / 200)
        return ("bearish", round(confidence, 2))
    else:
        return ("neutral", round(0.40 + abs(score) / 200, 2))
