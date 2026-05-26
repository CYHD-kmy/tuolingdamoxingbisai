"""
Agent 层单元测试 — 覆盖风控、数据模型、辩论等关键模块。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.models import (
    DebateResult, DebateRound, ResearchVerdict,
    PositionLimit, FinalDecision, PortfolioResult,
)
from src.agents.base import AnalystReport
from src.agents.managers.risk_manager import RiskManager
from src.utils.validators import extract_json


# ── 风控主管测试 ────────────────────────────

def test_risk_manager_basic():
    """风控: 基本仓位计算"""
    rm = RiskManager(total_capital=500_000.0)
    verdicts = [
        ResearchVerdict(code="600519", name="茅台", direction="buy",
                        confidence=0.8, risk_level="low"),
        ResearchVerdict(code="000858", name="五粮液", direction="buy",
                        confidence=0.7, risk_level="medium"),
    ]
    daily = {
        "600519": [_make_record(50.0, 1.5)],
        "000858": [_make_record(30.0, 2.5)],
    }
    limits = rm.compute_limits(verdicts, daily, {})

    assert len(limits) == 2
    assert limits["600519"].max_shares > 0
    assert limits["000858"].max_shares > 0
    # 低波动 + 高置信度 → 仓位比例更高
    assert limits["600519"].max_position_pct >= limits["000858"].max_position_pct


def test_risk_manager_hold_direction():
    """风控: hold/sell 方向仓位为 0"""
    rm = RiskManager(total_capital=500_000.0)
    verdicts = [
        ResearchVerdict(code="000001", name="测试股", direction="hold", confidence=0.5),
    ]
    daily = {"000001": [_make_record(50.0, 2.0)]}
    limits = rm.compute_limits(verdicts, daily, {})

    assert limits["000001"].max_shares == 0
    assert limits["000001"].max_value == 0


def test_risk_manager_high_volatility():
    """风控: 高波动率限制仓位"""
    rm = RiskManager(total_capital=500_000.0)
    verdicts = [
        ResearchVerdict(code="002594", name="高波股", direction="buy",
                        confidence=0.8, risk_level="high"),
    ]
    # 10条记录, 波动率 > 4% → vol_mult = 0.50
    daily = {"002594": [_make_record(100.0, 5.0) for _ in range(10)]}
    limits = rm.compute_limits(verdicts, daily, {})

    assert limits["002594"].max_position_pct <= 0.20  # 不超过硬上限
    assert limits["002594"].volatility >= 4.0


def test_risk_manager_drawdown_check():
    """风控: 日内熔断检测"""
    rm = RiskManager(total_capital=500_000.0)

    # 未触发
    assert rm.check_drawdown(490_000) is False
    # 触发 (回撤 > 5%)
    assert rm.check_drawdown(470_000) is True  # 回撤 6%
    # 边界: 刚好 5%
    assert rm.check_drawdown(475_000) is True


# ── 数据模型测试 ────────────────────────────

def test_final_decision_to_dict():
    """FinalDecision.to_dict() 输出正确格式"""
    d = FinalDecision(symbol="600519", symbol_name="茅台", volume=200)
    result = d.to_dict()
    assert result == {"symbol": "600519", "symbol_name": "茅台", "volume": 200}


def test_final_decision_to_dict_with_entry_price():
    """entry_price > 0 时 to_dict() 包含入场价"""
    d = FinalDecision(symbol="600519", symbol_name="茅台", volume=200, entry_price=1680.50)
    result = d.to_dict()
    assert result == {"symbol": "600519", "symbol_name": "茅台", "volume": 200, "entry_price": 1680.50}

    # entry_price=0 时不输出该字段
    d2 = FinalDecision(symbol="000858", symbol_name="五粮液", volume=500, entry_price=0.0)
    result2 = d2.to_dict()
    assert "entry_price" not in result2


def test_position_limit_defaults():
    """PositionLimit 默认值"""
    limit = PositionLimit(code="600519", name="茅台",
                          max_position_pct=0.2, max_shares=300, max_value=500_000)
    assert limit.volatility == 0.0
    assert limit.risk_flags == []


def test_debate_result_rounds():
    """DebateResult 辩论记录结构"""
    result = DebateResult(
        code="600519", name="茅台",
        rounds=[
            DebateRound(round_num=1, bull_argument="多头论点", bear_argument="空头反驳"),
            DebateRound(round_num=2, bull_argument="多头论点",
                        bear_argument="空头反驳",
                        bull_rebuttal="多头回应",
                        bear_summary="空头总结"),
        ],
        total_rounds=2,
    )
    assert result.total_rounds == 2
    assert len(result.rounds) == 2
    assert result.rounds[0].bull_argument == "多头论点"


# ── 分析师报告测试 ──────────────────────────

def test_analyst_report_from_json():
    """AnalystReport.from_json 解析正确"""
    data = {
        "analyst_type": "technical",
        "code": "600519",
        "name": "茅台",
        "signal": "bullish",
        "confidence": 0.75,
        "reasoning": "均线多头排列",
        "key_points": ["金叉", "放量"],
        "risks": ["超买"],
    }
    report = AnalystReport.from_json(data)
    assert report.analyst_type == "technical"
    assert report.signal == "bullish"
    assert report.confidence == 0.75
    assert len(report.key_points) == 2


def test_analyst_report_defaults():
    """AnalystReport.from_json 缺失字段使用默认值"""
    report = AnalystReport.from_json({})
    assert report.signal == "neutral"
    assert report.confidence == 0.5
    assert report.reasoning == ""


# ── JSON 提取测试 ──────────────────────────

def test_extract_json_code_block():
    """提取 ```json 代码块"""
    raw = '一些文本\n```json\n{"signal": "bullish"}\n```\n更多文本'
    assert extract_json(raw) == '{"signal": "bullish"}'


def test_extract_json_plain_code_block():
    """提取普通 ``` 代码块"""
    raw = '```\n{"key": "value"}\n```'
    result = extract_json(raw)
    assert "key" in result


def test_extract_json_inline():
    """提取行内 JSON"""
    raw = '结论: {"direction": "buy", "confidence": 0.8} 完成'
    result = extract_json(raw)
    assert "buy" in result


def test_extract_json_array():
    """提取 JSON 数组"""
    raw = '[{"symbol": "600519"}, {"symbol": "000858"}]'
    result = extract_json(raw)
    assert result == raw


def test_extract_json_no_json():
    """无 JSON 时返回原文"""
    raw = "这是一段普通文本，没有 JSON"
    assert extract_json(raw) == raw


# ── 辅助 ────────────────────────────────────

def _make_record(close: float, pct_chg: float = 1.0):
    return type("Record", (), {"close": close, "pct_chg": pct_chg, "date": "2026-05-20"})()


if __name__ == "__main__":
    test_risk_manager_basic()
    test_risk_manager_hold_direction()
    test_risk_manager_high_volatility()
    test_risk_manager_drawdown_check()
    test_final_decision_to_dict()
    test_final_decision_to_dict_with_entry_price()
    test_position_limit_defaults()
    test_debate_result_rounds()
    test_analyst_report_from_json()
    test_analyst_report_defaults()
    test_extract_json_code_block()
    test_extract_json_plain_code_block()
    test_extract_json_inline()
    test_extract_json_array()
    test_extract_json_no_json()
    print("agents: 全部通过")
