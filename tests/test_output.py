"""
测试输出模块 — JSON 格式化和校验。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.models import FinalDecision, PositionLimit
from src.output.json_formatter import format_decisions, validate_decisions


def test_format_decisions():
    """format_decisions 输出正确 JSON 格式"""
    decisions = [
        FinalDecision(symbol="600519", symbol_name="贵州茅台", volume=200),
        FinalDecision(symbol="000858", symbol_name="五粮液", volume=500),
    ]
    result = format_decisions(decisions)
    assert len(result) == 2
    assert result[0]["symbol"] == "600519"
    assert result[0]["symbol_name"] == "贵州茅台"
    assert result[0]["volume"] == 200


def test_format_empty():
    """空列表返回空列表"""
    assert format_decisions([]) == []


def test_validate_round_down():
    """volume 自动取整到 100 的倍数"""
    decisions = [
        FinalDecision(symbol="600519", symbol_name="贵州茅台", volume=299),
    ]
    limits: dict[str, PositionLimit] = {}
    daily_data = {
        "600519": [_make_record(100.0)],
    }
    result = validate_decisions(
        decisions, limits, daily_data,
        cash_available=500_000, total_capital=500_000,
    )
    assert len(result) == 1
    assert result[0].volume == 200  # 299 → 200


def test_validate_suspended_skip():
    """停牌股被跳过"""
    decisions = [
        FinalDecision(symbol="000001", symbol_name="停牌股", volume=100),
    ]
    limits: dict[str, PositionLimit] = {}
    daily_data = {
        "000001": [_make_record(50.0)],
    }
    result = validate_decisions(
        decisions, limits, daily_data,
        cash_available=500_000,
        suspended_codes={"000001"},
    )
    assert len(result) == 0


def test_validate_budget_exceeded():
    """预算不足时截断"""
    decisions = [
        FinalDecision(symbol="600519", symbol_name="贵州茅台", volume=10000),
    ]
    limits: dict[str, PositionLimit] = {}
    daily_data = {
        "600519": [_make_record(100.0)],
    }
    result = validate_decisions(
        decisions, limits, daily_data,
        cash_available=500_000, total_capital=500_000,
    )
    # 10000 * 100 = 1,000,000 > 500,000，应该被裁剪
    assert len(result) >= 1, "应有至少1笔决策被裁剪后保留"
    for d in result:
        assert d.volume * 100 <= 500_000 * 0.9  # 留10%现金


def test_validate_empty():
    """空决策列表返回空"""
    result = validate_decisions(
        [], {}, {},
        cash_available=500_000,
    )
    assert result == []


def _make_record(close: float):
    return type("Record", (), {"close": close})()


if __name__ == "__main__":
    test_format_decisions()
    test_format_empty()
    test_validate_round_down()
    test_validate_suspended_skip()
    test_validate_budget_exceeded()
    test_validate_empty()
    print("output: 全部通过")
