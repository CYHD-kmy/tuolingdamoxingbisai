"""
端到端集成测试 — 验证 --demo 模式全链路可跑通。

覆盖:
  - demo.py: 生成完整的 PipelineState
  - main.py: 校验和格式化决策
  - 各阶段数据完整性
"""

import sys
import os

from src.demo import generate_demo_state
from src.output.json_formatter import format_decisions, validate_decisions
from src.output.trace_logger import build_trace
from src.output.report_generator import generate_daily_report
from src.utils.validators import validate_and_clip, get_latest_price, LOT_SIZE


def test_demo_state_structure():
    """demo 模式生成完整 PipelineState"""
    state = generate_demo_state()

    assert state is not None
    assert state.stage == "done"
    assert state.total_capital == 500_000.0

    # 阶段 1: 候选池
    assert len(state.candidates) == 6
    for c in state.candidates:
        assert c.code
        assert c.name
        assert 0 <= c.composite <= 100

    # 阶段 2: 分析
    assert len(state.analyst_reports) == 6
    for code, reports in state.analyst_reports.items():
        assert len(reports) == 4  # 四维分析师各一份
        for r in reports:
            assert r.analyst_type in ("technical", "fundamentals", "fund_flow", "news")
            assert r.signal in ("bullish", "bearish", "neutral")
            assert 0.0 <= r.confidence <= 1.0

    assert len(state.debates) >= 3  # 至少3只有辩论
    for code, debate in state.debates.items():
        assert debate.code == code

    assert len(state.verdicts) == 6
    for code, v in state.verdicts.items():
        assert v.direction in ("buy", "sell", "hold")
        assert 0.0 <= v.confidence <= 1.0

    # 阶段 3: 风控
    assert len(state.position_limits) == 6
    buyable = sum(1 for l in state.position_limits.values() if l.max_shares > 0)
    assert buyable >= 4  # 至少4只可买

    # 阶段 4: 组合决策
    assert state.final_result is not None
    assert len(state.final_result.decisions) >= 1  # demo 至少买入1只
    assert state.final_result.cash_used > 0


def test_demo_decisions_are_valid():
    """demo 决策经过校验后保持有效"""
    state = generate_demo_state()

    validated = validate_decisions(
        state.final_result.decisions,
        state.position_limits,
        state.daily_data,
        cash_available=state.total_capital,
        total_capital=state.total_capital,
    )
    assert len(validated) >= 1

    for d in validated:
        assert d.volume >= LOT_SIZE
        assert d.volume % LOT_SIZE == 0
        assert d.symbol
        assert d.symbol_name


def test_demo_output_formatting():
    """demo 决策可格式化为赛道 JSON"""
    state = generate_demo_state()

    json_output = format_decisions(state.final_result.decisions)
    assert isinstance(json_output, list)
    assert len(json_output) >= 1

    for item in json_output:
        assert "symbol" in item
        assert "symbol_name" in item
        assert "volume" in item
        assert isinstance(item["volume"], int)
        assert item["volume"] % LOT_SIZE == 0


def test_demo_trace_serializable():
    """demo 结果可构建完整 trace JSON"""
    state = generate_demo_state()
    total_elapsed = sum(state.elapsed.values())

    trace = build_trace(state, total_elapsed)
    assert "pipeline_version" in trace
    assert "screening" in trace
    assert "analysis" in trace
    assert "debates" in trace
    assert "verdicts" in trace
    assert "risk" in trace
    assert "decisions" in trace

    # 验证可 JSON 序列化
    import json
    json_str = json.dumps(trace, ensure_ascii=False)
    assert len(json_str) > 1000  # 有实质内容


def test_demo_report_generation():
    """demo 结果可生成 Markdown 日报"""
    state = generate_demo_state()
    report = generate_daily_report(state)

    assert "# 智投未来" in report
    assert "操作摘要" in report
    assert "决策推理链" in report
    assert "持仓快照" in report
    assert "明日关注" in report
    assert len(report) > 500


def test_demo_all_candidates_have_scores():
    """demo 模式下所有候选股都有完整的因子评分 (10因子)"""
    state = generate_demo_state()

    for c in state.candidates:
        assert len(c.scores) == 10
        for factor in ("trend", "momentum", "volume_price", "capital_flow",
                       "northbound", "sentiment", "quality", "risk",
                       "liquidity", "shareholder_conc"):
            assert factor in c.scores
            assert 0 <= c.scores[factor] <= 100


def test_validators_shared_module():
    """共享校验模块基本功能"""
    # 测试 get_latest_price
    daily = {"600519": [type("R", (), {"close": 1680.50})()]}
    assert get_latest_price("600519", daily) == 1680.50
    assert get_latest_price("000001", daily) == 0.0
    assert get_latest_price("000001", {}) == 0.0

    # 测试 validate_and_clip
    decisions = [
        type("D", (), {"symbol": "600519", "symbol_name": "茅台", "volume": 299})(),
    ]
    limits = {
        "600519": type("L", (), {"max_shares": 500, "max_position_pct": 0.20})(),
    }
    result = validate_and_clip(
        decisions, limits, daily,
        cash_available=500_000, total_capital=500_000,
    )
    assert len(result) == 1
    assert result[0].volume == 200  # 299 → 200


if __name__ == "__main__":
    test_demo_state_structure()
    test_demo_decisions_are_valid()
    test_demo_output_formatting()
    test_demo_trace_serializable()
    test_demo_report_generation()
    test_demo_all_candidates_have_scores()
    test_validators_shared_module()
    print("integration: 全部通过 ✓")
