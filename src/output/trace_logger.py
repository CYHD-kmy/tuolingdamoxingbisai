"""
推理追踪日志 — 保存完整的决策轨迹 JSON 文件。

每笔决策保留从数据→信号→辩论→决策的完整审计链路。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from ..graph.state import PipelineState

logger = logging.getLogger(__name__)


def build_trace(state: PipelineState, total_elapsed: float) -> dict[str, Any]:
    """构建完整追踪数据结构"""

    date_str = datetime.now().strftime("%Y%m%d")

    # 候选池
    candidates = [
        {
            "code": c.code, "name": c.name, "score": c.composite,
            **{k: c.scores.get(k, 0)
               for k in ["trend", "momentum", "volume_price",
                         "capital_flow", "sentiment", "quality", "risk", "liquidity"]}
        }
        for c in state.candidates
    ]

    # 研判结论
    verdicts = {
        code: {
            "name": v.name,
            "direction": v.direction,
            "confidence": v.confidence,
            "target_price": v.target_price,
            "risk_level": v.risk_level,
            "core_reasoning": v.core_reasoning,
            "key_risks": v.key_risks,
        }
        for code, v in state.verdicts.items()
    }

    # 分析师报告摘要
    analysis: dict[str, dict[str, Any]] = {}
    for code, reports in state.analyst_reports.items():
        analysis[code] = {}
        for r in reports:
            analysis[code][r.analyst_type] = {
                "signal": r.signal,
                "confidence": r.confidence,
                "reasoning": r.reasoning[:300],
                "key_points": r.key_points,
                "risks": r.risks,
            }

    # 辩论记录
    debates = {
        code: {
            "name": d.name,
            "total_rounds": d.total_rounds,
            "rounds": [
                {
                    "round": r.round_num,
                    "bull_argument": r.bull_argument[:500],
                    "bear_argument": r.bear_argument[:500],
                    "bull_rebuttal": r.bull_rebuttal[:500],
                    "bear_summary": r.bear_summary[:500],
                }
                for r in d.rounds
            ],
        }
        for code, d in state.debates.items()
    }

    # 风控约束
    risk_limits = {
        code: {
            "name": l.name,
            "max_position_pct": l.max_position_pct,
            "max_shares": l.max_shares,
            "max_value": l.max_value,
            "volatility": l.volatility,
            "risk_flags": l.risk_flags,
        }
        for code, l in state.position_limits.items()
    }

    # 最终决策
    decisions = [d.to_dict() for d in (state.final_result.decisions if state.final_result else [])]

    return {
        "pipeline_version": "1.1.0",
        "date": date_str,
        "timestamp": datetime.now().isoformat(),
        "total_capital": state.total_capital,
        "elapsed": {
            "total": round(total_elapsed, 2),
            "stages": state.elapsed,
        },
        "screening": {
            "total_candidates": len(candidates),
            "candidates": candidates,
        },
        "analysis": analysis,
        "debates": debates,
        "verdicts": verdicts,
        "risk": risk_limits,
        "decisions": decisions,
        "data_quality": state.data_quality,
        "errors": state.errors,
        "portfolio": {
            "cash_used": state.final_result.cash_used if state.final_result else 0,
            "cash_remaining": state.final_result.cash_remaining if state.final_result else state.total_capital,
            "total_positions": state.final_result.total_positions if state.final_result else 0,
        } if state.final_result else {},
    }


def save_trace(
    state: PipelineState,
    total_elapsed: float,
    results_dir: str = "./results",
) -> str:
    """
    保存完整追踪 JSON 到文件。

    返回: 文件路径
    """
    trace = build_trace(state, total_elapsed)

    os.makedirs(results_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    filename = f"trace_{date_str}.json"
    filepath = os.path.join(results_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)

    logger.info("追踪日志已保存: %s (%.1f KB)", filepath, os.path.getsize(filepath) / 1024)
    return filepath


def load_trace(date_str: str, results_dir: str = "./results") -> dict[str, Any] | None:
    """加载指定日期的追踪文件"""
    filepath = os.path.join(results_dir, f"trace_{date_str}.json")
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
