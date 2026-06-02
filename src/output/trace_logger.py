"""
推理追踪日志 — 保存完整的决策轨迹 JSON 文件。

每笔决策保留从数据→信号→辩论→决策的完整审计链路。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

from ..graph.state import PipelineState

logger = logging.getLogger(__name__)


def build_trace(state: PipelineState, total_elapsed: float) -> dict[str, Any]:
    """构建完整追踪数据结构"""

    date_str = datetime.now().strftime("%Y%m%d")

    # 候选池
    _candidates = getattr(state, "candidates", [])
    candidates = [
        {
            "code": c.code, "name": c.name, "score": c.composite,
            **{k: c.scores.get(k, 0)
               for k in ["trend", "momentum", "volume_price",
                         "capital_flow", "northbound", "sentiment", "quality",
                         "risk", "liquidity", "shareholder_conc"]}
        }
        for c in _candidates
    ]

    # 研判结论
    _verdicts = getattr(state, "verdicts", {})
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
        for code, v in _verdicts.items()
    }

    # 分析师报告摘要
    _analyst_reports = getattr(state, "analyst_reports", {})
    analysis: dict[str, dict[str, Any]] = {}
    for code, reports in _analyst_reports.items():
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
        for code, d in getattr(state, "debates", {}).items()
    }

    # 风控约束
    _position_limits = getattr(state, "position_limits", {})
    risk_limits = {
        code: {
            "name": l.name,
            "max_position_pct": l.max_position_pct,
            "max_shares": l.max_shares,
            "max_value": l.max_value,
            "volatility": l.volatility,
            "risk_flags": l.risk_flags,
        }
        for code, l in _position_limits.items()
    }

    # 最终决策
    _final = getattr(state, "final_result", None)
    decisions = [
        {**d.to_dict(), "entry_price": d.entry_price, "asset_type": d.asset_type}
        for d in (_final.decisions if _final else [])
    ]

    return {
        "pipeline_version": "1.1.0",
        "date": date_str,
        "timestamp": datetime.now().isoformat(),
        "total_capital": getattr(state, "total_capital", 0),
        "total_equity": 0.0,   # 由 main.py 回写覆盖为实际权益
        "total_return": 0.0,   # 由 main.py 回写覆盖为实际收益率
        "elapsed": {
            "total": round(total_elapsed, 2),
            "stages": getattr(state, "elapsed", {}),
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
        "data_quality": getattr(state, "data_quality", {}),
        "errors": getattr(state, "errors", []),
        "portfolio": {
            "cash_used": _final.cash_used if _final else 0,
            "cash_remaining": _final.cash_remaining if _final else getattr(state, "total_capital", 0),
            "total_positions": _final.total_positions if _final else 0,
        } if _final else {},
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
