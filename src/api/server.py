"""
FastAPI 看板服务 — 读取流水线结果并提供 Web 展示。

启动方式:
    python -m src.api.server
    或: uvicorn src.api.server:app --reload --port 8000

访问: http://localhost:8000
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

app = FastAPI(title="智投未来 看板", version="1.0.0")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def _latest_trace() -> dict[str, Any] | None:
    """加载最新的 trace JSON 文件"""
    if not os.path.isdir(RESULTS_DIR):
        return None
    files = sorted(
        [f for f in os.listdir(RESULTS_DIR) if f.startswith("trace_") and f.endswith(".json")],
        reverse=True,
    )
    if not files:
        return None
    with open(os.path.join(RESULTS_DIR, files[0]), encoding="utf-8") as f:
        return json.load(f)


def _list_traces() -> list[dict[str, str]]:
    """列出所有可用的 trace 文件"""
    if not os.path.isdir(RESULTS_DIR):
        return []
    files = sorted(
        [f for f in os.listdir(RESULTS_DIR) if f.startswith("trace_") and f.endswith(".json")],
        reverse=True,
    )
    result = []
    for f in files:
        date_str = f.replace("trace_", "").replace(".json", "")
        path = os.path.join(RESULTS_DIR, f)
        size_kb = round(os.path.getsize(path) / 1024, 1)
        result.append({"date": date_str, "file": f, "size_kb": size_kb})
    return result


def _load_trace_by_date(date_str: str) -> dict[str, Any] | None:
    """按日期加载 trace"""
    filepath = os.path.join(RESULTS_DIR, f"trace_{date_str}.json")
    if not os.path.isfile(filepath):
        return None
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def _latest_report() -> str | None:
    """加载最新的日报 Markdown"""
    if not os.path.isdir(RESULTS_DIR):
        return None
    files = sorted(
        [f for f in os.listdir(RESULTS_DIR) if f.startswith("report_") and f.endswith(".md")],
        reverse=True,
    )
    if not files:
        return None
    with open(os.path.join(RESULTS_DIR, files[0]), encoding="utf-8") as f:
        return f.read()


# ── 静态文件 ──────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """看板首页"""
    html_path = os.path.join(STATIC_DIR, "dashboard.html")
    if not os.path.isfile(html_path):
        return HTMLResponse("<h1>dashboard.html 未找到</h1>", status_code=404)
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ── API ───────────────────────────────────────

@app.get("/api/status")
async def api_status():
    """系统状态摘要"""
    trace = _latest_trace()
    if not trace:
        return {"ready": False, "message": "暂无数据，请先运行流水线 (python -m src.main)"}

    return {
        "ready": True,
        "date": trace.get("date", ""),
        "total_capital": trace.get("total_capital", 0),
        "pipeline_version": trace.get("pipeline_version", ""),
        "elapsed": trace.get("elapsed", {}),
        "candidates_count": trace.get("screening", {}).get("total_candidates", 0),
        "decisions_count": len(trace.get("decisions", [])),
        "errors_count": len(trace.get("errors", [])),
        "errors": trace.get("errors", [])[:5],
    }


@app.get("/api/decisions")
async def api_decisions(date: str | None = None):
    """最终决策 (赛道 JSON 格式)"""
    trace = _load_trace_by_date(date) if date else _latest_trace()
    if not trace:
        return {"decisions": [], "date": None}

    decisions = trace.get("decisions", [])
    portfolio = trace.get("portfolio", {})
    return {
        "date": trace.get("date", ""),
        "decisions": decisions,
        "cash_used": portfolio.get("cash_used", 0),
        "cash_remaining": portfolio.get("cash_remaining", 0),
        "total_positions": portfolio.get("total_positions", 0),
        "total_capital": trace.get("total_capital", 0),
    }


@app.get("/api/candidates")
async def api_candidates(date: str | None = None):
    """候选池数据"""
    trace = _load_trace_by_date(date) if date else _latest_trace()
    if not trace:
        return {"candidates": [], "date": None}

    screening = trace.get("screening", {})
    return {
        "date": trace.get("date", ""),
        "total": screening.get("total_candidates", 0),
        "candidates": screening.get("candidates", []),
    }


@app.get("/api/analysis")
async def api_analysis(code: str | None = None, date: str | None = None):
    """分析详情: 四维报告 + 辩论 + 研判"""
    trace = _load_trace_by_date(date) if date else _latest_trace()
    if not trace:
        return {"analysis": {}, "debates": {}, "verdicts": {}}

    analysis = trace.get("analysis", {})
    debates = trace.get("debates", {})
    verdicts = trace.get("verdicts", {})

    if code:
        return {
            "code": code,
            "analysis": analysis.get(code, {}),
            "debate": debates.get(code, {}),
            "verdict": verdicts.get(code, {}),
        }

    return {"analysis": analysis, "debates": debates, "verdicts": verdicts}


@app.get("/api/risk")
async def api_risk(date: str | None = None):
    """风控约束"""
    trace = _load_trace_by_date(date) if date else _latest_trace()
    if not trace:
        return {"limits": {}}

    return {"limits": trace.get("risk", {})}


@app.get("/api/report")
async def api_report(date: str | None = None):
    """返回 Markdown 日报原文"""
    if date:
        path = os.path.join(RESULTS_DIR, f"report_{date}.md")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                return {"date": date, "report": f.read()}
        return {"date": date, "report": None}

    report = _latest_report()
    trace = _latest_trace()
    return {
        "date": trace.get("date", "") if trace else "",
        "report": report,
    }


@app.get("/api/history")
async def api_history():
    """历史记录列表"""
    return {"traces": _list_traces()}


@app.get("/api/trace")
async def api_trace(date: str | None = None):
    """完整 trace JSON"""
    trace = _load_trace_by_date(date) if date else _latest_trace()
    if not trace:
        raise HTTPException(status_code=404, detail="无数据")
    return trace


# ── 启动入口 ──────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
