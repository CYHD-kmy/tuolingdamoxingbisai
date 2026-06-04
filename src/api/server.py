"""
FastAPI 看板服务 — 多页面常驻网站。

启动方式:
    python -m src.api.server
    或: uvicorn src.api.server:app --host 0.0.0.0 --port 8000
    或: ./manage.sh start

访问: http://localhost:8000

页面:
    /home       首页 (落地页)
    /dashboard  看板 (实时决策数据)
    /history    历史记录 (所有运行记录)
    /report     日报 (Markdown 渲染)

API (11 个端点):
    /api/health /api/holdings /api/status /api/decisions
    /api/candidates /api/analysis /api/risk /api/report
    /api/history /api/trace /api/review
"""

from __future__ import annotations

import json
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# ── 日志配置 ────────────────────────────────────

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("src.api")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# 模块级初始化标记 (防止重复添加 handler)
_logger_initialized = False


def _setup_logging() -> None:
    """配置日志 (仅执行一次)。"""
    global _logger_initialized
    if _logger_initialized:
        return
    _logger_initialized = True

    _file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "server.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    _file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    _file_handler.setLevel(logging.INFO)

    _root_logger = logging.getLogger()
    _root_logger.addHandler(_file_handler)

    _access_logger = logging.getLogger("uvicorn.access")
    _access_logger.addHandler(_file_handler)

# ── App ─────────────────────────────────────────

_setup_logging()

app = FastAPI(title="智投未来 看板", version="1.1.0")

# 确保 static 目录存在 (防止模块导入时崩溃)
os.makedirs(STATIC_DIR, exist_ok=True)

# 挂载静态资源 (CSS / JS)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _serve_html(filename: str) -> HTMLResponse:
    """读取并返回静态 HTML 文件"""
    html_path = os.path.join(STATIC_DIR, filename)
    if not os.path.isfile(html_path):
        return HTMLResponse(f"<h1>{filename} 未找到</h1>", status_code=404)
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ── trace 工具函数 ──────────────────────────────

def _latest_trace() -> dict[str, Any] | None:
    """加载最新的 trace JSON 文件 (跳过 _vN 版本备份)"""
    if not os.path.isdir(RESULTS_DIR):
        return None
    files = sorted(
        [f for f in os.listdir(RESULTS_DIR)
         if f.startswith("trace_") and f.endswith(".json") and "_v" not in f],
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
        [f for f in os.listdir(RESULTS_DIR)
         if f.startswith("trace_") and f.endswith(".json") and "_v" not in f],
        reverse=True,
    )
    result = []
    for f in files:
        date_str = f.replace("trace_", "").replace(".json", "")
        path = os.path.join(RESULTS_DIR, f)
        size_kb = round(os.path.getsize(path) / 1024, 1)
        result.append({"date": date_str, "file": f, "size_kb": size_kb})
    return result


def _valid_date(date_str: str) -> bool:
    """校验日期格式为 YYYYMMDD (防止路径遍历)。"""
    return bool(date_str) and date_str.isdigit() and len(date_str) == 8


def _load_trace_by_date(date_str: str) -> dict[str, Any] | None:
    """按日期加载 trace"""
    if not _valid_date(date_str):
        return None
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


# ── 页面路由 ────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """首页 (落地页)"""
    return _serve_html("home.html")


@app.get("/home", response_class=HTMLResponse)
async def home():
    """首页"""
    return _serve_html("home.html")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """看板页面"""
    return _serve_html("dashboard.html")


@app.get("/history", response_class=HTMLResponse)
async def history_page():
    """历史记录页面"""
    return _serve_html("history.html")


@app.get("/report", response_class=HTMLResponse)
async def report_page():
    """日报页面"""
    return _serve_html("report.html")


# ── API 路由 ────────────────────────────────────

@app.get("/api/health")
async def api_health():
    """健康检查 (供负载均衡器/监控使用)"""
    return {"status": "ok"}


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
        "total_equity": trace.get("total_equity", trace.get("total_capital", 0)),
        "total_return": trace.get("total_return", 0),
        "pipeline_version": trace.get("pipeline_version", ""),
        "elapsed": trace.get("elapsed", {}),
        "candidates_count": trace.get("screening", {}).get("total_candidates", 0),
        "decisions_count": len(trace.get("decisions", [])),
        "errors_count": len(trace.get("errors", [])),
        "errors": trace.get("errors", [])[:5],
    }


@app.get("/api/holdings")
async def api_holdings():
    """当前持仓状态 (从 positions.json 读取)"""
    positions_path = os.path.join(RESULTS_DIR, "positions.json")
    if not os.path.isfile(positions_path):
        return {"date": None, "cash": 0, "total_equity": 0, "positions": []}

    try:
        with open(positions_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"date": None, "cash": 0, "total_equity": 0, "positions": []}

    positions = []
    total_mv = 0.0
    for code, pos in data.get("positions", {}).items():
        mv = pos["shares"] * pos.get("last_price", pos["avg_cost"])
        cost = pos["shares"] * pos["avg_cost"]
        pnl = mv - cost
        pnl_pct = (pnl / cost * 100) if cost > 0 else 0
        total_mv += mv
        positions.append({
            "code": code,
            "name": pos.get("name", ""),
            "shares": pos["shares"],
            "avg_cost": round(pos["avg_cost"], 2),
            "last_price": round(pos.get("last_price", pos["avg_cost"]), 2),
            "market_value": round(mv, 2),
            "cost_value": round(cost, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "weight_pct": 0,
        })

    cash = data.get("cash", 0)
    equity = cash + total_mv
    for p in positions:
        p["weight_pct"] = round(p["market_value"] / equity * 100, 1) if equity > 0 else 0

    return {
        "date": data.get("date", ""),
        "cash": round(cash, 2),
        "total_equity": round(equity, 2),
        "total_market_value": round(total_mv, 2),
        "total_return": round(data.get("cumulative_pnl", 0) / data.get("total_capital", 500000) * 100, 2),
        "positions": positions,
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
        "total_equity": trace.get("total_equity", trace.get("total_capital", 0)),
        "total_return": trace.get("total_return", 0),
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
        if not _valid_date(date):
            raise HTTPException(status_code=400, detail="日期格式错误，需为 YYYYMMDD")
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


@app.get("/api/review")
async def api_review(date: str | None = None):
    """持仓复盘结果 (P0风控红线 + P1合理性评分 + P2卖飞检测 + P3绩效归因)"""
    if date:
        if not _valid_date(date):
            raise HTTPException(status_code=400, detail="日期格式错误，需为 YYYYMMDD")
        path = os.path.join(RESULTS_DIR, f"review_{date}.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return {"date": date, "available": False, "message": "该日期无复盘数据"}
    # 返回最新复盘结果
    if not os.path.isdir(RESULTS_DIR):
        return {"date": None, "available": False, "message": "暂无复盘数据"}
    files = sorted(
        [f for f in os.listdir(RESULTS_DIR)
         if f.startswith("review_") and f.endswith(".json")],
        reverse=True,
    )
    if not files:
        return {"date": None, "available": False, "message": "暂无复盘数据，请先运行流水线"}
    with open(os.path.join(RESULTS_DIR, files[0]), encoding="utf-8") as f:
        return json.load(f)


# ── 启动入口 ─────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("ZHITOU_HOST", "0.0.0.0")
    port = int(os.getenv("ZHITOU_PORT", "8000"))
    uvicorn.run("src.api.server:app", host=host, port=port, reload=False)
