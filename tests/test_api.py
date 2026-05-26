"""
API / Web 看板单元测试 — 覆盖 FastAPI 端点和静态页面挂载。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

# 导入前设置环境，避免缺失 .env 报错
os.environ.setdefault("LLM_API_KEY", "sk-test-key-for-ci")
os.environ.setdefault("ZHITOU_HOST", "127.0.0.1")
os.environ.setdefault("ZHITOU_PORT", "8000")

from src.api.server import app

client = TestClient(app)


# ── 页面路由测试 ──────────────────────────────

def test_home_page():
    """GET /home 返回首页 HTML"""
    resp = client.get("/home")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_dashboard_page():
    """GET /dashboard 返回看板 HTML"""
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_history_page():
    """GET /history 返回历史 HTML"""
    resp = client.get("/history")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_report_page():
    """GET /report 返回日报 HTML"""
    resp = client.get("/report")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


# ── API 端点测试 ─────────────────────────────

def test_api_health():
    """GET /api/health 返回健康检查"""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_api_status():
    """GET /api/status 返回系统状态"""
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "date" in data
    assert "total_capital" in data
    assert "ready" in data


def test_api_decisions():
    """GET /api/decisions 返回决策对象"""
    resp = client.get("/api/decisions")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert "decisions" in data


def test_api_decisions_with_date():
    """GET /api/decisions?date=20260525 按日期查询"""
    resp = client.get("/api/decisions?date=20260525")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)


def test_api_candidates():
    """GET /api/candidates 返回候选池"""
    resp = client.get("/api/candidates")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert "candidates" in data


def test_api_analysis():
    """GET /api/analysis 返回分析详情"""
    resp = client.get("/api/analysis")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert "analysis" in data


def test_api_analysis_with_code():
    """GET /api/analysis?code=600519 按股票筛选"""
    resp = client.get("/api/analysis?code=600519")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)


def test_api_risk():
    """GET /api/risk 返回风控详情"""
    resp = client.get("/api/risk")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert "limits" in data


def test_api_report():
    """GET /api/report 返回日报原文"""
    resp = client.get("/api/report")
    assert resp.status_code in (200, 404)  # 可能无日报文件


def test_api_history():
    """GET /api/history 返回历史记录"""
    resp = client.get("/api/history")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)


def test_api_trace():
    """GET /api/trace 返回追踪 JSON"""
    resp = client.get("/api/trace")
    assert resp.status_code in (200, 404)  # 可能无最新追踪


# ── 静态文件测试 ─────────────────────────────

def test_static_css():
    """GET /static/css/style.css 返回样式"""
    resp = client.get("/static/css/style.css")
    assert resp.status_code == 200
    assert "text/css" in resp.headers["content-type"]


def test_static_js():
    """GET /static/js/common.js 返回脚本"""
    resp = client.get("/static/js/common.js")
    assert resp.status_code == 200


# ── 404 测试 ─────────────────────────────────

def test_404_page():
    """GET /nonexistent 返回 404"""
    resp = client.get("/nonexistent")
    assert resp.status_code == 404


if __name__ == "__main__":
    test_home_page()
    test_dashboard_page()
    test_history_page()
    test_report_page()
    test_api_health()
    test_api_status()
    test_api_decisions()
    test_api_decisions_with_date()
    test_api_candidates()
    test_api_analysis()
    test_api_analysis_with_code()
    test_api_risk()
    test_api_report()
    test_api_history()
    test_api_trace()
    test_static_css()
    test_static_js()
    test_404_page()
    print("api: 全部通过")
