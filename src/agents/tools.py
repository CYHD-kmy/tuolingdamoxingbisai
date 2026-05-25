"""
数据工具注册表 — 定义分析师可用的工具。

每个工具对应 UnifiedDataInterface 的一个查询方法，
LLM 通过 Tool Calling 自主决定何时调用。
"""

from __future__ import annotations

from ..llm.schema import Tool, ToolParam


# ── 工具定义 ──────────────────────────────────

TOOL_GET_DAILY = Tool(
    name="get_daily_data",
    description="获取股票日线数据，包含 OHLCV、均线(MA5/10/20)、MACD、RSI 等技术指标",
    parameters=[
        ToolParam(name="code", type="string", description="股票代码，如 600519", required=True),
        ToolParam(name="days", type="integer", description="回看天数，默认 30", required=False),
    ],
)

TOOL_GET_REALTIME = Tool(
    name="get_realtime_quote",
    description="获取股票实时行情：最新价、涨跌幅、量比、换手率、市盈率、市净率、总市值",
    parameters=[
        ToolParam(name="code", type="string", description="股票代码，如 600519", required=True),
    ],
)

TOOL_GET_FUND_FLOW = Tool(
    name="get_fund_flow",
    description="获取个股资金流向：主力/超大单/大单/中单/小单 的净流入金额和占比",
    parameters=[
        ToolParam(name="code", type="string", description="股票代码，如 600519", required=True),
        ToolParam(name="days", type="integer", description="回看天数，默认 5", required=False),
    ],
)

TOOL_GET_STOCK_INFO = Tool(
    name="get_stock_info",
    description="获取股票基本信息：公司名称、所属行业、上市日期、市盈率、市净率、总市值",
    parameters=[
        ToolParam(name="code", type="string", description="股票代码，如 600519", required=True),
    ],
)

TOOL_GET_NEWS = Tool(
    name="get_news",
    description="搜索相关财经新闻，返回标题、内容摘要、时间和来源",
    parameters=[
        ToolParam(name="keyword", type="string", description="搜索关键词，如'新能源'或股票代码", required=True),
        ToolParam(name="days", type="integer", description="回看天数，默认 3", required=False),
    ],
)

TOOL_GET_ANNOUNCEMENTS = Tool(
    name="get_announcements",
    description="获取个股近期公告：业绩预告、重大合同、增减持等",
    parameters=[
        ToolParam(name="code", type="string", description="股票代码，如 600519", required=True),
        ToolParam(name="days", type="integer", description="回看天数，默认 7", required=False),
    ],
)


# ── 工具集 (按分析师类型分组) ────────────────

def tools_for(analyst_type: str) -> list[Tool]:
    """返回指定分析师类型的工具集"""
    return {
        "technical":    [TOOL_GET_DAILY, TOOL_GET_REALTIME],
        "fundamentals": [TOOL_GET_STOCK_INFO, TOOL_GET_DAILY, TOOL_GET_ANNOUNCEMENTS],
        "fund_flow":    [TOOL_GET_FUND_FLOW, TOOL_GET_REALTIME],
        "news":         [TOOL_GET_NEWS, TOOL_GET_ANNOUNCEMENTS, TOOL_GET_REALTIME],
    }.get(analyst_type, [])
