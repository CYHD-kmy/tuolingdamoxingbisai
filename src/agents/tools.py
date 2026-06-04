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

TOOL_GET_NORTHBOUND = Tool(
    name="get_northbound_flow",
    description="获取北向资金(沪深股通)净流向：当日净流入额，沪股通/深股通分别流入",
    parameters=[
        ToolParam(name="days", type="integer", description="回看天数，默认 5", required=False),
    ],
)

TOOL_GET_NORTHBOUND_STOCK = Tool(
    name="get_northbound_stock",
    description="获取个股沪深股通持仓变化：外资持股数量和占比变化",
    parameters=[
        ToolParam(name="code", type="string", description="股票代码，如 600519", required=True),
        ToolParam(name="days", type="integer", description="回看天数，默认 10", required=False),
    ],
)

TOOL_GET_MARGIN = Tool(
    name="get_margin_detail",
    description="获取个股融资融券明细：融资余额、融资买入额、融券余量变化",
    parameters=[
        ToolParam(name="code", type="string", description="股票代码，如 600519", required=True),
        ToolParam(name="days", type="integer", description="回看天数，默认 10", required=False),
    ],
)

TOOL_GET_FINANCIALS = Tool(
    name="get_financials",
    description="获取深度财务指标：ROE/ROA/毛利率/净利率/营收增速/利润增速/负债率/每股收益/现金流，多报告期趋势",
    parameters=[
        ToolParam(name="code", type="string", description="股票代码，如 600519", required=True),
    ],
)

TOOL_GET_RESEARCH = Tool(
    name="get_research_reports",
    description="获取个股分析师研报：研究机构、评级、日期",
    parameters=[
        ToolParam(name="code", type="string", description="股票代码，如 600519", required=True),
        ToolParam(name="days", type="integer", description="回看天数，默认 30", required=False),
    ],
)

TOOL_GET_SHAREHOLDER = Tool(
    name="get_shareholder_count",
    description="获取股东人数变化趋势：股东人数增减反映筹码集中/分散程度",
    parameters=[
        ToolParam(name="code", type="string", description="股票代码，如 600519", required=True),
    ],
)


# ── 工具集 (按分析师类型分组) ────────────────

def tools_for(analyst_type: str) -> list[Tool]:
    """返回指定分析师类型的工具集"""
    return {
        "technical":    [TOOL_GET_DAILY, TOOL_GET_REALTIME],
        "fundamentals": [TOOL_GET_STOCK_INFO, TOOL_GET_DAILY, TOOL_GET_ANNOUNCEMENTS, TOOL_GET_FINANCIALS, TOOL_GET_RESEARCH],
        "fund_flow":    [TOOL_GET_FUND_FLOW, TOOL_GET_REALTIME, TOOL_GET_NORTHBOUND, TOOL_GET_NORTHBOUND_STOCK, TOOL_GET_MARGIN],
        "news":           [TOOL_GET_NEWS, TOOL_GET_ANNOUNCEMENTS, TOOL_GET_REALTIME, TOOL_GET_RESEARCH],
        "policy":         [TOOL_GET_NEWS, TOOL_GET_ANNOUNCEMENTS, TOOL_GET_STOCK_INFO, TOOL_GET_REALTIME, TOOL_GET_RESEARCH],
        "sector_hunter":  [TOOL_GET_REALTIME, TOOL_GET_FUND_FLOW, TOOL_GET_STOCK_INFO, TOOL_GET_NEWS],
    }.get(analyst_type, [])
