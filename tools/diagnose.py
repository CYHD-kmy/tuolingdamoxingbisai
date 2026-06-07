"""
全流程诊断脚本 — 逐模块测试，排查哪些功能可用。
用法: python tools/diagnose.py
"""
from __future__ import annotations

import sys
import time
import json
import traceback
from datetime import datetime

sys.path.insert(0, ".")

def check(name: str, fn, *args) -> tuple[bool, str]:
    """执行检查，返回 (是否通过, 详情)"""
    try:
        t0 = time.monotonic()
        result = fn(*args)
        elapsed = time.monotonic() - t0
        return True, f"通过 ({elapsed:.1f}s) → {result}" if not isinstance(result, str) else f"通过 ({elapsed:.1f}s) → {result[:200]}"
    except Exception as e:
        return False, f"失败 ({type(e).__name__}: {e})"

def summarize(result: str) -> str:
    if isinstance(result, list):
        return f"返回 {len(result)} 条"
    elif isinstance(result, dict):
        return f"返回 {len(result)} 个字段"
    elif isinstance(result, str):
        return result[:150]
    else:
        return str(result)[:150]

print("=" * 60)
print("  智投未来 — 全流程诊断")
print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

results: dict[str, dict] = {}
TEST_CODE = "600519"  # 贵州茅台 (最稳定)

# ── 1. 配置检查 ─────────────────────────────
print("\n【1/8 配置检查】")
from src.utils.config import get_config
cfg = get_config()

checks = [
    ("LLM API Key", lambda: "已配置" if cfg.llm_api_key else "未配置"),
    ("LLM Quick Model", lambda: cfg.llm_quick),
    ("LLM Deep Model", lambda: cfg.llm_deep),
    ("LLM Base URL", lambda: cfg.llm_base_url),
    ("Tushare Token", lambda: "已配置" if cfg.tushare_available else "未配置"),
    ("Tushare Priority", lambda: str(cfg.fetcher_priority("tushare"))),
    ("AKShare Priority", lambda: str(cfg.fetcher_priority("akshare"))),
]
for name, fn in checks:
    ok, detail = check(name, fn)
    print(f"  {'✓' if ok else '✗'} {name}: {detail}")
    results[name] = {"ok": ok, "detail": detail}

# ── 2. 数据源测试 ───────────────────────────
print("\n【2/8 数据源连接】")
from src.data.interface import UnifiedDataInterface
data = UnifiedDataInterface()

data_checks = [
    ("get_stock_name(600519)", lambda: data.get_stock_name(TEST_CODE)),
    ("get_daily_data(600519, 30)", lambda: summarize(data.get_daily_data(TEST_CODE, days=30))),
    ("get_realtime_quote(600519)", lambda: summarize(str(data.get_realtime_quote(TEST_CODE)))),
    ("get_stock_info(600519)", lambda: summarize(str(data.get_stock_info(TEST_CODE)))[:200]),
    ("get_fund_flow(600519, 5)", lambda: summarize(data.get_fund_flow(TEST_CODE, days=5))),
    ("get_financial_indicators(600519)", lambda: summarize(data.get_financial_indicators(TEST_CODE))),
    ("get_northbound_flow(5)", lambda: summarize(data.get_northbound_flow(days=5))),
    ("get_northbound_stock(600519)", lambda: summarize(data.get_northbound_stock(TEST_CODE, days=10))),
    ("get_margin_detail(600519)", lambda: summarize(data.get_margin_detail(TEST_CODE, days=10))),
    ("get_research_reports(600519)", lambda: summarize(data.get_research_reports(TEST_CODE, days=30))),
    ("get_shareholder_count(600519)", lambda: summarize(data.get_shareholder_count(TEST_CODE))),
    ("get_market_breadth()", lambda: summarize(str(data.get_market_breadth()))[:200]),
    ("get_limit_up_pool()", lambda: summarize(data.get_limit_up_pool())),
    ("get_auction_data()", lambda: summarize(data.get_auction_data())),
    ("get_news(贵州茅台, 3)", lambda: summarize(data.get_news("贵州茅台", days=3))),
]
for name, fn in data_checks:
    ok, detail = check(name, fn)
    icon = "✓" if ok else "✗"
    print(f"  {icon} {name}: {detail}")
    results[name] = {"ok": ok, "detail": detail}

# ── 3. LLM 连接测试 ─────────────────────────
print("\n【3/8 LLM 连接】")
try:
    from src.llm.factory import get_quick_llm, get_deep_llm
    from src.llm.schema import Message

    quick_llm = get_quick_llm()
    deep_llm = get_deep_llm()

    # Quick LLM 测试
    ok, detail = check("Quick LLM ping", lambda: summarize(quick_llm.chat([
        Message(role="user", content="回复'OK'")
    ], tools=None).content))
    print(f"  {'✓' if ok else '✗'} Quick LLM: {detail}")
    results["Quick LLM"] = {"ok": ok, "detail": detail}

    # Deep LLM 测试
    ok, detail = check("Deep LLM ping", lambda: summarize(deep_llm.chat([
        Message(role="user", content="回复'OK'")
    ], tools=None).content))
    print(f"  {'✓' if ok else '✗'} Deep LLM: {detail}")
    results["Deep LLM"] = {"ok": ok, "detail": detail}

except Exception as e:
    print(f"  ✗ LLM 连接失败: {e}")
    results["LLM"] = {"ok": False, "detail": str(e)}

# ── 4. 分析师测试 ───────────────────────────
print("\n【4/8 分析师 (单只股票: 600519)】")

try:
    from src.agents.analysts.technical import TechnicalAnalyst
    from src.agents.analysts.fundamentals import FundamentalsAnalyst
    from src.agents.analysts.fund_flow import FundFlowAnalyst
    from src.agents.analysts.news_sentiment import NewsSentimentAnalyst
    from src.agents.analysts.policy import PolicyAnalyst
    from src.agents.analysts.sector_hunter import SectorHunterAnalyst

    analysts_to_test = [
        ("TechnicalAnalyst", TechnicalAnalyst(quick_llm, data)),
        ("FundamentalsAnalyst", FundamentalsAnalyst(quick_llm, data)),
        ("FundFlowAnalyst", FundFlowAnalyst(quick_llm, data)),
        ("NewsSentimentAnalyst", NewsSentimentAnalyst(quick_llm, data)),
        ("PolicyAnalyst", PolicyAnalyst(quick_llm, data)),
        ("SectorHunterAnalyst", SectorHunterAnalyst(quick_llm, data)),
    ]

    for name, analyst in analysts_to_test:
        ok, detail = check(f"{name}.analyze(600519)", lambda a=analyst: (
            r := a.analyze(TEST_CODE),
            f"signal={r.signal} conf={r.confidence:.2f} [{r.reasoning[:80]}...]"
        )[-1])
        print(f"  {'✓' if ok else '✗'} {name}: {detail}")
        results[name] = {"ok": ok, "detail": detail}
except Exception as e:
    print(f"  ✗ 分析师测试失败: {e}")
    traceback.print_exc()

# ── 5. 筛选模块 ─────────────────────────────
print("\n【5/8 筛选模块】")
try:
    from src.screening.pipeline import ScreeningPipeline
    pipeline = ScreeningPipeline(data)
    ok, detail = check("ScreeningPipeline.run()", lambda: (
        r := pipeline.run(),
        f"候选 {len(r.candidates)} 只, 错误 {len(r.errors)} 个"
    )[-1])
    print(f"  {'✓' if ok else '✗'} ScreeningPipeline: {detail}")
    results["ScreeningPipeline"] = {"ok": ok, "detail": detail}
except Exception as e:
    print(f"  ✗ 筛选失败: {e}")
    results["ScreeningPipeline"] = {"ok": False, "detail": str(e)}

# ── 6. 分析/辩论/风控/组合 ──────────────────
print("\n【6/8 分析+辩论+风控+组合 (模块可用性)】")
try:
    from src.agents.researchers.engine import DebateEngine
    from src.agents.managers.research_manager import ResearchManager
    from src.agents.managers.risk_manager import RiskManager
    from src.agents.managers.portfolio_manager import PortfolioManager

    checks_to_run: list[tuple[str, callable]] = []

    # 获取筛选结果做候选
    scr = pipeline.run()
    codes = [c.code for c in scr.candidates[:5]]
    if codes:
        daily = data.batch_daily_data(codes, days=30, max_workers=4)

        # Risk Manager (不需要 LLM)
        risk_mgr = RiskManager(total_capital=cfg.initial_capital)
        # 用简单的 mock verdict 测试
        from src.agents.managers.risk_manager import ResearchVerdict
        mock_verdicts = [
            ResearchVerdict(code=c, name=data.get_stock_name(c), direction="buy",
                            confidence=0.70, target_price=0, risk_level="medium",
                            core_reasoning="test", key_risks=[])
            for c in codes
        ]
        ok, detail = check("RiskManager.compute_limits()", lambda: (
            limits := risk_mgr.compute_limits(mock_verdicts, daily),
            f"限制 {len(limits)} 只, 可买 {sum(1 for l in limits.values() if l.max_shares > 0)} 只"
        )[-1])
        print(f"  {'✓' if ok else '✗'} RiskManager: {detail}")
        results["RiskManager"] = {"ok": ok, "detail": detail}

        # Portfolio Manager ETF (确定性规则，不需要 LLM)
        pm = PortfolioManager(deep_llm)
        limits = risk_mgr.compute_limits(mock_verdicts, daily)
        ok, detail = check("PortfolioManager.construct_etf()", lambda: (
            pf := pm.construct_etf(mock_verdicts, limits, daily,
                                   cash_available=cfg.initial_capital,
                                   total_capital=cfg.initial_capital),
            f"决策 {len(pf.decisions)} 笔, 用款 {pf.cash_used:.0f}"
        )[-1])
        print(f"  {'✓' if ok else '✗'} PortfolioManager: {detail}")
        results["PortfolioManager"] = {"ok": ok, "detail": detail}
    else:
        print("  无候选股，跳过 RiskManager/PortfolioManager")

    # 模块导入检查
    ok, detail = check("ResearchManager", lambda: "已导入")
    print(f"  {'✓' if ok else '✗'} ResearchManager: {detail}")
    results["ResearchManager"] = {"ok": ok, "detail": detail}

    ok, detail = check("DebateEngine", lambda: "已导入")
    print(f"  {'✓' if ok else '✗'} DebateEngine: {detail}")
    results["DebateEngine"] = {"ok": ok, "detail": detail}

except Exception as e:
    print(f"  ✗ 分析阶段失败: {e}")
    traceback.print_exc()

# ── 7. Analytics 模块 ────────────────────────
print("\n【7/8 Analytics 分析模块】")
try:
    from src.analytics.market_sentiment import MarketSentimentAnalyzer
    from src.analytics.limit_up import LimitUpAnalyzer
    from src.analytics.auction import AuctionAnalyzer
    from src.analytics.dragon_tiger import DragonTigerAnalyzer
    from src.analytics.volume_price import VolumePriceAnalyzer

    # Market Sentiment
    breadth = data.get_market_breadth()
    ok, detail = check("MarketSentimentAnalyzer", lambda: (
        msa := MarketSentimentAnalyzer(),
        r := msa.analyze(breadth=breadth),
        f"regime={r.regime}, position_advice={r.position_advice}"
    )[-1])
    print(f"  {'✓' if ok else '✗'} MarketSentiment: {detail}")
    results["MarketSentiment"] = {"ok": ok, "detail": detail}

    # Limit Up
    limit_data = data.get_limit_up_pool()
    ok, detail = check("LimitUpAnalyzer", lambda: (
        lua := LimitUpAnalyzer(),
        r := lua.analyze(limit_data),
        f"涨停 {len(r)} 只"
    )[-1])
    print(f"  {'✓' if ok else '✗'} LimitUpAnalyzer: {detail}")
    results["LimitUpAnalyzer"] = {"ok": ok, "detail": detail}

    # Auction
    auction_data = data.get_auction_data()
    ok, detail = check("AuctionAnalyzer", lambda: (
        aa := AuctionAnalyzer(),
        r := aa.analyze(auction_data[:100]),
        f"竞价信号 {len(r)} 条"
    )[-1])
    print(f"  {'✓' if ok else '✗'} AuctionAnalyzer: {detail}")
    results["AuctionAnalyzer"] = {"ok": ok, "detail": detail}

    # Dragon Tiger
    dt_data = data.get_dragon_tiger_stats(days=10)
    ok, detail = check("DragonTigerAnalyzer", lambda: (
        dta := DragonTigerAnalyzer(),
        r := dta.analyze(dt_data),
        f"龙虎榜信号 {len(r)} 条"
    )[-1])
    print(f"  {'✓' if ok else '✗'} DragonTigerAnalyzer: {detail}")
    results["DragonTigerAnalyzer"] = {"ok": ok, "detail": detail}

    # Volume Price
    if codes:
        vpa = VolumePriceAnalyzer()
        ok, detail = check("VolumePriceAnalyzer", lambda: (
            r := vpa.batch_detect(daily),
            f"量价信号 {len(r)} 只"
        )[-1])
        print(f"  {'✓' if ok else '✗'} VolumePriceAnalyzer: {detail}")
        results["VolumePriceAnalyzer"] = {"ok": ok, "detail": detail}

except Exception as e:
    print(f"  ✗ Analytics 失败: {e}")
    traceback.print_exc()

# ── 8. 汇总 ──────────────────────────────────
print("\n" + "=" * 60)
print("  诊断汇总")
print("=" * 60)

all_checks = [
    "LLM API Key", "LLM Quick Model", "Tushare Token",
    "get_stock_name(600519)", "get_daily_data(600519, 30)",
    "get_realtime_quote(600519)", "get_fund_flow(600519, 5)",
    "get_financial_indicators(600519)", "get_northbound_flow(5)",
    "get_market_breadth()", "get_limit_up_pool()", "get_auction_data()",
    "Quick LLM", "Deep LLM",
    "TechnicalAnalyst", "FundamentalsAnalyst", "FundFlowAnalyst",
    "NewsSentimentAnalyst", "PolicyAnalyst", "SectorHunterAnalyst",
    "ScreeningPipeline", "ResearchManager", "RiskManager",
    "DebateEngine", "PortfolioManager",
    "MarketSentiment", "LimitUpAnalyzer", "AuctionAnalyzer",
    "DragonTigerAnalyzer", "VolumePriceAnalyzer",
]

pass_count = 0
fail_count = 0
for name in all_checks:
    r = results.get(name, {})
    ok = r.get("ok", False)
    if ok:
        pass_count += 1
        print(f"  ✓ {name}")
    else:
        fail_count += 1
        print(f"  ✗ {name} — {r.get('detail', '未执行')}")

print(f"\n通过: {pass_count}/{pass_count+fail_count}, 失败: {fail_count}")
print("=" * 60)
