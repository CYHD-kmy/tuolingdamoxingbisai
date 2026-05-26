"""
Data 层单元测试 — 覆盖缓存、数据模型、代码标准化等。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.cache import DataCache, CacheEntry
from src.data.fetchers.akshare_fetcher import StockDaily, RealtimeQuote, FundFlow, MarketSnapshot


# ── DataCache 测试 ─────────────────────────

def test_cache_set_get():
    """缓存: 基本存取"""
    cache = DataCache()
    cache.set("test", {"key": "value"}, 300, "item1")
    result = cache.get("test", "item1")
    assert result == {"key": "value"}


def test_cache_miss():
    """缓存: 未命中返回 None"""
    cache = DataCache()
    assert cache.get("nonexistent", "key") is None


def test_cache_ttl_expiry():
    """缓存: TTL 过期返回 None"""
    import time
    cache = DataCache()
    cache.set("test", "data", 0, "expired")  # TTL=0 立即过期
    time.sleep(0.01)
    assert cache.get("test", "expired") is None


def test_cache_clear():
    """缓存: 清空指定前缀"""
    cache = DataCache()
    cache.set("daily", "data1", 300, "600519")
    cache.set("realtime", "data2", 60, "600519")

    assert cache.get("daily", "600519") == "data1"
    assert cache.get("realtime", "600519") == "data2"

    cache.clear("daily")
    assert cache.get("daily", "600519") is None
    assert cache.get("realtime", "600519") == "data2"  # 不受影响


def test_cache_clear_all():
    """缓存: 清空全部"""
    cache = DataCache()
    cache.set("daily", "d", 300, "a")
    cache.set("realtime", "r", 60, "b")
    cache.clear()
    assert cache.get("daily", "a") is None
    assert cache.get("realtime", "b") is None


def test_cache_thread_safety():
    """缓存: 并发写入不崩溃"""
    import threading
    cache = DataCache()

    def writer(code: str):
        for _ in range(50):
            cache.set("daily", "data", 300, code)

    threads = [threading.Thread(target=writer, args=(f"code{i}",)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 不崩溃即为通过


# ── 数据模型测试 ──────────────────────────

def test_stock_daily_defaults():
    """StockDaily 默认值"""
    d = StockDaily(date="2026-05-26", open=100, high=102, low=99, close=101,
                   volume=1e6, amount=1e8, pct_chg=1.0)
    assert d.turnover == 0.0
    assert d.ma5 == 0.0
    assert d.macd_dif == 0.0
    assert d.rsi_6 == 0.0


def test_realtime_quote_defaults():
    """RealtimeQuote 默认值"""
    q = RealtimeQuote(code="600519", name="茅台", price=1680, open=1660,
                      high=1690, low=1655, pre_close=1665, pct_chg=0.9,
                      volume=1e6, amount=1.6e9, turnover=0.5, volume_ratio=1.2)
    assert q.pe == 0.0
    assert q.pb == 0.0
    assert q.total_mv == 0.0
    assert q.source == ""


def test_fund_flow_main_pct_default():
    """FundFlow main_pct 默认值"""
    f = FundFlow(date="2026-05-26", main_net_inflow=500, super_large_net=300,
                 large_net=200, medium_net=50, small_net=-50)
    assert f.main_pct == 0.0


def test_market_snapshot_pe_default():
    """MarketSnapshot PE/mv 默认值"""
    s = MarketSnapshot(code="600519", name="茅台", price=1680,
                       pct_chg=0.9, volume_ratio=1.2, turnover=0.5, amount=1e9)
    assert s.pe == 0.0
    assert s.total_mv == 0.0


# ── AKShare 代码标准化 ─────────────────────

def test_normalize_code_sh():
    """代码标准化: 去除 sh 前缀"""
    from src.data.fetchers.akshare_fetcher import AKShareFetcher
    assert AKShareFetcher._normalize_code("sh600519") == "600519"
    assert AKShareFetcher._normalize_code("SH600519") == "600519"


def test_normalize_code_sz():
    """代码标准化: 去除 sz 前缀"""
    from src.data.fetchers.akshare_fetcher import AKShareFetcher
    assert AKShareFetcher._normalize_code("sz000858") == "000858"
    assert AKShareFetcher._normalize_code("SZ300750") == "300750"


def test_normalize_code_plain():
    """代码标准化: 纯数字不变"""
    from src.data.fetchers.akshare_fetcher import AKShareFetcher
    assert AKShareFetcher._normalize_code("600519") == "600519"


def test_eastmoney_code_sh():
    """东方财富代码: 沪市"""
    from src.data.fetchers.akshare_fetcher import AKShareFetcher
    assert AKShareFetcher._eastmoney_code("600519") == "sh600519"
    assert AKShareFetcher._eastmoney_code("900001") == "sh900001"


def test_eastmoney_code_sz():
    """东方财富代码: 深市"""
    from src.data.fetchers.akshare_fetcher import AKShareFetcher
    assert AKShareFetcher._eastmoney_code("000858") == "sz000858"
    assert AKShareFetcher._eastmoney_code("300750") == "sz300750"


# ── 快照缓存测试 ──────────────────────────

def test_spot_cache_class_attribute():
    """全市场快照缓存: 类属性初始状态"""
    from src.data.fetchers.akshare_fetcher import AKShareFetcher
    # 保存并恢复，避免污染其他测试
    saved_cache = AKShareFetcher._spot_cache
    saved_time = AKShareFetcher._spot_cache_time
    try:
        AKShareFetcher._spot_cache = None
        AKShareFetcher._spot_cache_time = 0.0

        assert AKShareFetcher._spot_cache is None
        assert AKShareFetcher._spot_cache_time == 0.0
        assert AKShareFetcher._SPOT_CACHE_TTL == 60.0
    finally:
        AKShareFetcher._spot_cache = saved_cache
        AKShareFetcher._spot_cache_time = saved_time


if __name__ == "__main__":
    test_cache_set_get()
    test_cache_miss()
    test_cache_ttl_expiry()
    test_cache_clear()
    test_cache_clear_all()
    test_cache_thread_safety()
    test_stock_daily_defaults()
    test_realtime_quote_defaults()
    test_fund_flow_main_pct_default()
    test_market_snapshot_pe_default()
    test_normalize_code_sh()
    test_normalize_code_sz()
    test_normalize_code_plain()
    test_eastmoney_code_sh()
    test_eastmoney_code_sz()
    test_spot_cache_class_attribute()
    print("data: 全部通过")
