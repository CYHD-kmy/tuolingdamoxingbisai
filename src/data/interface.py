"""
统一数据接口 — 多源降级编排层。

设计模式: 策略模式 + 优先级降级
- 主数据源 (AKShare) → 增强数据源 (Tushare) → 兜底数据源 (BaoStock)
- 单只股票失败不阻断整体流程
- 所有数据经过缓存层减少重复请求

数据质量标记:
- "live":    来自主数据源的实时数据
- "cached":  来自缓存的历史数据
- "fallback": 来自兜底数据源
- "stale":   数据源全部失败，使用过期缓存
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from .cache import DataCache
from .fetchers.akshare_fetcher import (
    AKShareFetcher, StockDaily, RealtimeQuote, FundFlow, MarketSnapshot,
)
from .fetchers.tushare_fetcher import TushareFetcher
from .fetchers.baostock_fetcher import BaoStockFetcher
from ..utils.config import get_config

logger = logging.getLogger(__name__)

_fetchers_cache: list | None = None
_fetchers_cache_key: tuple | None = None


def _build_fetchers(config) -> list:
    """构建并排序 fetcher 列表，结果按 config 的优先级缓存。"""
    global _fetchers_cache, _fetchers_cache_key
    key = (
        config.fetcher_priority("akshare"),
        config.fetcher_priority("tushare"),
        config.fetcher_priority("baostock"),
        config.tushare_available,
    )
    if _fetchers_cache is not None and _fetchers_cache_key == key:
        return _fetchers_cache

    fetchers = [
        AKShareFetcher(),
        TushareFetcher(),
        BaoStockFetcher(),
    ]
    fetchers.sort(key=lambda f: config.fetcher_priority(f.name))
    _fetchers_cache = fetchers
    _fetchers_cache_key = key
    return fetchers


class UnifiedDataInterface:
    """
    统一数据接口 — 对外暴露的唯一数据入口。

    使用示例:
        udi = UnifiedDataInterface()
        daily = udi.get_daily_data("600519", days=60)
        quote = udi.get_realtime_quote("600519")
        snapshot = udi.get_market_snapshot()
    """

    def __init__(self) -> None:
        self._config = get_config()
        self._cache = DataCache()
        self._fetchers = _build_fetchers(self._config)

    # ── 公开 API ─────────────────────────────

    def get_daily_data(self, code: str, days: int = 60, force_refresh: bool = False) -> list[StockDaily]:
        """
        获取标准化日线数据 (含技术指标)。

        code: 600519 / 000858
        days: 回看天数
        force_refresh: 跳过缓存
        """
        if not force_refresh:
            cached = self._cache.daily_data(code)
            if cached is not None:
                return cached

        for fetcher in self._fetchers:
            try:
                data = fetcher.get_daily_data(code, days)
                if data:
                    self._cache.set_daily_data(code, data)
                    return data
            except Exception:
                logger.debug("%s: get_daily_data 失败, 尝试下一个", fetcher.name)
                continue

        # 所有数据源失败，尝试返回过期缓存
        stale = self._cache.daily_data(code)
        if stale is not None:
            logger.warning("%s: 所有数据源失败，使用过期缓存", code)
            return stale

        logger.error("%s: 所有数据源均失败，无数据可用", code)
        return []

    def get_realtime_quote(self, code: str) -> Optional[RealtimeQuote]:
        """获取实时行情"""
        cached = self._cache.realtime_quote(code)
        if cached is not None:
            return cached

        for fetcher in self._fetchers:
            try:
                quote = fetcher.get_realtime_quote(code)
                if quote is not None and quote.price > 0:
                    self._cache.set_realtime_quote(code, quote)
                    return quote
            except Exception:
                logger.debug("%s: get_realtime_quote 失败", fetcher.name)
                continue

        return None

    def get_stock_name(self, code: str) -> str:
        """获取股票名称，失败返回代码本身"""
        quote = self.get_realtime_quote(code)
        if quote and quote.name:
            return quote.name

        for fetcher in self._fetchers:
            try:
                name = fetcher.get_stock_name(code)
                if name and name != code:
                    return name
            except Exception:
                continue

        return code

    def get_stock_info(self, code: str) -> dict:
        """获取股票基本信息: 名称、行业、上市日期"""
        for fetcher in self._fetchers:
            try:
                info = fetcher.get_stock_info(code)
                if info:
                    return info
            except Exception:
                continue
        return {}

    def get_fund_flow(self, code: str, days: int = 5) -> list[FundFlow]:
        """获取近期资金流向"""
        for fetcher in self._fetchers:
            try:
                flows = fetcher.get_fund_flow(code, days)
                if flows:
                    return flows
            except Exception:
                continue
        return []

    def get_market_snapshot(self) -> list[MarketSnapshot]:
        """获取全市场快照 (按成交额降序 Top 3000)"""
        for fetcher in self._fetchers:
            try:
                snapshot = fetcher.get_market_snapshot()
                if snapshot:
                    return snapshot
            except Exception:
                continue
        return []

    def get_news(self, keyword: str, days: int = 3) -> list[dict]:
        """新闻搜索"""
        for fetcher in self._fetchers:
            try:
                result = fetcher.get_news(keyword, days)
                if result:
                    return result
            except Exception:
                continue
        return []

    def get_announcements(self, code: str, days: int = 7) -> list[dict]:
        """个股公告"""
        for fetcher in self._fetchers:
            try:
                result = fetcher.get_announcements(code, days)
                if result:
                    return result
            except Exception:
                continue
        return []

    # ── 别名方法 (与设计文档命名保持一致) ─────

    def get_stock_daily(self, code: str, days: int = 60) -> list[StockDaily]:
        """get_daily_data 的别名"""
        return self.get_daily_data(code, days)

    def get_news_sentiment(self, keyword: str, days: int = 3) -> list[dict]:
        """get_news 的别名，情感分析由 LLM 完成"""
        return self.get_news(keyword, days)

    # ── 批量操作 ─────────────────────────────

    def batch_realtime_quotes(
        self, codes: list[str], max_workers: int = 8
    ) -> dict[str, Optional[RealtimeQuote]]:
        """并发获取多个股票的实时行情"""
        results: dict[str, Optional[RealtimeQuote]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self.get_realtime_quote, c): c for c in codes}
            for future in as_completed(futures):
                code = futures[future]
                try:
                    results[code] = future.result(timeout=self._config.request_timeout)
                except Exception:
                    logger.exception("batch: 获取行情失败 %s", code)
                    results[code] = None
            del futures
        return results

    def batch_daily_data(
        self, codes: list[str], days: int = 60, max_workers: int = 4
    ) -> dict[str, list[StockDaily]]:
        """并发获取多个股票的日线数据"""
        results: dict[str, list[StockDaily]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self.get_daily_data, c, days): c for c in codes}
            for future in as_completed(futures):
                code = futures[future]
                try:
                    results[code] = future.result(timeout=self._config.request_timeout * 3)
                except Exception:
                    logger.exception("batch: 获取日线失败 %s", code)
                    results[code] = []
            del futures
        return results

    def batch_stock_info(
        self, codes: list[str], max_workers: int = 8
    ) -> dict[str, dict]:
        """并发获取多个股票的基本信息"""
        results: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self.get_stock_info, c): c for c in codes}
            for future in as_completed(futures):
                code = futures[future]
                try:
                    results[code] = future.result(timeout=self._config.request_timeout)
                except Exception:
                    results[code] = {}
            del futures
        return results

    def batch_fund_flows(
        self, codes: list[str], days: int = 5, max_workers: int = 6
    ) -> dict[str, list[FundFlow]]:
        """并发获取多个股票的资金流向"""
        results: dict[str, list[FundFlow]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self.get_fund_flow, c, days): c for c in codes}
            for future in as_completed(futures):
                code = futures[future]
                try:
                    results[code] = future.result(timeout=self._config.request_timeout * 2)
                except Exception:
                    results[code] = []
            del futures
        return results

    # ── 数据质量 ─────────────────────────────

    def is_tradable(self, code: str) -> bool:
        """
        检查股票当日是否可交易。

        排除: ST/*ST, 退市整理, 停牌
        """
        q = self.get_realtime_quote(code)
        if q is None:
            return False
        # 名称为空或带 ST 标记
        if not q.name or "ST" in q.name.upper():
            return False
        # 价格为 0 或换手率为 0 可能停牌
        if q.price <= 0.01:
            return False
        return True

    def get_tradable_codes(self, codes: list[str]) -> list[str]:
        """过滤出可交易的股票代码"""
        quotes = self.batch_realtime_quotes(codes)
        return [
            c for c, q in quotes.items()
            if q is not None and q.price > 0.01 and "ST" not in q.name.upper()
        ]
