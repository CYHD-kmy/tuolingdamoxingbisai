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
    NorthboundFlow, MarginData, FinancialIndicator, ETFSpot,
    UnlockShares, ShareholderCount, InstitutionalVisit, MarketActivity,
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
        self._quality: dict[str, str] = {}  # key: "code:data_type" → quality_level

    # ── 数据质量 ─────────────────────────────

    _QUALITY_LIVE = "live"
    _QUALITY_CACHED = "cached"
    _QUALITY_FALLBACK = "fallback"
    _QUALITY_STALE = "stale"

    def get_data_quality(self, code: str, data_type: str) -> str:
        """查询某只股票某类数据的质量等级 (live/cached/fallback/stale)"""
        return self._quality.get(f"{code}:{data_type}", "")

    def get_batch_quality(self, codes: list[str], data_type: str) -> dict[str, str]:
        """批量查询数据质量 {code: quality_level}"""
        return {c: self._quality.get(f"{c}:{data_type}", "") for c in codes}

    def _set_quality(self, code: str, data_type: str, level: str) -> None:
        self._quality[f"{code}:{data_type}"] = level

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
                self._set_quality(code, "daily", self._QUALITY_CACHED)
                return cached

        primary = self._fetchers[0].name if self._fetchers else ""
        for fetcher in self._fetchers:
            try:
                data = fetcher.get_daily_data(code, days)
                if data:
                    self._cache.set_daily_data(code, data)
                    level = self._QUALITY_LIVE if fetcher.name == primary else self._QUALITY_FALLBACK
                    self._set_quality(code, "daily", level)
                    return data
            except Exception:
                logger.debug("%s: get_daily_data 失败, 尝试下一个", fetcher.name)
                continue

        # 所有数据源失败，尝试返回过期缓存
        stale = self._cache.daily_data(code)
        if stale is not None:
            logger.warning("%s: 所有数据源失败，使用过期缓存", code)
            self._set_quality(code, "daily", self._QUALITY_STALE)
            return stale

        logger.error("%s: 所有数据源均失败，无数据可用", code)
        return []

    def get_realtime_quote(self, code: str) -> Optional[RealtimeQuote]:
        """获取实时行情"""
        cached = self._cache.realtime_quote(code)
        if cached is not None:
            self._set_quality(code, "realtime", self._QUALITY_CACHED)
            return cached

        primary = self._fetchers[0].name if self._fetchers else ""
        for fetcher in self._fetchers:
            try:
                quote = fetcher.get_realtime_quote(code)
                if quote is not None and quote.price > 0:
                    self._cache.set_realtime_quote(code, quote)
                    level = self._QUALITY_LIVE if fetcher.name == primary else self._QUALITY_FALLBACK
                    self._set_quality(code, "realtime", level)
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
        primary = self._fetchers[0].name if self._fetchers else ""
        for fetcher in self._fetchers:
            try:
                flows = fetcher.get_fund_flow(code, days)
                if flows:
                    level = self._QUALITY_LIVE if fetcher.name == primary else self._QUALITY_FALLBACK
                    self._set_quality(code, "fund_flow", level)
                    return flows
            except Exception:
                continue
        return []

    def get_market_snapshot(self) -> list[MarketSnapshot]:
        """获取全市场快照 (按成交额降序 Top 3000)"""
        primary = self._fetchers[0].name if self._fetchers else ""
        for fetcher in self._fetchers:
            try:
                snapshot = fetcher.get_market_snapshot()
                if snapshot:
                    level = self._QUALITY_LIVE if fetcher.name == primary else self._QUALITY_FALLBACK
                    self._set_quality("__market__", "snapshot", level)
                    return snapshot
            except Exception:
                continue
        return []

    def get_news(self, keyword: str, days: int = 3) -> list[dict]:
        """新闻搜索"""
        primary = self._fetchers[0].name if self._fetchers else ""
        for fetcher in self._fetchers:
            try:
                result = fetcher.get_news(keyword, days)
                if result:
                    level = self._QUALITY_LIVE if fetcher.name == primary else self._QUALITY_FALLBACK
                    self._set_quality(keyword, "news", level)
                    return result
            except Exception:
                continue
        return []

    def get_announcements(self, code: str, days: int = 7) -> list[dict]:
        """个股公告"""
        primary = self._fetchers[0].name if self._fetchers else ""
        for fetcher in self._fetchers:
            try:
                result = fetcher.get_announcements(code, days)
                if result:
                    level = self._QUALITY_LIVE if fetcher.name == primary else self._QUALITY_FALLBACK
                    self._set_quality(code, "announcements", level)
                    return result
            except Exception:
                continue
        return []

    # ── ETF 数据 ─────────────────────────────

    def get_etf_spot(self) -> list[ETFSpot]:
        """获取场内 ETF 实时行情"""
        for fetcher in self._fetchers:
            try:
                result = fetcher.get_etf_spot()
                if result:
                    return result
            except Exception:
                continue
        return []

    def get_etf_daily(self, code: str, days: int = 60) -> list[StockDaily]:
        """获取 ETF 日线数据"""
        for fetcher in self._fetchers:
            try:
                result = fetcher.get_etf_daily(code, days)
                if result:
                    return result
            except Exception:
                continue
        return []

    # ── 北向资金 ─────────────────────────────

    def get_northbound_flow(self, days: int = 5) -> list[NorthboundFlow]:
        """获取北向资金净流向 (全市场)"""
        primary = self._fetchers[0].name if self._fetchers else ""
        for fetcher in self._fetchers:
            try:
                result = fetcher.get_northbound_flow(days)
                if result:
                    level = self._QUALITY_LIVE if fetcher.name == primary else self._QUALITY_FALLBACK
                    self._set_quality("__market__", "northbound", level)
                    return result
            except Exception:
                continue
        return []

    def get_northbound_stock(self, code: str, days: int = 10) -> list[dict]:
        """获取个股沪深股通持仓变化"""
        primary = self._fetchers[0].name if self._fetchers else ""
        for fetcher in self._fetchers:
            try:
                result = fetcher.get_northbound_stock(code, days)
                if result:
                    level = self._QUALITY_LIVE if fetcher.name == primary else self._QUALITY_FALLBACK
                    self._set_quality(code, "northbound_stock", level)
                    return result
            except Exception:
                continue
        return []

    # ── 融资融券 ─────────────────────────────

    def get_margin_summary(self) -> dict:
        """获取全市场融资融券概况"""
        for fetcher in self._fetchers:
            try:
                result = fetcher.get_margin_summary()
                if result:
                    return result
            except Exception:
                continue
        return {}

    def get_margin_detail(self, code: str, days: int = 10) -> list[MarginData]:
        """获取个股融资融券明细"""
        primary = self._fetchers[0].name if self._fetchers else ""
        for fetcher in self._fetchers:
            try:
                result = fetcher.get_margin_detail(code, days)
                if result:
                    level = self._QUALITY_LIVE if fetcher.name == primary else self._QUALITY_FALLBACK
                    self._set_quality(code, "margin", level)
                    return result
            except Exception:
                continue
        return []

    # ── 深度财务指标 ─────────────────────────

    def get_financial_indicators(self, code: str) -> list[FinancialIndicator]:
        """获取深度财务指标 (多报告期趋势)"""
        primary = self._fetchers[0].name if self._fetchers else ""
        for fetcher in self._fetchers:
            try:
                result = fetcher.get_financial_indicators(code)
                if result:
                    level = self._QUALITY_LIVE if fetcher.name == primary else self._QUALITY_FALLBACK
                    self._set_quality(code, "financials", level)
                    return result
            except Exception:
                continue
        return []

    # ── 财联社电报 ───────────────────────────

    def get_telegraph(self, limit: int = 30) -> list[dict]:
        """获取财联社电报 (实时快讯)"""
        for fetcher in self._fetchers:
            try:
                result = fetcher.get_telegraph(limit)
                if result:
                    return result
            except Exception:
                continue
        return []

    # ── 分析师研报 ───────────────────────────

    def get_research_reports(self, code: str, days: int = 30) -> list[dict]:
        """获取个股分析师研报"""
        for fetcher in self._fetchers:
            try:
                result = fetcher.get_research_reports(code, days)
                if result:
                    return result
            except Exception:
                continue
        return []

    # ── 行业成分股 ───────────────────────────

    def get_industry_stocks(self, industry: str) -> list[str]:
        """获取指定行业的所有成分股代码"""
        for fetcher in self._fetchers:
            try:
                result = fetcher.get_industry_stocks(industry)
                if result:
                    return result
            except Exception:
                continue
        return []

    # ── 限售解禁 ─────────────────────────────

    def get_unlock_shares(self, days_ahead: int = 30) -> list[UnlockShares]:
        """获取近期限售股解禁列表"""
        primary = self._fetchers[0].name if self._fetchers else ""
        for fetcher in self._fetchers:
            try:
                result = fetcher.get_unlock_shares(days_ahead)
                if result:
                    level = self._QUALITY_LIVE if fetcher.name == primary else self._QUALITY_FALLBACK
                    self._set_quality("__market__", "unlock_shares", level)
                    return result
            except Exception:
                continue
        return []

    # ── 股东人数 (筹码集中度) ────────────────

    def get_shareholder_count(self, code: str) -> list[ShareholderCount]:
        """获取股东人数变化趋势"""
        primary = self._fetchers[0].name if self._fetchers else ""
        for fetcher in self._fetchers:
            try:
                result = fetcher.get_shareholder_count(code)
                if result:
                    level = self._QUALITY_LIVE if fetcher.name == primary else self._QUALITY_FALLBACK
                    self._set_quality(code, "shareholder", level)
                    return result
            except Exception:
                continue
        return []

    # ── 机构调研 ──────────────────────────────

    def get_institutional_visits(self, days: int = 30) -> list[InstitutionalVisit]:
        """获取近期机构调研记录"""
        for fetcher in self._fetchers:
            try:
                result = fetcher.get_institutional_visits(days)
                if result:
                    return result
            except Exception:
                continue
        return []

    # ── 市场异动 ──────────────────────────────

    def get_market_activity(self) -> list[MarketActivity]:
        """获取盘口异动"""
        for fetcher in self._fetchers:
            try:
                result = fetcher.get_market_activity()
                if result:
                    return result
            except Exception:
                continue
        return []

    # ── 大宗交易 ──────────────────────────────

    def get_block_trades(self, days: int = 10) -> list[dict]:
        """获取近期大宗交易明细"""
        for fetcher in self._fetchers:
            try:
                result = fetcher.get_block_trades(days)
                if result:
                    return result
            except Exception:
                continue
        return []

    # ── 集合竞价 ──────────────────────────────

    def get_auction_data(self, codes: list[str] | None = None) -> list[dict]:
        """获取集合竞价数据 (开盘前调用)"""
        for fetcher in self._fetchers:
            try:
                result = fetcher.get_auction_data(codes)
                if result:
                    return result
            except Exception:
                continue
        return []

    # ── 涨停板池 ──────────────────────────────

    def get_limit_up_pool(self, date: str = "") -> list[dict]:
        """获取当日涨停板股票池"""
        for fetcher in self._fetchers:
            try:
                result = fetcher.get_limit_up_pool(date)
                if result:
                    return result
            except Exception:
                continue
        return []

    # ── 市场广度 ──────────────────────────────

    def get_market_breadth(self) -> dict:
        """获取全市场广度数据 (涨跌家数/涨停跌停数/成交额)"""
        for fetcher in self._fetchers:
            try:
                result = fetcher.get_market_breadth()
                if result:
                    return result
            except Exception:
                continue
        return {}

    # ── 龙虎榜深化 ────────────────────────────

    def get_dragon_tiger_stats(self, days: int = 10) -> list[dict]:
        """龙虎榜个股上榜统计"""
        for fetcher in self._fetchers:
            try:
                result = fetcher.get_dragon_tiger_stats(days)
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

    def _get_akshare(self) -> AKShareFetcher | None:
        """获取 AKShare fetcher 实例，用于直连数据拉取（跳过优先级链）"""
        for f in self._fetchers:
            if f.name == "akshare":
                return f
        return None

    def batch_fund_flows(
        self, codes: list[str], days: int = 5, max_workers: int = 6
    ) -> dict[str, list[FundFlow]]:
        """并发获取多个股票的资金流向 — 走标准优先级链 (Tushare→BaoStock→AKShare)"""
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

    def batch_market_snapshot(self) -> list[MarketSnapshot]:
        """获取全市场快照 (已存在，别名兼容)"""
        return self.get_market_snapshot()

    def batch_northbound_stocks(
        self, codes: list[str], days: int = 10, max_workers: int = 6
    ) -> dict[str, list[dict]]:
        """并发获取多个股票的北向资金持仓变化 — 走标准优先级链"""
        results: dict[str, list[dict]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self.get_northbound_stock, c, days): c for c in codes}
            for future in as_completed(futures):
                code = futures[future]
                try:
                    results[code] = future.result(timeout=self._config.request_timeout * 2)
                except Exception:
                    results[code] = []
            del futures
        return results

    def batch_financials(
        self, codes: list[str], max_workers: int = 4
    ) -> dict[str, list[FinancialIndicator]]:
        """并发获取多个股票的深度财务指标 — 走标准优先级链"""
        results: dict[str, list[FinancialIndicator]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self.get_financial_indicators, c): c for c in codes}
            for future in as_completed(futures):
                code = futures[future]
                try:
                    results[code] = future.result(timeout=self._config.request_timeout * 3)
                except Exception:
                    results[code] = []
            del futures
        return results

    def batch_shareholders(
        self, codes: list[str], max_workers: int = 6
    ) -> dict[str, list]:
        """并发获取多个股票的股东人数变化 (筹码集中度) — 走标准优先级链"""
        results: dict[str, list[ShareholderCount]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self.get_shareholder_count, c): c for c in codes}
            for future in as_completed(futures):
                code = futures[future]
                try:
                    results[code] = future.result(timeout=self._config.request_timeout * 2)
                except Exception:
                    results[code] = []
            del futures
        return results

    def batch_etf_daily(
        self, codes: list[str], days: int = 20, max_workers: int = 4
    ) -> dict[str, list[StockDaily]]:
        """并发获取多个 ETF 的日线数据"""
        results: dict[str, list[StockDaily]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self.get_etf_daily, c, days): c for c in codes}
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
