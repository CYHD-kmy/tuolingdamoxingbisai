"""
BaoStock 数据适配器 — 兜底免费数据源。

特点: 完全免费、无需Token、稳定可靠
局限: 数据更新有延迟(通常T+1)、无实时行情、无资金流向

文档: http://baostock.com
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Optional

from .akshare_fetcher import StockDaily, RealtimeQuote

logger = logging.getLogger(__name__)


class BaoStockFetcher:
    """BaoStock 数据适配器。完全免费，无需 Token。"""

    name = "baostock"

    _logged_in = False
    _login_attempted = False
    _stock_basic_cache: dict[str, dict] | None = None  # code → {name, ipo_date}
    _stock_basic_cache_time: float = 0.0
    _stock_basic_lock: threading.Lock = threading.Lock()

    @classmethod
    def _warm_stock_basic_cache(cls):
        """预热全市场股票基本信息缓存 (线程安全，一次查询，全量缓存)"""
        with cls._stock_basic_lock:
            now = time.time()
            if cls._stock_basic_cache is not None:
                return
            # 最近已尝试过且失败，短时间内不再重试
            if cls._stock_basic_cache_time > 0 and now - cls._stock_basic_cache_time <= 120:
                return
            cls._ensure_login()
            import baostock as bs
            rs = bs.query_stock_basic()
            if rs.error_code != "0":
                logger.warning("baostock: stock_basic 查询失败")
                cls._stock_basic_cache = {}
                cls._stock_basic_cache_time = now
                return
            cache: dict[str, dict] = {}
            while rs.next():
                row = rs.get_row_data()
                code = row[0].replace("sh.", "").replace("sz.", "").replace("bj.", "")
                cache[code] = {"name": row[1], "ipo_date": row[2] if len(row) > 2 else ""}
            cls._stock_basic_cache = cache
            cls._stock_basic_cache_time = now
            logger.info("baostock: 全市场股票基本信息缓存 %d 只", len(cache))

    @classmethod
    def _ensure_login(cls):
        if cls._logged_in:
            return
        if cls._login_attempted:
            return
        import baostock as bs
        for attempt in range(3):
            lg = bs.login()
            if lg.error_code == "0":
                cls._logged_in = True
                return
            if attempt < 2:
                import time
                time.sleep(2 ** attempt)
        cls._login_attempted = True
        logger.warning("baostock 登录失败 (已重试3次): %s", lg.error_msg)

    @classmethod
    def _logout(cls):
        if not cls._logged_in:
            return
        import baostock as bs
        bs.logout()
        cls._logged_in = False


    @staticmethod
    def _bs_code(code: str) -> str:
        """转 BaoStock 格式: sh.600519 / sz.000858"""
        prefix = "sh" if code.startswith(("5", "6", "9")) else ("bj" if code.startswith(("4", "8")) else "sz")
        return f"{prefix}.{code}"

    def get_daily_data(self, code: str, days: int = 60) -> list[StockDaily]:
        self._ensure_login()
        import baostock as bs

        bs_code = self._bs_code(code)
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=days * 2)).strftime("%Y-%m-%d")

        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume,amount,pctChg,turn",
            start_date=start, end_date=end,
            frequency="d", adjustflag="2",  # 前复权
        )

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())

        if not rows:
            return []

        records = []
        for row in rows[-days:]:
            try:
                records.append(StockDaily(
                    date=row[0],
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    amount=float(row[6]) if row[6] else 0,
                    pct_chg=float(row[7]) if row[7] else 0,
                    turnover=float(row[8]) if row[8] else 0,
                ))
            except (ValueError, IndexError):
                continue
        return records

    def get_realtime_quote(self, code: str) -> Optional[RealtimeQuote]:
        """BaoStock 不提供实时行情，用最近日线模拟"""
        daily = self.get_daily_data(code, days=1)
        if not daily:
            return None
        d = daily[-1]
        return RealtimeQuote(
            code=code,
            name=self.get_stock_name(code),
            price=d.close,
            open=d.open,
            high=d.high,
            low=d.low,
            pre_close=d.close - d.close * d.pct_chg / (100 + d.pct_chg) if abs(d.pct_chg) < 99 else d.close * 0.5,
            pct_chg=d.pct_chg,
            volume=d.volume,
            amount=d.amount,
            turnover=d.turnover,
            volume_ratio=0,
            source="baostock:daily_last",
        )

    def get_stock_name(self, code: str) -> str:
        self._warm_stock_basic_cache()
        info = self._stock_basic_cache.get(code) if self._stock_basic_cache else None
        if info:
            return info.get("name", code)
        return code

    def get_stock_info(self, code: str) -> dict:
        self._warm_stock_basic_cache()
        info = self._stock_basic_cache.get(code) if self._stock_basic_cache else None
        if info:
            return {
                "code": code,
                "name": info.get("name", ""),
                "ipo_date": info.get("ipo_date", ""),
            }
        return {}

    def get_fund_flow(self, code: str, days: int = 5) -> list:
        """BaoStock 不支持资金流向"""
        return []

    def get_market_snapshot(self) -> list:
        """BaoStock 不支持全市场快照"""
        return []

    def get_news(self, keyword: str, days: int = 3) -> list[dict]:
        """BaoStock 不支持新闻搜索"""
        return []

    def get_announcements(self, code: str, days: int = 7) -> list[dict]:
        """BaoStock 不支持公告查询"""
        return []

    # ── 以下方法 BaoStock 暂不支持，返回空 ──────

    def get_etf_spot(self) -> list:          return []
    def get_etf_daily(self, code: str, days: int = 60) -> list:  return []
    def get_northbound_flow(self, days: int = 5) -> list:       return []
    def get_northbound_stock(self, code: str, days: int = 10) -> list[dict]:  return []
    def get_margin_summary(self) -> dict:                        return {}
    def get_margin_detail(self, code: str, days: int = 10) -> list:  return []
    def get_financial_indicators(self, code: str) -> list:       return []
    def get_telegraph(self, limit: int = 30) -> list[dict]:      return []
    def get_research_reports(self, code: str, days: int = 30) -> list[dict]:  return []
    def get_industry_stocks(self, industry: str) -> list[str]:    return []
    def get_unlock_shares(self, days_ahead: int = 30) -> list:   return []
    def get_shareholder_count(self, code: str) -> list:           return []
    def get_institutional_visits(self, days: int = 30) -> list:   return []
    def get_market_activity(self) -> list:                        return []
    def get_block_trades(self, days: int = 10) -> list[dict]:     return []
    def get_dragon_tiger_stats(self, days: int = 10) -> list[dict]:  return []


import atexit
atexit.register(BaoStockFetcher._logout)
