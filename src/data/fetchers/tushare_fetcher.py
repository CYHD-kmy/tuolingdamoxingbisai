"""
Tushare 数据适配器 — 增强数据源 (需要 Token)。

特点: 数据质量高、覆盖全，但需要注册 Token (免费)
优先级: Token 可用时自动升到最高优先级

文档: https://tushare.pro
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from .akshare_fetcher import StockDaily, RealtimeQuote, FundFlow, MarketSnapshot
from ...utils.config import get_config

logger = logging.getLogger(__name__)


class TushareFetcher:
    """Tushare 数据适配器。需要 TUSHARE_TOKEN。"""

    name = "tushare"

    def __init__(self) -> None:
        self._api = None
        self._config = get_config()

    @property
    def available(self) -> bool:
        return self._config.tushare_available

    def _get_api(self):
        if self._api is not None:
            return self._api
        if not self.available:
            raise RuntimeError("Tushare token 未配置")
        import tushare as ts
        ts.set_token(self._config.tushare_token)
        self._api = ts.pro_api()
        return self._api

    def get_daily_data(self, code: str, days: int = 60) -> list[StockDaily]:
        if not self.available:
            raise RuntimeError("Tushare 不可用")

        api = self._get_api()
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")
        code_ts = f"{code}.SH" if code.startswith("6") else f"{code}.SZ"

        df = api.daily(ts_code=code_ts, start_date=start, end_date=end)
        if df is None or df.empty:
            return []

        df = df.sort_values("trade_date").tail(days)

        records = []
        for _, row in df.iterrows():
            try:
                records.append(StockDaily(
                    date=str(row["trade_date"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["vol"]),
                    amount=float(row.get("amount", 0) or 0),
                    pct_chg=float(row.get("pct_chg", 0) or 0),
                    # Tushare 日线不直接提供换手率/技术指标，留作补充
                ))
            except (ValueError, KeyError):
                continue
        return records

    def get_realtime_quote(self, code: str) -> Optional[RealtimeQuote]:
        """Tushare 的实时行情需要 pro 权限，这里用日线末条模拟"""
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
            turnover=0,
            volume_ratio=0,
            source="tushare:daily_last",
        )

    def get_stock_name(self, code: str) -> str:
        try:
            api = self._get_api()
            code_ts = f"{code}.SH" if code.startswith("6") else (f"{code}.BJ" if code.startswith(("4", "8")) else f"{code}.SZ")
            df = api.stock_basic(ts_code=code_ts, fields="name")
            if df is not None and not df.empty:
                return str(df.iloc[0]["name"])
        except Exception:
            pass
        return code

    def get_stock_info(self, code: str) -> dict:
        try:
            api = self._get_api()
            code_ts = f"{code}.SH" if code.startswith("6") else (f"{code}.BJ" if code.startswith(("4", "8")) else f"{code}.SZ")
            df = api.stock_basic(ts_code=code_ts, fields="ts_code,name,industry,list_date")
            if df is not None and not df.empty:
                r = df.iloc[0]
                return {
                    "code": code,
                    "name": str(r.get("name", "")),
                    "industry": str(r.get("industry", "")),
                    "ipo_date": str(r.get("list_date", "")),
                }
        except Exception:
            pass
        return {}

    def get_fund_flow(self, code: str, days: int = 5) -> list[FundFlow]:
        """Tushare 资金流向需要较高权限，返回空"""
        return []

    def get_market_snapshot(self) -> list[MarketSnapshot]:
        """Tushare 不支持全市场快照，返回空"""
        return []

    def get_news(self, keyword: str, days: int = 3) -> list[dict]:
        """Tushare 不支持新闻搜索"""
        return []

    def get_announcements(self, code: str, days: int = 7) -> list[dict]:
        """Tushare 不支持公告查询"""
        return []

    # ── 以下方法 Tushare 暂不支持，返回空 ──────

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
