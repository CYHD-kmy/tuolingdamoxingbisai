"""
Tushare 数据适配器 — 增强数据源 (需要 Token)。

特点: 数据质量高、覆盖全，但需要注册 Token (免费)
优先级: Token 可用时自动升到最高优先级

文档: https://tushare.pro
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

from .akshare_fetcher import (
    StockDaily, RealtimeQuote, FundFlow, MarketSnapshot,
    NorthboundFlow, MarginData, FinancialIndicator, ETFSpot,
    UnlockShares, ShareholderCount, InstitutionalVisit, MarketActivity,
)
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

    @staticmethod
    def _ts_code(code: str) -> str:
        """标准化代码 → Tushare ts_code 格式 (600519.SH / 000858.SZ / 430047.BJ)"""
        if code.startswith("6"):
            return f"{code}.SH"
        if code.startswith(("4", "8")):
            return f"{code}.BJ"
        return f"{code}.SZ"

    def _sleep(self) -> None:
        """Tushare 免费版有频率限制，调用间稍作等待"""
        time.sleep(0.15)

    # ── 基础数据 ────────────────────────────────

    def get_daily_data(self, code: str, days: int = 60) -> list[StockDaily]:
        if not self.available:
            raise RuntimeError("Tushare 不可用")

        api = self._get_api()
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")
        code_ts = self._ts_code(code)

        self._sleep()
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
            pre_close=round(d.close / (1 + d.pct_chg / 100), 2) if d.pct_chg > -100 else d.close,
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
            code_ts = self._ts_code(code)
            self._sleep()
            df = api.stock_basic(ts_code=code_ts, fields="name")
            if df is not None and not df.empty:
                return str(df.iloc[0]["name"])
        except Exception:
            pass
        return code

    def get_stock_info(self, code: str) -> dict:
        try:
            api = self._get_api()
            code_ts = self._ts_code(code)
            self._sleep()
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

    def get_market_snapshot(self) -> list[MarketSnapshot]:
        """Tushare 不支持全市场快照，返回空"""
        return []

    def get_news(self, keyword: str, days: int = 3) -> list[dict]:
        """Tushare 不支持新闻搜索"""
        return []

    def get_announcements(self, code: str, days: int = 7) -> list[dict]:
        """Tushare 的公告接口(disclosure)不稳定，返回空交给 AKShare"""
        return []

    # ── 资金流向 ─────────────────────────────────

    def get_fund_flow(self, code: str, days: int = 5) -> list[FundFlow]:
        """个股资金流向 (moneyflow 接口, 免费版可获取最近约30个交易日)"""
        if not self.available:
            return []
        try:
            api = self._get_api()
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=days * 3)).strftime("%Y%m%d")
            code_ts = self._ts_code(code)
            self._sleep()
            df = api.moneyflow(ts_code=code_ts, start_date=start, end_date=end)
            if df is None or df.empty:
                return []
            df = df.sort_values("trade_date").tail(days)
            results = []
            for _, r in df.iterrows():
                try:
                    results.append(FundFlow(
                        date=str(r["trade_date"]),
                        main_net_inflow=float(r.get("buy_elg_vol", 0) or 0)
                                      + float(r.get("buy_lg_vol", 0) or 0)
                                      - float(r.get("sell_elg_vol", 0) or 0)
                                      - float(r.get("sell_lg_vol", 0) or 0),
                        super_large_net=float(r.get("buy_elg_vol", 0) or 0)
                                      - float(r.get("sell_elg_vol", 0) or 0),
                        large_net=float(r.get("buy_lg_vol", 0) or 0)
                                 - float(r.get("sell_lg_vol", 0) or 0),
                        medium_net=float(r.get("buy_md_vol", 0) or 0)
                                  - float(r.get("sell_md_vol", 0) or 0),
                        small_net=float(r.get("buy_sm_vol", 0) or 0)
                                 - float(r.get("sell_sm_vol", 0) or 0),
                        main_pct=0.0,
                    ))
                except (ValueError, KeyError):
                    continue
            return results
        except Exception:
            logger.debug("tushare: get_fund_flow 失败 %s", code, exc_info=True)
            return []

    # ── 北向资金 ─────────────────────────────────

    def get_northbound_flow(self, days: int = 5) -> list[NorthboundFlow]:
        """北向资金净流向 (moneyflow_hsgt 接口)"""
        if not self.available:
            return []
        try:
            api = self._get_api()
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")
            self._sleep()
            df = api.moneyflow_hsgt(start_date=start, end_date=end)
            if df is None or df.empty:
                return []
            df = df.sort_values("trade_date").tail(days)
            results = []
            for _, r in df.iterrows():
                try:
                    results.append(NorthboundFlow(
                        date=str(r["trade_date"]),
                        net_inflow=float(r.get("north_money", 0) or 0),
                        sh_inflow=float(r.get("ggt_ss", 0) or 0),
                        sz_inflow=float(r.get("ggt_sz", 0) or 0),
                    ))
                except (ValueError, KeyError):
                    continue
            return results
        except Exception:
            logger.debug("tushare: get_northbound_flow 失败", exc_info=True)
            return []

    def get_northbound_stock(self, code: str, days: int = 10) -> list[dict]:
        """个股沪深股通持仓变化 (hk_hold 接口)"""
        if not self.available:
            return []
        try:
            api = self._get_api()
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")
            code_ts = self._ts_code(code)
            self._sleep()
            df = api.hk_hold(ts_code=code_ts, start_date=start, end_date=end)
            if df is None or df.empty:
                return []
            df = df.sort_values("trade_date").tail(days)
            result = []
            for _, r in df.iterrows():
                try:
                    result.append({
                        "date": str(r["trade_date"]),
                        "hold_shares": float(r.get("vol", 0) or 0),
                        "hold_pct": float(r.get("ratio", 0) or 0),
                    })
                except (ValueError, KeyError):
                    continue
            return result
        except Exception:
            logger.debug("tushare: get_northbound_stock 失败 %s", code, exc_info=True)
            return []

    # ── 融资融券 ─────────────────────────────────

    def get_margin_summary(self) -> dict:
        """全市场融资融券概况 (margin 接口)"""
        if not self.available:
            return {}
        try:
            api = self._get_api()
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=5)).strftime("%Y%m%d")
            self._sleep()
            df = api.margin(start_date=start, end_date=end)
            if df is None or df.empty:
                return {}
            latest = df.iloc[-1]
            return {
                "date": str(latest.get("trade_date", "")),
                "sh_margin_balance": float(latest.get("rzye", 0) or 0),
                "sh_margin_buy": float(latest.get("rzmre", 0) or 0),
                "total_margin_balance": float(latest.get("rzye", 0) or 0),
            }
        except Exception:
            logger.debug("tushare: get_margin_summary 失败", exc_info=True)
            return {}

    def get_margin_detail(self, code: str, days: int = 10) -> list[MarginData]:
        """个股融资融券明细 (margin_detail 接口)"""
        if not self.available:
            return []
        try:
            api = self._get_api()
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")
            code_ts = self._ts_code(code)
            self._sleep()
            df = api.margin_detail(ts_code=code_ts, start_date=start, end_date=end)
            if df is None or df.empty:
                return []
            df = df.sort_values("trade_date").tail(days)
            results = []
            for _, r in df.iterrows():
                try:
                    results.append(MarginData(
                        date=str(r["trade_date"]),
                        margin_balance=float(r.get("rzye", 0) or 0) / 1e4,
                        margin_buy=float(r.get("rzmre", 0) or 0) / 1e4,
                        short_balance=float(r.get("rqye", 0) or 0) / 1e4,
                    ))
                except (ValueError, KeyError):
                    continue
            return results
        except Exception:
            logger.debug("tushare: get_margin_detail 失败 %s", code, exc_info=True)
            return []

    # ── 深度财务指标 ─────────────────────────────

    def get_financial_indicators(self, code: str) -> list[FinancialIndicator]:
        """深度财务指标 (fina_indicator + income 组合)"""
        if not self.available:
            return []
        try:
            api = self._get_api()
            code_ts = self._ts_code(code)
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=1000)).strftime("%Y%m%d")

            # 1. 财务指标
            self._sleep()
            df_fina = api.fina_indicator(ts_code=code_ts, start_date=start, end_date=end)

            # 2. 利润表 (营收增速)
            self._sleep()
            df_income = api.income(ts_code=code_ts, start_date=start, end_date=end,
                                   fields="ts_code,end_date,revenue,operate_profit,n_profit")

            if df_fina is None or df_fina.empty:
                return []

            df_fina = df_fina.sort_values("end_date")
            income_map: dict[str, dict] = {}
            if df_income is not None and not df_income.empty:
                df_income = df_income.sort_values("end_date")
                for _, r in df_income.iterrows():
                    income_map[str(r["end_date"])[:7]] = {
                        "revenue": float(r.get("revenue", 0) or 0),
                        "profit": float(r.get("n_profit", 0) or 0),
                    }

            results = []
            prev_revenue = None
            prev_profit = None
            for _, r in df_fina.tail(8).iterrows():
                try:
                    end_dt = str(r["end_date"])
                    period_key = end_dt[:7]

                    inc = income_map.get(period_key, {})
                    revenue = inc.get("revenue", 0)
                    profit = inc.get("profit", 0)

                    # 计算同比增速
                    revenue_yoy = 0.0
                    profit_yoy = 0.0
                    if prev_revenue and prev_revenue > 0 and revenue > 0:
                        revenue_yoy = round((revenue / prev_revenue - 1) * 100, 2)
                    if prev_profit and prev_profit > 0 and profit > 0:
                        profit_yoy = round((profit / prev_profit - 1) * 100, 2)

                    results.append(FinancialIndicator(
                        date=end_dt,
                        roe=float(r.get("roe", 0) or 0),
                        roa=float(r.get("roa", 0) or 0),
                        gross_margin=float(r.get("grossprofit_margin", 0) or 0),
                        net_margin=float(r.get("netprofit_margin", 0) or 0),
                        revenue_yoy=revenue_yoy,
                        profit_yoy=profit_yoy,
                        debt_ratio=float(r.get("debt_to_assets", 0) or 0),
                        eps=float(r.get("eps", 0) or 0),
                        current_ratio=float(r.get("current_ratio", 0) or 0),
                        quick_ratio=float(r.get("quick_ratio", 0) or 0),
                        cf_operating=float(r.get("ocfps", 0) or 0),
                    ))
                    prev_revenue = revenue
                    prev_profit = profit
                except (ValueError, KeyError):
                    continue
            return results
        except Exception:
            logger.debug("tushare: get_financial_indicators 失败 %s", code, exc_info=True)
            return []

    # ── 分析师研报 ───────────────────────────────

    def get_research_reports(self, code: str, days: int = 30) -> list[dict]:
        """个股分析师研报 (broker_recommend 接口)"""
        if not self.available:
            return []
        try:
            api = self._get_api()
            code_ts = self._ts_code(code)
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
            self._sleep()
            df = api.broker_recommend(ts_code=code_ts, start_date=start, end_date=end)
            if df is None or df.empty:
                return []
            return [
                {
                    "org": str(r.get("broker", "")),
                    "rating": str(r.get("recommend", "")),
                    "date": str(r.get("trade_date", r.get("ann_date", "")))[:10],
                }
                for _, r in df.head(15).iterrows()
            ]
        except Exception:
            logger.debug("tushare: get_research_reports 失败 %s", code, exc_info=True)
            return []

    # ── 限售解禁 ─────────────────────────────────

    def get_unlock_shares(self, days_ahead: int = 30) -> list[UnlockShares]:
        """近期限售股解禁 (share_float 接口)"""
        if not self.available:
            return []
        try:
            api = self._get_api()
            end = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y%m%d")
            start = datetime.now().strftime("%Y%m%d")
            self._sleep()
            df = api.share_float(ann_date="", start_date=start, end_date=end)
            if df is None or df.empty:
                return []
            results = []
            for _, r in df.head(50).iterrows():
                try:
                    code_raw = str(r.get("ts_code", ""))
                    code_clean = code_raw.split(".")[0] if "." in code_raw else code_raw
                    results.append(UnlockShares(
                        code=code_clean,
                        name=str(r.get("ts_code", "")),
                        unlock_date=str(r.get("float_date", r.get("ann_date", "")))[:10],
                        unlock_shares=float(r.get("float_share", 0) or 0),
                        unlock_value=0.0,
                        unlock_ratio=float(r.get("float_ratio", 0) or 0),
                    ))
                except (ValueError, KeyError):
                    continue
            return results
        except Exception:
            logger.debug("tushare: get_unlock_shares 失败", exc_info=True)
            return []

    # ── 股东人数 (筹码集中度) ────────────────────

    def get_shareholder_count(self, code: str) -> list[ShareholderCount]:
        """股东人数变化趋势 (stk_holdernumber 接口)"""
        if not self.available:
            return []
        try:
            api = self._get_api()
            code_ts = self._ts_code(code)
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=400)).strftime("%Y%m%d")
            self._sleep()
            df = api.stk_holdernumber(ts_code=code_ts, start_date=start, end_date=end)
            if df is None or df.empty:
                return []
            df = df.sort_values("end_date").tail(6)
            results = []
            prev_count = 0
            for _, r in df.iterrows():
                try:
                    count = int(float(r.get("holder_num", 0) or 0))
                    change_pct = 0.0
                    if prev_count > 0:
                        change_pct = round((count / prev_count - 1) * 100, 2)
                    results.append(ShareholderCount(
                        date=str(r["end_date"]),
                        holder_count=count,
                        change_pct=change_pct,
                    ))
                    prev_count = count
                except (ValueError, KeyError):
                    continue
            return results
        except Exception:
            logger.debug("tushare: get_shareholder_count 失败 %s", code, exc_info=True)
            return []

    # ── ETF ───────────────────────────────────────

    def get_etf_spot(self) -> list[ETFSpot]:
        """Tushare 免费版不直接支持 ETF 实时行情，返回空"""
        return []

    def get_etf_daily(self, code: str, days: int = 60) -> list[StockDaily]:
        """ETF 日线 (fund_daily 接口)"""
        if not self.available:
            return []
        try:
            api = self._get_api()
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")
            code_ts = f"{code}.SH" if code.startswith("5") else f"{code}.SZ"
            self._sleep()
            df = api.fund_daily(ts_code=code_ts, start_date=start, end_date=end)
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
                    ))
                except (ValueError, KeyError):
                    continue
            return records
        except Exception:
            logger.debug("tushare: get_etf_daily 失败 %s", code, exc_info=True)
            return []

    # ── 以下仍不支持 ──────────────────────────────

    def get_telegraph(self, limit: int = 30) -> list[dict]:
        return []

    def get_industry_stocks(self, industry: str) -> list[str]:
        return []

    def get_institutional_visits(self, days: int = 30) -> list[InstitutionalVisit]:
        return []

    def get_market_activity(self) -> list[MarketActivity]:
        return []

    def get_block_trades(self, days: int = 10) -> list[dict]:
        return []

    def get_dragon_tiger_stats(self, days: int = 10) -> list[dict]:
        return []
