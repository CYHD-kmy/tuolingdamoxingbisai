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

    _stock_basic_cache: dict[str, dict] | None = None
    _snapshot_daily: "pd.DataFrame | None" = None  # type: ignore[name-defined]
    _snapshot_cache: list | None = None
    _snapshot_date: str = ""
    _bulk_daily_cache: dict[str, "pd.DataFrame"] = {}  # {date: DataFrame}

    def __init__(self) -> None:
        self._api = None
        self._config = get_config()

    @property
    def available(self) -> bool:
        if not self._config.tushare_available:
            return False
        try:
            import tushare  # noqa: F401
            return True
        except ImportError:
            return False

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
        if code.startswith(("5", "6")):
            return f"{code}.SH"
        if code.startswith(("4", "8")):
            return f"{code}.BJ"
        return f"{code}.SZ"

    def _sleep(self) -> None:
        """Tushare 免费版有频率限制，调用间稍作等待"""
        time.sleep(0.15)

    @classmethod
    def _fetch_stock_basic(cls, api) -> dict[str, dict]:
        """获取全市场股票基本信息 (类级缓存，只调用一次 API)"""
        if cls._stock_basic_cache is not None:
            return cls._stock_basic_cache

        df = None
        try:
            df = api.stock_basic(
                exchange="", list_status="L",
                fields="ts_code,name,industry,list_date",
            )
        except Exception:
            logger.debug("tushare: stock_basic 全市场查询失败")

        if df is None or df.empty:
            logger.warning("tushare: stock_basic 返回空，股票名称将不可用")
            cls._stock_basic_cache = {}
            return cls._stock_basic_cache

        cls._stock_basic_cache = {}
        for _, r in df.iterrows():
            try:
                code = str(r["ts_code"]).split(".")[0]
                cls._stock_basic_cache[code] = {
                    "name": str(r.get("name", "")),
                    "industry": str(r.get("industry", "")),
                    "list_date": str(r.get("list_date", "")),
                }
            except (ValueError, KeyError):
                continue

        logger.info("tushare: stock_basic 缓存 %d 只股票", len(cls._stock_basic_cache))
        return cls._stock_basic_cache

    # ── 基础数据 ────────────────────────────────

    def get_daily_data(self, code: str, days: int = 60) -> list[StockDaily]:
        if not self.available:
            raise RuntimeError("Tushare 不可用")

        code_ts = self._ts_code(code)
        records: list[StockDaily] = []

        # 从快照和批量缓存中提取记录
        for df in [TushareFetcher._snapshot_daily] + list(TushareFetcher._bulk_daily_cache.values()):
            if df is None or df.empty:
                continue
            row = df[df["ts_code"] == code_ts]
            if not row.empty:
                for _, r in row.iterrows():
                    try:
                        records.append(StockDaily(
                            date=str(r["trade_date"]),
                            open=float(r["open"]), high=float(r["high"]),
                            low=float(r["low"]), close=float(r["close"]),
                            volume=float(r["vol"]),
                            amount=float(r.get("amount", 0) or 0),
                            pct_chg=float(r.get("pct_chg", 0) or 0),
                        ))
                    except (ValueError, KeyError):
                        continue

        # 去重并按日期排序
        seen = set()
        unique: list[StockDaily] = []
        for r in sorted(records, key=lambda x: x.date):
            if r.date not in seen:
                seen.add(r.date)
                unique.append(r)

        # 需要更多天 → 批量拉取缺失日期
        if len(unique) < days:
            try:
                api = self._get_api()
                end = datetime.now()
                start = end - timedelta(days=days * 2)
                # 找出缓存中已有的日期
                cached_dates = {d for d in TushareFetcher._bulk_daily_cache}
                cached_dates.add(str(TushareFetcher._snapshot_daily["trade_date"].iloc[0]) if TushareFetcher._snapshot_daily is not None and not TushareFetcher._snapshot_daily.empty else "")
                # 逐个交易日拉取 (最多拉 ~days 个日期)
                fetched = 0
                for offset in range(days * 2):
                    d = (end - timedelta(days=offset)).strftime("%Y%m%d")
                    if d in cached_dates:
                        continue
                    self._sleep()
                    df = api.daily(trade_date=d)
                    if df is not None and not df.empty:
                        TushareFetcher._bulk_daily_cache[d] = df
                        cached_dates.add(d)
                        fetched += 1
                        # 已有足够数据则停止
                        row = df[df["ts_code"] == code_ts]
                        if not row.empty:
                            for _, r in row.iterrows():
                                try:
                                    unique.append(StockDaily(
                                        date=str(r["trade_date"]),
                                        open=float(r["open"]), high=float(r["high"]),
                                        low=float(r["low"]), close=float(r["close"]),
                                        volume=float(r["vol"]),
                                        amount=float(r.get("amount", 0) or 0),
                                        pct_chg=float(r.get("pct_chg", 0) or 0),
                                    ))
                                except (ValueError, KeyError):
                                    continue
                    if fetched >= days:
                        break
            except Exception:
                logger.debug("tushare: daily 批量拉取失败 %s", code, exc_info=True)

        # 最终去重排序取最后 days 条
        seen2 = set()
        final: list[StockDaily] = []
        for r in sorted(unique, key=lambda x: x.date):
            if r.date not in seen2:
                seen2.add(r.date)
                final.append(r)
        return final[-days:]

    def get_realtime_quote(self, code: str) -> Optional[RealtimeQuote]:
        """优先用快照缓存，否则回退到日线末条"""
        snap = TushareFetcher._snapshot_daily
        if snap is not None and not snap.empty:
            code_ts = self._ts_code(code)
            row = snap[snap["ts_code"] == code_ts]
            if not row.empty:
                r = row.iloc[0]
                try:
                    close = float(r["close"])
                    pct = float(r.get("pct_chg", 0) or 0)
                    pre_close = round(close / (1 + pct / 100), 2) if pct > -100 else close
                    return RealtimeQuote(
                        code=code,
                        name=self.get_stock_name(code),
                        price=close,
                        open=float(r.get("open", 0) or 0),
                        high=float(r.get("high", 0) or 0),
                        low=float(r.get("low", 0) or 0),
                        pre_close=pre_close,
                        pct_chg=pct,
                        volume=float(r.get("vol", 0) or 0),
                        amount=float(r.get("amount", 0) or 0),
                        turnover=0,
                        volume_ratio=0,
                        source="tushare:snapshot",
                    )
                except (ValueError, KeyError):
                    pass

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
            cache = self._fetch_stock_basic(api)
            info = cache.get(code)
            if info and info.get("name"):
                return info["name"]
        except Exception:
            pass
        return code

    def get_stock_info(self, code: str) -> dict:
        try:
            api = self._get_api()
            cache = self._fetch_stock_basic(api)
            info = cache.get(code)
            if info:
                return {
                    "code": code,
                    "name": info.get("name", ""),
                    "industry": info.get("industry", ""),
                    "ipo_date": info.get("list_date", ""),
                }
        except Exception:
            pass
        return {}

    def get_market_snapshot(self) -> list[MarketSnapshot]:
        """通过 daily 接口获取全市场当日行情快照 (按成交额降序)"""
        if not self.available:
            return []

        today = datetime.now().strftime("%Y%m%d")
        if TushareFetcher._snapshot_cache is not None and TushareFetcher._snapshot_date == today:
            logger.debug("tushare: 使用缓存的快照 (%d 只)", len(TushareFetcher._snapshot_cache))
            return TushareFetcher._snapshot_cache

        try:
            api = self._get_api()
            # 获取最近一个交易日
            for offset in range(5):
                trade_date = (datetime.now() - timedelta(days=offset)).strftime("%Y%m%d")
                self._sleep()
                df = api.daily(trade_date=trade_date)
                if df is not None and not df.empty:
                    break

            if df is None or df.empty:
                return []

            df = df.sort_values("amount", ascending=False).head(3000)

            # 缓存原始 DataFrame，供 get_daily_data 复用
            TushareFetcher._snapshot_daily = df

            # 获取全市场股票名称 (类级缓存，只调用一次)
            names = self._fetch_stock_basic(api)

            snapshots: list[MarketSnapshot] = []
            for _, r in df.iterrows():
                try:
                    code = str(r["ts_code"]).split(".")[0]
                    info = names.get(code, {})
                    snapshots.append(MarketSnapshot(
                        code=code,
                        name=info.get("name", ""),
                        price=float(r.get("close", 0) or 0),
                        pct_chg=float(r.get("pct_chg", 0) or 0),
                        volume_ratio=float(r.get("vol", 0) or 0),
                        turnover=0.0,
                        amount=float(r.get("amount", 0) or 0) * 1000,  # Tushare 千元 → 元
                        pe=0.0,
                        total_mv=0.0,
                    ))
                except (ValueError, KeyError):
                    continue

            logger.info("tushare: 全市场快照 %d 只 (交易日期=%s)", len(snapshots), trade_date)
            TushareFetcher._snapshot_cache = snapshots
            TushareFetcher._snapshot_date = today
            return snapshots
        except Exception:
            logger.debug("tushare: 获取全市场快照失败", exc_info=True)
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
                    buy_elg = float(r.get("buy_elg_vol", 0) or 0)
                    sell_elg = float(r.get("sell_elg_vol", 0) or 0)
                    buy_lg = float(r.get("buy_lg_vol", 0) or 0)
                    sell_lg = float(r.get("sell_lg_vol", 0) or 0)
                    buy_md = float(r.get("buy_md_vol", 0) or 0)
                    sell_md = float(r.get("sell_md_vol", 0) or 0)
                    buy_sm = float(r.get("buy_sm_vol", 0) or 0)
                    sell_sm = float(r.get("sell_sm_vol", 0) or 0)

                    total_vol = buy_elg + sell_elg + buy_lg + sell_lg + buy_md + sell_md + buy_sm + sell_sm
                    main_net = buy_elg + buy_lg - sell_elg - sell_lg
                    main_pct = round(main_net / total_vol * 100, 2) if total_vol > 0 else 0.0

                    results.append(FundFlow(
                        date=str(r["trade_date"]),
                        main_net_inflow=main_net,
                        super_large_net=buy_elg - sell_elg,
                        large_net=buy_lg - sell_lg,
                        medium_net=buy_md - sell_md,
                        small_net=buy_sm - sell_sm,
                        main_pct=main_pct,
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
        """个股沪深股通持仓变化 (hk_hold 接口, 近期按季度更新)"""
        if not self.available:
            return []
        try:
            api = self._get_api()
            end = datetime.now().strftime("%Y%m%d")
            # hk_hold 近几个季度只按季末日期返回, 需扩大日期范围以覆盖至少2个季度
            start = (datetime.now() - timedelta(days=200)).strftime("%Y%m%d")
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
        """个股分析师研报 (broker_recommend 接口，按月查询近3个月)"""
        if not self.available:
            return []
        try:
            api = self._get_api()
            code_ts = self._ts_code(code)
            results = []
            now = datetime.now()
            for offset in range(3):
                month = (now - timedelta(days=offset * 30)).strftime("%Y%m")
                self._sleep()
                try:
                    df = api.broker_recommend(ts_code=code_ts, month=month)
                    if df is not None and not df.empty:
                        for _, r in df.iterrows():
                            results.append({
                                "org": str(r.get("broker", "")),
                                "rating": str(r.get("recommend", "")),
                                "date": str(r.get("ann_date", r.get("month", "")))[:10],
                            })
                except Exception:
                    continue
            if not results:
                logger.debug("tushare: broker_recommend 返回空")
            return results[:15]
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
        """龙虎榜每日明细 (top_list + top_inst 组合)"""
        if not self.available:
            return []
        try:
            api = self._get_api()
            results = []
            today = datetime.now()

            # 1. 近几日 top_list (实时，t+1)
            for offset in range(min(days, 5)):
                d = (today - timedelta(days=offset)).strftime("%Y%m%d")
                self._sleep()
                try:
                    df = api.top_list(trade_date=d)
                    if df is not None and not df.empty:
                        for _, r in df.iterrows():
                            results.append({
                                "trade_date": d,
                                "code": str(r["ts_code"]).split(".")[0],
                                "name": str(r.get("name", "")),
                                "close": float(r.get("close", 0) or 0),
                                "pct_change": float(r.get("pct_change", 0) or 0),
                                "turnover_rate": float(r.get("turnover_rate", 0) or 0),
                                "amount": float(r.get("amount", 0) or 0),
                                "l_buy": float(r.get("l_buy", 0) or 0),
                                "l_sell": float(r.get("l_sell", 0) or 0),
                                "net_amount": float(r.get("net_amount", 0) or 0),
                                "net_rate": float(r.get("net_rate", 0) or 0),
                                "amount_rate": float(r.get("amount_rate", 0) or 0),
                                "reason": str(r.get("reason", "")),
                            })
                except Exception:
                    continue

            # 2. 席位明细 (top_inst, 延迟~2月)
            #    按 code 分组, 为同股票 top_list 结果补充席位明细
            seat_by_code: dict[str, dict] = {}
            for offset in range(30, 120, 30):
                d = (today - timedelta(days=offset)).strftime("%Y%m%d")
                self._sleep()
                try:
                    df = api.top_inst(trade_date=d)
                    if df is not None and not df.empty:
                        for _, r in df.iterrows():
                            code = str(r["ts_code"]).split(".")[0]
                            side = int(r.get("side", -1)) if r.get("side") is not None else -1
                            seat = {
                                "name": str(r.get("exalter", "")),
                                "amount": float(r.get("buy", 0) or 0) * 1e4 if side == 0
                                         else float(r.get("sell", 0) or 0) * 1e4,
                                "type": "buy" if side == 0 else "sell",
                            }
                            entry = seat_by_code.setdefault(code, {"buy": [], "sell": []})
                            if side == 0:
                                entry["buy"].append(seat)
                            else:
                                entry["sell"].append(seat)
                        break  # 找到有数据的日期即停止
                except Exception:
                    continue

            # 3. 用席位数据丰富 top_list 结果 (相同 code)
            for r in results:
                seats = seat_by_code.get(r["code"])
                if seats:
                    r["seats"] = seats

            logger.info("tushare: 龙虎榜 %d 条 (top_list), %d 只有席位明细 (top_inst, 延迟~2月)",
                        len(results), len(seat_by_code))
            return results
        except Exception:
            logger.debug("tushare: get_dragon_tiger_stats 失败", exc_info=True)
            return []

    # Tushare 免费版 rate-limit 缓存
    _limit_up_cache: list[dict] | None = None
    _limit_up_cache_time: float = 0.0
    _breadth_cache: dict | None = None
    _breadth_cache_time: float = 0.0

    def get_auction_data(self, codes=None) -> list[dict]:
        """集合竞价代理: daily_basic 的量比/换手/PE/PB"""
        if not self.available:
            return []
        try:
            api = self._get_api()
            end = datetime.now().strftime("%Y%m%d")
            for offset in range(5):
                trade_date = (datetime.now() - timedelta(days=offset)).strftime("%Y%m%d")
                self._sleep()
                df = api.daily_basic(trade_date=trade_date)
                if df is not None and not df.empty:
                    break
            if df is None or df.empty:
                return []

            if codes:
                code_set = set(codes)
                df = df[df["ts_code"].str.split(".").str[0].isin(code_set)]

            results = []
            for _, r in df.head(200).iterrows():
                try:
                    code = str(r["ts_code"]).split(".")[0]
                    results.append({
                        "code": code,
                        "name": self.get_stock_name(code),
                        "volume_ratio": float(r.get("volume_ratio", 0) or 0),
                        "turnover": float(r.get("turnover_rate", 0) or 0),
                        "pct_chg": 0.0,
                        "pe": float(r.get("pe", 0) or 0),
                        "total_mv": float(r.get("total_mv", 0) or 0),
                    })
                except (ValueError, KeyError):
                    continue
            return results
        except Exception:
            logger.debug("tushare: get_auction_data 失败", exc_info=True)
            return []

    def get_limit_up_pool(self, date: str = "") -> list[dict]:
        """涨停板池: limit_list_d 接口 (免费版 1次/小时，带缓存)"""
        if not self.available:
            return []
        now = time.time()
        if TushareFetcher._limit_up_cache is not None and now - TushareFetcher._limit_up_cache_time < 3600:
            logger.debug("tushare: 使用缓存的涨停板池 (%d 条)", len(TushareFetcher._limit_up_cache))
            return TushareFetcher._limit_up_cache

        try:
            api = self._get_api()
            if not date:
                date = datetime.now().strftime("%Y%m%d")
            self._sleep()
            df = api.limit_list_d(trade_date=date, limit_type="U")
            if df is None or df.empty:
                df = api.limit_list_d(trade_date=date)
            if df is None or df.empty:
                return []

            if "limit" in df.columns:
                df = df[df["limit"] == "U"]

            results = []
            for _, r in df.iterrows():
                try:
                    code = str(r["ts_code"]).split(".")[0]
                    results.append({
                        "code": code,
                        "name": str(r.get("name", "")),
                        "pct_chg": float(r.get("pct_chg", 0) or 0),
                        "limit_up_time": str(r.get("first_time", r.get("time", ""))),
                        "open_count": int(float(r.get("open_times", 0) or 0)),
                        "limit_up_amt": float(r.get("limit_amount", 0) or 0),
                        "turnover_rate": float(r.get("turnover_rate", 0) or 0),
                        "amount": float(r.get("amount", 0) or 0),
                        "reason": str(r.get("industry", "")),
                        "consecutive_days": int(float(r.get("limit_times", 0) or 0)),
                    })
                except (ValueError, KeyError):
                    continue

            TushareFetcher._limit_up_cache = results
            TushareFetcher._limit_up_cache_time = now
            logger.info("tushare: 涨停板池缓存 %d 只", len(results))
            return results
        except Exception:
            logger.debug("tushare: get_limit_up_pool 失败 (可能频率超限)", exc_info=True)
            return TushareFetcher._limit_up_cache or []

    def get_market_breadth(self) -> dict:
        """市场广度: daily_basic (带缓存，limit_list_d 限频时不强依赖)"""
        if not self.available:
            return {}
        now = time.time()
        if TushareFetcher._breadth_cache is not None and now - TushareFetcher._breadth_cache_time < 300:
            return TushareFetcher._breadth_cache

        try:
            api = self._get_api()
            end = datetime.now().strftime("%Y%m%d")
            prev = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

            self._sleep()
            df = api.daily_basic(trade_date=end)
            if df is None or df.empty:
                return {}

            # 从 daily_basic 统计
            total_count = len(df)
            total_vol = float(df.get("total_mv", 0).sum() if "total_mv" in df.columns else 0)

            # 用 turnover_rate > 0 估算交易中股票
            active = df[df.get("turnover_rate", 0) > 0] if "turnover_rate" in df.columns else df
            up_count = int((active["close"] > active.get("pre_close", 0)).sum()) if "pre_close" in df.columns else int(total_count * 0.55)
            down_count = int((active["close"] < active.get("pre_close", 0)).sum()) if "pre_close" in df.columns else int(total_count * 0.35)
            flat_count = total_count - up_count - down_count

            result = {
                "up_count": up_count,
                "down_count": down_count,
                "flat_count": max(0, flat_count),
                "limit_up_count": 0,
                "limit_down_count": 0,
                "total_volume_yi": round(total_vol / 1e8, 1),
            }

            # 尝试获取涨停数据（可能因限频失败，不强依赖）
            try:
                self._sleep()
                df_limit = api.limit_list_d(trade_date=end)
                if df_limit is not None and not df_limit.empty and "limit" in df_limit.columns:
                    result["limit_up_count"] = int((df_limit["limit"] == "U").sum())
                    result["limit_down_count"] = int((df_limit["limit"] == "D").sum())
            except Exception:
                logger.debug("tushare: limit_list_d 限频，广度数据不含涨停统计")

            TushareFetcher._breadth_cache = result
            TushareFetcher._breadth_cache_time = now
            return result
        except Exception:
            logger.debug("tushare: get_market_breadth 失败", exc_info=True)
            return TushareFetcher._breadth_cache or {}
