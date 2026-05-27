"""
AKShare 数据适配器 — 主力免费数据源。

数据来源:
- 东方财富 API (通过 akshare 库)
- 新浪/腾讯财经接口 (备选)

支持:
- A股日线 OHLCV + 均线 + MACD + RSI
- 实时行情 (量比、换手率、市盈率、市净率、市值)
- 资金流向 (主力/大单/中单/小单)
- 股票基本信息 (名称、行业、上市日期)
- 全市场快照 (涨跌幅排序)
- 公告与新闻

设计原则:
- 每次请求前随机休眠 1-3s，防止被封
- 指数退避重试 (最多3次)
- 单只股票失败不影响整体流程
"""

import logging
import random
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── 数据模型 ──────────────────────────────────

@dataclass
class StockDaily:
    """标准化日线数据"""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    pct_chg: float          # 涨跌幅 %
    turnover: float = 0.0   # 换手率 %
    ma5: float = 0.0
    ma10: float = 0.0
    ma20: float = 0.0
    macd_dif: float = 0.0
    macd_dea: float = 0.0
    macd_bar: float = 0.0
    rsi_6: float = 0.0
    rsi_14: float = 0.0


@dataclass
class RealtimeQuote:
    """标准化实时行情"""
    code: str
    name: str
    price: float
    open: float
    high: float
    low: float
    pre_close: float
    pct_chg: float
    volume: float
    amount: float
    turnover: float          # 换手率 %
    volume_ratio: float      # 量比
    pe: float = 0.0          # 市盈率
    pb: float = 0.0          # 市净率
    total_mv: float = 0.0    # 总市值 (亿)
    float_mv: float = 0.0    # 流通市值 (亿)
    source: str = ""         # 数据来源


@dataclass
class FundFlow:
    """资金流向 (单位: 万元, 数据来源于东方财富)"""
    date: str
    main_net_inflow: float   # 主力净流入 (万元)
    super_large_net: float   # 超大单净流入 (万元)
    large_net: float         # 大单净流入 (万元)
    medium_net: float        # 中单净流入 (万元)
    small_net: float         # 小单净流入 (万元)
    main_pct: float = 0.0    # 主力净流入占比 %


@dataclass
class MarketSnapshot:
    """全市场快照 (amount: 元, total_mv: 亿元, 数据来源于东方财富)"""
    code: str
    name: str
    price: float
    pct_chg: float
    volume_ratio: float
    turnover: float
    amount: float            # 成交额 (元)
    pe: float = 0.0
    total_mv: float = 0.0    # 总市值 (亿元)


@dataclass
class NorthboundFlow:
    """北向资金流向 (单位: 万元 / 亿)"""
    date: str
    net_inflow: float          # 当日净流入 (万元)
    balance: float = 0.0       # 累计余额 (亿元)
    sh_inflow: float = 0.0     # 沪股通净流入
    sz_inflow: float = 0.0     # 深股通净流入


@dataclass
class MarginData:
    """融资融券数据 (单位: 亿元)"""
    date: str
    margin_balance: float      # 融资余额
    short_balance: float = 0.0 # 融券余额
    margin_buy: float = 0.0    # 融资买入额
    short_sell: float = 0.0    # 融券卖出量


@dataclass
class FinancialIndicator:
    """深度财务指标 (多季度)"""
    date: str                  # 报告期
    roe: float = 0.0           # ROE (%)
    roa: float = 0.0           # 总资产收益率 (%)
    gross_margin: float = 0.0  # 毛利率 (%)
    net_margin: float = 0.0    # 净利率 (%)
    revenue_yoy: float = 0.0   # 营收同比增速 (%)
    profit_yoy: float = 0.0    # 净利润同比增速 (%)
    debt_ratio: float = 0.0    # 资产负债率 (%)
    eps: float = 0.0           # 每股收益
    current_ratio: float = 0.0 # 流动比率
    quick_ratio: float = 0.0   # 速动比率
    inventory_turnover: float = 0.0  # 存货周转率
    cf_operating: float = 0.0  # 经营活动现金流 (亿)


@dataclass
class ETFSpot:
    """ETF 实时行情"""
    code: str
    name: str
    price: float
    pct_chg: float
    volume: float
    amount: float
    turnover: float = 0.0
    fund_size: float = 0.0     # 基金规模 (亿)


@dataclass
class UnlockShares:
    """限售股解禁信息"""
    code: str
    name: str
    unlock_date: str
    unlock_shares: float       # 解禁股数 (万股)
    unlock_value: float        # 解禁市值 (万元)
    unlock_ratio: float = 0.0  # 解禁占总股本比例 (%)


@dataclass
class ShareholderCount:
    """股东人数变化"""
    date: str
    holder_count: int          # 股东总人数
    change_pct: float = 0.0    # 环比变化 (%)


@dataclass
class InstitutionalVisit:
    """机构调研记录"""
    date: str
    institution: str
    visitors: int = 0
    summary: str = ""


@dataclass
class MarketActivity:
    """市场异动 (盘口异动)"""
    time: str
    code: str
    name: str
    activity_type: str         # 异动类型: 大单买入/大单卖出/涨跌速异常等
    description: str


# ── Fetcher ────────────────────────────────────

class AKShareFetcher:
    """
    AKShare 数据适配器。

    特点: 免费、无需 Token、覆盖面广
    风险: 爬虫机制可能被反爬
    """

    name = "akshare"

    # 全市场快照缓存 (避免每次获取单只股票行情时重复下载 5000+ 条数据)
    _spot_cache: pd.DataFrame | None = None
    _spot_cache_time: float = 0.0
    _SPOT_CACHE_TTL: float = 60.0  # 快照缓存 60 秒
    _spot_cache_lock: threading.Lock = threading.Lock()
    _eastmoney_unavailable: bool = False  # 东方财富不可用时快速失败

    @classmethod
    def _warm_spot_cache(cls):
        """预热全市场快照缓存 (线程安全), 供批量操作前调用"""
        with cls._spot_cache_lock:
            now = time.time()
            if cls._spot_cache is not None and now - cls._spot_cache_time <= cls._SPOT_CACHE_TTL:
                return
            # 最近已尝试过且失败，短时间内不再重试
            if cls._spot_cache is None and now - cls._spot_cache_time <= cls._SPOT_CACHE_TTL * 2:
                return
            try:
                import akshare as ak
                cls._spot_cache = ak.stock_zh_a_spot_em()
                cls._spot_cache_time = now
                logger.debug("akshare: 预热快照缓存 (%d 条)", len(cls._spot_cache))
            except Exception:
                cls._spot_cache_time = now  # 记录失败时间，避免短时间内重复尝试
                raise

    def _sleep(self) -> None:
        time.sleep(random.uniform(1.0, 3.0))

    def _retry(self, fn, *args, max_tries: int = 3, **kwargs):
        """指数退避重试 (首次不等待). AttributeError 不重试"""
        last_err = None
        for attempt in range(max_tries):
            try:
                if attempt > 0:
                    self._sleep()
                return fn(*args, **kwargs)
            except AttributeError:
                raise  # 函数/属性缺失，重试无意义
            except Exception as e:
                last_err = e
                if attempt >= max_tries - 1:
                    raise
                wait = 2 ** attempt
                logger.warning(
                    "akshare: %s 第%d次失败 (%.1fs后重试): %s",
                    fn.__name__, attempt + 1, wait, e,
                )
                time.sleep(wait)
        raise last_err  # type: ignore[misc]

    # ── 公开 API ─────────────────────────────

    def get_daily_data(self, code: str, days: int = 60) -> list[StockDaily]:
        """获取个股日线数据 (含技术指标)"""
        if AKShareFetcher._eastmoney_unavailable:
            return []
        try:
            df = self._retry(self._fetch_daily_from_eastmoney, code, days)
            return self._to_stock_daily_list(df)
        except Exception:
            AKShareFetcher._eastmoney_unavailable = True
            return []

    def get_realtime_quote(self, code: str) -> Optional[RealtimeQuote]:
        """获取实时行情"""
        if AKShareFetcher._eastmoney_unavailable:
            return None
        try:
            return self._retry(self._fetch_realtime, code)
        except Exception:
            logger.exception("akshare: 获取实时行情失败 %s", code)
            return None

    def get_stock_name(self, code: str) -> str:
        """获取股票名称"""
        try:
            q = self.get_realtime_quote(code)
            if q:
                return q.name
        except Exception:
            pass
        return code

    def get_stock_info(self, code: str) -> dict:
        """获取股票基本信息: 名称、行业、上市日期、总股本"""
        try:
            return self._retry(self._fetch_stock_info, code)
        except Exception:
            logger.exception("akshare: 获取股票信息失败 %s", code)
            return {}

    def get_fund_flow(self, code: str, days: int = 5) -> list[FundFlow]:
        """获取近期资金流向"""
        if AKShareFetcher._eastmoney_unavailable:
            return []
        try:
            return self._retry(self._fetch_fund_flow, code, days)
        except Exception:
            AKShareFetcher._eastmoney_unavailable = True
            logger.debug("akshare: 获取资金流向失败 %s", code)
            return []

    def get_market_snapshot(self) -> list[MarketSnapshot]:
        """获取全市场快照 (按成交额降序, Top 3000)"""
        try:
            return self._retry(self._fetch_market_snapshot)
        except Exception:
            logger.exception("akshare: 获取市场快照失败")
            return []

    def get_news(self, keyword: str, days: int = 3) -> list[dict]:
        """
        获取财经新闻 (基于关键词搜索)。

        返回: [{"title": ..., "content": ..., "time": ..., "source": ...}, ...]
        """
        try:
            return self._retry(self._fetch_news, keyword, days)
        except Exception:
            logger.exception("akshare: 获取新闻失败 %s", keyword)
            return []

    def get_announcements(self, code: str, days: int = 7) -> list[dict]:
        """获取个股近期公告"""
        try:
            return self._retry(self._fetch_announcements, code, days)
        except Exception:
            logger.exception("akshare: 获取公告失败 %s", code)
            return []

    # ── 内部实现 ─────────────────────────────

    @staticmethod
    def _normalize_code(code: str) -> str:
        """标准化股票代码: 去除 sh/sz 前缀, 返回纯数字"""
        code = code.replace("sh", "").replace("sz", "").replace("SH", "").replace("SZ", "")
        return code.strip()

    @staticmethod
    def _eastmoney_code(code: str) -> str:
        """转为东方财富代码格式 (sh600519 / sz000858)"""
        code = AKShareFetcher._normalize_code(code)
        prefix = "sh" if code.startswith(("6", "9")) else ("bj" if code.startswith(("4", "8")) else "sz")
        return f"{prefix}{code}"

    def _fetch_daily_from_eastmoney(self, code: str, days: int) -> pd.DataFrame:
        """从东方财富拉 K 线数据"""
        import akshare as ak

        em_code = self._eastmoney_code(code)
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")

        df = ak.stock_zh_a_hist(
            symbol=em_code,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="qfq",  # 前复权
        )
        return df.tail(days)

    def _to_stock_daily_list(self, df: pd.DataFrame) -> list[StockDaily]:
        """DataFrame → list[StockDaily] (含技术指标)"""
        if df.empty:
            return []

        closes = df["收盘"].astype(float).values
        highs = df["最高"].astype(float).values
        lows = df["最低"].astype(float).values
        volumes = df["成交量"].astype(float).values

        # 均线
        ma5 = _rolling(closes, 5)
        ma10 = _rolling(closes, 10)
        ma20 = _rolling(closes, 20)

        # MACD
        ema12 = _ema(closes, 12)
        ema26 = _ema(closes, 26)
        dif = ema12 - ema26
        dea = _ema(dif, 9)
        macd_bar = 2 * (dif - dea)

        # RSI
        rsi6 = _rsi(closes, 6)
        rsi14 = _rsi(closes, 14)

        records = []
        for i, (_, row) in enumerate(df.iterrows()):
            try:
                records.append(StockDaily(
                    date=str(row["日期"])[:10],
                    open=float(row["开盘"]),
                    high=float(highs[i]),
                    low=float(lows[i]),
                    close=float(closes[i]),
                    volume=float(volumes[i]),
                    amount=float(row.get("成交额", 0)),
                    pct_chg=float(row.get("涨跌幅", 0)),
                    turnover=float(row.get("换手率", 0)),
                    ma5=round(ma5[i], 2) if i >= 4 else 0,
                    ma10=round(ma10[i], 2) if i >= 9 else 0,
                    ma20=round(ma20[i], 2) if i >= 19 else 0,
                    macd_dif=round(float(dif[i]), 4) if i >= 25 else 0,
                    macd_dea=round(float(dea[i]), 4) if i >= 34 else 0,
                    macd_bar=round(float(macd_bar[i]), 4) if i >= 34 else 0,
                    rsi_6=round(float(rsi6[i]), 1) if i >= 6 else 50,
                    rsi_14=round(float(rsi14[i]), 1) if i >= 14 else 50,
                ))
            except (ValueError, KeyError, IndexError):
                continue
        return records

    def _fetch_realtime(self, code: str) -> RealtimeQuote:
        """从东方财富获取实时行情 (全市场快照缓存 60s，避免重复下载)"""
        import akshare as ak

        em_code = self._eastmoney_code(code)
        try:
            AKShareFetcher._warm_spot_cache()
            df = AKShareFetcher._spot_cache
            if df is None:
                raise ValueError("快照缓存为空")
            code_digits = self._normalize_code(code)
            mask = df["代码"].str.replace(r"[^0-9]", "", regex=True) == code_digits
            row = df[mask]
            if row.empty:
                raise ValueError(f"代码 {em_code} 未找到")
            r = row.iloc[0]
            return RealtimeQuote(
                code=self._normalize_code(code),
                name=str(r.get("名称", "")),
                price=float(r.get("最新价", 0)),
                open=float(r.get("今开", 0)),
                high=float(r.get("最高", 0)),
                low=float(r.get("最低", 0)),
                pre_close=float(r.get("昨收", 0)),
                pct_chg=float(r.get("涨跌幅", 0)),
                volume=float(r.get("成交量", 0)),
                amount=float(r.get("成交额", 0)),
                turnover=float(r.get("换手率", 0)),
                volume_ratio=float(r.get("量比", 0)),
                pe=float(r.get("市盈率-动态", 0) or 0),
                pb=float(r.get("市净率", 0) or 0),
                total_mv=float(r.get("总市值", 0) or 0) / 1e8,
                float_mv=float(r.get("流通市值", 0) or 0) / 1e8,
                source="akshare:eastmoney",
            )
        except Exception:
            # 备选: 新浪接口
            return self._fetch_realtime_sina(code)

    def _fetch_realtime_sina(self, code: str) -> RealtimeQuote:
        """新浪财经接口 (备选)"""
        import requests

        raw = self._eastmoney_code(code)
        url = f"https://hq.sinajs.cn/list={raw}"
        resp = requests.get(url, headers={"Referer": "https://finance.sina.com.cn"}, timeout=10)
        resp.encoding = "gbk"
        parts = resp.text.split('"')[1].split(",")
        return RealtimeQuote(
            code=self._normalize_code(code),
            name=parts[0],
            price=float(parts[3]),
            open=float(parts[1]),
            high=float(parts[4]),
            low=float(parts[5]),
            pre_close=float(parts[2]),
            pct_chg=round((float(parts[3]) - float(parts[2])) / float(parts[2]) * 100, 2),
            volume=float(parts[8]),
            amount=float(parts[9]),
            turnover=0,
            volume_ratio=0,
            source="akshare:sina",
        )

    def _fetch_stock_info(self, code: str) -> dict:
        """获取个股基本信息 (含上市日期)，优先复用快照缓存避免重复 akshare 调用"""
        import akshare as ak
        import pandas as pd

        em_code = self._eastmoney_code(code)
        code_digits = self._normalize_code(code)

        # 优先从全市场快照缓存获取 (线程安全，避免 3000 个线程同时下载)
        try:
            AKShareFetcher._warm_spot_cache()
            df = AKShareFetcher._spot_cache
            if df is None:
                return {}
        except Exception:
            logger.debug("akshare: stock_zh_a_spot_em 失败 (网络不通)，返回空信息")
            return {}

        mask = df["代码"].str.replace(r"[^0-9]", "", regex=True) == code_digits
        row = df[mask]
        if row.empty:
            return {}
        r = row.iloc[0]
        info = {
            "code": code_digits,
            "name": str(r.get("名称", "")),
            "industry": str(r.get("所属行业", "")),
            "pe": float(r.get("市盈率-动态", 0) or 0),
            "pb": float(r.get("市净率", 0) or 0),
            "total_mv": float(r.get("总市值", 0) or 0) / 1e8,
            "float_mv": float(r.get("流通市值", 0) or 0) / 1e8,
        }

        # 补充上市日期 (需额外 API 调用)
        try:
            self._sleep()
            detail = ak.stock_individual_info_em(symbol=em_code)
            if detail is not None and not detail.empty:
                detail_dict = dict(zip(detail["item"], detail["value"]))
                ipo_date = detail_dict.get("上市时间", "")
                if ipo_date and str(ipo_date) != "None":
                    info["ipo_date"] = str(ipo_date)[:10]
        except Exception:
            logger.debug("akshare: 获取 %s 上市日期失败，跳过新股过滤", code)

        return info

    def _fetch_fund_flow(self, code: str, days: int) -> list[FundFlow]:
        """获取资金流向"""
        import akshare as ak

        em_code = self._eastmoney_code(code)
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=max(days, 10))).strftime("%Y%m%d")

        df = ak.stock_individual_fund_flow(
            stock=em_code, market="sh" if code.startswith("6") else "sz"
        )
        df = df.tail(days)

        records = []
        for _, row in df.iterrows():
            try:
                records.append(FundFlow(
                    date=str(row.get("日期", ""))[:10],
                    main_net_inflow=float(row.get("主力净流入-净额", 0) or 0),
                    super_large_net=float(row.get("超大单净流入-净额", 0) or 0),
                    large_net=float(row.get("大单净流入-净额", 0) or 0),
                    medium_net=float(row.get("中单净流入-净额", 0) or 0),
                    small_net=float(row.get("小单净流入-净额", 0) or 0),
                    main_pct=float(row.get("主力净流入-净占比", 0) or 0),
                ))
            except (ValueError, KeyError):
                continue
        return records

    def _fetch_market_snapshot(self) -> list[MarketSnapshot]:
        """全市场快照 — 优先用 curl (绕过 TLS 指纹问题), akshare 做备选"""
        # 先尝试 curl 直连 (系统 curl 的 TLS 指纹不被封锁)
        try:
            return self._fetch_market_snapshot_via_curl()
        except Exception:
            logger.debug("curl 获取快照失败，回退到 akshare")

        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        df = df.sort_values("成交额", ascending=False).head(3000)

        snapshots = []
        for _, r in df.iterrows():
            try:
                snapshots.append(MarketSnapshot(
                    code=str(r["代码"]).replace("sh", "").replace("sz", ""),
                    name=str(r.get("名称", "")),
                    price=float(r.get("最新价", 0)),
                    pct_chg=float(r.get("涨跌幅", 0)),
                    volume_ratio=float(r.get("量比", 0) or 1),
                    turnover=float(r.get("换手率", 0) or 0),
                    amount=float(r.get("成交额", 0)),
                    pe=float(r.get("市盈率-动态", 0) or 0),
                    total_mv=float(r.get("总市值", 0) or 0) / 1e8,
                ))
            except (ValueError, KeyError):
                continue
        return snapshots

    def _fetch_market_snapshot_via_curl(self) -> list[MarketSnapshot]:
        """通过系统 curl 获取全市场快照 (绕过 Python TLS 指纹封锁)"""
        import subprocess
        import json as _json

        snapshots: list[MarketSnapshot] = []
        all_rows: list[dict] = []  # 缓存原始行供 _fetch_stock_info 复用
        # 分批获取: 每页100只, 取30页 = 3000只
        for page in range(1, 31):
            url = (
                "https://push2.eastmoney.com/api/qt/clist/get"
                f"?pn={page}&pz=100&po=1&np=1"
                "&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f12"
                "&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048"
                "&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,"
                "f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152"
            )
            try:
                result = subprocess.run(
                    ["curl", "-s", "--max-time", "15", url],
                    capture_output=True, text=True, timeout=20,
                )
                if result.returncode != 0:
                    continue
                data = _json.loads(result.stdout)
                rows = data.get("data", {}).get("diff", [])
                if not rows:
                    break
                all_rows.extend(rows)
                for r in rows:
                    try:
                        snapshots.append(MarketSnapshot(
                            code=str(r.get("f12", "")),
                            name=str(r.get("f14", "")),
                            price=float(r.get("f2", 0) or 0),
                            pct_chg=float(r.get("f3", 0) or 0),
                            volume_ratio=float(r.get("f10", 0) or 1),
                            turnover=float(r.get("f8", 0) or 0),
                            amount=float(r.get("f6", 0) or 0),
                            pe=float(r.get("f9", 0) or 0),
                            total_mv=float(r.get("f20", 0) or 0) / 1e8,
                        ))
                    except (ValueError, KeyError):
                        continue
            except Exception:
                continue

        if not snapshots:
            raise RuntimeError("curl 获取全市场快照为空")

        # 缓存原始行数据供 _fetch_stock_info 复用 (避免重复 akshare 调用)
        AKShareFetcher._spot_cache_rows = all_rows
        pages_fetched = page + 1 if page < 30 else page
        logger.info("curl: 获取 %d 只股票快照 (%d 页)", len(snapshots), pages_fetched)
        return snapshots

    def _fetch_news(self, keyword: str, days: int) -> list[dict]:
        """财经新闻搜索 — 优先用个股新闻接口，失败则尝试关键词搜索"""
        import akshare as ak

        results: list[dict] = []

        # 首先尝试个股新闻 (keyword 为股票代码时工作)
        try:
            df = ak.stock_zh_a_news(symbol=keyword)
            if df is not None and not df.empty:
                for _, row in df.head(20).iterrows():
                    results.append({
                        "title": str(row.get("title", row.get("标题", ""))),
                        "content": str(row.get("content", row.get("内容", "")))[:500],
                        "time": str(row.get("time", row.get("时间", ""))),
                        "source": str(row.get("source", row.get("来源", ""))),
                    })
                if results:
                    return results
        except Exception:
            pass

        # 备选: 使用东方财富关键词搜索 (适用于非代码关键词)
        try:
            from urllib.parse import urlencode
            em_code = self._eastmoney_code(keyword) if keyword.isdigit() else ""
            symbol = em_code or keyword
            params = urlencode({"cb": "callback", "keyword": symbol, "pageindex": 1, "pagesize": 15})
            url = f"https://search-api-web.eastmoney.com/search/jsonp?{params}"
            import requests as _requests
            resp = _requests.get(url, timeout=10, headers={
                "Referer": "https://so.eastmoney.com/",
            })
            if resp.status_code == 200:
                text = resp.text
                import json as _json
                if text.startswith("callback("):
                    text = text[9:-1]
                data = _json.loads(text)
                articles = data.get("Data", {}).get("Report", [])
                for a in articles[:15]:
                    results.append({
                        "title": a.get("Title", a.get("title", "")),
                        "content": a.get("Summary", a.get("Content", ""))[:500],
                        "time": a.get("PublishTime", a.get("time", "")),
                        "source": a.get("Source", a.get("source", "")),
                    })
        except Exception:
            pass

        return results

    def _fetch_announcements(self, code: str, days: int) -> list[dict]:
        """个股公告"""
        import akshare as ak

        try:
            df = ak.stock_notice_report(symbol=self._eastmoney_code(code))
            if df is None or df.empty:
                return []
            results = []
            for _, row in df.head(10).iterrows():
                results.append({
                    "title": str(row.get("title", row.get("公告标题", ""))),
                    "time": str(row.get("notice_date", row.get("公告日期", ""))),
                    "url": str(row.get("url", "")),
                })
            return results
        except Exception:
            return []

    # ── ETF 数据 ─────────────────────────────

    def get_etf_spot(self) -> list[ETFSpot]:
        """获取场内 ETF 实时行情"""
        try:
            return self._retry(self._fetch_etf_spot)
        except Exception:
            logger.exception("akshare: 获取ETF行情失败")
            return []

    def _fetch_etf_spot(self) -> list[ETFSpot]:
        import akshare as ak
        df = ak.fund_etf_spot_em()
        results = []
        for _, r in df.iterrows():
            try:
                results.append(ETFSpot(
                    code=str(r.get("代码", "")),
                    name=str(r.get("名称", "")),
                    price=float(r.get("最新价", 0) or 0),
                    pct_chg=float(r.get("涨跌幅", 0) or 0),
                    volume=float(r.get("成交量", 0) or 0),
                    amount=float(r.get("成交额", 0) or 0),
                    turnover=float(r.get("换手率", 0) or 0),
                ))
            except (ValueError, KeyError):
                continue
        return results

    def get_etf_daily(self, code: str, days: int = 60) -> list[StockDaily]:
        """获取 ETF 日线数据 (复用 StockDaily 结构)"""
        try:
            return self._retry(self._fetch_etf_daily, code, days)
        except Exception:
            logger.exception("akshare: 获取ETF日线失败 %s", code)
            return []

    def _fetch_etf_daily(self, code: str, days: int) -> list[StockDaily]:
        import akshare as ak
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")
        df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
        return self._to_stock_daily_list(df.tail(days))

    # ── 北向资金 ─────────────────────────────

    def get_northbound_flow(self, days: int = 5) -> list[NorthboundFlow]:
        """获取北向资金净流向 (全市场)"""
        try:
            return self._retry(self._fetch_northbound_flow, days)
        except Exception:
            logger.exception("akshare: 获取北向资金流向失败")
            return []

    def _fetch_northbound_flow(self, days: int) -> list[NorthboundFlow]:
        import akshare as ak
        df = ak.stock_hsgt_north_net_flow_in_em()
        results = []
        for _, r in df.tail(days).iterrows():
            try:
                results.append(NorthboundFlow(
                    date=str(r.get("date", r.get("日期", "")))[:10],
                    net_inflow=float(r.get("value", r.get("净流入", 0)) or 0),
                    sh_inflow=float(r.get("sh", r.get("沪股通", 0)) or 0),
                    sz_inflow=float(r.get("sz", r.get("深股通", 0)) or 0),
                ))
            except (ValueError, KeyError):
                continue
        return results

    def get_northbound_stock(self, code: str, days: int = 10) -> list[dict]:
        """获取个股沪深股通持仓变化"""
        try:
            return self._retry(self._fetch_northbound_stock, code, days)
        except Exception:
            logger.exception("akshare: 获取个股北向数据失败 %s", code)
            return []

    def _fetch_northbound_stock(self, code: str, days: int) -> list[dict]:
        import akshare as ak
        em_code = self._eastmoney_code(code)
        try:
            df = ak.stock_hsgt_individual_em(symbol=em_code)
            return [
                {"date": str(r.get("日期", ""))[:10],
                 "hold_shares": float(r.get("持股数量", 0) or 0),
                 "hold_value": float(r.get("持股市值", 0) or 0),
                 "hold_pct": float(r.get("持股占比", 0) or 0)}
                for _, r in df.tail(days).iterrows()
            ]
        except Exception:
            return []

    # ── 融资融券 ─────────────────────────────

    def get_margin_summary(self) -> dict:
        """获取全市场融资融券概况"""
        try:
            return self._retry(self._fetch_margin_summary)
        except Exception:
            logger.exception("akshare: 获取融资融券概况失败")
            return {}

    def _fetch_margin_summary(self) -> dict:
        import akshare as ak
        try:
            df_sh = ak.stock_margin_sh()
            sh_row = df_sh.iloc[-1] if not df_sh.empty else None
        except Exception:
            sh_row = None
        try:
            df_sz = ak.stock_margin_sz()
            sz_row = df_sz.iloc[-1] if not df_sz.empty else None
        except Exception:
            sz_row = None

        result = {}
        if sh_row is not None:
            result["sh_margin_balance"] = float(sh_row.get("融资余额", 0) or 0)
            result["sh_margin_buy"] = float(sh_row.get("融资买入额", 0) or 0)
        if sz_row is not None:
            result["sz_margin_balance"] = float(sz_row.get("融资余额", 0) or 0)
            result["sz_margin_buy"] = float(sz_row.get("融资买入额", 0) or 0)
        return result

    def get_margin_detail(self, code: str, days: int = 10) -> list[MarginData]:
        """获取个股融资融券明细"""
        try:
            return self._retry(self._fetch_margin_detail, code, days)
        except Exception:
            logger.exception("akshare: 获取个股融资融券失败 %s", code)
            return []

    def _fetch_margin_detail(self, code: str, days: int) -> list[MarginData]:
        import akshare as ak
        market = "sh" if code.startswith(("6", "9")) else "sz"
        try:
            fn = ak.stock_margin_detail_sh if market == "sh" else ak.stock_margin_detail_sz
            today_str = datetime.now().strftime("%Y%m%d")
            df = fn(date=today_str)
            if df is None or df.empty:
                return []
            mask = df["股票代码"].astype(str).str.replace(r"[^0-9]", "", regex=True) == self._normalize_code(code)
            df = df[mask].tail(days)
            return [
                MarginData(
                    date=str(r.get("日期", ""))[:10],
                    margin_balance=float(r.get("融资余额", 0) or 0),
                    margin_buy=float(r.get("融资买入额", 0) or 0),
                    short_balance=float(r.get("融券余量", 0) or 0),
                )
                for _, r in df.iterrows()
            ]
        except Exception:
            return []

    # ── 深度财务指标 ─────────────────────────

    def get_financial_indicators(self, code: str) -> list[FinancialIndicator]:
        """获取深度财务指标 (多报告期趋势)"""
        try:
            return self._retry(self._fetch_financial_indicators, code)
        except Exception:
            logger.exception("akshare: 获取财务指标失败 %s", code)
            return []

    def _fetch_financial_indicators(self, code: str) -> list[FinancialIndicator]:
        import akshare as ak
        em_code = self._eastmoney_code(code)
        df = ak.stock_financial_analysis_indicator(symbol=em_code)
        if df is None or df.empty:
            return []
        results = []
        for _, r in df.tail(8).iterrows():
            try:
                results.append(FinancialIndicator(
                    date=str(r.get("日期", r.get("报告期", "")))[:10],
                    roe=float(r.get("净资产收益率", 0) or 0),
                    roa=float(r.get("总资产净利润率", 0) or 0),
                    gross_margin=float(r.get("销售毛利率", 0) or 0),
                    net_margin=float(r.get("销售净利率", 0) or 0),
                    revenue_yoy=float(r.get("营业收入同比增长率", 0) or 0),
                    profit_yoy=float(r.get("净利润同比增长率", 0) or 0),
                    debt_ratio=float(r.get("资产负债率", 0) or 0),
                    eps=float(r.get("每股收益", 0) or 0),
                    current_ratio=float(r.get("流动比率", 0) or 0),
                    quick_ratio=float(r.get("速动比率", 0) or 0),
                    inventory_turnover=float(r.get("存货周转率", 0) or 0),
                    cf_operating=float(r.get("经营活动现金流量净额", 0) or 0) / 1e8,
                ))
            except (ValueError, KeyError):
                continue
        return results

    # ── 财联社电报 ───────────────────────────

    def get_telegraph(self, limit: int = 30) -> list[dict]:
        """获取财联社电报 (实时快讯)"""
        try:
            return self._retry(self._fetch_telegraph, limit)
        except Exception:
            logger.exception("akshare: 获取电报失败")
            return []

    def _fetch_telegraph(self, limit: int) -> list[dict]:
        import akshare as ak
        df = ak.stock_telegraph_cls()
        if df is None or df.empty:
            return []
        return [
            {"title": str(r.get("title", r.get("标题", ""))),
             "content": str(r.get("content", r.get("内容", "")))[:500],
             "time": str(r.get("ctime", r.get("时间", "")))}
            for _, r in df.head(limit).iterrows()
        ]

    # ── 分析师研报 ───────────────────────────

    def get_research_reports(self, code: str, days: int = 30) -> list[dict]:
        """获取个股分析师研报"""
        try:
            return self._retry(self._fetch_research_reports, code, days)
        except Exception:
            logger.exception("akshare: 获取研报失败 %s", code)
            return []

    def _fetch_research_reports(self, code: str, days: int) -> list[dict]:
        import akshare as ak
        try:
            df = ak.stock_research_report_em(symbol=self._eastmoney_code(code))
            if df is None or df.empty:
                return []
            return [
                {"title": str(r.get("title", r.get("研报标题", ""))),
                 "org": str(r.get("org", r.get("研究机构", ""))),
                 "rating": str(r.get("rating", r.get("评级", ""))),
                 "date": str(r.get("date", r.get("日期", "")))[:10]}
                for _, r in df.head(15).iterrows()
            ]
        except Exception:
            return []

    # ── 行业成分股 ───────────────────────────

    def get_industry_stocks(self, industry: str) -> list[str]:
        """获取指定行业的所有成分股代码"""
        try:
            return self._retry(self._fetch_industry_stocks, industry)
        except Exception:
            logger.exception("akshare: 获取行业成分股失败 %s", industry)
            return []

    def _fetch_industry_stocks(self, industry: str) -> list[str]:
        import akshare as ak
        df = ak.stock_board_industry_cons_em(symbol=industry)
        if df is None or df.empty:
            return []
        codes = df["代码"].astype(str).tolist()
        # 清理代码格式 (去除 sh/sz 前缀不一致的情况)
        cleaned = []
        for c in codes:
            first_digit_pos = next((i for i, ch in enumerate(c) if ch.isdigit()), -1)
            cleaned.append(c[first_digit_pos:] if first_digit_pos >= 0 else c)
        return cleaned

    # ── 限售解禁 ─────────────────────────────

    def get_unlock_shares(self, days_ahead: int = 30) -> list[UnlockShares]:
        """获取近期限售股解禁列表"""
        try:
            return self._retry(self._fetch_unlock_shares, days_ahead)
        except Exception:
            logger.exception("akshare: 获取解禁数据失败")
            return []

    def _fetch_unlock_shares(self, days_ahead: int) -> list[UnlockShares]:
        import akshare as ak
        try:
            df = ak.stock_restricted_release_queue_em()
            if df is None or df.empty:
                return []
            results = []
            for _, r in df.head(50).iterrows():
                try:
                    results.append(UnlockShares(
                        code=str(r.get("代码", "")),
                        name=str(r.get("名称", "")),
                        unlock_date=str(r.get("解禁日期", ""))[:10],
                        unlock_shares=float(r.get("解禁数量", 0) or 0),
                        unlock_value=float(r.get("解禁市值", 0) or 0),
                        unlock_ratio=float(r.get("占总股本比例", 0) or 0),
                    ))
                except (ValueError, KeyError):
                    continue
            return results
        except Exception:
            return []

    # ── 股东人数变化 (筹码集中度) ─────────────

    def get_shareholder_count(self, code: str) -> list[ShareholderCount]:
        """获取股东人数变化趋势 (筹码集中度)"""
        try:
            return self._retry(self._fetch_shareholder_count, code)
        except Exception:
            logger.exception("akshare: 获取股东人数失败 %s", code)
            return []

    def _fetch_shareholder_count(self, code: str) -> list[ShareholderCount]:
        import akshare as ak
        try:
            df = ak.stock_zh_a_gdhs_em(symbol=self._eastmoney_code(code))
            if df is None or df.empty:
                return []
            results = []
            for _, r in df.tail(6).iterrows():
                holder_count = float(r.get("股东人数", 0) or 0)
                if holder_count <= 0:
                    continue
                results.append(ShareholderCount(
                    date=str(r.get("日期", ""))[:10],
                    holder_count=int(holder_count),
                    change_pct=float(r.get("环比增减", 0) or 0),
                ))
            return results
        except Exception:
            return []

    # ── 机构调研 ──────────────────────────────

    def get_institutional_visits(self, days: int = 30) -> list[InstitutionalVisit]:
        """获取近期机构调研记录"""
        try:
            return self._retry(self._fetch_institutional_visits, days)
        except Exception:
            logger.exception("akshare: 获取机构调研失败")
            return []

    def _fetch_institutional_visits(self, days: int) -> list[InstitutionalVisit]:
        import akshare as ak
        try:
            df = ak.stock_institute_visit_em()
            if df is None or df.empty:
                return []
            return [
                InstitutionalVisit(
                    date=str(r.get("日期", ""))[:10],
                    institution=str(r.get("机构名称", r.get("调研机构", ""))),
                    visitors=int(r.get("调研人员", 0) or 0),
                    summary=str(r.get("调研摘要", ""))[:300],
                )
                for _, r in df.head(30).iterrows()
            ]
        except Exception:
            return []

    # ── 市场异动 ──────────────────────────────

    def get_market_activity(self) -> list[MarketActivity]:
        """获取盘口异动 (大单买入/卖出、涨跌速异常等)"""
        try:
            return self._retry(self._fetch_market_activity)
        except Exception:
            logger.exception("akshare: 获取市场异动失败")
            return []

    def _fetch_market_activity(self) -> list[MarketActivity]:
        import akshare as ak
        try:
            df = ak.stock_market_activity_em()
            if df is None or df.empty:
                return []
            return [
                MarketActivity(
                    time=str(r.get("时间", "")),
                    code=str(r.get("代码", "")),
                    name=str(r.get("名称", "")),
                    activity_type=str(r.get("异动类型", r.get("异动", ""))),
                    description=str(r.get("异动描述", r.get("描述", ""))),
                )
                for _, r in df.head(30).iterrows()
            ]
        except Exception:
            return []

    # ── 大宗交易 ──────────────────────────────

    def get_block_trades(self, days: int = 10) -> list[dict]:
        """获取近期大宗交易明细"""
        try:
            return self._retry(self._fetch_block_trades, days)
        except Exception:
            logger.exception("akshare: 获取大宗交易失败")
            return []

    def _fetch_block_trades(self, days: int) -> list[dict]:
        import akshare as ak
        try:
            df = ak.stock_dzjy_mrmx()
            if df is None or df.empty:
                return []
            return [
                {"date": str(r.get("日期", ""))[:10],
                 "code": str(r.get("代码", "")),
                 "name": str(r.get("名称", "")),
                 "price": float(r.get("成交价", 0) or 0),
                 "volume": float(r.get("成交额", 0) or 0),
                 "premium": float(r.get("溢价率", 0) or 0)}
                for _, r in df.head(30).iterrows()
            ]
        except Exception:
            return []

    # ── 龙虎榜深化 ────────────────────────────

    def get_dragon_tiger_stats(self, days: int = 10) -> list[dict]:
        """龙虎榜个股上榜统计 (游资活跃度)"""
        try:
            return self._retry(self._fetch_dragon_tiger_stats, days)
        except Exception:
            logger.exception("akshare: 获取龙虎榜统计失败")
            return []

    def _fetch_dragon_tiger_stats(self, days: int) -> list[dict]:
        import akshare as ak
        try:
            df = ak.stock_lhb_stock_statistic_em()
            if df is None or df.empty:
                return []
            return [
                {"code": str(r.get("代码", "")),
                 "name": str(r.get("名称", "")),
                 "count": int(r.get("上榜次数", 0) or 0),
                 "buy_amount": float(r.get("买入金额", 0) or 0),
                 "sell_amount": float(r.get("卖出金额", 0) or 0)}
                for _, r in df.head(30).iterrows()
            ]
        except Exception:
            return []


# ── 技术指标计算 ────────────────────────────

def _rolling(arr, window: int):
    """简单移动平均"""
    out = pd.Series(arr).rolling(window=window, min_periods=1).mean()
    return out.fillna(0).values


def _ema(arr, period: int):
    """指数移动平均"""
    s = pd.Series(arr).ewm(span=period, adjust=False).mean()
    return s.fillna(arr[0]).values


def _rsi(closes, period: int = 14):
    """RSI 指标"""
    delta = pd.Series(closes).diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).values
