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
    """资金流向"""
    date: str
    main_net_inflow: float   # 主力净流入 (万)
    super_large_net: float   # 超大单净流入
    large_net: float         # 大单净流入
    medium_net: float        # 中单净流入
    small_net: float         # 小单净流入
    main_pct: float = 0.0    # 主力净流入占比 %


@dataclass
class MarketSnapshot:
    """全市场快照"""
    code: str
    name: str
    price: float
    pct_chg: float
    volume_ratio: float
    turnover: float
    amount: float
    pe: float = 0.0
    total_mv: float = 0.0


# ── Fetcher ────────────────────────────────────

class AKShareFetcher:
    """
    AKShare 数据适配器。

    特点: 免费、无需 Token、覆盖面广
    风险: 爬虫机制可能被反爬
    """

    name = "akshare"

    def _sleep(self) -> None:
        time.sleep(random.uniform(1.0, 3.0))

    def _retry(self, fn, *args, max_tries: int = 3, **kwargs):
        """指数退避重试"""
        last_err = None
        for attempt in range(max_tries):
            try:
                self._sleep()
                return fn(*args, **kwargs)
            except Exception as e:
                last_err = e
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
        df = self._retry(self._fetch_daily_from_eastmoney, code, days)
        return self._to_stock_daily_list(df)

    def get_realtime_quote(self, code: str) -> Optional[RealtimeQuote]:
        """获取实时行情"""
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
        try:
            return self._retry(self._fetch_fund_flow, code, days)
        except Exception:
            logger.exception("akshare: 获取资金流向失败 %s", code)
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
        prefix = "sh" if code.startswith(("6", "9")) else "sz"
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
        """从东方财富获取实时行情"""
        import akshare as ak

        em_code = self._eastmoney_code(code)
        try:
            df = ak.stock_zh_a_spot_em()
            row = df[df["代码"] == em_code]
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
        """获取个股基本信息"""
        import akshare as ak

        em_code = self._eastmoney_code(code)
        df = ak.stock_zh_a_spot_em()
        row = df[df["代码"] == em_code]
        if row.empty:
            return {}
        r = row.iloc[0]
        return {
            "code": self._normalize_code(code),
            "name": str(r.get("名称", "")),
            "industry": str(r.get("所属行业", "")),
            "pe": float(r.get("市盈率-动态", 0) or 0),
            "pb": float(r.get("市净率", 0) or 0),
            "total_mv": float(r.get("总市值", 0) or 0) / 1e8,
            "float_mv": float(r.get("流通市值", 0) or 0) / 1e8,
        }

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
        logger.info("curl: 获取 %d 只股票快照 (%d 页)", len(snapshots), page)
        return snapshots

    def _fetch_news(self, keyword: str, days: int) -> list[dict]:
        """财经新闻搜索"""
        import akshare as ak

        try:
            df = ak.stock_zh_a_news(symbol=keyword)
            if df is None or df.empty:
                return []
            results = []
            for _, row in df.head(20).iterrows():
                results.append({
                    "title": str(row.get("title", row.get("标题", ""))),
                    "content": str(row.get("content", row.get("内容", "")))[:500],
                    "time": str(row.get("time", row.get("时间", ""))),
                    "source": str(row.get("source", row.get("来源", ""))),
                })
            return results
        except Exception:
            return []

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
