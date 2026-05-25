"""
海选过滤器 — 确定性规则，不消耗 LLM Token。

过滤链:
  全市场 5000+
  → 剔除 ST/*ST/退市
  → 剔除停牌 (价格或换手率为0)
  → 剔除新股 (上市 < 60 天)
  → 剔除流动性不足 (日均成交额 < 阈值)
  → 进入多因子打分
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional

from ..data.fetchers.akshare_fetcher import MarketSnapshot

logger = logging.getLogger(__name__)

# IPO 最短天数 — 新股上市不足此天数直接排除
MIN_LISTING_DAYS = 60

# 日均成交额下限 (元) — 低于此值的股票流动性太差
MIN_DAILY_AMOUNT = 50_000_000  # 5000万


def filter_tradable(
    snapshots: list[MarketSnapshot],
    stock_infos: Optional[dict[str, dict]] = None,
) -> list[MarketSnapshot]:
    """
    剔除 ST/*ST、停牌、新股。

    snapshots: 全市场快照列表
    stock_infos: {code: {name, ipo_date, ...}} 批量获取的基本信息

    返回: 通过过滤的快照列表
    """
    today = datetime.now().date()
    cutoff = today - timedelta(days=MIN_LISTING_DAYS)

    passed = []
    st_count = 0
    suspended_count = 0
    new_count = 0

    for s in snapshots:
        # 1. ST 过滤
        if "ST" in s.name.upper():
            st_count += 1
            continue

        # 2. 停牌过滤 (价格或换手率接近0)
        if s.price <= 0.01 or s.turnover <= 0.001:
            suspended_count += 1
            continue

        # 3. 新股过滤
        if stock_infos and s.code in stock_infos:
            ipo_date = stock_infos[s.code].get("ipo_date", "")
            if ipo_date:
                try:
                    ipo_dt = _parse_ipo_date(ipo_date)
                    if ipo_dt and ipo_dt > cutoff:
                        new_count += 1
                        continue
                except (ValueError, IndexError):
                    pass  # 无法解析日期则放行

        passed.append(s)

    logger.info(
        "filter_tradable: %d -> %d (剔除 ST:%d 停牌:%d 新股:%d)",
        len(snapshots), len(passed), st_count, suspended_count, new_count,
    )
    return passed


def filter_liquidity(
    snapshots: list[MarketSnapshot],
    min_amount: float = MIN_DAILY_AMOUNT,
) -> list[MarketSnapshot]:
    """
    剔除日均成交额不足的股票，确保入选标的可交易。

    MarketSnapshot.amount 已是当日成交额，作为日均近似值。
    """
    passed = [s for s in snapshots if s.amount >= min_amount]
    logger.info(
        "filter_liquidity: %d -> %d (剔除 成交额<%d万)",
        len(snapshots), len(passed), min_amount // 10000,
    )
    return passed


def filter_volatility(
    daily_data: dict[str, list],
    max_volatility_pct: float = 15.0,
) -> set[str]:
    """
    剔除近期波动异常剧烈的股票。

    daily_data: {code: [StockDaily, ...]}
    max_volatility_pct: 单日涨跌幅绝对值上限

    返回: 应被剔除的 code 集合
    """
    excluded: set[str] = set()
    for code, records in daily_data.items():
        if not records:
            continue
        for r in records:
            if abs(r.pct_chg) > max_volatility_pct:
                excluded.add(code)
                break

    logger.info("filter_volatility: 剔除 %d 只异常波动股", len(excluded))
    return excluded


def extract_codes(snapshots: list[MarketSnapshot]) -> list[str]:
    """从快照列表提取纯代码列表"""
    return [s.code for s in snapshots]


def _parse_ipo_date(raw: str) -> Optional[date]:
    """
    解析上市日期，支持两种格式:
    - YYYY-MM-DD (BaoStock)
    - YYYYMMDD   (Tushare)
    """
    raw = raw.strip()
    try:
        return datetime.strptime(raw[:10], "%Y-%m-%d").date()
    except ValueError:
        pass
    try:
        return datetime.strptime(raw[:8], "%Y%m%d").date()
    except ValueError:
        return None
