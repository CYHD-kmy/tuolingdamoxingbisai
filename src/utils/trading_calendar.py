"""
A 股交易日历 — 判断指定日期是否为交易日。

数据来源: 基于中国法定节假日和交易所公告的简化实现。
注意: 假期数据仅硬编码了 2026 年，其他年份需手动更新 _FIXED_HOLIDAYS。
生产环境建议使用 akshare 的 tool_trade_date_hist_sina() 获取官方交易日历。
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

logger = logging.getLogger(__name__)

# 2026 年中国法定节假日 (国务院已公布)
# 元旦: 1月1-3日 / 春节: 2月15-21日 / 清明节: 4月5-7日
# 劳动节: 5月1-5日 / 端午节: 6月19-21日 / 中秋节+国庆节: 9月25日-10月8日
_HOLIDAYS_2026: set[str] = set()

_FIXED_HOLIDAYS: dict[int, list[tuple[int, int]]] = {
    2026: [
        (1, 1), (1, 2), (1, 3),                          # 元旦
        (2, 15), (2, 16), (2, 17), (2, 18), (2, 19), (2, 20), (2, 21),  # 春节
        (4, 5), (4, 6), (4, 7),                           # 清明节
        (5, 1), (5, 2), (5, 3), (5, 4), (5, 5),          # 劳动节
        (6, 19), (6, 20), (6, 21),                        # 端午节
        (9, 25), (9, 26), (9, 27), (9, 28), (9, 29), (9, 30),           # 中秋节+国庆节
        (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8),
    ],
}


def _build_holidays(year: int) -> set[str]:
    """构建指定年份的节假日集合"""
    holidays: set[str] = set()
    if year in _FIXED_HOLIDAYS:
        for m, d in _FIXED_HOLIDAYS[year]:
            holidays.add(date(year, m, d).isoformat())
    return holidays


def _get_holidays(year: int) -> set[str]:
    global _HOLIDAYS_2026
    if year == 2026 and not _HOLIDAYS_2026:
        _HOLIDAYS_2026 = _build_holidays(2026)
    if year == 2026:
        return _HOLIDAYS_2026
    return _build_holidays(year)


def is_trading_day(d: date | datetime | None = None) -> bool:
    """
    判断是否为 A 股交易日。

    规则:
    - 周六/周日: 非交易日
    - 法定节假日: 非交易日
    - 其余: 交易日
    """
    if d is None:
        d = date.today()
    if isinstance(d, datetime):
        d = d.date()

    # 周末
    if d.weekday() >= 5:
        return False

    # 法定节假日
    holidays = _get_holidays(d.year)
    if d.isoformat() in holidays:
        return False

    return True


def next_trading_day(d: date | datetime | None = None) -> date:
    """返回下一个交易日 (不含当天)"""
    if d is None:
        d = date.today()
    if isinstance(d, datetime):
        d = d.date()

    next_day = d + timedelta(days=1)
    while not is_trading_day(next_day):
        next_day = next_day + timedelta(days=1)
    return next_day


def prev_trading_day(d: date | datetime | None = None) -> date:
    """返回上一个交易日 (不含当天)"""
    if d is None:
        d = date.today()
    if isinstance(d, datetime):
        d = d.date()

    prev_day = d - timedelta(days=1)
    while not is_trading_day(prev_day):
        prev_day = prev_day - timedelta(days=1)
    return prev_day


def trading_days_between(start: date, end: date) -> int:
    """计算两个日期之间的交易日数量 (含起止日)"""
    count = 0
    current = start
    while current <= end:
        if is_trading_day(current):
            count += 1
        current += timedelta(days=1)
    return count
