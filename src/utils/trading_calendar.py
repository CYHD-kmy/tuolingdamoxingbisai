"""
A 股交易日历 — 判断指定日期是否为交易日。

数据来源: 基于中国法定节假日和交易所公告的简化实现。
支持: 法定节假日、调休上班日 (补班周六)、特殊休市日。
生产环境建议使用 akshare 的 tool_trade_date_hist_sina() 获取官方交易日历。
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

logger = logging.getLogger(__name__)

_HOLIDAYS_CACHE: dict[int, set[str]] = {}

# ── 法定节假日 (国务院已公布) ──────────────────
# 元旦: 1月1-3日 / 春节: 2月15-21日 / 清明节: 4月5-7日
# 劳动节: 5月1-5日 / 端午节: 6月19-21日 / 中秋节+国庆节: 9月25日-10月8日

_FIXED_HOLIDAYS: dict[int, list[tuple[int, int]]] = {
    2026: [
        (1, 1), (1, 2), (1, 3),                                      # 元旦
        (2, 15), (2, 16), (2, 17), (2, 18), (2, 19), (2, 20), (2, 21),  # 春节
        (4, 5), (4, 6), (4, 7),                                       # 清明节
        (5, 1), (5, 2), (5, 3), (5, 4), (5, 5),                     # 劳动节
        (6, 19), (6, 20), (6, 21),                                    # 端午节
        (9, 25), (9, 26), (9, 27), (9, 28), (9, 29), (9, 30),       # 中秋节+国庆节
        (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8),
    ],
}

# ── 调休上班日 (周末补班, 实际是交易日) ───────
_WORK_SATURDAYS: dict[int, list[tuple[int, int]]] = {
    2026: [
        (2, 14),   # 春节前补班 (周六)
        (4, 25),   # 劳动节前补班 (周六)
        (6, 13),   # 端午节前补班 (周六)
        (9, 19),   # 中秋节前补班 (周六)
        (10, 10),  # 国庆节后补班 (周六)
    ],
}

# ── 特殊休市日 (台风/重大事件) ────────────────
_SPECIAL_CLOSED: dict[int, list[tuple[int, int]]] = {
    2026: [],
}


def _build_holidays(year: int) -> set[str]:
    if year in _HOLIDAYS_CACHE:
        return _HOLIDAYS_CACHE[year]
    holidays: set[str] = set()
    if year in _FIXED_HOLIDAYS:
        for m, d in _FIXED_HOLIDAYS[year]:
            holidays.add(date(year, m, d).isoformat())
    if year in _SPECIAL_CLOSED:
        for m, d in _SPECIAL_CLOSED[year]:
            holidays.add(date(year, m, d).isoformat())
    _HOLIDAYS_CACHE[year] = holidays
    return holidays


def _get_holidays(year: int) -> set[str]:
    if year in _HOLIDAYS_CACHE:
        return _HOLIDAYS_CACHE[year]
    if year not in _FIXED_HOLIDAYS:
        logger.warning(
            "交易日历: %d 年无硬编码节假日数据，所有工作日均视为交易日。"
            "建议使用 akshare.tool_trade_date_hist_sina() 获取官方交易日历。", year
        )
    return _build_holidays(year)


def _is_work_saturday(d: date) -> bool:
    year_saturdays = _WORK_SATURDAYS.get(d.year, [])
    for m, day in year_saturdays:
        if d.month == m and d.day == day:
            return True
    return False


def _is_special_closed(d: date) -> bool:
    closed = _SPECIAL_CLOSED.get(d.year, [])
    for m, day in closed:
        if d.month == m and d.day == day:
            return True
    return False


def is_trading_day(d: date | datetime | None = None) -> bool:
    """
    判断是否为 A 股交易日。

    规则:
    - 调休上班的周末: 交易日
    - 周六/周日: 非交易日
    - 特殊休市日: 非交易日
    - 法定节假日: 非交易日
    - 其余: 交易日
    """
    if d is None:
        d = date.today()
    if isinstance(d, datetime):
        d = d.date()

    # 调休上班日 (补班周六)
    if _is_work_saturday(d):
        return True

    # 周末
    if d.weekday() >= 5:
        return False

    # 特殊休市
    if _is_special_closed(d):
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
