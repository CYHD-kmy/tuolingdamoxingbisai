"""
测试 trading_calendar 模块。
"""

import sys
import os
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.trading_calendar import is_trading_day, next_trading_day, prev_trading_day


def test_weekend_not_trading():
    """周末不是交易日"""
    # 2026-05-23 是周六
    assert not is_trading_day(date(2026, 5, 23))
    # 2026-05-24 是周日
    assert not is_trading_day(date(2026, 5, 24))


def test_weekday_is_trading():
    """普通工作日是交易日"""
    # 2026-05-25 是周一（非节假日）
    assert is_trading_day(date(2026, 5, 25))


def test_holiday_not_trading():
    """法定节假日不是交易日"""
    # 2026-01-01 元旦
    assert not is_trading_day(date(2026, 1, 1))
    # 2026-02-18 春节
    assert not is_trading_day(date(2026, 2, 18))
    # 2026-10-01 国庆节
    assert not is_trading_day(date(2026, 10, 1))


def test_next_trading_day():
    """next_trading_day 跳过周末"""
    # 周五的下一个交易日是周一
    friday = date(2026, 5, 22)
    next_day = next_trading_day(friday)
    assert next_day == date(2026, 5, 25)  # 跳过周六日
    assert next_day.weekday() == 0  # 周一


def test_prev_trading_day():
    """prev_trading_day 跳过周末"""
    # 周一的上一个交易日是周五
    monday = date(2026, 5, 25)
    prev_day = prev_trading_day(monday)
    assert prev_day == date(2026, 5, 22)  # 周五
    assert prev_day.weekday() == 4


if __name__ == "__main__":
    test_weekend_not_trading()
    test_weekday_is_trading()
    test_holiday_not_trading()
    test_next_trading_day()
    test_prev_trading_day()
    print("trading_calendar: 全部通过")
