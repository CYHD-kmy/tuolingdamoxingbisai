"""
测试 validators 模块。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.validators import (
    validate_symbol,
    validate_volume,
    ValidationResult,
    LOT_SIZE,
)


def test_validate_symbol_valid():
    """有效的6位数字代码"""
    assert validate_symbol("600519")
    assert validate_symbol("000858")
    assert validate_symbol("300750")


def test_validate_symbol_invalid():
    """无效的代码格式"""
    assert not validate_symbol("60051")    # 5位
    assert not validate_symbol("6005191")  # 7位
    assert not validate_symbol("abc")      # 非数字


def test_validate_volume_exact():
    """股数已是100的整数倍"""
    corrected, warning = validate_volume(500)
    assert corrected == 500
    assert warning is None


def test_validate_volume_round_down():
    """股数向下取整到100的倍数"""
    corrected, warning = validate_volume(599)
    assert corrected == 500
    assert warning is not None


def test_validate_volume_zero():
    """股数为0"""
    corrected, warning = validate_volume(0)
    assert corrected == 0
    assert warning is not None


def test_validate_volume_too_small():
    """股数小于100取整后为0"""
    corrected, warning = validate_volume(50)
    assert corrected == 0
    assert warning is not None


def test_validation_result_default():
    """ValidationResult 默认值"""
    result = ValidationResult()
    assert result.valid is True
    assert result.errors == []
    assert result.warnings == []
    assert result.corrected_decisions == []


if __name__ == "__main__":
    test_validate_symbol_valid()
    test_validate_symbol_invalid()
    test_validate_volume_exact()
    test_validate_volume_round_down()
    test_validate_volume_zero()
    test_validate_volume_too_small()
    test_validation_result_default()
    print("validators: 全部通过")
