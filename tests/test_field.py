from fractions import Fraction

import pytest

from lsss.algebra.field import QQ, RR, Fp, parse_coeff, serialize_coeff


def test_fp_arithmetic_wraps():
    field = Fp(7)
    assert field.add(5, 4) == 2
    assert field.mul(3, 5) == 1
    assert field.neg(3) == 4
    assert field.inv(3) == 5  # 3 * 5 = 15 = 1 mod 7


def test_fp_inverse_zero_raises():
    field = Fp(7)
    with pytest.raises(ZeroDivisionError):
        field.inv(0)


def test_qq_arithmetic_exact():
    field = QQ()
    assert field.add(Fraction(1, 2), Fraction(1, 3)) == Fraction(5, 6)
    assert field.mul(Fraction(2, 3), Fraction(3, 4)) == Fraction(1, 2)
    assert field.inv(Fraction(2, 3)) == Fraction(3, 2)


def test_rr_arithmetic_float():
    field = RR()
    assert field.add(0.5, 0.25) == 0.75
    assert field.mul(2.0, 3.0) == 6.0


def test_serialize_roundtrip_fp():
    field = Fp(7)
    assert parse_coeff(serialize_coeff(3, field), field) == 3


def test_serialize_roundtrip_qq():
    field = QQ()
    c = Fraction(-5, 3)
    assert parse_coeff(serialize_coeff(c, field), field) == c


def test_serialize_qq_format_is_string_pair():
    field = QQ()
    assert serialize_coeff(Fraction(2, 7), field) == ["2", "7"]
