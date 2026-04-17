from fractions import Fraction

import pytest

from lsss.algebra.field import QQ, Fp
from lsss.algebra.polynomial import Poly, lex_compare


def test_lex_compare_single_var():
    # x^3 > x^2 > x > 1 under lex
    assert lex_compare((3,), (2,)) > 0
    assert lex_compare((1,), (0,)) > 0
    assert lex_compare((2,), (2,)) == 0


def test_lex_compare_multi_var():
    # x1^2 > x1 x2 > x2^2 (x1 has index 0, higher priority)
    assert lex_compare((2, 0), (1, 1)) > 0
    assert lex_compare((1, 1), (0, 2)) > 0


def test_poly_normalizes_and_sorts_terms():
    field = Fp(7)
    # 2*x^2 + 3 + x^2 -> 3*x^2 + 3 after combining, sorted high-to-low
    p = Poly.from_terms(
        [(2, (2,)), (3, (0,)), (1, (2,))], n_vars=1, field=field
    )
    assert p.terms == ((3, (2,)), (3, (0,)))


def test_poly_drops_zero_terms():
    field = Fp(7)
    # 7*x (which is 0 mod 7) + 2 -> just 2
    p = Poly.from_terms([(7, (1,)), (2, (0,))], n_vars=1, field=field)
    assert p.terms == ((2, (0,)),)


def test_poly_zero_is_empty_terms():
    field = Fp(7)
    p = Poly.zero(n_vars=2, field=field)
    assert p.terms == ()
    assert p.is_zero()


def test_poly_add_fp():
    field = Fp(7)
    p1 = Poly.from_terms([(3, (1,)), (5, (0,))], 1, field)
    p2 = Poly.from_terms([(4, (1,)), (2, (0,))], 1, field)
    # (3+4) x + (5+2) = 7x + 7 = 0 in Fp(7)
    assert (p1 + p2).is_zero()


def test_poly_mul_qq():
    field = QQ()
    # (1/2 x + 1) * (x - 1) = 1/2 x^2 + 1/2 x - 1
    p1 = Poly.from_terms([(Fraction(1, 2), (1,)), (Fraction(1), (0,))], 1, field)
    p2 = Poly.from_terms([(Fraction(1), (1,)), (Fraction(-1), (0,))], 1, field)
    product = p1 * p2
    assert product.terms == (
        (Fraction(1, 2), (2,)),
        (Fraction(1, 2), (1,)),
        (Fraction(-1), (0,)),
    )


def test_poly_leading_term():
    field = Fp(7)
    p = Poly.from_terms([(3, (1, 0)), (2, (0, 2)), (1, (2, 0))], 2, field)
    lc, lm = p.leading_term()
    assert lm == (2, 0)
    assert lc == 1


def test_poly_rejects_mismatched_nvars():
    field = Fp(7)
    with pytest.raises(ValueError):
        Poly.from_terms([(1, (1, 0, 0))], n_vars=2, field=field)


def test_poly_equality():
    field = Fp(7)
    p1 = Poly.from_terms([(1, (2,)), (3, (0,))], 1, field)
    p2 = Poly.from_terms([(3, (0,)), (1, (2,))], 1, field)
    assert p1 == p2
