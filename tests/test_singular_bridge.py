from fractions import Fraction

import pytest

from lsss.algebra.field import QQ, Fp
from lsss.algebra.polynomial import Poly
from lsss.algebra.singular import SingularSession


@pytest.fixture(scope="module")
def session():
    with SingularSession() as s:
        yield s


@pytest.mark.integration
def test_roundtrip_fp_polynomial(session):
    field = Fp(7)
    p = Poly.from_terms([(3, (2, 0)), (5, (0, 1)), (1, (0, 0))], 2, field)
    serialized = session.serialize_poly(p)
    parsed = session.parse_poly(serialized, n_vars=2, field=field)
    assert parsed == p


@pytest.mark.integration
def test_reduced_groebner_trivial_qq(session):
    field = QQ()
    # F = {x - 1, y - 2}. Reduced lex GB = {x - 1, y - 2}.
    f1 = Poly.from_terms(
        [(Fraction(1), (1, 0)), (Fraction(-1), (0, 0))], 2, field
    )
    f2 = Poly.from_terms(
        [(Fraction(1), (0, 1)), (Fraction(-2), (0, 0))], 2, field
    )
    G = session.reduced_groebner([f1, f2], n_vars=2, field=field, order="lex")
    assert len(G) == 2
    # Singular returns them sorted; just check sets
    assert set((tuple(p.terms),) for p in [f1, f2]) == set(
        (tuple(p.terms),) for p in G
    )


@pytest.mark.integration
def test_reduced_groebner_reduces_redundant_fp(session):
    field = Fp(7)
    # F = {x, 2x}. Reduced GB = {x}.
    f1 = Poly.from_terms([(1, (1,))], 1, field)
    f2 = Poly.from_terms([(2, (1,))], 1, field)
    G = session.reduced_groebner([f1, f2], n_vars=1, field=field, order="lex")
    assert len(G) == 1
    assert G[0].terms == ((1, (1,)),)


@pytest.mark.integration
def test_ideal_equal_true(session):
    field = Fp(7)
    f1 = Poly.from_terms([(1, (1,)), (6, (0,))], 1, field)  # x - 1 mod 7
    f2 = Poly.from_terms([(2, (1,)), (5, (0,))], 1, field)  # 2x - 2 mod 7
    assert session.ideal_equal([f1], [f2], n_vars=1, field=field)


@pytest.mark.integration
def test_ideal_equal_false(session):
    field = Fp(7)
    f1 = Poly.from_terms([(1, (1,))], 1, field)  # x
    f2 = Poly.from_terms([(1, (1,)), (1, (0,))], 1, field)  # x + 1
    assert not session.ideal_equal([f1], [f2], n_vars=1, field=field)
