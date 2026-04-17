import random

import pytest

from lsss.algebra.field import Fp
from lsss.algebra.shape_position import generate_pair
from lsss.algebra.singular import SingularSession
from lsss.algebra.verify import check_generated_pair, check_reduced_gb


@pytest.fixture(scope="module")
def session():
    with SingularSession() as s:
        yield s


@pytest.mark.integration
def test_check_reduced_gb_accepts_shape_position(session):
    rng = random.Random(0)
    field = Fp(7)
    from lsss.algebra.shape_position import sample_shape_position_basis

    G = sample_shape_position_basis(n_vars=2, d_max=4, field=field, rng=rng)
    assert check_reduced_gb(session, G, n_vars=2, field=field, order="lex")


@pytest.mark.integration
def test_check_generated_pair_passes_on_algorithm_output(session):
    rng = random.Random(42)
    field = Fp(7)
    for _ in range(5):
        F, G = generate_pair(
            n_vars=2, d_max=4, d_prime=2, s_max=3,
            field=field, density_sigma=1.0, rng=rng,
        )
        assert check_generated_pair(
            session, F, G, n_vars=2, field=field, order="lex"
        )


@pytest.mark.integration
def test_check_reduced_gb_rejects_redundant(session):
    # F = {x, 2x}: G = {x}, but passing F in place of G should fail the check
    field = Fp(7)
    from lsss.algebra.polynomial import Poly
    f1 = Poly.from_terms([(1, (1,))], 1, field)
    f2 = Poly.from_terms([(2, (1,))], 1, field)
    assert not check_reduced_gb(session, [f1, f2], n_vars=1, field=field, order="lex")
