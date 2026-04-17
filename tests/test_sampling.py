import random

import pytest

from lsss.algebra.field import Fp
from lsss.algebra.polynomial import Poly
from lsss.algebra.sampling import (
    sample_permutation_matrix,
    sample_random_poly,
    sample_unimodular_upper_triangular,
)


def test_permutation_matrix_is_permutation():
    rng = random.Random(0)
    P = sample_permutation_matrix(5, rng)
    assert len(P) == 5
    for row in P:
        assert sum(row) == 1
        assert all(v in (0, 1) for v in row)
    # Each column sums to 1
    for j in range(5):
        assert sum(P[i][j] for i in range(5)) == 1


def test_unimodular_upper_triangular_shape():
    rng = random.Random(0)
    field = Fp(7)
    U = sample_unimodular_upper_triangular(
        size=3, n_vars=2, d_prime=2, field=field, rng=rng, density_sigma=1.0
    )
    assert len(U) == 3
    assert all(len(row) == 3 for row in U)
    # Diagonal is polynomial 1 (a single term c=1, exps=(0,0))
    for i in range(3):
        assert U[i][i].terms == ((1, (0, 0)),)
    # Strictly below diagonal is zero
    for i in range(3):
        for j in range(i):
            assert U[i][j].is_zero()


def test_unimodular_upper_triangular_density_zero_gives_identity():
    rng = random.Random(0)
    field = Fp(7)
    U = sample_unimodular_upper_triangular(
        size=3, n_vars=2, d_prime=2, field=field, rng=rng, density_sigma=0.0
    )
    for i in range(3):
        for j in range(3):
            if i == j:
                assert U[i][j].terms == ((1, (0, 0)),)
            else:
                assert U[i][j].is_zero()


def test_random_poly_respects_max_degree():
    rng = random.Random(0)
    field = Fp(7)
    for _ in range(20):
        p = sample_random_poly(
            n_vars=3, max_total_degree=4, field=field, rng=rng, density_sigma=1.0
        )
        for _, exps in p.terms:
            assert sum(exps) <= 4


def test_random_poly_density_zero_is_zero():
    rng = random.Random(0)
    field = Fp(7)
    p = sample_random_poly(
        n_vars=2, max_total_degree=3, field=field, rng=rng, density_sigma=0.0
    )
    assert p.is_zero()
