from __future__ import annotations

import random

from lsss.algebra.field import Field
from lsss.algebra.polynomial import Poly
from lsss.algebra.sampling import (
    sample_permutation_matrix,
    sample_random_poly,
    sample_unimodular_upper_triangular,
)


def _univariate_in_last(
    n_vars: int,
    degree: int,
    field: Field,
    rng: random.Random,
    monic: bool,
) -> Poly:
    """Sample a polynomial of exact degree in x_{n-1} only."""
    from lsss.algebra.sampling import _sample_nonzero_coeff

    if degree < 0:
        return Poly.zero(n_vars=n_vars, field=field)
    terms = []
    for d in range(degree + 1):
        exps = tuple(0 if i < n_vars - 1 else d for i in range(n_vars))
        if d == degree and monic:
            terms.append((field.one(), exps))
        else:
            if rng.random() < 0.9:  # mostly dense, but allow some sparsity
                c = _sample_nonzero_coeff(field, rng)
                terms.append((c, exps))
    if not any(sum(exps) > 0 for _, exps in terms):
        # Guarantee a nonzero constant if we generated nothing
        terms.append((_sample_nonzero_coeff(field, rng), (0,) * n_vars))
    return Poly.from_terms(terms, n_vars=n_vars, field=field)


def sample_shape_position_basis(
    n_vars: int,
    d_max: int,
    field: Field,
    rng: random.Random,
) -> list[Poly]:
    """Sample a reduced lex-GB G = {h, x_0 - g_0, ..., x_{n-2} - g_{n-2}}."""
    d_h = rng.randint(max(1, 1), d_max)
    h = _univariate_in_last(n_vars, d_h, field, rng, monic=True)
    G = [h]
    for i in range(n_vars - 1):
        # x_i - g_i  where deg(g_i) < d_h
        g_deg = rng.randint(0, d_h - 1)
        g_i = _univariate_in_last(n_vars, g_deg, field, rng, monic=False)
        xi_exps = tuple(1 if k == i else 0 for k in range(n_vars))
        xi = Poly.from_terms([(field.one(), xi_exps)], n_vars, field)
        G.append(xi - g_i)
    return G


def _matmul_poly_vec(
    M: list[list[Poly]], v: list[Poly], field: Field, n_vars: int
) -> list[Poly]:
    rows = len(M)
    cols = len(M[0])
    assert len(v) == cols
    out: list[Poly] = []
    zero = Poly.zero(n_vars=n_vars, field=field)
    for i in range(rows):
        acc = zero
        for j in range(cols):
            acc = acc + (M[i][j] * v[j])
        out.append(acc)
    return out


def _permute_vec(P: list[list[int]], v: list[Poly]) -> list[Poly]:
    n = len(P)
    out: list[Poly] = []
    for i in range(n):
        # Row i has a single 1 at some column j
        j = next(k for k in range(n) if P[i][k] == 1)
        out.append(v[j])
    return out


def _pad_rectangular_U2(
    U2_prime: list[list[Poly]], s: int, n_vars: int, field: Field
) -> list[list[Poly]]:
    """Extend n x n U2' to s x n by appending zero rows below."""
    n = len(U2_prime)
    assert all(len(row) == n for row in U2_prime)
    zero = Poly.zero(n_vars=n_vars, field=field)
    padded = [list(row) for row in U2_prime]
    for _ in range(s - n):
        padded.append([zero for _ in range(n)])
    return padded


def generate_pair(
    n_vars: int,
    d_max: int,
    d_prime: int,
    s_max: int,
    field: Field,
    density_sigma: float,
    rng: random.Random,
) -> tuple[list[Poly], list[Poly]]:
    """Generate (F, G): G is the shape-position lex-GB, F = U1 * P * U2 * G."""
    G = sample_shape_position_basis(n_vars, d_max, field, rng)
    s = rng.randint(n_vars, s_max)
    U1 = sample_unimodular_upper_triangular(
        size=s,
        n_vars=n_vars,
        d_prime=d_prime,
        field=field,
        rng=rng,
        density_sigma=density_sigma,
    )
    U2_prime = sample_unimodular_upper_triangular(
        size=n_vars,
        n_vars=n_vars,
        d_prime=d_prime,
        field=field,
        rng=rng,
        density_sigma=density_sigma,
    )
    U2 = _pad_rectangular_U2(U2_prime, s=s, n_vars=n_vars, field=field)
    P = sample_permutation_matrix(s, rng)
    # F = U1 * P * U2 * G
    step1 = _matmul_poly_vec(U2, G, field, n_vars)        # length s
    step2 = _permute_vec(P, step1)                         # length s
    F = _matmul_poly_vec(U1, step2, field, n_vars)         # length s
    return F, G
