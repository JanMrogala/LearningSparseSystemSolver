from __future__ import annotations

import random
from itertools import product

from lsss.algebra.field import Field, Fp, QQ, RR
from lsss.algebra.polynomial import Poly


def _all_monomials(n_vars: int, max_total_degree: int) -> list[tuple[int, ...]]:
    """All exponent vectors (e0, ..., e_{n-1}) with sum <= max_total_degree."""
    out: list[tuple[int, ...]] = []
    for exps in product(range(max_total_degree + 1), repeat=n_vars):
        if sum(exps) <= max_total_degree:
            out.append(exps)
    return out


def _sample_nonzero_coeff(field: Field, rng: random.Random) -> int | float:
    if isinstance(field, Fp):
        return rng.randint(1, field.p - 1)
    if isinstance(field, QQ):
        from fractions import Fraction

        num = rng.randint(-5, 5)
        while num == 0:
            num = rng.randint(-5, 5)
        den = rng.randint(1, 5)
        return Fraction(num, den)
    if isinstance(field, RR):
        return rng.uniform(-5.0, 5.0)
    raise TypeError(f"Unsupported field {field!r}")


def sample_random_poly(
    n_vars: int,
    max_total_degree: int,
    field: Field,
    rng: random.Random,
    density_sigma: float = 1.0,
) -> Poly:
    monomials = _all_monomials(n_vars, max_total_degree)
    terms = []
    for m in monomials:
        if rng.random() < density_sigma:
            c = _sample_nonzero_coeff(field, rng)
            terms.append((c, m))
    return Poly.from_terms(terms, n_vars=n_vars, field=field)


def sample_permutation_matrix(n: int, rng: random.Random) -> list[list[int]]:
    perm = list(range(n))
    rng.shuffle(perm)
    P = [[0] * n for _ in range(n)]
    for i, j in enumerate(perm):
        P[i][j] = 1
    return P


def sample_unimodular_upper_triangular(
    size: int,
    n_vars: int,
    d_prime: int,
    field: Field,
    rng: random.Random,
    density_sigma: float = 1.0,
) -> list[list[Poly]]:
    """Upper-triangular polynomial matrix with all-one diagonal.

    Entries strictly above the diagonal are random polynomials of
    total degree <= d_prime sampled with per-monomial probability
    density_sigma. Diagonal is the constant polynomial 1. Strictly
    lower-triangular entries are zero. Such matrices have det=1 and
    are called unimodular upper-triangular (paper Sec. 4.3).
    """
    one = Poly.from_terms([(field.one(), (0,) * n_vars)], n_vars, field)
    zero = Poly.zero(n_vars=n_vars, field=field)
    M: list[list[Poly]] = [[zero for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if i == j:
                M[i][j] = one
            elif j > i:
                M[i][j] = sample_random_poly(
                    n_vars=n_vars,
                    max_total_degree=d_prime,
                    field=field,
                    rng=rng,
                    density_sigma=density_sigma,
                )
    return M
