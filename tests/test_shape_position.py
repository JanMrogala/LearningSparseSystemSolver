import random

from lsss.algebra.field import Fp
from lsss.algebra.shape_position import generate_pair, sample_shape_position_basis


def test_sampled_G_has_shape_position_form():
    rng = random.Random(0)
    field = Fp(7)
    G = sample_shape_position_basis(
        n_vars=3, d_max=4, field=field, rng=rng
    )
    # Expected: {h(x_{n-1}), x_0 - g_0(x_{n-1}), x_1 - g_1(x_{n-1})}
    assert len(G) == 3
    # First polynomial is univariate in x_{n-1}
    h = G[0]
    for _, exps in h.terms:
        assert exps[0] == 0 and exps[1] == 0
    # Leading coefficient of h is 1 (monic)
    lc, _ = h.leading_term()
    assert lc == 1
    # deg(h) must exceed deg of each g_i
    h_deg = max(exps[-1] for _, exps in h.terms)
    for gi in G[1:]:
        gi_deg = max(exps[-1] for _, exps in gi.terms if exps[-1] > 0) if any(
            exps[-1] > 0 for _, exps in gi.terms
        ) else 0
        assert h_deg > gi_deg


def test_generate_pair_returns_non_trivial_F():
    rng = random.Random(42)
    field = Fp(7)
    F, G = generate_pair(
        n_vars=2,
        d_max=4,
        d_prime=2,
        s_max=3,
        field=field,
        density_sigma=1.0,
        rng=rng,
    )
    assert len(G) == 2
    assert len(F) >= 2
    # F should have at least one nonzero polynomial
    assert any(not f.is_zero() for f in F)


def test_generate_pair_F_differs_from_G():
    """With d_prime >= 1 and density > 0, F should almost surely != G."""
    rng = random.Random(7)
    field = Fp(7)
    F, G = generate_pair(
        n_vars=2,
        d_max=4,
        d_prime=2,
        s_max=3,
        field=field,
        density_sigma=1.0,
        rng=rng,
    )
    # Compare as sets of term-tuples
    f_set = {tuple(p.terms) for p in F if not p.is_zero()}
    g_set = {tuple(p.terms) for p in G}
    assert f_set != g_set
