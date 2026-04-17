from __future__ import annotations

from lsss.algebra.field import Field
from lsss.algebra.polynomial import Poly
from lsss.algebra.singular import SingularSession, _poly_set_equal


def check_reduced_gb(
    session: SingularSession,
    G: list[Poly],
    n_vars: int,
    field: Field,
    order: str = "lex",
) -> bool:
    """Return True iff G is the reduced Gröbner basis of <G>."""
    recomputed = session.reduced_groebner(G, n_vars, field, order)
    return _poly_set_equal(G, recomputed)


def check_generated_pair(
    session: SingularSession,
    F: list[Poly],
    G: list[Poly],
    n_vars: int,
    field: Field,
    order: str = "lex",
) -> bool:
    """Return True iff G is reduced GB of <G> and <F> = <G>."""
    if not check_reduced_gb(session, G, n_vars, field, order):
        return False
    return session.ideal_equal(F, G, n_vars, field, order)
