from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from lsss.algebra.field import Coeff, Field, Fp, QQ, RR

ExponentVector = tuple[int, ...]
Term = tuple[Coeff, ExponentVector]


def lex_compare(a: ExponentVector, b: ExponentVector) -> int:
    """Lex comparator: higher index 0 wins, ties broken by index 1, etc."""
    for x, y in zip(a, b, strict=True):
        if x != y:
            return 1 if x > y else -1
    return 0


@dataclass(frozen=True)
class Poly:
    terms: tuple[Term, ...]  # sorted descending by lex order
    n_vars: int
    field: Field

    @classmethod
    def zero(cls, n_vars: int, field: Field) -> "Poly":
        return cls(terms=(), n_vars=n_vars, field=field)

    @classmethod
    def from_terms(
        cls,
        raw_terms: Iterable[Term],
        n_vars: int,
        field: Field,
    ) -> "Poly":
        merged: dict[ExponentVector, Coeff] = {}
        for coeff, exps in raw_terms:
            if len(exps) != n_vars:
                raise ValueError(
                    f"exponent vector length {len(exps)} != n_vars {n_vars}"
                )
            prev = merged.get(exps, field.zero())
            merged[exps] = field.add(prev, coeff)
        cleaned = [
            (c, e) for e, c in merged.items() if not _coeff_is_zero(c, field)
        ]
        cleaned.sort(key=lambda t: t[1], reverse=True)  # lex order, descending
        # Convert to tuple form
        final: list[Term] = [(c, tuple(e)) for c, e in cleaned]
        return cls(terms=tuple(final), n_vars=n_vars, field=field)

    def is_zero(self) -> bool:
        return len(self.terms) == 0

    def leading_term(self) -> Term:
        if self.is_zero():
            raise ValueError("leading_term of zero polynomial")
        return self.terms[0]

    def __add__(self, other: "Poly") -> "Poly":
        _check_compat(self, other)
        return Poly.from_terms(
            list(self.terms) + list(other.terms), self.n_vars, self.field
        )

    def __sub__(self, other: "Poly") -> "Poly":
        _check_compat(self, other)
        negated = [(self.field.neg(c), e) for c, e in other.terms]
        return Poly.from_terms(
            list(self.terms) + negated, self.n_vars, self.field
        )

    def __neg__(self) -> "Poly":
        return Poly.from_terms(
            [(self.field.neg(c), e) for c, e in self.terms],
            self.n_vars,
            self.field,
        )

    def __mul__(self, other: "Poly") -> "Poly":
        _check_compat(self, other)
        out: list[Term] = []
        for c1, e1 in self.terms:
            for c2, e2 in other.terms:
                out.append(
                    (
                        self.field.mul(c1, c2),
                        tuple(a + b for a, b in zip(e1, e2, strict=True)),
                    )
                )
        return Poly.from_terms(out, self.n_vars, self.field)


def _check_compat(p: Poly, q: Poly) -> None:
    if p.n_vars != q.n_vars:
        raise ValueError("n_vars mismatch")
    if p.field != q.field:
        raise ValueError("field mismatch")


def _coeff_is_zero(c: Coeff, field: Field) -> bool:
    if isinstance(field, Fp):
        return c % field.p == 0
    if isinstance(field, QQ):
        return c == 0
    if isinstance(field, RR):
        return c == 0.0
    raise TypeError(f"Unknown field {field!r}")
