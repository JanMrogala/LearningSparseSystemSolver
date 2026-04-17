from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Union

Coeff = Union[int, Fraction, float]


@dataclass(frozen=True)
class Fp:
    p: int

    def zero(self) -> int:
        return 0

    def one(self) -> int:
        return 1

    def add(self, a: int, b: int) -> int:
        return (a + b) % self.p

    def sub(self, a: int, b: int) -> int:
        return (a - b) % self.p

    def neg(self, a: int) -> int:
        return (-a) % self.p

    def mul(self, a: int, b: int) -> int:
        return (a * b) % self.p

    def inv(self, a: int) -> int:
        if a % self.p == 0:
            raise ZeroDivisionError("0 has no inverse in Fp")
        return pow(a, -1, self.p)

    @property
    def name(self) -> str:
        return f"Fp{self.p}"


@dataclass(frozen=True)
class QQ:
    def zero(self) -> Fraction:
        return Fraction(0)

    def one(self) -> Fraction:
        return Fraction(1)

    def add(self, a: Fraction, b: Fraction) -> Fraction:
        return a + b

    def sub(self, a: Fraction, b: Fraction) -> Fraction:
        return a - b

    def neg(self, a: Fraction) -> Fraction:
        return -a

    def mul(self, a: Fraction, b: Fraction) -> Fraction:
        return a * b

    def inv(self, a: Fraction) -> Fraction:
        if a == 0:
            raise ZeroDivisionError("0 has no inverse in QQ")
        return Fraction(a.denominator, a.numerator)

    @property
    def name(self) -> str:
        return "QQ"


@dataclass(frozen=True)
class RR:
    def zero(self) -> float:
        return 0.0

    def one(self) -> float:
        return 1.0

    def add(self, a: float, b: float) -> float:
        return a + b

    def sub(self, a: float, b: float) -> float:
        return a - b

    def neg(self, a: float) -> float:
        return -a

    def mul(self, a: float, b: float) -> float:
        return a * b

    def inv(self, a: float) -> float:
        if a == 0.0:
            raise ZeroDivisionError("0.0 has no inverse in RR")
        return 1.0 / a

    @property
    def name(self) -> str:
        return "RR"


Field = Union[Fp, QQ, RR]


def serialize_coeff(c: Coeff, field: Field):
    if isinstance(field, Fp):
        return int(c) % field.p
    if isinstance(field, QQ):
        frac = Fraction(c)
        return [str(frac.numerator), str(frac.denominator)]
    if isinstance(field, RR):
        return float(c)
    raise TypeError(f"Unknown field {field!r}")


def parse_coeff(raw, field: Field) -> Coeff:
    if isinstance(field, Fp):
        return int(raw) % field.p
    if isinstance(field, QQ):
        num, den = raw
        return Fraction(int(num), int(den))
    if isinstance(field, RR):
        return float(raw)
    raise TypeError(f"Unknown field {field!r}")
