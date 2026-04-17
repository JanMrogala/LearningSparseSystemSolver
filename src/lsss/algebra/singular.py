from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable

import pexpect

from lsss.algebra.field import Coeff, Field, Fp, QQ, RR
from lsss.algebra.polynomial import Poly

_PROMPT = r">\s*"

# Singular emits ANSI bracket-paste-mode sequences around output.
# Strip all ANSI escape sequences and carriage returns before parsing.
_ANSI_ESCAPE = re.compile(r"\x1b\[[^a-zA-Z]*[a-zA-Z]|\x1b\[\?[0-9]+[hl]")


def _clean(text: str) -> str:
    """Remove ANSI escape codes and carriage returns from Singular output."""
    text = _ANSI_ESCAPE.sub("", text)
    text = text.replace("\r", "")
    return text.strip()


@dataclass
class SingularSession:
    """Long-lived Singular subprocess.

    Use as a context manager:
        with SingularSession() as s:
            G = s.reduced_groebner(F, n_vars, field, order="lex")
    """

    binary: str = "Singular"
    timeout_seconds: int = 30
    _child: pexpect.spawn | None = field(default=None, init=False, repr=False)

    def __enter__(self):
        self._child = pexpect.spawn(
            f"{self.binary} -q",  # -q = quiet, suppress banner
            timeout=self.timeout_seconds,
            encoding="utf-8",
        )
        self._child.expect(_PROMPT)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._child is not None:
            self._child.sendline("quit;")
            self._child.close(force=True)
        self._child = None

    # ------ Serialization ----------------------------------------------------

    def _var_names(self, n_vars: int) -> list[str]:
        return [f"x{i}" for i in range(n_vars)]

    def _setup_ring(self, n_vars: int, field: Field, order: str) -> None:
        vars_str = ",".join(self._var_names(n_vars))
        if isinstance(field, Fp):
            char = str(field.p)
        elif isinstance(field, QQ):
            char = "0"
        elif isinstance(field, RR):
            char = "real"
        else:
            raise TypeError(f"Unsupported field {field!r}")
        order_str = {"lex": "lp", "grlex": "Dp", "grevlex": "dp"}[order]
        self._send(f"ring R = {char},({vars_str}),{order_str};")

    def serialize_poly(self, p: Poly) -> str:
        if p.is_zero():
            return "0"
        parts: list[str] = []
        for c, exps in p.terms:
            parts.append(self._term_to_singular(c, exps, p.field))
        return "+".join(parts).replace("+-", "-")

    def _term_to_singular(self, c: Coeff, exps: tuple[int, ...], field: Field) -> str:
        if isinstance(field, Fp):
            coeff_str = str(int(c) % field.p)
        elif isinstance(field, QQ):
            frac = Fraction(c)
            coeff_str = f"({frac.numerator}/{frac.denominator})"
        elif isinstance(field, RR):
            coeff_str = repr(float(c))
        else:
            raise TypeError(f"Unsupported field {field!r}")
        mono_parts = []
        for i, e in enumerate(exps):
            if e == 0:
                continue
            mono_parts.append(f"x{i}^{e}" if e > 1 else f"x{i}")
        if not mono_parts:
            return coeff_str
        mono = "*".join(mono_parts)
        if coeff_str == "1":
            return mono
        if coeff_str == "-1":
            return f"-{mono}"
        return f"{coeff_str}*{mono}"

    def parse_poly(self, s: str, n_vars: int, field: Field) -> Poly:
        """Parse a Singular-output polynomial string back to Poly.

        Handles standard Singular output: terms separated by '+' or '-',
        each term of the form "[coeff][*]x0^e0*x1^e1*...".
        Also strips ANSI escape sequences that Singular may include.
        """
        s = _clean(s).replace(" ", "")
        if s in ("0", ""):
            return Poly.zero(n_vars=n_vars, field=field)
        # Attach a leading '+' so every term has a sign prefix.
        if not s.startswith(("+", "-")):
            s = "+" + s
        # Split on top-level sign boundaries.
        tokens = re.findall(r"[+-][^+-]+", s)
        terms: list[tuple[Coeff, tuple[int, ...]]] = []
        for tok in tokens:
            terms.append(self._parse_term(tok, n_vars, field))
        return Poly.from_terms(terms, n_vars=n_vars, field=field)

    def _parse_term(
        self, tok: str, n_vars: int, field: Field
    ) -> tuple[Coeff, tuple[int, ...]]:
        sign = 1
        if tok.startswith("+"):
            tok = tok[1:]
        elif tok.startswith("-"):
            sign = -1
            tok = tok[1:]
        # Split into coefficient and monomial factors.
        parts = tok.split("*")
        exps = [0] * n_vars
        coeff_raw: str | None = None
        for part in parts:
            m = re.fullmatch(r"x(\d+)(?:\^(\d+))?", part)
            if m:
                idx = int(m.group(1))
                exp = int(m.group(2) or "1")
                exps[idx] += exp
            else:
                coeff_raw = part if coeff_raw is None else coeff_raw + "*" + part
        if coeff_raw is None:
            coeff_val: Coeff = field.one()
        else:
            coeff_val = self._parse_coeff_str(coeff_raw, field)
        if sign == -1:
            coeff_val = field.neg(coeff_val)
        return coeff_val, tuple(exps)

    def _parse_coeff_str(self, s: str, field: Field) -> Coeff:
        if isinstance(field, Fp):
            return int(s) % field.p
        if isinstance(field, QQ):
            s = s.strip("()")
            if "/" in s:
                num, den = s.split("/")
                return Fraction(int(num), int(den))
            return Fraction(int(s))
        if isinstance(field, RR):
            return float(s)
        raise TypeError(f"Unsupported field {field!r}")

    # ------ Protocol ---------------------------------------------------------

    def _send(self, cmd: str) -> str:
        """Send a command and return cleaned output up to the next prompt."""
        assert self._child is not None
        self._child.sendline(cmd)
        self._child.expect(_PROMPT)
        before = self._child.before or ""
        cleaned = _clean(before)
        # Strip the echoed command if it appears as the first non-empty line.
        lines = cleaned.splitlines()
        if lines and lines[0].strip() == cmd.strip():
            lines = lines[1:]
        return "\n".join(lines).strip()

    # ------ Operations -------------------------------------------------------

    def _has_ring(self) -> bool:
        out = self._send('if (defined(R)) { "yes"; } else { "no"; };')
        return "yes" in out

    def reduced_groebner(
        self,
        F: Iterable[Poly],
        n_vars: int,
        field: Field,
        order: str = "lex",
    ) -> list[Poly]:
        F = list(F)
        if self._has_ring():
            self._send("kill R;")
        self._setup_ring(n_vars, field, order)
        if not F:
            return []
        poly_strs = [self.serialize_poly(p) for p in F]
        self._send("ideal I = " + ",".join(poly_strs) + ";")
        self._send("option(redSB);")
        self._send("ideal G = std(I);")
        self._send("G = simplify(G, 1);")  # 1 = remove zero entries
        size_str = self._send("size(G);")
        size = int(size_str.splitlines()[-1].strip())
        out: list[Poly] = []
        for i in range(1, size + 1):
            poly_str = self._send(f"print(G[{i}]);")
            out.append(self.parse_poly(poly_str, n_vars=n_vars, field=field))
        return out

    def ideal_equal(
        self,
        F1: Iterable[Poly],
        F2: Iterable[Poly],
        n_vars: int,
        field: Field,
        order: str = "lex",
    ) -> bool:
        g1 = self.reduced_groebner(F1, n_vars, field, order)
        g2 = self.reduced_groebner(F2, n_vars, field, order)
        return _poly_set_equal(g1, g2)


def _poly_set_equal(a: list[Poly], b: list[Poly]) -> bool:
    if len(a) != len(b):
        return False
    key = lambda p: tuple(p.terms)
    return sorted((key(p) for p in a)) == sorted((key(p) for p in b))
