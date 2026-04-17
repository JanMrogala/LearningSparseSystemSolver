# Gröbner-Basis MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end MVP pipeline that generates (F, G) Gröbner-basis pairs via the paper's backward algorithm, trains a small Transformer to map F → G, and evaluates it against three metrics including a Singular-based ideal-equality check. Target slice: shape position, 𝔽₇, n=2, discrete embedding.

**Architecture:** Two-stage pipeline with on-disk JSONL dataset. `gen_dataset.py` and `evaluate.py` call Singular via a long-lived pexpect subprocess; `train.py` has zero algebra dependency. PyTorch Lightning wraps a from-scratch encoder-decoder Transformer; Hydra/OmegaConf drives configuration; W&B logs training and eval. See `docs/superpowers/specs/2026-04-18-learning-to-compute-grobner-bases-design.md` for full design.

**Tech Stack:** Python 3.11, PyTorch 2.x (from scratch, no HuggingFace), PyTorch Lightning, Weights & Biases, Hydra/OmegaConf, Singular (via pexpect), pytest.

---

## Task 0: Project Scaffolding and Environment Setup

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore` (augment existing)
- Create: `src/lsss/__init__.py`
- Create: `tests/__init__.py`
- Create: `README.md` (augment existing)

- [ ] **Step 1: Install Singular (system dependency)**

Check current state:

```bash
which Singular
```

If the command prints nothing, install Singular. On Debian/Ubuntu:

```bash
sudo apt-get update && sudo apt-get install -y singular
```

On Arch: `sudo pacman -S singular`. On macOS: `brew install Singular`. Confirm:

```bash
Singular --version
```

Expected: a line like `Singular for x86_64-Linux version 4.3.x ...`.

- [ ] **Step 2: Create `pyproject.toml`**

File: `/home/jan/projects/CIIRC/colabs/tpajdla/LearningSparseSystemSolver/pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "lsss"
version = "0.1.0"
description = "Learning Sparse System Solver: Gröbner-basis Transformer reimplementation"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2",
    "pytorch-lightning>=2.2",
    "wandb>=0.17",
    "hydra-core>=1.3",
    "omegaconf>=2.3",
    "pexpect>=4.9",
    "numpy>=1.26",
    "tqdm>=4.66",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-xdist>=3.5",
    "ruff>=0.5",
]

[tool.hatch.build.targets.wheel]
packages = ["src/lsss"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "integration: requires Singular subprocess",
    "gpu: requires CUDA-capable GPU",
]

[tool.ruff]
line-length = 100
target-version = "py311"
```

- [ ] **Step 3: Augment `.gitignore`**

Add the following lines to `/home/jan/projects/CIIRC/colabs/tpajdla/LearningSparseSystemSolver/.gitignore` (append, do not overwrite existing entries):

```
# Project outputs
outputs/
wandb/
data/
*.ckpt
__pycache__/
*.egg-info/
.pytest_cache/
```

- [ ] **Step 4: Create package skeleton**

Create these files, all with content `""` (empty):

- `src/lsss/__init__.py`
- `src/lsss/algebra/__init__.py`
- `src/lsss/data/__init__.py`
- `src/lsss/model/__init__.py`
- `src/lsss/train/__init__.py`
- `src/lsss/eval/__init__.py`
- `src/lsss/utils/__init__.py`
- `tests/__init__.py`

- [ ] **Step 5: Install the package in editable mode with dev extras**

```bash
cd /home/jan/projects/CIIRC/colabs/tpajdla/LearningSparseSystemSolver
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Expected: installs torch, pytorch-lightning, wandb, hydra-core, pexpect, numpy, tqdm, pytest, pytest-xdist, ruff. No errors.

- [ ] **Step 6: Verify pytest discovers an empty test suite**

```bash
pytest
```

Expected: `no tests ran` (exit code 5). This confirms pytest is configured correctly.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore src/ tests/
git commit -m "chore: bootstrap Python package and dev dependencies"
```

---

## Task 1: Field and Coefficient Types

**Files:**
- Create: `src/lsss/algebra/field.py`
- Test: `tests/test_field.py`

- [ ] **Step 1: Write the failing test**

File: `tests/test_field.py`

```python
from fractions import Fraction

import pytest

from lsss.algebra.field import QQ, RR, Fp, parse_coeff, serialize_coeff


def test_fp_arithmetic_wraps():
    field = Fp(7)
    assert field.add(5, 4) == 2
    assert field.mul(3, 5) == 1
    assert field.neg(3) == 4
    assert field.inv(3) == 5  # 3 * 5 = 15 = 1 mod 7


def test_fp_inverse_zero_raises():
    field = Fp(7)
    with pytest.raises(ZeroDivisionError):
        field.inv(0)


def test_qq_arithmetic_exact():
    field = QQ()
    assert field.add(Fraction(1, 2), Fraction(1, 3)) == Fraction(5, 6)
    assert field.mul(Fraction(2, 3), Fraction(3, 4)) == Fraction(1, 2)
    assert field.inv(Fraction(2, 3)) == Fraction(3, 2)


def test_rr_arithmetic_float():
    field = RR()
    assert field.add(0.5, 0.25) == 0.75
    assert field.mul(2.0, 3.0) == 6.0


def test_serialize_roundtrip_fp():
    field = Fp(7)
    assert parse_coeff(serialize_coeff(3, field), field) == 3


def test_serialize_roundtrip_qq():
    field = QQ()
    c = Fraction(-5, 3)
    assert parse_coeff(serialize_coeff(c, field), field) == c


def test_serialize_qq_format_is_string_pair():
    field = QQ()
    assert serialize_coeff(Fraction(2, 7), field) == ["2", "7"]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_field.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'lsss.algebra.field'`.

- [ ] **Step 3: Write the implementation**

File: `src/lsss/algebra/field.py`

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_field.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/lsss/algebra/field.py tests/test_field.py
git commit -m "feat(algebra): add field types (Fp, QQ, RR) with coeff serialization"
```

---

## Task 2: Polynomial Dataclass and Arithmetic

**Files:**
- Create: `src/lsss/algebra/polynomial.py`
- Test: `tests/test_polynomial.py`

- [ ] **Step 1: Write the failing test**

File: `tests/test_polynomial.py`

```python
from fractions import Fraction

import pytest

from lsss.algebra.field import QQ, Fp
from lsss.algebra.polynomial import Poly, lex_compare


def test_lex_compare_single_var():
    # x^3 > x^2 > x > 1 under lex
    assert lex_compare((3,), (2,)) > 0
    assert lex_compare((1,), (0,)) > 0
    assert lex_compare((2,), (2,)) == 0


def test_lex_compare_multi_var():
    # x1^2 > x1 x2 > x2^2 (x1 has index 0, higher priority)
    assert lex_compare((2, 0), (1, 1)) > 0
    assert lex_compare((1, 1), (0, 2)) > 0


def test_poly_normalizes_and_sorts_terms():
    field = Fp(7)
    # 2*x^2 + 3 + x^2 -> 3*x^2 + 3 after combining, sorted high-to-low
    p = Poly.from_terms(
        [(2, (2,)), (3, (0,)), (1, (2,))], n_vars=1, field=field
    )
    assert p.terms == ((3, (2,)), (3, (0,)))


def test_poly_drops_zero_terms():
    field = Fp(7)
    # 7*x (which is 0 mod 7) + 2 -> just 2
    p = Poly.from_terms([(7, (1,)), (2, (0,))], n_vars=1, field=field)
    assert p.terms == ((2, (0,)),)


def test_poly_zero_is_empty_terms():
    field = Fp(7)
    p = Poly.zero(n_vars=2, field=field)
    assert p.terms == ()
    assert p.is_zero()


def test_poly_add_fp():
    field = Fp(7)
    p1 = Poly.from_terms([(3, (1,)), (5, (0,))], 1, field)
    p2 = Poly.from_terms([(4, (1,)), (2, (0,))], 1, field)
    # (3+4) x + (5+2) = 7x + 7 = 0 in Fp(7)
    assert (p1 + p2).is_zero()


def test_poly_mul_qq():
    field = QQ()
    # (1/2 x + 1) * (x - 1) = 1/2 x^2 + 1/2 x - 1
    p1 = Poly.from_terms([(Fraction(1, 2), (1,)), (Fraction(1), (0,))], 1, field)
    p2 = Poly.from_terms([(Fraction(1), (1,)), (Fraction(-1), (0,))], 1, field)
    product = p1 * p2
    assert product.terms == (
        (Fraction(1, 2), (2,)),
        (Fraction(1, 2), (1,)),
        (Fraction(-1), (0,)),
    )


def test_poly_leading_term():
    field = Fp(7)
    p = Poly.from_terms([(3, (1, 0)), (2, (0, 2)), (1, (2, 0))], 2, field)
    lc, lm = p.leading_term()
    assert lm == (2, 0)
    assert lc == 1


def test_poly_rejects_mismatched_nvars():
    field = Fp(7)
    with pytest.raises(ValueError):
        Poly.from_terms([(1, (1, 0, 0))], n_vars=2, field=field)


def test_poly_equality():
    field = Fp(7)
    p1 = Poly.from_terms([(1, (2,)), (3, (0,))], 1, field)
    p2 = Poly.from_terms([(3, (0,)), (1, (2,))], 1, field)
    assert p1 == p2
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_polynomial.py -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Write the implementation**

File: `src/lsss/algebra/polynomial.py`

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_polynomial.py -v
```

Expected: all 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/lsss/algebra/polynomial.py tests/test_polynomial.py
git commit -m "feat(algebra): add Poly dataclass with lex order, arithmetic"
```

---

## Task 3: Singular Subprocess Bridge

**Files:**
- Create: `src/lsss/algebra/singular.py`
- Test: `tests/test_singular_bridge.py`

- [ ] **Step 1: Write the failing integration test**

File: `tests/test_singular_bridge.py`

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_singular_bridge.py -v -m integration
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the Singular bridge implementation**

File: `src/lsss/algebra/singular.py`

```python
from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

import pexpect

from lsss.algebra.field import Coeff, Field, Fp, QQ, RR
from lsss.algebra.polynomial import Poly

_PROMPT = r">\s*"


@dataclass
class SingularSession:
    """Long-lived Singular subprocess.

    Use as a context manager:
        with SingularSession() as s:
            G = s.reduced_groebner(F, n_vars, field, order="lex")
    """

    binary: str = "Singular"
    timeout_seconds: int = 30
    _child: pexpect.spawn | None = None

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
        # Join with + (each part carries its own sign)
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
        """
        s = s.strip().replace(" ", "")
        if s in ("0", ""):
            return Poly.zero(n_vars=n_vars, field=field)
        # Split on signs but keep them attached. Insert separator before each + or -
        # (except leading sign).
        if not s.startswith(("+", "-")):
            s = "+" + s
        # Regex: a sign followed by anything up to the next top-level sign.
        # Singular outputs do not include parenthesized coefficients except
        # for rationals we wrote; after reduced_groebner the output is flat.
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
        # Split coefficient and monomial.
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
        """Send a command, return output up to the next prompt."""
        assert self._child is not None
        self._child.sendline(cmd)
        self._child.expect(_PROMPT)
        before = self._child.before or ""
        # Strip the echoed command and trailing prompt artifacts.
        lines = before.splitlines()
        # First line is often the echoed command; drop it if it matches.
        if lines and lines[0].strip() == cmd:
            lines = lines[1:]
        return "\n".join(lines).strip()

    # ------ Operations -------------------------------------------------------

    def reduced_groebner(
        self,
        F: Iterable[Poly],
        n_vars: int,
        field: Field,
        order: str = "lex",
    ) -> list[Poly]:
        F = list(F)
        self._send("kill R;") if self._has_ring() else None
        self._setup_ring(n_vars, field, order)
        if not F:
            return []
        # Build the ideal in Singular.
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

    def _has_ring(self) -> bool:
        # Singular's listvar shows defined vars; we use a cheaper sentinel.
        out = self._send('if (defined(R)) { "yes"; } else { "no"; };')
        return "yes" in out


def _poly_set_equal(a: list[Poly], b: list[Poly]) -> bool:
    if len(a) != len(b):
        return False
    # Canonical comparison via sorted term-tuples.
    key = lambda p: tuple(p.terms)
    return sorted((key(p) for p in a)) == sorted((key(p) for p in b))
```

- [ ] **Step 4: Run the integration test**

```bash
pytest tests/test_singular_bridge.py -v -m integration
```

Expected: all 5 tests pass. If Singular output parsing fails on a specific case, inspect the raw output by running `Singular -q` manually with the same input and fix `parse_poly`.

- [ ] **Step 5: Commit**

```bash
git add src/lsss/algebra/singular.py tests/test_singular_bridge.py
git commit -m "feat(algebra): add Singular pexpect bridge with reduced_groebner and ideal_equal"
```

---

## Task 4: Matrix Sampling (Unimodular, Permutation, Random Polys)

**Files:**
- Create: `src/lsss/algebra/sampling.py`
- Test: `tests/test_sampling.py`

- [ ] **Step 1: Write the failing test**

File: `tests/test_sampling.py`

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_sampling.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

File: `src/lsss/algebra/sampling.py`

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_sampling.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/lsss/algebra/sampling.py tests/test_sampling.py
git commit -m "feat(algebra): add matrix and polynomial samplers"
```

---

## Task 5: Shape-Position Dataset Generation (Algorithm 1)

**Files:**
- Create: `src/lsss/algebra/shape_position.py`
- Test: `tests/test_shape_position.py`

- [ ] **Step 1: Write the failing test**

File: `tests/test_shape_position.py`

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_shape_position.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

File: `src/lsss/algebra/shape_position.py`

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_shape_position.py -v
```

Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/lsss/algebra/shape_position.py tests/test_shape_position.py
git commit -m "feat(algebra): implement shape-position generator (paper Alg. 1, sans FGLM)"
```

---

## Task 6: Verification (reduced-GB and ideal-equality checks)

**Files:**
- Create: `src/lsss/algebra/verify.py`
- Test: `tests/test_verify.py`

- [ ] **Step 1: Write the failing test**

File: `tests/test_verify.py`

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_verify.py -v -m integration
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

File: `src/lsss/algebra/verify.py`

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_verify.py -v -m integration
```

Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/lsss/algebra/verify.py tests/test_verify.py
git commit -m "feat(algebra): add reduced-GB and ideal-equality verification"
```

---

## Task 7: Dataset Format (JSONL Schema)

**Files:**
- Create: `src/lsss/data/dataset_format.py`
- Test: `tests/test_dataset_format.py`

- [ ] **Step 1: Write the failing test**

File: `tests/test_dataset_format.py`

```python
from fractions import Fraction
from pathlib import Path

from lsss.algebra.field import Fp, QQ
from lsss.algebra.polynomial import Poly
from lsss.data.dataset_format import (
    SCHEMA_VERSION,
    deserialize_sample,
    serialize_sample,
    write_shard,
    read_shard,
)


def test_schema_version_is_int():
    assert isinstance(SCHEMA_VERSION, int)


def test_roundtrip_sample_fp():
    field = Fp(7)
    F = [Poly.from_terms([(3, (1, 0)), (2, (0, 1))], 2, field)]
    G = [Poly.from_terms([(1, (1, 0))], 2, field)]
    sample = {"id": 0, "F": F, "G": G, "field": field, "n": 2, "seed": 1, "verified": True}
    blob = serialize_sample(sample)
    restored = deserialize_sample(blob, field=field)
    assert restored["id"] == 0
    assert restored["F"] == F
    assert restored["G"] == G


def test_roundtrip_sample_qq():
    field = QQ()
    F = [Poly.from_terms([(Fraction(1, 2), (1, 0)), (Fraction(-3), (0, 0))], 2, field)]
    G = [Poly.from_terms([(Fraction(1), (1, 0))], 2, field)]
    sample = {"id": 7, "F": F, "G": G, "field": field, "n": 2, "seed": 2, "verified": False}
    blob = serialize_sample(sample)
    restored = deserialize_sample(blob, field=field)
    assert restored["F"] == F
    assert restored["G"] == G


def test_write_and_read_shard(tmp_path: Path):
    field = Fp(7)
    samples = []
    for i in range(3):
        F = [Poly.from_terms([(1 + i, (1,))], 1, field)]
        G = [Poly.from_terms([(1, (1,))], 1, field)]
        samples.append(
            {"id": i, "F": F, "G": G, "field": field, "n": 1, "seed": i, "verified": True}
        )
    path = tmp_path / "shard.jsonl"
    write_shard(path, samples)
    restored = list(read_shard(path, field=field))
    assert len(restored) == 3
    assert [s["id"] for s in restored] == [0, 1, 2]
    assert restored[0]["F"] == samples[0]["F"]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_dataset_format.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

File: `src/lsss/data/dataset_format.py`

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from lsss.algebra.field import Field, parse_coeff, serialize_coeff
from lsss.algebra.polynomial import Poly

SCHEMA_VERSION = 1


def _serialize_poly(p: Poly) -> list[list[Any]]:
    return [[serialize_coeff(c, p.field), list(exps)] for c, exps in p.terms]


def _deserialize_poly(raw: list[list[Any]], n_vars: int, field: Field) -> Poly:
    terms = [(parse_coeff(coeff, field), tuple(exps)) for coeff, exps in raw]
    return Poly.from_terms(terms, n_vars=n_vars, field=field)


def serialize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": sample["id"],
        "F": [_serialize_poly(p) for p in sample["F"]],
        "G": [_serialize_poly(p) for p in sample["G"]],
        "field": sample["field"].name,
        "n": sample["n"],
        "seed": sample["seed"],
        "verified": sample["verified"],
    }


def deserialize_sample(blob: dict[str, Any], field: Field) -> dict[str, Any]:
    n = blob["n"]
    return {
        "id": blob["id"],
        "F": [_deserialize_poly(raw, n_vars=n, field=field) for raw in blob["F"]],
        "G": [_deserialize_poly(raw, n_vars=n, field=field) for raw in blob["G"]],
        "field": field,
        "n": n,
        "seed": blob["seed"],
        "verified": blob["verified"],
    }


def write_shard(path: Path, samples: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(serialize_sample(s)) + "\n")


def read_shard(path: Path, field: Field) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield deserialize_sample(json.loads(line), field=field)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_dataset_format.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/lsss/data/dataset_format.py tests/test_dataset_format.py
git commit -m "feat(data): add JSONL dataset format with roundtrip tests"
```

---

## Task 8: Hydra Configuration Files

**Files:**
- Create: `configs/mvp.yaml`
- Create: `configs/paper.yaml`
- Create: `configs/eval.yaml`
- Create: `configs/data/shape_position_fp7_n2.yaml`
- Create: `configs/generation/mvp.yaml`
- Create: `configs/generation/paper.yaml`
- Create: `configs/model/mvp.yaml`
- Create: `configs/model/paper.yaml`
- Create: `configs/embedding/discrete.yaml`
- Create: `configs/embedding/hybrid.yaml`
- Create: `configs/train/mvp.yaml`
- Create: `configs/train/paper.yaml`
- Create: `configs/wandb/default.yaml`
- Test: `tests/test_configs.py`

- [ ] **Step 1: Write the failing test**

File: `tests/test_configs.py`

```python
from pathlib import Path

from hydra import compose, initialize_config_dir


CONFIG_DIR = str(Path(__file__).parent.parent / "configs")


def test_mvp_config_composes():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name="mvp")
    assert cfg.seed == 0
    assert cfg.data.field == "Fp7"
    assert cfg.data.n_vars == 2
    assert cfg.model.n_layers_encoder == 3
    assert cfg.embedding.kind == "discrete"
    assert cfg.train.epochs >= 1
    assert cfg.generation.verify.enabled is True


def test_paper_config_composes():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name="paper")
    assert cfg.model.n_layers_encoder == 6
    assert cfg.model.d_model == 512
    assert cfg.train.epochs == 8
    assert cfg.generation.verify.enabled is False


def test_embedding_override():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name="mvp", overrides=["embedding=hybrid"])
    assert cfg.embedding.kind == "hybrid"
    assert cfg.embedding.regression_loss_weight == 0.01
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_configs.py -v
```

Expected: FAIL with `MissingConfigException` (no config dir yet).

- [ ] **Step 3: Create the configs**

File: `configs/mvp.yaml`

```yaml
defaults:
  - data: shape_position_fp7_n2
  - model: mvp
  - embedding: discrete
  - train: mvp
  - generation: mvp
  - wandb: default
  - _self_

seed: 0
output_dir: ${hydra:runtime.output_dir}
```

File: `configs/paper.yaml`

```yaml
defaults:
  - data: shape_position_fp7_n2
  - model: paper
  - embedding: discrete
  - train: paper
  - generation: paper
  - wandb: default
  - _self_

seed: 0
output_dir: ${hydra:runtime.output_dir}
```

File: `configs/eval.yaml`

```yaml
defaults:
  - data: shape_position_fp7_n2
  - model: paper
  - embedding: discrete
  - wandb: default
  - _self_

seed: 0
eval:
  ckpt_path: ???
  check_ideal_equality: true
  beam_width: 1
  max_decode_len: ${model.max_tgt_len}
  output_json: ${hydra:runtime.output_dir}/results.json
```

File: `configs/data/shape_position_fp7_n2.yaml`

```yaml
kind: shape_position
field: Fp7
n_vars: 2
d_max: 5
d_prime: 3
s_max: 5
term_order: lex
coeff_range_G: [-5, 5]
coeff_range_F: [-100, 100]
density_sigma: 1.0
dataset_name: shape_position_fp7_n2
dataset_path: ${oc.env:LSSS_DATA_DIR,data}/${.dataset_name}
```

File: `configs/generation/mvp.yaml`

```yaml
n_train: 10000
n_test: 1000
num_workers: 4
verify:
  enabled: true
  check_reduced_gb: true
  check_ideal_equality: true
  resample_on_failure: true
  max_resample_attempts: 10
singular:
  binary: Singular
  timeout_seconds: 30
```

File: `configs/generation/paper.yaml`

```yaml
n_train: 1000000
n_test: 1000
num_workers: 48
verify:
  enabled: false
  check_reduced_gb: false
  check_ideal_equality: false
  resample_on_failure: false
  max_resample_attempts: 0
singular:
  binary: Singular
  timeout_seconds: 30
```

File: `configs/model/mvp.yaml`

```yaml
n_layers_encoder: 3
n_layers_decoder: 3
n_heads: 4
d_model: 256
d_ff: 1024
dropout: 0.1
max_src_len: 2000
max_tgt_len: 1000
```

File: `configs/model/paper.yaml`

```yaml
n_layers_encoder: 6
n_layers_decoder: 6
n_heads: 8
d_model: 512
d_ff: 2048
dropout: 0.1
max_src_len: 5000
max_tgt_len: 2000
```

File: `configs/embedding/discrete.yaml`

```yaml
kind: discrete
regression_loss_weight: 0.0
```

File: `configs/embedding/hybrid.yaml`

```yaml
kind: hybrid
mlp_hidden_layers: 1
mlp_hidden_width: ${model.d_model}
coeff_scale_mode: per_sample
regression_loss_weight: 0.01
regression_correct_threshold: 0.1
```

File: `configs/train/mvp.yaml`

```yaml
epochs: 3
batch_size: 16
lr: 1.0e-4
schedule: linear_decay
grad_clip: 1.0
precision: bf16-mixed
devices: 1
val_check_interval: 1.0
val_decode_subset_size: 32
```

File: `configs/train/paper.yaml`

```yaml
epochs: 8
batch_size: 16
lr: 1.0e-4
schedule: linear_decay
grad_clip: 1.0
precision: bf16-mixed
devices: 1
val_check_interval: 0.25
val_decode_subset_size: 64
```

File: `configs/wandb/default.yaml`

```yaml
project: lsss-groebner
entity: null
name: null
tags: []
mode: online
run_id: null
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_configs.py -v
```

Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add configs/ tests/test_configs.py
git commit -m "feat(config): add Hydra config groups for MVP and paper-scale"
```

---

## Task 9: Dataset Generation Entry Point

**Files:**
- Create: `src/lsss/data/generate.py`
- Create: `scripts/gen_dataset.py`
- Test: `tests/test_generate_pipeline.py`

- [ ] **Step 1: Write the failing test**

File: `tests/test_generate_pipeline.py`

```python
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from lsss.algebra.field import Fp
from lsss.data.dataset_format import read_shard
from lsss.data.generate import generate_dataset


CONFIG_DIR = str(Path(__file__).parent.parent / "configs")


@pytest.mark.integration
def test_generate_dataset_writes_verified_samples(tmp_path):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="mvp",
            overrides=[
                "generation.n_train=10",
                "generation.n_test=5",
                "generation.num_workers=1",
                f"+data.dataset_path_override={tmp_path}",
            ],
        )
    # Override the dataset path to tmp_path via cfg mutation.
    from omegaconf import OmegaConf, open_dict

    with open_dict(cfg.data):
        cfg.data.dataset_path = str(tmp_path)

    generate_dataset(cfg)

    train_shards = list((tmp_path / "train").glob("shard-*.jsonl"))
    test_shards = list((tmp_path / "test").glob("shard-*.jsonl"))
    assert len(train_shards) >= 1
    assert len(test_shards) >= 1

    field = Fp(7)
    train_samples = []
    for sh in train_shards:
        train_samples.extend(read_shard(sh, field=field))
    assert len(train_samples) == 10
    for s in train_samples:
        assert s["verified"] is True
        assert len(s["G"]) == 2  # n_vars=2 -> shape-position has 2 elements
        assert len(s["F"]) >= 2

    meta = tmp_path / "meta.json"
    assert meta.exists()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_generate_pipeline.py -v -m integration
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the generator module**

File: `src/lsss/data/generate.py`

```python
from __future__ import annotations

import json
import random
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from lsss.algebra.field import Fp, QQ, RR
from lsss.algebra.shape_position import generate_pair
from lsss.algebra.singular import SingularSession
from lsss.algebra.verify import check_generated_pair
from lsss.data.dataset_format import SCHEMA_VERSION, write_shard


def _resolve_field(name: str):
    if name.startswith("Fp"):
        return Fp(int(name[2:]))
    if name == "QQ":
        return QQ()
    if name == "RR":
        return RR()
    raise ValueError(f"Unknown field {name!r}")


def _worker(args):
    (
        worker_id,
        base_seed,
        n_samples,
        cfg_dict,
        split,
        out_path,
    ) = args
    cfg = OmegaConf.create(cfg_dict)
    field = _resolve_field(cfg.data.field)
    rng = random.Random((base_seed, worker_id, split))

    samples = []
    session_ctx = None
    session = None
    if cfg.generation.verify.enabled:
        session_ctx = SingularSession(
            binary=cfg.generation.singular.binary,
            timeout_seconds=cfg.generation.singular.timeout_seconds,
        )
        session = session_ctx.__enter__()

    try:
        sample_id = worker_id * n_samples
        produced = 0
        attempt_budget = n_samples * (
            cfg.generation.verify.max_resample_attempts + 1
        )
        attempts = 0
        while produced < n_samples and attempts < attempt_budget:
            attempts += 1
            sample_seed = rng.randint(0, 2**31 - 1)
            sample_rng = random.Random(sample_seed)
            F, G = generate_pair(
                n_vars=cfg.data.n_vars,
                d_max=cfg.data.d_max,
                d_prime=cfg.data.d_prime,
                s_max=cfg.data.s_max,
                field=field,
                density_sigma=cfg.data.density_sigma,
                rng=sample_rng,
            )
            verified = False
            if cfg.generation.verify.enabled:
                verified = check_generated_pair(
                    session, F, G,
                    n_vars=cfg.data.n_vars,
                    field=field,
                    order=cfg.data.term_order,
                )
                if not verified:
                    if cfg.generation.verify.resample_on_failure:
                        continue
            samples.append(
                {
                    "id": sample_id,
                    "F": F,
                    "G": G,
                    "field": field,
                    "n": cfg.data.n_vars,
                    "seed": sample_seed,
                    "verified": verified or not cfg.generation.verify.enabled,
                }
            )
            sample_id += 1
            produced += 1
    finally:
        if session_ctx is not None:
            session_ctx.__exit__(None, None, None)

    write_shard(out_path, samples)
    return len(samples)


def generate_dataset(cfg: DictConfig) -> None:
    dataset_path = Path(cfg.data.dataset_path)
    dataset_path.mkdir(parents=True, exist_ok=True)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    for split, n_samples in [
        ("train", cfg.generation.n_train),
        ("test", cfg.generation.n_test),
    ]:
        split_dir = dataset_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        num_workers = max(1, cfg.generation.num_workers)
        per_worker = _split_counts(n_samples, num_workers)
        jobs = []
        for w, count in enumerate(per_worker):
            if count == 0:
                continue
            jobs.append(
                (
                    w,
                    cfg.seed,
                    count,
                    cfg_dict,
                    split,
                    split_dir / f"shard-{w:05d}.jsonl",
                )
            )
        if num_workers == 1:
            for job in tqdm(jobs, desc=f"{split}"):
                _worker(job)
        else:
            with Pool(num_workers) as pool:
                for _ in tqdm(
                    pool.imap_unordered(_worker, jobs),
                    total=len(jobs),
                    desc=split,
                ):
                    pass

    meta = {
        "schema_version": SCHEMA_VERSION,
        "config": cfg_dict,
    }
    (dataset_path / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def _split_counts(total: int, workers: int) -> list[int]:
    base = total // workers
    rem = total % workers
    return [base + (1 if i < rem else 0) for i in range(workers)]
```

- [ ] **Step 4: Write the CLI entry point**

File: `scripts/gen_dataset.py`

```python
import hydra
from omegaconf import DictConfig

from lsss.data.generate import generate_dataset


@hydra.main(config_path="../configs", config_name="mvp", version_base="1.3")
def main(cfg: DictConfig) -> None:
    generate_dataset(cfg)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the test**

```bash
pytest tests/test_generate_pipeline.py -v -m integration
```

Expected: test passes; 10 training samples and 5 test samples written, all verified.

- [ ] **Step 6: Smoke-test the CLI entry point**

```bash
python scripts/gen_dataset.py generation.n_train=20 generation.n_test=5 generation.num_workers=1 data.dataset_path=/tmp/lsss_smoke
```

Expected: exits 0, creates `/tmp/lsss_smoke/train/shard-00000.jsonl` with 20 lines, `/tmp/lsss_smoke/test/shard-00000.jsonl` with 5 lines, and `/tmp/lsss_smoke/meta.json`.

- [ ] **Step 7: Commit**

```bash
git add src/lsss/data/generate.py scripts/gen_dataset.py tests/test_generate_pipeline.py
git commit -m "feat(data): dataset generation entry point with multi-worker Singular verification"
```

---

## Task 10: Tokenizer (prefix notation, discrete path)

**Files:**
- Create: `src/lsss/data/tokenizer.py`
- Test: `tests/test_tokenizer.py`

- [ ] **Step 1: Write the failing test**

File: `tests/test_tokenizer.py`

```python
from lsss.algebra.field import Fp
from lsss.algebra.polynomial import Poly
from lsss.data.tokenizer import Tokenizer, build_vocab_discrete


def test_build_vocab_fp7_n2_has_expected_sizes():
    vocab = build_vocab_discrete(field_name="Fp7", n_vars=2, max_exp=5)
    # 2 variables + 6 exponent tokens (0..5) + 7 Fp coeff tokens + 4 structural + 4 ops
    # Keep assertions loose to allow for implementation variation but bound the size.
    assert vocab.size >= 2 + 6 + 7 + 4
    assert vocab.size <= 100
    assert vocab.pad_id is not None
    assert vocab.bos_id is not None
    assert vocab.eos_id is not None
    assert vocab.sep_id is not None


def test_tokenize_detokenize_roundtrip_fp7():
    vocab = build_vocab_discrete(field_name="Fp7", n_vars=2, max_exp=5)
    tok = Tokenizer(vocab=vocab, field_name="Fp7", n_vars=2)
    field = Fp(7)
    polys = [
        Poly.from_terms([(3, (2, 0)), (5, (0, 1)), (1, (0, 0))], 2, field),
        Poly.from_terms([(1, (1, 1))], 2, field),
    ]
    ids = tok.encode_polys(polys)
    restored = tok.decode_polys(ids)
    assert restored == polys


def test_tokenize_handles_zero_polynomial():
    vocab = build_vocab_discrete(field_name="Fp7", n_vars=1, max_exp=3)
    tok = Tokenizer(vocab=vocab, field_name="Fp7", n_vars=1)
    field = Fp(7)
    polys = [Poly.zero(n_vars=1, field=field)]
    ids = tok.encode_polys(polys)
    restored = tok.decode_polys(ids)
    assert restored == polys
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_tokenizer.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

File: `src/lsss/data/tokenizer.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Iterable

from lsss.algebra.field import Fp, QQ, RR
from lsss.algebra.polynomial import Poly


@dataclass
class Vocab:
    id2tok: list[str]
    tok2id: dict[str, int]

    @property
    def size(self) -> int:
        return len(self.id2tok)

    @property
    def pad_id(self) -> int:
        return self.tok2id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.tok2id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.tok2id["<eos>"]

    @property
    def sep_id(self) -> int:
        return self.tok2id["<sep>"]


def build_vocab_discrete(field_name: str, n_vars: int, max_exp: int) -> Vocab:
    tokens: list[str] = []
    # Structural
    tokens += ["<pad>", "<bos>", "<eos>", "<sep>"]
    # Operators (prefix notation uses + only; term structure uses E/C directly).
    tokens += ["+"]
    # Variables
    for i in range(n_vars):
        tokens.append(f"E{i}")
    # Exponent tokens
    for e in range(max_exp + 1):
        tokens.append(f"X{e}")
    # Coefficient tokens for Fp
    if field_name.startswith("Fp"):
        p = int(field_name[2:])
        for v in range(p):
            tokens.append(f"C{v}")
    else:
        raise NotImplementedError(
            f"Discrete vocab not supported for field {field_name!r}"
        )
    # Dedup while preserving order
    seen = set()
    ordered = []
    for t in tokens:
        if t not in seen:
            ordered.append(t)
            seen.add(t)
    tok2id = {t: i for i, t in enumerate(ordered)}
    return Vocab(id2tok=ordered, tok2id=tok2id)


@dataclass
class Tokenizer:
    vocab: Vocab
    field_name: str
    n_vars: int

    def encode_polys(self, polys: list[Poly]) -> list[int]:
        """Encode a list of polynomials as a token id sequence."""
        out: list[int] = [self.vocab.bos_id]
        for i, p in enumerate(polys):
            if i > 0:
                out.append(self.vocab.sep_id)
            out.extend(self._encode_single(p))
        out.append(self.vocab.eos_id)
        return out

    def _encode_single(self, p: Poly) -> list[int]:
        if p.is_zero():
            return [self.vocab.tok2id["C0"]]
        ids: list[int] = []
        for idx, (coeff, exps) in enumerate(p.terms):
            if idx > 0:
                ids.append(self.vocab.tok2id["+"])
            ids.append(self._encode_coeff(coeff))
            for var_idx, exp in enumerate(exps):
                ids.append(self.vocab.tok2id[f"E{var_idx}"])
                ids.append(self.vocab.tok2id[f"X{exp}"])
        return ids

    def _encode_coeff(self, c) -> int:
        if self.field_name.startswith("Fp"):
            p = int(self.field_name[2:])
            return self.vocab.tok2id[f"C{int(c) % p}"]
        raise NotImplementedError(self.field_name)

    def decode_polys(self, ids: list[int]) -> list[Poly]:
        field = self._field()
        tokens = [self.vocab.id2tok[i] for i in ids]
        if tokens[0] == "<bos>":
            tokens = tokens[1:]
        if tokens and tokens[-1] == "<eos>":
            tokens = tokens[:-1]
        # Split into per-polynomial token runs at <sep>.
        poly_token_runs: list[list[str]] = [[]]
        for t in tokens:
            if t == "<sep>":
                poly_token_runs.append([])
            else:
                poly_token_runs[-1].append(t)
        polys = [self._decode_single(run, field) for run in poly_token_runs]
        return polys

    def _decode_single(self, tokens: list[str], field) -> Poly:
        # Terms separated by "+"; each term is coeff + n_vars * (Ei, Xj) pairs.
        if not tokens:
            return Poly.zero(n_vars=self.n_vars, field=field)
        term_groups: list[list[str]] = [[]]
        for t in tokens:
            if t == "+":
                term_groups.append([])
            else:
                term_groups[-1].append(t)
        raw_terms = []
        for group in term_groups:
            if not group:
                continue
            coeff_tok = group[0]
            assert coeff_tok.startswith("C"), f"expected coeff got {coeff_tok!r}"
            coeff_val = int(coeff_tok[1:])
            if self.field_name.startswith("Fp"):
                p = int(self.field_name[2:])
                coeff = coeff_val % p
            else:
                coeff = coeff_val
            exps = [0] * self.n_vars
            i = 1
            while i < len(group):
                var_tok = group[i]
                exp_tok = group[i + 1]
                var_idx = int(var_tok[1:])
                exps[var_idx] = int(exp_tok[1:])
                i += 2
            raw_terms.append((coeff, tuple(exps)))
        return Poly.from_terms(raw_terms, n_vars=self.n_vars, field=field)

    def _field(self):
        if self.field_name.startswith("Fp"):
            return Fp(int(self.field_name[2:]))
        if self.field_name == "QQ":
            return QQ()
        if self.field_name == "RR":
            return RR()
        raise ValueError(self.field_name)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_tokenizer.py -v
```

Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/lsss/data/tokenizer.py tests/test_tokenizer.py
git commit -m "feat(data): prefix-notation tokenizer with discrete Fp vocab"
```

---

## Task 11: PyTorch Dataset and DataModule

**Files:**
- Create: `src/lsss/data/torch_dataset.py`
- Create: `src/lsss/train/lit_datamodule.py`
- Test: `tests/test_torch_dataset.py`

- [ ] **Step 1: Write the failing test**

File: `tests/test_torch_dataset.py`

```python
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from lsss.data.generate import generate_dataset
from lsss.train.lit_datamodule import GroebnerDataModule


CONFIG_DIR = str(Path(__file__).parent.parent / "configs")


@pytest.mark.integration
def test_datamodule_yields_correct_tensor_shapes(tmp_path):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="mvp",
            overrides=[
                "generation.n_train=16",
                "generation.n_test=8",
                "generation.num_workers=1",
                "train.batch_size=4",
            ],
        )
    from omegaconf import open_dict
    with open_dict(cfg.data):
        cfg.data.dataset_path = str(tmp_path)
    generate_dataset(cfg)

    dm = GroebnerDataModule(cfg)
    dm.setup("fit")

    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert "src_tokens" in batch
    assert "tgt_tokens" in batch
    assert batch["src_tokens"].dtype == torch.long
    assert batch["src_tokens"].shape[0] == 4
    assert batch["src_tokens"].ndim == 2
    assert batch["tgt_tokens"].shape[0] == 4


@pytest.mark.integration
def test_datamodule_val_test_split(tmp_path):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="mvp",
            overrides=[
                "generation.n_train=20",
                "generation.n_test=5",
                "generation.num_workers=1",
                "train.batch_size=2",
            ],
        )
    from omegaconf import open_dict
    with open_dict(cfg.data):
        cfg.data.dataset_path = str(tmp_path)
    generate_dataset(cfg)

    dm = GroebnerDataModule(cfg)
    dm.setup("fit")
    dm.setup("test")

    train_n = sum(len(b["src_tokens"]) for b in dm.train_dataloader())
    val_n = sum(len(b["src_tokens"]) for b in dm.val_dataloader())
    test_n = sum(len(b["src_tokens"]) for b in dm.test_dataloader())
    # 1K held-out val policy is n_test=5 here; we carve val from train
    assert train_n + val_n == 20
    assert test_n == 5
    assert val_n > 0
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_torch_dataset.py -v -m integration
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the torch dataset**

File: `src/lsss/data/torch_dataset.py`

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from lsss.algebra.field import Fp, QQ, RR
from lsss.data.dataset_format import read_shard
from lsss.data.tokenizer import Tokenizer


@dataclass
class GroebnerSample:
    src_tokens: list[int]
    tgt_tokens: list[int]


def _field(name: str):
    if name.startswith("Fp"):
        return Fp(int(name[2:]))
    if name == "QQ":
        return QQ()
    if name == "RR":
        return RR()
    raise ValueError(name)


class GroebnerDataset(Dataset):
    """In-memory map-style dataset. Tokenizes all shards on construction."""

    def __init__(self, shard_paths: list[Path], tokenizer: Tokenizer, field_name: str):
        self.tokenizer = tokenizer
        field = _field(field_name)
        self._samples: list[GroebnerSample] = []
        for path in shard_paths:
            for raw in read_shard(path, field=field):
                self._samples.append(
                    GroebnerSample(
                        src_tokens=tokenizer.encode_polys(raw["F"]),
                        tgt_tokens=tokenizer.encode_polys(raw["G"]),
                    )
                )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> GroebnerSample:
        return self._samples[idx]


def collate_fn(samples: list[GroebnerSample], pad_id: int) -> dict[str, torch.Tensor]:
    src_lens = [len(s.src_tokens) for s in samples]
    tgt_lens = [len(s.tgt_tokens) for s in samples]
    src_max = max(src_lens)
    tgt_max = max(tgt_lens)
    B = len(samples)
    src = torch.full((B, src_max), pad_id, dtype=torch.long)
    tgt = torch.full((B, tgt_max), pad_id, dtype=torch.long)
    src_mask = torch.zeros((B, src_max), dtype=torch.bool)
    tgt_mask = torch.zeros((B, tgt_max), dtype=torch.bool)
    for i, s in enumerate(samples):
        src[i, : len(s.src_tokens)] = torch.tensor(s.src_tokens, dtype=torch.long)
        tgt[i, : len(s.tgt_tokens)] = torch.tensor(s.tgt_tokens, dtype=torch.long)
        src_mask[i, : len(s.src_tokens)] = True
        tgt_mask[i, : len(s.tgt_tokens)] = True
    return {
        "src_tokens": src,
        "tgt_tokens": tgt,
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
    }
```

- [ ] **Step 4: Write the DataModule**

File: `src/lsss/train/lit_datamodule.py`

```python
from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split

from lsss.data.tokenizer import Tokenizer, build_vocab_discrete
from lsss.data.torch_dataset import GroebnerDataset, collate_fn


class GroebnerDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer: Tokenizer | None = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage: str | None = None):
        dataset_path = Path(self.cfg.data.dataset_path)
        meta = json.loads((dataset_path / "meta.json").read_text())
        field_name = meta["config"]["data"]["field"]
        n_vars = meta["config"]["data"]["n_vars"]
        max_exp = self.cfg.data.d_max + 2 * self.cfg.data.d_prime
        vocab = build_vocab_discrete(field_name, n_vars, max_exp=max_exp)
        self.tokenizer = Tokenizer(vocab=vocab, field_name=field_name, n_vars=n_vars)

        if stage in (None, "fit"):
            train_shards = sorted((dataset_path / "train").glob("shard-*.jsonl"))
            full = GroebnerDataset(train_shards, self.tokenizer, field_name)
            val_n = max(1, len(full) // 10)  # 10% of train for val
            train_n = len(full) - val_n
            self.train_ds, self.val_ds = random_split(
                full,
                [train_n, val_n],
                generator=self._gen(),
            )
        if stage in (None, "test"):
            test_shards = sorted((dataset_path / "test").glob("shard-*.jsonl"))
            self.test_ds = GroebnerDataset(test_shards, self.tokenizer, field_name)

    def _gen(self):
        import torch
        g = torch.Generator()
        g.manual_seed(self.cfg.seed)
        return g

    def _collate(self):
        assert self.tokenizer is not None
        return partial(collate_fn, pad_id=self.tokenizer.vocab.pad_id)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            collate_fn=self._collate(),
            num_workers=0,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            collate_fn=self._collate(),
            num_workers=0,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            collate_fn=self._collate(),
            num_workers=0,
            pin_memory=True,
        )
```

- [ ] **Step 5: Run the test**

```bash
pytest tests/test_torch_dataset.py -v -m integration
```

Expected: both tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/lsss/data/torch_dataset.py src/lsss/train/lit_datamodule.py tests/test_torch_dataset.py
git commit -m "feat(data): torch Dataset and Lightning DataModule with val split"
```

---

## Task 12: Embeddings and Transformer Blocks

**Files:**
- Create: `src/lsss/model/embedding.py`
- Create: `src/lsss/model/transformer.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write the failing test**

File: `tests/test_model.py`

```python
import torch

from lsss.model.embedding import DiscreteEmbedding
from lsss.model.transformer import EncoderDecoder


def test_discrete_embedding_forward_shape():
    emb = DiscreteEmbedding(vocab_size=32, d_model=16, max_len=100)
    tokens = torch.randint(0, 32, (2, 10))
    out = emb(tokens)
    assert out.shape == (2, 10, 16)


def test_encoder_decoder_forward_shape():
    model = EncoderDecoder(
        vocab_size=32,
        d_model=16,
        n_heads=2,
        n_layers_encoder=2,
        n_layers_decoder=2,
        d_ff=32,
        dropout=0.0,
        max_src_len=50,
        max_tgt_len=50,
        pad_id=0,
    )
    src = torch.randint(1, 32, (2, 10))
    tgt = torch.randint(1, 32, (2, 8))
    src_mask = torch.ones(2, 10, dtype=torch.bool)
    tgt_mask = torch.ones(2, 8, dtype=torch.bool)
    logits = model(src, tgt, src_mask, tgt_mask)
    assert logits.shape == (2, 8, 32)


def test_encoder_decoder_ignores_pad_in_attention():
    model = EncoderDecoder(
        vocab_size=32,
        d_model=16,
        n_heads=2,
        n_layers_encoder=1,
        n_layers_decoder=1,
        d_ff=32,
        dropout=0.0,
        max_src_len=50,
        max_tgt_len=50,
        pad_id=0,
    )
    src = torch.tensor([[5, 6, 0, 0]])
    tgt = torch.tensor([[2, 3]])
    src_mask = torch.tensor([[True, True, False, False]])
    tgt_mask = torch.ones(1, 2, dtype=torch.bool)
    logits1 = model(src, tgt, src_mask, tgt_mask)

    # Same but with different pad token values; masked-out positions should not matter
    src_alt = torch.tensor([[5, 6, 0, 0]])
    logits2 = model(src_alt, tgt, src_mask, tgt_mask)
    assert torch.allclose(logits1, logits2)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_model.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the embedding**

File: `src/lsss/model/embedding.py`

```python
from __future__ import annotations

import torch
from torch import nn


class DiscreteEmbedding(nn.Module):
    """Token embedding + learned absolute positional embedding."""

    def __init__(self, vocab_size: int, d_model: int, max_len: int):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.max_len = max_len

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        assert L <= self.max_len, f"seq len {L} > max_len {self.max_len}"
        pos_ids = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, L)
        return self.token(tokens) + self.pos(pos_ids)
```

- [ ] **Step 4: Write the Transformer**

File: `src/lsss/model/transformer.py`

```python
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from lsss.model.embedding import DiscreteEmbedding


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_mask: torch.Tensor | None,  # (B, Lk) True = keep
        causal: bool,
    ) -> torch.Tensor:
        B, Lq, _ = q.shape
        Lk = k.shape[1]
        q = self.q_proj(q).view(B, Lq, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(k).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(v).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)

        attn_mask: torch.Tensor | None = None
        if key_mask is not None:
            attn_mask = key_mask[:, None, None, :].expand(B, self.n_heads, Lq, Lk)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor | None) -> torch.Tensor:
        h = self.ln1(x)
        x = x + self.drop(self.attn(h, h, h, key_mask=key_mask, causal=False))
        h = self.ln2(x)
        x = x + self.drop(self.ff(h))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln3 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None,
        memory_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        h = self.ln1(x)
        x = x + self.drop(self.self_attn(h, h, h, key_mask=tgt_mask, causal=True))
        h = self.ln2(x)
        x = x + self.drop(self.cross_attn(h, memory, memory, key_mask=memory_mask, causal=False))
        h = self.ln3(x)
        x = x + self.drop(self.ff(h))
        return x


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers_encoder: int,
        n_layers_decoder: int,
        d_ff: int,
        dropout: float,
        max_src_len: int,
        max_tgt_len: int,
        pad_id: int,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.src_emb = DiscreteEmbedding(vocab_size, d_model, max_src_len)
        self.tgt_emb = DiscreteEmbedding(vocab_size, d_model, max_tgt_len)
        self.encoder = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers_encoder)]
        )
        self.decoder = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers_decoder)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight-tying to source embedding.
        self.lm_head.weight = self.src_emb.token.weight

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.src_emb(src)
        for blk in self.encoder:
            x = blk(x, key_mask=src_mask)
        return x

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.tgt_emb(tgt)
        for blk in self.decoder:
            x = blk(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return self.ln_f(x)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        hidden = self.decode(tgt, memory, tgt_mask, src_mask)
        return self.lm_head(hidden)
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
pytest tests/test_model.py -v
```

Expected: all 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/lsss/model/embedding.py src/lsss/model/transformer.py tests/test_model.py
git commit -m "feat(model): from-scratch encoder-decoder Transformer with discrete embedding"
```

---

## Task 13: Lightning Module (training step, greedy decode, metrics)

**Files:**
- Create: `src/lsss/train/lit_module.py`
- Create: `src/lsss/eval/decode.py`
- Create: `src/lsss/eval/metrics.py`
- Test: `tests/test_lit_module.py`

- [ ] **Step 1: Write the failing test**

File: `tests/test_lit_module.py`

```python
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from lsss.data.generate import generate_dataset
from lsss.train.lit_datamodule import GroebnerDataModule
from lsss.train.lit_module import GroebnerTransformer


CONFIG_DIR = str(Path(__file__).parent.parent / "configs")


@pytest.mark.integration
def test_lit_module_training_step_runs(tmp_path):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="mvp",
            overrides=[
                "generation.n_train=16",
                "generation.n_test=4",
                "generation.num_workers=1",
                "train.batch_size=2",
                "model.d_model=16",
                "model.n_heads=2",
                "model.d_ff=32",
                "model.n_layers_encoder=1",
                "model.n_layers_decoder=1",
                "model.max_src_len=128",
                "model.max_tgt_len=64",
            ],
        )
    with open_dict(cfg.data):
        cfg.data.dataset_path = str(tmp_path)
    generate_dataset(cfg)

    dm = GroebnerDataModule(cfg)
    dm.setup("fit")
    model = GroebnerTransformer(cfg, vocab_size=dm.tokenizer.vocab.size,
                                 pad_id=dm.tokenizer.vocab.pad_id)

    loader = dm.train_dataloader()
    batch = next(iter(loader))
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert loss.item() > 0


@pytest.mark.integration
def test_lit_module_greedy_decode_runs(tmp_path):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="mvp",
            overrides=[
                "generation.n_train=8",
                "generation.n_test=4",
                "generation.num_workers=1",
                "train.batch_size=2",
                "model.d_model=16",
                "model.n_heads=2",
                "model.d_ff=32",
                "model.n_layers_encoder=1",
                "model.n_layers_decoder=1",
                "model.max_src_len=128",
                "model.max_tgt_len=64",
            ],
        )
    with open_dict(cfg.data):
        cfg.data.dataset_path = str(tmp_path)
    generate_dataset(cfg)

    dm = GroebnerDataModule(cfg)
    dm.setup("fit")
    model = GroebnerTransformer(cfg, vocab_size=dm.tokenizer.vocab.size,
                                 pad_id=dm.tokenizer.vocab.pad_id)
    model.eval()

    loader = dm.val_dataloader()
    batch = next(iter(loader))
    with torch.no_grad():
        preds = model.greedy_decode(batch["src_tokens"], batch["src_mask"],
                                    bos_id=dm.tokenizer.vocab.bos_id,
                                    eos_id=dm.tokenizer.vocab.eos_id,
                                    max_len=cfg.model.max_tgt_len)
    assert preds.shape[0] == batch["src_tokens"].shape[0]
    assert preds.dtype == torch.long
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_lit_module.py -v -m integration
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `eval/decode.py`**

File: `src/lsss/eval/decode.py`

```python
from __future__ import annotations

import torch

from lsss.model.transformer import EncoderDecoder


@torch.no_grad()
def greedy_decode(
    model: EncoderDecoder,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int,
) -> torch.Tensor:
    """Greedy decoding. Returns (B, L_out) with padding after <eos>."""
    device = src.device
    B = src.shape[0]
    memory = model.encode(src, src_mask)
    tgt = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_len - 1):
        tgt_mask = torch.ones_like(tgt, dtype=torch.bool)
        hidden = model.decode(tgt, memory, tgt_mask, src_mask)
        logits = model.lm_head(hidden[:, -1, :])
        next_tok = logits.argmax(dim=-1)
        next_tok = torch.where(finished, torch.full_like(next_tok, pad_id), next_tok)
        tgt = torch.cat([tgt, next_tok.unsqueeze(1)], dim=1)
        finished = finished | (next_tok == eos_id)
        if finished.all():
            break
    return tgt
```

- [ ] **Step 4: Write `eval/metrics.py` (token-level metrics only for now)**

File: `src/lsss/eval/metrics.py`

```python
from __future__ import annotations

import torch

from lsss.algebra.polynomial import Poly
from lsss.data.tokenizer import Tokenizer


def token_ids_to_polys(ids: torch.Tensor, tokenizer: Tokenizer) -> list[Poly] | None:
    """Decode a token-id sequence to a list of Polys. Returns None on parse error."""
    try:
        flat = ids.tolist()
        # Drop trailing pad
        eos = tokenizer.vocab.eos_id
        if eos in flat:
            flat = flat[: flat.index(eos) + 1]
        return tokenizer.decode_polys(flat)
    except Exception:
        return None


def polynomial_accuracy(pred: list[Poly] | None, truth: list[Poly]) -> bool:
    if pred is None:
        return False
    if len(pred) != len(truth):
        return False
    pred_sorted = sorted((tuple(p.terms) for p in pred))
    truth_sorted = sorted((tuple(p.terms) for p in truth))
    return pred_sorted == truth_sorted


def support_accuracy(pred: list[Poly] | None, truth: list[Poly]) -> bool:
    if pred is None:
        return False
    if len(pred) != len(truth):
        return False
    pred_supp = sorted(tuple(sorted(e for _, e in p.terms)) for p in pred)
    truth_supp = sorted(tuple(sorted(e for _, e in p.terms)) for p in truth)
    return pred_supp == truth_supp
```

- [ ] **Step 5: Write `train/lit_module.py`**

File: `src/lsss/train/lit_module.py`

```python
from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

from lsss.eval.decode import greedy_decode
from lsss.eval.metrics import polynomial_accuracy, support_accuracy, token_ids_to_polys
from lsss.model.transformer import EncoderDecoder


class GroebnerTransformer(pl.LightningModule):
    def __init__(self, cfg: DictConfig, vocab_size: int, pad_id: int):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg
        self.pad_id = pad_id
        self.model = EncoderDecoder(
            vocab_size=vocab_size,
            d_model=cfg.model.d_model,
            n_heads=cfg.model.n_heads,
            n_layers_encoder=cfg.model.n_layers_encoder,
            n_layers_decoder=cfg.model.n_layers_decoder,
            d_ff=cfg.model.d_ff,
            dropout=cfg.model.dropout,
            max_src_len=cfg.model.max_src_len,
            max_tgt_len=cfg.model.max_tgt_len,
            pad_id=pad_id,
        )
        self.tokenizer = None  # set by training script before fit
        self.bos_id = None
        self.eos_id = None

    def attach_tokenizer(self, tokenizer, bos_id, eos_id):
        self.tokenizer = tokenizer
        self.bos_id = bos_id
        self.eos_id = eos_id

    def _compute_loss(self, batch) -> torch.Tensor:
        src = batch["src_tokens"]
        tgt = batch["tgt_tokens"]
        src_mask = batch["src_mask"]
        tgt_mask = batch["tgt_mask"]
        # Teacher forcing: feed tgt[:-1], predict tgt[1:]
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        tgt_in_mask = tgt_mask[:, :-1]
        logits = self.model(src, tgt_in, src_mask, tgt_in_mask)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1),
            ignore_index=self.pad_id,
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        if self.tokenizer is None or batch_idx > 0:
            # Greedy decode only on the first val batch to keep it cheap
            return
        preds = self.greedy_decode(
            batch["src_tokens"],
            batch["src_mask"],
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            max_len=self.cfg.model.max_tgt_len,
        )
        poly_acc = 0
        supp_acc = 0
        total = 0
        for i in range(preds.shape[0]):
            pred_polys = token_ids_to_polys(preds[i], self.tokenizer)
            truth_polys = token_ids_to_polys(batch["tgt_tokens"][i], self.tokenizer)
            if truth_polys is None:
                continue
            total += 1
            if polynomial_accuracy(pred_polys, truth_polys):
                poly_acc += 1
            if support_accuracy(pred_polys, truth_polys):
                supp_acc += 1
        if total > 0:
            self.log("val/poly_acc", poly_acc / total, on_epoch=True, prog_bar=True)
            self.log("val/support_acc", supp_acc / total, on_epoch=True, prog_bar=True)

    def greedy_decode(self, src, src_mask, bos_id, eos_id, max_len):
        return greedy_decode(
            self.model, src, src_mask,
            bos_id=bos_id, eos_id=eos_id, pad_id=self.pad_id, max_len=max_len,
        )

    def configure_optimizers(self):
        opt = AdamW(
            self.parameters(),
            lr=self.cfg.train.lr,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )
        total_steps = self.trainer.estimated_stepping_batches
        sched = LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=total_steps)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }
```

- [ ] **Step 6: Run the test**

```bash
pytest tests/test_lit_module.py -v -m integration
```

Expected: both tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/lsss/train/lit_module.py src/lsss/eval/decode.py src/lsss/eval/metrics.py tests/test_lit_module.py
git commit -m "feat(train+eval): LightningModule with greedy decode and token-level metrics"
```

---

## Task 14: Training Entry Point

**Files:**
- Create: `scripts/train.py`
- Test: manual smoke test

- [ ] **Step 1: Write the training entry point**

File: `scripts/train.py`

```python
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from lsss.train.lit_datamodule import GroebnerDataModule
from lsss.train.lit_module import GroebnerTransformer


@hydra.main(config_path="../configs", config_name="mvp", version_base="1.3")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    dm = GroebnerDataModule(cfg)
    dm.setup("fit")

    model = GroebnerTransformer(
        cfg,
        vocab_size=dm.tokenizer.vocab.size,
        pad_id=dm.tokenizer.vocab.pad_id,
    )
    model.attach_tokenizer(
        dm.tokenizer,
        bos_id=dm.tokenizer.vocab.bos_id,
        eos_id=dm.tokenizer.vocab.eos_id,
    )

    logger_cfg = OmegaConf.to_container(cfg.wandb, resolve=True)
    wandb_mode = logger_cfg.pop("mode", "online")
    logger_cfg.pop("run_id", None)
    logger = WandbLogger(**{k: v for k, v in logger_cfg.items() if v is not None},
                          offline=(wandb_mode == "offline"))

    callbacks = [
        ModelCheckpoint(
            monitor="val/support_acc",
            mode="max",
            save_top_k=2,
            save_last=True,
            dirpath=f"{cfg.output_dir}/checkpoints",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="gpu" if cfg.train.devices else "cpu",
        devices=cfg.train.devices if cfg.train.devices else 1,
        precision=cfg.train.precision,
        logger=logger if wandb_mode != "disabled" else False,
        callbacks=callbacks,
        gradient_clip_val=cfg.train.grad_clip,
        val_check_interval=cfg.train.val_check_interval,
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the training script end-to-end**

First generate a tiny dataset:

```bash
rm -rf /tmp/lsss_train_smoke
python scripts/gen_dataset.py \
  generation.n_train=40 \
  generation.n_test=10 \
  generation.num_workers=1 \
  data.dataset_path=/tmp/lsss_train_smoke
```

Then run 1 epoch of training with W&B disabled:

```bash
python scripts/train.py \
  data.dataset_path=/tmp/lsss_train_smoke \
  train.epochs=1 \
  train.batch_size=4 \
  train.devices=1 \
  train.precision=32 \
  wandb.mode=disabled \
  model.d_model=32 \
  model.n_heads=2 \
  model.d_ff=64 \
  model.n_layers_encoder=2 \
  model.n_layers_decoder=2 \
  model.max_src_len=500 \
  model.max_tgt_len=300
```

Expected: training completes, `val/loss` is logged, checkpoint written under `outputs/*/checkpoints/last.ckpt`.

- [ ] **Step 3: Commit**

```bash
git add scripts/train.py
git commit -m "feat(train): Hydra entry point for training with W&B and checkpointing"
```

---

## Task 15: Evaluation with Ideal-Equality Metric

**Files:**
- Modify: `src/lsss/eval/metrics.py` (add ideal-equality function)
- Create: `src/lsss/eval/evaluate.py`
- Create: `scripts/evaluate.py`
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Write the failing test**

File: `tests/test_evaluate.py`

```python
from pathlib import Path

import pytest
import pytorch_lightning as pl
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from lsss.data.generate import generate_dataset
from lsss.eval.evaluate import run_evaluation
from lsss.train.lit_datamodule import GroebnerDataModule
from lsss.train.lit_module import GroebnerTransformer


CONFIG_DIR = str(Path(__file__).parent.parent / "configs")


@pytest.mark.integration
def test_run_evaluation_end_to_end(tmp_path):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="mvp",
            overrides=[
                "generation.n_train=8",
                "generation.n_test=4",
                "generation.num_workers=1",
                "train.batch_size=2",
                "model.d_model=16",
                "model.n_heads=2",
                "model.d_ff=32",
                "model.n_layers_encoder=1",
                "model.n_layers_decoder=1",
                "model.max_src_len=128",
                "model.max_tgt_len=64",
            ],
        )
    with open_dict(cfg.data):
        cfg.data.dataset_path = str(tmp_path)
    generate_dataset(cfg)

    dm = GroebnerDataModule(cfg)
    dm.setup("fit")
    dm.setup("test")

    model = GroebnerTransformer(cfg, vocab_size=dm.tokenizer.vocab.size,
                                 pad_id=dm.tokenizer.vocab.pad_id)
    model.attach_tokenizer(dm.tokenizer,
                           bos_id=dm.tokenizer.vocab.bos_id,
                           eos_id=dm.tokenizer.vocab.eos_id)

    # Skip actual training - evaluate a randomly initialized model.
    results = run_evaluation(
        model=model,
        datamodule=dm,
        check_ideal_equality=True,
        max_decode_len=cfg.model.max_tgt_len,
    )
    assert "polynomial_accuracy" in results
    assert "support_accuracy" in results
    assert "ideal_equality" in results
    assert 0.0 <= results["polynomial_accuracy"] <= 1.0
    assert 0.0 <= results["support_accuracy"] <= 1.0
    assert 0.0 <= results["ideal_equality"] <= 1.0
    assert results["n_samples"] == 4
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_evaluate.py -v -m integration
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Add `ideal_equality` to metrics**

File: `src/lsss/eval/metrics.py` (append, do not replace existing content)

Add at the bottom:

```python
from lsss.algebra.field import Field
from lsss.algebra.singular import SingularSession


def ideal_equality(
    session: SingularSession,
    F: list[Poly],
    G_pred: list[Poly] | None,
    n_vars: int,
    field: Field,
    order: str = "lex",
) -> bool:
    if G_pred is None or len(G_pred) == 0:
        return False
    try:
        return session.ideal_equal(F, G_pred, n_vars=n_vars, field=field, order=order)
    except Exception:
        return False
```

- [ ] **Step 4: Write `eval/evaluate.py`**

File: `src/lsss/eval/evaluate.py`

```python
from __future__ import annotations

from typing import Any

import torch
from tqdm import tqdm

from lsss.algebra.singular import SingularSession
from lsss.data.dataset_format import read_shard
from lsss.eval.metrics import (
    ideal_equality,
    polynomial_accuracy,
    support_accuracy,
    token_ids_to_polys,
)
from lsss.train.lit_datamodule import GroebnerDataModule
from lsss.train.lit_module import GroebnerTransformer


def run_evaluation(
    model: GroebnerTransformer,
    datamodule: GroebnerDataModule,
    check_ideal_equality: bool,
    max_decode_len: int,
) -> dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device

    tokenizer = datamodule.tokenizer
    assert tokenizer is not None

    # Reread test shards for ground-truth F,G Polys (for ideal-equality check).
    from pathlib import Path

    test_dir = Path(datamodule.cfg.data.dataset_path) / "test"
    raw_samples: list[dict[str, Any]] = []
    for sh in sorted(test_dir.glob("shard-*.jsonl")):
        raw_samples.extend(read_shard(sh, field=datamodule.tokenizer._field()))

    poly_correct = 0
    supp_correct = 0
    ideal_correct = 0
    n = 0

    session_ctx = None
    session = None
    if check_ideal_equality:
        session_ctx = SingularSession()
        session = session_ctx.__enter__()

    try:
        loader = datamodule.test_dataloader()
        sample_offset = 0
        for batch in tqdm(loader, desc="eval"):
            src = batch["src_tokens"].to(device)
            src_mask = batch["src_mask"].to(device)
            with torch.no_grad():
                preds = model.greedy_decode(
                    src, src_mask,
                    bos_id=tokenizer.vocab.bos_id,
                    eos_id=tokenizer.vocab.eos_id,
                    max_len=max_decode_len,
                )
            for i in range(preds.shape[0]):
                raw = raw_samples[sample_offset + i]
                pred_polys = token_ids_to_polys(preds[i], tokenizer)
                truth_polys = raw["G"]
                if polynomial_accuracy(pred_polys, truth_polys):
                    poly_correct += 1
                if support_accuracy(pred_polys, truth_polys):
                    supp_correct += 1
                if check_ideal_equality and session is not None:
                    ok = ideal_equality(
                        session,
                        raw["F"],
                        pred_polys,
                        n_vars=raw["n"],
                        field=raw["field"],
                    )
                    if ok:
                        ideal_correct += 1
                n += 1
            sample_offset += preds.shape[0]
    finally:
        if session_ctx is not None:
            session_ctx.__exit__(None, None, None)

    return {
        "polynomial_accuracy": poly_correct / max(n, 1),
        "support_accuracy": supp_correct / max(n, 1),
        "ideal_equality": ideal_correct / max(n, 1) if check_ideal_equality else None,
        "n_samples": n,
    }
```

- [ ] **Step 5: Write the CLI entry point**

File: `scripts/evaluate.py`

```python
import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from lsss.eval.evaluate import run_evaluation
from lsss.train.lit_datamodule import GroebnerDataModule
from lsss.train.lit_module import GroebnerTransformer


@hydra.main(config_path="../configs", config_name="eval", version_base="1.3")
def main(cfg: DictConfig) -> None:
    dm = GroebnerDataModule(cfg)
    dm.setup("fit")
    dm.setup("test")

    model = GroebnerTransformer.load_from_checkpoint(
        cfg.eval.ckpt_path,
        cfg=cfg,
        vocab_size=dm.tokenizer.vocab.size,
        pad_id=dm.tokenizer.vocab.pad_id,
    )
    model.attach_tokenizer(
        dm.tokenizer,
        bos_id=dm.tokenizer.vocab.bos_id,
        eos_id=dm.tokenizer.vocab.eos_id,
    )

    results = run_evaluation(
        model=model,
        datamodule=dm,
        check_ideal_equality=cfg.eval.check_ideal_equality,
        max_decode_len=cfg.eval.max_decode_len,
    )
    print(json.dumps(results, indent=2))
    Path(cfg.eval.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.eval.output_json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the test**

```bash
pytest tests/test_evaluate.py -v -m integration
```

Expected: test passes, all three metrics present.

- [ ] **Step 7: Commit**

```bash
git add src/lsss/eval/metrics.py src/lsss/eval/evaluate.py scripts/evaluate.py tests/test_evaluate.py
git commit -m "feat(eval): full evaluation pipeline with Singular ideal-equality metric"
```

---

## Task 16: MVP End-to-End Run and Documentation

**Files:**
- Modify: `README.md` (replace content)

- [ ] **Step 1: Generate MVP dataset (10K/1K, verified)**

```bash
rm -rf data/shape_position_fp7_n2
python scripts/gen_dataset.py --config-name mvp
```

Expected: `data/shape_position_fp7_n2/train/` contains 10K verified samples across shards; `data/shape_position_fp7_n2/test/` contains 1K samples; `data/shape_position_fp7_n2/meta.json` exists. Runtime: a few minutes on 4 workers.

- [ ] **Step 2: Train MVP model**

```bash
python scripts/train.py --config-name mvp wandb.mode=offline
```

Expected: 3 epochs complete, `val/loss` decreases monotonically, `val/support_acc` is logged and non-zero by the end of training. W&B run written offline under `wandb/`. Best checkpoint under `outputs/*/checkpoints/`.

- [ ] **Step 3: Evaluate the checkpoint**

```bash
CKPT=$(ls -t outputs/*/checkpoints/last.ckpt | head -1)
python scripts/evaluate.py --config-name eval \
  eval.ckpt_path=$CKPT \
  wandb.mode=offline \
  model=mvp \
  eval.max_decode_len=1000
```

Expected: prints a JSON with `polynomial_accuracy`, `support_accuracy`, `ideal_equality` (all ∈ [0, 1]) and `n_samples=1000`. `results.json` written under `outputs/*/`.

- [ ] **Step 4: Update README with MVP usage**

File: `/home/jan/projects/CIIRC/colabs/tpajdla/LearningSparseSystemSolver/README.md`

```markdown
# LearningSparseSystemSolver

Reimplementation of *Learning to Compute Gröbner Bases* (Kera et al., arXiv:2311.12904). MVP targets shape-position ideals over 𝔽₇ with n=2 variables.

## Setup

Requires Singular on `PATH` and Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quickstart (MVP)

```bash
# 1. Generate dataset (10K train / 1K test, all verified via Singular)
python scripts/gen_dataset.py --config-name mvp

# 2. Train a small Transformer
python scripts/train.py --config-name mvp wandb.mode=offline

# 3. Evaluate with polynomial-acc, support-acc, and ideal-equality metrics
python scripts/evaluate.py --config-name eval \
  eval.ckpt_path=outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/last.ckpt \
  wandb.mode=offline
```

## Structure

- `src/lsss/algebra/`: polynomial ring, Singular bridge, shape-position generator, verification.
- `src/lsss/data/`: JSONL dataset format, prefix-notation tokenizer, PyTorch dataset.
- `src/lsss/model/`: from-scratch encoder-decoder Transformer.
- `src/lsss/train/`: Lightning module and data module.
- `src/lsss/eval/`: greedy decode, metrics, evaluation entry point.
- `configs/`: Hydra config groups (MVP and paper-scale presets).
- `docs/superpowers/`: design specs and implementation plans.

## Testing

```bash
pytest                          # unit + config tests (fast)
pytest -m integration           # requires Singular
```

## References

- Design spec: `docs/superpowers/specs/2026-04-18-learning-to-compute-grobner-bases-design.md`
- Implementation plan: `docs/superpowers/plans/2026-04-18-grobner-mvp-implementation.md`
```

- [ ] **Step 5: Commit the end-to-end run artifacts**

```bash
git add README.md
git commit -m "docs: README with MVP quickstart and repo structure"
```

The generated dataset and W&B offline runs are in `.gitignore` and are not committed.

---

## Definition of MVP Done

When all 16 tasks above are green:

1. `pytest` passes all unit tests and config tests.
2. `pytest -m integration` passes all Singular-dependent tests.
3. `scripts/gen_dataset.py --config-name mvp` produces 10K verified `(F, G)` pairs on disk.
4. `scripts/train.py --config-name mvp wandb.mode=offline` trains without errors, loss decreases, best checkpoint saved.
5. `scripts/evaluate.py --config-name eval eval.ckpt_path=...` prints all three metrics with non-degenerate values.

Out of scope for this plan and deferred to M7/M8: paper-scale reproduction (full 1M/1K, n ∈ {2,3,4,5}, all four fields, hybrid embedding for ℚ/ℝ), FGLM change-of-order, Cauchy module generator, sparse-systems extension.
