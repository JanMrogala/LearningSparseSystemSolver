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
