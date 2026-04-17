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
