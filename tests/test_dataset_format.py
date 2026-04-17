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
