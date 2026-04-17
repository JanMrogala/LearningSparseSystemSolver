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


def collate_fn(
    samples: list[GroebnerSample],
    pad_id: int,
    max_src_len: int | None = None,
    max_tgt_len: int | None = None,
) -> dict[str, torch.Tensor]:
    src_lens = [len(s.src_tokens) for s in samples]
    tgt_lens = [len(s.tgt_tokens) for s in samples]
    src_max = max(src_lens)
    tgt_max = max(tgt_lens)
    if max_src_len is not None:
        src_max = min(src_max, max_src_len)
    if max_tgt_len is not None:
        tgt_max = min(tgt_max, max_tgt_len)
    B = len(samples)
    src = torch.full((B, src_max), pad_id, dtype=torch.long)
    tgt = torch.full((B, tgt_max), pad_id, dtype=torch.long)
    src_mask = torch.zeros((B, src_max), dtype=torch.bool)
    tgt_mask = torch.zeros((B, tgt_max), dtype=torch.bool)
    for i, s in enumerate(samples):
        sl = min(len(s.src_tokens), src_max)
        tl = min(len(s.tgt_tokens), tgt_max)
        src[i, :sl] = torch.tensor(s.src_tokens[:sl], dtype=torch.long)
        tgt[i, :tl] = torch.tensor(s.tgt_tokens[:tl], dtype=torch.long)
        src_mask[i, :sl] = True
        tgt_mask[i, :tl] = True
    return {
        "src_tokens": src,
        "tgt_tokens": tgt,
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
    }
