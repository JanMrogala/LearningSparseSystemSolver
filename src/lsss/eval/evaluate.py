from __future__ import annotations

from pathlib import Path
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

    # Reread test shards for ground-truth F, G Polys (needed for ideal-equality check).
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
