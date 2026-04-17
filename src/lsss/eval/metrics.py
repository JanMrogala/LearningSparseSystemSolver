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
