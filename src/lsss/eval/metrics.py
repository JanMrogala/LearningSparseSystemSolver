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


from lsss.algebra.field import Field  # noqa: E402
from lsss.algebra.singular import SingularSession  # noqa: E402


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
