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
