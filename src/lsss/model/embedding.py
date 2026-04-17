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
