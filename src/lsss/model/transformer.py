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

        # Build attention mask. scaled_dot_product_attention does not allow
        # combining is_causal=True with an explicit attn_mask, so when causal
        # masking is needed we construct the mask manually.
        attn_mask: torch.Tensor | None = None
        if causal:
            # Upper-triangular causal mask: True = keep (additive 0), False = block (-inf).
            causal_mask = torch.ones(Lq, Lk, dtype=torch.bool, device=q.device).tril()
            attn_mask = causal_mask[None, None, :, :].expand(B, self.n_heads, Lq, Lk)
            if key_mask is not None:
                # key_mask: (B, Lk) True = keep; broadcast over heads and queries.
                key_mask_4d = key_mask[:, None, None, :].expand(B, self.n_heads, Lq, Lk)
                attn_mask = attn_mask & key_mask_4d
            # Convert bool mask to float additive mask for sdpa.
            attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill(
                ~attn_mask, float("-inf")
            )
        elif key_mask is not None:
            key_mask_4d = key_mask[:, None, None, :].expand(B, self.n_heads, Lq, Lk)
            attn_mask = torch.zeros(B, self.n_heads, Lq, Lk, dtype=q.dtype, device=q.device)
            attn_mask = attn_mask.masked_fill(~key_mask_4d, float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
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
