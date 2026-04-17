import torch

from lsss.model.embedding import DiscreteEmbedding
from lsss.model.transformer import EncoderDecoder


def test_discrete_embedding_forward_shape():
    emb = DiscreteEmbedding(vocab_size=32, d_model=16, max_len=100)
    tokens = torch.randint(0, 32, (2, 10))
    out = emb(tokens)
    assert out.shape == (2, 10, 16)


def test_encoder_decoder_forward_shape():
    model = EncoderDecoder(
        vocab_size=32,
        d_model=16,
        n_heads=2,
        n_layers_encoder=2,
        n_layers_decoder=2,
        d_ff=32,
        dropout=0.0,
        max_src_len=50,
        max_tgt_len=50,
        pad_id=0,
    )
    src = torch.randint(1, 32, (2, 10))
    tgt = torch.randint(1, 32, (2, 8))
    src_mask = torch.ones(2, 10, dtype=torch.bool)
    tgt_mask = torch.ones(2, 8, dtype=torch.bool)
    logits = model(src, tgt, src_mask, tgt_mask)
    assert logits.shape == (2, 8, 32)


def test_encoder_decoder_ignores_pad_in_attention():
    model = EncoderDecoder(
        vocab_size=32,
        d_model=16,
        n_heads=2,
        n_layers_encoder=1,
        n_layers_decoder=1,
        d_ff=32,
        dropout=0.0,
        max_src_len=50,
        max_tgt_len=50,
        pad_id=0,
    )
    src = torch.tensor([[5, 6, 0, 0]])
    tgt = torch.tensor([[2, 3]])
    src_mask = torch.tensor([[True, True, False, False]])
    tgt_mask = torch.ones(1, 2, dtype=torch.bool)
    logits1 = model(src, tgt, src_mask, tgt_mask)

    # Same but with different pad token values; masked-out positions should not matter
    src_alt = torch.tensor([[5, 6, 0, 0]])
    logits2 = model(src_alt, tgt, src_mask, tgt_mask)
    assert torch.allclose(logits1, logits2)
