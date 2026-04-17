from lsss.algebra.field import Fp
from lsss.algebra.polynomial import Poly
from lsss.data.tokenizer import Tokenizer, build_vocab_discrete


def test_build_vocab_fp7_n2_has_expected_sizes():
    vocab = build_vocab_discrete(field_name="Fp7", n_vars=2, max_exp=5)
    # 2 variables + 6 exponent tokens (0..5) + 7 Fp coeff tokens + 4 structural + 4 ops
    # Keep assertions loose to allow for implementation variation but bound the size.
    assert vocab.size >= 2 + 6 + 7 + 4
    assert vocab.size <= 100
    assert vocab.pad_id is not None
    assert vocab.bos_id is not None
    assert vocab.eos_id is not None
    assert vocab.sep_id is not None


def test_tokenize_detokenize_roundtrip_fp7():
    vocab = build_vocab_discrete(field_name="Fp7", n_vars=2, max_exp=5)
    tok = Tokenizer(vocab=vocab, field_name="Fp7", n_vars=2)
    field = Fp(7)
    polys = [
        Poly.from_terms([(3, (2, 0)), (5, (0, 1)), (1, (0, 0))], 2, field),
        Poly.from_terms([(1, (1, 1))], 2, field),
    ]
    ids = tok.encode_polys(polys)
    restored = tok.decode_polys(ids)
    assert restored == polys


def test_tokenize_handles_zero_polynomial():
    vocab = build_vocab_discrete(field_name="Fp7", n_vars=1, max_exp=3)
    tok = Tokenizer(vocab=vocab, field_name="Fp7", n_vars=1)
    field = Fp(7)
    polys = [Poly.zero(n_vars=1, field=field)]
    ids = tok.encode_polys(polys)
    restored = tok.decode_polys(ids)
    assert restored == polys
