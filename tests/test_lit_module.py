from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from lsss.data.generate import generate_dataset
from lsss.train.lit_datamodule import GroebnerDataModule
from lsss.train.lit_module import GroebnerTransformer


CONFIG_DIR = str(Path(__file__).parent.parent / "configs")


@pytest.mark.integration
def test_lit_module_training_step_runs(tmp_path):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="mvp",
            overrides=[
                "generation.n_train=16",
                "generation.n_test=4",
                "generation.num_workers=1",
                "train.batch_size=2",
                "model.d_model=16",
                "model.n_heads=2",
                "model.d_ff=32",
                "model.n_layers_encoder=1",
                "model.n_layers_decoder=1",
                "model.max_src_len=512",
                "model.max_tgt_len=256",
            ],
        )
    with open_dict(cfg.data):
        cfg.data.dataset_path = str(tmp_path)
    generate_dataset(cfg)

    dm = GroebnerDataModule(cfg)
    dm.setup("fit")
    model = GroebnerTransformer(cfg, vocab_size=dm.tokenizer.vocab.size,
                                 pad_id=dm.tokenizer.vocab.pad_id)

    loader = dm.train_dataloader()
    batch = next(iter(loader))
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert loss.item() > 0


@pytest.mark.integration
def test_lit_module_greedy_decode_runs(tmp_path):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="mvp",
            overrides=[
                "generation.n_train=8",
                "generation.n_test=4",
                "generation.num_workers=1",
                "train.batch_size=2",
                "model.d_model=16",
                "model.n_heads=2",
                "model.d_ff=32",
                "model.n_layers_encoder=1",
                "model.n_layers_decoder=1",
                "model.max_src_len=512",
                "model.max_tgt_len=256",
            ],
        )
    with open_dict(cfg.data):
        cfg.data.dataset_path = str(tmp_path)
    generate_dataset(cfg)

    dm = GroebnerDataModule(cfg)
    dm.setup("fit")
    model = GroebnerTransformer(cfg, vocab_size=dm.tokenizer.vocab.size,
                                 pad_id=dm.tokenizer.vocab.pad_id)
    model.eval()

    loader = dm.val_dataloader()
    batch = next(iter(loader))
    with torch.no_grad():
        preds = model.greedy_decode(batch["src_tokens"], batch["src_mask"],
                                    bos_id=dm.tokenizer.vocab.bos_id,
                                    eos_id=dm.tokenizer.vocab.eos_id,
                                    max_len=cfg.model.max_tgt_len)
    assert preds.shape[0] == batch["src_tokens"].shape[0]
    assert preds.dtype == torch.long
