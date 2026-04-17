from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from lsss.data.generate import generate_dataset
from lsss.train.lit_datamodule import GroebnerDataModule


CONFIG_DIR = str(Path(__file__).parent.parent / "configs")


@pytest.mark.integration
def test_datamodule_yields_correct_tensor_shapes(tmp_path):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="mvp",
            overrides=[
                "generation.n_train=16",
                "generation.n_test=8",
                "generation.num_workers=1",
                "train.batch_size=4",
            ],
        )
    from omegaconf import open_dict
    with open_dict(cfg.data):
        cfg.data.dataset_path = str(tmp_path)
    generate_dataset(cfg)

    dm = GroebnerDataModule(cfg)
    dm.setup("fit")

    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert "src_tokens" in batch
    assert "tgt_tokens" in batch
    assert batch["src_tokens"].dtype == torch.long
    assert batch["src_tokens"].shape[0] == 4
    assert batch["src_tokens"].ndim == 2
    assert batch["tgt_tokens"].shape[0] == 4


@pytest.mark.integration
def test_datamodule_val_test_split(tmp_path):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="mvp",
            overrides=[
                "generation.n_train=20",
                "generation.n_test=5",
                "generation.num_workers=1",
                "train.batch_size=2",
            ],
        )
    from omegaconf import open_dict
    with open_dict(cfg.data):
        cfg.data.dataset_path = str(tmp_path)
    generate_dataset(cfg)

    dm = GroebnerDataModule(cfg)
    dm.setup("fit")
    dm.setup("test")

    train_n = sum(len(b["src_tokens"]) for b in dm.train_dataloader())
    val_n = sum(len(b["src_tokens"]) for b in dm.val_dataloader())
    test_n = sum(len(b["src_tokens"]) for b in dm.test_dataloader())
    # 1K held-out val policy is n_test=5 here; we carve val from train
    assert train_n + val_n == 20
    assert test_n == 5
    assert val_n > 0
