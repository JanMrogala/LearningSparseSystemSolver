from pathlib import Path

import pytest
import pytorch_lightning as pl
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from lsss.data.generate import generate_dataset
from lsss.eval.evaluate import run_evaluation
from lsss.train.lit_datamodule import GroebnerDataModule
from lsss.train.lit_module import GroebnerTransformer


CONFIG_DIR = str(Path(__file__).parent.parent / "configs")


@pytest.mark.integration
def test_run_evaluation_end_to_end(tmp_path):
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
    dm.setup("test")

    model = GroebnerTransformer(cfg, vocab_size=dm.tokenizer.vocab.size,
                                 pad_id=dm.tokenizer.vocab.pad_id)
    model.attach_tokenizer(dm.tokenizer,
                           bos_id=dm.tokenizer.vocab.bos_id,
                           eos_id=dm.tokenizer.vocab.eos_id)

    # Skip actual training - evaluate a randomly initialized model.
    results = run_evaluation(
        model=model,
        datamodule=dm,
        check_ideal_equality=True,
        max_decode_len=cfg.model.max_tgt_len,
    )
    assert "polynomial_accuracy" in results
    assert "support_accuracy" in results
    assert "ideal_equality" in results
    assert 0.0 <= results["polynomial_accuracy"] <= 1.0
    assert 0.0 <= results["support_accuracy"] <= 1.0
    assert 0.0 <= results["ideal_equality"] <= 1.0
    assert results["n_samples"] == 4
