from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from lsss.algebra.field import Fp
from lsss.data.dataset_format import read_shard
from lsss.data.generate import generate_dataset


CONFIG_DIR = str(Path(__file__).parent.parent / "configs")


@pytest.mark.integration
def test_generate_dataset_writes_verified_samples(tmp_path):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="mvp",
            overrides=[
                "generation.n_train=10",
                "generation.n_test=5",
                "generation.num_workers=1",
                f"+data.dataset_path_override={tmp_path}",
            ],
        )
    # Override the dataset path to tmp_path via cfg mutation.
    from omegaconf import OmegaConf, open_dict

    with open_dict(cfg.data):
        cfg.data.dataset_path = str(tmp_path)

    generate_dataset(cfg)

    train_shards = list((tmp_path / "train").glob("shard-*.jsonl"))
    test_shards = list((tmp_path / "test").glob("shard-*.jsonl"))
    assert len(train_shards) >= 1
    assert len(test_shards) >= 1

    field = Fp(7)
    train_samples = []
    for sh in train_shards:
        train_samples.extend(read_shard(sh, field=field))
    assert len(train_samples) == 10
    for s in train_samples:
        assert s["verified"] is True
        assert len(s["G"]) == 2  # n_vars=2 -> shape-position has 2 elements
        assert len(s["F"]) >= 2

    meta = tmp_path / "meta.json"
    assert meta.exists()
