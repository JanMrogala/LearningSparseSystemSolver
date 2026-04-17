from pathlib import Path

from hydra import compose, initialize_config_dir


CONFIG_DIR = str(Path(__file__).parent.parent / "configs")


def test_mvp_config_composes():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name="mvp")
    assert cfg.seed == 0
    assert cfg.data.field == "Fp7"
    assert cfg.data.n_vars == 2
    assert cfg.model.n_layers_encoder == 3
    assert cfg.embedding.kind == "discrete"
    assert cfg.train.epochs >= 1
    assert cfg.generation.verify.enabled is True


def test_paper_config_composes():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name="paper")
    assert cfg.model.n_layers_encoder == 6
    assert cfg.model.d_model == 512
    assert cfg.train.epochs == 8
    assert cfg.generation.verify.enabled is False


def test_embedding_override():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name="mvp", overrides=["embedding=hybrid"])
    assert cfg.embedding.kind == "hybrid"
    assert cfg.embedding.regression_loss_weight == 0.01
