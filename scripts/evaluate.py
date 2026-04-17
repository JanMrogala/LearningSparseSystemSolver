import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from lsss.eval.evaluate import run_evaluation
from lsss.train.lit_datamodule import GroebnerDataModule
from lsss.train.lit_module import GroebnerTransformer


@hydra.main(config_path="../configs", config_name="eval", version_base="1.3")
def main(cfg: DictConfig) -> None:
    dm = GroebnerDataModule(cfg)
    dm.setup("fit")
    dm.setup("test")

    model = GroebnerTransformer.load_from_checkpoint(
        cfg.eval.ckpt_path,
        cfg=cfg,
        vocab_size=dm.tokenizer.vocab.size,
        pad_id=dm.tokenizer.vocab.pad_id,
    )
    model.attach_tokenizer(
        dm.tokenizer,
        bos_id=dm.tokenizer.vocab.bos_id,
        eos_id=dm.tokenizer.vocab.eos_id,
    )

    results = run_evaluation(
        model=model,
        datamodule=dm,
        check_ideal_equality=cfg.eval.check_ideal_equality,
        max_decode_len=cfg.eval.max_decode_len,
    )
    print(json.dumps(results, indent=2))
    Path(cfg.eval.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.eval.output_json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
