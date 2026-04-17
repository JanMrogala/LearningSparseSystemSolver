import torch
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from lsss.train.lit_datamodule import GroebnerDataModule
from lsss.train.lit_module import GroebnerTransformer


@hydra.main(config_path="../configs", config_name="mvp", version_base="1.3")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    dm = GroebnerDataModule(cfg)
    dm.setup("fit")

    model = GroebnerTransformer(
        cfg,
        vocab_size=dm.tokenizer.vocab.size,
        pad_id=dm.tokenizer.vocab.pad_id,
    )
    model.attach_tokenizer(
        dm.tokenizer,
        bos_id=dm.tokenizer.vocab.bos_id,
        eos_id=dm.tokenizer.vocab.eos_id,
    )

    logger_cfg = OmegaConf.to_container(cfg.wandb, resolve=True)
    wandb_mode = logger_cfg.pop("mode", "online")
    logger_cfg.pop("run_id", None)

    if wandb_mode == "disabled":
        logger = False
    else:
        logger = WandbLogger(
            **{k: v for k, v in logger_cfg.items() if v is not None},
            offline=(wandb_mode == "offline"),
        )

    callbacks = [
        ModelCheckpoint(
            monitor="val/support_acc",
            mode="max",
            save_top_k=2,
            save_last=True,
            dirpath=f"{cfg.output_dir}/checkpoints",
        ),
    ]
    if logger is not False:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    accelerator = "gpu" if (cfg.train.devices and torch.cuda.is_available()) else "cpu"
    devices = cfg.train.devices if accelerator == "gpu" else 1

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator=accelerator,
        devices=devices,
        precision=cfg.train.precision,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=cfg.train.grad_clip,
        val_check_interval=cfg.train.val_check_interval,
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
