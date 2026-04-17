from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split

from lsss.data.tokenizer import Tokenizer, build_vocab_discrete
from lsss.data.torch_dataset import GroebnerDataset, collate_fn


class GroebnerDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer: Tokenizer | None = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage: str | None = None):
        dataset_path = Path(self.cfg.data.dataset_path)
        meta = json.loads((dataset_path / "meta.json").read_text())
        field_name = meta["config"]["data"]["field"]
        n_vars = meta["config"]["data"]["n_vars"]
        max_exp = self.cfg.data.d_max + 2 * self.cfg.data.d_prime
        vocab = build_vocab_discrete(field_name, n_vars, max_exp=max_exp)
        self.tokenizer = Tokenizer(vocab=vocab, field_name=field_name, n_vars=n_vars)

        if stage in (None, "fit"):
            train_shards = sorted((dataset_path / "train").glob("shard-*.jsonl"))
            full = GroebnerDataset(train_shards, self.tokenizer, field_name)
            val_n = max(1, len(full) // 10)  # 10% of train for val
            train_n = len(full) - val_n
            self.train_ds, self.val_ds = random_split(
                full,
                [train_n, val_n],
                generator=self._gen(),
            )
        if stage in (None, "test"):
            test_shards = sorted((dataset_path / "test").glob("shard-*.jsonl"))
            self.test_ds = GroebnerDataset(test_shards, self.tokenizer, field_name)

    def _gen(self):
        import torch
        g = torch.Generator()
        g.manual_seed(self.cfg.seed)
        return g

    def _collate(self):
        assert self.tokenizer is not None
        max_src = getattr(self.cfg.model, "max_src_len", None)
        max_tgt = getattr(self.cfg.model, "max_tgt_len", None)
        return partial(collate_fn, pad_id=self.tokenizer.vocab.pad_id,
                       max_src_len=max_src, max_tgt_len=max_tgt)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            collate_fn=self._collate(),
            num_workers=0,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            collate_fn=self._collate(),
            num_workers=0,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            collate_fn=self._collate(),
            num_workers=0,
            pin_memory=True,
        )
