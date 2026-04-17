from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

from lsss.eval.decode import greedy_decode
from lsss.eval.metrics import polynomial_accuracy, support_accuracy, token_ids_to_polys
from lsss.model.transformer import EncoderDecoder


class GroebnerTransformer(pl.LightningModule):
    def __init__(self, cfg: DictConfig, vocab_size: int, pad_id: int):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg
        self.pad_id = pad_id
        self.model = EncoderDecoder(
            vocab_size=vocab_size,
            d_model=cfg.model.d_model,
            n_heads=cfg.model.n_heads,
            n_layers_encoder=cfg.model.n_layers_encoder,
            n_layers_decoder=cfg.model.n_layers_decoder,
            d_ff=cfg.model.d_ff,
            dropout=cfg.model.dropout,
            max_src_len=cfg.model.max_src_len,
            max_tgt_len=cfg.model.max_tgt_len,
            pad_id=pad_id,
        )
        self.tokenizer = None  # set by training script before fit
        self.bos_id = None
        self.eos_id = None

    def attach_tokenizer(self, tokenizer, bos_id, eos_id):
        self.tokenizer = tokenizer
        self.bos_id = bos_id
        self.eos_id = eos_id

    def _compute_loss(self, batch) -> torch.Tensor:
        src = batch["src_tokens"]
        tgt = batch["tgt_tokens"]
        src_mask = batch["src_mask"]
        tgt_mask = batch["tgt_mask"]
        # Teacher forcing: feed tgt[:-1], predict tgt[1:]
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        tgt_in_mask = tgt_mask[:, :-1]
        logits = self.model(src, tgt_in, src_mask, tgt_in_mask)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1),
            ignore_index=self.pad_id,
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        try:
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        except Exception:
            pass  # running outside a Trainer (e.g., direct test)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        try:
            self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        except Exception:
            pass
        if self.tokenizer is None or batch_idx > 0:
            # Greedy decode only on the first val batch to keep it cheap
            return
        preds = self.greedy_decode(
            batch["src_tokens"],
            batch["src_mask"],
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            max_len=self.cfg.model.max_tgt_len,
        )
        poly_acc = 0
        supp_acc = 0
        total = 0
        for i in range(preds.shape[0]):
            pred_polys = token_ids_to_polys(preds[i], self.tokenizer)
            truth_polys = token_ids_to_polys(batch["tgt_tokens"][i], self.tokenizer)
            if truth_polys is None:
                continue
            total += 1
            if polynomial_accuracy(pred_polys, truth_polys):
                poly_acc += 1
            if support_accuracy(pred_polys, truth_polys):
                supp_acc += 1
        if total > 0:
            try:
                self.log("val/poly_acc", poly_acc / total, on_epoch=True, prog_bar=True)
                self.log("val/support_acc", supp_acc / total, on_epoch=True, prog_bar=True)
            except Exception:
                pass

    def greedy_decode(self, src, src_mask, bos_id, eos_id, max_len):
        return greedy_decode(
            self.model, src, src_mask,
            bos_id=bos_id, eos_id=eos_id, pad_id=self.pad_id, max_len=max_len,
        )

    def configure_optimizers(self):
        opt = AdamW(
            self.parameters(),
            lr=self.cfg.train.lr,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )
        total_steps = self.trainer.estimated_stepping_batches
        sched = LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=total_steps)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }
