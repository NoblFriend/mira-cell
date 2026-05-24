from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torchvision import models

from form_reader_and_evaluator.constants import EMPTY_IDX, JUNK_IDX, NUM_CLASSES
from form_reader_and_evaluator.data.hint import no_hint_mask, sample_hint_mask


def _build_backbone(name: str) -> tuple[nn.Module, int]:
    if name == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        return nn.Sequential(*list(net.children())[:-1]), 512
    raise ValueError(f"Unknown backbone: {name}")


class LetterClassifier(L.LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters({"cfg": dict(cfg) if hasattr(cfg, "keys") else cfg})

        self.backbone, feat_dim = _build_backbone(cfg.backbone)

        self.hint_encoder = nn.Sequential(
            nn.Linear(NUM_CLASSES, cfg.context_dim),
            nn.ReLU(),
            nn.Linear(cfg.context_dim, cfg.context_dim),
            nn.ReLU(),
        )

        self.image_proj = nn.Linear(feat_dim, cfg.fusion_proj_dim)
        self.hint_proj = nn.Linear(cfg.context_dim, cfg.fusion_proj_dim)

        head_in = feat_dim + cfg.context_dim + cfg.fusion_proj_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, cfg.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.head_dropout_1),
            nn.Linear(cfg.head_hidden_dim, cfg.head_hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(cfg.head_dropout_2),
            nn.Linear(cfg.head_hidden_dim_2, NUM_CLASSES),
        )

        self.hint_residual = nn.Sequential(
            nn.Linear(NUM_CLASSES, cfg.hint_residual_dim),
            nn.ReLU(),
            nn.Linear(cfg.hint_residual_dim, NUM_CLASSES),
        )

        self.alpha = nn.Parameter(torch.tensor(cfg.hint_residual_alpha_init, dtype=torch.float32))

        self.criterion = nn.CrossEntropyLoss()

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).flatten(1)

    def forward(
        self,
        x: torch.Tensor,
        hint_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        if hint_mask is None:
            hint_mask = no_hint_mask(batch_size, x.device)
        hint_mask = hint_mask.to(device=x.device, dtype=torch.float32)

        f_img = self.encode_image(x)
        f_hint = self.hint_encoder(hint_mask)

        p_img = self.image_proj(f_img)
        p_hint = self.hint_proj(f_hint)
        interaction = p_img * p_hint

        joint = torch.cat([f_img, f_hint, interaction], dim=1)
        joint_logits = self.head(joint)
        residual = self.hint_residual(hint_mask)
        return joint_logits + self.alpha * residual

    def _build_train_hint_batch(self, y: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [sample_hint_mask(label.item(), y.device, self.cfg.hint_prob) for label in y]
        )

    def training_step(self, batch, _batch_idx):
        x, y = batch
        hint_mask = self._build_train_hint_batch(y)
        logits = self(x, hint_mask)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()

        self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train/acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train/alpha", self.alpha.detach(), on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _batch_idx):
        x, y = batch
        logits = self(x, None)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()

        self.log("val/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val/acc_no_hint", acc, prog_bar=True, on_epoch=True, on_step=False)

        hint_mask = torch.zeros(x.size(0), NUM_CLASSES, device=x.device, dtype=torch.float32)
        hint_mask[:, : self.cfg.hint_ah_end_idx] = 1.0
        hint_mask[:, EMPTY_IDX] = 1.0
        hint_mask[:, JUNK_IDX] = 1.0

        pred_hint = self(x, hint_mask).argmax(dim=-1)
        eval_mask = y < self.cfg.hint_ah_end_idx
        if eval_mask.any():
            acc_hint = (pred_hint[eval_mask] == y[eval_mask]).float().mean()
            self.log("val/acc_hint_AH", acc_hint, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.scheduler_t_max,
            eta_min=self.cfg.scheduler_eta_min,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @torch.inference_mode()
    def predict_one(
        self,
        image: torch.Tensor,
        hint_mask: torch.Tensor | None = None,
    ) -> dict:
        self.eval()
        probs = F.softmax(self(image, hint_mask), dim=-1)
        conf, idx = probs.max(dim=-1)
        return {
            "class_idx": int(idx.item()),
            "confidence": float(conf.item()),
            "probs": probs.squeeze(0).cpu(),
        }
