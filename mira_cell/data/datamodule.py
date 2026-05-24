from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import lightning as L
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from mira_cell.constants import (
    CLASS_NAMES,
    LETTER_TO_IDX,
    NUM_CLASSES,
)
from mira_cell.data.transforms import build_transforms


class TargetRemap:
    def __init__(self, remap: dict[int, int]) -> None:
        self.remap = remap

    def __call__(self, target: int) -> int:
        return self.remap[target]


def _make_sampler(
    targets: Sequence[int],
    mode: str,
    class_prior: Sequence[float] | None = None,
) -> WeightedRandomSampler | None:
    targets = torch.as_tensor(targets, dtype=torch.long)
    if mode == "natural":
        return None

    counts = torch.clamp(torch.bincount(targets, minlength=NUM_CLASSES).float(), min=1)

    if mode == "uniform":
        prior = torch.ones(NUM_CLASSES) / NUM_CLASSES
    elif mode == "prior":
        if class_prior is None:
            raise ValueError("class_prior is required for mode='prior'")
        prior = torch.as_tensor(class_prior, dtype=torch.float32)
        prior = prior / prior.sum()
    else:
        raise ValueError(f"Unknown sampler mode: {mode}")

    weights = (prior / counts)[targets]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(targets),
        replacement=True,
    )


class NISTDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.data_dir = Path(cfg.data_dir)
        self.train_ds: ImageFolder | None = None
        self.val_ds: ImageFolder | None = None
        self.train_targets: list[int] = []

    def _build_folder(self, subdir: str, mode: str) -> tuple[ImageFolder, list[int]]:
        tf = build_transforms(
            mode=mode,
            num_channels=self.cfg.num_channels,
            norm_mean=tuple(self.cfg.norm_mean),
            norm_std=tuple(self.cfg.norm_std),
            cell_variants_dir=self.cfg.cell_variants_dir,
            img_size=self.cfg.img_size,
            elastic_alpha=self.cfg.elastic_alpha,
            elastic_sigma=self.cfg.elastic_sigma,
            elastic_p=self.cfg.elastic_p,
            affine_degrees=self.cfg.affine_degrees,
            affine_translate=tuple(self.cfg.affine_translate),
            affine_scale=tuple(self.cfg.affine_scale),
            affine_shear=self.cfg.affine_shear,
            affine_fill=self.cfg.affine_fill,
            affine_p=self.cfg.affine_p,
            cell_shift_translate=tuple(self.cfg.cell_shift_translate),
            blur_kernel=self.cfg.blur_kernel,
            blur_sigma=tuple(self.cfg.blur_sigma),
            blur_p=self.cfg.blur_p,
        )
        dataset = ImageFolder(self.data_dir / subdir, transform=tf)

        unknown = [name for name in dataset.classes if name not in LETTER_TO_IDX]
        if unknown:
            raise ValueError(f"Unknown class folder(s): {unknown}")

        remap = {idx: LETTER_TO_IDX[name] for name, idx in dataset.class_to_idx.items()}
        dataset.target_transform = TargetRemap(remap)
        dataset.classes = CLASS_NAMES
        dataset.class_to_idx = LETTER_TO_IDX

        mapped_targets = [remap[t] for t in dataset.targets]
        return dataset, mapped_targets

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            if self.train_ds is None:
                self.train_ds, self.train_targets = self._build_folder(
                    self.cfg.train_subdir, "train"
                )
            if self.val_ds is None:
                self.val_ds, _ = self._build_folder(self.cfg.val_subdir, "val")

    def train_dataloader(self) -> DataLoader:
        self.setup("fit")
        assert self.train_ds is not None
        sampler = _make_sampler(self.train_targets, self.cfg.sampler_mode)
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=(self.cfg.num_workers > 0 and self.cfg.persistent_workers),
        )

    def val_dataloader(self) -> DataLoader:
        self.setup("fit")
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=(self.cfg.num_workers > 0 and self.cfg.persistent_workers),
        )
