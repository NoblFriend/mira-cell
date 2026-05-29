from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class AddFrameTensor:
    def __init__(self, variants_dir: str, out_size: int) -> None:
        self.cells: list[torch.Tensor] = []
        for path in Path(variants_dir).glob("*"):
            img = Image.open(path).convert("L").resize((out_size, out_size))
            arr = np.array(img, dtype=np.float32) / 255.0
            self.cells.append(torch.from_numpy(arr).unsqueeze(0))
        if not self.cells:
            raise FileNotFoundError(f"No cell-frame variants found in {variants_dir!r}")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        c, _h, _w = x.shape
        letter = x.mean(0, keepdim=True).clamp(0, 1)
        idx = int(torch.randint(0, len(self.cells), (1,)).item())
        cell = self.cells[idx].to(dtype=letter.dtype, device=letter.device)
        out = torch.minimum(letter, cell)
        if c > 1:
            out = out.repeat(c, 1, 1)
        return out.to(x.dtype)


def build_transforms(
    *,
    mode: str,
    num_channels: int,
    norm_mean: tuple[float, float, float],
    norm_std: tuple[float, float, float],
    cell_variants_dir: str,
    img_size: int,
    elastic_alpha: float,
    elastic_sigma: float,
    elastic_p: float,
    affine_degrees: float,
    affine_translate: tuple[float, float],
    affine_scale: tuple[float, float],
    affine_shear: float,
    affine_fill: int,
    affine_p: float,
    cell_shift_translate: tuple[float, float],
    blur_kernel: int,
    blur_sigma: tuple[float, float],
    blur_p: float,
) -> transforms.Compose:
    letter_aug = transforms.Compose(
        [
            transforms.RandomApply(
                [transforms.ElasticTransform(alpha=elastic_alpha, sigma=elastic_sigma)],
                p=elastic_p,
            ),
            transforms.RandomApply(
                [
                    transforms.RandomAffine(
                        degrees=affine_degrees,
                        translate=tuple(affine_translate),
                        scale=tuple(affine_scale),
                        shear=affine_shear,
                        fill=affine_fill,
                    )
                ],
                p=affine_p,
            ),
        ]
    )

    cell_shift = transforms.RandomAffine(
        degrees=0,
        translate=tuple(cell_shift_translate),
        fill=1,
    )

    printer = transforms.RandomApply(
        [transforms.GaussianBlur(blur_kernel, sigma=tuple(blur_sigma))],
        p=blur_p,
    )

    pipeline: list = [
        transforms.Grayscale(num_output_channels=num_channels),
    ]
    if mode == "train":
        pipeline.append(letter_aug)
    pipeline.extend(
        [
            transforms.ToTensor(),
            AddFrameTensor(cell_variants_dir, img_size),
        ]
    )
    if mode == "train":
        pipeline.extend([cell_shift, printer])
    pipeline.append(transforms.Normalize(tuple(norm_mean), tuple(norm_std)))

    return transforms.Compose(pipeline)
