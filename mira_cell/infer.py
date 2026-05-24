from __future__ import annotations

from pathlib import Path

import fire
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from mira_cell.constants import CLASS_NAMES
from mira_cell.data.hint import allowed_letters_to_mask
from mira_cell.models.classifier import LetterClassifier


def _build_eval_transform(img_size: int, num_channels: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=num_channels),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * num_channels, (0.5,) * num_channels),
        ]
    )


def predict(
    image_path: str,
    checkpoint: str,
    model_config: str = "configs/model/resnet18_hint.yaml",
    allowed: str | None = None,
    img_size: int = 128,
    num_channels: int = 3,
) -> dict:
    cfg = OmegaConf.load(model_config)
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {"lr": 0.0, "weight_decay": 0.0, "scheduler_t_max": 1, "scheduler_eta_min": 0.0}
        ),
    )

    model = LetterClassifier.load_from_checkpoint(checkpoint, cfg=cfg, map_location="cpu")
    model.eval()

    tf = _build_eval_transform(img_size, num_channels)
    image = tf(Image.open(Path(image_path))).unsqueeze(0)

    allowed_list = list(allowed) if allowed else None
    hint_mask = allowed_letters_to_mask(allowed_list, batch_size=1, device=image.device)

    result = model.predict_one(image, hint_mask=hint_mask)
    result["label"] = CLASS_NAMES[result["class_idx"]]
    result.pop("probs", None)
    return result


def cli() -> None:
    fire.Fire(predict)


if __name__ == "__main__":
    cli()
