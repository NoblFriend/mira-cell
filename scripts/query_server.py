from __future__ import annotations

from pathlib import Path

import fire
import httpx
import numpy as np
from PIL import Image

from mira_cell.constants import CLASS_NAMES, EMPTY_IDX, JUNK_IDX, LETTER_TO_IDX, NUM_CLASSES


def _preprocess(image_path: str, img_size: int, num_channels: int) -> np.ndarray:
    image = Image.open(Path(image_path)).convert("L").resize((img_size, img_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5
    array = np.repeat(array[None, ...], num_channels, axis=0)
    return array[None, ...].astype(np.float32)


def _hint_mask(allowed: str | None) -> np.ndarray:
    mask = np.zeros((1, NUM_CLASSES), dtype=np.float32)
    if allowed is None:
        mask[:] = 1.0
        return mask
    for letter in allowed:
        idx = LETTER_TO_IDX.get(letter)
        if idx is not None:
            mask[0, idx] = 1.0
    mask[0, EMPTY_IDX] = 1.0
    mask[0, JUNK_IDX] = 1.0
    return mask


def query(
    image_path: str,
    allowed: str | None = None,
    url: str = "http://127.0.0.1:5001/invocations",
    img_size: int = 128,
    num_channels: int = 3,
) -> dict:
    payload = {
        "inputs": {
            "image": _preprocess(image_path, img_size, num_channels).tolist(),
            "hint_mask": _hint_mask(allowed).tolist(),
        }
    }
    response = httpx.post(url, json=payload, timeout=30.0)
    response.raise_for_status()
    predictions = response.json()["predictions"]
    if isinstance(predictions, dict):
        predictions = predictions["logits"]
    logits = np.asarray(predictions, dtype=np.float32)[0]
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    order = probs.argsort()[::-1]
    top1, top2 = order[0], order[1]
    result = {
        "label": CLASS_NAMES[int(top1)],
        "confidence": float(probs[top1]),
        "margin": float(probs[top1] - probs[top2]),
    }
    print(result)
    return result


def cli() -> None:
    fire.Fire(query)


if __name__ == "__main__":
    cli()
