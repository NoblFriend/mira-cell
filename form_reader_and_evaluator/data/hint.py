from __future__ import annotations

import random

import numpy as np
import torch

from form_reader_and_evaluator.constants import (
    EMPTY_IDX,
    JUNK_IDX,
    LETTER_TO_IDX,
    LETTERS,
    NUM_CLASSES,
)

_HINT_WEIGHTS_RAW = {
    "F": 100,
    "E": 90,
    "H": 70,
    "D": 40,
    "G": 40,
    "I": 5,
    "J": 3,
    "K": 2,
    "L": 1,
    "M": 1,
    "N": 1,
    "O": 1,
    "P": 1,
    "Q": 1,
    "R": 1,
    "S": 1,
    "T": 1,
    "U": 1,
    "V": 1,
    "W": 1,
    "X": 1,
    "Y": 1,
    "Z": 1,
    "A": 1,
    "B": 1,
    "C": 1,
}


def sample_hint_mask(true_class_idx: int, device, prob: float) -> torch.Tensor:
    mask = torch.zeros(NUM_CLASSES, dtype=torch.float32, device=device)

    if true_class_idx in (EMPTY_IDX, JUNK_IDX):
        true_class_idx = random.choice(range(len(LETTERS)))

    if random.random() < prob:
        mask[:] = 1.0
        return mask

    weights = np.array(
        [_HINT_WEIGHTS_RAW[c] if LETTER_TO_IDX[c] >= true_class_idx else 0.0 for c in LETTERS],
        dtype=np.float64,
    )
    weights /= float(weights.sum())
    end_letter = np.random.choice(LETTERS, p=weights)
    end_idx = LETTERS.index(end_letter)

    mask[: end_idx + 1] = 1.0
    mask[EMPTY_IDX] = 1.0
    mask[JUNK_IDX] = 1.0
    return mask.to(device=device, dtype=torch.float32)


def no_hint_mask(batch_size: int, device) -> torch.Tensor:
    return torch.ones(batch_size, NUM_CLASSES, device=device, dtype=torch.float32)


def allowed_letters_to_mask(
    allowed: list[str] | None,
    batch_size: int,
    device,
) -> torch.Tensor | None:
    if allowed is None:
        return None
    mask = torch.zeros(batch_size, NUM_CLASSES, device=device, dtype=torch.float32)
    for letter in allowed:
        idx = LETTER_TO_IDX.get(letter)
        if idx is not None:
            mask[:, idx] = 1.0
    mask[:, EMPTY_IDX] = 1.0
    mask[:, JUNK_IDX] = 1.0
    return mask
