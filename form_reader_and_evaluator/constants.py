"""Static class layout for the letter classifier.

These never change across configs; they pin the model output ordering.
"""

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CLASS_NAMES = LETTERS + ["<empty>", "<junk>"]
NUM_CLASSES = len(CLASS_NAMES)
LETTER_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
EMPTY_IDX = LETTER_TO_IDX["<empty>"]
JUNK_IDX = LETTER_TO_IDX["<junk>"]
