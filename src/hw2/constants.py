"""Project-wide constants shared by every module."""

CLASS_NAMES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "bus",
    "motor",
)
NUM_CLASSES: int = 8
MAX_DETS: int = 500
SCORE_THRESH: float = 0.001
IGNORE_FILL: int = 114
SEED: int = 42
