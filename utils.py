"""Utility functions for reproducible CIFAR-100 open-set noise experiments."""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: dict) -> None:
    """Save JSON with pretty formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_known_classes(known_classes_file: str) -> List[int]:
    """Load fixed known-class ids from file (JSON list or newline-separated ints)."""
    with open(known_classes_file, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if raw.startswith("["):
        classes = json.loads(raw)
    else:
        classes = [int(line.strip()) for line in raw.splitlines() if line.strip()]

    return [int(c) for c in classes]


def split_known_unknown_classes(
    num_total_classes: int = 100,
    num_unknown_classes: int = 20,
    seed: int = 0,
    fixed_known_classes: Optional[Sequence[int]] = None,
) -> Tuple[List[int], List[int], Dict[int, int]]:
    """
    Split class ids into known/unknown groups and create known label remapping.

    Returns:
        known_classes: sorted original class ids used as known classes.
        unknown_classes: sorted original class ids used as unknown classes.
        known_remap: dict mapping original known class id -> new id in [0, K-1].
    """
    if num_unknown_classes <= 0 or num_unknown_classes >= num_total_classes:
        raise ValueError("num_unknown_classes must be in (0, num_total_classes)")

    all_classes = list(range(num_total_classes))

    if fixed_known_classes is not None:
        known_classes = sorted(int(c) for c in fixed_known_classes)
        expected_known = num_total_classes - num_unknown_classes
        if len(known_classes) != expected_known:
            raise ValueError(
                f"Expected {expected_known} known classes, got {len(known_classes)}"
            )
        if len(set(known_classes)) != len(known_classes):
            raise ValueError("known_classes contains duplicates")
        if min(known_classes) < 0 or max(known_classes) >= num_total_classes:
            raise ValueError("known_classes contains out-of-range ids")
    else:
        rng = random.Random(seed)
        unknown_classes = sorted(rng.sample(all_classes, k=num_unknown_classes))
        known_classes = sorted(list(set(all_classes) - set(unknown_classes)))

    unknown_classes = sorted(list(set(all_classes) - set(known_classes)))
    known_remap = {orig_cls: new_id for new_id, orig_cls in enumerate(known_classes)}

    return known_classes, unknown_classes, known_remap
