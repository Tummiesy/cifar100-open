"""Dataset utilities for known/unknown splits and final noisy training set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100


@dataclass
class KnownSample:
    index: int
    original_label: int
    remapped_label: int


@dataclass
class UnknownSample:
    index: int
    original_unknown_label: int


class CIFAR100SubsetByClass(Dataset):
    """
    CIFAR-100 subset by class with optional label remapping.

    Returns dict samples to keep metadata visible throughout the pipeline.
    """

    def __init__(
        self,
        base_dataset: CIFAR100,
        target_classes: List[int],
        label_remap: Optional[Dict[int, int]] = None,
        return_original_label: bool = True,
    ):
        self.base = base_dataset
        self.target_classes = set(target_classes)
        self.label_remap = label_remap
        self.return_original_label = return_original_label

        self.indices: List[int] = []
        self.original_labels: List[int] = []
        self.train_labels: List[int] = []

        targets = self.base.targets
        for idx, y in enumerate(targets):
            if y in self.target_classes:
                self.indices.append(idx)
                self.original_labels.append(y)
                self.train_labels.append(label_remap[y] if label_remap is not None else y)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        base_idx = self.indices[i]
        image, _ = self.base[base_idx]

        item = {
            "image": image,
            "label": int(self.train_labels[i]),
            "index": int(base_idx),
        }
        if self.return_original_label:
            item["original_label"] = int(self.original_labels[i])
        return item


class FinalOpenSetNoisyDataset(Dataset):
    """
    Final training dataset for clean/closed/open noisy samples.

    Backward compatible with the original open-set-only return format while exposing
    richer metadata for mixed-noise workflows.
    """

    def __init__(self, samples: List[dict], include_metadata: bool = True):
        self.samples = samples
        self.include_metadata = include_metadata

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        if self.include_metadata:
            return {
                "image": s["image"],
                "label": s["label"],
                "original_label": s["original_label"],
                "source_index": s.get("source_index", -1),
                "is_open_set_noise": s.get("is_open_set_noise", False),
                "is_closed_set_noise": s.get("is_closed_set_noise", False),
                "noise_type": s.get("noise_type", "open" if s.get("is_open_set_noise", False) else "clean"),
                "flip_rate": s.get("flip_rate", None),
                "max_prob": s.get("max_prob", None),
                "entropy": s.get("entropy", None),
            }
        return s["image"], s["label"]


def _resolve_target_noisy_count(
    n_known: int,
    available_noisy: int,
    open_set_noise_ratio: float,
    ratio_mode: str,
) -> int:
    if ratio_mode not in {"fraction_total", "relative_clean"}:
        raise ValueError("ratio_mode must be one of {'fraction_total', 'relative_clean'}")

    if ratio_mode == "fraction_total":
        if not (0.0 <= open_set_noise_ratio < 1.0):
            raise ValueError("fraction_total ratio must be in [0, 1)")
        target_noisy = int(round(open_set_noise_ratio * n_known / (1.0 - open_set_noise_ratio)))
    else:
        if open_set_noise_ratio < 0:
            raise ValueError("relative_clean ratio must be >= 0")
        target_noisy = int(round(open_set_noise_ratio * n_known))

    return min(target_noisy, available_noisy)


def build_final_open_set_dataset(
    clean_known_dataset: CIFAR100SubsetByClass,
    unknown_assignments: List[dict],
    open_set_noise_ratio: float,
    ratio_mode: str = "fraction_total",
) -> FinalOpenSetNoisyDataset:
    """
    Build final training samples with configurable ratio semantics.

    ratio_mode:
      - fraction_total: ratio = M / (N_clean + M)
      - relative_clean: ratio = M / N_clean
    """
    n_clean = len(clean_known_dataset)
    target_noisy = _resolve_target_noisy_count(
        n_known=n_clean,
        available_noisy=len(unknown_assignments),
        open_set_noise_ratio=open_set_noise_ratio,
        ratio_mode=ratio_mode,
    )
    selected_unknown = unknown_assignments[:target_noisy]

    final_samples: List[dict] = []

    for i in range(len(clean_known_dataset)):
        s = clean_known_dataset[i]
        final_samples.append(
            {
                "image": s["image"],
                "label": int(s["label"]),
                "is_open_set_noise": False,
                "is_closed_set_noise": False,
                "noise_type": "clean",
                "original_label": int(s["original_label"]),
                "source_index": int(s["index"]),
            }
        )

    for s in selected_unknown:
        final_samples.append(
            {
                "image": s["image"],
                "label": int(s["noisy_label"]),
                "is_open_set_noise": True,
                "is_closed_set_noise": False,
                "noise_type": "open",
                "original_label": int(s["original_unknown_class"]),
                "source_index": int(s["index"]),
                "max_prob": float(s["max_prob"]),
                "entropy": float(s["entropy"]),
            }
        )

    return FinalOpenSetNoisyDataset(samples=final_samples, include_metadata=True)


def build_final_mixed_noise_dataset(
    closed_known_samples: List[dict],
    unknown_assignments: List[dict],
    open_set_noise_ratio: float,
    ratio_mode: str = "fraction_total",
) -> FinalOpenSetNoisyDataset:
    """
    Build final mixed-noise dataset:
      - known samples (clean and/or closed-set noisy) from closed_known_samples
      - open-set noisy unknown samples from unknown_assignments

    ratio_mode:
      - fraction_total: ratio = M_open / (N_known + M_open)
      - relative_clean: ratio = M_open / N_known
    """
    n_known = len(closed_known_samples)
    target_noisy = _resolve_target_noisy_count(
        n_known=n_known,
        available_noisy=len(unknown_assignments),
        open_set_noise_ratio=open_set_noise_ratio,
        ratio_mode=ratio_mode,
    )
    selected_unknown = unknown_assignments[:target_noisy]

    final_samples: List[dict] = []

    for s in closed_known_samples:
        final_samples.append(
            {
                "image": s["image"],
                "label": int(s["label"]),
                "original_label": int(s["original_label"]),
                "source_index": int(s.get("source_index", s.get("index", -1))),
                "is_open_set_noise": False,
                "is_closed_set_noise": bool(s.get("is_closed_set_noise", False)),
                "noise_type": "closed" if s.get("is_closed_set_noise", False) else "clean",
                "flip_rate": None if s.get("flip_rate") is None else float(s["flip_rate"]),
                "max_prob": None if s.get("max_prob") is None else float(s["max_prob"]),
                "entropy": None if s.get("entropy") is None else float(s["entropy"]),
            }
        )

    for s in selected_unknown:
        final_samples.append(
            {
                "image": s["image"],
                "label": int(s["noisy_label"]),
                "original_label": int(s["original_unknown_class"]),
                "source_index": int(s["index"]),
                "is_open_set_noise": True,
                "is_closed_set_noise": False,
                "noise_type": "open",
                "flip_rate": None,
                "max_prob": float(s["max_prob"]),
                "entropy": float(s["entropy"]),
            }
        )

    return FinalOpenSetNoisyDataset(samples=final_samples, include_metadata=True)
