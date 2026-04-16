"""Dataset utilities for known/unknown splits and final noisy training set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

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
    Final training dataset = clean known samples + generated open-set noisy samples.

    All returned training labels are in [0, num_known_classes - 1].
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
                "is_open_set_noise": s["is_open_set_noise"],
                "original_label": s["original_label"],
                "source_index": s.get("source_index", -1),
                "max_prob": s.get("max_prob", None),
                "entropy": s.get("entropy", None),
            }
        return s["image"], s["label"]


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
    if ratio_mode not in {"fraction_total", "relative_clean"}:
        raise ValueError("ratio_mode must be one of {'fraction_total', 'relative_clean'}")

    n_clean = len(clean_known_dataset)
    available_noisy = len(unknown_assignments)

    if ratio_mode == "fraction_total":
        if not (0.0 <= open_set_noise_ratio < 1.0):
            raise ValueError("fraction_total ratio must be in [0, 1)")
        target_noisy = int(round(open_set_noise_ratio * n_clean / (1.0 - open_set_noise_ratio)))
    else:
        if open_set_noise_ratio < 0:
            raise ValueError("relative_clean ratio must be >= 0")
        target_noisy = int(round(open_set_noise_ratio * n_clean))

    target_noisy = min(target_noisy, available_noisy)
    selected_unknown = unknown_assignments[:target_noisy]

    final_samples: List[dict] = []

    for i in range(len(clean_known_dataset)):
        s = clean_known_dataset[i]
        final_samples.append(
            {
                "image": s["image"],
                "label": int(s["label"]),
                "is_open_set_noise": False,
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
                "original_label": int(s["original_unknown_class"]),
                "source_index": int(s["index"]),
                "max_prob": float(s["max_prob"]),
                "entropy": float(s["entropy"]),
            }
        )

    return FinalOpenSetNoisyDataset(samples=final_samples, include_metadata=True)
