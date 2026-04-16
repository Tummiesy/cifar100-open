"""Generate instance-dependent closed-set noisy labels for known CIFAR-100 classes."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100

from datasets import CIFAR100SubsetByClass
from model import CIFARResNet18
from utils import ensure_dir, save_json, set_seed, split_known_unknown_classes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate instance-dependent closed-set noisy labels")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ref_ckpt", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_unknown_classes", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--closed_set_noise_rate", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def entropy_of_probs(probs: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    return -(probs * (probs + eps).log()).sum(dim=1)


def sample_truncated_flip_rate(mean: float, size: int, std: float = 0.1) -> np.ndarray:
    return np.clip(np.random.normal(loc=mean, scale=std, size=size), 0.0, 1.0)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    ckpt = torch.load(args.ref_ckpt, map_location="cpu")
    known_classes = ckpt["known_classes"]
    unknown_classes = ckpt["unknown_classes"]
    known_remap = {int(k): int(v) for k, v in ckpt["known_remap"].items()}
    num_known = int(ckpt["num_known_classes"])

    if known_classes is None or unknown_classes is None:
        known_classes, unknown_classes, known_remap = split_known_unknown_classes(
            num_total_classes=100,
            num_unknown_classes=args.num_unknown_classes,
            seed=args.seed,
            fixed_known_classes=None,
        )

    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    train_base = CIFAR100(root=args.data_root, train=True, download=True, transform=tfm)
    known_ds = CIFAR100SubsetByClass(train_base, known_classes, label_remap=known_remap)

    known_loader = DataLoader(
        known_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device(args.device)
    model = CIFARResNet18(num_classes=num_known).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    closed_samples: List[dict] = []
    assignments_for_csv: List[Dict[str, object]] = []

    with torch.no_grad():
        for batch in known_loader:
            images = batch["image"].to(device)
            original_labels = batch["original_label"]
            remapped_labels = batch["label"].to(device)
            sample_indices = batch["index"]

            logits = model(images)
            probs = torch.softmax(logits / args.temperature, dim=1)
            ents = entropy_of_probs(probs)

            q = sample_truncated_flip_rate(
                mean=args.closed_set_noise_rate,
                size=images.shape[0],
                std=0.1,
            )
            q_tensor = torch.from_numpy(q).float().to(device)

            masked_probs = probs.clone()
            masked_probs[torch.arange(images.shape[0], device=device), remapped_labels] = 0.0
            masked_sum = masked_probs.sum(dim=1, keepdim=True)

            fallback = torch.ones_like(masked_probs)
            fallback[torch.arange(images.shape[0], device=device), remapped_labels] = 0.0
            fallback = fallback / fallback.sum(dim=1, keepdim=True)

            renorm_probs = torch.where(masked_sum > 0, masked_probs / masked_sum.clamp_min(1e-12), fallback)

            final_probs = renorm_probs * q_tensor.unsqueeze(1)
            final_probs[torch.arange(images.shape[0], device=device), remapped_labels] = 1.0 - q_tensor
            final_probs = final_probs / final_probs.sum(dim=1, keepdim=True).clamp_min(1e-12)

            sampled_labels = torch.multinomial(final_probs, num_samples=1).squeeze(1)
            max_probs, _ = probs.max(dim=1)

            probs_cpu = probs.cpu()
            renorm_cpu = renorm_probs.cpu()
            final_cpu = final_probs.cpu()
            sampled_cpu = sampled_labels.cpu()

            for i in range(images.shape[0]):
                orig_remapped = int(remapped_labels[i].item())
                new_label = int(sampled_cpu[i].item())
                closed_samples.append(
                    {
                        "image": batch["image"][i],
                        "label": new_label,
                        "original_label": int(original_labels[i].item()),
                        "original_known_label": orig_remapped,
                        "source_index": int(sample_indices[i].item()),
                        "is_closed_set_noise": new_label != orig_remapped,
                        "flip_rate": float(q[i]),
                        "max_prob": float(max_probs[i].item()),
                        "entropy": float(ents[i].item()),
                        "prob_vector": probs_cpu[i].tolist(),
                        "masked_prob_vector": renorm_cpu[i].tolist(),
                        "final_prob_vector": final_cpu[i].tolist(),
                    }
                )
                assignments_for_csv.append(
                    {
                        "index": int(sample_indices[i].item()),
                        "original_label": int(original_labels[i].item()),
                        "original_known_label": orig_remapped,
                        "new_training_label": new_label,
                        "is_flipped": int(new_label != orig_remapped),
                        "flip_rate": float(q[i]),
                        "max_prob": float(max_probs[i].item()),
                        "entropy": float(ents[i].item()),
                        "masked_prob_vector": renorm_cpu[i].tolist(),
                        "final_prob_vector": final_cpu[i].tolist(),
                    }
                )

    n_known = len(closed_samples)
    n_flipped = sum(1 for s in closed_samples if s["is_closed_set_noise"])
    n_clean = n_known - n_flipped
    empirical_ratio = n_flipped / max(n_known, 1)

    closed_to_hist = Counter(int(s["label"]) for s in closed_samples if s["is_closed_set_noise"])
    relabel_pairs = Counter(
        f"{int(s['original_known_label'])}->{int(s['label'])}"
        for s in closed_samples
        if s["is_closed_set_noise"]
    )

    print("==== Closed-set Noise Stats ====")
    print(f"Known sample count: {n_known}")
    print(f"Known clean samples: {n_clean}")
    print(f"Known closed-set flipped samples: {n_flipped}")
    print(f"Empirical closed-set flip ratio: {empirical_ratio:.6f}")

    save_json(
        os.path.join(args.output_dir, "split_info.json"),
        {
            "known_classes": known_classes,
            "unknown_classes": unknown_classes,
            "known_remap": known_remap,
        },
    )

    save_json(
        os.path.join(args.output_dir, "closed_set_noise_stats.json"),
        {
            "num_known_samples": n_known,
            "num_known_clean_samples": n_clean,
            "num_known_closed_set_flipped_samples": n_flipped,
            "empirical_closed_set_flip_ratio": empirical_ratio,
            "requested_closed_set_noise_rate": args.closed_set_noise_rate,
            "temperature": args.temperature,
            "avg_max_prob": float(np.mean([s["max_prob"] for s in closed_samples])) if closed_samples else 0.0,
            "avg_entropy": float(np.mean([s["entropy"] for s in closed_samples])) if closed_samples else 0.0,
            "closed_relabel_to_class_histogram": {
                str(k): int(v) for k, v in sorted(closed_to_hist.items())
            },
            "closed_relabel_pair_histogram": dict(sorted(relabel_pairs.items())),
        },
    )

    assignment_csv = os.path.join(args.output_dir, "closed_set_assignments.csv")
    with open(assignment_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "original_label",
                "original_known_label",
                "new_training_label",
                "is_flipped",
                "flip_rate",
                "max_prob",
                "entropy",
                "masked_prob_vector_json",
                "final_prob_vector_json",
            ]
        )
        for a in assignments_for_csv:
            writer.writerow(
                [
                    a["index"],
                    a["original_label"],
                    a["original_known_label"],
                    a["new_training_label"],
                    a["is_flipped"],
                    f"{a['flip_rate']:.8f}",
                    f"{a['max_prob']:.8f}",
                    f"{a['entropy']:.8f}",
                    json.dumps(a["masked_prob_vector"]),
                    json.dumps(a["final_prob_vector"]),
                ]
            )

    tensor_path = os.path.join(args.output_dir, "closed_set_samples.pt")
    torch.save(closed_samples, tensor_path)

    print(f"Saved stats: {os.path.join(args.output_dir, 'closed_set_noise_stats.json')}")
    print(f"Saved assignments: {assignment_csv}")
    print(f"Saved closed known samples: {tensor_path}")


if __name__ == "__main__":
    main()
