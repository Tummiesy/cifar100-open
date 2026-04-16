"""Generate instance-dependent open-set noisy labels from unknown CIFAR-100 classes.

Why this is instance-dependent open-set noise:
- Unknown-class samples are not discarded.
- Each unknown image is mapped to a known-class noisy label by sampling from
  p(c|x_unknown) predicted by a clean reference model over known classes.
- Therefore noisy labels depend on each sample's visual content, not uniform random noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100

from datasets import CIFAR100SubsetByClass, build_final_open_set_dataset
from model import CIFARResNet18
from utils import ensure_dir, save_json, set_seed, split_known_unknown_classes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate instance-dependent open-set noisy labels")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ref_ckpt", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_unknown_classes", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--hardness_mode", type=str, default="all", choices=["all", "hard", "easy", "topk"])
    parser.add_argument("--hardness_threshold", type=float, default=0.3)
    parser.add_argument("--topk", type=int, default=5000)
    parser.add_argument("--open_set_noise_ratio", type=float, default=0.2)
    parser.add_argument(
        "--ratio_mode",
        type=str,
        default="fraction_total",
        choices=["fraction_total", "relative_clean"],
        help="fraction_total: M/(N_clean+M); relative_clean: M/N_clean",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def entropy_of_probs(probs: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    return -(probs * (probs + eps).log()).sum(dim=1)


def filter_unknown(assignments: List[dict], mode: str, threshold: float, topk: int) -> List[dict]:
    if mode == "all":
        return assignments
    if mode == "hard":
        return [a for a in assignments if a["max_prob"] >= threshold]
    if mode == "easy":
        return [a for a in assignments if a["max_prob"] < threshold]
    if mode == "topk":
        assignments_sorted = sorted(assignments, key=lambda x: x["max_prob"], reverse=True)
        return assignments_sorted[: min(topk, len(assignments_sorted))]
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    ckpt = torch.load(args.ref_ckpt, map_location="cpu")
    known_classes = ckpt["known_classes"]
    unknown_classes = ckpt["unknown_classes"]
    known_remap = {int(k): int(v) for k, v in ckpt["known_remap"].items()}
    num_known = int(ckpt["num_known_classes"])

    # Fallback: if checkpoint lacks split metadata, reconstruct from seed.
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
    unknown_ds = CIFAR100SubsetByClass(train_base, unknown_classes, label_remap=None)

    unknown_loader = DataLoader(
        unknown_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device(args.device)
    model = CIFARResNet18(num_classes=num_known).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    assignments: List[dict] = []

    with torch.no_grad():
        for batch in unknown_loader:
            images = batch["image"].to(device)
            original_unknown_labels = batch["original_label"]
            sample_indices = batch["index"]

            logits = model(images)
            probs = torch.softmax(logits / args.temperature, dim=1)
            max_probs, _ = probs.max(dim=1)
            ents = entropy_of_probs(probs)

            sampled_labels = torch.multinomial(probs, num_samples=1).squeeze(1)

            probs_cpu = probs.cpu()
            for i in range(images.shape[0]):
                assignments.append(
                    {
                        "image": batch["image"][i],
                        "index": int(sample_indices[i].item()),
                        "original_unknown_class": int(original_unknown_labels[i].item()),
                        "noisy_label": int(sampled_labels[i].item()),
                        "prob_vector": probs_cpu[i].tolist(),
                        "max_prob": float(max_probs[i].item()),
                        "entropy": float(ents[i].item()),
                    }
                )

    selected = filter_unknown(
        assignments,
        mode=args.hardness_mode,
        threshold=args.hardness_threshold,
        topk=args.topk,
    )

    final_ds = build_final_open_set_dataset(
        clean_known_dataset=known_ds,
        unknown_assignments=selected,
        open_set_noise_ratio=args.open_set_noise_ratio,
        ratio_mode=args.ratio_mode,
    )

    n_clean = len(known_ds)
    n_unknown = len(unknown_ds)
    n_selected = len(selected)
    n_final = len(final_ds)
    n_open = sum(1 for s in final_ds.samples if s["is_open_set_noise"])
    final_ratio = n_open / max(n_final, 1)

    noisy_label_hist = Counter(a["noisy_label"] for a in selected)
    avg_max_prob = sum(a["max_prob"] for a in selected) / max(len(selected), 1)
    avg_entropy = sum(a["entropy"] for a in selected) / max(len(selected), 1)

    per_unknown_summary = defaultdict(lambda: Counter())
    for a in selected:
        per_unknown_summary[a["original_unknown_class"]][a["noisy_label"]] += 1

    print("==== Open-set Noise Stats ====")
    print(f"Known classes ({len(known_classes)}): {known_classes}")
    print(f"Unknown classes ({len(unknown_classes)}): {unknown_classes}")
    print(f"Clean known samples: {n_clean}")
    print(f"Unknown samples (pool): {n_unknown}")
    print(f"Selected unknown samples: {n_selected}")
    print(f"Final dataset size: {n_final}")
    print(f"Final open-set noise ratio: {final_ratio:.6f}")
    print(f"Average selected max probability: {avg_max_prob:.6f}")
    print(f"Average selected entropy: {avg_entropy:.6f}")

    save_json(
        os.path.join(args.output_dir, "split_info.json"),
        {
            "known_classes": known_classes,
            "unknown_classes": unknown_classes,
            "known_remap": known_remap,
        },
    )

    save_json(
        os.path.join(args.output_dir, "noise_stats.json"),
        {
            "num_clean_known_samples": n_clean,
            "num_unknown_pool_samples": n_unknown,
            "num_selected_unknown_samples": n_selected,
            "num_final_dataset_samples": n_final,
            "num_open_set_noisy_samples_used": n_open,
            "final_open_set_noise_ratio": final_ratio,
            "hardness_mode": args.hardness_mode,
            "hardness_threshold": args.hardness_threshold,
            "topk": args.topk,
            "temperature": args.temperature,
            "ratio_mode": args.ratio_mode,
            "requested_open_set_noise_ratio": args.open_set_noise_ratio,
            "avg_selected_max_prob": avg_max_prob,
            "avg_selected_entropy": avg_entropy,
            "noisy_label_histogram": {str(k): int(v) for k, v in sorted(noisy_label_hist.items())},
        },
    )

    assignment_csv = os.path.join(args.output_dir, "unknown_noisy_assignments.csv")
    with open(assignment_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "original_unknown_class",
                "noisy_label",
                "max_prob",
                "entropy",
                "prob_vector_json",
            ]
        )
        for a in selected:
            writer.writerow(
                [
                    a["index"],
                    a["original_unknown_class"],
                    a["noisy_label"],
                    f"{a['max_prob']:.8f}",
                    f"{a['entropy']:.8f}",
                    json.dumps(a["prob_vector"]),
                ]
            )

    per_unknown_csv = os.path.join(args.output_dir, "per_unknown_assignment_summary.csv")
    with open(per_unknown_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["original_unknown_class", "assigned_known_label", "count"])
        for unk_cls in sorted(per_unknown_summary.keys()):
            for noisy_lbl, count in sorted(per_unknown_summary[unk_cls].items()):
                writer.writerow([unk_cls, noisy_lbl, count])

    final_meta_csv = os.path.join(args.output_dir, "final_dataset_metadata.csv")
    with open(final_meta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source_index",
                "train_label",
                "is_open_set_noise",
                "original_label",
                "max_prob",
                "entropy",
            ]
        )
        for s in final_ds.samples:
            writer.writerow(
                [
                    s.get("source_index", -1),
                    s["label"],
                    int(s["is_open_set_noise"]),
                    s["original_label"],
                    "" if s.get("max_prob") is None else f"{s['max_prob']:.8f}",
                    "" if s.get("entropy") is None else f"{s['entropy']:.8f}",
                ]
            )

    tensor_path = os.path.join(args.output_dir, "final_dataset_samples.pt")
    torch.save(final_ds.samples, tensor_path)

    print(f"Saved split info: {os.path.join(args.output_dir, 'split_info.json')}")
    print(f"Saved stats: {os.path.join(args.output_dir, 'noise_stats.json')}")
    print(f"Saved assignments: {assignment_csv}")
    print(f"Saved per-unknown summary: {per_unknown_csv}")
    print(f"Saved final dataset metadata: {final_meta_csv}")
    print(f"Saved final dataset tensor list: {tensor_path}")


if __name__ == "__main__":
    main()
