"""Generate mixed closed-set + open-set instance-dependent noisy training data."""

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

from datasets import CIFAR100SubsetByClass, build_final_mixed_noise_dataset
from model import CIFARResNet18
from utils import ensure_dir, save_json, set_seed, split_known_unknown_classes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mixed closed/open instance-dependent noisy dataset")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ref_ckpt", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_unknown_classes", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--closed_set_noise_rate", type=float, default=0.2)
    parser.add_argument("--open_set_noise_ratio", type=float, default=0.2)
    parser.add_argument(
        "--ratio_mode",
        type=str,
        default="fraction_total",
        choices=["fraction_total", "relative_clean"],
    )
    parser.add_argument("--hardness_mode", type=str, default="all", choices=["all", "hard", "easy", "topk"])
    parser.add_argument("--hardness_threshold", type=float, default=0.3)
    parser.add_argument("--topk", type=int, default=5000)
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


def sample_truncated_flip_rate(mean: float, size: int, std: float = 0.1) -> np.ndarray:
    return np.clip(np.random.normal(loc=mean, scale=std, size=size), 0.0, 1.0)


def generate_closed_set_samples(
    model: torch.nn.Module,
    known_loader: DataLoader,
    temperature: float,
    closed_set_noise_rate: float,
    device: torch.device,
) -> List[dict]:
    closed_samples: List[dict] = []

    with torch.no_grad():
        for batch in known_loader:
            images = batch["image"].to(device)
            original_labels = batch["original_label"]
            remapped_labels = batch["label"].to(device)
            sample_indices = batch["index"]

            logits = model(images)
            probs = torch.softmax(logits / temperature, dim=1)
            ents = entropy_of_probs(probs)

            q = sample_truncated_flip_rate(mean=closed_set_noise_rate, size=images.shape[0], std=0.1)
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

    return closed_samples


def generate_open_set_assignments(
    model: torch.nn.Module,
    unknown_loader: DataLoader,
    temperature: float,
    device: torch.device,
) -> List[dict]:
    assignments: List[dict] = []

    with torch.no_grad():
        for batch in unknown_loader:
            images = batch["image"].to(device)
            original_unknown_labels = batch["original_label"]
            sample_indices = batch["index"]

            logits = model(images)
            probs = torch.softmax(logits / temperature, dim=1)
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

    return assignments


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
    unknown_ds = CIFAR100SubsetByClass(train_base, unknown_classes, label_remap=None)

    known_loader = DataLoader(
        known_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
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

    closed_known_samples = generate_closed_set_samples(
        model=model,
        known_loader=known_loader,
        temperature=args.temperature,
        closed_set_noise_rate=args.closed_set_noise_rate,
        device=device,
    )

    all_unknown_assignments = generate_open_set_assignments(
        model=model,
        unknown_loader=unknown_loader,
        temperature=args.temperature,
        device=device,
    )
    selected_unknown = filter_unknown(
        assignments=all_unknown_assignments,
        mode=args.hardness_mode,
        threshold=args.hardness_threshold,
        topk=args.topk,
    )

    final_ds = build_final_mixed_noise_dataset(
        closed_known_samples=closed_known_samples,
        unknown_assignments=selected_unknown,
        open_set_noise_ratio=args.open_set_noise_ratio,
        ratio_mode=args.ratio_mode,
    )

    n_known = len(closed_known_samples)
    n_closed_flipped = sum(1 for s in closed_known_samples if s["is_closed_set_noise"])
    n_known_clean = n_known - n_closed_flipped
    closed_ratio = n_closed_flipped / max(n_known, 1)

    n_unknown_pool = len(unknown_ds)
    n_unknown_selected = len(selected_unknown)
    n_open_used = sum(1 for s in final_ds.samples if s["noise_type"] == "open")
    n_final = len(final_ds)

    clean_count = sum(1 for s in final_ds.samples if s["noise_type"] == "clean")
    closed_count = sum(1 for s in final_ds.samples if s["noise_type"] == "closed")
    open_count = sum(1 for s in final_ds.samples if s["noise_type"] == "open")

    closed_to_hist = Counter(int(s["label"]) for s in closed_known_samples if s["is_closed_set_noise"])
    open_hist = Counter(int(s["label"]) for s in final_ds.samples if s["noise_type"] == "open")

    proportions = {
        "clean": clean_count / max(n_final, 1),
        "closed": closed_count / max(n_final, 1),
        "open": open_count / max(n_final, 1),
    }

    print("==== Mixed Noise Stats ====")
    print(f"Known clean samples: {n_known_clean}")
    print(f"Known closed-set flipped samples: {n_closed_flipped}")
    print(f"Empirical closed-set flip ratio: {closed_ratio:.6f}")
    print(f"Unknown pool samples: {n_unknown_pool}")
    print(f"Selected open-set noisy samples: {n_unknown_selected}")
    print(f"Final open-set noisy samples used: {n_open_used}")
    print(f"Final dataset size: {n_final}")
    print(
        "Final proportions (clean/closed/open): "
        f"{proportions['clean']:.6f}/{proportions['closed']:.6f}/{proportions['open']:.6f}"
    )

    save_json(
        os.path.join(args.output_dir, "split_info.json"),
        {
            "known_classes": known_classes,
            "unknown_classes": unknown_classes,
            "known_remap": known_remap,
        },
    )

    save_json(
        os.path.join(args.output_dir, "mixed_noise_stats.json"),
        {
            "num_known_samples": n_known,
            "num_known_clean_samples": n_known_clean,
            "num_known_closed_set_flipped_samples": n_closed_flipped,
            "empirical_closed_set_flip_ratio": closed_ratio,
            "num_unknown_pool_samples": n_unknown_pool,
            "num_selected_open_set_samples": n_unknown_selected,
            "num_open_set_noisy_samples_used": n_open_used,
            "num_final_dataset_samples": n_final,
            "final_proportions": proportions,
            "closed_relabel_to_class_histogram": {str(k): int(v) for k, v in sorted(closed_to_hist.items())},
            "open_assigned_label_histogram": {str(k): int(v) for k, v in sorted(open_hist.items())},
            "temperature": args.temperature,
            "closed_set_noise_rate": args.closed_set_noise_rate,
            "requested_open_set_noise_ratio": args.open_set_noise_ratio,
            "ratio_mode": args.ratio_mode,
            "hardness_mode": args.hardness_mode,
            "hardness_threshold": args.hardness_threshold,
            "topk": args.topk,
        },
    )

    closed_csv = os.path.join(args.output_dir, "closed_set_assignments.csv")
    with open(closed_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source_index",
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
        for s in closed_known_samples:
            writer.writerow(
                [
                    s["source_index"],
                    s["original_label"],
                    s["original_known_label"],
                    s["label"],
                    int(s["is_closed_set_noise"]),
                    f"{s['flip_rate']:.8f}",
                    f"{s['max_prob']:.8f}",
                    f"{s['entropy']:.8f}",
                    json.dumps(s["masked_prob_vector"]),
                    json.dumps(s["final_prob_vector"]),
                ]
            )

    unknown_csv = os.path.join(args.output_dir, "unknown_noisy_assignments.csv")
    with open(unknown_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["index", "original_unknown_class", "noisy_label", "max_prob", "entropy", "prob_vector_json"]
        )
        for a in selected_unknown:
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

    final_meta_csv = os.path.join(args.output_dir, "final_dataset_metadata.csv")
    with open(final_meta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source_index",
                "train_label",
                "original_label",
                "noise_type",
                "is_closed_set_noise",
                "is_open_set_noise",
                "flip_rate",
                "max_prob",
                "entropy",
            ]
        )
        for s in final_ds.samples:
            writer.writerow(
                [
                    s.get("source_index", -1),
                    s["label"],
                    s["original_label"],
                    s["noise_type"],
                    int(s["is_closed_set_noise"]),
                    int(s["is_open_set_noise"]),
                    "" if s.get("flip_rate") is None else f"{s['flip_rate']:.8f}",
                    "" if s.get("max_prob") is None else f"{s['max_prob']:.8f}",
                    "" if s.get("entropy") is None else f"{s['entropy']:.8f}",
                ]
            )

    torch.save(closed_known_samples, os.path.join(args.output_dir, "closed_set_samples.pt"))
    torch.save(final_ds.samples, os.path.join(args.output_dir, "final_dataset_samples.pt"))

    print(f"Saved stats: {os.path.join(args.output_dir, 'mixed_noise_stats.json')}")
    print(f"Saved closed assignments: {closed_csv}")
    print(f"Saved open assignments: {unknown_csv}")
    print(f"Saved final metadata: {final_meta_csv}")
    print(f"Saved final samples: {os.path.join(args.output_dir, 'final_dataset_samples.pt')}")


if __name__ == "__main__":
    main()
