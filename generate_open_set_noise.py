"""Generate final CIFAR-100 training data with close-set and open-set noise.

Pipeline:
1. Load a clean reference classifier. The new default reference model predicts all
   100 CIFAR-100 classes.
2. Use the saved 80/20 known/unknown split only for noise generation and final
   label-space construction.
3. For known samples, generate close-set labels by restricting the 100-class
   reference probabilities to known classes, masking the true class, and sampling
   a wrong known-class label.
4. For unknown samples, generate open-set labels by restricting the 100-class
   reference probabilities to known classes and sampling a known-class label.
5. Save D_train = D_clean ∪ D_csn ∪ D_osn, where every final label is in [0, 79].
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from typing import List

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100

from datasets import CIFAR100SubsetByClass, build_final_clean_close_open_dataset
from model import CIFARResNet18
from utils import ensure_dir, save_json, set_seed, split_known_unknown_classes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate instance-dependent close-set and open-set noisy CIFAR-100 labels"
    )
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ref_ckpt", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_unknown_classes", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--open_set_noise_ratio", type=float, default=0.2)
    parser.add_argument("--hardness_mode", type=str, default="all", choices=["all", "hard", "easy", "topk"])
    parser.add_argument("--hardness_threshold", type=float, default=0.3)
    parser.add_argument("--topk", type=int, default=5000)

    parser.add_argument("--close_set_noise_ratio", type=float, default=0.0)
    parser.add_argument(
        "--close_set_mode",
        type=str,
        default="topk_hard",
        choices=["random", "hard", "topk_hard"],
        help="How to choose known samples for close-set corruption.",
    )
    parser.add_argument(
        "--close_set_topk",
        type=int,
        default=8000,
        help="Hard candidate pool size for --close_set_mode topk_hard.",
    )
    parser.add_argument(
        "--close_set_hardness_metric",
        type=str,
        default="margin",
        choices=["margin", "true_prob"],
        help="Difficulty metric: smaller margin or smaller true-class probability is harder.",
    )

    parser.add_argument(
        "--ratio_mode",
        type=str,
        default="fraction_total",
        choices=["fraction_total", "relative_clean"],
        help="fraction_total: M_open/(N_known+M_open); relative_clean: M_open/N_known",
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


def infer_num_ref_classes(ckpt: dict) -> int:
    if "num_ref_classes" in ckpt:
        return int(ckpt["num_ref_classes"])
    state = ckpt["model_state"]
    for key in ("backbone.fc.weight", "fc.weight"):
        if key in state:
            return int(state[key].shape[0])
    if "num_known_classes" in ckpt:
        return int(ckpt["num_known_classes"])
    raise KeyError("Could not infer reference classifier output dimension from checkpoint")


def load_split_from_checkpoint(ckpt: dict, args: argparse.Namespace):
    known_classes = ckpt.get("known_classes")
    unknown_classes = ckpt.get("unknown_classes")
    raw_remap = ckpt.get("known_remap")

    if known_classes is None or unknown_classes is None or raw_remap is None:
        known_classes, unknown_classes, known_remap = split_known_unknown_classes(
            num_total_classes=100,
            num_unknown_classes=args.num_unknown_classes,
            seed=args.seed,
            fixed_known_classes=None,
        )
    else:
        known_classes = [int(c) for c in known_classes]
        unknown_classes = [int(c) for c in unknown_classes]
        known_remap = {int(k): int(v) for k, v in raw_remap.items()}

    return known_classes, unknown_classes, known_remap


def known_space_probs(
    logits: torch.Tensor,
    known_class_index: torch.Tensor,
    num_ref_classes: int,
    num_known_classes: int,
    temperature: float,
) -> torch.Tensor:
    """Return probabilities over remapped known labels [0, K-1]."""
    if num_ref_classes == num_known_classes:
        return torch.softmax(logits / temperature, dim=1)
    if num_ref_classes != 100:
        raise ValueError(
            f"Expected reference classifier to have either {num_known_classes} known-only outputs "
            f"or 100 full-CIFAR outputs, got {num_ref_classes}"
        )

    full_probs = torch.softmax(logits / temperature, dim=1)
    probs = full_probs.index_select(dim=1, index=known_class_index)
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)


def known_metrics(probs: torch.Tensor, true_labels: torch.Tensor | None = None):
    max_probs, _ = probs.max(dim=1)
    ents = entropy_of_probs(probs)
    if true_labels is None:
        return max_probs, ents, None, None

    batch_idx = torch.arange(probs.shape[0], device=probs.device)
    true_probs = probs[batch_idx, true_labels]
    competitors = probs.clone()
    competitors[batch_idx, true_labels] = -1.0
    strongest_competitor, _ = competitors.max(dim=1)
    margins = true_probs - strongest_competitor
    return max_probs, ents, true_probs, margins


def select_close_set_assignments(
    candidates: List[dict],
    ratio: float,
    mode: str,
    topk: int,
    hardness_metric: str,
) -> List[dict]:
    if not (0.0 <= ratio <= 1.0):
        raise ValueError("close_set_noise_ratio must be in [0, 1]")

    target = min(int(round(ratio * len(candidates))), len(candidates))
    if target == 0:
        return []

    if mode == "random":
        perm = torch.randperm(len(candidates)).tolist()
        return [candidates[i] for i in perm[:target]]

    metric_key = "margin" if hardness_metric == "margin" else "true_class_prob"
    hardest = sorted(candidates, key=lambda x: x[metric_key])
    if mode == "hard":
        return hardest[:target]
    if mode == "topk_hard":
        pool_size = min(max(int(topk), target), len(hardest))
        pool = hardest[:pool_size]
        return pool[:target]
    raise ValueError(f"Unsupported close_set_mode: {mode}")


def generate_close_set_candidates(
    model: torch.nn.Module,
    known_loader: DataLoader,
    known_class_index: torch.Tensor,
    num_ref_classes: int,
    num_known_classes: int,
    temperature: float,
    device: torch.device,
) -> List[dict]:
    candidates: List[dict] = []

    with torch.no_grad():
        for batch in known_loader:
            images = batch["image"].to(device)
            original_labels = batch["original_label"]
            remapped_labels = batch["label"].to(device)
            sample_indices = batch["index"]

            logits = model(images)
            probs = known_space_probs(
                logits=logits,
                known_class_index=known_class_index,
                num_ref_classes=num_ref_classes,
                num_known_classes=num_known_classes,
                temperature=temperature,
            )
            max_probs, ents, true_probs, margins = known_metrics(probs, true_labels=remapped_labels)

            masked_probs = probs.clone()
            masked_probs[torch.arange(images.shape[0], device=device), remapped_labels] = 0.0
            masked_sum = masked_probs.sum(dim=1, keepdim=True)

            fallback = torch.ones_like(masked_probs)
            fallback[torch.arange(images.shape[0], device=device), remapped_labels] = 0.0
            fallback = fallback / fallback.sum(dim=1, keepdim=True).clamp_min(1e-12)
            relabel_probs = torch.where(
                masked_sum > 0, masked_probs / masked_sum.clamp_min(1e-12), fallback
            )

            sampled_labels = torch.multinomial(relabel_probs, num_samples=1).squeeze(1)

            probs_cpu = probs.cpu()
            relabel_cpu = relabel_probs.cpu()
            sampled_cpu = sampled_labels.cpu()
            for i in range(images.shape[0]):
                original_known_label = int(remapped_labels[i].item())
                noisy_label = int(sampled_cpu[i].item())
                candidates.append(
                    {
                        "image": batch["image"][i],
                        "label": noisy_label,
                        "original_label": int(original_labels[i].item()),
                        "original_known_label": original_known_label,
                        "source_index": int(sample_indices[i].item()),
                        "is_close_set_noise": True,
                        "is_closed_set_noise": True,
                        "true_class_prob": float(true_probs[i].item()),
                        "margin": float(margins[i].item()),
                        "max_prob": float(max_probs[i].item()),
                        "entropy": float(ents[i].item()),
                        "known_prob_vector": probs_cpu[i].tolist(),
                        "masked_known_prob_vector": relabel_cpu[i].tolist(),
                    }
                )

    return candidates


def generate_open_set_assignments(
    model: torch.nn.Module,
    unknown_loader: DataLoader,
    known_class_index: torch.Tensor,
    num_ref_classes: int,
    num_known_classes: int,
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
            probs = known_space_probs(
                logits=logits,
                known_class_index=known_class_index,
                num_ref_classes=num_ref_classes,
                num_known_classes=num_known_classes,
                temperature=temperature,
            )
            max_probs, ents, _, _ = known_metrics(probs)
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


def validate_final_labels(samples: List[dict], num_known_classes: int) -> None:
    bad = [s for s in samples if int(s["label"]) < 0 or int(s["label"]) >= num_known_classes]
    if bad:
        raise ValueError(f"Found {len(bad)} final labels outside [0, {num_known_classes - 1}]")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    ckpt = torch.load(args.ref_ckpt, map_location="cpu")
    known_classes, unknown_classes, known_remap = load_split_from_checkpoint(ckpt, args)
    num_known_classes = len(known_classes)
    num_ref_classes = infer_num_ref_classes(ckpt)

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
    known_class_index = torch.tensor(known_classes, dtype=torch.long, device=device)
    model = CIFARResNet18(num_classes=num_ref_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    close_candidates = generate_close_set_candidates(
        model=model,
        known_loader=known_loader,
        known_class_index=known_class_index,
        num_ref_classes=num_ref_classes,
        num_known_classes=num_known_classes,
        temperature=args.temperature,
        device=device,
    )
    selected_close = select_close_set_assignments(
        candidates=close_candidates,
        ratio=args.close_set_noise_ratio,
        mode=args.close_set_mode,
        topk=args.close_set_topk,
        hardness_metric=args.close_set_hardness_metric,
    )

    all_unknown_assignments = generate_open_set_assignments(
        model=model,
        unknown_loader=unknown_loader,
        known_class_index=known_class_index,
        num_ref_classes=num_ref_classes,
        num_known_classes=num_known_classes,
        temperature=args.temperature,
        device=device,
    )
    selected_unknown_pool = filter_unknown(
        assignments=all_unknown_assignments,
        mode=args.hardness_mode,
        threshold=args.hardness_threshold,
        topk=args.topk,
    )

    final_ds = build_final_clean_close_open_dataset(
        known_dataset=known_ds,
        close_set_assignments=selected_close,
        unknown_assignments=selected_unknown_pool,
        open_set_noise_ratio=args.open_set_noise_ratio,
        ratio_mode=args.ratio_mode,
    )
    validate_final_labels(final_ds.samples, num_known_classes)

    n_known = len(known_ds)
    n_unknown_pool = len(unknown_ds)
    n_close = sum(1 for s in final_ds.samples if s["is_close_set_noise"])
    n_open = sum(1 for s in final_ds.samples if s["is_open_set_noise"])
    n_clean = sum(1 for s in final_ds.samples if s["noise_type"] == "clean")
    n_final = len(final_ds)
    close_ratio = n_close / max(n_known, 1)
    open_ratio = n_open / max(n_final, 1)

    close_hist = Counter(s["label"] for s in selected_close)
    open_hist = Counter(s["label"] for s in final_ds.samples if s["is_open_set_noise"])
    avg_close_true_prob = sum(s["true_class_prob"] for s in selected_close) / max(len(selected_close), 1)
    avg_close_margin = sum(s["margin"] for s in selected_close) / max(len(selected_close), 1)
    avg_open_max_prob = sum(a["max_prob"] for a in selected_unknown_pool) / max(len(selected_unknown_pool), 1)
    avg_open_entropy = sum(a["entropy"] for a in selected_unknown_pool) / max(len(selected_unknown_pool), 1)

    per_unknown_summary = defaultdict(lambda: Counter())
    for a in selected_unknown_pool:
        per_unknown_summary[a["original_unknown_class"]][a["noisy_label"]] += 1

    print("==== Close/Open Noise Stats ====")
    print(f"Reference output classes: {num_ref_classes}")
    print(f"Known classes ({len(known_classes)}): {known_classes}")
    print(f"Unknown classes ({len(unknown_classes)}): {unknown_classes}")
    print(f"Known samples: {n_known}")
    print(f"Clean known samples kept: {n_clean}")
    print(f"Close-set noisy known samples: {n_close} (ratio over known: {close_ratio:.6f})")
    print(f"Unknown samples (pool): {n_unknown_pool}")
    print(f"Selected unknown pool after hardness filter: {len(selected_unknown_pool)}")
    print(f"Open-set noisy unknown samples used: {n_open} (ratio in final: {open_ratio:.6f})")
    print(f"Final dataset size: {n_final}")
    print(f"Average selected close-set true prob: {avg_close_true_prob:.6f}")
    print(f"Average selected close-set margin: {avg_close_margin:.6f}")
    print(f"Average selected open-set max probability: {avg_open_max_prob:.6f}")
    print(f"Average selected open-set entropy: {avg_open_entropy:.6f}")

    save_json(
        os.path.join(args.output_dir, "split_info.json"),
        {
            "known_classes": known_classes,
            "unknown_classes": unknown_classes,
            "known_remap": known_remap,
            "num_known_classes": num_known_classes,
            "num_unknown_classes": len(unknown_classes),
            "num_ref_classes": num_ref_classes,
            "ref_train_full_dataset": bool(ckpt.get("ref_train_full_dataset", num_ref_classes == 100)),
        },
    )

    stats = {
        "num_known_samples": n_known,
        "num_known_clean_samples": n_clean,
        "num_close_set_noisy_samples": n_close,
        "empirical_close_set_noise_ratio": close_ratio,
        "num_unknown_pool_samples": n_unknown_pool,
        "num_selected_unknown_pool_samples": len(selected_unknown_pool),
        "num_open_set_noisy_samples_used": n_open,
        "num_final_dataset_samples": n_final,
        "final_open_set_noise_ratio": open_ratio,
        "final_proportions": {
            "clean": n_clean / max(n_final, 1),
            "close": n_close / max(n_final, 1),
            "open": n_open / max(n_final, 1),
        },
        "temperature": args.temperature,
        "requested_close_set_noise_ratio": args.close_set_noise_ratio,
        "close_set_mode": args.close_set_mode,
        "close_set_topk": args.close_set_topk,
        "close_set_hardness_metric": args.close_set_hardness_metric,
        "requested_open_set_noise_ratio": args.open_set_noise_ratio,
        "ratio_mode": args.ratio_mode,
        "hardness_mode": args.hardness_mode,
        "hardness_threshold": args.hardness_threshold,
        "topk": args.topk,
        "avg_selected_close_true_class_prob": avg_close_true_prob,
        "avg_selected_close_margin": avg_close_margin,
        "avg_selected_open_max_prob": avg_open_max_prob,
        "avg_selected_open_entropy": avg_open_entropy,
        "close_relabel_to_class_histogram": {str(k): int(v) for k, v in sorted(close_hist.items())},
        "open_assigned_label_histogram": {str(k): int(v) for k, v in sorted(open_hist.items())},
    }
    save_json(os.path.join(args.output_dir, "noise_stats.json"), stats)
    save_json(os.path.join(args.output_dir, "mixed_noise_stats.json"), stats)

    close_csv = os.path.join(args.output_dir, "close_set_assignments.csv")
    with open(close_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source_index",
                "original_label",
                "original_known_label",
                "new_training_label",
                "true_class_prob",
                "margin",
                "max_prob",
                "entropy",
                "known_prob_vector_json",
                "masked_known_prob_vector_json",
            ]
        )
        for s in selected_close:
            writer.writerow(
                [
                    s["source_index"],
                    s["original_label"],
                    s["original_known_label"],
                    s["label"],
                    f"{s['true_class_prob']:.8f}",
                    f"{s['margin']:.8f}",
                    f"{s['max_prob']:.8f}",
                    f"{s['entropy']:.8f}",
                    json.dumps(s["known_prob_vector"]),
                    json.dumps(s["masked_known_prob_vector"]),
                ]
            )

    # Legacy alias for scripts that still expect the old closed-set filename.
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
                "true_class_prob",
                "margin",
                "max_prob",
                "entropy",
                "masked_prob_vector_json",
            ]
        )
        for s in selected_close:
            writer.writerow(
                [
                    s["source_index"],
                    s["original_label"],
                    s["original_known_label"],
                    s["label"],
                    1,
                    f"{s['true_class_prob']:.8f}",
                    f"{s['margin']:.8f}",
                    f"{s['max_prob']:.8f}",
                    f"{s['entropy']:.8f}",
                    json.dumps(s["masked_known_prob_vector"]),
                ]
            )

    unknown_csv = os.path.join(args.output_dir, "unknown_noisy_assignments.csv")
    with open(unknown_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "original_unknown_class", "noisy_label", "max_prob", "entropy", "prob_vector_json"])
        for a in selected_unknown_pool:
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
                "original_label",
                "noise_type",
                "is_close_set_noise",
                "is_open_set_noise",
                "true_class_prob",
                "margin",
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
                    int(s["is_close_set_noise"]),
                    int(s["is_open_set_noise"]),
                    "" if s.get("true_class_prob") is None else f"{s['true_class_prob']:.8f}",
                    "" if s.get("margin") is None else f"{s['margin']:.8f}",
                    "" if s.get("max_prob") is None else f"{s['max_prob']:.8f}",
                    "" if s.get("entropy") is None else f"{s['entropy']:.8f}",
                ]
            )

    torch.save(selected_close, os.path.join(args.output_dir, "close_set_samples.pt"))
    torch.save(selected_close, os.path.join(args.output_dir, "closed_set_samples.pt"))
    tensor_path = os.path.join(args.output_dir, "final_dataset_samples.pt")
    torch.save(final_ds.samples, tensor_path)

    print(f"Saved split info: {os.path.join(args.output_dir, 'split_info.json')}")
    print(f"Saved stats: {os.path.join(args.output_dir, 'noise_stats.json')}")
    print(f"Saved close-set assignments: {close_csv}")
    print(f"Saved open-set assignments: {unknown_csv}")
    print(f"Saved per-unknown summary: {per_unknown_csv}")
    print(f"Saved final dataset metadata: {final_meta_csv}")
    print(f"Saved final dataset tensor list: {tensor_path}")


if __name__ == "__main__":
    main()
