"""Train a clean CIFAR-100 reference model for instance-dependent noise generation.

By default the reference classifier is trained on the full clean CIFAR-100 training
set with all 100 classes. A known/unknown split is still saved so later scripts can
restrict generated noisy labels to the 80-class known label space. For legacy
experiments, pass ``--no-ref_train_full_dataset`` to train the reference model on
known classes only.
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100

from datasets import CIFAR100SubsetByClass, CIFAR100WithIndex
from model import CIFARResNet18
from utils import ensure_dir, load_known_classes, save_json, set_seed, split_known_unknown_classes


BooleanOptionalAction = getattr(argparse, "BooleanOptionalAction", None)

if BooleanOptionalAction is None:
    class BooleanOptionalAction(argparse.Action):
        """Backport argparse.BooleanOptionalAction for Python < 3.9."""

        def __init__(
            self,
            option_strings,
            dest,
            default=None,
            type=None,
            choices=None,
            required=False,
            help=None,
            metavar=None,
        ):
            boolean_option_strings = []
            for option_string in option_strings:
                boolean_option_strings.append(option_string)
                if option_string.startswith("--"):
                    boolean_option_strings.append("--no-" + option_string[2:])

            super().__init__(
                option_strings=boolean_option_strings,
                dest=dest,
                nargs=0,
                default=default,
                type=type,
                choices=choices,
                required=required,
                help=help,
                metavar=metavar,
            )

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, not (option_string or "").startswith("--no-"))

        def format_usage(self):
            return " | ".join(self.option_strings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train clean CIFAR-100 reference model")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--known_classes_file", type=str, default=None)
    parser.add_argument("--num_unknown_classes", type=int, default=20)
    parser.add_argument(
        "--ref_train_full_dataset",
        action=BooleanOptionalAction,
        default=True,
        help=(
            "Train the reference classifier on all 100 CIFAR-100 classes. "
            "Use --no-ref_train_full_dataset for legacy known-only training."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def make_loaders(
    args: argparse.Namespace,
    known_classes,
    known_remap,
) -> Tuple[DataLoader, DataLoader, int]:
    train_tfm = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    test_tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    train_base = CIFAR100(root=args.data_root, train=True, download=True, transform=train_tfm)
    test_base = CIFAR100(root=args.data_root, train=False, download=True, transform=test_tfm)

    if args.ref_train_full_dataset:
        train_ds = CIFAR100WithIndex(train_base)
        test_ds = CIFAR100WithIndex(test_base)
        num_ref_classes = 100
    else:
        train_ds = CIFAR100SubsetByClass(train_base, known_classes, label_remap=known_remap)
        test_ds = CIFAR100SubsetByClass(test_base, known_classes, label_remap=known_remap)
        num_ref_classes = len(known_classes)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader, num_ref_classes


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            pred = torch.argmax(logits, dim=1)
            total_correct += (pred == y).sum().item()
            total += x.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    fixed_known = load_known_classes(args.known_classes_file) if args.known_classes_file else None
    known_classes, unknown_classes, known_remap = split_known_unknown_classes(
        num_total_classes=100,
        num_unknown_classes=args.num_unknown_classes,
        seed=args.seed,
        fixed_known_classes=fixed_known,
    )

    split_payload = {
        "known_classes": known_classes,
        "unknown_classes": unknown_classes,
        "known_remap": known_remap,
        "seed": args.seed,
        "num_unknown_classes": args.num_unknown_classes,
        "num_total_classes": 100,
        "num_known_classes": len(known_classes),
        "ref_train_full_dataset": bool(args.ref_train_full_dataset),
        "num_ref_classes": 100 if args.ref_train_full_dataset else len(known_classes),
    }
    save_json(os.path.join(args.output_dir, "split_info.json"), split_payload)

    train_loader, test_loader, num_ref_classes = make_loaders(args, known_classes, known_remap)
    print(f"Known classes: {len(known_classes)}, unknown classes: {len(unknown_classes)}")
    print(f"Reference training mode: {'full CIFAR-100' if args.ref_train_full_dataset else 'known-only'}")
    print(f"Reference classifier output classes: {num_ref_classes}")
    print(f"Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")

    device = torch.device(args.device)
    model = CIFARResNet18(num_classes=num_ref_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0

        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            pred = torch.argmax(logits, dim=1)
            total_correct += (pred == y).sum().item()
            total += x.size(0)

        scheduler.step()

        train_loss = total_loss / max(total, 1)
        train_acc = total_correct / max(total, 1)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join(args.output_dir, "reference_model_best.pth")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_acc": best_acc,
                    "known_classes": known_classes,
                    "unknown_classes": unknown_classes,
                    "known_remap": known_remap,
                    "num_known_classes": len(known_classes),
                    "num_unknown_classes": args.num_unknown_classes,
                    "num_total_classes": 100,
                    "num_ref_classes": num_ref_classes,
                    "ref_train_full_dataset": bool(args.ref_train_full_dataset),
                    "seed": args.seed,
                },
                ckpt_path,
            )
            print(f"Saved best checkpoint to {ckpt_path} (val_acc={best_acc:.4f})")

    final_path = os.path.join(args.output_dir, "reference_model_last.pth")
    torch.save(
        {
            "model_state": model.state_dict(),
            "epoch": args.epochs,
            "best_acc": best_acc,
            "known_classes": known_classes,
            "unknown_classes": unknown_classes,
            "known_remap": known_remap,
            "num_known_classes": len(known_classes),
            "num_unknown_classes": args.num_unknown_classes,
            "num_total_classes": 100,
            "num_ref_classes": num_ref_classes,
            "ref_train_full_dataset": bool(args.ref_train_full_dataset),
            "seed": args.seed,
        },
        final_path,
    )
    print(f"Saved last checkpoint to {final_path}")


if __name__ == "__main__":
    main()
