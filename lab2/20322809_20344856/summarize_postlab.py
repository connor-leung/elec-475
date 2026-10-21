"""
ELEC 475: Lab 2 - Print the multi-seed (Mean ± Std) and extended ablation (Δ) tables
from the post-lab training logs written by run_postlab.sh.

Usage:
    python summarize_postlab.py
"""
import os
import re

from utils.seed import format_mean_std

LOG_DIR = "logs/postlab"

# Rows of train.py's epoch table: epoch | train loss | train acc | val loss | val acc | val f1 | time | ...
EPOCH_ROW = re.compile(r"^\s*(\d+)\s*\|" + r"\s*([\d.]+)\s*\|" * 6)


def read_epochs(name: str):
    path = os.path.join(LOG_DIR, f"{name}.log")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for line in f:
            m = EPOCH_ROW.match(line)
            if m:
                ep, tr_loss, tr_acc, va_loss, va_acc, va_f1, t = m.groups()
                rows.append({"epoch": int(ep), "train_acc": float(tr_acc), "val_acc": float(va_acc),
                             "val_f1": float(va_f1), "val_loss": float(va_loss), "time": float(t)})
    return rows or None


def best(rows):
    return max(rows, key=lambda r: r["val_acc"])


def print_table(headers, rows):
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    print("| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths)) + " |")
    print("|-" + "-|-".join("-" * w for w in widths) + "-|")
    for r in rows:
        print("| " + " | ".join(f"{str(x):<{w}}" for x, w in zip(r, widths)) + " |")


def multi_seed():
    print("\nMulti-seed normalization study (CIFAR-10, 10 epochs, seeds 42/0/1, best epoch per run)\n")
    table = []
    for norm, label in [("none", "AlexNet (No Norm)"), ("lrn", "AlexNet + LRN"), ("bn", "AlexNet + BatchNorm")]:
        runs = [read_epochs(n) for n in (f"alexnet_{norm}", f"alexnet_{norm}_s0", f"alexnet_{norm}_s1")]
        runs = [best(r) for r in runs if r]
        table.append([label, len(runs),
                      format_mean_std([r["val_acc"] for r in runs]),
                      format_mean_std([r["val_f1"] for r in runs]),
                      format_mean_std([r["train_acc"] - r["val_acc"] for r in runs], unit="")])
    print_table(["Model", "Seeds", "Val Acc", "Val Macro F1", "Δ (pts)"], table)


def extended():
    print("\nExtended ablation (CIFAR-10, AlexNet + BN; Δ = train acc - val acc at the best epoch)\n")
    configs = [
        ("alexnet_bn", "Baseline (lr 1e-3, bs 128)"),
        ("alexnet_bn_lr1e-2", "lr 1e-2"),
        ("alexnet_bn_lr1e-4", "lr 1e-4"),
        ("alexnet_bn_bs64", "batch 64"),
        ("alexnet_bn_bs256", "batch 256"),
        ("alexnet_bn_aug_es", "aug + early stop (≤30 ep)"),
    ]
    table = []
    for name, label in configs:
        rows = read_epochs(name)
        if not rows:
            table.append([label, "not run", "", "", "", "", ""])
            continue
        b = best(rows)
        mean_t = sum(r["time"] for r in rows) / len(rows)
        table.append([label, f"{len(rows)} (best {b['epoch']})", f"{b['train_acc']:.2f}", f"{b['val_acc']:.2f}",
                      f"{b['val_f1']:.2f}", f"{b['train_acc'] - b['val_acc']:+.2f}", f"{mean_t:.2f}"])
    print_table(["Config", "Epochs", "Train Acc", "Val Acc", "Val F1", "Δ", "Epoch (s)"], table)


if __name__ == "__main__":
    multi_seed()
    extended()
