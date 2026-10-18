"""
ELEC 475: Lab 2 - Build plots/ablation_summary.png from the post-lab training logs.
Parses the per-epoch Val Acc column of each run in logs/postlab/ and feeds it to
utils.visualization.plot_ablation_summary.

Usage:
    python plot_ablation.py                          # CIFAR-10
    python plot_ablation.py --dataset tiny_imagenet  # Tiny ImageNet
"""
import argparse
import os
import re

from utils.visualization import plot_ablation_summary

RUNS = {
    "1. Base LeNet-5": "lenet_baseline",
    "2. Modernized LeNet": "lenet_modern",
    "3. AlexNet (No Norm)": "alexnet_none",
    "4. AlexNet + LRN": "alexnet_lrn",
    "5. AlexNet + BatchNorm": "alexnet_bn",
    "6. AlexNet + BN + Aug": "alexnet_bn_aug",
    "7. PyTorch Zoo AlexNet": "zoo_alexnet",
}

# Matches rows of train.py's epoch table: " 3 | loss | acc | loss | val acc | ..."
EPOCH_ROW = re.compile(r"^\s*\d+\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*([\d.]+)\s*\|")


def read_val_accs(log_path: str):
    with open(log_path) as f:
        return [float(m.group(1)) for line in f if (m := EPOCH_ROW.match(line))]


def main():
    parser = argparse.ArgumentParser(description="Plot master ablation validation accuracy curves")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "tiny_imagenet"])
    parser.add_argument("--log-dir", type=str, default="logs/postlab")
    args = parser.parse_args()

    prefix = "tiny_" if args.dataset == "tiny_imagenet" else ""
    results = {}
    for label, name in RUNS.items():
        path = os.path.join(args.log_dir, f"{prefix}{name}.log")
        if not os.path.exists(path):
            print(f"Skipping {label}: {path} not found")
            continue
        results[label] = {"val_accs": read_val_accs(path)}

    suffix = "_tiny_imagenet" if prefix else ""
    plot_ablation_summary(results, save_path=f"plots/ablation_summary{suffix}.png")


if __name__ == "__main__":
    main()
