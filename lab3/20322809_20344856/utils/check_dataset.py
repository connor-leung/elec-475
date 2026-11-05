"""
ELEC 475: Lab 3 - Dataset Reality Check (Pre-Lab Task 3, Step 2)
    1. Iterates every sample of both splits and confirms image shape and label bounds
    2. Plots eight training images with their nose labels to plots/dataset_check.png

Run from the lab folder:
    python -m utils.check_dataset
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from utils.dataset import PetNoseDataset, denormalize, IMG_SIZE


def check_split(split: str) -> None:
    ds = PetNoseDataset(split=split)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    n_checked, n_bad_shape, n_out_of_frame = 0, 0, 0
    for images, labels, _ in loader:
        if images.shape[1:] != (3, IMG_SIZE, IMG_SIZE):
            n_bad_shape += images.shape[0]
        inside = (labels >= 0).all(dim=1) & (labels <= IMG_SIZE).all(dim=1)
        n_out_of_frame += int((~inside).sum())
        n_checked += images.shape[0]
    status = "OK" if (n_bad_shape == 0 and n_out_of_frame == 0) else "FAIL"
    print(f"  {split:5s}: {n_checked:,} samples | bad shape {n_bad_shape} "
          f"| label out of frame {n_out_of_frame} --> {status}")


def plot_samples(out_path: str = "plots/dataset_check.png", n: int = 8) -> None:
    ds = PetNoseDataset(split="train")
    fig, axes = plt.subplots(2, n // 2, figsize=(3 * n // 2, 6))
    for i, ax in enumerate(axes.flat):
        image, label, orig_size = ds[i]
        ax.imshow(denormalize(image).permute(1, 2, 0).numpy())
        ax.scatter(label[0].item(), label[1].item(), s=40, c="red", edgecolors="white")
        ax.set_title(f"{ds.samples[i][0]}\n{int(orig_size[0])}x{int(orig_size[1])}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    print("Checking every sample of both splits:")
    for split in ["train", "test"]:
        check_split(split)
    plot_samples()
