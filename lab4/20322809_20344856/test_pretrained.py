"""
ELEC 475: Lab 4 - Pre-Lab Step 1: Pretrained FCN-ResNet50 Baseline
Loads torchvision's pretrained FCN-ResNet50, downloads the PASCAL VOC 2012
segmentation dataset and evaluates the model on it using mean IoU (mIoU).

Run from the lab4 directory:
  python test_pretrained.py --limit 20     (quick check on 20 images)
  python test_pretrained.py                (full validation set)
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from tqdm import tqdm

NUM_CLASSES = 21  # 20 VOC classes + background
IGNORE_INDEX = 255  # VOC marks object boundaries with 255: excluded from the metric
VOC_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
               "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]


def get_device() -> torch.device:
    # MPS is the Apple Silicon GPU backend (e.g. M4)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mask_to_tensor(mask) -> torch.Tensor:
    return torch.as_tensor(np.array(mask), dtype=torch.int64)  # palette indices: 0-20, 255


def unbatch(batch):
    # Module-level (not a lambda) so macOS's spawned workers can pickle it
    return batch[0]


def load_dataset(data_dir: str, image_set: str, preprocess=None) -> VOCSegmentation:
    # download=True re-extracts the whole archive on every run, so only use it when the data is missing
    download = not os.path.isdir(os.path.join(data_dir, "VOCdevkit", "VOC2012"))
    return VOCSegmentation(root=data_dir, year="2012", image_set=image_set, download=download,
                           transform=preprocess, target_transform=mask_to_tensor)


def update_confusion(confusion: torch.Tensor, pred: torch.Tensor, target: torch.Tensor) -> None:
    """
    Adds one image to the confusion matrix, skipping IGNORE_INDEX pixels.

    Args:
        confusion : torch.Tensor, shape (C, C), int64. Rows are ground truth, columns are predictions
        pred      : torch.Tensor, shape (H, W), int64. Predicted class of each pixel
        target    : torch.Tensor, shape (H, W), int64. Ground-truth class of each pixel
    """
    valid = target != IGNORE_INDEX
    pairs = target[valid] * NUM_CLASSES + pred[valid]
    confusion += torch.bincount(pairs, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def iou_from_confusion(confusion: torch.Tensor) -> torch.Tensor:
    """
    Computes per-class IoU = TP / (TP + FP + FN) from the dataset-wide confusion matrix.

    Returns:
        torch.Tensor, shape (C,), float64. NaN for a class absent from both predictions and labels
    """
    confusion = confusion.double()
    tp = confusion.diag()
    union = confusion.sum(dim=0) + confusion.sum(dim=1) - tp
    return tp / union


@torch.inference_mode()
def evaluate(model, dataset, device, limit=None, num_workers=4) -> torch.Tensor:
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))

    # Batch size 1: VOC images vary in size. Workers decode and preprocess images in parallel with the GPU
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                        collate_fn=unbatch)
    for image, target in tqdm(loader, desc="Evaluating"):
        logits = model(image.unsqueeze(0).to(device))["out"]
        # Score at the mask's original resolution, so the labels are never resampled
        logits = F.interpolate(logits, size=target.shape, mode="bilinear", align_corners=False)
        update_confusion(confusion, logits.argmax(dim=1)[0].cpu(), target)

    return confusion


def print_results(iou: torch.Tensor, confusion: torch.Tensor) -> None:
    print("-" * 40)
    print(f"{'Class':<16}{'IoU':>10}")
    print("-" * 40)
    for name, value in zip(VOC_CLASSES, iou.tolist()):
        print(f"{name:<16}{value * 100:>9.2f}%")
    print("-" * 40)
    print(f"{'mIoU':<16}{iou.nanmean().item() * 100:>9.2f}%")
    pixel_acc = confusion.diag().sum().item() / confusion.sum().item()
    print(f"{'Pixel accuracy':<16}{pixel_acc * 100:>9.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained FCN-ResNet50 on PASCAL VOC 2012.")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--image_set", type=str, default="val", choices=["train", "val", "trainval"])
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N images")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--output", type=str, default="results/baseline_fcn_resnet50.json")
    args = parser.parse_args()

    device = get_device()
    weights = FCN_ResNet50_Weights.DEFAULT

    print("=" * 72)
    print("ELEC 475: Lab 4 Pretrained FCN-ResNet50 Baseline")
    print(f"Weights : {weights}")
    print(f"Device  : {device}")
    print("=" * 72)

    # 1. Load the Pretrained Model
    model = fcn_resnet50(weights=weights).to(device).eval()
    preprocess = weights.transforms()

    # 2. Load the Dataset
    dataset = load_dataset(args.data_dir, args.image_set, preprocess)
    print(f"VOC 2012 {args.image_set}: {len(dataset)} images")

    # 3. Evaluate
    confusion = evaluate(model, dataset, device, args.limit, args.num_workers)
    iou = iou_from_confusion(confusion)
    print_results(iou, confusion)

    # 4. Save the Baseline (full runs only, so a quick check never overwrites it)
    if args.limit is None:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"weights": str(weights), "image_set": args.image_set, "n_images": len(dataset),
                       "miou": iou.nanmean().item(),
                       "pixel_acc": confusion.diag().sum().item() / confusion.sum().item(),
                       "class_iou": dict(zip(VOC_CLASSES, iou.tolist()))}, f, indent=2)
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
