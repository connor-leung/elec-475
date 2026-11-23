import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.datasets import VOCSegmentation

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class SegTransform:
    """Resizes an (image, mask) pair to a fixed size and converts both to tensors.
    With augment=True (training only) it first applies a random horizontal flip and a
    random scale (0.5x to 1.5x) followed by a random crop / pad back to the fixed size."""

    def __init__(self, size=320, augment=False, scale=(0.5, 1.5)):
        self.size = [size, size]
        self.augment = augment
        self.scale = scale

    def __call__(self, img, mask):
        if not self.augment:
            img = TF.resize(img, self.size, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)
        else:
            if random.random() < 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            size = self.size[0]
            s = round(size * random.uniform(*self.scale))
            img = TF.resize(img, [s, s], interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, [s, s], interpolation=TF.InterpolationMode.NEAREST)
            if s < size:
                # Pad to the fixed size: grey for the image, 255 (ignored by the loss) for the mask
                left, top = random.randint(0, size - s), random.randint(0, size - s)
                pad = [left, top, size - s - left, size - s - top]
                img = TF.pad(img, pad, fill=tuple(round(255 * m) for m in MEAN))
                mask = TF.pad(mask, pad, fill=255)
            elif s > size:
                left, top = random.randint(0, s - size), random.randint(0, s - size)
                img = TF.crop(img, top, left, size, size)
                mask = TF.crop(mask, top, left, size, size)
        img = TF.normalize(TF.to_tensor(img), MEAN, STD)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return img, mask


def get_voc(data_dir, image_set, size=320, augment=False):
    return VOCSegmentation(data_dir, year='2012', image_set=image_set, download=False,
                           transforms=SegTransform(size, augment))
