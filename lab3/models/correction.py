"""
ELEC 475: Lab 3 - Correction Model (Starter Template)
A small CNN that predicts the offset from a patch centre to the true nose.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrectionNet(nn.Module):
    """
    CorrectionNet for (3, 64, 64) ImageNet-normalized patches.

    Spatial dimension arithmetic (each block halves the size with 2x2 max-pooling):
        Input        : (B, 3, 64, 64)
        Block 1      : (B, 32, 32, 32)
        Block 2      : (B, 64, 16, 16)
        Block 3      : (B, 128, 8, 8)
        Block 4      : (B, 128, 4, 4)
        Flatten      : 128 * 4 * 4 = 2,048 features
        FC1          : 2,048 -> 256
        FC2          : 256 -> 2, the normalized offset (du, dv) / HALF_PATCH
    """
    def __init__(self):
        super(CorrectionNet, self).__init__()

        # ======================================================================
        # TODO [POST-LAB STEP 2]: Implement the CorrectionNet Layers
        #
        # Requirements:
        #   1. Four conv blocks, with output channels 32, 64, 128 and 128.
        #      Each block: nn.Conv2d(kernel_size=3, padding=1) -> nn.BatchNorm2d
        #      -> ReLU -> nn.MaxPool2d(kernel_size=2, stride=2)
        #   2. fc1: in_features = 128 * 4 * 4 (2,048), out_features = 256
        #   3. fc2: in_features = 256, out_features = 2
        # ======================================================================
        self.block1 = self._block(3, 32)
        self.block2 = self._block(32, 64)
        self.block3 = self._block(64, 128)
        self.block4 = self._block(128, 128)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 2)

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ======================================================================
        # TODO [POST-LAB STEP 2]: Implement the Forward Pass
        #
        # 1. Apply the four conv blocks
        # 2. Flatten using torch.flatten(x, start_dim=1)
        # 3. Apply fc1 -> ReLU, then fc2 (no activation)
        # 4. Return the (B, 2) normalized offsets
        # ======================================================================
        x = self.block4(self.block3(self.block2(self.block1(x))))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class CorrectionNetModified(nn.Module):
    """
    Your modified CorrectionNet (Post-Lab Step 5). Same input (B, 3, 64, 64) and
    output (B, 2) normalized offsets as CorrectionNet.

    Changes from CorrectionNet:
      1. Each block has two 3x3 conv -> BN -> ReLU layers before pooling (VGG-style),
         which doubles the depth and widens the receptive field without losing resolution
         before the second conv. Block 4 widens to 256 channels.
      2. BatchNorm1d after fc1. Without it, an early gradient spike drives most fc1
         units negative for every input ("dying ReLU"), and the model collapses to
         predicting a zero offset. BatchNorm re-centres them on every batch.
      3. Dropout (p=0.3) before fc2 to reduce overfitting of the larger head.
    The flatten keeps the 4x4 spatial grid (no global pooling), since the
    position of the nose within the patch is exactly what must be regressed.

        Input        : (B, 3, 64, 64)
        Block 1      : (B, 32, 32, 32)
        Block 2      : (B, 64, 16, 16)
        Block 3      : (B, 128, 8, 8)
        Block 4      : (B, 256, 4, 4)
        Flatten      : 256 * 4 * 4 = 4,096 features
        FC1          : 4,096 -> 256
        FC2          : 256 -> 2
    """
    def __init__(self):
        super(CorrectionNetModified, self).__init__()
        # ======================================================================
        # TODO [POST-LAB STEP 5]: Implement Your Modified Correction Model
        # ======================================================================
        self.features = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 256),
        )
        self.head = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ======================================================================
        # TODO [POST-LAB STEP 5]: Implement the Forward Pass
        # ======================================================================
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.head(x)


if __name__ == "__main__":
    from torchinfo import summary
    for model in [CorrectionNet(), CorrectionNetModified()]:
        print(f"Testing {type(model).__name__} Architecture:")
        try:
            summary(model, input_size=(1, 3, 64, 64))
        except Exception as e:
            print(f"Model not fully implemented yet: {e}")
