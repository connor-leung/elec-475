import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def conv_bn_relu(in_ch, out_ch, k=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True))


def sep_conv(in_ch, out_ch, dilation=1):
    """Depthwise-separable 3x3 convolution: depthwise 3x3 (optionally atrous) -> BN -> ReLU,
    then pointwise 1x1 -> BN -> ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, 3, padding=dilation, dilation=dilation, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True))


class LiteASPP(nn.Module):
    """ASPP built from depthwise-separable atrous convolutions, plus an image-pooling branch."""

    def __init__(self, in_ch, out_ch=96, rates=(3, 6, 9)):
        super().__init__()
        self.branches = nn.ModuleList(
            [conv_bn_relu(in_ch, out_ch, 1)] +
            [sep_conv(in_ch, out_ch, dilation=r) for r in rates])
        # No BatchNorm here: the pooled map is 1x1, which BN cannot normalize for a batch of one
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, out_ch, 1),
                                  nn.ReLU(inplace=True))
        self.project = conv_bn_relu(out_ch * (2 + len(rates)), out_ch, 1)

    def forward(self, x):
        pooled = self.pool(x).expand(-1, -1, *x.shape[-2:])
        return self.project(torch.cat([b(x) for b in self.branches] + [pooled], dim=1))


class CompactSegNet(nn.Module):
    """Custom student: dilated MobileNetV3-Small encoder (output stride 16), a depthwise-separable
    ASPP, and a two-stage decoder with skip connections at stride 8 and stride 4."""

    feat_channels = {'low': 24, 'mid': 48, 'high': 576}

    def __init__(self, num_classes=21, pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        # dilated=True replaces the last stride-2 stage by dilation 2, so the encoder stops at stride 16
        features = mobilenet_v3_small(weights=weights, dilated=True).features
        self.stem = features[:2]    # 16 channels, stride 4
        self.low = features[2:4]    # 24 channels, stride 8
        self.mid = features[4:9]    # 48 channels, stride 16
        self.high = features[9:]    # 576 channels, stride 16 (dilated)
        self.aspp = LiteASPP(576, 96)
        self.skip8 = conv_bn_relu(24, 32, 1)
        self.fuse8 = sep_conv(96 + 32, 96)
        self.skip4 = conv_bn_relu(16, 16, 1)
        self.fuse4 = sep_conv(96 + 16, 64)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        size = x.shape[-2:]
        stem = self.stem(x)
        low = self.low(stem)
        mid = self.mid(low)
        high = self.high(mid)
        y = self.aspp(high)
        y = F.interpolate(y, size=low.shape[-2:], mode='bilinear', align_corners=False)
        y = self.fuse8(torch.cat([y, self.skip8(low)], dim=1))
        y = F.interpolate(y, size=stem.shape[-2:], mode='bilinear', align_corners=False)
        y = self.fuse4(torch.cat([y, self.skip4(stem)], dim=1))
        y = self.classifier(self.dropout(y))
        out = F.interpolate(y, size=size, mode='bilinear', align_corners=False)
        return {'out': out, 'low': low, 'mid': mid, 'high': high}


if __name__ == '__main__':
    from torchinfo import summary
    from utils.profiler import ms_per_image

    model = CompactSegNet().eval()
    summary(model, input_size=(1, 3, 320, 320), depth=1)
    with torch.no_grad():
        out = model(torch.randn(1, 3, 320, 320))
    for k, v in out.items():
        print(f'{k:<5} {tuple(v.shape)}')
    print(f'ms/img (CPU): {ms_per_image(model, torch.device("cpu")):.1f}')
