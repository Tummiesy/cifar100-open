"""Reference model definitions."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18
import inspect


class CIFARResNet18(nn.Module):
    """
    ResNet-18 adapted for CIFAR-sized images (32x32).

    Exposes optional feature extraction to support future prototype-based
    open-set noisy label generation.
    """

    def __init__(self, num_classes: int = 80):
        super().__init__()
        # Support both newer torchvision APIs (weights=...) and legacy
        # versions that still expect pretrained=....
        resnet18_sig = inspect.signature(resnet18)
        if "weights" in resnet18_sig.parameters:
            self.backbone = resnet18(weights=None, num_classes=num_classes)
        else:
            self.backbone = resnet18(pretrained=False, num_classes=num_classes)
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.backbone.maxpool = nn.Identity()

    def forward(self, x: torch.Tensor, return_features: bool = False):
        # Re-implement forward to expose pre-classifier embedding.
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        feat = torch.flatten(x, 1)
        logits = self.backbone.fc(feat)

        if return_features:
            return logits, feat
        return logits
