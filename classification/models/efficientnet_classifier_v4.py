# ============================================================
# classification/models/efficientnet_classifier_v4.py
# EfficientNet-B0 3-class flood classifier (v4)
# ============================================================

import torch.nn as nn
import torchvision.models as models


class FloodClassifierV4(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 3),
        )

    def forward(self, x):
        return self.backbone(x)
