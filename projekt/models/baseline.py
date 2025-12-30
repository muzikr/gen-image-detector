import torch
from torchvision import models

import torch
import torch.nn as nn
from torchvision import models

class EfficientNetBasedModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = models.efficientnet_b0(weights="DEFAULT")

        in_features = self.backbone.classifier[1].in_features  # 1280

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 2)  # SINGLE LOGIT
        )

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x).squeeze(1)
