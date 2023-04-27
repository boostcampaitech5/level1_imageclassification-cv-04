"""
model.py is a file that you use when you load a pre-trained model and use it as it is, or change it.
"""

import torch.nn as nn
import timm


class CustomModel(nn.Module):
    def __init__(self, num_classes : int, backbone : str, pretrained : bool):
        super(CustomModel, self).__init__()

        self.backbone = timm.create_model(backbone, num_classes=num_classes, pretrained=pretrained)

    def forward(self, x):
        x = self.backbone(x)
        return x