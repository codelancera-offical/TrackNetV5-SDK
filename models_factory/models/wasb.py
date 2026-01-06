import torch.nn as nn
from ..builder import MODELS, build_backbone, build_head

@MODELS.register_module
class WASB(nn.Module):
    def __init__(self, backbone, head, neck=None):
        super().__init__()
        self.backbone = build_backbone(backbone)
        self.head = build_head(head)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits