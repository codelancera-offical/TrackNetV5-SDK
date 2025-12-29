# 文件: models_factory/basic/conv.py

import torch.nn as nn

class BasicConvBlock(nn.Module):
    """
    Conv -> ReLU -> BN
    """
    def __init__(self, in_channels, out_channels, k=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=k,
                padding=(k - 1) // 2,
                bias=False
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.conv(x)

