import torch
import torch.nn as nn
from numpy.f2py.auxfuncs import throw_error
from numpy.ma.core import argmax

from ..builder import LOSSES


@LOSSES.register_module
class TrackNetV2Loss(nn.Module):
    """
        一个实现了您提供的 "WBCE" (变种 Focal Loss, 伽马=2) 的 nn.Module。

        它接收 Logits (模型的原始输出，未经过 Sigmoid 激活) 以保证数值稳定性。

        公式: -sum [ (1-P)^2 * Y * log(P) + P^2 * (1-Y) * log(1-P) ]
        其中 P = sigmoid(logits)
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """
           计算损失值。

           参数:
               logits (torch.Tensor):
                   来自模型头部的原始输出。
                   期望形状: [B, 3, H, W]

               targets (torch.Tensor):
                   真实的灰度标签图 (Ground Truth Grayscale Map)。
                   期望形状: [B, 3, H, W]，其中每个像素的值是0或255
                   期望数据类型: torch.long (int64)。

           返回:
               torch.Tensor: 计算出的交叉熵损失值，一个标量张量。
       """

        # 1. 将 Logits 转换维度
        # prob:
        prob = logits

        # 2. 将targets的Gt图灰度值大于0的都设为正样本
        y = torch.where(targets == 255, 1.0, 0.0)

        # 3. 计算权重（基于预测概率）
        # 对于正样本：权重 = (1 - prob)^2
        # 对于负样本：权重 = prob^2
        pos_weight = (1.0 - prob).pow(2)
        neg_weight = prob.pow(2)

        # 4. 计算标准 BCE 损失的 Log 部分
        eps = 1e-6
        # prob 限制在[eps，1-eps]
        prob = torch.clamp(prob, eps, 1.0 - eps)
        weight_bce_loss = -(
                pos_weight * y * torch.log(prob + eps) +
                neg_weight * (1.0 - y) * torch.log((1.0 - prob + eps))
        )

        # 5. 检查是否有NaN值
        if torch.isnan(weight_bce_loss).any():
            print("警告: 损失中出现NaN值!")
            print(f"logits范围: [{logits.min():.6f}, {logits.max():.6f}]")
            print(f"prob范围: [{prob.min():.6f}, {prob.max():.6f}]")
            print(f"y中正样本数量: {(y == 1).sum().item()}")
            print(f"y中负样本数量: {(y == 0).sum().item()}")

        # 3. 聚合损失
        if self.reduction == 'mean':
            return weight_bce_loss.mean()
        elif self.reduction == 'sum':
            return weight_bce_loss.sum()
        else:
            return weight_bce_loss  # 返回逐元素的损失


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("--- 测试 TrackNetV1Loss (基于256灰度等级分类原理) ---")

    # 定义模拟参数
    B, C, H, W = 4, 3, 640, 360  # 批大小=4, 类别/灰度等级=256, 尺寸=64x64

    # 1. 初始化损失函数
    loss_fn = TrackNetV2Loss()
    print("损失函数已初始化")

    # 2. 创建模拟输入
    # 模型的输出 logits, 形状 [B, 1, H, W]
    mock_logits = torch.randn(B, C, H, W)
    # 真实的灰度标签图, 形状 [B, H, W], 值为 0 到 255
    mock_targets = torch.randint(0, 255, (B, 3, H, W), dtype=torch.long)

    print(f"\n模拟 Logits 形状: {mock_logits.shape}")
    print(f"模拟 Targets 形状: {mock_targets.shape}")
    print(f"Targets 数据类型: {mock_targets.dtype}")

    # 3. 计算损失
    try:
        loss_value = loss_fn(torch.sigmoid(mock_logits), mock_targets)
        print(f"\n计算得到的损失值: {loss_value.item():.4f}")
        print(f"损失值是一个标量: {loss_value.dim() == 0}")
        print("\n✅ 测试通过：损失函数成功处理了符合原理的输入维度。")
    except Exception as e:
        print(f"\n❌ 测试失败. 错误: {e}")