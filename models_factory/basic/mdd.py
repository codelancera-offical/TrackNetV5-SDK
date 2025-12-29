import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Motion Direction Decouple layer
class MDD(nn.Module):
    def __init__(self,):
        super().__init__()
        # 您当前的测试值
        self.a = nn.Parameter(torch.tensor(0.2))
        self.b = nn.Parameter(torch.tensor(0.15))
        self.register_buffer('R_WEIGHT', torch.tensor(0.299).view(1, 1, 1, 1))
        self.register_buffer('G_WEIGHT', torch.tensor(0.587).view(1, 1, 1, 1))
        self.register_buffer('B_WEIGHT', torch.tensor(0.114).view(1, 1, 1, 1))
    
    def _rgb_to_luminance(self, img_tensor: torch.Tensor) -> torch.Tensor:
        R = img_tensor[:, 0:1]
        G = img_tensor[:, 1:2]
        B = img_tensor[:, 2:3]
        luminance = (self.R_WEIGHT * R + self.G_WEIGHT * G + self.B_WEIGHT * B)
        return luminance
    
    def _power_normalization(self, input_tensor, a, b):
        return 1.0 / (1.0 + torch.exp(
            -(5.0 / (0.45 * torch.abs(torch.tanh(a)) + 1e-6)) * (torch.abs(input_tensor) - 0.6 * torch.tanh(b))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_prev_tensor = x[:, 0:3] 
        img_curr_tensor = x[:, 3:6]
        img_next_tensor = x[:, 6:9] 
        img_prev_tensor_lum = self._rgb_to_luminance(img_prev_tensor)
        img_curr_tensor_lum = self._rgb_to_luminance(img_curr_tensor)
        img_next_tensor_lum = self._rgb_to_luminance(img_next_tensor)
        diff1 = img_curr_tensor_lum - img_prev_tensor_lum
        diff2 = img_next_tensor_lum - img_curr_tensor_lum
        
        brighten1_att = self._power_normalization(torch.relu(diff1), self.a, self.b)
        darken1_att = self._power_normalization(torch.relu(-diff1), self.a, self.b)
        brighten2_att = self._power_normalization(torch.relu(diff2), self.a, self.b)
        darken2_att = self._power_normalization(torch.relu(-diff2), self.a, self.b)
    
        return torch.cat([
            # img_prev_tensor, 
            brighten1_att, 
            darken1_att, 
            # img_curr_tensor, 
            brighten2_att, 
            darken2_att, 
            # img_next_tensor
        ], dim=1)
# --- 模块定义结束 ---


def load_images(folder_path, image_names=['1.png', '2.png', '3.png']):
    """加载、预处理图像并堆叠为9通道张量"""
    transform = T.Compose([
        T.ToTensor()
    ])
    
    tensor_list = []
    rgb_images = [] # 存储原始 RGB numpy 图像用于显示
    
    print(f"--- 正在从 {folder_path} 加载图片 ---")
    for name in image_names:
        img_path = os.path.join(folder_path, name)
        if not os.path.exists(img_path):
            print(f"❌ 错误：找不到图片 {img_path}")
            return None, None
            
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"❌ 错误：无法读取图片 {img_path}")
            return None, None
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rgb_images.append(img_rgb) # 存储原始图像
        
        img_tensor = transform(img_rgb)
        tensor_list.append(img_tensor)
        print(f"✅ 成功加载: {name} (形状: {img_tensor.shape})")

    stacked_tensor = torch.cat(tensor_list, dim=0)
    input_tensor = stacked_tensor.unsqueeze(0) # 增加 Batch 维度
    
    print(f"--- 最终输入张量形状: {input_tensor.shape} ---")
    
    return input_tensor, rgb_images

def visualize_results(rgb_images, att_maps, output_folder_name):
    """
    使用 Matplotlib 可视化结果, 并将合并的 R/B 叠加图保存到指定文件夹。
    """
    
    # 提取原始图像
    img1, img2, img3 = rgb_images
    
    # 提取注意力图 (从 [1, 1, H, W] 转换为 [H, W] numpy 数组)
    # 注意力图的值域为 [0, 1]
    b1 = att_maps['b1'].squeeze().detach().cpu().numpy()
    d1 = att_maps['d1'].squeeze().detach().cpu().numpy()
    b2 = att_maps['b2'].squeeze().detach().cpu().numpy()
    d2 = att_maps['d2'].squeeze().detach().cpu().numpy()
    
    # --- 【修改】创建合并的 R/B 叠加图 ---
    # R (Brighten) + G (0) + B (Darken) => 红色表示变亮，蓝色表示变暗
    
    # 帧 1 -> 帧 2 的运动叠加图
    motion_overlay_1_to_2 = np.stack([b1, np.zeros_like(b1), d1], axis=-1)
    
    # 帧 2 -> 帧 3 的运动叠加图
    motion_overlay_2_to_3 = np.stack([b2, np.zeros_like(b2), d2], axis=-1)

    # --- 【修改】保存合并后的注意力图 ---
    maps_to_save = {
        'motion_1_to_2_overlay.png': motion_overlay_1_to_2,
        'motion_2_to_3_overlay.png': motion_overlay_2_to_3,
    }

    for filename, np_map_rgb in maps_to_save.items():
        # 1. 转换回 [0, 255] uint8 格式
        # np_map_rgb 形状为 [H, W, 3]
        img_to_save = (np_map_rgb * 255).astype(np.uint8)
        
        # 2. 创建完整保存路径
        save_path = os.path.join(output_folder_name, filename)
        
        # 3. 保存图像 (cv2 需要 BGR 格式，所以需要转换)
        img_bgr = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img_bgr)
        
    print(f"--- 成功保存 2 张 R/B 叠加注意力图到 {output_folder_name} ---")
    # --- 保存逻辑修正结束 ---


    # --- 绘图 (与之前相同，但现在直接使用上面生成的变量) ---
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("运动方向注意力图可视化", fontsize=20)

    axes[0, 0].imshow(img1)
    axes[0, 0].set_title("原始帧 1 (Prev)")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2)
    axes[0, 1].set_title("原始帧 2 (Curr)")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img3)
    axes[0, 2].set_title("原始帧 3 (Next)")
    axes[0, 2].axis('off')

    axes[1, 0].imshow(motion_overlay_1_to_2)
    axes[1, 0].set_title("运动 (帧1 -> 帧2)\n[红色=变亮, 蓝色=变暗]")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(motion_overlay_2_to_3)
    axes[1, 1].set_title("运动 (帧2 -> 帧3)\n[红色=变亮, 蓝色=变暗]")
    axes[1, 1].axis('off')
    
    # 显示 a 和 b 参数的值
    a_val = att_maps['a_val']
    b_val = att_maps['b_val']
    axes[1, 2].text(0.5, 0.5, f"Power Norm Params:\na = {a_val:.3f}\nb = {b_val:.3f}", 
                    fontsize=14, ha='center', va='center')
    axes[1, 2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print("\n--- 正在显示可视化结果... (请关闭图像窗口以继续) ---")
    plt.show()


# --- 【修改】main 函数以传递 a 和 b 的值 ---
def main():
    parser = argparse.ArgumentParser(description="测试 MotionDirectionAttentionLayer")
    parser.add_argument(
        '--folder_path', 
        type=str, 
        required=True, 
        help="包含 1.png, 2.png, 3.png 图像的文件夹路径"
    )
    args = parser.parse_args()

    input_tensor, rgb_images = load_images(args.folder_path)
    if input_tensor is None:
        return

    model = MDD()
    model.eval()
    
    # 获取 a 和 b 的值
    a_val = model.a.item()
    b_val = model.b.item()
    
    # 创建文件夹名称，格式化为3位小数
    output_folder_name = f"att_maps_a_{a_val:.3f}_b_{b_val:.3f}"
    
    # 创建文件夹
    os.makedirs(output_folder_name, exist_ok=True)
    print(f"--- 正在准备保存注意力图到: {output_folder_name} ---")
    
    with torch.no_grad():
        output_tensor = model(input_tensor)

    print(f"--- 模型输出张量形状: {output_tensor.shape} ---")

    # 注意力图字典中新增 a 和 b 的值，用于在 visualize_results 中显示
    att_maps = {
        'b1': output_tensor[:, 3:4],
        'd1': output_tensor[:, 4:5],
        'b2': output_tensor[:, 8:9],
        'd2': output_tensor[:, 9:10],
        'a_val': a_val, # 新增
        'b_val': b_val  # 新增
    }

    # 传递 output_folder_name 给可视化函数
    visualize_results(rgb_images, att_maps, output_folder_name)
    
    print("--- 测试完成 ---")

if __name__ == "__main__":
    main()