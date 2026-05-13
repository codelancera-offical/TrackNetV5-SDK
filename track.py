# -*- coding: utf-8 -*-
import torch
import cv2
import numpy as np
import argparse
import os
import math # 确保 math 被导入
import csv
import json
from pathlib import Path
from tqdm import tqdm
from collections import deque
import time

# 导入你项目里的构建器和模型！
from models_factory.builder import build_model
from datasets_factory.transforms.tracknet_transforms import (
    Resize, ConcatChannels
)

# --- 1. “模型配置库” ---
MODEL_CONFIGS = {
    'v2': dict(
        type='TrackNetV2',
        backbone=dict(type='TrackNetV2Backbone', in_channels=9),
        neck=dict(type='TrackNetV2Neck'),
        head=dict(type='TrackNetV2Head', in_channels=64, out_channels=3)
    ),
    'v4': dict(
        type='TrackNetV4',
        backbone=dict(type='TrackNetV4Backbone', in_channels=9),
        neck=dict(type='TrackNetV4Neck'),
        head=dict(type='TrackNetV2Head', in_channels=64, out_channels=3)
    ),
    'v5': dict(
        type='TrackNetV5',
        backbone=dict(type='TrackNetV2Backbone', in_channels=13),
        neck=dict(type='TrackNetV2Neck'),
        head=dict(type='R_STRHead', in_channels=64, out_channels=3)
    )
}

# --- 2. 辅助函数 (✨ 已修改，与你的 Metric 脚本对齐) ---
def _heatmap_to_points(
    heatmap: np.ndarray,
    threshold: int = 127,
    min_area: float = 1.0,
    max_points: int | None = None,
):
    """
    将单张 uint8 热力图转换为多个候选坐标点。
    """
    if heatmap.dtype != np.uint8:
        heatmap = heatmap.astype(np.uint8)

    _, binary_map = cv2.threshold(heatmap, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # 创建一个掩码，只关注当前轮廓内的区域
            mask = np.zeros(heatmap.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)

            # 在原始热力图中找到该区域内的最大值作为置信度
            _, max_val, _, _ = cv2.minMaxLoc(heatmap, mask=mask)
            conf = round(max_val / 255.0, 4)

            points.append({
                "x": cx,
                "y": cy,
                "conf": conf,
                "area": area
            })

    # 按 conf 从高到低排序
    points.sort(key=lambda p: p['conf'], reverse=True)

    # 支持 max_points 截断
    if max_points is not None and max_points > 0:
        points = points[:max_points]

    return points

def draw_candidate_points(frame, points):
    """
    在帧上画出所有候选点。
    - conf 最高的点用红色实心圆；
    - 其他点用黄色圆；
    - 在点旁边标注 index 和 conf。
    """
    for i, pt in enumerate(points):
        x, y = int(pt['x']), int(pt['y'])
        conf = pt['conf']
        
        if i == 0:
            # 最高置信度点：红色实心
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        else:
            # 其他候选点：黄色空心
            cv2.circle(frame, (x, y), 5, (0, 255, 255), 2)
        
        # 标注 index 和 conf
        label = f"{i}:{conf:.2f}"
        cv2.putText(frame, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return frame

def draw_comet_tail(frame, points_deque, head_radius=8):
    """
    基于半径衰减的圆点轨迹可视化 (无连线版)
    :param frame: 当前图像帧 (BGR)
    :param points_deque: 存储坐标的队列，支持 (x, y) 或 (x, y, vis)
    :param head_radius: 最前端圆点的最大半径
    """
    # 无需创建全黑 overlay，因为我们直接在原图绘制实心圆
    # 如果需要半透明效果，可以保留 overlay 逻辑，这里采用你要求的直接绘制
    
    q_len = len(points_deque)
    if q_len == 0:
        return frame

    for i, pt in enumerate(points_deque):
        # 1. 安全检查：跳过空点或 NaN
        if pt is None:
            continue
            
        # 兼容处理：支持 (x, y) 或 (x, y, vis/conf)
        if len(pt) >= 3:
            tx, ty, tvis = pt[:3]
            if tvis == 0 or tvis is None: continue 
        else:
            tx, ty = pt
            
        if tx is None or ty is None:
            continue

        # 2. 计算半径衰减 (核心逻辑)
        # i=0 (最旧) -> scale 最小; i=q_len-1 (最新) -> scale=1.0
        scale = (i + 1) / q_len
        current_radius = int(head_radius * scale)

        # 确保半径至少为 1
        current_radius = max(1, current_radius)

        # 3. 绘制实心球
        # 使用 LINE_AA 开启抗锯齿，让圆点边缘更丝滑
        cv2.circle(
            frame, 
            (int(tx), int(ty)), 
            current_radius, 
            (0, 0, 255), # 纯红色
            -1,          # 实心
            lineType=cv2.LINE_AA
        )

    return frame


# --- 3. “核心加工车间”: ✨ process_video (✨ 已修改) ✨ ---
def process_video(video_path: Path, model, device, args, output_root_dir: Path) -> dict:
    """
    处理单个视频文件，并生成所有需要的输出文件。
    新逻辑：一次读取 3 帧，推理 3 帧，写入 3 帧，然后跳 3 帧。
    ✨ 新增: 返回一个包含统计数据的字典。
    """
    print(f"\n🏭 Processing video: {video_path.name}")
    
    video_output_dir = output_root_dir / video_path.stem
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))

    # --- ✨ 新增：获取视频原始分辨率 ---
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution_str = f"{width}x{height}"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    input_size = (288, 512)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    trajectory_video_path = video_output_dir / f"{video_path.stem}_trajectory.mp4"
    comparison_video_path = video_output_dir / f"{video_path.stem}_comparison.mp4"
    csv_path = video_output_dir / f"{video_path.stem}_data.csv"
    
    writer_traj = cv2.VideoWriter(str(trajectory_video_path), fourcc, fps, (input_size[1], input_size[0]))
    writer_comp = cv2.VideoWriter(str(comparison_video_path), fourcc, fps, (input_size[1] * 2, input_size[0]))

    trajectory_points = deque(maxlen=fps) 
    
    csv_data = []
    detected_frames_count = 0
    
    # 预处理转换（保持不变）
    resizer = Resize(keys=['path_prev', 'path', 'path_next'], size=input_size)
    concatenator = ConcatChannels(
        keys=['path_prev', 'path', 'path_next'],
        output_key='image'
    )
    
    # --- 新的循环逻辑 ---
    frame_idx_counter = 0
    iteration_count = 0
    pbar = tqdm(total=total_frames, desc=f"Processing {video_path.stem}")
    start_time = time.time() # 记录开始时间

    while cap.isOpened():
        # 1. 一次性读取 3 帧
        ret1, frame1 = cap.read()
        ret2, frame2 = cap.read()
        ret3, frame3 = cap.read()

        # 如果任何一帧读取失败（视频末尾），则终止循环
        if not ret1 or not ret2 or not ret3:
            break

        # 2. 准备模型输入
        frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        frame3_rgb = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)
        
        data_dict = {'path_prev': frame1_rgb, 'path': frame2_rgb, 'path_next': frame3_rgb}
        data_dict = resizer(data_dict)
        data_dict = concatenator(data_dict)
        
        resized_frames = [data_dict['path_prev'], data_dict['path'], data_dict['path_next']]
        
        image_np = data_dict['image']
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float().div(255).unsqueeze(0).to(device)

        # 3. 批量推理
        with torch.no_grad():
            # heatmap_preds 的形状是 [1, 3, H, W]
            heatmap_preds = model(image_tensor)
        
        # 移除 batch 维度，得到 (3, H, W) 的 NumPy 数组
        heatmaps_np = heatmap_preds.squeeze(0).cpu().numpy()
        threshold_uint8 = int(args.threshold * 255) # 阈值仍然由参数控制

        # 4. 循环处理这 3 帧的结果
        for i in range(3):
            current_frame_idx = frame_idx_counter + i
            # 确保不会因为最后几帧凑不满3帧而出错
            if current_frame_idx >= total_frames:
                continue

            single_heatmap_np = heatmaps_np[i] # 形状 (H, W)
            heatmap_uint8 = (single_heatmap_np * 255).astype(np.uint8)

            # (A) 提取坐标 (✨ 已修改：支持多点输出)
            points = _heatmap_to_points(
                heatmap_uint8,
                threshold=threshold_uint8,
                min_area=args.min_area,
                max_points=args.max_points
            )
            
            # (B) 记录 CSV 和轨迹
            if points:
                detected_frames_count += 1
                # 取 points[0] 作为主点加入 trajectory_points
                main_point = points[0]
                trajectory_points.append((main_point['x'], main_point['y'], main_point['conf']))
                csv_row = {
                    'frame_number': current_frame_idx,
                    'detected': 1,
                    'num_points': len(points),
                    'points': json.dumps(points)
                }
            else:
                trajectory_points.append(None)
                csv_row = {
                    'frame_number': current_frame_idx,
                    'detected': 0,
                    'num_points': 0,
                    'points': "[]"
                }
            csv_data.append(csv_row)
            
            # (C) 绘制和写入视频
            frame_to_draw = cv2.cvtColor(resized_frames[i], cv2.COLOR_RGB2BGR)
            
            # 绘制所有候选点
            frame_to_draw = draw_candidate_points(frame_to_draw, points)
            
            # 绘制轨迹视频
            final_traj_frame = draw_comet_tail(frame_to_draw, trajectory_points)
            writer_traj.write(final_traj_frame)

            # 绘制对比视频
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            combined_frame = np.hstack((final_traj_frame, heatmap_color))
            writer_comp.write(combined_frame)

        # 5. 更新计数器和进度条 (关键！)
        frame_idx_counter += 3
        iteration_count += 1
        pbar.update(3)
    
    # --- 循环结束后的清理 ---

    end_time = time.time()
    total_duration = end_time - start_time
    # 平均每秒处理多少个 iteration (每 iteration 处理 3 帧)
    avg_it_per_sec = iteration_count * 3 / total_duration if total_duration > 0 else 0
    print(f"⏱️  Processed {iteration_count * 3} frames of {resolution_str} in {total_duration:.2f} seconds. Avg: {avg_it_per_sec:.2f} frames/sec.")
    pbar.close() # 关闭进度条

    detection_ratio = (detected_frames_count / total_frames) if total_frames > 0 else 0
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame_number', 'detected', 'num_points', 'points'])
        writer.writeheader()
        writer.writerows(csv_data)
        f.write("\n")
        f.write(f"total_detected_frame,{detected_frames_count}\n")
        f.write(f"detection_ratio,{detection_ratio:.4f}\n")

    cap.release()
    writer_traj.release()
    writer_comp.release()
    print(f"✅ Finished processing. Results saved in: {video_output_dir}")
    
    # ✨ 新增：返回统计结果
    stats = {
        'video_name': video_path.name,
        'detected_frames': detected_frames_count,
        'total_frames': total_frames,
        'detection_ratio': round(detection_ratio, 4)
    }
    return stats


# --- 4. “总调度室”: ✨ main (✨ 已修改) ✨ ---
def main():
    parser = argparse.ArgumentParser(description="TrackNet Batch Inference Pipeline")
    parser.add_argument('input_dir', type=str, help='Path to the directory containing input videos.')
    parser.add_argument('weights_path', type=str, help='Path to the model weights (.pth file).')
    
    # ✨ 新增架构选择参数
    parser.add_argument(
        '--arch', 
        type=str, 
        required=True, 
        choices=['v2', 'v4', 'v5'], 
        help='Model architecture to use (v2, v4, or v5).'
    )
    
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for inference (e.g., "cuda:0" or "cpu").')
    
    # ✨ 唯一可调的后处理参数 ✨
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold for detection (0-1).')
    parser.add_argument('--min-area', type=float, default=1.0, help='Minimum area to filter small candidate regions.')
    parser.add_argument('--max-points', type=int, default=10, help='Maximum number of candidate points to output per heatmap (<=0 for unlimited).')

    args = parser.parse_args()

    # 兼容处理 max_points
    if args.max_points <= 0:
        args.max_points = None

    # ✨ 动态获取模型配置
    model_cfg = MODEL_CONFIGS.get(args.arch)
    if model_cfg is None:
        print(f"❌ 错误：未知的架构 '{args.arch}'。请从 'v2', 'v4', 'v5' 中选择。")
        return
        
    print(f"🚀 Starting Batch Inference Pipeline for [TrackNet {args.arch.upper()}]...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    model = build_model(model_cfg)
    model.load_state_dict(torch.load(args.weights_path, map_location='cpu'))
    model.to(device).eval()
    print(f"✅ Model loaded from {args.weights_path} and sent to {device}.")

    input_dir = Path(args.input_dir)
    
    # ✨ 动态设置输出目录
    output_root_dir = input_dir / args.arch 
    output_root_dir.mkdir(exist_ok=True)
    
    print("🔎 Searching for .mp4 and .mov files...")
    video_files = []
    supported_formats = ['*.mp4', '*.mov', '*.MOV', '*.MP4']
    for fmt in supported_formats:
        video_files.extend(input_dir.glob(fmt))
    
    if not video_files:
        print(f"❌ No supported video files (.mp4, .mov) found in {input_dir}. Exiting.")
        return
        
    video_files = sorted(list(set(video_files)))
    print(f"Found {len(video_files)} videos to process.")
    
    # ✨ 1. 初始化汇总列表
    summary_data_list = [] 
    
    for video_path in video_files:
        # ✨ 2. 收集每个视频的返回结果
        try:
            video_stats = process_video(video_path, model, device, args, output_root_dir)
            if video_stats:
                summary_data_list.append(video_stats)
        except Exception as e:
            print(f"❌ ERROR processing {video_path.name}: {e}")
            print("Skipping this video and continuing...")

    # ✨ 3. 循环结束后，写入全局汇总CSV
    if summary_data_list:
        summary_csv_path = output_root_dir / f"_summary_report_{args.arch}.csv"
        print(f"\n📊 Writing summary report to {summary_csv_path}")
        
        fieldnames = ['video_name', 'detected_frames', 'total_frames', 'detection_ratio']
        # 定义中文表头
        chinese_header_map = {
            'video_name': '视频名',
            'detected_frames': '检测到的球帧数',
            'total_frames': '视频总帧数',
            'detection_ratio': '检测比率'
        }
        
        try:
            with open(summary_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                # 写入UTF-8 BOM头和中文表头
                writer = csv.writer(f)
                writer.writerow([chinese_header_map[field] for field in fieldnames])
                
                # 使用 DictWriter 写入数据行
                dict_writer = csv.DictWriter(f, fieldnames=fieldnames)
                dict_writer.writerows(summary_data_list)
        except Exception as e:
            print(f"❌ ERROR writing summary CSV: {e}")
            
    print(f"\n🎉🎉🎉 All videos processed! Check the results in: {output_root_dir} 🎉🎉🎉")


if __name__ == '__main__':
    main()
