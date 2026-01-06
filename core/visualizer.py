# -*- coding: utf-8 -*-
import cv2
import numpy as np
from collections import deque
from typing import Tuple, Optional
from .pipeline import BallPoint

class BallVisualizer:
    """
    Stage 3: 渲染器
    负责将修复后的轨迹数据绘制到视频帧上。
    实现效果：基于时间滑动窗口的“半径衰减”残影轨迹。
    """

    def __init__(self, fps: int, head_radius: int = 8, trail_color: Tuple[int, int, int] = (0, 0, 255)):
        """
        初始化可视化器
        :param fps: 视频帧率 (决定轨迹保留的时长，默认为1秒)
        :param head_radius: 轨迹最前端(当前帧)圆点的最大半径
        :param trail_color: 轨迹颜色 (BGR 元组, 默认红色)
        """
        self.head_radius = head_radius
        self.trail_color = trail_color
        
        # 核心状态：滑动窗口缓存队列
        # maxlen=fps 意味着只保留最近 1 秒内的历史点
        self.history = deque(maxlen=int(fps))

    def render(self, frame: np.ndarray, current_point: BallPoint) -> np.ndarray:
        """
        处理单帧图像：更新历史记录并绘制轨迹
        :param frame: 原始视频帧 (BGR)
        :param current_point: 当前帧对应的 BallPoint 对象(来自 refined_dict)
        :return: 绘制完成后的视频帧
        """
        # 1. 更新状态：将当前点压入队列
        # 即使 is_detected=False 也要压入，以保持时间连续性
        self.history.append(current_point)

        # 2. 渲染循环：遍历队列绘制所有有效点
        q_len = len(self.history)
        for i, pt in enumerate(self.history):
            # 跳过无效点（漏检帧）
            if pt is None or not pt.is_detected:
                continue
            
            # 安全检查：确保坐标不为 NaN 或 Inf
            if np.isnan(pt.x) or np.isnan(pt.y) or np.isinf(pt.x) or np.isinf(pt.y):
                continue

            # --- 核心逻辑：计算半径衰减比例 ---
            # i 是点在队列中的索引。
            # i=0 表示队列中最老的点，i=q_len-1 表示最新的点。
            # scale 范围从接近 0 (最老) 到 1.0 (最新) 渐变。
            scale = (i + 1) / q_len
            
            # 根据比例计算当前点的半径
            current_radius = int(self.head_radius * scale)
            
            # 确保半径至少为 1 个像素，否则看不见
            current_radius = max(1, current_radius)

            # --- 绘图 ---
            # 使用 cv2.LINE_AA 开启抗锯齿，让小圆点边缘更平滑
            cv2.circle(
                frame,
                (int(pt.x), int(pt.y)),
                current_radius,
                self.trail_color,
                -1,  # 实心圆
                lineType=cv2.LINE_AA
            )
            
        return frame

    def reset(self):
        """清空历史轨迹缓存 (用于处理新视频前)"""
        self.history.clear()