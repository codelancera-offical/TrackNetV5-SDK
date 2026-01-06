# -*- coding: utf-8 -*-
import cv2
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class BallPoint:
    """传送带上的标准数据包"""
    x: float = 0.0
    y: float = 0.0
    conf: float = 0.0
    is_detected: bool = False

class TennisPipeline:
    def __init__(self, detector, tracker, visualizer):
        """
        自动化流水线调度中心
        :param detector: Stage 1 处理器 (Detector 实例)
        :param tracker: Stage 2 处理器 (BaseTracker 的子类实例)
        :param visualizer: Stage 3 处理器 (Visualizer 实例)
        """
        self.detector = detector
        self.tracker = tracker
        self.visualizer = visualizer
        
        # 内部状态存储
        self.raw_points: List[BallPoint] = []
        self.refined_dict: Dict[int, BallPoint] = {}

    def run(self, video_path: str, output_dir: str):
        """执行全流程：检测 -> 追踪 -> 渲染"""
        video_path = Path(video_path)
        output_root = Path(output_dir) / video_path.stem
        output_root.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Starting Pipeline for: {video_path.name}")

        # --- Stage 1: Detection Pass ---
        # 产出包含 conf 的原始坐标列表
        self.raw_points = self._pass1_detection(video_path)

        # --- Stage 2: Tracking Pass ---
        # 算法介入，利用全局信息修复 raw_points，产出坐标字典
        self.refined_dict = self._pass2_tracking(self.raw_points)

        # --- Stage 3: Rendering Pass ---
        # 根据修复后的字典进行二次遍历，高清渲染
        output_video_path = output_root / f"{video_path.stem}_refined.mp4"
        self._pass3_rendering(video_path, str(output_video_path), self.refined_dict)

        print(f"✅ All stages completed. Result saved to: {output_video_path}")

    def _pass1_detection(self, video_path: Path) -> List[BallPoint]:
        """第一遍扫描：GPU 密集型检测"""
        print("\n[Stage 1/3] Running Neural Inference...")
        # 调用 detector 的接口，获取全视频原始坐标
        return self.detector.detect_video(str(video_path))

    def _pass2_tracking(self, raw_points: List[BallPoint]) -> Dict[int, BallPoint]:
        """第二遍扫描：算法级轨迹优化"""
        print("\n[Stage 2/3] Refining Trajectory with Tracker...")
        # 将原始列表（含 conf）交给追踪器进行逻辑修复
        return self.tracker.refine(raw_points)

    def _pass3_rendering(self, video_path: Path, output_path: str, refined_dict: Dict[int, BallPoint]):
        """第三遍扫描：I/O 密集型渲染"""
        print("\n[Stage 3/3] Rendering Final Video...")
        
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        pbar = tqdm(total=total_frames, desc="Rendering")
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 从‘坐标手册’中提取当前帧经过修复的 BallPoint
            current_point = refined_dict.get(frame_idx, BallPoint(is_detected=False))
            
            # 调用 visualizer 执行渲染逻辑（半径衰减等）
            rendered_frame = self.visualizer.render(frame, current_point)
            
            out.write(rendered_frame)
            frame_idx += 1
            pbar.update(1)
            
        pbar.close()
        cap.release()
        out.release()