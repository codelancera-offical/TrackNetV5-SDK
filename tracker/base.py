from abc import ABC, abstractmethod

class BaseTracker(ABC):
    def __init__(self, fps):
        self.fps = fps

    @abstractmethod
    def refine(self, raw_points):
        """
        raw_points: List[Point] (检测阶段原始输出, 包含置信度)
        return: Dict[int, point] (以帧号为 key 的优化坐标)
        """