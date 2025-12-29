import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from ..builder import DATASETS, build_pipeline

@DATASETS.register_module
class TennisDataset(Dataset):
    def __init__(self, csv_path: str, pipeline: list, data_dir: str = '',
                 input_height: int = 360, input_width: int = 640):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.data_info = pd.read_csv(csv_path) # 读入类
        self.pipeline = build_pipeline(pipeline) # 构建pipeline
        self.input_height = input_height
        self.input_width = input_width
        print(f"Dataset '{self.__class__.__name__}' initialized.")
        print(f"Loaded {Path(csv_path).name}, total samples = {len(self.data_info)}")

        # 打印列名用于调试
        print(f"CSV columns: {self.data_info.columns.tolist()}")

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx: int) -> dict:
        row_info = self.data_info.iloc[idx].to_dict()
        """
            {
                'path_prev': 'frame_0001.jpg',      # 字符串 (String)
                'path': 'frame_0002.jpg',           # 字符串
                'path_next': 'frame_0003.jpg',      # 字符串
                'x_prev': 450.5,                    # 浮点数 (Float)
                'y_prev': 200.0,                    # 浮点数
                'x-coordinate': 460.2,              # 浮点数
                'y-coordinate': 210.5,              # 浮点数
                'visibility': 1,                    # 整数 (Integer)
                'gt_path': 'mask_0002.png'          # 字符串
            }
        """

        results = {
            'data_dir': self.data_dir,
            'input_height': self.input_height,
            'input_width': self.input_width,
            'original_info': row_info,
            **row_info
        }

        # ✨✨✨ 整合坐标信息 ✨✨✨
        # 识别所有坐标字段
        x_fields = [k for k in row_info.keys() if k.startswith('x_') or k == 'x-coordinate']
        y_fields = [k for k in row_info.keys() if k.startswith('y_') or k == 'y-coordinate']

        # 确保字段顺序正确（prev, current, next）
        x_fields_sorted = sorted(x_fields, key=lambda x:
        {'x_prev': 0, 'x-coordinate': 1, 'x_current': 1, 'x_next': 2}.get(x, 3)) # 把列表排序，使得prev在前，current在中间，next在后
        y_fields_sorted = sorted(y_fields, key=lambda x:
        {'y_prev': 0, 'y-coordinate': 1, 'y_current': 1, 'y_next': 2}.get(x, 3)) # 把列表排序，使得prev在前，current在中间，next在后

        # 构建坐标列表
        coords_list = []
        for x_key, y_key in zip(x_fields_sorted, y_fields_sorted):
            if x_key in row_info and y_key in row_info:
                coords_list.append((row_info[x_key], row_info[y_key]))

        if coords_list:
            results['coords'] = coords_list

        # ✨✨✨ 整合可见性信息 ✨✨✨
        # 识别所有可见性字段
        vis_fields = [k for k in row_info.keys() if k.startswith('visibility')]

        # 确保字段顺序正确（prev, current, next）
        vis_fields_sorted = sorted(vis_fields, key=lambda x:
        {'visibility_prev': 0, 'visibility': 1, 'visibility_current': 1, 'visibility_next': 2}.get(x, 3))

        # 构建可见性列表
        visibility_list = []
        for vis_key in vis_fields_sorted:
            if vis_key in row_info:
                visibility_list.append(row_info[vis_key])

        if visibility_list:
            results['visibility'] = visibility_list

        # 识别所有图片路径字段（不包括gt路径）
        img_fields = [k for k in row_info.keys() if 'path' in k and not k.startswith('gt_')]
        results['img_fields'] = img_fields

        # ✨✨✨ 关键修正：处理所有路径字段 ✨✨✨
        all_path_fields = img_fields.copy()

        # 添加所有gt路径字段
        gt_fields = [k for k in row_info.keys() if k.startswith('gt_path')]
        all_path_fields.extend(gt_fields)

        # 调试信息：打印路径字段
        # print(f"DEBUG - Processing path fields: {all_path_fields}")

        # 构建完整路径
        for key in all_path_fields:
            if key in results and pd.notna(results[key]):
                # 确保路径是字符串
                path_str = str(results[key])
                full_path = self.data_dir / path_str
                results[key] = full_path

                # # 调试信息：检查文件是否存在
                # if key in gt_fields:  # 只检查gt文件
                #     exists = full_path.exists()
                #     print(f"  {key}: {full_path} -> exists: {exists}")
                #     if not exists:
                #         print(f"  ⚠️ WARNING: GT file does not exist: {full_path}")
            else:
                print(f"  ⚠️ WARNING: Missing or NaN value for {key}: {results.get(key)}")

        return self.pipeline(results)