# -*- coding: utf-8 -*-
import argparse
import torch
import csv
from pathlib import Path
from torch.utils.data import DataLoader
import importlib.util
import sys

# 导入工厂函数
import models_factory
import datasets_factory
import losses_factory
import metrics_factory
import engine

# 导入 Runner
from engine.runner import Runner

def load_config_from_path(config_path: str):
    """从 .py 文件路径中加载配置模块。"""
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
        
    spec = importlib.util.spec_from_file_location(config_path.stem, config_path)
    cfg_module = importlib.util.module_from_spec(spec)
    if config_path.stem in sys.modules:
        del sys.modules[config_path.stem]
    spec.loader.exec_module(cfg_module)
    return cfg_module

def run_test_experiment(config_path_str: str, checkpoint_path: str):
    """组装测试机器并执行测试。"""
    try:
        print(f"\n{'='*80}")
        print(f"🔍 [TESTING] 实验配置: {config_path_str}")
        print(f"📦 权重路径: {checkpoint_path}")
        print(f"{'='*80}\n")
        
        # A. 加载配置
        cfg = load_config_from_path(config_path_str)
        
        # B. 组装零件
        print("Assembling components for testing...")
        model = models_factory.build_model(cfg.model)
        
        # 加载测试数据集 (优先找 cfg.data['test']，找不到则用 val)
        test_cfg = cfg.data.get('test', cfg.data['val'])

        test_dataset = datasets_factory.build_dataset(test_cfg)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=cfg.data['samples_per_gpu'],
            num_workers=cfg.data['workers_per_gpu'],
            shuffle=False,
            pin_memory=True
        )

        criterion = losses_factory.build_loss(cfg.loss)
        metric = metrics_factory.build_metric(cfg.evaluation['metric'])
        
        # 构建钩子 (如可视化钩子)
        hooks = engine.build_hooks(cfg.log_config['hooks'])
        if hasattr(cfg, 'custom_hooks'):
            hooks.extend(engine.build_hooks(cfg.custom_hooks))

        # C. 实例化“测试模式”下的 Runner
        # 注意：测试不需要 optimizer 和 scheduler，传 None 即可
        runner = Runner(
            model=model,
            optimizer=None,
            criterion=criterion,
            metric=metric,
            train_loader=None,
            val_loader=test_loader,
            lr_scheduler=None,
            hooks=hooks,
            cfg=cfg
        )
        
        # D. 按下启动键 
        test_results = runner.test(checkpoint_path=checkpoint_path)
        
        return test_results

    except Exception as e:
        print(f"❌ [FAILED] 测试遇到错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """交互式测试调度中心。"""
    config_dir = Path('./configs')
    print(f"🔍 正在扫描配置目录: {config_dir}")
    config_paths = sorted([p for p in config_dir.glob('**/*.py') if p.name != '__init__.py'])
    
    if not config_paths:
        print("❌ 未找到配置文件。")
        return

    print("\n--- 📜 可用测试规格 ---")
    for i, path in enumerate(config_paths, 1):
        print(f"  {i:>2}: {path.name}")
    
    try:
        idx = int(input("\n👉 请输入要测试的配置序号: "))
        selected_cfg = str(config_paths[idx - 1])
        
        # 自动尝试定位 best_model.pth (基于约定优于配置)
        work_dir = Path('./work_dirs') / config_paths[idx - 1].stem
        default_pth = work_dir / 'best_model.pth'
        
        pth_input = input(f"👉 请输入权重路径 (回车默认为 {default_pth}): ").strip()
        selected_pth = pth_input if pth_input else str(default_pth)
        
        if not Path(selected_pth).exists():
            print(f"❌ 错误: 找不到权重文件 {selected_pth}")
            return

        run_test_experiment(selected_cfg, selected_pth)

    except (ValueError, IndexError):
        print("❌ 输入无效。")

if __name__ == '__main__':
    main()