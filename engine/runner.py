import torch
from tqdm import tqdm
from pathlib import Path
import numpy as np

class Runner:
    """
    一个封装了完整训练/验证循环的执行器。
    它负责管理训练状态、执行训练循环、调用钩子、进行验证和保存模型。
    """
    def __init__(self, model, optimizer, criterion, metric,
                 train_loader, val_loader, lr_scheduler, 
                 hooks, cfg):
        
        # --- 核心组件 ---
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric = metric
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr_scheduler = lr_scheduler
        self.hooks = hooks
        self.cfg = cfg
        
        # --- 环境与路径 ---
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.work_dir = Path(cfg.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True) # 确保工作目录存在
        
        # --- 训练过程中的状态变量 ---
        self.epoch = 0
        self.global_iter = 0
        self.inner_iter = 0
        self.max_epochs = cfg.total_epochs
        
        if hasattr(cfg, 'steps_per_epoch'):
            self.max_iters_per_epoch = cfg.steps_per_epoch
        else:
            # 只有在 train_loader 不为 None 时才获取长度，否则设为 0（测试模式常见情况）
            self.max_iters_per_epoch = len(self.train_loader) if self.train_loader is not None else 0

        self.outputs = {}      # 用于在钩子之间传递临时数据 (如loss, metrics)
        self.best_metric = 0.0 # 用于保存最佳模型的判断依据

    def call_hooks(self, event_name):
        """调用所有钩子中名为 event_name 的方法。"""
        for hook in self.hooks:
            getattr(hook, event_name)(self)

    def train_epoch(self):
        """执行一个完整的训练 epoch。"""
        self.model.train()
        progress_bar = tqdm(self.train_loader, total=self.max_iters_per_epoch, 
                            desc=f"Train Epoch {self.epoch + 1}/{self.max_epochs}")

        base_lr = self.optimizer.param_groups[0]['lr']
                            
        for i, data_batch in enumerate(progress_bar):
            if i >= self.max_iters_per_epoch:
                break

            self.inner_iter = i
            self.call_hooks('before_iter')
            
            inputs = data_batch['image'].to(self.device)
            targets = data_batch['target'].to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss = self.criterion(logits, targets, epoch_num=self.epoch+1)
            loss.backward()

            # ✨✨✨ 【【【 紧急添加：手动 Warmup 逻辑 】】】 ✨✨✨
            # 1. 检查配置中是否启用了 Warmup
            warmup_enabled = hasattr(self.cfg, 'lr_config') and \
                             self.cfg.lr_config is not None and \
                             self.cfg.lr_config.get('warmup') == 'linear'
                             
            if warmup_enabled:
                # 2. 从配置获取 Warmup 参数
                warmup_iters = self.cfg.lr_config.get('warmup_iters', 0) # 总 Warmup 迭代次数
                warmup_ratio = self.cfg.lr_config.get('warmup_ratio', 1e-6) # 初始 LR 比例
                
                # 3. 检查当前是否处于 Warmup 阶段
                if self.global_iter < warmup_iters:
                    # 4. 计算当前的 Warmup 学习率
                    #    k 是一个从 warmup_ratio 线性增长到 1.0 的因子
                    k = (1 - warmup_ratio) * self.global_iter / warmup_iters + warmup_ratio
                    current_warmup_lr = base_lr * k
                    
                    # 5. 手动设置优化器中所有参数组的学习率
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = current_warmup_lr
                
                # 6. (可选) Warmup 结束时，确保恢复到基础学习率
                elif self.global_iter == warmup_iters:
                     for param_group in self.optimizer.param_groups:
                        param_group['lr'] = base_lr # 恢复基础 LR，防止浮点误差
            # ✨✨✨ 【【【 Warmup 逻辑结束 】】】 ✨✨✨

            # ✨✨✨ 【【【 紧急添加：手动梯度裁剪 】】】 ✨✨✨
            # 1. 检查配置文件中是否存在 optimizer_config 和 grad_clip 设置
            if hasattr(self.cfg, 'optimizer_config') and \
               self.cfg.optimizer_config is not None and \
               'grad_clip' in self.cfg.optimizer_config and \
               self.cfg.optimizer_config['grad_clip'] is not None:
                
                # 2. 从配置中获取 max_norm 值
                max_norm = self.cfg.optimizer_config['grad_clip'].get('max_norm')
                
                # 3. 如果 max_norm 有效，则执行梯度裁剪
                if max_norm is not None and max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
                    # (可选) 可以在这里加一行 print 确认裁剪已执行
                    # print(f"  -- Grad clipped with max_norm={max_norm}")
            # ✨✨✨ 【【【 梯度裁剪代码结束 】】】 ✨✨✨

            self.optimizer.step()
            
            self.global_iter += 1
            self.outputs['loss'] = loss.item()
            self.outputs['batch_size'] = inputs.size(0)
            self.current_lr = self.optimizer.param_groups[0]['lr']

            self.call_hooks('after_iter')
            progress_bar.set_postfix(loss=loss.item())

    @torch.no_grad()
    def validate_epoch(self):
        """
        执行一个完整的验证 epoch，并正确地调用钩子，为可视化提供支持。
        """
        self.model.eval()
        self.metric.reset()
        
        # 1. 新增：广播“验证epoch开始”事件
        # 这会触发 ValidationVisualizerHook 的 before_val_epoch 方法
        self.call_hooks('before_val_epoch')

        val_losses = []
        progress_bar = tqdm(self.val_loader, desc=f"Validate Epoch {self.epoch + 1}")
        for i, data_batch in enumerate(progress_bar):
            self.inner_iter = i
            
            inputs = data_batch['image'].to(self.device)
            targets = data_batch['target'].to(self.device)
            logits = self.model(inputs)
            
            # 2. 新增：将当前批次的数据暂存到 outputs 中，供钩子访问
            self.outputs['val_batch'] = data_batch
            self.outputs['val_logits'] = logits

            # 计算损失和指标
            loss = self.criterion(logits, targets)
            val_losses.append(loss.item())
            self.metric.update(logits, data_batch) # metric传入的是logits和对应的batch，算出是tp还是啥别的
            
            # 3. 新增：广播“验证iter结束”事件
            # 这是 ValidationVisualizerHook 工作的关键！
            self.call_hooks('after_val_iter')

        # 计算并打印最终结果
        eval_results = self.metric.compute() # 这里才算出来F1
        eval_results['loss'] = np.mean(val_losses)
        self.outputs['val_metrics'] = eval_results 
        print(f"Validation Results: {eval_results}")
        
        # 4. 新增：广播“验证epoch结束”事件
        self.call_hooks('after_val_epoch')

    def run(self):
        """启动完整的训练流程。"""
        print("🚀 Starting Runner...")
        self.call_hooks('before_run')
        
        for self.epoch in range(self.max_epochs):
            self.call_hooks('before_epoch')
            self.train_epoch()
            
            if (self.epoch + 1) % self.cfg.evaluation['interval'] == 0:
                self.validate_epoch()
                
                # --- 新增代码 START ---
                # 1. 每次验证后，都保存当前 epoch 的模型快照
                # 使用 f-string 创建一个独一无二的文件名，如 'epoch_5.pth'
                checkpoint_path = self.work_dir / f'epoch_{self.epoch + 1}.pth'
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"✅ Checkpoint saved for epoch {self.epoch + 1} to {checkpoint_path}")
                # --- 新增代码 END ---

                # 2. 保留原有的逻辑，用于保存和更新性能最佳的模型
                current_f1 = self.outputs.get('val_metrics', {}).get('F1-Score', 0.0)
                if current_f1 > self.best_metric:
                    self.best_metric = current_f1
                    best_model_path = self.work_dir / 'best_model.pth'
                    torch.save(self.model.state_dict(), best_model_path)
                    print(f"🏆 New best model saved to {best_model_path} with F1-score: {self.best_metric:.4f}")
            
            # 只有在学习率调度器存在时，才执行 .step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.call_hooks('after_epoch')
            
        self.call_hooks('after_run')
        print("\n🎉 Training finished!")
        print(f"Best F1-Score on validation set: {self.best_metric:.4f}")


    # ... 您原有的代码 ...

    @torch.no_grad()
    def test(self, test_loader=None, checkpoint_path=None):
        """
        执行完整的测试流程。
        参数:
            test_loader: 指定测试数据集加载器。若为 None 则尝试使用 val_loader。
            checkpoint_path: 指定要加载的权重路径 (.pth)。
        """
        print(f"🔍 Starting Testing Mode...")
        
        # 1. 加载指定的权重 (如果是从 test.py 启动)
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print(f"✅ Loaded checkpoint from {checkpoint_path}")

        self.model.eval()
        self.metric.reset()
        
        # 使用传入的 loader 或默认的 val_loader
        loader = test_loader if test_loader is not None else self.val_loader
        
        # 2. 触发测试专用的钩子 (建议在 BaseHook 中也增加对应的接口)
        self.call_hooks('before_test_epoch')

        test_losses = []
        progress_bar = tqdm(loader, desc="Testing")
        
        for i, data_batch in enumerate(progress_bar):
            self.inner_iter = i
            
            inputs = data_batch['image'].to(self.device)
            targets = data_batch['target'].to(self.device)
            logits = self.model(inputs)
            
            # 将数据存入 outputs 供 VisualizerHook 等使用
            self.outputs['test_batch'] = data_batch
            self.outputs['test_logits'] = logits

            # 计算
            loss = self.criterion(logits, targets)
            test_losses.append(loss.item())
            self.metric.update(logits, data_batch)
            
            # 3. 触发测试迭代钩子
            self.call_hooks('after_test_iter')

        # 4. 汇总结果
        eval_results = self.metric.compute()
        eval_results['loss'] = np.mean(test_losses)
        self.outputs['test_metrics'] = eval_results
        
        print(f"\n" + "="*30)
        print(f" Final Test Results: ")
        for k, v in eval_results.items():
            print(f"  - {k}: {v:.4f}" if isinstance(v, float) else f"  - {k}: {v}")
        print("="*30 + "\n")
        
        self.call_hooks('after_test_epoch')
        return eval_results