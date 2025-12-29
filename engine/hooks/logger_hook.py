from .base_hook import BaseHook
from ..builder import HOOKS

# 确保您已经安装了 tensorboardX: pip install tensorboardX
try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None

@HOOKS.register_module
class TextLoggerHook(BaseHook):
    """
    一个简单的钩子，用于在控制台打印训练日志。
    """
    def after_iter(self, runner):
        """在每次迭代结束后被 Runner 调用。"""
        # ✨ 关键修正：通过 runner.cfg 访问配置
        if self.every_n_iters(runner, runner.cfg.log_config['interval']):
            # 从 runner 对象中获取需要打印的实时信息
            log_str = (f"Epoch [{runner.epoch + 1}][{runner.inner_iter + 1}/{runner.max_iters_per_epoch}]  "
                       f"lr: {runner.current_lr:.5f},  loss: {runner.outputs['loss']:.4f}")
            print(log_str)

    def every_n_iters(self, runner, n):
        """一个辅助函数，用于判断是否达到了间隔次数。"""
        return (runner.inner_iter + 1) % n == 0 if n > 0 else False


@HOOKS.register_module
class TensorboardLoggerHook(BaseHook):
    """
    一个用于将日志记录到 TensorBoard 的钩子。
    """
    def __init__(self):
        if SummaryWriter is None:
            raise ImportError('Please install tensorboardX to use TensorboardLoggerHook.')
        self.writer = None

    def before_run(self, runner):
        """在训练开始前，创建 SummaryWriter 实例。"""
        # 使用 pathlib 的 / 操作符来拼接路径，更健壮
        log_dir = runner.work_dir / 'tf_logs'
        self.writer = SummaryWriter(log_dir)
        print(f"Tensorboard log will be saved to: {log_dir}")

    def after_iter(self, runner):
        """在每次迭代结束后，记录训练相关的标量数据。"""
        # ✨ 关键修正：通过 runner.cfg 访问配置
        if self.every_n_iters(runner, runner.cfg.log_config['interval']):
            # runner.global_iter 是全局总迭代次数，适合作为x轴
            self.writer.add_scalar('train/loss', runner.outputs['loss'], runner.global_iter)
            self.writer.add_scalar('train/lr', runner.current_lr, runner.global_iter)
            
    def after_epoch(self, runner):
        """在每个epoch结束后，记录验证相关的标量数据。"""
        # 检查 runner.outputs 中是否有验证结果
        if 'val_metrics' in runner.outputs:
            for metric, value in runner.outputs['val_metrics'].items():
                # runner.epoch 是当前的epoch数，适合作为x轴
                self.writer.add_scalar(f'val/{metric}', value, runner.epoch + 1)

    def after_run(self, runner):
        """在训练结束后，关闭 writer。"""
        self.writer.close()

    def every_n_iters(self, runner, n):
        """一个辅助函数，用于判断是否达到了间隔次数。"""
        return (runner.inner_iter + 1) % n == 0 if n > 0 else False