import torch
from torch.nn import Module

# 1. 我们仍然需要 Registry，来管理我们【自定义】的优化器
class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def register_module(self, cls):
        self._module_dict[cls.__name__] = cls
        return cls

    def get(self, module_name):
        ret = self._module_dict.get(module_name)
        # 注意：这里我们让 get 在找不到时返回 None，而不是直接报错
        return ret

# 为我们自定义的优化器创建一个专属花名册
OPTIMIZERS = Registry('optimizer')


def build_optimizer(model: Module, cfg: dict):
    """
    一个混合模式的优化器构建器。
    它会先在我们自己的 OPTIMIZERS 注册表中查找，如果找不到，
    则会去 torch.optim 中查找。
    """
    optimizer_cfg = cfg.copy()
    optimizer_name = optimizer_cfg.pop('type')

    # 2. 优先从我们自己的花名册中查找
    optimizer_cls = OPTIMIZERS.get(optimizer_name)

    # 3. 如果在我们自己的花名册中找不到，再去 torch.optim 中查找
    if optimizer_cls is None:
        try:
            optimizer_cls = getattr(torch.optim, optimizer_name)
        except AttributeError:
            raise KeyError(
                f"Optimizer '{optimizer_name}' is not found in either the "
                f"custom OPTIMIZERS registry or in torch.optim."
            )
    
    # 4. 实例化优化器
    return optimizer_cls(model.parameters(), **optimizer_cfg)