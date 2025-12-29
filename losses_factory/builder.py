# 我们可以从一个公共的地方导入Registry类，或者在这里再定义一次
# 为保持独立性，我们在这里再定义一次
class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def register_module(self, cls):
        self._module_dict[cls.__name__] = cls
        return cls

    def get(self, module_name):
        ret = self._module_dict.get(module_name)
        if ret is None:
            raise KeyError(f"'{module_name}' is not registered in the '{self._name}' registry.")
        return ret

# --- 为损失函数创建专属的“花名册” ---
LOSSES = Registry('loss')

# --- 通用的构建函数 ---
def build(cfg: dict, registry: Registry):
    args = cfg.copy()
    module_name = args.pop('type')
    module_cls = registry.get(module_name)
    return module_cls(**args)

# --- 专用于构建损失函数的便捷函数 ---
def build_loss(cfg: dict):
    """根据配置构建一个损失函数实例。"""
    return build(cfg, LOSSES)