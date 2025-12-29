# 您可以把 Registry 类的定义放在一个更通用的地方，比如 utils/registry.py
# 但为了简单，我们先在这里复制一份
from torchvision.transforms import Compose

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
            raise KeyError(f"{module_name} is not registered in {self._name}")
        return ret

# --- 只保留和数据相关的花名册 ---
DATASETS = Registry('dataset')
TRANSFORMS = Registry('transform') # 也常被称为 PIPELINE

# --- 通用的构建函数 ---
def build(cfg, registry):
    args = cfg.copy()
    module_name = args.pop('type')
    module_cls = registry.get(module_name)
    return module_cls(**args)

# --- 只保留和数据相关的构建函数 ---
def build_dataset(cfg):
    return build(cfg, DATASETS)

def build_pipeline(cfg):
    # 流水线是一系列transforms的组合
    pipeline = [build(c, TRANSFORMS) for c in cfg]
    # 在torchvision中，通常会用Compose将它们组合起来
    # from torchvision.transforms import Compose
    # return Compose(pipeline)
    
    # 2. 用“传送带”把所有零件串起来，返回一个可调用的对象
    return Compose(pipeline)