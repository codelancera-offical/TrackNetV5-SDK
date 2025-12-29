# Ball-Tracking/models/builder.py

# 模型车间 + 注册表

class Registry: # 注册表类
    def __init__(self, name):
        self._name = name          # 注册表的名称
        self._module_dict = dict() # 模块的注册表 模块名-模块类 键值对

    def register_module(self, cls):
        self._module_dict[cls.__name__] = cls # 把 模块名-模块类 键值对 放进注册表里面
        return cls

    def get(self, module_name):
        ret = self._module_dict.get(module_name)
        if ret is None:
            raise KeyError(f"{module_name} is not registered in {self._name}")
        return ret

# 创建不同组件的注册表

BACKBONES = Registry('backbone')
NECKS = Registry('neck')
HEADS = Registry('head')
MODELS = Registry('models')

def build(cfg, registry):
    args = cfg.copy()
    module_name = args.pop('type')
    module_cls = registry.get(module_name)
    return module_cls(**args)

# 创建专用的构建函数

def build_backbone(cfg):
    return build(cfg, BACKBONES)

def build_neck(cfg):
    return build(cfg, NECKS)

def build_head(cfg):
    return build(cfg, HEADS)

def build_model(cfg):
    return build(cfg, MODELS)