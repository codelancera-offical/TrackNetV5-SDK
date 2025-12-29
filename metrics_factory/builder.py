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

# --- 为“评估指标” (Metrics) 创建专属的“花名册” ---
METRICS = Registry('metric')

def build(cfg: dict, registry: Registry):
    args = cfg.copy()
    module_name = args.pop('type')
    module_cls = registry.get(module_name)
    return module_cls(**args)

# --- 专用于构建评估指标的便捷函数 ---
def build_metric(cfg: dict):
    """根据配置构建一个评估指标实例。"""
    return build(cfg, METRICS)