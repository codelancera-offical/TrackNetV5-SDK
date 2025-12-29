class Registry:
    """一个通用的注册表类。"""
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

# --- 为“钩子” (Hooks) 创建专属的“花名册” ---
HOOKS = Registry('hook')


# --- 通用的构建函数 ---
def build(cfg: dict, registry: Registry):
    """根据配置从指定的注册表中构建一个模块实例。"""
    args = cfg.copy()
    module_name = args.pop('type')
    module_cls = registry.get(module_name)
    return module_cls(**args)


# --- 专用于构建钩子列表的便捷函数 ---
def build_hooks(cfg_list: list) -> list:
    """
    根据配置列表，构建一个由多个钩子(hook)实例组成的列表。
    """
    hooks = [build(cfg, HOOKS) for cfg in cfg_list]
    return hooks