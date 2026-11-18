"""Load inherited DataLoader factory without prescribing dataset or transforms."""
import importlib


def create_loaders(factory, kwargs=None):
    if not factory or ":" not in factory:
        raise ValueError("Set data.loader_factory to 'package.module:create_loaders'")
    module_name, function_name = factory.split(":", 1)
    return getattr(importlib.import_module(module_name), function_name)(**(kwargs or {}))
