"""Load a model factory from the inherited project using module:function syntax."""
import importlib


def create_model(factory, kwargs=None):
    if not factory or ":" not in factory:
        raise ValueError("Set model.factory to an import path like 'package.module:create_model'")
    module_name, function_name = factory.split(":", 1)
    fn = getattr(importlib.import_module(module_name), function_name)
    return fn(**(kwargs or {}))
