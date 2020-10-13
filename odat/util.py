import os
import importlib_resources  # replace when moving to 3.9+


class ModuleNotFoundIgnore:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is ModuleNotFoundError:
            pass
        return True


class IgnoreAllErrors:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True


def is_submodule_path(path):
    path = str(path)
    return path.endswith('.py')


def module_name(path):
    name, ext = os.path.splitext(os.path.basename(path))
    return name


def submodules_of(pkg, include_init=True):
    f = importlib_resources.files(pkg)
    g = map(module_name, filter(is_submodule_path, f.iterdir()))
    if include_init:
        return g
    else:
        return filter(lambda name: name != '__init__', g)
