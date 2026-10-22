"""Load bundled scenarios without the imp module removed in Python 3.12."""
import importlib.util
import os.path as osp


def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    spec = importlib.util.spec_from_file_location(osp.splitext(osp.basename(name))[0], pathname)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load scenario: {pathname}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
