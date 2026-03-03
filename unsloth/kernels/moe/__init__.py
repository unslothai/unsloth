import importlib
import sys


_MODULE_SUFFIXES = (
    "",
    ".autotune_cache",
    ".grouped_gemm",
    ".grouped_gemm.interface",
)


def _alias_module(suffix):
    legacy_name = f"{__name__}{suffix}"
    target_name = f"unsloth_zoo.kernels.moe{suffix}"
    try:
        module = importlib.import_module(target_name)
    except ModuleNotFoundError:
        return None
    sys.modules[legacy_name] = module
    return module


_canonical_module = _alias_module("")
autotune_cache = _alias_module(".autotune_cache")
grouped_gemm = _alias_module(".grouped_gemm")

for _suffix in _MODULE_SUFFIXES[1:]:
    _alias_module(_suffix)


def __getattr__(name):
    return getattr(_canonical_module, name)


__all__ = getattr(_canonical_module, "__all__", ())
