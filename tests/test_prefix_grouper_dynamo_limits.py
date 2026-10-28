"""prefix_grouper_kernel raises dynamo's recompile limits at import under either config name.

torch < 2.7 only has cache_size_limit / accumulated_cache_size_limit, and dynamo's config refuses
unknown names with AttributeError. Both lazy importers swallow that and turn PrefixGrouper off, so
the import must not depend on the torch >= 2.7 names.
"""

import importlib.util
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch.nn.attention.flex_attention")

KERNEL = (
    pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "utils" / "prefix_grouper_kernel.py"
)
NEW_NAMES = ("recompile_limit", "accumulated_recompile_limit")


class _OldDynamoConfig:
    """torch 2.6 view of torch._dynamo.config: no recompile_limit names, unknown writes raise."""

    def __init__(self, real):
        object.__setattr__(self, "_real", real)
        object.__setattr__(
            self, "_local", {"cache_size_limit": 8, "accumulated_cache_size_limit": 256}
        )

    def __getattr__(self, name):
        if name in NEW_NAMES:
            raise AttributeError(name)
        local = object.__getattribute__(self, "_local")
        if name in local:
            return local[name]
        return getattr(object.__getattribute__(self, "_real"), name)

    def __setattr__(self, name, value):
        local = object.__getattribute__(self, "_local")
        if name not in local:
            raise AttributeError(f"torch._dynamo.config.{name} does not exist")
        local[name] = value


def _load_kernel():
    name = "_pg_kernel_under_test"
    spec = importlib.util.spec_from_file_location(name, KERNEL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve their module through sys.modules
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_import_raises_the_old_names_on_torch_26_config(monkeypatch):
    fake = _OldDynamoConfig(torch._dynamo.config)
    monkeypatch.setattr(torch._dynamo, "config", fake)
    module = _load_kernel()
    assert callable(module.flex_shared_prefix_attention)
    assert fake.cache_size_limit == 256
    assert fake.accumulated_cache_size_limit == 2048


def test_import_raises_the_new_names_when_present(monkeypatch):
    config = torch._dynamo.config
    if not hasattr(config, "recompile_limit"):
        pytest.skip("torch < 2.7 has no recompile_limit")
    monkeypatch.setattr(config, "recompile_limit", 8)
    monkeypatch.setattr(config, "accumulated_recompile_limit", 256)
    _load_kernel()
    assert config.recompile_limit == 256
    assert config.accumulated_recompile_limit == 2048
    # Never lowers a limit the user already raised.
    monkeypatch.setattr(config, "recompile_limit", 4096)
    _load_kernel()
    assert config.recompile_limit == 4096
