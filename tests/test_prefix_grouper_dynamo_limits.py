# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""PrefixGrouper on torch < 2.7: kernel imports cleanly, stays off with one log, dynamo untouched."""

import importlib.util
import pathlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch.nn.attention.flex_attention")

UTILS = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "utils"
KERNEL = UTILS / "prefix_grouper_kernel.py"
PKG = "_pg_pkg_under_test"
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


def test_import_raises_the_old_names_when_only_they_exist(monkeypatch):
    fake = _OldDynamoConfig(torch._dynamo.config)
    monkeypatch.setattr(torch._dynamo, "config", fake)
    monkeypatch.setattr(torch, "__version__", "2.7.0")
    module = _load_kernel()
    assert callable(module.flex_shared_prefix_attention)
    assert fake.cache_size_limit == 256
    assert fake.accumulated_cache_size_limit == 2048


@pytest.mark.skipif(
    not hasattr(torch._dynamo.config, "recompile_limit"),
    reason = "torch < 2.7 has no recompile_limit; the old names are covered by the test above",
)
def test_import_raises_the_new_names_when_present(monkeypatch):
    config = torch._dynamo.config
    monkeypatch.setattr(config, "recompile_limit", 8)
    monkeypatch.setattr(config, "accumulated_recompile_limit", 256)
    _load_kernel()
    assert config.recompile_limit == 256
    assert config.accumulated_recompile_limit == 2048
    monkeypatch.setattr(config, "recompile_limit", 4096)
    _load_kernel()
    assert config.recompile_limit == 4096


def _load_prefix_grouper():
    """Load prefix_grouper.py + its kernel under a throwaway package (no full unsloth import)."""
    pkg = types.ModuleType(PKG)
    pkg.__path__ = [str(UTILS)]
    names = (PKG, f"{PKG}.prefix_grouper_kernel", f"{PKG}.prefix_grouper")
    sys.modules[PKG] = pkg
    try:
        name = f"{PKG}.prefix_grouper"
        spec = importlib.util.spec_from_file_location(name, UTILS / "prefix_grouper.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        for n in names:
            sys.modules.pop(n, None)
    return module


@pytest.mark.parametrize(
    "version, ok",
    [
        ("2.6.0+cu124", False),
        ("2.5.1", False),
        ("1.13.1", False),
        ("2.7.0a0+git1234", True),
        ("2.7.1", True),
        ("2.10.0", True),
        ("3.0.0", True),
        ("not-a-version", True),
    ],
)
def test_torch_version_gate(version, ok):
    assert _load_kernel().torch_supports_prefix_grouper(version) is ok


def test_torch_26_import_is_clean_and_leaves_dynamo_alone(monkeypatch):
    fake = _OldDynamoConfig(torch._dynamo.config)
    monkeypatch.setattr(torch._dynamo, "config", fake)
    monkeypatch.setattr(torch, "__version__", "2.6.0+cu124")
    module = _load_kernel()
    assert module.TORCH_SUPPORTS_PREFIX_GROUPER is False
    assert fake.cache_size_limit == 8
    assert fake.accumulated_cache_size_limit == 256


def test_torch_26_prefix_grouper_off_with_one_log(monkeypatch, capsys):
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", "1")
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING", "1")
    monkeypatch.setattr(torch._dynamo, "config", _OldDynamoConfig(torch._dynamo.config))
    monkeypatch.setattr(torch, "__version__", "2.6.0+cu124")
    pg = _load_prefix_grouper()
    assert capsys.readouterr().out == ""
    assert pg.prefix_grouper_enabled() is False
    out = capsys.readouterr().out
    assert out.count("\n") == 1 and "PrefixGrouper is off on torch 2.6.0+cu124" in out
    assert pg.prefix_grouper_enabled() is False
    assert capsys.readouterr().out == ""


def test_torch_26_env_off_stays_quiet(monkeypatch, capsys):
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", "0")
    monkeypatch.setattr(torch, "__version__", "2.6.0")
    pg = _load_prefix_grouper()
    assert pg.prefix_grouper_enabled() is False
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("version", ["2.7.1", "2.8.0", "2.14.1"])
def test_torch_27_plus_unchanged(monkeypatch, capsys, version):
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", "1")
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING", "1")
    monkeypatch.setattr(torch, "__version__", version)
    config = torch._dynamo.config
    for name in ("recompile_limit", "accumulated_recompile_limit"):
        if hasattr(config, name):
            monkeypatch.setattr(config, name, getattr(config, name))
    pg = _load_prefix_grouper()
    assert pg.prefix_grouper_enabled() is True
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING", "0")
    assert _load_prefix_grouper().prefix_grouper_enabled() is False
    assert capsys.readouterr().out == ""
