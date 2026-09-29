# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``/api/system.quantised_streaming``: whether group offload can stream torchao weights.

The picker offers MiniMax-H3's streamed 30 GiB tier only when this is true, so an install whose
diffusers predates torchao-aware group offload routes to the GGUF row instead of a refused load.
"""

import ast
import sys
import types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_MAIN = (_BACKEND / "main.py").read_text(encoding = "utf-8")


def _src(name: str) -> str:
    node = next(
        n for n in ast.walk(ast.parse(_MAIN)) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return ast.get_source_segment(_MAIN, node)


def test_the_bit_is_published_on_the_system_route():
    assert '"quantised_streaming": _quantised_streaming()' in _src("get_system_info")


def test_the_bit_defaults_to_unknown_until_resolved():
    assert "_quantised_streaming_capability: Optional[bool] = None" in _MAIN


@pytest.mark.parametrize("cached, expected", [(None, False), (False, False), (True, True)])
def test_the_polled_reader_answers_from_the_cache_without_importing(monkeypatch, cached, expected):
    """A poll never imports diffusers: with every import poisoned it still answers the cached bit."""
    for name in ("diffusers", "diffusers.hooks", "torch", "core.inference.diffusion_prequant"):
        monkeypatch.setitem(sys.modules, name, None)
    reader = _src("_quantised_streaming")
    assert "import" not in reader
    namespace: dict = {"_quantised_streaming_capability": cached}
    exec(reader, namespace)  # noqa: S102 -- the real body
    assert namespace["_quantised_streaming"]() is expected


@pytest.mark.parametrize("supported", [True, False])
def test_the_refresh_caches_the_prequant_verdict(monkeypatch, supported):
    fake = types.ModuleType("core.inference.diffusion_prequant")
    fake.torchao_group_offload_supported = lambda: supported
    monkeypatch.setitem(sys.modules, "core.inference.diffusion_prequant", fake)
    namespace: dict = {"_quantised_streaming_capability": None}
    exec(_src("_refresh_quantised_streaming_capability"), namespace)  # noqa: S102
    exec(_src("_quantised_streaming"), namespace)  # noqa: S102
    assert namespace["_refresh_quantised_streaming_capability"]() is supported
    assert namespace["_quantised_streaming_capability"] is supported
    assert namespace["_quantised_streaming"]() is supported


def test_the_post_warm_worker_resolves_it_behind_the_torch_guard():
    warm = _src("_post_warm_background_work")
    assert "_refresh_quantised_streaming_capability()" in warm
    guard = warm.index('if "torch" in sys.modules:')
    assert guard < warm.index("_refresh_quantised_streaming_capability()")


def _group_offloading(**attrs):
    hooks = types.ModuleType("diffusers.hooks")
    hooks.apply_group_offloading = lambda *a, **k: None
    go = types.ModuleType("diffusers.hooks.group_offloading")
    for name, value in attrs.items():
        setattr(go, name, value)
    hooks.group_offloading = go
    return hooks, go


@pytest.mark.parametrize(
    "attrs, expected",
    [
        ({"_is_torchao_tensor": lambda t: False, "_swap_torchao_tensor": lambda *a: None}, True),
        ({}, False),
    ],
)
def test_the_probe_follows_diffusers_group_offload(monkeypatch, attrs, expected):
    from core.inference.diffusion_prequant import torchao_group_offload_supported

    hooks, go = _group_offloading(**attrs)
    diffusers = types.ModuleType("diffusers")
    diffusers.hooks = hooks
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.hooks", hooks)
    monkeypatch.setitem(sys.modules, "diffusers.hooks.group_offloading", go)
    assert torchao_group_offload_supported() is expected
