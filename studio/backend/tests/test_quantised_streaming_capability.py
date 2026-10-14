# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``/api/system.quantised_streaming``: whether group offload can stream torchao weights.

The picker offers MiniMax-H3's streamed 30 GiB tier only when this is true, so an install whose
diffusers predates torchao-aware group offload, or a GPU without INT8 cores, routes to the GGUF row
instead of a refused load.
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
    namespace: dict = {"_quantised_streaming_capability": cached, "sys": sys}
    exec(reader, namespace)  # noqa: S102 -- the real body
    assert namespace["_quantised_streaming"]() is expected


def _exec_refresh(
    monkeypatch,
    supported,
    device_count = 1,
):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: device_count > 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: device_count)
    fake = types.ModuleType("core.inference.video")
    fake.h3_streamed_int8_supported = supported
    monkeypatch.setitem(sys.modules, "core.inference.video", fake)
    device = types.ModuleType("core.inference.diffusion_device")
    device.resolve_diffusion_device_target = lambda ordinal = None: ordinal
    monkeypatch.setitem(sys.modules, "core.inference.diffusion_device", device)
    namespace: dict = {"_quantised_streaming_capability": None, "sys": sys, "Any": object}
    exec(_src("_probe_quantised_streaming"), namespace)  # noqa: S102
    exec(_src("_refresh_quantised_streaming_capability"), namespace)  # noqa: S102
    return namespace


@pytest.mark.parametrize("supported", [True, False])
def test_the_refresh_caches_the_prequant_verdict(monkeypatch, supported):
    namespace = _exec_refresh(monkeypatch, lambda target = None: supported)
    exec(_src("_quantised_streaming"), namespace)  # noqa: S102
    assert namespace["_refresh_quantised_streaming_capability"]() is supported
    assert namespace["_quantised_streaming_capability"] is supported
    assert namespace["_quantised_streaming"]() is supported


@pytest.mark.parametrize("loaded", [True, False])
def test_a_cold_warm_resolves_it_once_a_load_has_loaded_the_stack(monkeypatch, loaded):
    """UNSLOTH_STUDIO_DISABLE_TORCH_WARM=1 skips the warm refresh; a later load must still publish the bit."""
    for name in _STREAMING_MODULES:
        if loaded:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        else:
            monkeypatch.delitem(sys.modules, name, raising = False)
    calls: list = []
    namespace: dict = {"_quantised_streaming_capability": None, "sys": sys}
    exec(_src("_quantised_streaming"), namespace)  # noqa: S102
    namespace["_refresh_quantised_streaming_capability"] = lambda: calls.append(1) or True
    assert namespace["_quantised_streaming"]() is loaded
    assert calls == ([1] if loaded else [])


def test_an_image_load_alone_resolves_it(monkeypatch):
    """An early image load makes the post-warm probe stand down and never loads core.inference.video,
    so requiring it kept the streamed tier hidden for the rest of the process."""
    for name in _STREAMING_MODULES:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.delitem(sys.modules, "core.inference.video", raising = False)
    namespace: dict = {"_quantised_streaming_capability": None, "sys": sys}
    exec(_src("_quantised_streaming"), namespace)  # noqa: S102
    namespace["_refresh_quantised_streaming_capability"] = lambda: True
    assert namespace["_quantised_streaming"]() is True


_STREAMING_MODULES = (
    "torch",
    "torchao",
    "diffusers",
    "diffusers.hooks",
    "diffusers.hooks.group_offloading",
)


@pytest.mark.parametrize("half_built", _STREAMING_MODULES)
def test_the_poll_waits_until_a_loads_import_has_finished(monkeypatch, half_built):
    """Probing while a load still imports diffusers failed that load ("Failed to import
    diffusers.loaders.peft"), so a module still initialising defers the refresh."""
    for name in _STREAMING_MODULES:
        module = types.ModuleType(name)
        module.__spec__ = types.SimpleNamespace(_initializing = name == half_built)
        monkeypatch.setitem(sys.modules, name, module)
    calls: list = []
    namespace: dict = {"_quantised_streaming_capability": None, "sys": sys}
    exec(_src("_quantised_streaming"), namespace)  # noqa: S102
    namespace["_refresh_quantised_streaming_capability"] = lambda: calls.append(1) or True
    assert namespace["_quantised_streaming"]() is False
    assert calls == []
    sys.modules[half_built].__spec__._initializing = False
    assert namespace["_quantised_streaming"]() is True
    assert calls == [1]


def test_a_mixed_host_offers_the_tier_only_when_every_card_qualifies(monkeypatch):
    """A load can be pinned to any card; one resolving to float16 would keep the bf16 denoiser."""
    namespace = _exec_refresh(monkeypatch, lambda target = None: target != 1, device_count = 2)
    assert namespace["_refresh_quantised_streaming_capability"]() is False
    namespace = _exec_refresh(monkeypatch, lambda target = None: True, device_count = 2)
    assert namespace["_refresh_quantised_streaming_capability"]() is True


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


@pytest.mark.parametrize(
    "device, dtype_name, streaming, readable, expected",
    [
        ("cuda", "bfloat16", True, True, True),
        # Pre-Ampere NVIDIA resolves to float16, and its INT8 path is not supported.
        ("cuda", "float16", True, True, False),
        ("cuda", "bfloat16", False, True, False),
        # An unreadable hosted INT8 keeps auto on bf16, whose load the tier cannot hold.
        ("cuda", "bfloat16", True, False, False),
        ("mps", "bfloat16", True, True, False),
        ("cpu", "float32", True, True, False),
    ],
)
def test_the_streamed_tier_needs_an_int8_capable_gpu(
    monkeypatch, device, dtype_name, streaming, readable, expected
):
    torch = pytest.importorskip("torch")
    from core.inference import diffusion_prequant, video

    monkeypatch.setattr(diffusion_prequant, "torchao_group_offload_supported", lambda: streaming)
    monkeypatch.setattr(
        diffusion_prequant, "restricted_prequant_load_supported", lambda *a, **k: readable
    )
    target = types.SimpleNamespace(device = device, dtype = getattr(torch, dtype_name))
    assert video.h3_streamed_int8_supported(target) is expected
