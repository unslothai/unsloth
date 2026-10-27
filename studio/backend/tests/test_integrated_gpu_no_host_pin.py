# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""No page-locked host memory on an integrated GPU (Strix Halo, Jetson, DGX Spark), as ComfyUI does.

The device is stubbed through ``torch.cuda.get_device_properties(...).is_integrated``: 1 must reach no pinning call
at any diffusion / video site, 0 (a discrete card) must reach the same calls as before. GPU-free throughout."""

from __future__ import annotations

import ast
import inspect
import os
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_memory as dm  # noqa: E402

# Imported before any test stubs the device properties: importing diffusers under the stub reads fields it lacks.
pytest.importorskip("diffusers.hooks.group_offloading")

BACKEND = Path(__file__).resolve().parents[1]


class _Stop(Exception):
    pass


def _props(integrated: bool, arch: str = "") -> types.SimpleNamespace:
    flag = 1 if integrated else 0
    return types.SimpleNamespace(
        is_integrated = flag,
        integrated = flag,
        name = "Fake GPU",
        gcnArchName = arch,
        total_memory = 16 << 30,
        major = 9,
        minor = 0,
    )


def _forget_answers() -> None:
    try:
        from core.inference import host_pinning
    except ImportError:
        return
    host_pinning.reset_cache()


@pytest.fixture
def gpu(monkeypatch):
    """``gpu(True)`` stubs an integrated device, ``gpu(False)`` a discrete one."""

    def make(integrated: bool, arch: str = "") -> None:
        props = _props(integrated, arch)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a, **k: props)
        _forget_answers()

    yield make
    _forget_answers()


@pytest.fixture
def pins(monkeypatch):
    """Every pinning request (``torch.empty(pin_memory=True)``, ``Tensor.pin_memory()``), served as plain host memory."""
    calls: list = []
    real_empty = torch.empty

    def empty(*args, **kwargs):
        if kwargs.pop("pin_memory", False):
            calls.append("empty")
        return real_empty(*args, **kwargs)

    def pin_memory(self, *args, **kwargs):
        calls.append("pin_memory")
        return self.clone()

    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch.Tensor, "pin_memory", pin_memory)
    return calls


@pytest.mark.parametrize("integrated", [True, False])
def test_the_predicate_reads_the_driver_flag(gpu, integrated):
    from core.inference import host_pinning

    gpu(integrated)
    assert host_pinning.device_is_integrated() is integrated
    assert host_pinning.host_pinning_allowed() is (not integrated)
    assert host_pinning.host_pinning_allowed("cuda:0") is (not integrated)
    assert host_pinning.host_pinning_allowed(torch.device("cuda")) is (not integrated)


def test_the_predicate_keeps_the_discrete_answer_without_a_gpu_or_on_another_device(
    monkeypatch, gpu
):
    from core.inference import host_pinning

    gpu(True)
    assert host_pinning.host_pinning_allowed("cpu") is True
    assert host_pinning.host_pinning_allowed("mps") is True
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    host_pinning.reset_cache()
    assert host_pinning.host_pinning_allowed() is True

    def boom(*a, **k):
        raise RuntimeError("no driver")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", boom)
    host_pinning.reset_cache()
    assert host_pinning.host_pinning_allowed() is True


@pytest.mark.parametrize(
    "arch, flag, integrated",
    [("gfx1151", 1, True), ("gfx1151", 0, True), ("gfx1100", 0, False), ("gfx942", 0, False)],
)
def test_rocm_uses_studios_apu_classifier(monkeypatch, gpu, arch, flag, integrated):
    from core.inference import host_pinning

    gpu(bool(flag), arch)
    monkeypatch.setattr(torch.version, "hip", "6.4.0", raising = False)
    host_pinning.reset_cache()
    assert host_pinning.device_is_integrated() is integrated


def test_the_answer_is_cached_per_device(monkeypatch, gpu):
    from core.inference import host_pinning

    gpu(True)
    asked: list = []
    props = _props(True)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda i, *a, **k: (asked.append(i), props)[1]
    )
    for _ in range(3):
        host_pinning.host_pinning_allowed("cuda:0")
    assert asked == [0]


def _host_of(module):
    return {name: p.data for name, p in module.named_parameters()}


@pytest.mark.parametrize("integrated", [True, False])
def test_keep_cpu_weights_are_not_pinned_on_an_integrated_gpu(monkeypatch, gpu, pins, integrated):
    gpu(integrated)
    monkeypatch.delenv(dm.OFFLOAD_PIN_ENV, raising = False)
    monkeypatch.setattr(dm, "_available_system_memory_mib", lambda: 64 << 10)
    monkeypatch.setattr(dm, "_host_ram_capacity_mib", lambda: 128 << 10)
    lin = torch.nn.Linear(8, 8)
    pinned = dm._pin_host_weights(lin, _host_of(lin))
    if integrated:
        assert pinned == 0 and pins == []
    else:
        assert pinned > 0 and pins


@pytest.mark.parametrize("integrated", [True, False])
def test_the_streamed_pin_plan_pins_nothing_on_an_integrated_gpu(monkeypatch, gpu, integrated):
    gpu(integrated)
    monkeypatch.delenv(dm.GROUP_OFFLOAD_PIN_ENV, raising = False)
    monkeypatch.setattr(dm, "_pinned_memory_capped", lambda: False)
    monkeypatch.setattr(dm, "_pin_budget_mib", lambda: 1 << 20)
    want = (False, False) if integrated else (True, True)
    assert dm._streamed_pin_plan(1024, 1024) == want


@pytest.mark.parametrize("integrated", [True, False])
def test_background_and_registered_pins_are_off_on_an_integrated_gpu(monkeypatch, gpu, integrated):
    gpu(integrated)
    monkeypatch.delenv(dm.BACKGROUND_PIN_ENV, raising = False)
    monkeypatch.delenv(dm.FAST_PIN_ENV, raising = False)
    monkeypatch.setattr(dm.sys, "platform", "linux")
    monkeypatch.setattr(dm, "_pinned_memory_capped", lambda: False)
    registered: list = []
    cudart = types.SimpleNamespace(
        cudaHostRegister = lambda *a: registered.append(a) or 0,
        cudaHostUnregister = lambda *a: 0,
    )
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)
    assert dm._background_pin_enabled() is (not integrated)
    assert dm._fast_pin_supported() is (not integrated)
    assert registered == []


@pytest.mark.parametrize("integrated", [True, False])
def test_the_top_level_group_is_not_pinned_on_an_integrated_gpu(monkeypatch, gpu, integrated):
    gpu(integrated)
    monkeypatch.delenv(dm.PIN_TOP_GROUP_ENV, raising = False)
    looked: list = []
    module = types.SimpleNamespace(
        _diffusers_hook = types.SimpleNamespace(get_hook = lambda name: looked.append(name))
    )
    assert dm._pin_top_level_group(module) is False
    assert bool(looked) is (not integrated)


def _record_group_offload(monkeypatch):
    from diffusers import hooks as dh

    real = dh.apply_group_offloading
    seen: list = []

    def apply_group_offloading(module, **kwargs):
        seen.append(kwargs)

    apply_group_offloading.__signature__ = inspect.signature(real)
    monkeypatch.setattr(dh, "apply_group_offloading", apply_group_offloading)
    monkeypatch.setattr(dm, "install_group_offload_buffer_restore", lambda: None)
    monkeypatch.setattr(dm, "install_group_offload_hooks_eager", lambda: None)
    return seen


@pytest.mark.parametrize("integrated", [True, False])
def test_streamed_text_encoders_do_not_use_a_pinning_stream_on_an_integrated_gpu(
    monkeypatch, gpu, integrated
):
    pytest.importorskip("diffusers")
    gpu(integrated)
    seen = _record_group_offload(monkeypatch)
    monkeypatch.setattr(dm, "_pin_vision_embedding_device", lambda module: 0)
    pipe = types.SimpleNamespace(components = {"text_encoder": torch.nn.Linear(4, 4)})
    assert dm.stream_text_encoders_onto(pipe, "cuda") == ["text_encoder"]
    (kwargs,) = seen
    assert kwargs["use_stream"] is (not integrated)
    if integrated:
        assert "record_stream" not in kwargs and "non_blocking" not in kwargs


@pytest.mark.parametrize("integrated", [True, False])
def test_the_controlnet_offload_streams_without_pinning_on_an_integrated_gpu(
    monkeypatch, gpu, integrated
):
    pytest.importorskip("diffusers")
    from core.inference import diffusion

    gpu(integrated)
    seen = _record_group_offload(monkeypatch)
    assert diffusion._offload_controlnet_module(torch.nn.Linear(4, 4), "cuda", None) is True
    assert seen[0]["use_stream"] is (not integrated)


@pytest.mark.parametrize("integrated", [True, False])
def test_the_fast_upload_ring_is_skipped_on_an_integrated_gpu(monkeypatch, gpu, integrated):
    from core.inference import diffusion_fast_load as fl

    gpu(integrated)
    monkeypatch.delenv(fl.FAST_UPLOAD_ENV, raising = False)
    asked: list = []
    monkeypatch.setattr(
        fl, "_upload_candidates", lambda modules, device: asked.append(device) or []
    )
    with fl.fast_upload([torch.nn.Linear(2, 2)], "cuda") as staged:
        assert staged == 0
    assert bool(asked) is (not integrated)


@pytest.mark.parametrize("integrated", [True, False])
def test_the_encoder_prefetch_ring_is_skipped_on_an_integrated_gpu(monkeypatch, gpu, integrated):
    from core.inference import diffusion_small_host as sh

    gpu(integrated)
    monkeypatch.delenv(dm.GROUP_OFFLOAD_PIN_ENV, raising = False)
    monkeypatch.setattr(sh, "encoder_prefetch_disabled", lambda: False)
    asked: list = []
    monkeypatch.setattr(dm, "_offload_groups", lambda module: asked.append(module) or [])
    assert sh.install_encoder_prefetch(torch.nn.Linear(2, 2), "cuda") == 0
    assert bool(asked) is (not integrated)


@pytest.mark.parametrize("integrated", [True, False])
def test_the_torchao_top_group_is_not_adopted_on_an_integrated_gpu(monkeypatch, gpu, integrated):
    from core.inference import diffusion_offload_prefetch as op

    gpu(integrated)
    monkeypatch.delenv(dm.GROUP_OFFLOAD_PIN_ENV, raising = False)
    monkeypatch.delenv(dm.PIN_TOP_GROUP_ENV, raising = False)
    asked: list = []
    monkeypatch.setattr(op, "_go", lambda: asked.append(1) or types.SimpleNamespace())
    assert op._adopt_top_group(types.SimpleNamespace(), object()) is None
    assert bool(asked) is (not integrated)


@pytest.mark.parametrize("integrated", [True, False])
def test_video_frame_readback_is_pageable_on_an_integrated_gpu(monkeypatch, gpu, integrated):
    from core.inference import video_frames

    gpu(integrated)
    asked: list = []
    real_empty = torch.empty

    def empty(*args, **kwargs):
        asked.append(bool(kwargs.pop("pin_memory", False)))
        if len(asked) > 1 or not asked[0]:
            raise _Stop
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    frames = types.SimpleNamespace(
        device = torch.device("cuda", 0), numel = lambda: 6, shape = (1, 3, 1, 2)
    )
    if integrated:
        with pytest.raises(_Stop):
            video_frames.device_uint8(frames)
        assert asked == [False]
    else:
        with pytest.raises(Exception):  # noqa: B017 - the fake frames cannot be sliced
            video_frames.device_uint8(frames)
        assert asked[0] is True


@pytest.mark.parametrize("integrated", [True, False])
def test_the_ltx2_direct_load_staging_is_pageable_on_an_integrated_gpu(
    monkeypatch, gpu, integrated
):
    from core.inference import video_ltx2

    gpu(integrated)
    asked: list = []

    def empty(*args, **kwargs):
        asked.append(bool(kwargs.get("pin_memory", False)))
        raise _Stop

    monkeypatch.setattr(torch, "empty", empty)
    with pytest.raises(_Stop):
        video_ltx2._copy_file_chunks("unused", [(0, 4, None, 0)], "cuda", 4, 1, 1)
    assert asked == [not integrated]


class _FakeHost:
    def __init__(self, calls):
        self.calls = calls

    def pin_memory(self):
        self.calls.append("pin_memory")
        return self

    def to(self, *a, **k):
        return self

    def view(self, *a):
        return self

    def copy_(self, *a, **k):
        return self


def _fake_torch(calls):
    cuda = types.SimpleNamespace(
        Stream = lambda device = None: types.SimpleNamespace(wait_event = lambda e: None),
        Event = lambda: types.SimpleNamespace(record = lambda *a: None),
        current_stream = lambda device = None: object(),
        stream = lambda s: __import__("contextlib").nullcontext(),
    )

    def empty(*a, **k):
        if k.get("pin_memory"):
            calls.append("empty")
        return _FakeHost(calls)

    return types.SimpleNamespace(
        tensor = lambda *a, **k: _FakeHost(calls), float32 = None, uint8 = None, empty = empty, cuda = cuda
    )


@pytest.mark.parametrize("integrated", [True, False])
def test_stream_encode_host_buffers_are_pageable_on_an_integrated_gpu(monkeypatch, gpu, integrated):
    from core.inference import video_stream_encode as vse

    gpu(integrated)
    calls: list = []
    fake = types.SimpleNamespace(_torch = _fake_torch(calls), _pixel = ((0.5,), (0.5,)))
    vse.H3StreamEncode._pixel_tensors(fake, torch.device("cuda", 0))
    assert bool(calls) is (not integrated)

    calls.clear()

    class _X:
        device = torch.device("cuda", 0)
        shape = (1, 1, 1, 3)

        def to(self, *a, **k):
            return self

        def __mul__(self, other):
            return self

        def round_(self):
            return self

        def permute(self, *a):
            return self

        def contiguous(self):
            return self

        def record_stream(self, s):
            pass

    import queue

    sess = types.SimpleNamespace(
        _torch = fake._torch, _side = None, chunks = [], _thread = object(), _queue = queue.SimpleQueue()
    )
    vse.StreamEncode._push_uint8(sess, _X())
    assert calls == ([] if integrated else ["empty"])


@pytest.mark.parametrize("integrated", [True, False])
def test_the_h3_pin_gate_refuses_an_integrated_gpu_even_when_forced(monkeypatch, gpu, integrated):
    from core.inference import video_minimax_h3_te as te

    gpu(integrated)
    monkeypatch.setenv(dm.GROUP_OFFLOAD_PIN_ENV, "1")
    assert te.h3_te_pin_allowed(1 << 20) is (not integrated)


# Sites with no gate of their own, each reached only through one that asks host_pinning_allowed.
_REACHED_THROUGH_A_GATE = {
    ("core/inference/diffusion_fast_load.py", "_ring_copy"): "fast_upload",
    (
        "core/inference/diffusion_memory.py",
        "install_torchao_v1_int8_pin_ops._pinned",
    ): "torchao's pin op: runs when a gated site pins",
    ("core/inference/diffusion_memory.py", "_RegisteredHostBuffer.register"): "_fast_pin_supported",
    (
        "core/inference/diffusion_memory.py",
        "_GroupPinner._fast_result",
    ): "_background_pin_enabled / use_stream",
    (
        "core/inference/diffusion_memory.py",
        "_GroupPinner._run",
    ): "_background_pin_enabled / use_stream",
    (
        "core/inference/diffusion_offload_prefetch.py",
        "GroupPrefetcher._issue",
    ): "streamed groups only (use_stream)",
    (
        "core/inference/diffusion_pinned_arena.py",
        "_alloc_slab",
    ): "group offload with use_stream only",
    (
        "core/inference/diffusion_pinned_arena.py",
        "PinnedArena.pin_plain",
    ): "group offload with use_stream only",
    (
        "core/inference/diffusion_segment.py",
        "_HostWeights._repack",
    ): "MaskSegmenter park passes pin=",
    (
        "core/inference/diffusion_small_host.py",
        "_EncoderPrefetcher._ensure_stage",
    ): "install_encoder_prefetch",
    (
        "core/inference/video_minimax_h3_te.py",
        "pin_module_in_place",
    ): "h3_te_pin_allowed / stream_h3_text_encoder",
    (
        "core/inference/video_minimax_h3_te.py",
        "_pinned_arena",
    ): "pin_module_in_place",
    (
        "core/inference/video_minimax_h3_residency.py",
        "_pin_host_tensor",
    ): "_H3BlockPinner._run only, built by defer_streamed_block_pins behind _background_pin_enabled / "
    "_fast_pin_supported / h3_te_pin_allowed (the admit too), all on the target device",
    ("core/inference/diffusion_preview.py", "_device_map"): "exempt: tiny, async preview",
    (
        "core/inference/diffusion_preview.py",
        "LatentPreviewer.__init__",
    ): "exempt: tiny, async preview",
}


def _pin_sites():
    for path in sorted((BACKEND / "core").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding = "utf-8"))
        rel = path.relative_to(BACKEND).as_posix()

        def visit(node, chain):
            for child in ast.iter_child_nodes(node):
                here = chain
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    here = chain + [child]
                if isinstance(child, ast.Call):
                    func = child.func
                    hit = isinstance(func, ast.Attribute) and func.attr in (
                        "pin_memory",
                        "cudaHostRegister",
                    )
                    for kw in child.keywords:
                        if kw.arg == "pin_memory" and not (
                            isinstance(kw.value, ast.Constant) and kw.value.value is False
                        ):
                            hit = True
                    if hit:
                        yield rel, ".".join(n.name for n in here), here, child.lineno
                yield from visit(child, here)

        yield from visit(tree, [])


def test_every_pinning_site_asks_the_integrated_gpu_predicate_or_is_reached_through_one():
    ungated = []
    for rel, qualname, chain, line in _pin_sites():
        funcs = [n for n in chain if not isinstance(n, ast.ClassDef)]
        if any("host_pinning_allowed" in ast.unparse(n) for n in funcs):
            continue
        if (rel, qualname) in _REACHED_THROUGH_A_GATE:
            continue
        ungated.append(f"{rel}:{line} {qualname}")
    assert not ungated, "pinning without asking host_pinning_allowed: " + ", ".join(ungated)


def test_every_cuda_group_offload_stream_asks_the_predicate():
    # diffusers pins every streamed group's host copy, so use_stream is itself a pinning decision.
    loose = []
    for path in sorted((BACKEND / "core" / "inference").glob("*.py")):
        tree = ast.parse(path.read_text(encoding = "utf-8"))
        for node in ast.walk(tree):
            value = None
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "use_stream" for t in node.targets
            ):
                value = node.value
            elif isinstance(node, ast.keyword) and node.arg == "use_stream":
                value = node.value
            if value is None:
                continue
            text = ast.unparse(value)
            if "'cuda'" in text and "host_pinning_allowed" not in text and "pinning" not in text:
                loose.append(f"{path.name}:{getattr(value, 'lineno', '?')} {text}")
    assert not loose, loose


def test_the_h3_block_pin_helper_is_reached_only_through_the_gated_deferral():
    # _pin_host_tensor has no gate of its own: its one caller is _H3BlockPinner, built only by the deferral.
    tree = ast.parse(
        (BACKEND / "core/inference/video_minimax_h3_residency.py").read_text(encoding = "utf-8")
    )
    funcs = {
        n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def users(name):
        return sorted(
            f
            for f, node in funcs.items()
            if f != name and any(isinstance(x, ast.Name) and x.id == name for x in ast.walk(node))
        )

    assert users("_pin_host_tensor") == ["_h3_block_pinner_cls", "_run"]
    assert users("_h3_block_pinner_cls") == ["defer_streamed_block_pins"]
    gate = ast.unparse(funcs["defer_streamed_block_pins"])
    for asked in (
        "_background_pin_enabled(device)",
        "_fast_pin_supported(device)",
        "h3_te_pin_allowed(group_payload_bytes(lazy[0]), device)",
    ):
        assert asked in gate
    assert gate.index("_fast_pin_supported(device)") < gate.index("_h3_block_pinner_cls()")


# Mixed hosts: the current device (0) and the target (1) differ, one integrated and one discrete.
@pytest.fixture
def mixed_gpu(monkeypatch):
    """``mixed_gpu(current_integrated, target_integrated)``: device 0 is current, device 1 is the target."""

    def make(current_integrated: bool, target_integrated: bool) -> None:
        props = {0: _props(current_integrated), 1: _props(target_integrated)}
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i, *a, **k: props[int(i)])
        _forget_answers()

    yield make
    _forget_answers()


_MIXED = [(True, False), (False, True)]
_TARGET = "cuda:1"


@pytest.mark.parametrize("current_integrated, target_integrated", _MIXED)
def test_keep_cpu_weights_follow_the_target_device(
    monkeypatch, mixed_gpu, pins, current_integrated, target_integrated
):
    mixed_gpu(current_integrated, target_integrated)
    monkeypatch.delenv(dm.OFFLOAD_PIN_ENV, raising = False)
    monkeypatch.setattr(dm, "_available_system_memory_mib", lambda: 64 << 10)
    monkeypatch.setattr(dm, "_host_ram_capacity_mib", lambda: 128 << 10)
    lin = torch.nn.Linear(8, 8)
    pinned = dm._pin_host_weights(lin, _host_of(lin), device = _TARGET)
    if target_integrated:
        assert pinned == 0 and pins == []
    else:
        assert pinned > 0 and pins


@pytest.mark.parametrize("current_integrated, target_integrated", _MIXED)
def test_pin_planners_follow_the_target_device(
    monkeypatch, mixed_gpu, current_integrated, target_integrated
):
    from core.inference import video_minimax_h3_te as te

    mixed_gpu(current_integrated, target_integrated)
    allowed = not target_integrated
    monkeypatch.delenv(dm.BACKGROUND_PIN_ENV, raising = False)
    monkeypatch.delenv(dm.FAST_PIN_ENV, raising = False)
    monkeypatch.delenv(dm.PARTIAL_ENCODER_PIN_ENV, raising = False)
    monkeypatch.delenv(dm.GROUP_OFFLOAD_PIN_ENV, raising = False)
    monkeypatch.setattr(dm.sys, "platform", "linux")
    monkeypatch.setattr(dm, "_pinned_memory_capped", lambda: False)
    monkeypatch.setattr(dm, "_pin_budget_mib", lambda: 1 << 20)
    cudart = types.SimpleNamespace(cudaHostRegister = lambda *a: 0, cudaHostUnregister = lambda *a: 0)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)
    assert dm._streamed_pin_plan(1024, 1024, device = _TARGET) == (allowed, allowed)
    assert (dm._streamed_pin_leftover_mib(0, False, _TARGET) > 0) is allowed
    assert dm._background_pin_enabled(_TARGET) is allowed
    assert dm._fast_pin_supported(_TARGET) is allowed
    assert te.h3_te_pin_allowed(1 << 20, _TARGET) is allowed
    pipe = types.SimpleNamespace(**{dm._ENCODER_PIN_REQUEST_ATTR: True})
    assert dm._encoder_background_pin_wanted(pipe, _TARGET) is allowed
    # forced on: the user's switch never overrides an integrated target
    monkeypatch.setenv(dm.GROUP_OFFLOAD_PIN_ENV, "1")
    assert dm._streamed_pin_plan(1024, 1024, device = _TARGET) == (allowed, allowed)
    assert te.h3_te_pin_allowed(1 << 20, _TARGET) is allowed


@pytest.mark.parametrize("current_integrated, target_integrated", _MIXED)
def test_top_level_group_pins_follow_the_target_device(
    monkeypatch, mixed_gpu, current_integrated, target_integrated
):
    from core.inference import diffusion_offload_prefetch as op

    mixed_gpu(current_integrated, target_integrated)
    monkeypatch.delenv(dm.PIN_TOP_GROUP_ENV, raising = False)
    monkeypatch.delenv(dm.GROUP_OFFLOAD_PIN_ENV, raising = False)
    looked: list = []
    module = types.SimpleNamespace(
        _diffusers_hook = types.SimpleNamespace(get_hook = lambda name: looked.append(name))
    )
    assert dm._pin_top_level_group(module, device = _TARGET) is False
    assert bool(looked) is (not target_integrated)
    asked: list = []
    monkeypatch.setattr(op, "_go", lambda: asked.append(1) or types.SimpleNamespace())
    assert op._adopt_top_group(types.SimpleNamespace(), object(), device = _TARGET) is None
    assert bool(asked) is (not target_integrated)


@pytest.mark.parametrize("current_integrated, target_integrated", _MIXED)
def test_streamed_text_encoders_pin_by_the_target_device(
    monkeypatch, mixed_gpu, current_integrated, target_integrated
):
    pytest.importorskip("diffusers")
    mixed_gpu(current_integrated, target_integrated)
    monkeypatch.setenv(dm.GROUP_OFFLOAD_PIN_ENV, "1")
    seen = _record_group_offload(monkeypatch)
    monkeypatch.setattr(dm, "_pin_vision_embedding_device", lambda module: 0)
    pipe = types.SimpleNamespace(components = {"text_encoder": torch.nn.Linear(4, 4)})
    assert dm.stream_text_encoders_onto(pipe, _TARGET) == ["text_encoder"]
    (kwargs,) = seen
    assert kwargs["use_stream"] is (not target_integrated)
    if not target_integrated:
        # a forced pin on a discrete target pins up front instead of re-pinning on every onload
        assert kwargs["low_cpu_mem_usage"] is False


def test_only_deviceless_sites_ask_the_predicate_without_a_device():
    # Every other site knows its onload target; asking the current device there misclassifies mixed hosts.
    allowed = {
        # probes and keys the current device by contract
        ("core/inference/diffusion_memory.py", "_offload_cost_measurements"),
        # device == "cuda" (the current device) is checked first
        ("core/training/diffusion_dit_trainer.py", "_build_latent_cache"),
        ("core/training/diffusion_dit_trainer.py", "_load_warm_conditioning"),
        ("core/training/diffusion_lora_trainer.py", "_build_sdxl_latent_cache"),
    }
    bare = set()
    for path in sorted((BACKEND / "core").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding = "utf-8"))
        rel = path.relative_to(BACKEND).as_posix()
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "host_pinning_allowed"
                    and not node.args
                    and not node.keywords
                ):
                    bare.add((rel, fn.name))
    assert bare <= allowed, sorted(bare - allowed)
