# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermetic CPU tests for dual-GPU CFG branch parallelism (``diffusion_cfg_parallel``).

torch is stubbed via ``sys.modules`` (the module imports it lazily), the DiT modules are
fakes that record calls, and the guider is a plain namespace -- so the gating matrix, the
proxy's routing/fan-out semantics, the per-generation dispatch policy, and teardown are
all exercised without a GPU, a replica download, or diffusers."""

from __future__ import annotations

import contextlib
import sys
import types

import pytest

from core.inference.diffusion_cfg_parallel import (
    CFG_PARALLEL_AUTO,
    CFG_PARALLEL_OFF,
    CFG_PARALLEL_ON,
    CFGParallelProxy,
    _pick_secondary_device,
    maybe_enable_cfg_parallel,
    normalize_cfg_parallel,
    teardown_cfg_parallel,
)


# ── normalisation ─────────────────────────────────────────────────────────────────
def test_normalize_unset_and_auto():
    for value in (None, "", "  ", "auto", "AUTO"):
        assert normalize_cfg_parallel(value) == CFG_PARALLEL_AUTO


def test_normalize_modes_and_casing():
    assert normalize_cfg_parallel("off") == CFG_PARALLEL_OFF
    assert normalize_cfg_parallel("none") == CFG_PARALLEL_OFF
    assert normalize_cfg_parallel(" ON ") == CFG_PARALLEL_ON


def test_normalize_rejects_unknown():
    with pytest.raises(ValueError):
        normalize_cfg_parallel("both")


# ── fakes ─────────────────────────────────────────────────────────────────────────
class _FakeDevice:
    def __init__(self, type_ = "cuda", index = 0):
        self.type = type_
        self.index = index


class _FakeTensor:
    """Just enough tensor for the proxy's _move / guider resolve paths."""

    def __init__(self, device, tag = "t", nbytes = 8):
        self.device = device
        self.tag = tag
        self._nbytes = nbytes

    def numel(self):
        return self._nbytes

    def element_size(self):
        return 1

    def to(self, device, non_blocking = False):
        return _FakeTensor(device, tag = self.tag, nbytes = self._nbytes)


class _FakeDiT:
    def __init__(self, device_index = 0, fail_enable = False):
        self._device = _FakeDevice(index = device_index)
        self.fail_enable = fail_enable
        self.enabled_with = None
        self.disables = 0
        self.resets = 0
        self.contexts: list = []
        self.calls: list = []
        self._mods = [self, types.SimpleNamespace(name = f"block{device_index}")]

    def parameters(self):
        return iter([types.SimpleNamespace(
            numel = lambda: 100, element_size = lambda: 2, device = self._device
        )])

    def modules(self):
        return list(self._mods)

    def enable_cache(self, config):
        if self.fail_enable:
            raise RuntimeError("replica enable boom")
        self.enabled_with = config

    def disable_cache(self):
        self.disables += 1

    def _reset_stateful_cache(self):
        self.resets += 1

    @contextlib.contextmanager
    def cache_context(self, name):
        self.contexts.append(name)
        yield

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return (_FakeTensor(self._device, tag = "pred"),)


def _stub_torch(monkeypatch, *, device_count = 2, free = None):
    torch = types.ModuleType("torch")
    torch.Tensor = _FakeTensor
    free = free if free is not None else {}

    def _mem_get_info(idx):
        return free.get(idx, (64 << 30, 80 << 30))

    torch.cuda = types.SimpleNamespace(
        is_available = lambda: device_count > 0,
        device_count = lambda: device_count,
        mem_get_info = _mem_get_info,
        empty_cache = lambda: None,
    )
    torch.inference_mode = contextlib.nullcontext
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def _make_proxy(monkeypatch, *, compiled = False, explicit_on = False, fail_enable = False):
    _stub_torch(monkeypatch)
    primary = _FakeDiT(device_index = 0)
    replica = _FakeDiT(device_index = 1, fail_enable = fail_enable)
    guider = types.SimpleNamespace(forward = lambda *a, **k: ("combined", a, k), num_conditions = 2)
    proxy = CFGParallelProxy(
        primary, replica, guider, compiled = compiled, explicit_on = explicit_on
    )
    return proxy, primary, replica, guider


# ── gating matrix ─────────────────────────────────────────────────────────────────
class _CtxPipe:
    """A pipeline whose __call__ opens transformer.cache_context (the branch signal)."""

    def __init__(self, transformer):
        self.transformer = transformer
        self.guider = types.SimpleNamespace(forward = lambda *a, **k: None)

    def __call__(self):
        with self.transformer.cache_context("pred_cond"):
            pass


def _fam(name = "hunyuanvideo-1.5-720p", guider = True):
    return types.SimpleNamespace(name = name, guidance_via_guider = guider)


def _gate(monkeypatch, pipe, fam, **overrides):
    # compiled=False = the eager tier, the only stack auto parallelises (bit-identity).
    kwargs = dict(
        requested = None,
        kind = "pipeline",
        transformer_source = "repo",
        hf_token = None,
        dtype = "bf16",
        quant_engaged = None,
        offload_active = False,
        compiled = False,
        attention_backend = "_native_cudnn",
        speed_active = True,
    )
    kwargs.update(overrides)
    return maybe_enable_cfg_parallel(pipe, fam, **kwargs)


def test_gate_disabled_by_request(monkeypatch):
    proxy, reason = _gate(monkeypatch, _CtxPipe(_FakeDiT()), _fam(), requested = "off")
    assert proxy is None and reason == "disabled by request"


def test_gate_family_allowlist(monkeypatch):
    _stub_torch(monkeypatch)
    proxy, reason = _gate(monkeypatch, _CtxPipe(_FakeDiT()), _fam(name = "wan2.2-ti2v-5b"))
    assert proxy is None and "allowlist" in reason


def test_gate_auto_refuses_compiled_stack(monkeypatch):
    # The per-device inductor artifacts drift ~1 ulp/step; auto is bit-identical-only,
    # so a compiled load never engages (and never spends the replica VRAM).
    _stub_torch(monkeypatch)
    proxy, reason = _gate(monkeypatch, _CtxPipe(_FakeDiT()), _fam(), compiled = True)
    assert proxy is None and "cfg_parallel=on" in reason


def test_gate_requires_guider_pipeline(monkeypatch):
    _stub_torch(monkeypatch)
    proxy, reason = _gate(monkeypatch, _CtxPipe(_FakeDiT()), _fam(guider = False))
    assert proxy is None and "guider" in reason


def test_gate_requires_pipeline_kind(monkeypatch):
    _stub_torch(monkeypatch)
    proxy, reason = _gate(monkeypatch, _CtxPipe(_FakeDiT()), _fam(), kind = "gguf")
    assert proxy is None and "second transformer source" in reason


def test_gate_skips_quantized_dit(monkeypatch):
    _stub_torch(monkeypatch)
    proxy, reason = _gate(monkeypatch, _CtxPipe(_FakeDiT()), _fam(), quant_engaged = "int8")
    assert proxy is None and "int8" in reason


def test_gate_skips_offload(monkeypatch):
    _stub_torch(monkeypatch)
    proxy, reason = _gate(monkeypatch, _CtxPipe(_FakeDiT()), _fam(), offload_active = True)
    assert proxy is None and "offload" in reason


def test_gate_needs_two_gpus(monkeypatch):
    _stub_torch(monkeypatch, device_count = 1)
    proxy, reason = _gate(monkeypatch, _CtxPipe(_FakeDiT()), _fam())
    assert proxy is None and "2+ CUDA devices" in reason


def test_gate_needs_secondary_vram(monkeypatch):
    # 1 GiB free on the only other device < weights + headroom -> stay single-device.
    _stub_torch(monkeypatch, free = {1: (1 << 30, 80 << 30)})
    proxy, reason = _gate(monkeypatch, _CtxPipe(_FakeDiT()), _fam())
    assert proxy is None and "free" in reason and "needs" in reason


def test_gate_replica_load_failure_is_soft(monkeypatch):
    # Every gate passes; the replica from_pretrained blows up (download / VRAM race):
    # the load must proceed single-device, never raise.
    _stub_torch(monkeypatch)
    proxy, reason = _gate(monkeypatch, _CtxPipe(_FakeDiT()), _fam())
    assert proxy is None and reason == "replica load failed"


def test_explicit_on_skips_family_allowlist(monkeypatch):
    # "on" bypasses the measured-family list; it still fails soft at the replica load
    # (the fake DiT class has no from_pretrained), proving the gate ORDER.
    _stub_torch(monkeypatch)
    proxy, reason = _gate(
        monkeypatch, _CtxPipe(_FakeDiT()), _fam(name = "some-future-family"), requested = "on"
    )
    assert proxy is None and reason == "replica load failed"


def test_pick_secondary_prefers_most_free(monkeypatch):
    _stub_torch(
        monkeypatch,
        device_count = 3,
        free = {1: (10 << 30, 80 << 30), 2: (40 << 30, 80 << 30)},
    )
    idx, free = _pick_secondary_device(0)
    assert idx == 2 and free == 40 << 30


# ── proxy semantics ───────────────────────────────────────────────────────────────
def test_proxy_delegates_reads_to_primary(monkeypatch):
    proxy, primary, _, _ = _make_proxy(monkeypatch)
    primary.some_flag = "x"
    assert proxy.some_flag == "x"
    proxy.shutdown()


def test_proxy_modules_covers_both(monkeypatch):
    # The cache-hook inner arming walks transformer.modules(); missing the replica's
    # blocks would leave its computed steps eager and erase the parallel win.
    proxy, primary, replica, _ = _make_proxy(monkeypatch)
    mods = proxy.modules()
    for m in primary.modules() + replica.modules():
        assert any(m is x for x in mods)
    proxy.shutdown()


def test_enable_cache_fans_out(monkeypatch):
    proxy, primary, replica, _ = _make_proxy(monkeypatch)
    proxy.enable_cache({"threshold": 0.12})
    assert primary.enabled_with == {"threshold": 0.12}
    assert replica.enabled_with == {"threshold": 0.12}
    proxy.disable_cache()
    assert primary.disables == 1 and replica.disables == 1
    proxy.shutdown()


def test_replica_enable_failure_reraises_and_breaks(monkeypatch):
    # A half-cached pair would skip differently per branch; the raise lets the caller's
    # best-effort path disable both, and _broken pins the sequential passthrough.
    proxy, primary, _, _ = _make_proxy(monkeypatch, fail_enable = True)
    with pytest.raises(RuntimeError):
        proxy.enable_cache({})
    assert primary.enabled_with == {}  # primary was hooked before the replica failed
    plan = proxy.plan_generation(
        cache_engaged = True, steps = 30, width = 1280, height = 720, frames = 33
    )
    assert plan["enabled"] is False
    proxy.shutdown()


def test_reset_stateful_cache_fans_out(monkeypatch):
    proxy, primary, replica, _ = _make_proxy(monkeypatch)
    proxy._reset_stateful_cache()
    assert primary.resets == 1 and replica.resets == 1
    proxy.shutdown()


def test_cache_context_enters_both_only_when_parallel_inline(monkeypatch):
    proxy, primary, replica, _ = _make_proxy(monkeypatch)
    proxy.enabled, proxy.dispatch = True, "inline"
    with proxy.cache_context("pred_cond"):
        pass
    assert primary.contexts == ["pred_cond"] and replica.contexts == ["pred_cond"]
    proxy.enabled = False
    with proxy.cache_context("pred_uncond"):
        pass
    assert replica.contexts == ["pred_cond"]  # sequential: primary only
    proxy.shutdown()


def test_routing_pred_cond_to_replica_inline(monkeypatch):
    proxy, primary, replica, _ = _make_proxy(monkeypatch)
    proxy.enabled, proxy.dispatch = True, "inline"
    with proxy.cache_context("pred_cond"):
        proxy("latents")
    with proxy.cache_context("pred_uncond"):
        proxy("latents")
    assert len(replica.calls) == 1 and len(primary.calls) == 1
    proxy.shutdown()


def test_routing_passthrough_when_disabled(monkeypatch):
    proxy, primary, replica, _ = _make_proxy(monkeypatch)
    proxy.enabled = False
    with proxy.cache_context("pred_cond"):
        proxy("latents")
    assert len(primary.calls) == 1 and len(replica.calls) == 0
    proxy.shutdown()


def test_thread_dispatch_resolves_through_guider(monkeypatch):
    proxy, primary, replica, guider = _make_proxy(monkeypatch)
    proxy.enabled, proxy.dispatch = True, "thread"
    with proxy.cache_context("pred_cond"):
        out = proxy("latents")
    # The worker resolves the pending prediction; the patched guider forward joins it
    # and hands a primary-device tensor to the original combine.
    combined, args, _ = guider.forward(out[0], _FakeTensor(_FakeDevice(index = 0)))
    assert combined == "combined"
    assert args[0].device.index == 0  # replica output copied to the primary device
    assert len(replica.calls) == 1
    proxy.shutdown()


# ── per-generation dispatch policy ──────────────────────────────────────────────────
def test_plan_parallel_on_eager_settles_to_thread(monkeypatch):
    proxy, _, _, _ = _make_proxy(monkeypatch, compiled = False)
    plan = proxy.plan_generation(
        cache_engaged = True, steps = 30, width = 1280, height = 720, frames = 33
    )
    assert plan["enabled"] is True and plan["dispatch"] == "inline"  # first run: compile-safe
    proxy.note_generation_done()
    plan = proxy.plan_generation(
        cache_engaged = True, steps = 30, width = 1280, height = 720, frames = 33
    )
    assert plan["dispatch"] == "thread"  # settled key: full overlap
    proxy.shutdown()


def test_plan_sequential_for_compiled_stack_even_with_cache(monkeypatch):
    # Compiled per-device artifacts drift regardless of the cache state (its computed
    # steps run the per-device compiled inners): auto stays sequential, an explicit
    # "on" accepts the fp-noise divergence.
    proxy, _, _, _ = _make_proxy(monkeypatch, compiled = True)
    for cache_engaged in (True, False):
        plan = proxy.plan_generation(
            cache_engaged = cache_engaged, steps = 30, width = 1280, height = 720, frames = 33
        )
        assert plan["enabled"] is False and plan["lossless"] is False
    proxy.shutdown()
    proxy_on, _, _, _ = _make_proxy(monkeypatch, compiled = True, explicit_on = True)
    plan = proxy_on.plan_generation(
        cache_engaged = False, steps = 10, width = 1280, height = 720, frames = 33
    )
    assert plan["enabled"] is True and plan["lossless"] is False
    proxy_on.shutdown()


def test_plan_parallel_for_eager_stack(monkeypatch):
    proxy, _, _, _ = _make_proxy(monkeypatch, compiled = False)
    plan = proxy.plan_generation(
        cache_engaged = False, steps = 10, width = 1280, height = 720, frames = 33
    )
    assert plan["enabled"] is True and plan["lossless"] is True
    proxy.shutdown()


def test_plan_requires_cfg_conditions(monkeypatch):
    # guidance ~1 collapses the guider to one condition: nothing to overlap.
    proxy, _, _, guider = _make_proxy(monkeypatch)
    guider.num_conditions = 1
    plan = proxy.plan_generation(
        cache_engaged = True, steps = 30, width = 1280, height = 720, frames = 33
    )
    assert plan["enabled"] is False
    proxy.shutdown()


def test_shape_change_forces_inline_once(monkeypatch):
    proxy, _, _, _ = _make_proxy(monkeypatch)
    proxy.plan_generation(cache_engaged = True, steps = 30, width = 1280, height = 720, frames = 33)
    proxy.note_generation_done()
    plan = proxy.plan_generation(
        cache_engaged = True, steps = 30, width = 960, height = 544, frames = 33
    )
    assert plan["dispatch"] == "inline"  # new shape may recompile: serialize
    proxy.shutdown()


def test_cancelled_generation_stays_inline(monkeypatch):
    proxy, _, _, _ = _make_proxy(monkeypatch)
    proxy.plan_generation(cache_engaged = True, steps = 30, width = 1280, height = 720, frames = 33)
    # No note_generation_done (cancel/failure): the same key must stay inline.
    plan = proxy.plan_generation(
        cache_engaged = True, steps = 30, width = 1280, height = 720, frames = 33
    )
    assert plan["dispatch"] == "inline"
    proxy.shutdown()


# ── teardown ──────────────────────────────────────────────────────────────────────
def test_teardown_restores_pipe_and_guider(monkeypatch):
    proxy, primary, _, guider = _make_proxy(monkeypatch)
    orig_forward = proxy._orig_guider_forward
    pipe = types.SimpleNamespace(transformer = proxy)
    teardown_cfg_parallel(pipe, proxy)
    assert pipe.transformer is primary
    assert guider.forward is orig_forward
    assert proxy._replica is None


def test_teardown_tolerates_foreign_object():
    teardown_cfg_parallel(types.SimpleNamespace(transformer = None), object())
