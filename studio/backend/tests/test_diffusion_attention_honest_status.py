# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An explicit attention pick (sage, flash, cuDNN, native, INT8 On) is what runs, or the status says what ran instead
and why. Routes, probes and kernels are stubs; the module census is real."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_attention as A
import core.inference.diffusion_int8_attention as I8
import core.inference.diffusion_int8_attention_dispatch as D
from core.inference.diffusion_auto_policy import build_resolved_record

N_BLOCKS = 4


class _Proc:
    """A diffusers-style attention processor: the dispatcher backend it calls with."""

    def __init__(self):
        self._attention_backend = None
        self._parallel_config = None


class _Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 2
        self.processor = _Proc()


class _Dit(torch.nn.Module):
    def __init__(self, head_dim = 128):
        super().__init__()
        self.config = {"attention_head_dim": head_dim}
        self.blocks = torch.nn.ModuleList([_Attn() for _ in range(N_BLOCKS)])
        self.set_calls: list = []
        self.fail_on: set = set()

    def set_attention_backend(self, name):
        self.set_calls.append(name)
        if name in self.fail_on:
            raise RuntimeError(f"{name} is not available")
        for m in self.blocks:
            m.processor._attention_backend = None if name == "native" else name


def _pipe(head_dim = 128):
    return types.SimpleNamespace(transformer = _Dit(head_dim))


def _target(device = "cuda", dtype = torch.bfloat16):
    return types.SimpleNamespace(device = device, dtype = dtype, torch_device = device)


@pytest.fixture(autouse = True)
def _isolated(monkeypatch):
    monkeypatch.delenv(I8.INT8_ATTENTION_ENV, raising = False)
    monkeypatch.setattr(A, "_SAGE_PROBE_CACHE", {})
    monkeypatch.setattr(A, "_SAGE_HUB_PROBE_CACHE", {})
    monkeypatch.setattr(A, "_ensure_attention_backend_installed", lambda *a, **k: None)
    monkeypatch.setattr(A, "_active_attention_backend", lambda: "native")
    monkeypatch.setattr(A, "_reset_global_backend_to_native", lambda *a, **k: None)
    monkeypatch.setattr(A, "_restore_native_backend", lambda *a, **k: None)
    monkeypatch.setattr(A, "_configure_native_attention", lambda *a, **k: None)
    monkeypatch.setattr(A, "guard_rocm_fused_sdpa", lambda *a, **k: ())
    monkeypatch.setattr(A, "install_rocm_contiguous_kv", lambda *a, **k: False)
    monkeypatch.setattr(A, "_indexed_cuda_device", lambda device: device)
    monkeypatch.setattr(A, "_is_cuda_nvidia", lambda t: getattr(t, "device", None) == "cuda")
    monkeypatch.setattr(A, "_is_cuda_rocm", lambda t: False)
    monkeypatch.setattr(A, "_install_sage_dispatch_guard", lambda: True)
    monkeypatch.setattr(A, "_install_sage_hub_dispatch_guard", lambda: True)
    monkeypatch.setattr(A, "_sage_version_too_old", lambda: None)
    monkeypatch.setattr(A, "_cudnn_attention_supported", lambda: True)
    monkeypatch.setattr(A, "_cuda_capability", lambda: (8, 9))
    # the INT8 routes: each would take every module it is let at, so a call that should not run shows in the census
    calls = {"auto": 0, "dispatch": 0, "requested": 0}

    def route_all(pipe, label):
        calls[label] += 1
        for m in pipe.transformer.blocks:
            m.processor._unsloth_int8 = True
        return N_BLOCKS

    def install_auto(pipe, target, **k):
        return route_all(pipe, "auto") if I8.mode(k.get("policy")) == "auto" else 0

    def install_requested(pipe, request, target, **k):
        if I8.mode(k.get("policy")) != "force" and not I8.requested(request):
            return 0
        return route_all(pipe, "requested")

    def dispatch_install(pipe, target, **k):
        forced = I8.mode(k.get("policy")) == "force" or I8.requested(k.get("requested"))
        if I8.mode(k.get("policy")) == "off" or not (forced or k.get("speed_active", True)):
            return 0
        calls["dispatch"] += 1
        for m in pipe.transformer.blocks:
            if not getattr(m.processor, "_unsloth_int8", False):
                m.processor._attention_backend = D.Int8Route(
                    m.processor._attention_backend, 0, torch.bfloat16, 0, 128
                )
        return N_BLOCKS

    monkeypatch.setattr(I8, "install_auto", install_auto)
    monkeypatch.setattr(I8, "install_requested", install_requested)
    monkeypatch.setattr(I8, "forget_plans", lambda: None)
    monkeypatch.setattr(D, "install", dispatch_install)
    yield calls


def _sage_pip(monkeypatch, probe = ""):
    monkeypatch.setattr(A, "_pip_sage2_installed", lambda: True)
    monkeypatch.setattr(A, "_run_sage_probe", lambda d, dt, hd = 128, sageattn = None: probe)


def _sage_missing(monkeypatch):
    monkeypatch.setattr(A, "_pip_sage2_installed", lambda: False)
    monkeypatch.setattr(
        A,
        "_load_sage_hub_kernel",
        lambda: (None, "no kernels-community/sage-attention build loads here"),
    )


def _load(
    pipe,
    requested,
    target = None,
    int8_policy = None,
):
    """What a load does: select, apply, then the status entry read off the modules."""
    target = target or _target()
    backend = A.select_attention_backend(target, requested, speed_active = True)
    engaged = A.apply_attention_backend(
        pipe, backend, target = target, requested = requested, int8_policy = int8_policy
    )
    record = build_resolved_record(
        {"attention_backend": A.attention_resolved_entry(requested, engaged, pipe, target = target)}
    )
    return engaged, A.annotate_attention_resolved(record, pipe)["attention_backend"]


# ---------------------------------------------------------------- explicit sage wins, auto may race


@pytest.mark.parametrize("build", ["pip", "hub"])
def test_explicit_sage_wins_over_the_int8_race(monkeypatch, _isolated, build):
    if build == "pip":
        _sage_pip(monkeypatch)
    else:
        monkeypatch.setattr(A, "_pip_sage2_installed", lambda: False)
        monkeypatch.setattr(A, "_load_sage_hub_kernel", lambda: (lambda *a, **k: None, ""))
        monkeypatch.setattr(A, "_run_sage_probe", lambda d, dt, hd = 128, sageattn = None: "")
    pipe = _pipe()
    engaged, entry = _load(pipe, "sage")
    name = "sage" if build == "pip" else "sage_hub"
    assert engaged == name
    assert _isolated == {"auto": 0, "dispatch": 0, "requested": 0}
    assert all(m.processor._attention_backend == name for m in pipe.transformer.blocks)
    assert entry["status"] == "applied" and entry["value"] == name and entry["engaged"] == name
    assert entry["modules"] == N_BLOCKS and entry["routed"] == {name: N_BLOCKS}


def test_auto_still_races_and_the_status_counts_the_winner(monkeypatch, _isolated):
    pipe = _pipe()
    engaged, entry = _load(pipe, None)
    assert engaged == I8.BACKEND and _isolated["auto"] == 1
    assert entry["source"] == "auto" and entry["value"] == I8.BACKEND
    assert entry["modules"] == N_BLOCKS and entry["routed"] == {I8.BACKEND: N_BLOCKS}


# ---------------------------------------------------------------- sage cannot run: what ran and why


def test_sage_missing_says_sdpa_ran_and_why(monkeypatch, _isolated):
    _sage_missing(monkeypatch)
    pipe = _pipe()
    engaged, entry = _load(pipe, "sage")
    assert engaged is None
    # the INT8 race does not take a declined explicit pick either
    assert _isolated == {"auto": 0, "dispatch": 0, "requested": 0}
    assert (
        entry["requested"] == "sage" and entry["value"] == "native" and entry["engaged"] == "native"
    )
    assert entry["status"] == "fell_back"
    assert "sageattention is not installed" in entry["reason"]
    assert entry["modules"] == N_BLOCKS and entry["routed"] == {"native": N_BLOCKS}


def test_sage_probe_failure_says_why(monkeypatch):
    _sage_pip(
        monkeypatch, probe = "self-check: cosine 0.1000, relative L1 0.9000 vs fp32 at head_dim 128"
    )
    _, entry = _load(_pipe(), "sage")
    assert entry["value"] == "native" and entry["status"] == "fell_back"
    assert "SageAttention failed its check on this GPU" in entry["reason"]
    assert "cosine 0.1000" in entry["reason"]


def test_sage_import_failure_says_why(monkeypatch):
    monkeypatch.setattr(A, "_pip_sage2_installed", lambda: True)

    def probe(*a, **k):
        raise ImportError("libcuda.so.1: cannot open shared object file")

    monkeypatch.setattr(A, "_run_sage_probe", probe)
    _, entry = _load(_pipe(), "sage")
    assert (
        entry["status"] == "fell_back" and "sageattention could not be imported" in entry["reason"]
    )


@pytest.mark.parametrize("build", ["pip", "hub"])
def test_sage_unsupported_head_dim_says_why(monkeypatch, build):
    if build == "pip":
        _sage_pip(monkeypatch)
    else:
        monkeypatch.setattr(A, "_pip_sage2_installed", lambda: False)
    _, entry = _load(_pipe(head_dim = 256), "sage")
    assert entry["value"] == "native" and entry["status"] == "fell_back"
    assert "head_dim <= 128" in entry["reason"] and "256" in entry["reason"]


def test_sage_fp32_says_why(monkeypatch):
    _sage_pip(monkeypatch)
    _, entry = _load(_pipe(), "sage", target = _target(dtype = torch.float32))
    assert entry["status"] == "fell_back" and "float32" in entry["reason"]


@pytest.mark.parametrize(
    "device, rocm, where",
    [("mps", False, "Apple Silicon"), ("cpu", False, "the CPU"), ("cuda", True, "AMD ROCm")],
)
def test_sage_on_another_platform_is_unsupported(monkeypatch, device, rocm, where):
    _sage_pip(monkeypatch)
    if rocm:
        monkeypatch.setattr(A, "_is_cuda_nvidia", lambda t: False)
        monkeypatch.setattr(A, "_is_cuda_rocm", lambda t: True)
    engaged, entry = _load(_pipe(), "sage", target = _target(device = device))
    assert engaged is None
    assert entry["value"] == "native" and entry["status"] == "unsupported"
    assert "needs an NVIDIA CUDA GPU" in entry["reason"] and where in entry["reason"]


# ---------------------------------------------------------------- INT8 On, and never a backend with 0 modules


def test_int8_on_over_explicit_sage_says_int8_ran(monkeypatch, _isolated):
    _sage_pip(monkeypatch)
    pipe = _pipe()
    engaged, entry = _load(pipe, "sage", int8_policy = "on")
    # On layers the kernel over the pick, by design; the status says so instead of "sage, applied"
    assert engaged == I8.BACKEND and _isolated["dispatch"] == 1
    assert entry["value"] == I8.BACKEND and entry["status"] == "fell_back"
    assert (
        "INT8 attention is On" in entry["reason"]
        and f"{N_BLOCKS} attention modules" in entry["reason"]
    )
    assert entry["modules"] == N_BLOCKS


def test_int8_on_that_routes_nothing_never_claims_the_kernel(monkeypatch):
    monkeypatch.setattr(D, "install", lambda *a, **k: 1)  # claims a route, sets none
    monkeypatch.setattr(I8, "install_requested", lambda *a, **k: 0)
    pipe = _pipe()
    engaged, entry = _load(pipe, "unsloth_int8")
    assert engaged == I8.BACKEND
    assert entry["value"] == "_native_cudnn" and entry["status"] == "fell_back"
    assert entry["modules"] == N_BLOCKS and I8.BACKEND not in entry["routed"]


def test_auto_claim_with_no_module_names_what_runs():
    pipe = _pipe()
    for m in pipe.transformer.blocks:
        m.processor._attention_backend = "_native_flash"
    entry = build_resolved_record(
        {"attention_backend": A.attention_resolved_entry(None, I8.BACKEND, pipe)}
    )["attention_backend"]
    assert entry["value"] == "_native_flash"
    assert "no attention module runs it" in entry["reason"]


def test_a_split_census_is_named():
    pipe = _pipe()
    for i, m in enumerate(pipe.transformer.blocks):
        m.processor._attention_backend = "sage" if i else None
    entry = build_resolved_record(
        {"attention_backend": A.attention_resolved_entry("sage", "sage", pipe)}
    )["attention_backend"]
    assert entry["status"] == "applied"
    assert (
        f"{N_BLOCKS - 1} of {N_BLOCKS} attention modules run SageAttention, 1 run native SDPA"
        in entry["reason"]
    )


def test_live_entry_follows_modules_that_moved(monkeypatch):
    _sage_pip(monkeypatch)
    pipe = _pipe()
    _, entry = _load(pipe, None)
    assert entry["value"] == I8.BACKEND
    for m in pipe.transformer.blocks:  # a later race dropped the kernel
        m.processor._unsloth_int8 = False
    live = A.live_attention_entry(entry, pipe, I8.BACKEND)
    assert live["value"] == "_native_cudnn" and live["modules"] == N_BLOCKS
    assert A.live_attention_entry(live, pipe, I8.BACKEND) is live


# ---------------------------------------------------------------- the other explicit picks


def test_explicit_native_and_cudnn_are_honored(_isolated):
    for requested, value in (("native", "native"), ("cudnn", "_native_cudnn")):
        engaged, entry = _load(_pipe(), requested)
        assert entry["status"] == "applied" and entry["value"] == value, requested
        assert entry["modules"] == N_BLOCKS
    assert _isolated == {"auto": 0, "dispatch": 0, "requested": 0}


def test_cudnn_without_a_head_dim_kernel_says_why(monkeypatch):
    monkeypatch.setattr(A, "_cudnn_serves_pipe", lambda *a, **k: False)
    _, entry = _load(_pipe(), "cudnn")
    assert entry["value"] == "native" and entry["status"] == "fell_back"
    assert "cuDNN attention has no kernel for this model's head_dim" in entry["reason"]


def test_cudnn_on_a_pre_ampere_card_is_unsupported(monkeypatch):
    monkeypatch.setattr(A, "_cudnn_attention_supported", lambda: False)
    _, entry = _load(_pipe(), "cudnn")
    assert entry["status"] == "unsupported" and "Ampere" in entry["reason"]


def test_flash3_off_hopper_is_unsupported():
    _, entry = _load(_pipe(), "flash3")
    assert entry["value"] == "native" and entry["status"] == "unsupported"
    assert (
        "FlashAttention 3 needs a sm90 to below sm100 GPU and this one is sm89" in entry["reason"]
    )


def test_flash_that_diffusers_cannot_set_says_why():
    pipe = _pipe()
    pipe.transformer.fail_on = {"flash"}
    _, entry = _load(pipe, "flash")
    assert entry["value"] == "native" and entry["status"] == "fell_back"
    assert (
        "diffusers could not set it" in entry["reason"]
        and "flash is not available" in entry["reason"]
    )


def test_status_builders_use_the_census_entry():
    from pathlib import Path

    root = Path(A.__file__).parent
    image = (root / "diffusion.py").read_text(encoding = "utf-8")
    video = (root / "video.py").read_text(encoding = "utf-8")
    assert image.count("attention_resolved_entry(") == 3  # generic load, deferred profile, H3-Image
    assert image.count("annotate_attention_resolved(") == 3
    assert "live_attention_entry(" in image
    assert video.count("attention_resolved_entry(") == 2  # generic video, H3 video
    assert video.count("annotate_attention_resolved(") == 2


@pytest.mark.parametrize("response", ["DiffusionStatusResponse", "VideoStatusResponse"])
def test_the_census_survives_the_status_response_model(response):
    # the status routes return through the response model, which drops any field it does not declare
    import models.inference as M

    pipe = _pipe()
    for m in pipe.transformer.blocks[:3]:
        m.processor._attention_backend = "sage_hub"
    record = build_resolved_record(
        {
            "attention_backend": A.attention_resolved_entry(
                "sage", "sage_hub", pipe, target = _target()
            )
        }
    )
    A.annotate_attention_resolved(record, pipe)
    body = getattr(M, response)(loaded = True, resolved = record).model_dump()
    entry = body["resolved"]["attention_backend"]
    assert entry["value"] == "sage_hub" and entry["status"] == "applied"
    assert entry["engaged"] == "sage_hub"
    assert entry["modules"] == 3
    assert entry["routed"] == {"native": 1, "sage_hub": 3}


# ---------------------------------------------------------------- no graph break, recompile or tensor op added


class _CompiledDit(_Dit):
    """A DiT whose forward reads each processor's backend and runs diffusers' attention dispatch, like the stock
    processors: a status read that changed anything a traced forward reads would show as a recompile."""

    def forward(self, x):
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        for m in self.blocks:
            x = x + dispatch_attention_fn(x, x, x, backend = m.processor._attention_backend)
        return x


def test_status_reads_add_no_graph_break_recompile_or_tensor_op(monkeypatch):
    pytest.importorskip("diffusers.models.attention_dispatch")
    import torch._dynamo
    from torch._dynamo.testing import CompileCounter
    from torch._dynamo.utils import counters
    from torch.overrides import TorchFunctionMode

    _sage_missing(monkeypatch)
    pipe = types.SimpleNamespace(transformer = _CompiledDit(head_dim = 8))
    _load(pipe, "sage")
    x = torch.randn(1, 16, 2, 8)

    # the counters see a deliberate break (resources README 13.6)
    def broken(t):
        t = t + 1
        torch._dynamo.graph_break()
        return t * 2

    torch._dynamo.reset()
    counters.clear()
    torch.compile(broken, backend = CompileCounter())(x)
    assert sum(counters["graph_break"].values()) >= 1

    torch._dynamo.reset()
    counters.clear()
    cnt = CompileCounter()
    compiled = torch.compile(pipe.transformer, backend = cnt)
    compiled(x)
    compiled(x)
    frames, ops = cnt.frame_count, cnt.op_count
    assert frames == 1 and sum(counters["graph_break"].values()) == 0

    class _Ops(TorchFunctionMode):
        calls: list = []

        def __torch_function__(
            self,
            func,
            types_,
            args = (),
            kwargs = None,
        ):
            self.calls.append(func)
            return func(*args, **(kwargs or {}))

    with _Ops() as mode:
        # everything the status path runs between renders: entry, census, live re-read, decline lookup
        entry = build_resolved_record(
            {"attention_backend": A.attention_resolved_entry("sage", None, pipe, target = _target())}
        )
        A.annotate_attention_resolved(entry, pipe)
        A.live_attention_entry(entry["attention_backend"], pipe, None)
        A.attention_census([pipe, pipe])
        A.attention_decline(pipe)
    assert mode.calls == [], "the status path must not touch a tensor (no host sync)"
    assert entry["attention_backend"]["status"] == "fell_back"
    compiled(x)
    compiled(x)
    assert (cnt.frame_count, cnt.op_count) == (frames, ops), "a status read recompiled the DiT"
    assert sum(counters["graph_break"].values()) == 0


# ---------------------------------------------------------------- diffusers names, and an explicit pick that cannot run


def _auto_cudnn_runs(monkeypatch):
    """The automatic pick on this (stub) card is the cuDNN pin."""
    monkeypatch.setattr(A, "_cudnn_serves_pipe", lambda *a, **k: True)
    monkeypatch.setattr(A, "_native_dispatches_cudnn", lambda *a, **k: False)
    monkeypatch.setattr(A, "_native_flash_measured_faster", lambda *a, **k: False)
    monkeypatch.setattr(A, "available_sdpa_kernels", lambda *a, **k: (A.SDPA_FLASH, A.SDPA_CUDNN))


def _load_like_studio(
    pipe,
    requested,
    *,
    speed = True,
    int8_policy = "off",
):
    """``_load`` as the image and video loads call it, with the explicit-pick fallback."""
    target = _target()
    backend = A.select_attention_backend(target, requested, speed_active = speed)
    engaged = A.apply_attention_backend(
        pipe,
        backend,
        target = target,
        requested = requested,
        speed_active = speed,
        int8_policy = int8_policy,
        fallback = A.explicit_pick_fallback(target, requested, speed_active = speed),
    )
    record = build_resolved_record(
        {"attention_backend": A.attention_resolved_entry(requested, engaged, pipe, target = target)}
    )
    return engaged, A.annotate_attention_resolved(record, pipe)["attention_backend"]


@pytest.mark.parametrize(
    "name, alias",
    [
        ("_native_flash", "flash_sdpa"),
        (" _NATIVE_FLASH ", "flash_sdpa"),
        ("_native_cudnn", "cudnn"),
        ("_flash_3_hub", "flash3"),
        ("flash_4_hub", "flash4"),
        ("sage_hub", "sage"),
        ("flash", "flash"),
        ("auto", "auto"),
    ],
)
def test_diffusers_dispatcher_names_fold_onto_studio_aliases(name, alias):
    assert A.normalize_attention_backend(name) == alias


def test_unknown_name_still_refused():
    with pytest.raises(ValueError, match = "Unsupported attention_backend"):
        A.normalize_attention_backend("_native_bogus")


def test_explicit_native_flash_runs_flash_sdpa(monkeypatch, _isolated):
    _auto_cudnn_runs(monkeypatch)
    pipe = _pipe()
    engaged, entry = _load_like_studio(pipe, "_native_flash")
    assert engaged == "_native_flash"
    assert entry["status"] == "applied" and entry["value"] == "_native_flash"
    assert entry["routed"] == {"_native_flash": N_BLOCKS}


def test_flash_without_flash_attn_keeps_the_automatic_pick(monkeypatch, _isolated):
    _auto_cudnn_runs(monkeypatch)
    pipe = _pipe()
    pipe.transformer.fail_on = {"flash"}
    engaged, entry = _load_like_studio(pipe, "flash")
    # the cuDNN pin the automatic pick runs, not bare native SDPA
    assert engaged == "_native_cudnn"
    assert pipe.transformer.set_calls[-1] == "_native_cudnn"
    assert entry["value"] == "_native_cudnn" and entry["routed"] == {"_native_cudnn": N_BLOCKS}
    # the status still says the explicit pick did not run, and why
    assert entry["status"] == "fell_back"
    assert "could not set it" in entry["reason"] and "flash is not available" in entry["reason"]
    decline = A.attention_decline(pipe)
    assert decline["tried"] == "flash" and decline["engaged"] == "_native_cudnn"
    assert _isolated == {"auto": 0, "dispatch": 0, "requested": 0}


def test_arch_gated_pick_keeps_the_automatic_pick(monkeypatch, _isolated):
    _auto_cudnn_runs(monkeypatch)
    engaged, entry = _load_like_studio(_pipe(), "flash4")  # sm89 stub: FA4 needs sm100+
    assert engaged == "_native_cudnn"
    assert entry["status"] == "unsupported" and "sm100" in entry["reason"]


def test_flash_without_flash_attn_and_speed_off_stays_native(monkeypatch, _isolated):
    _auto_cudnn_runs(monkeypatch)
    pipe = _pipe()
    pipe.transformer.fail_on = {"flash"}
    engaged, entry = _load_like_studio(pipe, "flash", speed = False)
    assert engaged is None and entry["value"] == "native" and entry["status"] == "fell_back"
    assert "_native_cudnn" not in pipe.transformer.set_calls


@pytest.mark.parametrize(
    "requested", [None, "", "auto", "AUTO", "native", "sdpa", "unsloth_int8", "bogus"]
)
@pytest.mark.parametrize("speed", [True, False])
def test_no_fallback_for_studio_or_native_picks(monkeypatch, requested, speed):
    _auto_cudnn_runs(monkeypatch)
    assert A.explicit_pick_fallback(_target(), requested, speed_active = speed) is None


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_no_fallback_off_nvidia(monkeypatch, device):
    assert A.explicit_pick_fallback(_target(device = device), "flash", speed_active = True) is None


@pytest.mark.parametrize(
    "requested", [None, "auto", "native", "sdpa", "cudnn", "sage", "unsloth_int8"]
)
@pytest.mark.parametrize("speed", [True, False])
def test_default_and_working_picks_unchanged_by_the_fallback(monkeypatch, requested, speed):
    """Same engaged backend, status entry and setter calls with and without the fallback argument."""
    _auto_cudnn_runs(monkeypatch)
    _sage_pip(monkeypatch)
    results = []
    for with_fallback in (False, True):
        pipe = _pipe()
        target = _target()
        backend = A.select_attention_backend(target, requested, speed_active = speed)
        kwargs = dict(target = target, requested = requested, speed_active = speed)
        if with_fallback:
            kwargs["fallback"] = A.explicit_pick_fallback(target, requested, speed_active = speed)
        engaged = A.apply_attention_backend(pipe, backend, **kwargs)
        entry = A.attention_resolved_entry(requested, engaged, pipe, target = target)
        procs = [
            (str(m.processor._attention_backend), getattr(m.processor, "_unsloth_int8", False))
            for m in pipe.transformer.blocks
        ]
        results.append(
            (engaged, entry, pipe.transformer.set_calls, procs, A.attention_decline(pipe))
        )
    assert results[0] == results[1]


def test_h3_fallback_keeps_the_strided_processors(monkeypatch):
    """MiniMax-H3: the fallback goes through the H3 arch pick, and both it and an explicit _native_flash install the
    strided processors (fused QK-RoPE) that a bare native fallback drops."""
    import core.inference.video_minimax_h3_attn as H

    _auto_cudnn_runs(monkeypatch)
    monkeypatch.setattr(H, "_capability", lambda device = None: (8, 9))
    target = _target()
    fallback = A.explicit_pick_fallback(target, "flash", speed_active = True)
    fallback = H.h3_attention_backend(fallback, requested = None, target = target)
    assert fallback == "_native_flash" and H.strided_install_backend(fallback)
    explicit = A.select_attention_backend(target, "_native_flash", speed_active = True)
    explicit = H.h3_attention_backend(explicit, requested = "_native_flash", target = target)
    assert explicit == "_native_flash" and H.strided_install_backend(explicit)
    assert not H.strided_install_backend(None)


@pytest.mark.parametrize("model", ["DiffusionLoadRequest", "VideoLoadRequest"])
def test_routes_accept_diffusers_names(model):
    import models.inference as M

    cls = getattr(M, model)
    fields = (
        {"model_path": "org/repo"} if "model_path" in cls.model_fields else {"repo_id": "org/repo"}
    )
    assert cls(**fields, attention_backend = "_native_flash").attention_backend == "flash_sdpa"
    assert cls(**fields, attention_backend = "_Native_Cudnn").attention_backend == "cudnn"
    with pytest.raises(Exception):
        cls(**fields, attention_backend = "_native_bogus")
