# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for transformer quantisation (``diffusion_transformer_quant.py``).

Hermetic: torch + torchao are stubbed via ``sys.modules``, and the per-scheme smoke
probe (``_scheme_supported`` / ``_smoke_probe``) is monkeypatched where the test cares
about the selection ladder rather than the GPU probe, so everything runs CPU-only.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

import core.inference.diffusion_transformer_quant as tq
from core.inference.diffusion_transformer_quant import (
    TQ_FP8,
    TQ_INT8,
    TQ_MXFP8,
    TQ_NVFP4,
    dense_transformer_supported,
    make_filter_fn,
    normalize_transformer_quant,
    quantize_transformer,
    select_transformer_quant_scheme,
)


def _target(*, device = "cuda", dtype = "bfloat16"):
    return types.SimpleNamespace(device = device, dtype = dtype)


def _stub_torch(
    monkeypatch,
    *,
    cc = (10, 0),
    with_fp8 = True,
    cuda_available = True,
    device_name = "NVIDIA B200",
):
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.float16 = "float16"
    if with_fp8:
        torch.float8_e4m3fn = "float8_e4m3fn"
    torch.cuda = types.SimpleNamespace(
        is_available = lambda: cuda_available,
        get_device_capability = lambda *a: cc,
        # A data-center name by default; consumer tests pass a GeForce name (or monkeypatch _is_consumer_gpu).
        get_device_name = lambda *a: device_name,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


# ── normalisation ─────────────────────────────────────────────────────────────


def test_normalize_transformer_quant():
    assert normalize_transformer_quant(None) is None
    assert normalize_transformer_quant("") is None
    assert normalize_transformer_quant("none") is None
    assert normalize_transformer_quant("off") is None
    assert normalize_transformer_quant("AUTO") == "auto"
    assert normalize_transformer_quant("INT8") == TQ_INT8
    assert normalize_transformer_quant("fp8") == TQ_FP8
    with pytest.raises(ValueError):
        normalize_transformer_quant("int2")


# ── dense-source gate ───────────────────────────────────────────────────────────


def test_dense_transformer_supported_requires_cuda_bf16(monkeypatch):
    _stub_torch(monkeypatch)
    assert dense_transformer_supported(_target()) is True
    assert dense_transformer_supported(_target(device = "cpu")) is False
    assert dense_transformer_supported(_target(dtype = "float16")) is False


# ── scheme selection ladder ─────────────────────────────────────────────────────


def _allow(monkeypatch, allowed):
    """Force ``_scheme_supported`` to accept only ``allowed`` (simulates smoke results)."""
    monkeypatch.setattr(tq, "_scheme_supported", lambda scheme, device, **kw: scheme in allowed)


def test_auto_blackwell_prefers_fp8_then_falls_back(monkeypatch):
    _stub_torch(monkeypatch, cc = (10, 0))
    # Even with every scheme available, auto picks fp8 on Blackwell: measured on a B200 it is faster AND more accurate than nvfp4 at DiT shapes.
    _allow(monkeypatch, {TQ_NVFP4, TQ_MXFP8, TQ_FP8, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_FP8
    # fp8 unavailable: auto skips nvfp4 even though the hardware runs it, because nvfp4 is an
    # explicit opt-in only (slower AND less accurate at DiT shapes), and lands on mxfp8.
    _allow(monkeypatch, {TQ_NVFP4, TQ_MXFP8, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_MXFP8
    # Only mxfp8 + int8 left -> mxfp8 (still above int8).
    _allow(monkeypatch, {TQ_MXFP8, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_MXFP8
    # Only int8 usable -> int8.
    _allow(monkeypatch, {TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_INT8


def test_auto_consumer_blackwell_prefers_int8(monkeypatch):
    # Consumer Blackwell (RTX 50xx): fp8 FP32-accumulate is throughput-halved while int8 is full-rate, so auto prefers int8.
    _stub_torch(monkeypatch, cc = (10, 0), device_name = "NVIDIA GeForce RTX 5090")
    _allow(monkeypatch, {TQ_NVFP4, TQ_MXFP8, TQ_FP8, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_INT8
    # int8 unavailable, so it falls back to the rest of the tier (fp8 next).
    _allow(monkeypatch, {TQ_NVFP4, TQ_MXFP8, TQ_FP8})
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_FP8


def test_auto_consumer_ada_prefers_int8(monkeypatch):
    # Consumer Ada (RTX 4090): int8 runs ~2x fp8's nerfed FP32-accumulate rate.
    _stub_torch(monkeypatch, cc = (8, 9), device_name = "NVIDIA GeForce RTX 4090")
    _allow(monkeypatch, {TQ_FP8, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_INT8


def test_auto_workstation_unknown_prefers_int8(monkeypatch):
    # An unknown / workstation name is treated as consumer (the safe default), so int8 first.
    _stub_torch(monkeypatch, cc = (8, 9), device_name = "NVIDIA RTX A5000")
    _allow(monkeypatch, {TQ_FP8, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_INT8


def test_auto_professional_rtx_prefers_fp8(monkeypatch):
    # Professional parts (RTX PRO 6000 Blackwell, RTX 6000 Ada) count as datacenter elsewhere in the backend, so auto keeps fp8 first, matching llama_cpp.
    for device_name, cc in (
        ("NVIDIA RTX PRO 6000 Blackwell Server Edition", (10, 0)),
        ("NVIDIA RTX 6000 Ada Generation", (8, 9)),
    ):
        _stub_torch(monkeypatch, cc = cc, device_name = device_name)
        _allow(monkeypatch, {TQ_FP8, TQ_INT8})
        assert select_transformer_quant_scheme(_target(), "auto") == TQ_FP8


def test_auto_ada_hopper_prefers_fp8(monkeypatch):
    # Data-center Ada (L40S) / Hopper (H100) are not nerfed, so fp8 comes first.
    _stub_torch(monkeypatch, cc = (8, 9), device_name = "NVIDIA L40S")
    _allow(monkeypatch, {TQ_NVFP4, TQ_MXFP8, TQ_FP8, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_FP8
    _stub_torch(monkeypatch, cc = (9, 0), device_name = "NVIDIA H100 80GB HBM3")  # Hopper
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_FP8


def test_auto_ampere_prefers_int8(monkeypatch):
    _stub_torch(monkeypatch, cc = (8, 0))
    _allow(monkeypatch, {TQ_FP8, TQ_INT8})  # fp8 cores absent on Ampere -> int8 only in ladder
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_INT8
    _stub_torch(monkeypatch, cc = (8, 6))
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_INT8


def test_auto_pre_ampere_unsupported(monkeypatch):
    _stub_torch(monkeypatch, cc = (7, 5))  # Turing: below the int8-dynamic floor
    _allow(monkeypatch, {TQ_INT8, TQ_FP8})
    assert select_transformer_quant_scheme(_target(), "auto") is None


def test_explicit_scheme_honored_or_none(monkeypatch):
    _stub_torch(monkeypatch, cc = (8, 0))
    _allow(monkeypatch, {TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "int8") == TQ_INT8
    # An explicit unsupported scheme is NOT silently downgraded: None, so the GGUF fallback.
    assert select_transformer_quant_scheme(_target(), "fp8") is None
    assert select_transformer_quant_scheme(_target(), "nvfp4") is None


def test_select_none_when_disabled_or_non_cuda(monkeypatch):
    _stub_torch(monkeypatch)
    _allow(monkeypatch, {TQ_INT8, TQ_FP8, TQ_NVFP4})
    assert select_transformer_quant_scheme(_target(), None) is None
    assert select_transformer_quant_scheme(_target(device = "cpu"), "auto") is None


# ── _scheme_supported / _smoke_probe ────────────────────────────────────────────


def test_scheme_supported_shortcircuits(monkeypatch):
    # No CUDA gives False without running the smoke probe.
    _stub_torch(monkeypatch, cuda_available = False)
    monkeypatch.setattr(tq, "_smoke_probe", lambda *a: pytest.fail("probe should not run"))
    assert tq._scheme_supported(TQ_INT8, "cuda") is False
    # fp8 requested but the fp8 dtype is missing gives False before the probe.
    _stub_torch(monkeypatch, with_fp8 = False)
    monkeypatch.setattr(tq, "_smoke_probe", lambda *a: pytest.fail("probe should not run"))
    assert tq._scheme_supported(TQ_FP8, "cuda") is False


class _FakeTensor:
    """Records the probe's zero-row write so the assertion below can check it happened."""

    def __init__(self):
        self.zeroed = []

    def __setitem__(self, key, value):
        self.zeroed.append((key, value))


class _FakeBool:
    def __init__(self, value):
        self.value = value

    def all(self):
        return self

    def item(self):
        return self.value


def test_smoke_probe_caches_and_tolerates_failure(monkeypatch):
    tq._SMOKE_CACHE.clear()
    calls = {"n": 0}
    finite = {"ok": True}

    class _Lin:
        def __init__(self, *a, **k):
            pass

        def to(self, **k):
            return self

    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.nn = types.SimpleNamespace(Linear = _Lin)
    torch.randn = lambda *a, **k: _FakeTensor()
    torch.isfinite = lambda t: _FakeBool(finite["ok"])
    torch.no_grad = lambda: __import__("contextlib").nullcontext()
    torch.cuda = types.SimpleNamespace(is_available = lambda: True, synchronize = lambda: None)
    monkeypatch.setitem(sys.modules, "torch", torch)

    tqz = types.ModuleType("torchao.quantization")

    def _quantize_ok(
        module,
        config,
        filter_fn = None,
    ):
        calls["n"] += 1

    tqz.quantize_ = _quantize_ok
    tqz.Int8DynamicActivationInt8WeightConfig = lambda: "int8cfg"
    tqz.Float8DynamicActivationFloat8WeightConfig = lambda: "fp8cfg"
    monkeypatch.setitem(sys.modules, "torchao.quantization", tqz)
    # _Lin must be callable or the forward lin(x) would fail, so make instances callable.
    _Lin.__call__ = lambda self, x: x

    assert tq._smoke_probe(TQ_INT8, "cuda") is True
    assert tq._smoke_probe(TQ_INT8, "cuda") is True  # cached, no second quantize_
    assert calls["n"] == 1

    # A scheme whose quantize_ raises probes False (and is cached).
    tq._SMOKE_CACHE.clear()

    def _quantize_boom(
        module,
        config,
        filter_fn = None,
    ):
        raise RuntimeError("kernel unavailable")

    tqz.quantize_ = _quantize_boom
    assert tq._smoke_probe(TQ_FP8, "cuda") is False

    # A kernel that RUNS but returns non-finite values probes False too. torchao's fp8 scale
    # chooser has no eps clamp, so a zero activation row gives scale 0 and NaN qdata unless the
    # config floors it, and the floor is applied only on a torchao exposing activation_value_lb.
    # Without this the probe passed on such a build and every zero-padded text stream went black.
    # int8, not fp8: _make_quant_config(fp8) imports PerRow, which this stub module does not
    # carry, so an fp8 probe here would return False from the ImportError and prove nothing.
    tq._SMOKE_CACHE.clear()
    tqz.quantize_ = _quantize_ok
    finite["ok"] = False
    assert tq._smoke_probe(TQ_INT8, "cuda") is False


def test_the_smoke_probe_does_not_cache_an_out_of_memory(monkeypatch):
    # A full GPU is not a verdict on the scheme, and this probe now runs on the ROUTE thread --
    # before the arbiter evicts the resident chat model, which is the point of raising the refusal
    # early -- so it meets a full GPU by design. Caching that answer would refuse every later
    # EXPLICIT request for the scheme for the life of the process, on a host that runs it fine.
    class _OOM(RuntimeError):
        pass

    class _Lin:
        def __init__(self, *a, **k):
            pass

        def to(self, **k):
            return self

        def __call__(self, x):
            return x

    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.nn = types.SimpleNamespace(Linear = _Lin)
    torch.randn = lambda *a, **k: _FakeTensor()
    torch.isfinite = lambda t: _FakeBool(True)
    torch.no_grad = lambda: __import__("contextlib").nullcontext()
    torch.cuda = types.SimpleNamespace(
        is_available = lambda: True, synchronize = lambda: None, OutOfMemoryError = _OOM
    )
    torch.OutOfMemoryError = _OOM
    monkeypatch.setitem(sys.modules, "torch", torch)

    tqz = types.ModuleType("torchao.quantization")
    calls = {"n": 0}

    def _quantize(
        module,
        config,
        filter_fn = None,
    ):
        calls["n"] += 1

    tqz.quantize_ = _quantize
    monkeypatch.setitem(sys.modules, "torchao.quantization", tqz)
    # The config builder is where the allocation lands in practice; raising there keeps this test
    # off the real torchao, which _make_quant_config would otherwise import for its config classes.
    fault: list = [_OOM("CUDA out of memory. Tried to allocate 2.00 GiB")]

    def _config(scheme, fast_accum = None):
        if fault[0] is not None:
            raise fault[0]
        return "cfg"

    monkeypatch.setattr(tq, "_make_quant_config", _config)

    tq._SMOKE_CACHE.clear()
    assert tq._smoke_probe(TQ_INT8, "cuda") is False
    assert tq._SMOKE_CACHE == {}, "an OOM must not be remembered as 'this scheme cannot run'"
    # The eviction happens, memory frees, and the very next ask gets the real answer.
    fault[0] = None
    assert tq._smoke_probe(TQ_INT8, "cuda") is True
    assert calls["n"] == 1

    # A NON-memory failure is still cached: that one really is a property of the build.
    tq._SMOKE_CACHE.clear()
    fault[0] = RuntimeError("kernel unavailable")
    assert tq._smoke_probe(TQ_INT8, "cuda") is False
    assert tq._SMOKE_CACHE == {(TQ_INT8, "cuda"): False}

    # And the PRE-EVICTION caller gets "could not tell", not "cannot run": the gate it feeds turns
    # a False into a 409 before the arbiter has freed the VRAM the probe wanted, so answering
    # "unsupported" there refuses a load the eviction was about to make room for. A non-memory
    # failure stays False for that caller too -- that one is a real verdict.
    tq._SMOKE_CACHE.clear()
    fault[0] = _OOM("CUDA out of memory. Tried to allocate 2.00 GiB")
    assert tq._smoke_probe(TQ_INT8, "cuda", unproven_ok = True) is True
    assert tq._SMOKE_CACHE == {}
    fault[0] = RuntimeError("kernel unavailable")
    assert tq._smoke_probe(TQ_INT8, "cuda", unproven_ok = True) is False


# ── out-of-process probe ────────────────────────────────────────────────────────
#
# The probe allocates, and a CUDA context is process-wide and never given back: measured on a
# B200 host, one uncached probe takes the backend from 0 MiB to 806 MiB for the life of the
# process. /images/download-plan reaches this while the user is only STAGING a download, so the
# child is what keeps a plan from costing VRAM. Verdict parity with the in-process probe is the
# contract; everything below pins one half of it.


@pytest.fixture(autouse = True)
def _reset_child_probe_state():
    tq._SMOKE_CACHE.clear()
    tq._CHILD_PROBE_UNAVAILABLE = False
    tq._CHILD_PROBE_SPAWN_ERRORS = 0
    yield
    tq._SMOKE_CACHE.clear()
    tq._CHILD_PROBE_UNAVAILABLE = False
    tq._CHILD_PROBE_SPAWN_ERRORS = 0


def test_the_child_answers_for_every_scheme_in_one_go(monkeypatch):
    # One child, not one per ladder step: spawning costs 3.9 s on a B200 host (nearly all of it
    # importing torch), so an auto ladder walking three schemes would otherwise pay it three times.
    _stub_torch(monkeypatch)
    spawns = {"n": 0}

    def _table(device):
        spawns["n"] += 1
        return {TQ_INT8: True, TQ_FP8: True, TQ_NVFP4: False, TQ_MXFP8: False}

    monkeypatch.setattr(tq, "_child_probe_table", _table)
    monkeypatch.setattr(
        tq, "_run_smoke_probe", lambda *a: pytest.fail("probed in-process after the child answered")
    )
    assert [tq._scheme_supported(s, "cuda") for s in tq.TQ_SCHEMES] == [True, True, False, False]
    assert spawns["n"] == 1
    assert tq._SMOKE_CACHE == {
        (TQ_INT8, "cuda"): True,
        (TQ_FP8, "cuda"): True,
        (TQ_NVFP4, "cuda"): False,
        (TQ_MXFP8, "cuda"): False,
    }


def test_a_child_out_of_memory_is_not_cached_and_is_not_a_verdict(monkeypatch):
    # Same contract the in-process probe holds: a full GPU says nothing about the scheme. Falling
    # back in-process here would be worse than useless -- it would meet the same full GPU and pay
    # the context on the way -- so the OOM is answered without re-probing.
    _stub_torch(monkeypatch)
    monkeypatch.setattr(tq, "_child_probe_table", lambda device: {TQ_INT8: None, TQ_FP8: True})
    monkeypatch.setattr(
        tq, "_run_smoke_probe", lambda *a: pytest.fail("re-probed in-process after a child OOM")
    )
    assert tq._scheme_supported(TQ_INT8, "cuda") is False
    assert tq._scheme_supported(TQ_INT8, "cuda", unproven_ok = True) is True
    assert (TQ_INT8, "cuda") not in tq._SMOKE_CACHE
    # The schemes that DID answer are still cached; one scheme's OOM does not lose the table.
    assert tq._SMOKE_CACHE == {(TQ_FP8, "cuda"): True}


def test_no_child_falls_back_to_the_in_process_probe(monkeypatch):
    # A frozen desktop build or a sandbox that refuses to spawn must still be able to load a
    # model: the VRAM this saves is not worth failing a load over.
    _stub_torch(monkeypatch)
    monkeypatch.setattr(tq, "_child_probe_table", lambda device: None)
    monkeypatch.setattr(tq, "_smoke_probe", lambda *a, **k: True)
    assert tq._scheme_supported(TQ_INT8, "cuda") is True


def test_a_host_without_cuda_never_spawns_a_child(monkeypatch):
    # CPU-only, MPS and XPU hosts answer False above the child, so they pay nothing at all.
    _stub_torch(monkeypatch, cuda_available = False)
    monkeypatch.setattr(
        tq, "_child_probe_table", lambda device: pytest.fail("spawned on a CUDA-less host")
    )
    assert tq._scheme_supported(TQ_INT8, "cuda") is False
    # Same for an fp8 ask on a torch without the dtype.
    _stub_torch(monkeypatch, with_fp8 = False)
    assert tq._scheme_supported(TQ_FP8, "cuda") is False


def test_a_cached_scheme_does_not_spawn_a_child(monkeypatch):
    _stub_torch(monkeypatch)
    tq._SMOKE_CACHE[(TQ_INT8, "cuda")] = True
    monkeypatch.setattr(tq, "_child_probe_table", lambda device: pytest.fail("spawned on a hit"))
    assert tq._scheme_supported(TQ_INT8, "cuda") is True


def test_the_child_entry_posts_one_table_covering_every_scheme(monkeypatch):
    # Child side. Every scheme is attempted even after one fails: the verdicts are independent
    # and the whole point of the child is to pay the CUDA context once.
    posted = []
    monkeypatch.setattr(
        tq,
        "_run_smoke_probe",
        lambda scheme, device: {TQ_INT8: True, TQ_FP8: None}.get(scheme, False),
    )
    tq._child_probe_entry("cuda", tq.TQ_SCHEMES, types.SimpleNamespace(put = posted.append))
    assert posted == [{TQ_INT8: True, TQ_FP8: None, TQ_NVFP4: False, TQ_MXFP8: False}]


def test_a_spawn_failure_is_remembered_so_it_is_paid_once(monkeypatch):
    import multiprocessing

    def _no_spawn(name):
        raise ValueError(f"no {name} start method here")

    monkeypatch.setattr(multiprocessing, "get_context", _no_spawn)
    assert tq._child_probe_table("cuda") is None
    assert tq._CHILD_PROBE_UNAVAILABLE is True
    # Second time round it does not even reach multiprocessing.
    monkeypatch.setattr(
        multiprocessing, "get_context", lambda name: pytest.fail("retried a spawn known to fail")
    )
    assert tq._child_probe_table("cuda") is None


# ── the child's lifetime ────────────────────────────────────────────────────────


class _FakeProbeChild:
    """Stand-in for the spawned probe: alive from start() until sent the signal it obeys.

    ``dies_on = None`` is the wedged child -- uninterruptible in the driver, surviving SIGKILL --
    which is exactly the case that must not lose its lifetime record."""

    def __init__(self, dies_on = "terminate"):
        self.pid = 4321
        self.alive = False
        self.signals = []
        self._dies_on = dies_on

    def start(self):
        self.alive = self._dies_on != "start"

    def is_alive(self):
        return self.alive

    def join(self, timeout = None):
        pass

    def terminate(self):
        self.signals.append("terminate")
        if self._dies_on == "terminate":
            self.alive = False

    def kill(self):
        self.signals.append("kill")
        if self._dies_on == "kill":
            self.alive = False


class _FakeProbeQueue:
    def __init__(self, table = None):
        self._table = table
        self.closed = False

    def get(self, timeout = None):
        if self._table is None:
            raise ValueError("empty")
        table, self._table = self._table, None
        return table

    def close(self):
        self.closed = True

    def join_thread(self):
        pass


class _FakeSpawnContext:
    def __init__(self, child, queue):
        self._child = child
        self._queue = queue

    def Queue(self):
        return self._queue

    def Process(
        self,
        target = None,
        args = (),
        daemon = None,
    ):
        return self._child


@pytest.fixture
def _probe_lifetime_records(monkeypatch):
    """Capture what the probe adopts and forgets, without touching the real record."""
    import utils.process_lifetime as pl

    records = {"adopted": [], "forgotten": []}
    monkeypatch.setattr(pl, "adopt_pid", records["adopted"].append)
    monkeypatch.setattr(pl, "forget_pid", records["forgotten"].append)
    return records


def test_the_probe_child_is_adopted_so_a_shutdown_sweep_can_reach_it(
    monkeypatch, _probe_lifetime_records
):
    # The child-side PDEATHSIG bind is Linux only, and the Windows job object is documented to
    # fail when Studio already runs inside an incompatible host job. In that configuration this
    # record is the only thing left that can reach a probe still holding a CUDA context, both
    # from the shutdown sweep and from the next startup.
    import multiprocessing

    monkeypatch.setattr(tq, "_CHILD_PROBE_TIMEOUT", 0.0)
    child = _FakeProbeChild(dies_on = "start")
    monkeypatch.setattr(
        multiprocessing,
        "get_context",
        lambda name: _FakeSpawnContext(child, _FakeProbeQueue({TQ_INT8: True})),
    )
    assert tq._child_probe_table("cuda") == {TQ_INT8: True}
    assert _probe_lifetime_records["adopted"] == [child.pid]


def test_the_queue_is_built_with_the_lease_secret_already_scrubbed(
    monkeypatch, _probe_lifetime_records
):
    # On POSIX the first spawn-context queue creates the named semaphores that start
    # multiprocessing's resource tracker, and that tracker is exec'd with this process's
    # environment and then outlives every child. Built above the scrub it carries the
    # native-path lease secret for the life of the backend, where the child-side scrub can no
    # longer reach it. Measured: the secret is in the tracker's /proc/<pid>/environ when the
    # queue is built first, and absent when it is built here.
    import multiprocessing

    from utils.native_path_leases import LEASE_SECRET_ENV

    secret = "x" * 64
    monkeypatch.setenv(LEASE_SECRET_ENV, secret)
    monkeypatch.setattr(tq, "_CHILD_PROBE_TIMEOUT", 0.0)
    seen = {}

    class _WatchingContext(_FakeSpawnContext):
        def Queue(self):
            seen["at_queue"] = os.environ.get(LEASE_SECRET_ENV)
            return super().Queue()

    monkeypatch.setattr(
        multiprocessing,
        "get_context",
        lambda name: _WatchingContext(
            _FakeProbeChild(dies_on = "start"), _FakeProbeQueue({TQ_INT8: True})
        ),
    )
    assert tq._child_probe_table("cuda") == {TQ_INT8: True}
    assert seen["at_queue"] is None
    # And the parent has it back once the child is started.
    assert os.environ.get(LEASE_SECRET_ENV) == secret


# ── a spawn that failed but may not fail next time ──────────────────────────────


def test_a_transient_spawn_oserror_is_retried_rather_than_latched(
    monkeypatch, _probe_lifetime_records
):
    # Descriptors, process slots and /dev/shm all come back. Latching the OSError would hold the
    # backend on the in-process probe -- and so on the ~800 MiB the child exists to avoid -- for
    # every later miss, until Studio restarts.
    import multiprocessing

    calls = {"n": 0}

    def _get_context(name):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(24, "Too many open files")
        return _FakeSpawnContext(_FakeProbeChild(dies_on = "start"), _FakeProbeQueue({TQ_INT8: True}))

    monkeypatch.setattr(multiprocessing, "get_context", _get_context)
    monkeypatch.setattr(tq, "_CHILD_PROBE_TIMEOUT", 0.0)
    assert tq._child_probe_table("cuda") is None
    assert tq._CHILD_PROBE_UNAVAILABLE is False
    # The pressure clears and the next miss gets its child back.
    assert tq._child_probe_table("cuda") == {TQ_INT8: True}
    assert calls["n"] == 2
    assert tq._CHILD_PROBE_SPAWN_ERRORS == 0


def test_a_host_that_refuses_every_spawn_stops_being_asked(monkeypatch):
    # The retry is bounded: an OSError on every attempt is indistinguishable from a sandbox that
    # will never spawn, so the latch still lands after a short run of them.
    import multiprocessing

    calls = {"n": 0}

    def _always_refuses(name):
        calls["n"] += 1
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(multiprocessing, "get_context", _always_refuses)
    for _ in range(tq._CHILD_PROBE_SPAWN_ERROR_LIMIT):
        assert tq._child_probe_table("cuda") is None
    assert tq._CHILD_PROBE_UNAVAILABLE is True
    settled = calls["n"]
    assert tq._child_probe_table("cuda") is None
    assert calls["n"] == settled


def test_a_child_that_survives_terminate_is_killed(_probe_lifetime_records):
    # Five seconds of terminate and then giving up leaves the VRAM this probe exists to hand
    # back held for the whole 180 s timeout, or forever if the child is wedged.
    child = _FakeProbeChild(dies_on = "kill")
    child.start()
    assert tq._close_probe_child(child, _FakeProbeQueue()) is True
    assert child.signals == ["terminate", "kill"]
    assert _probe_lifetime_records["forgotten"] == [child.pid]


def test_a_child_that_survives_kill_keeps_its_breadcrumb_and_is_reported(_probe_lifetime_records):
    # Same call the chat worker makes: a survivor keeps its handle rather than being dropped
    # silently, so the sweep still has something to retry.
    child = _FakeProbeChild(dies_on = None)
    child.start()
    assert tq._close_probe_child(child, _FakeProbeQueue()) is False
    assert child.signals == ["terminate", "kill"]
    assert _probe_lifetime_records["forgotten"] == []


def test_a_clean_exit_forgets_the_pid(_probe_lifetime_records):
    # The child that posted its table and exited is not signalled at all, and its record goes.
    child = _FakeProbeChild(dies_on = "start")
    child.start()
    queue = _FakeProbeQueue()
    assert tq._close_probe_child(child, queue) is True
    assert child.signals == []
    assert queue.closed is True
    assert _probe_lifetime_records["forgotten"] == [child.pid]


def test_the_probe_body_separates_an_oom_from_a_verdict(monkeypatch):
    # _run_smoke_probe is what BOTH the child and the in-process fallback call, so this three-way
    # answer is the single place the two can be shown not to drift.
    class _OOM(RuntimeError):
        pass

    fault = [None]

    class _Lin:
        def __init__(self, *a, **k):
            pass

        def to(self, **k):
            return self

        def __call__(self, x):
            if fault[0] is not None:
                raise fault[0]
            return x

    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.nn = types.SimpleNamespace(Linear = _Lin)
    torch.randn = lambda *a, **k: _FakeTensor()
    torch.isfinite = lambda t: _FakeBool(True)
    torch.no_grad = lambda: __import__("contextlib").nullcontext()
    torch.cuda = types.SimpleNamespace(is_available = lambda: True, synchronize = lambda: None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    tqz = types.ModuleType("torchao.quantization")
    tqz.quantize_ = lambda module, config, filter_fn = None: None
    tqz.Int8DynamicActivationInt8WeightConfig = lambda: "int8cfg"
    tqz.Float8DynamicActivationFloat8WeightConfig = lambda: "fp8cfg"
    monkeypatch.setitem(sys.modules, "torchao.quantization", tqz)

    assert tq._run_smoke_probe(TQ_INT8, "cuda") is True
    fault[0] = _OOM("CUDA out of memory. Tried to allocate 2.00 GiB")
    assert tq._run_smoke_probe(TQ_INT8, "cuda") is None
    fault[0] = RuntimeError("kernel unavailable")
    assert tq._run_smoke_probe(TQ_INT8, "cuda") is False


def test_an_oom_is_recognised_however_it_is_spelled():
    # torch.OutOfMemoryError subclasses RuntimeError (not MemoryError) and has moved between
    # torch.cuda and torch across releases, so neither name alone is enough.
    assert tq._is_out_of_memory(MemoryError("no room")) is True
    assert tq._is_out_of_memory(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")) is True
    assert tq._is_out_of_memory(RuntimeError("kernel unavailable")) is False
    assert tq._is_out_of_memory(ImportError("cannot import name 'ScalingType'")) is False


def test_an_unusable_scheme_names_the_fault_the_user_can_actually_fix(monkeypatch):
    # select_transformer_quant_scheme folds three different faults into one None, and an EXPLICIT
    # scheme now fails CLOSED, so that None becomes the whole explanation on a 409. Measured on a
    # B200 whose torchao could not import (a torch/torchao skew): every explicit scheme was refused
    # with "not usable ... on this GPU", which is false and sends the owner hunting for hardware.
    monkeypatch.setattr(tq, "_TORCHAO_UNAVAILABLE", (None,))
    assert tq.explain_unusable_scheme("z-image-turbo", "fp8") == (
        "'fp8' is not usable for family 'z-image-turbo' on this GPU"
    )
    # The measured deny list wins over both: it holds on every GPU, so naming hardware would be wrong.
    # mxfp8, not fp8: fp8 on qwen-image is no longer denied, so it would take the GPU branch here.
    denied = tq.explain_unusable_scheme("qwen-image", "mxfp8")
    assert "measured accuracy gate" in denied and "whatever the GPU" in denied
    assert "on this GPU" not in denied.replace("whatever the GPU", "")

    # A torchao that cannot import is a package problem, and the message has to say so.
    monkeypatch.setattr(
        tq, "_TORCHAO_UNAVAILABLE", ("ImportError: cannot import name 'ScalingType'",)
    )
    broken = tq.explain_unusable_scheme("z-image-turbo", "fp8")
    assert "cannot import name 'ScalingType'" in broken
    assert "not a limit of the GPU" in broken
    # ...but a denied family is still reported as denied, whatever torchao is doing.
    assert "measured accuracy gate" in tq.explain_unusable_scheme("qwen-image", "mxfp8")


def test_torchao_unavailable_reason_is_resolved_once_and_covers_the_stub(monkeypatch):
    monkeypatch.setattr(tq, "_TORCHAO_UNAVAILABLE", None)
    monkeypatch.setattr(tq, "is_stubbed", lambda pkg: True)
    reason = tq.torchao_unavailable_reason()
    assert reason is not None and "stub" in reason
    # Cached: flipping the stub answer does not re-resolve within a process.
    monkeypatch.setattr(tq, "is_stubbed", lambda pkg: False)
    assert tq.torchao_unavailable_reason() == reason

    monkeypatch.setattr(tq, "_TORCHAO_UNAVAILABLE", None)
    monkeypatch.setitem(
        sys.modules, "torchao.quantization", types.ModuleType("torchao.quantization")
    )
    sys.modules["torchao.quantization"].quantize_ = lambda *a, **k: None
    assert tq.torchao_unavailable_reason() is None


def test_the_smoke_probe_feeds_zero_rows_not_only_noise(monkeypatch):
    # The finiteness check above is only meaningful if the input actually contains a zero row:
    # torch.randn alone never produces one, which is exactly how the silent degradation survived.
    tq._SMOKE_CACHE.clear()
    seen = {}

    class _Lin:
        def __init__(self, *a, **k):
            pass

        def to(self, **k):
            return self

        def __call__(self, x):
            seen["x"] = x
            return x

    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.nn = types.SimpleNamespace(Linear = _Lin)
    torch.randn = lambda *a, **k: _FakeTensor()
    torch.isfinite = lambda t: _FakeBool(True)
    torch.no_grad = lambda: __import__("contextlib").nullcontext()
    torch.cuda = types.SimpleNamespace(is_available = lambda: True, synchronize = lambda: None)
    monkeypatch.setitem(sys.modules, "torch", torch)

    tqz = types.ModuleType("torchao.quantization")
    tqz.quantize_ = lambda module, config, filter_fn = None: None
    tqz.Int8DynamicActivationInt8WeightConfig = lambda: "int8cfg"
    tqz.Float8DynamicActivationFloat8WeightConfig = lambda: "fp8cfg"
    monkeypatch.setitem(sys.modules, "torchao.quantization", tqz)

    assert tq._smoke_probe(TQ_INT8, "cuda") is True
    assert seen["x"].zeroed, "probe input was never zeroed anywhere"
    assert all(value == 0 for _, value in seen["x"].zeroed)


# ── consumer-vs-datacenter detection (fp8 fast-accumulate gate) ──────────────────


def _stub_device_name(monkeypatch, name):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(get_device_name = lambda device = None: name)
    monkeypatch.setitem(sys.modules, "torch", torch)


@pytest.mark.parametrize(
    "name",
    [
        "NVIDIA GeForce RTX 5090",
        "NVIDIA GeForce RTX 4090",
        "NVIDIA RTX A4000",  # workstation: A4000 token, NOT the data-center A40
        "NVIDIA RTX A5000",  # workstation: A5000 token, not professional/datacenter
        "NVIDIA Some Future Card 9000",  # unknown -> default consumer (fast accum is free on DC)
    ],
)
def test_is_consumer_gpu_true(monkeypatch, name):
    _stub_device_name(monkeypatch, name)
    assert tq._is_consumer_gpu() is True


@pytest.mark.parametrize(
    "name",
    [
        "NVIDIA B200",
        "NVIDIA B300",  # Blackwell Ultra (matches llama_cpp datacenter regex)
        "NVIDIA GH200 480GB",  # Grace-Hopper superchip (was misread as consumer)
        "NVIDIA H100 80GB HBM3",
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA A40",  # data-center Ampere (distinct token from RTX A4000)
        "NVIDIA L40S",
        "NVIDIA L4",
        "Tesla V100-SXM2-16GB",
        "NVIDIA RTX PRO 6000 Blackwell Server Edition",  # professional -> datacenter-class
        "NVIDIA RTX 6000 Ada Generation",  # professional -> datacenter-class
    ],
)
def test_is_consumer_gpu_false_for_datacenter(monkeypatch, name):
    _stub_device_name(monkeypatch, name)
    assert tq._is_consumer_gpu() is False


def test_is_consumer_gpu_defaults_true_on_probe_failure(monkeypatch):
    # No torch / no device name available assumes consumer (safe: fast accum is free on data center and a win on consumer).
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace()  # no get_device_name
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert tq._is_consumer_gpu() is True


# ── filter ──────────────────────────────────────────────────────────────────────


def test_make_filter_fn(monkeypatch):
    class _Lin:
        def __init__(self, i, o):
            self.in_features, self.out_features = i, o

    torch = types.ModuleType("torch")
    torch.nn = types.SimpleNamespace(Linear = _Lin)
    monkeypatch.setitem(sys.modules, "torch", torch)

    keep = make_filter_fn(512)
    assert keep(_Lin(1024, 4096), "blocks.0.attn.to_q") is True
    assert keep(_Lin(256, 4096), "time_proj") is False  # small in_features -> skip
    assert keep(_Lin(4096, 256), "out_proj") is False  # small out_features -> skip
    assert keep(object(), "not_linear") is False  # non-Linear -> skip
    assert keep(types.SimpleNamespace(), "no_attrs") is False


def test_require_bf16_schemes_excludes_nvfp4():
    # fp8 and mxfp8 assert a bf16 weight (torchao 0.17 / B200) so they gate on it; nvfp4 quantises fp32 fine and keeps its large fp32 projections.
    from core.inference.diffusion_transformer_quant import (
        _REQUIRE_BF16_SCHEMES,
        TQ_FP8,
        TQ_MXFP8,
        TQ_NVFP4,
        TQ_INT8,
    )

    assert TQ_FP8 in _REQUIRE_BF16_SCHEMES
    assert TQ_MXFP8 in _REQUIRE_BF16_SCHEMES
    assert TQ_NVFP4 not in _REQUIRE_BF16_SCHEMES
    assert TQ_INT8 not in _REQUIRE_BF16_SCHEMES


def test_make_filter_fn_require_bf16_skips_non_bf16(monkeypatch):
    # fp8 / mxfp8 assert a bf16 weight, so require_bf16 must skip an fp32 Linear (Wan / Hunyuan video DiTs keep some) or one such layer raises inside quantize_ and no-ops the whole pass.
    torch = types.ModuleType("torch")
    torch.bfloat16, torch.float32 = "bf16", "fp32"

    class _Lin:
        def __init__(self, i, o, dtype):
            self.in_features, self.out_features = i, o
            self.weight = types.SimpleNamespace(dtype = dtype)

    torch.nn = types.SimpleNamespace(Linear = _Lin)
    monkeypatch.setitem(sys.modules, "torch", torch)

    gated = make_filter_fn(512, require_bf16 = True)
    assert gated(_Lin(1024, 4096, torch.bfloat16), "blocks.0.attn.to_q") is True
    assert gated(_Lin(1024, 4096, torch.float32), "blocks.0.attn.to_q") is False  # fp32 -> skip
    assert gated(types.SimpleNamespace(in_features = 1024, out_features = 4096), "no_weight") is False
    # int8 (require_bf16 off, the default) still quantises the fp32 linear.
    assert make_filter_fn(512)(_Lin(1024, 4096, torch.float32), "blocks.0.attn.to_q") is True


def test_make_filter_fn_int8_excludes_modulation_and_embedders(monkeypatch):
    # The int8 path skips the M=1 AdaLN modulation / conditioning-embedder projections (below torch._int_mm's M floor of 16) while keeping the attention / FFN and sequence embedders. fp8 keeps everything.
    from core.inference.diffusion_transformer_quant import _INT8_EXCLUDE_NAME_TOKENS

    class _Lin:
        def __init__(self, i, o):
            self.in_features, self.out_features = i, o

    torch = types.ModuleType("torch")
    torch.nn = types.SimpleNamespace(Linear = _Lin)
    monkeypatch.setitem(sys.modules, "torch", torch)

    keep = make_filter_fn(512, exclude_name_tokens = _INT8_EXCLUDE_NAME_TOKENS)
    big = lambda: _Lin(3072, 18432)  # noqa: E731 — large enough to pass min_features
    # Excluded (M=1 modulation / conditioning embedders), despite large features:
    for fqn in (
        "transformer_blocks.0.norm1.linear",
        "transformer_blocks.0.norm1_context.linear",
        "single_transformer_blocks.0.norm.linear",
        "norm_out.linear",
        "transformer_blocks.0.img_mod.1",
        "transformer_blocks.0.txt_mod.1",
        "double_stream_modulation_img.linear",
        "time_text_embed.timestep_embedder.linear_2",
        "time_text_embed.guidance_embedder.linear_2",
        "time_guidance_embed.timestep_embedder.linear_2",
    ):
        assert keep(big(), fqn) is False, fqn
    # Kept (M=seq compute layers + sequence embedders), NOT matched by the modulation tokens:
    for fqn in (
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.ff.net.0.proj",
        "single_transformer_blocks.0.proj_mlp",
        "single_transformer_blocks.0.attn.to_qkv_mlp_proj",
        "context_embedder",  # "context" contains "text" -> must NOT be excluded
        "txt_in",
    ):
        assert keep(big(), fqn) is True, fqn
    # Without the exclusion (fp8 path), the modulation layer is kept.
    assert make_filter_fn(512)(big(), "transformer_blocks.0.norm1.linear") is True
    # A None / empty fqn must not crash the exclusion check; with no name nothing matches, so it is kept.
    assert keep(big(), None) is True
    assert keep(big(), "") is True


def test_exclude_tokens_for_scheme_shared_by_runtime_and_builder():
    # The runtime quantiser and the offline prequant builder must apply the SAME int8 exclusion, else an int8 artifact bakes the M=1 linears and reintroduces the crash.
    from core.inference.diffusion_transformer_quant import (
        _INT8_EXCLUDE_NAME_TOKENS,
        exclude_tokens_for_scheme,
    )
    assert exclude_tokens_for_scheme(TQ_INT8) == _INT8_EXCLUDE_NAME_TOKENS
    for scheme in (TQ_FP8, TQ_NVFP4, TQ_MXFP8):
        assert exclude_tokens_for_scheme(scheme) == ()


def test_exclude_tokens_for_scheme():
    # The shared scheme-to-exclusion decision for both paths: int8 excludes the M=1 modulation / embedder tokens, every scaled_mm scheme excludes none.
    from core.inference.diffusion_transformer_quant import (
        _INT8_EXCLUDE_NAME_TOKENS,
        exclude_tokens_for_scheme,
    )

    assert exclude_tokens_for_scheme(TQ_INT8) == _INT8_EXCLUDE_NAME_TOKENS
    assert exclude_tokens_for_scheme(TQ_FP8) == ()
    assert exclude_tokens_for_scheme(TQ_NVFP4) == ()
    assert exclude_tokens_for_scheme(TQ_MXFP8) == ()


def test_exclude_tokens_for_scheme_family():
    # Qwen-Image never pads its text stream (unlike FLUX's 512-token T5), so short prompts run those linears at M < 16 and torch._int_mm raises: they stay bf16 while the M ~ 4k image stream keeps int8.
    from core.inference.diffusion_transformer_quant import (
        _INT8_EXCLUDE_NAME_TOKENS,
        _QWENIMAGE_INT8_EXCLUDES,
        exclude_tokens_for_scheme,
    )

    for fam in ("qwen-image", "qwen-image-edit"):
        assert (
            exclude_tokens_for_scheme(TQ_INT8, fam)
            == _INT8_EXCLUDE_NAME_TOKENS + _QWENIMAGE_INT8_EXCLUDES
        )
    for token in ("txt_in", "add_q_proj", "to_add_out", "txt_mlp"):
        assert token in _QWENIMAGE_INT8_EXCLUDES
    assert exclude_tokens_for_scheme(TQ_INT8, "z-image") == _INT8_EXCLUDE_NAME_TOKENS
    assert exclude_tokens_for_scheme(TQ_FP8, "qwen-image") == ()


# ── apply ───────────────────────────────────────────────────────────────────────


def test_resolve_fast_accum(monkeypatch):
    # None means fast accumulate on every GPU class; an explicit bool forces it. Deriving it from the GPU class made fp8 2.05x slower than int8 on RTX 6000 Ada, and on B200 the flag is a measured no-op.
    for consumer in (True, False):
        monkeypatch.setattr(tq, "_is_consumer_gpu", lambda *a, _c = consumer: _c)
        assert tq._resolve_fast_accum(None) is True
    assert tq._resolve_fast_accum(True) is True
    assert tq._resolve_fast_accum(False) is False  # precise accumulate stays available explicitly


def test_fp8_config_uses_per_row_granularity():
    """FP8 must use PerRow (per-token activation + per-channel weight) scaling. torchao's
    default is per-TENSOR: on a DiT with extreme activation outliers (z-image's ~6.6e4) one
    outlier forces a tensor-wide scale that pushes normal values below fp8 resolution and the
    denoise collapses to noise. This is the regression guard for that fix (validated on B200:
    per-tensor fp8 = noise, per-row fp8 = matches bf16)."""
    torchao_quant = pytest.importorskip("torchao.quantization")
    per_row = getattr(torchao_quant, "PerRow", None)
    if per_row is None:
        pytest.skip("torchao build without PerRow granularity")
    cfg = tq._make_quant_config(TQ_FP8)
    gran = getattr(cfg, "granularity", None)
    assert gran is not None, "fp8 config must set an explicit granularity, not torchao's default"
    grans = gran if isinstance(gran, (list, tuple)) else [gran]
    assert grans and all(isinstance(g, per_row) for g in grans), f"expected all PerRow, got {gran}"


def test_fp8_config_pins_torch_kernel_preference():
    """FP8 must pin KernelPreference.TORCH. The AUTO default silently switches the weight
    quantize to the MSLK kernel whenever an mslk package is importable, which changes fp8
    scale rounding bitwise (measured 8/8 FLUX matrices differ) and would break the hosted
    prequant bit-identity invariant; the mslk path is also slower under torch.compile."""
    pytest.importorskip("torchao.quantization")
    try:
        from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
    except Exception:
        pytest.skip("torchao build without KernelPreference")
    cfg = tq._make_quant_config(TQ_FP8)
    if not hasattr(cfg, "kernel_preference"):
        pytest.skip("torchao config without kernel_preference")
    assert cfg.kernel_preference == KernelPreference.TORCH


def test_quantize_transformer_applies_and_marks(monkeypatch):
    monkeypatch.setattr(
        tq, "select_transformer_quant_scheme", lambda target, mode, family = None: TQ_FP8
    )
    seen: dict = {}

    def _mk(scheme, fast_accum = None):
        seen["scheme"], seen["fast_accum"] = scheme, fast_accum
        return f"{scheme}cfg"

    monkeypatch.setattr(tq, "_make_quant_config", _mk)
    recorder: list = []
    tqz = types.ModuleType("torchao.quantization")
    tqz.quantize_ = lambda module, config, filter_fn = None: recorder.append(
        (module, config, filter_fn)
    )
    monkeypatch.setitem(sys.modules, "torchao.quantization", tqz)

    transformer = types.SimpleNamespace()
    pipe = types.SimpleNamespace(transformer = transformer)
    assert quantize_transformer(pipe, _target(), mode = "fp8", fast_accum = False) == TQ_FP8
    assert len(recorder) == 1 and recorder[0][0] is transformer and recorder[0][1] == "fp8cfg"
    assert callable(recorder[0][2])  # a filter_fn was passed
    assert transformer._unsloth_runtime_quant == TQ_FP8  # diagnostic marker set
    assert seen["fast_accum"] is False  # the override is forwarded into the config


def test_quantize_transformer_none_when_unsupported(monkeypatch):
    monkeypatch.setattr(
        tq, "select_transformer_quant_scheme", lambda target, mode, family = None: None
    )
    pipe = types.SimpleNamespace(transformer = types.SimpleNamespace())
    assert quantize_transformer(pipe, _target(), mode = "auto") is None


def test_quantize_transformer_tolerates_failure(monkeypatch):
    monkeypatch.setattr(
        tq, "select_transformer_quant_scheme", lambda target, mode, family = None: TQ_INT8
    )
    monkeypatch.setattr(tq, "_make_quant_config", lambda scheme: "cfg")
    tqz = types.ModuleType("torchao.quantization")

    def _boom(
        module,
        config,
        filter_fn = None,
    ):
        raise RuntimeError("partial quant failure")

    tqz.quantize_ = _boom
    monkeypatch.setitem(sys.modules, "torchao.quantization", tqz)
    pipe = types.SimpleNamespace(transformer = types.SimpleNamespace())
    # A quantise failure returns None (the caller falls back to GGUF), never raises.
    assert quantize_transformer(pipe, _target(), mode = "int8") is None


# ── family scheme deny (measured model-level breakage) ────────────────────────


def test_family_deny_auto_skips_mx_and_nvfp4_for_qwen(monkeypatch):
    # B200 with every scheme available: mxfp8 and nvfp4 still damage the Qwen DiT, so auto skips
    # them. fp8 is no longer denied (activation_value_lb fixed the black frames), so auto now
    # takes fp8 first on a data-center part rather than falling all the way to int8.
    _stub_torch(monkeypatch, cc = (10, 0))
    _allow(monkeypatch, {TQ_FP8, TQ_NVFP4, TQ_MXFP8, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto", family = "qwen-image") == TQ_FP8
    assert select_transformer_quant_scheme(_target(), "auto", family = "qwen-image-edit") == TQ_FP8
    # With fp8 unavailable the deny still bites: mxfp8 / nvfp4 are skipped and int8 is the pick.
    _allow(monkeypatch, {TQ_NVFP4, TQ_MXFP8, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto", family = "qwen-image") == TQ_INT8


def test_family_deny_refuses_explicit_mxfp8_and_nvfp4_for_qwen(monkeypatch):
    # An explicit denied scheme returns None (same contract as an unsupported scheme). fp8 and int8
    # are both honored on qwen now, and fp8 outside the deny table is unaffected.
    _stub_torch(monkeypatch, cc = (10, 0))
    _allow(monkeypatch, {TQ_FP8, TQ_MXFP8, TQ_NVFP4, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "mxfp8", family = "qwen-image") is None
    assert select_transformer_quant_scheme(_target(), "nvfp4", family = "qwen-image") is None
    assert select_transformer_quant_scheme(_target(), "fp8", family = "qwen-image") == TQ_FP8
    assert select_transformer_quant_scheme(_target(), "int8", family = "qwen-image") == TQ_INT8
    assert select_transformer_quant_scheme(_target(), "fp8", family = "z-image") == TQ_FP8


def test_family_deny_no_family_keeps_ladder(monkeypatch):
    # Without a family (or an unknown one) the ladder is unchanged: fp8 first on B200.
    _stub_torch(monkeypatch, cc = (10, 0))
    _allow(monkeypatch, {TQ_FP8, TQ_INT8})
    assert select_transformer_quant_scheme(_target(), "auto") == TQ_FP8
    assert select_transformer_quant_scheme(_target(), "auto", family = "sdxl") == TQ_FP8


def test_quantize_transformer_threads_family(monkeypatch):
    # quantize_transformer passes the family down to the selector, so a denied (family, scheme) pair never reaches torchao.
    # mxfp8, not fp8: fp8 on qwen-image is no longer denied, so it would reach torchao and prove nothing.
    _stub_torch(monkeypatch, cc = (10, 0))
    _allow(monkeypatch, {TQ_FP8, TQ_INT8})
    pipe = types.SimpleNamespace(transformer = types.SimpleNamespace())
    called = {}
    tqz = types.ModuleType("torchao.quantization")

    def _quantize(
        module,
        config,
        filter_fn = None,
    ):
        called["scheme"] = True

    tqz.quantize_ = _quantize
    tqz.Int8DynamicActivationInt8WeightConfig = lambda: "int8-cfg"
    tqz.Float8DynamicActivationFloat8WeightConfig = lambda **kw: "fp8-cfg"
    tqz.PerRow = lambda: "per-row"
    monkeypatch.setitem(sys.modules, "torchao.quantization", tqz)
    assert quantize_transformer(pipe, _target(), mode = "mxfp8", family = "qwen-image") is None
    assert called == {}


def test_the_attention_trim_families_exclude_their_small_m_text_streams():
    """The trim in this PR is what makes these excludes necessary, so they ship together.

    It shrinks HunyuanVideo-1.5's text / image streams from padded length to valid tokens, and
    quantize_transformer runs BEFORE the trim hook is installed, so those tiny-M activations flow
    through already-int8 linears: M = 0 comes back unprojected (torchao passes the input through,
    so the 2048-wide cond-type add crashes) and M <= 16 trips torch._int_mm's floor."""
    from core.inference.diffusion_transformer_quant import TQ_INT8, exclude_tokens_for_scheme

    for family in ("hunyuanvideo-1.5", "hunyuanvideo-1.5-720p"):
        tokens = exclude_tokens_for_scheme(TQ_INT8, family)
        for name in (
            "context_embedder",  # also matches context_embedder_2, by substring
            "image_embedder",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
            "ff_context",
        ):
            assert name in tokens, f"{family} must exclude {name}"

    # Only int8 has the M floor: the per-row scaled_mm schemes are unaffected, and an unrelated
    # family keeps exactly the generic set.
    from core.inference.diffusion_transformer_quant import _INT8_EXCLUDE_NAME_TOKENS

    assert exclude_tokens_for_scheme("fp8", "hunyuanvideo-1.5") == ()
    # flux.1 stands in for the unrelated family here. ltx-2 no longer can: it is audiovisual, and
    # a video-only run feeds a one-token audio stream that hits the same M floor, so it now carries
    # its own audio exclusions.
    assert exclude_tokens_for_scheme(TQ_INT8, "flux.1") == _INT8_EXCLUDE_NAME_TOKENS
    assert exclude_tokens_for_scheme(TQ_INT8, None) == _INT8_EXCLUDE_NAME_TOKENS


def test_minimax_h3_int8_excludes_its_adaln_projection():
    """H3's adaLN projection is named ``adaln_proj``, which the generic token list does not match.

    "norm" is the closest generic token and it does not appear in the name, so on the DENSE
    checkpoint Linear(2688 -> 96768) clears min_features = 512, gets quantized, then runs at M = 1
    and raises "self.size(0) needs to be greater than 16, but got 1" at the first denoise. Measured:
    the offline builder bakes it and torch.compile dies on that module.

    The pruned-modulation form hides this rather than fixing it, since there adaln_proj is
    Linear(8 -> 96768) and falls under min_features anyway (verified on the fl2va_pruned build:
    all 51 of them are rejected for min_features, in_features = 8). So this exclusion is what
    makes the DENSE path correct and is a no-op on the pruned one.
    """
    from core.inference.diffusion_transformer_quant import TQ_INT8, exclude_tokens_for_scheme

    assert "adaln_proj" in exclude_tokens_for_scheme(TQ_INT8, "minimax-h3")

    # The generic list genuinely does not cover adaln_proj, which is why the entry is needed at all.
    # If a future generic token starts matching it, this assertion fails and the family entry can
    # be reconsidered rather than left as dead weight.
    from core.inference.diffusion_transformer_quant import _INT8_EXCLUDE_NAME_TOKENS

    assert not any(t in "adaln_proj" for t in _INT8_EXCLUDE_NAME_TOKENS)
    # fp8 has no M floor, so it must not inherit any of this.
    assert exclude_tokens_for_scheme("fp8", "minimax-h3") == ()


def test_minimax_h3_pads_its_text_stream_instead_of_excluding_it():
    """context_embedder and the two token_refiner blocks are QUANTIZED and padded, not skipped.

    They run at M = text tokens (10..19 across the seven eval prompts), which straddles
    ``_int_mm``'s floor of 16, so they used to be excluded. Padding the activation up to 32 rows
    is bitwise exact under per-row activation scaling and recovers 0.80 GB of weights, so the
    two names moved from the exclude list to the pad list. Both halves are asserted here: a
    change that dropped one without the other would either crash under compile or silently
    quantise nothing."""
    from core.inference.diffusion_transformer_quant import (
        TQ_INT8,
        exclude_tokens_for_scheme,
        pad_tokens_for_scheme,
    )

    pad = pad_tokens_for_scheme(TQ_INT8, "minimax-h3")
    exclude = exclude_tokens_for_scheme(TQ_INT8, "minimax-h3")
    for name in ("context_embedder", "token_refiner"):
        assert name in pad, f"minimax-h3 int8 must pad {name}"
        assert name not in exclude, f"{name} is padded, so excluding it would quantise nothing"


def test_pad_and_exclude_sets_never_overlap():
    """An excluded Linear is never quantized, so there would be nothing to pad. A name in both
    lists means one of them is dead, and which one is dead is not visible at the call site."""
    from core.inference.diffusion_transformer_quant import (
        _INT8_FAMILY_PAD_NAME_TOKENS,
        TQ_INT8,
        exclude_tokens_for_scheme,
        pad_tokens_for_scheme,
    )
    for family in _INT8_FAMILY_PAD_NAME_TOKENS:
        exclude = exclude_tokens_for_scheme(TQ_INT8, family)
        for pad_token in pad_tokens_for_scheme(TQ_INT8, family):
            assert not any(
                e in pad_token or pad_token in e for e in exclude
            ), f"{family}: {pad_token!r} is both padded and excluded"


def test_only_minimax_h3_pads_today():
    """Scoped deliberately. qwen-image, qwen-image-edit and hunyuanvideo-1.5 have the same
    small-M shape and could adopt this, but each has a PUBLISHED int8 prequant checkpoint whose
    metadata bakes the current exclusion set, and ``_validate_checkpoint`` compares that set
    against ``exclude_tokens_for_scheme``. Flipping one of them without rebuilding and
    republishing its artifact turns every hosted int8 load into a silent fallback."""
    from core.inference.diffusion_transformer_quant import TQ_INT8, pad_tokens_for_scheme

    for family in ("qwen-image", "qwen-image-edit", "hunyuanvideo-1.5", "hunyuanvideo-1.5-720p"):
        assert pad_tokens_for_scheme(TQ_INT8, family) == ()
    assert pad_tokens_for_scheme(TQ_INT8, "z-image") == ()
    assert pad_tokens_for_scheme(TQ_INT8, None) == ()


def test_only_int8_pads():
    """``_int_mm``'s row floor is int8's alone: scaled_mm and the MX/FP4 kernels have no
    equivalent, so no other scheme should be paying for a pad-and-slice."""
    from core.inference.diffusion_transformer_quant import pad_tokens_for_scheme
    for scheme in ("fp8", "nvfp4", "mxfp8", "auto"):
        assert pad_tokens_for_scheme(scheme, "minimax-h3") == ()


def test_quantize_transformer_pads_after_quantising(monkeypatch):
    """The padding runs on the RUNTIME dense-quantise path too, not only on the prequant one,
    and it runs AFTER quantize_ (it reparents Linears that must already hold quantized weights).
    """
    _stub_torch(monkeypatch, cc = (10, 0))
    _allow(monkeypatch, {TQ_INT8})
    order = []
    tqz = types.ModuleType("torchao.quantization")
    tqz.quantize_ = lambda module, config, filter_fn = None: order.append("quantize_")
    tqz.Int8DynamicActivationInt8WeightConfig = lambda: "int8-cfg"
    tqz.Float8DynamicActivationFloat8WeightConfig = lambda **kw: "fp8-cfg"
    tqz.PerRow = lambda: "per-row"
    monkeypatch.setitem(sys.modules, "torchao.quantization", tqz)
    monkeypatch.setattr(
        tq,
        "apply_small_m_padding",
        lambda transformer, scheme, family = None, logger = None: (
            order.append(("pad", scheme, family)) or ()
        ),
    )
    pipe = types.SimpleNamespace(transformer = types.SimpleNamespace())
    assert quantize_transformer(pipe, _target(), mode = "int8", family = "minimax-h3") == TQ_INT8
    assert order == ["quantize_", ("pad", TQ_INT8, "minimax-h3")]


def test_quantize_transformer_refuses_when_padding_cannot_be_proven(monkeypatch):
    """A half-padded transformer compiles on the modules that were wrapped and crashes inside
    ``_int_mm`` on the ones that were not, so a raise from the padding must fail the whole
    quantise and send the caller to GGUF -- not be swallowed into a partially padded model."""
    _stub_torch(monkeypatch, cc = (10, 0))
    _allow(monkeypatch, {TQ_INT8})
    tqz = types.ModuleType("torchao.quantization")
    tqz.quantize_ = lambda module, config, filter_fn = None: None
    tqz.Int8DynamicActivationInt8WeightConfig = lambda: "int8-cfg"
    tqz.Float8DynamicActivationFloat8WeightConfig = lambda **kw: "fp8-cfg"
    tqz.PerRow = lambda: "per-row"
    monkeypatch.setitem(sys.modules, "torchao.quantization", tqz)

    def _boom(
        transformer,
        scheme,
        family = None,
        logger = None,
    ):
        raise RuntimeError("cannot prove per-row granularity")

    monkeypatch.setattr(tq, "apply_small_m_padding", _boom)
    pipe = types.SimpleNamespace(transformer = types.SimpleNamespace())
    assert quantize_transformer(pipe, _target(), mode = "int8", family = "minimax-h3") is None
    assert not hasattr(pipe.transformer, "_unsloth_runtime_quant")


def test_apply_small_m_padding_is_inert_without_a_pad_list(monkeypatch):
    """No pad list means the padding module is never even IMPORTED.

    That matters beyond the wasted traversal: ``diffusion_transformer_quant`` deliberately keeps
    torch out of its own import path (every probe imports lazily), while ``diffusion_quant_pad``
    subclasses ``nn.Module`` and so must import torch at module scope. Shadowing the module with
    an empty stub makes the import observable: it raises for the family that pads, and must not
    be reached at all for anything else."""
    from core.inference.diffusion_transformer_quant import TQ_INT8, apply_small_m_padding

    stub = types.ModuleType("core.inference.diffusion_quant_pad")  # no names to import
    monkeypatch.setitem(sys.modules, "core.inference.diffusion_quant_pad", stub)

    assert apply_small_m_padding(object(), TQ_INT8, "z-image") == ()
    assert apply_small_m_padding(object(), "fp8", "minimax-h3") == ()
    assert apply_small_m_padding(object(), TQ_INT8, None) == ()
    with pytest.raises(ImportError):
        apply_small_m_padding(object(), TQ_INT8, "minimax-h3")


def test_the_training_deny_is_a_superset_of_the_inference_deny():
    # The two tables are separate because rendering evidence is not training evidence, but the
    # relationship must only ever go one way: anything inference refuses, training refuses too.
    # A regression making training MORE permissive than inference would let a scheme that cannot
    # even render reach a trainer, which is the one direction this split must not allow.
    from core.inference.diffusion_transformer_quant import (
        _FAMILY_SCHEME_DENY,
        TQ_SCHEMES,
        _family_denied,
        _family_train_denied,
    )

    families = set(_FAMILY_SCHEME_DENY) | {"qwen-image", "qwen-image-edit", "z-image", "sdxl", ""}
    for fam in families:
        for scheme in TQ_SCHEMES:
            if _family_denied(fam, scheme):
                assert _family_train_denied(fam, scheme), (fam, scheme)

    # And the specific split this change introduces: qwen-image fp8 renders (gate 28/28) but is not
    # cleared for training, so inference allows it and training does not.
    for fam in ("qwen-image", "qwen-image-edit"):
        assert not _family_denied(fam, TQ_FP8)
        assert _family_train_denied(fam, TQ_FP8)
        # int8 was never denied on either side and must stay available.
        assert not _family_train_denied(fam, TQ_INT8)


def test_auto_scheme_candidates_lists_the_whole_ladder_not_just_the_winner(monkeypatch):
    # select_transformer_quant_scheme returns one winner. When that winner has no hosted prequant
    # AND cannot fit dense, the loader needs to know what auto would have picked NEXT, or the pick
    # drops to GGUF even though a lower rung would have loaded. Same ladder, deny list and probe.
    from core.inference.diffusion_transformer_quant import auto_scheme_candidates

    _stub_torch(monkeypatch, cc = (10, 0))
    _allow(monkeypatch, {TQ_FP8, TQ_MXFP8, TQ_INT8})
    assert auto_scheme_candidates(_target()) == (TQ_FP8, TQ_MXFP8, TQ_INT8)
    # The deny list still applies: qwen-image keeps mxfp8 out, so fp8 then int8.
    assert auto_scheme_candidates(_target(), "qwen-image") == (TQ_FP8, TQ_INT8)
    # Whatever the probe refuses is absent, so the list can never offer an unusable scheme.
    _allow(monkeypatch, {TQ_INT8})
    assert auto_scheme_candidates(_target(), "qwen-image") == (TQ_INT8,)
    # A target the dense path cannot use has no candidates at all.
    assert auto_scheme_candidates(_target(device = "cpu")) == ()


def test_the_candidate_list_agrees_with_the_selector_on_the_winner(monkeypatch):
    # The two must never disagree about what auto is allowed to pick, so the selector's answer is
    # always the head of the candidate list. A drift here would let the retry path propose a scheme
    # auto itself would refuse.
    from core.inference.diffusion_transformer_quant import auto_scheme_candidates
    for cc, allowed, family in (
        ((10, 0), {TQ_FP8, TQ_MXFP8, TQ_INT8}, None),
        ((10, 0), {TQ_FP8, TQ_MXFP8, TQ_INT8}, "qwen-image"),
        ((8, 9), {TQ_FP8, TQ_INT8}, "qwen-image-edit"),
        ((8, 0), {TQ_INT8}, None),
    ):
        _stub_torch(monkeypatch, cc = cc)
        _allow(monkeypatch, allowed)
        chosen = select_transformer_quant_scheme(_target(), "auto", family = family)
        candidates = auto_scheme_candidates(_target(), family)
        assert (candidates[0] if candidates else None) == chosen, (cc, family)


def test_the_pre_eviction_gate_does_not_refuse_on_an_indeterminate_probe(monkeypatch):
    # The route-level precision gate asks the selector, not the probe, so unproven_ok has to reach
    # through select_transformer_quant_scheme for the leniency to exist where it matters.
    monkeypatch.setattr(tq, "dense_transformer_supported", lambda target: True)
    seen: list = []

    def _supported(
        scheme,
        device,
        *,
        unproven_ok = False,
    ):
        seen.append(unproven_ok)
        return unproven_ok

    monkeypatch.setattr(tq, "_scheme_supported", _supported)
    target = types.SimpleNamespace(device = "cuda")
    assert tq.select_transformer_quant_scheme(target, "fp8") is None
    assert tq.select_transformer_quant_scheme(target, "fp8", unproven_ok = True) == "fp8"
    assert seen == [False, True]


def test_a_refusal_reason_does_not_carry_server_paths(monkeypatch):
    # The torchao import error is interpolated into the precision-refusal RuntimeError, which both
    # load routes return verbatim as the 409 detail. An ImportError routinely names the absolute
    # file that raised it, so the reason has to be stripped while the log keeps the whole thing.
    monkeypatch.setattr(tq, "_TORCHAO_UNAVAILABLE", None)
    monkeypatch.setattr(tq, "is_stubbed", lambda pkg: False)
    broken = types.ModuleType("torchao.quantization")

    def _raise():
        raise ImportError(
            "cannot import name 'ScalingType' from 'torch.nn.functional' "
            "(/srv/unsloth/.venv/lib/python3.12/site-packages/torch/nn/functional.py)"
        )

    broken.__getattr__ = lambda name: _raise()
    monkeypatch.setitem(sys.modules, "torchao.quantization", broken)
    reason = tq.torchao_unavailable_reason()
    assert reason is not None
    assert "/srv/unsloth" not in reason and "site-packages" not in reason
    # The actionable half survives: the caller still learns WHICH import broke.
    assert "ScalingType" in reason and "torch.nn.functional" in reason


def test_paths_are_stripped_without_eating_dotted_module_names():
    assert tq._strip_paths("ImportError: no module 'torchao.quantization'") == (
        "ImportError: no module 'torchao.quantization'"
    )
    assert "C:\\Users" not in tq._strip_paths(
        r"ImportError: DLL load failed: C:\Users\me\.venv\Lib\site-packages\torchao\_C.pyd"
    )
