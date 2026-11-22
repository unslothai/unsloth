# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cold-card probe prewarm; CPU only (kernels faked). Verdicts must not depend on the warm-up."""

from __future__ import annotations

import threading

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_convrot_quant as cq
from core.inference import diffusion_flux_convrot_fused as fcf
from core.inference import diffusion_int8_gemm as gemm


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(cq.PROBE_PREWARM_ENV, raising = False)
    for name in (
        cq.INT8_ROTQUANT_ENV,
        cq.INT8_ROTQUANT_H16_ENV,
        cq.INT8_ROTQUANT_SWIGLU_ENV,
        cq.INT8_ROTQUANT_PRODUCER_ENV,
        "UNSLOTH_DIFFUSION_INT8_GEMM",
        "UNSLOTH_DIFFUSION_INT8_GEMM_TILES",
    ):
        monkeypatch.delenv(name, raising = False)
    for name in (
        "_ROTQ_DEVICE",
        "_ROTQ_DEVICE_K",
        "_H16_DEVICE",
        "_SWIGLU_DEVICE",
        "_PRODUCER_DEVICE",
    ):
        monkeypatch.setattr(cq, name, {})
    monkeypatch.setattr(cq, "PREWARM_COUNTS", {})
    monkeypatch.setattr(gemm, "_DEVICE_CFG", {})
    monkeypatch.setattr(gemm, "_DEVICE_TILES", {})
    monkeypatch.setattr(gemm, "_DEVICE_TMA", {})
    monkeypatch.setattr(cq, "_verdict_load", lambda index: None)
    monkeypatch.setattr(cq, "_verdict_store", lambda index, update: None)


def _fake_card(monkeypatch, cap = (8, 9)):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: cap)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    class _Ctx:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(torch.cuda, "device", _Ctx)
    monkeypatch.setattr(cq, "_triton_ok", lambda: True)
    monkeypatch.setattr(cq, "_kernels", lambda: object())
    monkeypatch.setattr(cq, "_rotq_op", lambda: "rotq")
    monkeypatch.setattr(cq, "_swiglu_op", lambda: "swiglu")
    monkeypatch.setattr(cq, "_producer_op", lambda: "producer")
    monkeypatch.setattr(cq, "is_rocm_torch", lambda t: False)
    import core.inference.diffusion_cuda_graph as cg

    monkeypatch.setattr(cg, "synchronize_outside_capture", lambda dev = None: None)


def _record_launches(monkeypatch):
    seen = []
    lock = threading.Lock()

    def note(key):
        with lock:
            seen.append((key, threading.current_thread().name))

    monkeypatch.setattr(
        cq, "_warm_input", lambda m, k, dev, seed = 0: torch.zeros(m, k, dtype = torch.bfloat16)
    )
    monkeypatch.setattr(
        cq, "_rotq_launch", lambda x, group, kind, cfg: note(("rotq", tuple(x.shape), kind, cfg))
    )
    monkeypatch.setattr(
        cq,
        "_h16_launch",
        lambda x, kind, v = None, tf32x3 = None: note(
            ("swiglu" if v is not None else "h16", tuple(x.shape), kind, tf32x3)
        ),
    )
    monkeypatch.setattr(
        cq,
        "_producer_probe_inputs",
        lambda m, pre, f, g, dev: (pre, torch.zeros(m, f), torch.zeros(m, f)),
    )
    monkeypatch.setattr(
        cq,
        "_h16x_launch",
        lambda prefix, gate, value, kind, tf32x3 = None: note(
            ("producer", (gate.shape[0], prefix, gate.shape[1]), kind, tf32x3)
        ),
    )
    return seen


def _expected(cap, producer):
    want = cq.rotquant_config(cap)
    cfgs = []
    for cfg in (want, cq._ROTQ_FALLBACK, *(rule[2] for rule in cq.rotq_k_tiles(cap))):
        if cfg not in cfgs:
            cfgs.append(cfg)
    tf32x3 = tuple(cap) in cq._H16_TF32X3_ARCHS
    out = set()
    for cfg in cfgs:
        for m, k in cq._ROTQ_PROBE_SHAPES:
            if cq._rotq_rows(k, 256, cfg) >= 1:
                out |= {("rotq", (m, k), kind, cfg) for kind in ("v1", "v2")}
    out |= {("h16", s, kind, tf32x3) for s in cq._H16_PROBE_SHAPES for kind in ("v1", "v2")}
    out |= {("swiglu", s, kind, tf32x3) for s in cq._SWIGLU_PROBE_SHAPES for kind in ("v1", "v2")}
    if producer:
        out |= {
            ("producer", (m, pre, f), kind, tf32x3)
            for m, pre, f in cq._PRODUCER_PROBE_SHAPES
            for kind in ("v1", "v2")
        }
    return out


@pytest.mark.parametrize("cap", [(8, 9), (8, 0), (12, 0)])
@pytest.mark.parametrize("producer", [False, True])
def test_warm_jobs_cover_every_cold_probe_launch(monkeypatch, cap, producer):
    _fake_card(monkeypatch, cap)
    seen = _record_launches(monkeypatch)
    if cq.rotquant_config(cap) is None:
        pytest.skip("no dense rotq tile on this arch")
    ran = cq.prewarm_probe_kernels(0, rotq = True, producer = producer)
    keys = {k for k, _ in seen}
    assert keys == _expected(cap, producer)
    assert ran == len(seen) == cq.PREWARM_COUNTS[0]
    assert all(name.startswith("unsloth-probe-prewarm") for _, name in seen)


def test_prewarm_spreads_over_threads_and_survives_failing_launches(monkeypatch):
    _fake_card(monkeypatch)
    names = set()
    gate = threading.Barrier(2, timeout = 5)

    def job_ok():
        names.add(threading.current_thread().name)
        gate.wait()  # two jobs must run at the same time

    def job_bad():
        raise RuntimeError("out of shared memory")

    monkeypatch.setattr(cq, "probe_warm_jobs", lambda index, rotq = True, producer = False: [])
    ran = cq.prewarm_probe_kernels(0, extra = (job_ok, job_ok, job_bad))
    assert ran == 2
    assert len(names) == 2


def test_kill_switch_and_capture_run_nothing(monkeypatch):
    _fake_card(monkeypatch)
    calls = []
    jobs = (lambda: calls.append(1), lambda: calls.append(2))
    monkeypatch.setenv(cq.PROBE_PREWARM_ENV, "0")
    assert cq.prewarm_probe_kernels(0, extra = jobs) == 0
    monkeypatch.delenv(cq.PROBE_PREWARM_ENV)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert cq.prewarm_probe_kernels(0, extra = jobs) == 0
    assert calls == []


def test_pending_follows_in_process_and_persisted_verdicts(monkeypatch):
    assert cq.rotq_probes_pending(0) and cq.producer_probe_pending(0)
    cq._ROTQ_DEVICE[0] = (64, 64, 4, 3)
    cq._PRODUCER_DEVICE[0] = False
    assert not cq.rotq_probes_pending(0) and not cq.producer_probe_pending(0)
    cq._ROTQ_DEVICE.clear()
    cq._PRODUCER_DEVICE.clear()
    monkeypatch.setattr(cq, "_verdict_load", lambda index: {"cfg": [1, 2, 3, 4], "producer": True})
    assert not cq.rotq_probes_pending(0) and not cq.producer_probe_pending(0)
    monkeypatch.setattr(cq, "_verdict_load", lambda index: {"cfg": [1, 2, 3, 4]})
    assert cq.producer_probe_pending(0)


def test_producer_expected_only_for_flux_denoisers(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DIFFUSION_FLUX_CONVROT_FUSED", raising = False)
    monkeypatch.setattr(fcf, "flux_convrot_fused_disabled", lambda: False)
    flux3 = type("Flux3Transformer2DModel", (), {})()
    qwen = type("QwenImageTransformer2DModel", (), {})()
    assert fcf.producer_probe_expected(flux3)
    assert not fcf.producer_probe_expected(qwen)
    monkeypatch.setattr(fcf, "flux_convrot_fused_disabled", lambda: True)
    assert not fcf.producer_probe_expected(flux3)


def test_gemm_probe_jobs_cover_device_config_candidates(monkeypatch):
    _fake_card(monkeypatch, (8, 9))
    monkeypatch.setattr(gemm, "_triton_version_ok", lambda version = None: True)
    monkeypatch.setattr(gemm, "_kernels", lambda: object())
    monkeypatch.setattr(gemm, "_triton38_sm8x", lambda cap: False)
    want = gemm.arch_config((8, 9))
    if want is None:
        pytest.skip("no int8 GEMM tile listed for sm89")
    probed = []
    monkeypatch.setattr(gemm, "_probe", lambda index, cfg, tma = False: probed.append(cfg) or True)
    for job in gemm._gemm_probe_jobs(0):
        job()
    expected = []
    for cfg in (want, gemm._FALLBACK_CONFIG, *(rule[4] for rule in gemm.shape_tiles((8, 9)))):
        if cfg not in expected:
            expected.append(cfg)
    assert probed == expected
    gemm._DEVICE_CFG[0] = want  # probed already: nothing to warm
    assert gemm._gemm_probe_jobs(0) == []


def test_swap_prewarms_before_the_serial_probes_and_keeps_their_verdicts(monkeypatch):
    order = []
    monkeypatch.setattr(
        gemm,
        "_prewarm_probes",
        lambda index, transformer, rotated, logger: order.append(("prewarm", rotated)) or 0,
    )
    monkeypatch.setattr(
        gemm, "device_config", lambda index: order.append(("device_config",)) or None
    )

    def rotq_cfg(index):
        order.append(("rotquant_device_config",))
        return None

    monkeypatch.setattr(cq, "rotquant_device_config", rotq_cfg)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: (8, 9))
    rec = ("v2", 256, True, torch.zeros(1))
    monkeypatch.setattr(
        gemm, "_eligible", lambda m: rec if isinstance(m, torch.nn.Linear) else None
    )
    monkeypatch.setattr(gemm, "_split_cards", lambda t: (0,))
    monkeypatch.setattr(gemm, "_cutlass_plan", lambda *a, **k: {})
    monkeypatch.setattr(gemm, "_v1_act_quant_matches", lambda index: True)
    monkeypatch.setattr(gemm, "_card_dtypes_ok", lambda m, r, f16: True)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    try:
        gemm._swap_on(model, None, 0)
    except Exception:  # noqa: BLE001 - the swap past the probes is not under test
        pass
    assert order[:3] == [("prewarm", True), ("device_config",), ("rotquant_device_config",)]
