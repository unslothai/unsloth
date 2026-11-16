# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A step CUDA graph is declined over an uncompiled denoiser whose native int8 W8A8 linears run compiled.

The compiled W8A8 function never runs inside a CUDA-graph capture, so the captured step bakes the eager rotate /
quantise / int_mm / rescale kernels. On a T4, FLUX.3 Klein int8 (fp16, whole-model compile declined) at Speed default
rendered 42.5 s per image with the graph and 34 s at Speed off; with the graph disabled, Speed default was back to 33.4 s.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_native_quant as nq  # noqa: E402
from core.inference import diffusion_speed as ds  # noqa: E402


def _dit(act_int8 = True):
    model = torch.nn.Sequential(
        torch.nn.Linear(256, 256, bias = False), torch.nn.Linear(256, 256, bias = False)
    )
    n = nq.apply_native_weight_quant(
        model, nq.NATIVE_INT8, filter_fn = lambda m, name: True, act_int8 = act_int8
    )
    assert n == 2
    return model


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setattr(ds, "torch_compile_runtime_available", lambda: True)
    monkeypatch.delenv(nq.NATIVE_INT8_FUSED_ENV, raising = False)
    monkeypatch.delenv(ds.STEP_GRAPH_NATIVE_W8A8_ENV, raising = False)
    monkeypatch.setitem(nq._FUSED_STATE, "failed", None)
    monkeypatch.setitem(nq._FUSED_STATE, "devices", {})


def test_a_card_whose_compiled_linear_failed_keeps_its_graph(runtime, monkeypatch):
    # A per-card failure leaves only eager kernels there, so there is no compiled linear to keep out of a graph.
    dit = _dit(act_int8 = True)
    card = nq._device_key(None)
    monkeypatch.setitem(nq._FUSED_STATE, "devices", {card: "RuntimeError('boom')"})
    assert not nq.fused_w8a8_serves(dit)
    monkeypatch.setitem(nq._FUSED_STATE, "devices", {card: None})
    assert nq.fused_w8a8_serves(dit)


def test_fused_w8a8_serves_only_act_int8_native_linears(runtime):
    assert nq.fused_w8a8_serves(_dit(act_int8 = True))
    assert not nq.fused_w8a8_serves(_dit(act_int8 = False))
    assert not nq.fused_w8a8_serves(torch.nn.Sequential(torch.nn.Linear(8, 8)))


def test_fused_w8a8_serves_follows_its_switches(runtime, monkeypatch):
    dit = _dit()
    monkeypatch.setenv(nq.NATIVE_INT8_FUSED_ENV, "0")
    assert not nq.fused_w8a8_serves(dit)
    monkeypatch.delenv(nq.NATIVE_INT8_FUSED_ENV)
    monkeypatch.setitem(nq._FUSED_STATE, "failed", "no inductor runtime")
    assert not nq.fused_w8a8_serves(dit)
    monkeypatch.setitem(nq._FUSED_STATE, "failed", None)
    monkeypatch.setattr(ds, "torch_compile_runtime_available", lambda: False)
    assert not nq.fused_w8a8_serves(dit)


def test_step_graph_check_and_its_kill_switch(runtime, monkeypatch):
    dit = _dit()
    monkeypatch.setattr(ds, "_denoiser_dits", lambda pipe: [dit])
    assert ds.step_graph_bakes_eager_w8a8(object())
    monkeypatch.setenv(ds.STEP_GRAPH_NATIVE_W8A8_ENV, "1")
    assert not ds.step_graph_bakes_eager_w8a8(object())
    monkeypatch.delenv(ds.STEP_GRAPH_NATIVE_W8A8_ENV)
    monkeypatch.setattr(ds, "_denoiser_dits", lambda pipe: [_dit(act_int8 = False)])
    assert not ds.step_graph_bakes_eager_w8a8(object())


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("native", [True, False])
def test_apply_speed_optims_declines_the_graph_only_over_an_eager_native_w8a8_denoiser(
    runtime, monkeypatch, compiled, native, dtype
):
    from core.inference import diffusion_cuda_graph as cg

    dit = _dit() if native else torch.nn.Sequential(torch.nn.Linear(8, 8))
    installed = []
    monkeypatch.setattr(ds, "_denoiser_dits", lambda pipe: [dit])
    monkeypatch.setattr(ds, "_denoiser_unet", lambda pipe: None)
    monkeypatch.setattr(ds, "compile_eligible", lambda *a, **k: compiled)
    monkeypatch.setattr(ds, "_compile_repeated_blocks", lambda *a, **k: compiled)
    monkeypatch.setattr(ds, "fp16_unet_offloaded", lambda *a, **k: False)
    monkeypatch.setattr(cg, "graph_eligible", lambda *a, **k: (True, "eligible"))
    monkeypatch.setattr(cg, "offload_graphs_enabled", lambda: False)
    monkeypatch.setattr(
        cg, "install_cuda_graphs", lambda pipe, logger = None: installed.append(pipe) or True
    )
    pipe = types.SimpleNamespace()
    target = types.SimpleNamespace(
        device = "cuda", backend = "cuda", torch_device = "cuda:0", dtype = getattr(torch, dtype)
    )
    try:
        applied = ds.apply_speed_optims(
            pipe,
            target,
            is_gguf = False,
            family = types.SimpleNamespace(supports_cuda_graph = True),
            speed_mode = ds.SPEED_DEFAULT,
        )
    except Exception as exc:  # noqa: BLE001 - CPU host: only the graph decision is under test
        pytest.skip(f"apply_speed_optims needs more of a CUDA host here: {exc!r}")
    # fp16 cards only: a bf16 load keeps its graph whatever its linears are.
    declined = native and not applied.get("compiled") and dtype == "float16"
    assert bool(installed) is not declined
    if declined:
        assert pipe._unsloth_cuda_graph_reason == ds.STEP_GRAPH_EAGER_W8A8_REASON
