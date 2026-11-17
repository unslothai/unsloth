# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows ROCm: SDPA calls with a head wider than 256 (Gemma 4's global layers) must run on the math backend, since
the fused kernels that take them there compute them wrong. Heads up to 256 and every other platform are untouched."""

import shutil
import sys
import types

import pytest

torch = pytest.importorskip("torch")
attention = pytest.importorskip("torch.nn.attention")

from core import _rocm_windows_sdpa as G  # noqa: E402


def _q(head_dim, device = "cuda"):
    return types.SimpleNamespace(
        shape = (1, 8, 128, head_dim), device = types.SimpleNamespace(type = device)
    )


@pytest.fixture
def recorder(monkeypatch):
    entered = []

    def fake_set(backends):
        entered.append(list(backends))

    monkeypatch.setattr(attention, "_sdpa_kernel", fake_set)
    monkeypatch.setattr(attention, "_cur_sdpa_kernel_backends", lambda: ["saved"])
    calls = []

    def sdpa(q, k, v, *args, **kwargs):
        calls.append((q, kwargs))
        return "out"

    return entered, calls, sdpa


@pytest.mark.parametrize("head_dim", [384, 512])
def test_wide_cuda_head_runs_on_math(recorder, head_dim):
    entered, calls, sdpa = recorder
    q = _q(head_dim)
    assert G.guard_sdpa(sdpa)(q, q, q, is_causal = True, enable_gqa = True) == "out"
    assert entered == [[attention.SDPBackend.MATH], ["saved"]]
    assert calls[0][1] == {"is_causal": True, "enable_gqa": True}


@pytest.mark.parametrize("head_dim, device", [(128, "cuda"), (256, "cuda"), (512, "cpu")])
def test_other_calls_keep_the_stock_dispatch(recorder, head_dim, device):
    entered, calls, sdpa = recorder
    q = _q(head_dim, device)
    assert G.guard_sdpa(sdpa)(q, q, q) == "out"
    assert entered == []


def test_install_wraps_sdpa_on_windows_rocm_only(monkeypatch):
    F = torch.nn.functional
    stock = F.scaled_dot_product_attention
    monkeypatch.setattr(F, "scaled_dot_product_attention", stock)
    monkeypatch.setattr(sys, "platform", "linux")
    assert G.install_windows_rocm_wide_head_sdpa_guard() is False
    assert F.scaled_dot_product_attention is stock

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(torch.version, "hip", "7.14.0", raising = False)
    assert G.install_windows_rocm_wide_head_sdpa_guard() is True
    wrapped = F.scaled_dot_product_attention
    assert getattr(wrapped, "_unsloth_windows_rocm_wide_head_guard", False)
    assert G.install_windows_rocm_wide_head_sdpa_guard() is True
    assert F.scaled_dot_product_attention is wrapped  # idempotent: no double wrap

    # A real CPU call still computes the stock answer through the wrapper.
    q = torch.randn(1, 2, 8, 16)
    assert torch.equal(wrapped(q, q, q, is_causal = True), stock(q, q, q, is_causal = True))


@pytest.mark.parametrize("worker", ["core/inference/worker.py", "core/training/worker.py"])
def test_llm_workers_install_the_guard(worker):
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / worker).read_text(encoding = "utf-8")
    assert "install_windows_rocm_wide_head_sdpa_guard(" in src
    # Before transformers or unsloth load the model: right after the torchao stub.
    assert src.index("install_windows_rocm_wide_head_sdpa_guard(") > src.index(
        "install_torchao_windows_rocm_stub()"
    )


def test_overlapping_wide_head_calls_restore_the_callers_backends_once():
    """The first of two overlapping wide-head calls returning first must not put the fused kernels back under the
    second; the caller's backends come back once, after the last returns."""
    import threading

    seen = {}
    inside = {n: threading.Event() for n in "ab"}
    release = {n: threading.Event() for n in "ab"}

    def sdpa(
        q,
        k,
        v,
        *args,
        name = None,
        **kwargs,
    ):
        inside[name].set()
        release[name].wait(5)
        seen[name] = set(attention._cur_sdpa_kernel_backends())
        return "out"

    guarded = G.guard_sdpa(sdpa)
    before = set(attention._cur_sdpa_kernel_backends())
    q = _q(512)
    threads = {
        n: threading.Thread(target = guarded, args = (q, q, q), kwargs = {"name": n}) for n in "ab"
    }
    threads["a"].start()
    assert inside["a"].wait(5)
    threads["b"].start()
    assert inside["b"].wait(5)
    release["a"].set()
    threads["a"].join(5)
    release["b"].set()
    threads["b"].join(5)
    assert seen == {"a": {attention.SDPBackend.MATH}, "b": {attention.SDPBackend.MATH}}
    assert set(attention._cur_sdpa_kernel_backends()) == before


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_guarded_wide_head_compiles_fullgraph(monkeypatch, backend):
    # Studio installs this guard before unsloth's, which then keeps it: its wide-head branch must trace with no break
    if backend == "inductor" and sys.platform == "win32" and shutil.which("cl") is None:
        # Inductor's CPU backend builds C++ with MSVC, which a plain Windows runner does not have on PATH.
        pytest.skip(
            "inductor CPU needs MSVC (cl) on Windows; the aot_eager case still traces the guard"
        )
    F = torch.nn.functional
    monkeypatch.setattr(G, "_wide_cuda_head", lambda q: q.shape[-1] > G.WIDEST_FUSED_HEAD_DIM)
    guarded = G.guard_sdpa(F.scaled_dot_product_attention)
    q, k, v = (torch.randn(1, 2, 8, 512) for _ in range(3))
    before = attention._cur_sdpa_kernel_backends()
    want = guarded(q, k, v, is_causal = True)
    torch._dynamo.reset()
    try:
        got = torch.compile(
            lambda q, k, v: guarded(q, k, v, is_causal = True), backend = backend, fullgraph = True
        )(q, k, v)
    finally:
        torch._dynamo.reset()
    assert torch.allclose(got, want, atol = 1e-6)
    assert attention._cur_sdpa_kernel_backends() == before and G._HELD == [0, None]
