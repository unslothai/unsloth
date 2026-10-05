# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""cuDNN attention is only pinned when the torch/cuDNN build has a kernel for every DiT head_dim.

Ideogram 4 (head_dim 256) under any speed profile raised "No available kernel" on torch 2.11-2.13, whose cuDNN
serves head_dim <= 128, while torch 2.14 runs it. Probe stubbed except the last test."""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion_attention as att
from core.inference.diffusion_attention import apply_attention_backend


class _Transformer:
    def __init__(
        self,
        config = None,
        modules = (),
    ):
        self.calls: list = []
        self.config = config
        self._modules_list = list(modules)

    def set_attention_backend(self, name):
        self.calls.append(name)

    def modules(self):
        return iter(self._modules_list)


class _Logger:
    def __init__(self):
        self.warnings: list = []

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)

    def info(self, *_a, **_k):
        pass


def _target(
    device = "cuda",
    dtype = "bf16",
    torch_device = None,
):
    return types.SimpleNamespace(device = device, dtype = dtype, torch_device = torch_device or device)


@pytest.fixture(autouse = True)
def _isolated(monkeypatch):
    monkeypatch.setattr(att, "_CUDNN_HEAD_DIM_CACHE", {}, raising = False)
    monkeypatch.setattr(att, "_ensure_attention_backend_installed", lambda *a, **k: None)
    monkeypatch.setattr(att, "_active_attention_backend", lambda: "native")
    monkeypatch.setattr(att, "warn_if_sdpa_math_only", lambda *a, **k: False)
    monkeypatch.setattr(att, "_indexed_cuda_device", lambda device: device)


def _stub_probe(
    monkeypatch,
    served = (64, 128),
    seen = None,
    raises = None,
):
    def _probe(device, dtype, head_dim):
        if seen is not None:
            seen.append((device, dtype, head_dim))
        if raises is not None:
            raise raises
        return head_dim in served

    monkeypatch.setattr(att, "_run_cudnn_head_dim_probe", _probe, raising = False)


def test_ideogram_head_dim_256_keeps_native_where_cudnn_has_no_kernel(monkeypatch):
    _stub_probe(monkeypatch, served = (64, 128))
    cond = _Transformer(config = {"attention_head_dim": 256})
    uncond = _Transformer(config = {"attention_head_dim": 256})
    pipe = types.SimpleNamespace(transformer = cond, unconditional_transformer = uncond)
    log = _Logger()
    assert apply_attention_backend(pipe, "_native_cudnn", logger = log, target = _target()) is None
    assert "_native_cudnn" not in cond.calls and "_native_cudnn" not in uncond.calls
    assert any("head_dim 256" in w for w in log.warnings)


def test_cudnn_stays_pinned_where_it_serves_the_head_dim(monkeypatch):
    _stub_probe(monkeypatch, served = (128, 256))
    t = _Transformer(config = {"attention_head_dim": 256})
    assert apply_attention_backend(
        types.SimpleNamespace(transformer = t), "_native_cudnn", target = _target()
    ) == ("_native_cudnn")
    assert t.calls == ["_native_cudnn"]


def test_head_dim_read_from_modules_when_config_lacks_it(monkeypatch):
    _stub_probe(monkeypatch, served = (64, 128))
    t = _Transformer(config = {"dim": 3840}, modules = [types.SimpleNamespace(head_dim = 256)])
    assert (
        apply_attention_backend(
            types.SimpleNamespace(transformer = t), "_native_cudnn", target = _target()
        )
        is None
    )


@pytest.mark.parametrize("exc", [ImportError("no torch"), RuntimeError("CUDA out of memory")])
def test_an_unanswerable_probe_keeps_cudnn(monkeypatch, exc):
    seen: list = []
    _stub_probe(monkeypatch, seen = seen, raises = exc)
    t = _Transformer(config = {"attention_head_dim": 256})
    assert apply_attention_backend(
        types.SimpleNamespace(transformer = t), "_native_cudnn", target = _target()
    ) == ("_native_cudnn")
    assert att._cudnn_runs_head_dim(_target(), 256) is None and len(seen) == 2


def test_probe_is_memoised_per_device_dtype_and_head_dim(monkeypatch):
    seen: list = []
    _stub_probe(monkeypatch, seen = seen)
    assert att._cudnn_runs_head_dim(_target(), 256) is False
    assert att._cudnn_runs_head_dim(_target(), 256) is False
    assert att._cudnn_runs_head_dim(_target(), 128) is True
    att._cudnn_runs_head_dim(_target(torch_device = "cuda:1"), 256)
    assert seen == [("cuda", "bf16", 256), ("cuda", "bf16", 128), ("cuda:1", "bf16", 256)]


def test_other_backends_unknown_head_dims_and_cpu_targets_never_probe(monkeypatch):
    seen: list = []
    _stub_probe(monkeypatch, seen = seen)
    t = _Transformer(config = {"attention_head_dim": 256})
    for backend in ("flash", "xformers", "aiter"):
        assert (
            apply_attention_backend(types.SimpleNamespace(transformer = t), backend, target = _target())
            == backend
        )
    bare = _Transformer()
    assert apply_attention_backend(
        types.SimpleNamespace(transformer = bare), "_native_cudnn", target = _target()
    ) == ("_native_cudnn")
    assert (
        apply_attention_backend(types.SimpleNamespace(transformer = t), "_native_cudnn")
        == "_native_cudnn"
    )
    assert att._cudnn_runs_head_dim(_target(device = "cpu"), 256) is None
    assert seen == []


def test_real_probe_agrees_with_a_real_pinned_attention():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip(
            reason = "the probe asks a real CUDA device; the stubbed tests above cover CPU-only CI"
        )
    from torch.nn.attention import SDPBackend, sdpa_kernel

    for head_dim in (128, 256, 512):
        q = torch.zeros((1, 4, 64, head_dim), device = "cuda", dtype = torch.bfloat16)
        try:
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                torch.nn.functional.scaled_dot_product_attention(q, q, q)
            runs = True
        except RuntimeError:
            runs = False
        assert att._run_cudnn_head_dim_probe("cuda", torch.bfloat16, head_dim) is runs


def test_kill_switch_pins_cudnn_without_asking(monkeypatch):
    seen: list = []
    _stub_probe(monkeypatch, served = (), seen = seen)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_CUDNN_HEAD_DIM_PROBE", "0")
    t = _Transformer(config = {"attention_head_dim": 256})
    assert apply_attention_backend(
        types.SimpleNamespace(transformer = t), "_native_cudnn", target = _target()
    ) == ("_native_cudnn")
    assert seen == []
