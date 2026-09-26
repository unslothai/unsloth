# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Explicit ``sage`` on a GPU the installed kernel does not serve falls back to native. Probe stubbed."""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion_attention as att
from core.inference.diffusion_attention import apply_attention_backend

_UNSUPPORTED = "ValueError: Unsupported CUDA architecture: sm100"


class _Transformer:
    def __init__(self):
        self.calls: list = []

    def set_attention_backend(self, name):
        self.calls.append(name)


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
    monkeypatch.setattr(att, "_SAGE_PROBE_CACHE", {})
    monkeypatch.setattr(att, "_ensure_attention_backend_installed", lambda *a, **k: None)
    monkeypatch.setattr(att, "_active_attention_backend", lambda: "native")
    monkeypatch.setattr(att, "warn_if_sdpa_math_only", lambda *a, **k: False)
    monkeypatch.setattr(att, "_indexed_cuda_device", lambda device: device)


def _stub_probe(
    monkeypatch,
    result,
    seen = None,
):
    def _probe(device, dtype):
        if seen is not None:
            seen.append((device, dtype))
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(att, "_run_sage_probe", _probe)


def test_sage_on_a_card_its_kernel_rejects_falls_back_to_native(monkeypatch):
    _stub_probe(monkeypatch, _UNSUPPORTED)
    t, log = _Transformer(), _Logger()
    assert (
        apply_attention_backend(
            types.SimpleNamespace(transformer = t), "sage", logger = log, target = _target()
        )
        is None
    )
    assert "sage" not in t.calls
    assert any("Unsupported CUDA architecture: sm100" in w for w in log.warnings)


def test_sage_stays_engaged_where_its_kernel_runs(monkeypatch):
    _stub_probe(monkeypatch, "")
    t = _Transformer()
    assert (
        apply_attention_backend(types.SimpleNamespace(transformer = t), "sage", target = _target())
        == "sage"
    )
    assert t.calls == ["sage"]


@pytest.mark.parametrize(
    "exc", [ImportError("no sageattention"), RuntimeError("CUDA out of memory")]
)
def test_an_unanswerable_probe_keeps_the_request(monkeypatch, exc):
    seen: list = []
    _stub_probe(monkeypatch, exc, seen)
    t = _Transformer()
    assert (
        apply_attention_backend(types.SimpleNamespace(transformer = t), "sage", target = _target())
        == "sage"
    )
    assert t.calls == ["sage"]
    assert att._sage_kernel_runs(_target()) is None and len(seen) == 2


def test_probe_is_memoised_per_device_and_dtype_and_warns_each_time(monkeypatch):
    seen: list = []
    _stub_probe(monkeypatch, _UNSUPPORTED, seen)
    log = _Logger()
    assert att._sage_kernel_runs(_target(), log) is False
    assert att._sage_kernel_runs(_target(), log) is False
    assert len(seen) == 1 and len(log.warnings) == 2
    att._sage_kernel_runs(_target(dtype = "fp16"))
    att._sage_kernel_runs(_target(torch_device = "cuda:1"))
    assert seen[1:] == [("cuda", "fp16"), ("cuda:1", "bf16")]


def test_other_backends_and_targetless_calls_never_probe(monkeypatch):
    seen: list = []
    _stub_probe(monkeypatch, _UNSUPPORTED, seen)
    for backend in ("_native_cudnn", "flash", "_flash_3_hub", "flash_4_hub", "xformers", "aiter"):
        t = _Transformer()
        assert (
            apply_attention_backend(types.SimpleNamespace(transformer = t), backend, target = _target())
            == backend
        )
    t = _Transformer()
    assert apply_attention_backend(types.SimpleNamespace(transformer = t), "sage") == "sage"
    assert seen == []


@pytest.mark.parametrize("device", ["cpu", "mps", "xpu", "", None])
def test_non_cuda_targets_are_not_probed(monkeypatch, device):
    seen: list = []
    _stub_probe(monkeypatch, _UNSUPPORTED, seen)
    assert att._sage_kernel_runs(types.SimpleNamespace(device = device, dtype = None)) is None
    assert seen == []


def test_run_probe_reports_the_kernels_own_error(monkeypatch):
    torch = pytest.importorskip("torch")

    def _sageattn(
        q,
        k,
        v,
        tensor_layout = "HND",
    ):
        assert tensor_layout == "NHD" and q.dtype == torch.float16 and q.shape == (1, 128, 2, 128)
        raise ValueError("Unsupported CUDA architecture: sm75")

    monkeypatch.setitem(
        __import__("sys").modules, "sageattention", types.SimpleNamespace(sageattn = _sageattn)
    )
    assert att._run_sage_probe("cpu", None) == "ValueError: Unsupported CUDA architecture: sm75"


def test_run_probe_raises_when_the_package_is_missing(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setitem(__import__("sys").modules, "sageattention", None)
    with pytest.raises(ImportError):
        att._run_sage_probe("cpu", None)


def test_bare_cuda_is_keyed_by_the_card_pinned_on_this_thread(monkeypatch):
    seen: list = []
    current = {"card": 0}
    monkeypatch.setattr(
        att, "_indexed_cuda_device", lambda d: f"cuda:{current['card']}" if d == "cuda" else d
    )
    monkeypatch.setattr(
        att,
        "_run_sage_probe",
        lambda d, dt: seen.append(d) or ("" if d == "cuda:1" else _UNSUPPORTED),
    )
    bare = types.SimpleNamespace(device = "cuda", dtype = "bf16")
    assert att._sage_kernel_runs(bare) is False
    current["card"] = 1
    assert att._sage_kernel_runs(bare) is True
    assert seen == ["cuda:0", "cuda:1"]


def test_indexed_cuda_device_leaves_indexed_names_alone():
    assert att._indexed_cuda_device("cuda:3") == "cuda:3"
