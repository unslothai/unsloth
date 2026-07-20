# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the UMA safetensors clone-then-move fast load.

The module loads in isolation with a fake ``transformers.modeling_utils``. The
CUDA correctness check needs a GPU; gating, passthrough, idempotency and opt-out
are GPU-free. The gate is lazy (wrapper-time), so the wrapper installs
everywhere and passes through when it's off.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")
import safetensors  # noqa: E402

_MODULE_PATH = Path(__file__).resolve().parent.parent / "unsloth" / "models" / "_uma_safetensors.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("uma_safetensors_under_test", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def uma():
    return _load_module()


@pytest.fixture()
def force_uma(uma, monkeypatch):
    """Force the UMA gate on (or off) and keep the lru_cache from sticking."""

    def _set(on):
        monkeypatch.setenv("UNSLOTH_FORCE_UMA", "1" if on else "0")
        uma.is_integrated_unified_memory_gpu.cache_clear()

    yield _set
    uma.is_integrated_unified_memory_gpu.cache_clear()


@pytest.fixture()
def tiny_safetensors(tmp_path):
    tensors = {
        "w": torch.arange(32, dtype = torch.float32).reshape(4, 8),
        "b": torch.tensor([1.0, 2.0, 3.0, 4.0], dtype = torch.float32),
    }
    path = tmp_path / "model.safetensors"
    safetensors_torch.save_file(tensors, str(path))
    return path, tensors


def _install_fake_modeling_utils(monkeypatch, safe_open_fn):
    fake_transformers = types.ModuleType("transformers")
    fake_mu = types.ModuleType("transformers.modeling_utils")
    fake_mu.safe_open = safe_open_fn
    fake_transformers.modeling_utils = fake_mu
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", fake_mu)
    return fake_mu


# --- detection / gate ---


def test_force_uma_on(uma, monkeypatch):
    monkeypatch.setenv("UNSLOTH_FORCE_UMA", "1")
    uma.is_integrated_unified_memory_gpu.cache_clear()
    assert uma.is_integrated_unified_memory_gpu() is True


def test_force_uma_off(uma, monkeypatch):
    monkeypatch.setenv("UNSLOTH_FORCE_UMA", "0")
    uma.is_integrated_unified_memory_gpu.cache_clear()
    assert uma.is_integrated_unified_memory_gpu() is False


@pytest.mark.parametrize(
    "device,expected",
    [
        (0, True),
        ("cuda", True),
        ("cuda:0", True),
        ("cpu", False),
        ("disk", False),
        (None, False),
        (True, False),  # a bool is not a device index
    ],
)
def test_is_cuda_target(uma, device, expected):
    assert uma._is_cuda_target(device) is expected


def test_is_cuda_target_torch_device(uma):
    assert uma._is_cuda_target(torch.device("cuda", 0)) is True
    assert uma._is_cuda_target(torch.device("cpu")) is False


# --- patch gating ---


def test_wrapper_passes_through_off_uma(uma, force_uma, monkeypatch):
    """Gate OFF: every call -- including CUDA targets -- passes straight through
    to the real safe_open (the gate is evaluated lazily inside the wrapper)."""
    force_uma(False)
    sentinel = object()
    calls = []

    def fake_safe_open(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    fake_mu = _install_fake_modeling_utils(monkeypatch, fake_safe_open)
    assert uma.patch_unified_memory_safetensors_load() is True
    assert getattr(fake_mu.safe_open, "_unsloth_uma_clone", False) is True
    out = fake_mu.safe_open("shard.safetensors", "pt", "cuda:0")
    assert out is sentinel
    assert calls == [(("shard.safetensors", "pt", "cuda:0"), {})]


def test_patch_install_does_not_evaluate_gate(uma, monkeypatch):
    """Installing the wrapper must NOT query the integrated-GPU property -- that
    would init CUDA at ``import unsloth`` (fork-unsafe, and before the Spark
    allocator config is set)."""

    def _boom():
        raise AssertionError("gate must not be evaluated at install time")

    _install_fake_modeling_utils(monkeypatch, safetensors.safe_open)
    monkeypatch.setattr(uma, "is_integrated_unified_memory_gpu", _boom)
    assert uma.patch_unified_memory_safetensors_load() is True


def test_patch_noop_when_opted_out(uma, force_uma, monkeypatch):
    force_uma(True)
    monkeypatch.setenv("UNSLOTH_DISABLE_UMA_CLONE_LOAD", "1")
    real = object()
    fake_mu = _install_fake_modeling_utils(monkeypatch, real)
    assert uma.patch_unified_memory_safetensors_load() is False
    assert fake_mu.safe_open is real


def test_patch_installs_and_is_idempotent(uma, force_uma, monkeypatch):
    force_uma(True)
    fake_mu = _install_fake_modeling_utils(monkeypatch, safetensors.safe_open)
    assert uma.patch_unified_memory_safetensors_load() is True
    wrapped = fake_mu.safe_open
    assert getattr(wrapped, "_unsloth_uma_clone", False) is True
    # second call must not double-wrap
    assert uma.patch_unified_memory_safetensors_load() is True
    assert fake_mu.safe_open is wrapped


# --- correctness ---


def test_cpu_target_is_passthrough(uma, force_uma, monkeypatch, tiny_safetensors):
    path, tensors = tiny_safetensors
    force_uma(True)
    fake_mu = _install_fake_modeling_utils(monkeypatch, safetensors.safe_open)
    uma.patch_unified_memory_safetensors_load()
    # device="cpu" must NOT be intercepted -> identical data, still on CPU.
    with fake_mu.safe_open(str(path), framework = "pt", device = "cpu") as f:
        for key, expected in tensors.items():
            got = f.get_slice(key)[:]
            assert got.device.type == "cpu"
            assert torch.equal(got, expected)


@pytest.mark.skipif(
    not (hasattr(torch, "cuda") and torch.cuda.is_available()),
    reason = "needs a GPU for the host->device clone-and-move path",
)
def test_cuda_target_clones_and_moves(uma, force_uma, monkeypatch, tiny_safetensors):
    path, tensors = tiny_safetensors
    force_uma(True)
    fake_mu = _install_fake_modeling_utils(monkeypatch, safetensors.safe_open)
    uma.patch_unified_memory_safetensors_load()
    # device="cuda" IS intercepted -> tensors land on cuda, byte-identical.
    with fake_mu.safe_open(str(path), framework = "pt", device = "cuda") as f:
        for key, expected in tensors.items():
            got = f.get_slice(key)[:]
            assert got.device.type == "cuda"
            assert torch.equal(got.cpu(), expected)
            got_full = f.get_tensor(key)
            assert got_full.device.type == "cuda"
            assert torch.equal(got_full.cpu(), expected)


@pytest.mark.skipif(
    not (hasattr(torch, "cuda") and torch.cuda.is_available()),
    reason = "needs a GPU for the low-memory fallback path",
)
def test_low_memory_falls_back_to_direct_move(uma, force_uma, monkeypatch, tiny_safetensors):
    path, tensors = tiny_safetensors
    force_uma(True)
    fake_mu = _install_fake_modeling_utils(monkeypatch, safetensors.safe_open)
    uma.patch_unified_memory_safetensors_load()
    # Clone OOMs (transient CPU doubling on a constrained UMA box): the wrapper
    # must fall back to the direct move and still succeed.
    real_clone = torch.Tensor.clone

    def _oom_clone(self, *a, **k):
        raise RuntimeError("[enforce fail] not enough memory")

    monkeypatch.setattr(torch.Tensor, "clone", _oom_clone)
    try:
        with fake_mu.safe_open(str(path), framework = "pt", device = "cuda") as f:
            for key, expected in tensors.items():
                got = f.get_slice(key)[:]
                assert got.device.type == "cuda"
                got_full = f.get_tensor(key)
                assert got_full.device.type == "cuda"
    finally:
        monkeypatch.setattr(torch.Tensor, "clone", real_clone)
    for key, expected in tensors.items():
        with fake_mu.safe_open(str(path), framework = "pt", device = "cuda") as f:
            assert torch.equal(f.get_tensor(key).cpu(), expected)
