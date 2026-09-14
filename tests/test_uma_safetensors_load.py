# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

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
from real_accelerator import (
    has_real_accelerator,
)  # tests/_shared, on sys.path via tests/conftest.py

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
    not has_real_accelerator(),
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
    not has_real_accelerator(),
    reason = "needs a GPU for the low-memory fallback path",
)
def test_low_memory_falls_back_to_direct_move(uma, force_uma, monkeypatch, tiny_safetensors):
    path, tensors = tiny_safetensors
    force_uma(True)
    fake_mu = _install_fake_modeling_utils(monkeypatch, safetensors.safe_open)
    uma.patch_unified_memory_safetensors_load()
    # Clone OOMs (transient CPU doubling on a constrained UMA box): the wrapper must fall back to
    # the direct move and still succeed.
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


# ── the budget question is per device, because the budget is per device ──────
def _fake_torch_cuda(uma, monkeypatch, flags, *, attr = "is_integrated"):
    """A torch whose device ``i`` reports ``flags[i]`` for the integrated flag."""

    def _props(index):
        props = types.SimpleNamespace()
        setattr(props, attr, flags[index])
        return props

    monkeypatch.setattr(
        uma,
        "torch",
        types.SimpleNamespace(
            cuda = types.SimpleNamespace(
                is_available = lambda: True,
                device_count = lambda: len(flags),
                get_device_properties = _props,
            )
        ),
    )
    monkeypatch.delenv("UNSLOTH_FORCE_UMA", raising = False)
    uma.device_is_integrated_unified_memory.cache_clear()
    uma.is_integrated_unified_memory_gpu.cache_clear()


def test_a_mixed_host_still_classifies_the_integrated_device(uma, monkeypatch):
    """A GB10 beside a discrete card. The process-wide gate answers False, which
    is right for the loader patch and wrong for a per-device memory budget: it
    would skip the cap on exactly the device whose pool is the host's RAM."""
    _fake_torch_cuda(uma, monkeypatch, {0: 0, 1: 1})
    assert uma.is_integrated_unified_memory_gpu() is False
    assert uma.device_is_integrated_unified_memory(0) is False
    assert uma.device_is_integrated_unified_memory(1) is True


def test_every_device_discrete_answers_false_throughout(uma, monkeypatch):
    _fake_torch_cuda(uma, monkeypatch, {0: 0, 1: 0})
    assert uma.device_is_integrated_unified_memory(0) is False
    assert uma.device_is_integrated_unified_memory(1) is False


def test_the_older_attribute_spelling_is_read_too(uma, monkeypatch):
    """torch renamed the flag; a wheel exposing neither reads discrete."""
    _fake_torch_cuda(uma, monkeypatch, {0: 1}, attr = "integrated")
    assert uma.device_is_integrated_unified_memory(0) is True
    _fake_torch_cuda(uma, monkeypatch, {0: 1}, attr = "neither_spelling")
    assert uma.device_is_integrated_unified_memory(0) is False


def test_an_out_of_range_device_is_not_classified(uma, monkeypatch):
    _fake_torch_cuda(uma, monkeypatch, {0: 1})
    assert uma.device_is_integrated_unified_memory(5) is False
    assert uma.device_is_integrated_unified_memory(-1) is False


def test_a_device_that_cannot_be_read_budgets_as_before(uma, monkeypatch):
    def _raise(index):
        raise RuntimeError("no driver")

    monkeypatch.setattr(
        uma,
        "torch",
        types.SimpleNamespace(
            cuda = types.SimpleNamespace(
                is_available = lambda: True,
                device_count = lambda: 1,
                get_device_properties = _raise,
            )
        ),
    )
    monkeypatch.delenv("UNSLOTH_FORCE_UMA", raising = False)
    uma.device_is_integrated_unified_memory.cache_clear()
    assert uma.device_is_integrated_unified_memory(0) is False


def test_the_force_override_still_applies_per_device(uma, monkeypatch):
    monkeypatch.setenv("UNSLOTH_FORCE_UMA", "1")
    uma.device_is_integrated_unified_memory.cache_clear()
    assert uma.device_is_integrated_unified_memory(0) is True
    monkeypatch.setenv("UNSLOTH_FORCE_UMA", "0")
    uma.device_is_integrated_unified_memory.cache_clear()
    assert uma.device_is_integrated_unified_memory(0) is False


# ── an absolute ceiling and an incremental headroom are not the same number ──
GIB = 1024**3


def _budget(**kwargs):
    from unsloth.save import _unified_memory_vram_budget

    return _unified_memory_vram_budget(**kwargs)


def test_the_host_headroom_is_anchored_to_what_is_already_resident():
    """8 GiB resident and 9 GiB of spendable host headroom.

    The caller tests ``memory_allocated + W.nbytes < budget``, so the budget must
    leave room for 9 GiB MORE, i.e. land at 8 + 9 = 17 GiB. Returning the bare
    9 GiB charges the resident bytes a second time and allows ~1 GiB more.
    """
    got = _budget(
        vram_budget_bytes = 40 * GIB,
        allocated_bytes = 8 * GIB,
        host_headroom_bytes = 9 * GIB,
    )
    assert got == 17 * GIB
    assert got > 8 * GIB, "the ceiling must not sit below what is already resident"


def test_a_tight_host_still_binds_the_budget_down():
    """Capping is the whole point: a 45 GiB pool with under 2 GiB spendable."""
    got = _budget(
        vram_budget_bytes = 45 * GIB,
        allocated_bytes = 0,
        host_headroom_bytes = 2 * GIB,
    )
    assert got == 2 * GIB


def test_it_never_raises_the_discrete_budget():
    """Plenty of host RAM does not license spending more than the pool allows."""
    got = _budget(
        vram_budget_bytes = 20 * GIB,
        allocated_bytes = 0,
        host_headroom_bytes = 500 * GIB,
    )
    assert got == 20 * GIB


def test_a_negative_host_reading_is_not_spendable():
    got = _budget(
        vram_budget_bytes = 20 * GIB,
        allocated_bytes = 4 * GIB,
        host_headroom_bytes = -1,
    )
    assert got == 4 * GIB


def test_the_shard_workspace_is_not_spent_twice_on_one_pool():
    """`max_ram` has already had the serialization workspace taken out and the
    fraction applied. On unified memory the retained tensors and that workspace
    are the same bytes, so the GPU ceiling has to be the SAME budget, not a fresh
    `available * fraction` that silently re-credits the shard reserve.

    39 GiB available, 5 GiB shard workspace, 0.9 usable: max_ram is 30.6 GiB, so
    that is the whole headroom. Deriving from raw availability would offer
    35.1 GiB and leave under a shard's room for the final save_pretrained.
    """
    available = 39 * GIB
    shard = 5 * GIB
    max_ram = int(max(0, available - shard) * 0.9)
    got = _budget(
        vram_budget_bytes = 46 * GIB, allocated_bytes = 0, host_headroom_bytes = max_ram,
    )
    assert got == max_ram
    assert got < int(available * 0.9), "the shard workspace must not be re-credited"


def test_the_cgroup_reader_the_save_path_imports_still_exists():
    """The save path reaches `_cgroup_free_bytes` through a lazy import wrapped in
    `except Exception`, which is right at runtime (it must not break a merge on a
    host without cgroups) and dangerous at review time: a rename would be swallowed
    and the container cap would silently stop being applied. Pin the name.
    """
    from unsloth.dataset_num_proc import _cgroup_free_bytes

    answer = _cgroup_free_bytes()
    assert answer is None or (isinstance(answer, int) and answer >= 0)


def test_the_save_path_bounds_max_ram_by_the_cgroup():
    """psutil reports the HOST inside a container, so the merge budget has to be
    the tighter of the two readings."""
    import inspect

    from unsloth import save as save_mod

    source = inspect.getsource(save_mod.unsloth_save_model)
    assert "_cgroup_free_bytes" in source
    assert "max_ram = min(max_ram, _cgroup_ram)" in source
