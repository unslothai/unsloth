# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

from __future__ import annotations

import traceback
import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_vae_upcast as vu  # noqa: E402


class _Budget:
    def __init__(
        self,
        cap: int,
        error_type = torch.OutOfMemoryError,
    ):
        self.cap, self.used = cap, 0
        self.ooms = 0
        self.error_type = error_type

    def alloc(self, n: int) -> None:
        if self.used + n > self.cap:
            self.ooms += 1
            raise self.error_type("out of memory")
        self.used += n


class _Tensor:
    def __init__(
        self,
        budget: _Budget,
        n: int,
        dtype,
        device = "cuda:0",
        value = 1,
    ):
        self.budget, self.n, self.dtype = budget, n, dtype
        self.device = torch.device(device)
        self.size = (
            n * (2 if dtype in (torch.float16, torch.bfloat16) else 4)
            if self.device.type != "cpu"
            else 0
        )
        self.shape = (n,)
        self._version, self.value = 0, value
        self.live = False
        if self.size:
            budget.alloc(self.size)
        self.live = True

    def __del__(self):
        if self.live:
            self.budget.used -= self.size

    def is_pinned(self):
        return False

    def is_floating_point(self):
        return True

    def is_inference(self):
        return False

    def data_ptr(self):
        return id(self)

    def to(
        self,
        *args,
        dtype = None,
        device = None,
        **kwargs,
    ):
        dtype = dtype or next((a for a in args if isinstance(a, torch.dtype)), self.dtype)
        device = torch.device(
            device or next((a for a in args if not isinstance(a, torch.dtype)), self.device)
        )
        return (
            self
            if dtype == self.dtype and device == self.device
            else _Tensor(self.budget, self.n, dtype, device, self.value)
        )


class _Param:
    def __init__(self, t):
        self.data = t
        self._version = 0

    @property
    def dtype(self):
        return self.data.dtype

    def is_floating_point(self):
        return True

    def is_inference(self):
        return False

    def data_ptr(self):
        return self.data.data_ptr()

    @property
    def device(self):
        return self.data.device


class _Model:
    def __init__(
        self,
        budget: _Budget,
        dtype = torch.float16,
        kind = "p",
    ):
        self._buffers = {}
        self._parameters = {
            name: _Param(_Tensor(budget, n, dtype))
            for name, n in [("large0", 4), ("bias", 1), ("large1", 4)]
        }
        if kind == "b":
            self._buffers = {name: p.data for name, p in self._parameters.items()}
            self._parameters = {}

    def modules(self):
        return [self]

    def named_modules(self):
        return [("", self)]

    def parameters(self):
        return iter(self._parameters.values())


def _kept_model(
    monkeypatch,
    budget,
    dtype = torch.float16,
    kind = "p",
):
    model = _Model(budget, dtype, kind)
    table = model._parameters if kind == "p" else model._buffers
    hosts = {
        name: _Tensor(budget, t.data.n if kind == "p" else t.n, dtype, "cpu")
        for name, t in table.items()
    }
    state = {
        "host" if kind == "p" else "buffer_host": hosts,
        "version" if kind == "p" else "buffer_version": {
            name: (t, 0, t.data_ptr()) for name, t in table.items()
        },
    }
    model._hf_hook = types.SimpleNamespace(_unsloth_offload_keep_cpu = True)
    model._unsloth_offload_keep_cpu = state
    monkeypatch.setattr(vu, "_host_backed", lambda state, kind, full, current: hosts[full])
    return model, state, table


def test_rollback_retries_a_cast_back_that_ran_out_of_room():
    # Large0's cast-back needs 8 free units; only the bias cast-back makes that room.
    budget = _Budget(34)
    model = _Model(budget)
    with pytest.raises(torch.OutOfMemoryError):
        vu.upcast_keeping_storage(model)
    dtypes = {name: p.dtype for name, p in model._parameters.items()}
    assert dtypes == {name: torch.float16 for name in dtypes}
    assert budget.used == 18
    assert budget.ooms == 1


@pytest.mark.parametrize("cap", range(18, 36))
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["p", "b"])
def test_kept_host_rollback_restores_tight_budgets(monkeypatch, cap, dtype, kind):
    budget = _Budget(cap)
    model, state, table = _kept_model(monkeypatch, budget, dtype, kind)
    with pytest.raises(torch.OutOfMemoryError):
        vu.upcast_keeping_storage(model)
    assert all(t.device.type == "cuda" and t.dtype == dtype for t in table.values())
    assert budget.used == 18
    versions = state["version" if kind == "p" else "buffer_version"]
    assert all(versions[name][2] == t.data_ptr() for name, t in table.items())


@pytest.mark.parametrize("chained", [False, True])
@pytest.mark.parametrize("error_type", [torch.OutOfMemoryError, RuntimeError])
def test_forward_oom_frees_dead_frame_activations_before_restoring(chained, error_type):
    budget = _Budget(54)
    model = _Model(budget)
    error = error_type("MPS backend out of memory")

    def inner():
        _activation = _Tensor(budget, 4, torch.float32)
        raise error_type("out of memory") if chained else error

    def forward():
        try:
            inner()
        except RuntimeError as cause:
            if chained:
                raise error from cause
            raise

    with pytest.raises(error_type) as caught:
        with vu.fp32_keeping_storage(model):
            forward()
    assert caught.value is error
    assert "forward" in [f.name for f in traceback.extract_tb(error.__traceback__)]
    assert all(p.dtype == torch.float16 for p in model._parameters.values())
    assert budget.used == 18


def test_kept_host_rollback_handles_backend_runtime_oom(monkeypatch):
    budget = _Budget(20, RuntimeError)
    model, state, table = _kept_model(monkeypatch, budget)
    with pytest.raises(RuntimeError, match = "out of memory"):
        vu.upcast_keeping_storage(model)
    assert all(t.device.type == "cuda" and t.dtype == torch.float16 for t in table.values())
    assert budget.used == 18


def test_backend_cache_failure_does_not_replace_the_original_oom(monkeypatch):
    def fail():
        raise RuntimeError("cache unavailable")

    monkeypatch.setattr(torch.cuda, "empty_cache", fail)
    budget = _Budget(34)
    model = _Model(budget)
    with pytest.raises(torch.OutOfMemoryError, match = "out of memory"):
        vu.upcast_keeping_storage(model)
    assert all(p.dtype == torch.float16 for p in model._parameters.values())


def test_host_fallback_preserves_inplace_parameter_changes(monkeypatch):
    budget = _Budget(100)
    model, state, table = _kept_model(monkeypatch, budget)
    token = vu.upcast_keeping_storage(model)
    table["large0"].data.value = 7
    table["large0"]._version += 1
    budget.cap = 20
    vu.restore_storage(token)
    assert table["large0"].data.value == 7
    assert not token


def test_no_progress_stops_and_preserves_non_oom_tracebacks(monkeypatch):
    attempts = []

    def fail(rec):
        attempts.append(rec)
        raise ValueError("persistent")

    monkeypatch.setattr(vu, "_restore_one", fail)
    token = [[None] * 7 for _ in range(3)]
    try:
        raise ValueError("caller")
    except ValueError as error:
        tb = error.__traceback__
        vu.restore_storage(token)
        assert error.__traceback__ is tb
    assert len(attempts) == 3
    assert not token
