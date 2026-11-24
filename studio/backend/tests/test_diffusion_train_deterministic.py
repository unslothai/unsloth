# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU tests: UNSLOTH_DIFFUSION_TRAIN_DETERMINISTIC=1 turns on strict deterministic algorithms (plus the cuBLAS
workspace it needs) for the training run only; off by default; the previous state comes back on exit."""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from core.training import diffusion_train_common as common  # noqa: E402
from core.training.diffusion_train_common import DiffusionLoraConfig  # noqa: E402

_ENV = "UNSLOTH_DIFFUSION_TRAIN_DETERMINISTIC"
_CUBLAS = "CUBLAS_WORKSPACE_CONFIG"


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    monkeypatch.delenv(_ENV, raising = False)
    monkeypatch.delenv(_CUBLAS, raising = False)
    monkeypatch.setattr(common, "_deterministic_logged", False)
    torch.use_deterministic_algorithms(False)
    yield
    torch.use_deterministic_algorithms(before[0], warn_only = before[1])


def _cfg():
    return DiffusionLoraConfig(base_model = "b", data_dir = "d", output_dir = "o")


def test_off_by_default():
    snap = common._apply_perf_flags(_cfg(), "cuda")
    assert torch.are_deterministic_algorithms_enabled() is False
    assert _CUBLAS not in os.environ
    common._restore_perf_flags(snap)
    assert torch.are_deterministic_algorithms_enabled() is False


def test_on_sets_strict_mode_and_cublas_workspace_then_restores(monkeypatch, caplog):
    monkeypatch.setenv(_ENV, "1")
    with caplog.at_level("WARNING"):
        snap = common._apply_perf_flags(_cfg(), "cuda")
    assert torch.are_deterministic_algorithms_enabled() is True
    assert torch.is_deterministic_algorithms_warn_only_enabled() is False
    assert os.environ[_CUBLAS] == ":4096:8"
    assert sum(_ENV in r.getMessage() for r in caplog.records) == 1
    assert "slower" in caplog.text
    common._restore_perf_flags(snap)
    assert torch.are_deterministic_algorithms_enabled() is False
    assert _CUBLAS not in os.environ


def test_logged_once_per_process(monkeypatch, caplog):
    monkeypatch.setenv(_ENV, "1")
    with caplog.at_level("WARNING"):
        for _ in range(2):
            common._restore_perf_flags(common._apply_perf_flags(_cfg(), "cuda"))
    assert sum(_ENV in r.getMessage() for r in caplog.records) == 1


def test_keeps_a_user_cublas_setting_and_a_prior_warn_only_mode(monkeypatch):
    monkeypatch.setenv(_ENV, "1")
    monkeypatch.setenv(_CUBLAS, ":16:8")
    torch.use_deterministic_algorithms(True, warn_only = True)
    snap = common._apply_perf_flags(_cfg(), "cuda")
    assert os.environ[_CUBLAS] == ":16:8"
    assert torch.is_deterministic_algorithms_warn_only_enabled() is False
    common._restore_perf_flags(snap)
    assert os.environ[_CUBLAS] == ":16:8"
    assert torch.are_deterministic_algorithms_enabled() is True
    assert torch.is_deterministic_algorithms_warn_only_enabled() is True


@pytest.mark.parametrize("value", ["0", "", "off"])
def test_falsy_values_stay_off(monkeypatch, value):
    monkeypatch.setenv(_ENV, value)
    common._restore_perf_flags(common._apply_perf_flags(_cfg(), "cuda"))
    assert torch.are_deterministic_algorithms_enabled() is False
