# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The NPU memory window and which catalog models fit it; RAM and OS are pinned, never the host's."""

from __future__ import annotations

import pytest

from core.inference import npu_backend as nb

# Approximate Lemonade catalog sizes of the flagship FastFlowLM models, plus a small one.
_SIZES_GB = {
    "qwen3.8-mtp:27b": 18.0,
    "qwen3.8-dflash2:27b": 18.0,
    "qwen3.6-moe:35b-a3b": 22.0,
    "qwen3:0.6b": 0.66,
}
_FLAGSHIPS = ("qwen3.8-mtp:27b", "qwen3.8-dflash2:27b", "qwen3.6-moe:35b-a3b")


class _Response:
    status_code = 200

    def json(self):
        return {
            "data": [
                {"id": name, "recipe": "flm", "checkpoint": name, "size": size, "labels": ["chat"]}
                for name, size in _SIZES_GB.items()
            ]
            + [{"id": "no-size", "recipe": "flm", "checkpoint": "no-size", "labels": ["chat"]}]
        }


class _Server:
    def request(self, method, path, **kwargs):
        return _Response()

    def is_alive(self):
        return True


def _catalog(tmp_path, monkeypatch, total_ram_gb, platform):
    monkeypatch.setattr(nb, "_total_ram_gb", lambda: total_ram_gb)
    monkeypatch.setattr(nb.sys, "platform", platform)
    backend = nb.LemonadeNpuBackend(root = tmp_path / "lemonade")
    monkeypatch.setattr(backend, "_ensure_running", lambda: _Server())
    monkeypatch.setattr(backend, "_flm_binary", lambda: None)
    monkeypatch.setattr(backend, "_model_files", lambda checkpoint, flm_binary: None)
    return {model.id: model for model in backend.catalog()}


def test_windows_window_is_half_of_ram():
    assert nb.npu_memory_window_gb(32.0, "win32") == 16.0
    assert nb.npu_memory_window_gb(64.0, "win32") == 32.0


def test_other_systems_keep_all_but_a_reserve():
    assert nb.npu_memory_window_gb(32.0, "linux") == 30.0
    assert nb.npu_memory_window_gb(1.0, "linux") == 0.0


@pytest.mark.parametrize("total", [None, 0, -1.0])
def test_unknown_ram_gives_no_window(total):
    assert nb.npu_memory_window_gb(total, "win32") is None
    assert nb.fits_npu_memory(18.0, nb.npu_memory_window_gb(total, "win32")) is None


def test_32gb_windows_flags_the_flagship_models(tmp_path, monkeypatch):
    models = _catalog(tmp_path, monkeypatch, 32.0, "win32")
    for name in _FLAGSHIPS:
        assert models[name].fits_memory is False, name
    assert models["qwen3.8-mtp:27b"].to_json()["memory_needed_gb"] == 19.0
    assert models["qwen3:0.6b"].fits_memory is True
    assert models["no-size"].fits_memory is None
    assert models["no-size"].to_json()["memory_needed_gb"] is None


def test_64gb_windows_flags_neither(tmp_path, monkeypatch):
    models = _catalog(tmp_path, monkeypatch, 64.0, "win32")
    assert all(models[name].fits_memory is True for name in _FLAGSHIPS)


def test_32gb_linux_is_not_held_to_the_windows_half(tmp_path, monkeypatch):
    models = _catalog(tmp_path, monkeypatch, 32.0, "linux")
    assert all(models[name].fits_memory is True for name in _FLAGSHIPS)


def test_unknown_ram_flags_nothing(tmp_path, monkeypatch):
    models = _catalog(tmp_path, monkeypatch, None, "win32")
    assert all(model.fits_memory is None for model in models.values())


def test_status_reports_the_window(tmp_path, monkeypatch):
    monkeypatch.setattr(nb, "_total_ram_gb", lambda: 32.0)
    monkeypatch.setattr(nb.sys, "platform", "win32")
    monkeypatch.setattr(nb, "detect_amd_npu", lambda: {"present": True, "supported": True})
    backend = nb.LemonadeNpuBackend(root = tmp_path / "lemonade")
    monkeypatch.setattr(backend, "_installed_lemond", lambda: None)
    monkeypatch.setattr(backend, "_upgrade_pending", lambda binary: False)
    assert backend.status()["memory_window_gb"] == 16.0
