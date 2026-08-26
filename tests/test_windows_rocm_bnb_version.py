# Unsloth - 2x faster, 60% less VRAM LLM training and finetuning
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

"""Tests for ``maybe_set_windows_rocm_bnb_version`` (loaded in isolation, no torch/GPU)."""

from __future__ import annotations

import importlib.util
import os
import types
from pathlib import Path

import pytest

_IMPORT_FIXES_PATH = Path(__file__).resolve().parent.parent / "unsloth" / "import_fixes.py"


def _load_import_fixes():
    spec = importlib.util.spec_from_file_location(
        "unsloth_import_fixes_under_test", _IMPORT_FIXES_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def import_fixes():
    return _load_import_fixes()


@pytest.fixture()
def clean_env(monkeypatch):
    """Unset the env vars and remove them afterwards; the function writes
    os.environ directly, which monkeypatch does not auto-revert."""
    for var in (
        "BNB_ROCM_VERSION",
        "UNSLOTH_SKIP_BNB_ROCM_VERSION",
        "UNSLOTH_BNB_ROCM_VERSION_SOURCE",
    ):
        monkeypatch.delenv(var, raising = False)
    yield monkeypatch
    for var in (
        "BNB_ROCM_VERSION",
        "UNSLOTH_SKIP_BNB_ROCM_VERSION",
        "UNSLOTH_BNB_ROCM_VERSION_SOURCE",
    ):
        os.environ.pop(var, None)


def _force(import_fixes, monkeypatch, *, win, rocm, detected):
    monkeypatch.setattr(import_fixes.sys, "platform", "win32" if win else "linux")
    monkeypatch.setattr(import_fixes, "_is_hip_torch_build", lambda: rocm)
    monkeypatch.setattr(import_fixes, "_detect_installed_bnb_rocm_version", lambda: detected)


# ---------------------------------------------------------------------------
# _detect_installed_bnb_rocm_version
# ---------------------------------------------------------------------------


def test_detect_picks_highest_rocm_suffix(import_fixes, tmp_path, monkeypatch):
    pkg = tmp_path / "bitsandbytes"
    pkg.mkdir()
    for name in (
        "libbitsandbytes_rocm72.dll",
        "libbitsandbytes_rocm713.dll",  # numerically highest -> wins
        "libbitsandbytes_cpu.dll",
        "__init__.py",
    ):
        (pkg / name).write_text("")
    fake_spec = types.SimpleNamespace(submodule_search_locations = [str(pkg)])
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: fake_spec)
    assert import_fixes._detect_installed_bnb_rocm_version() == "713"


def test_detect_none_when_only_non_rocm_dlls(import_fixes, tmp_path, monkeypatch):
    pkg = tmp_path / "bitsandbytes"
    pkg.mkdir()
    (pkg / "libbitsandbytes_cpu.dll").write_text("")
    (pkg / "libbitsandbytes_cuda124.dll").write_text("")
    fake_spec = types.SimpleNamespace(submodule_search_locations = [str(pkg)])
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: fake_spec)
    assert import_fixes._detect_installed_bnb_rocm_version() is None


def test_detect_none_when_bnb_absent(import_fixes, monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert import_fixes._detect_installed_bnb_rocm_version() is None


# ---------------------------------------------------------------------------
# maybe_set_windows_rocm_bnb_version
# ---------------------------------------------------------------------------


def test_sets_bnb_version_on_windows_rocm(import_fixes, clean_env):
    _force(import_fixes, clean_env, win = True, rocm = True, detected = "72")
    assert import_fixes.maybe_set_windows_rocm_bnb_version() == "72"
    assert os.environ["BNB_ROCM_VERSION"] == "72"
    assert os.environ["UNSLOTH_BNB_ROCM_VERSION_SOURCE"] == "detected"


def test_noop_off_windows(import_fixes, clean_env):
    # Linux ROCm resolves its backend correctly from torch.version.hip.
    _force(import_fixes, clean_env, win = False, rocm = True, detected = "72")
    assert import_fixes.maybe_set_windows_rocm_bnb_version() is None
    assert "BNB_ROCM_VERSION" not in os.environ


def test_noop_when_not_rocm_torch(import_fixes, clean_env):
    _force(import_fixes, clean_env, win = True, rocm = False, detected = "72")
    assert import_fixes.maybe_set_windows_rocm_bnb_version() is None
    assert "BNB_ROCM_VERSION" not in os.environ


def test_noop_when_no_rocm_dll_installed(import_fixes, clean_env):
    # No ROCm DLL ships -> don't force a backend name (would break non-ROCm bnb).
    _force(import_fixes, clean_env, win = True, rocm = True, detected = None)
    assert import_fixes.maybe_set_windows_rocm_bnb_version() is None
    assert "BNB_ROCM_VERSION" not in os.environ


def test_respects_user_provided_value(import_fixes, clean_env):
    clean_env.setenv("BNB_ROCM_VERSION", "999")
    _force(import_fixes, clean_env, win = True, rocm = True, detected = "72")
    assert import_fixes.maybe_set_windows_rocm_bnb_version() is None
    assert os.environ["BNB_ROCM_VERSION"] == "999"


def test_explicit_opt_out(import_fixes, clean_env):
    clean_env.setenv("UNSLOTH_SKIP_BNB_ROCM_VERSION", "1")
    _force(import_fixes, clean_env, win = True, rocm = True, detected = "72")
    assert import_fixes.maybe_set_windows_rocm_bnb_version() is None
    assert "BNB_ROCM_VERSION" not in os.environ


def test_redetects_sitecustomize_seeded_default(import_fixes, clean_env):
    # A sitecustomize-seeded default must be redetected (the wheel may have changed).
    clean_env.setenv("BNB_ROCM_VERSION", "72")
    clean_env.setenv("UNSLOTH_BNB_ROCM_VERSION_SOURCE", "sitecustomize")
    _force(import_fixes, clean_env, win = True, rocm = True, detected = "713")
    assert import_fixes.maybe_set_windows_rocm_bnb_version() == "713"
    assert os.environ["BNB_ROCM_VERSION"] == "713"
    assert os.environ["UNSLOTH_BNB_ROCM_VERSION_SOURCE"] == "detected"


def test_sitecustomize_default_kept_when_no_dll_found(import_fixes, clean_env):
    # A failed redetect must not discard the seeded value.
    clean_env.setenv("BNB_ROCM_VERSION", "72")
    clean_env.setenv("UNSLOTH_BNB_ROCM_VERSION_SOURCE", "sitecustomize")
    _force(import_fixes, clean_env, win = True, rocm = True, detected = None)
    assert import_fixes.maybe_set_windows_rocm_bnb_version() is None
    assert os.environ["BNB_ROCM_VERSION"] == "72"
    assert os.environ["UNSLOTH_BNB_ROCM_VERSION_SOURCE"] == "sitecustomize"


def test_user_value_with_non_sitecustomize_marker_untouched(import_fixes, clean_env):
    # Only the sitecustomize marker makes a value redetectable.
    clean_env.setenv("BNB_ROCM_VERSION", "999")
    clean_env.setenv("UNSLOTH_BNB_ROCM_VERSION_SOURCE", "detected")
    _force(import_fixes, clean_env, win = True, rocm = True, detected = "72")
    assert import_fixes.maybe_set_windows_rocm_bnb_version() is None
    assert os.environ["BNB_ROCM_VERSION"] == "999"


def test_opt_out_unseats_sitecustomize_seeded_value(import_fixes, clean_env):
    # Opt-out must also drop a sitecustomize-seeded default so bnb never sees it.
    clean_env.setenv("BNB_ROCM_VERSION", "72")
    clean_env.setenv("UNSLOTH_BNB_ROCM_VERSION_SOURCE", "sitecustomize")
    clean_env.setenv("UNSLOTH_SKIP_BNB_ROCM_VERSION", "1")
    _force(import_fixes, clean_env, win = True, rocm = True, detected = "713")
    assert import_fixes.maybe_set_windows_rocm_bnb_version() is None
    assert "BNB_ROCM_VERSION" not in os.environ
    assert "UNSLOTH_BNB_ROCM_VERSION_SOURCE" not in os.environ


def test_opt_out_keeps_explicit_user_value(import_fixes, clean_env):
    # Opt-out must never remove a value the user set themselves (no marker).
    clean_env.setenv("BNB_ROCM_VERSION", "999")
    clean_env.setenv("UNSLOTH_SKIP_BNB_ROCM_VERSION", "1")
    _force(import_fixes, clean_env, win = True, rocm = True, detected = "72")
    assert import_fixes.maybe_set_windows_rocm_bnb_version() is None
    assert os.environ["BNB_ROCM_VERSION"] == "999"


def test_empty_string_value_without_marker_is_respected(import_fixes, clean_env):
    # "" counts as present: without the sitecustomize marker it is not ours to overwrite.
    clean_env.setenv("BNB_ROCM_VERSION", "")
    _force(import_fixes, clean_env, win = True, rocm = True, detected = "72")
    assert import_fixes.maybe_set_windows_rocm_bnb_version() is None
    assert os.environ["BNB_ROCM_VERSION"] == ""


# ---------------------------------------------------------------------------
# _is_hip_torch_build: strict gate; HIP-SDK env hints (HIP_PATH) must NOT count
# (regression for the HIP-SDK-on-a-CUDA-box false positive).
# ---------------------------------------------------------------------------


def _fake_torch(hip):
    return types.SimpleNamespace(version = types.SimpleNamespace(hip = hip))


def test_hip_build_true_from_wheel_tag(import_fixes, monkeypatch):
    monkeypatch.setattr(import_fixes, "importlib_version", lambda name: "2.11.0+rocm7.13.0")
    assert import_fixes._is_hip_torch_build() is True


def test_hip_build_true_from_torch_version_hip(import_fixes, monkeypatch):
    # Custom/source HIP build without the +rocm wheel tag.
    monkeypatch.setattr(import_fixes, "importlib_version", lambda name: "2.11.0")
    monkeypatch.setitem(__import__("sys").modules, "torch", _fake_torch("7.2.0"))
    assert import_fixes._is_hip_torch_build() is True


def test_hip_build_false_for_cuda_torch_despite_rocm_env_hints(import_fixes, monkeypatch):
    """HIP SDK env vars but CUDA torch: gate must be False, else CUDA bnb raises."""
    monkeypatch.setenv("HIP_PATH", r"C:\Program Files\AMD\ROCm\6.2")
    monkeypatch.setenv("ROCM_PATH", r"C:\Program Files\AMD\ROCm\6.2")
    monkeypatch.setattr(import_fixes, "importlib_version", lambda name: "2.9.0+cu126")
    monkeypatch.setitem(__import__("sys").modules, "torch", _fake_torch(None))
    assert import_fixes._is_hip_torch_build() is False


def test_hip_build_false_when_torch_absent(import_fixes, monkeypatch):
    def _raise(name):
        raise Exception("no torch dist")

    monkeypatch.setattr(import_fixes, "importlib_version", _raise)
    monkeypatch.setitem(__import__("sys").modules, "torch", None)
    assert import_fixes._is_hip_torch_build() is False
