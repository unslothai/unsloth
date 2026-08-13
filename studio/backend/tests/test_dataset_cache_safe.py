# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import sys
import types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from utils.datasets import cache_safe


class WindowsSymlinkPrivilegeError(OSError):
    winerror = 1314


def _install_fake_datasets(monkeypatch, load_dataset):
    module = types.ModuleType("datasets")
    module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", module)
    monkeypatch.setattr("loggers.config.quiet_third_party_progress_bars", lambda: None)


def test_windows_symlink_privilege_failure_retries_with_regular_files(monkeypatch):
    calls = []

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs.copy()))
        if len(calls) == 1:
            raise WindowsSymlinkPrivilegeError("symlink denied")
        return {"loaded": True}

    _install_fake_datasets(monkeypatch, load_dataset)
    monkeypatch.setattr(cache_safe, "_is_native_windows", lambda: True)
    monkeypatch.delenv("HF_HUB_DISABLE_SYMLINKS", raising = False)

    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_DISABLE_SYMLINKS", False)

    assert cache_safe.load_dataset_cache_safe("Org/Data", split = "train") == {"loaded": True}
    assert len(calls) == 2
    assert os.environ["HF_HUB_DISABLE_SYMLINKS"] == "1"
    assert constants.HF_HUB_DISABLE_SYMLINKS is True


def test_success_does_not_change_windows_symlink_policy(monkeypatch):
    calls = []

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return {"loaded": True}

    _install_fake_datasets(monkeypatch, load_dataset)
    monkeypatch.setattr(cache_safe, "_is_native_windows", lambda: True)
    monkeypatch.delenv("HF_HUB_DISABLE_SYMLINKS", raising = False)

    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_DISABLE_SYMLINKS", False)

    assert cache_safe.load_dataset_cache_safe("Org/Data") == {"loaded": True}
    assert len(calls) == 1
    assert "HF_HUB_DISABLE_SYMLINKS" not in os.environ
    assert constants.HF_HUB_DISABLE_SYMLINKS is False


def test_permission_error_still_retries_in_studio_cache(monkeypatch, tmp_path):
    calls = []
    fallback = str(tmp_path / "fallback")

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs.copy(), os.environ.get("HF_DATASETS_CACHE")))
        if len(calls) == 1:
            raise PermissionError("shared cache is not writable")
        return {"loaded": True}

    _install_fake_datasets(monkeypatch, load_dataset)
    monkeypatch.setattr(cache_safe, "studio_datasets_cache", lambda: fallback)
    monkeypatch.setenv("HF_DATASETS_CACHE", "original")

    assert cache_safe.load_dataset_cache_safe("Org/Data") == {"loaded": True}
    assert calls == [
        (("Org/Data",), {}, "original"),
        (("Org/Data",), {"cache_dir": fallback}, fallback),
    ]
    assert os.environ["HF_DATASETS_CACHE"] == "original"


@pytest.mark.parametrize(
    ("is_windows", "winerror"),
    ((False, 1314), (True, 5)),
)
def test_unrelated_os_errors_are_not_retried(monkeypatch, is_windows, winerror):
    calls = []

    class UnrelatedError(OSError):
        pass

    error = UnrelatedError("unrelated")
    error.winerror = winerror

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        raise error

    _install_fake_datasets(monkeypatch, load_dataset)
    monkeypatch.setattr(cache_safe, "_is_native_windows", lambda: is_windows)

    with pytest.raises(UnrelatedError, match = "unrelated"):
        cache_safe.load_dataset_cache_safe("Org/Data")
    assert len(calls) == 1
