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


class WindowsSymlinkPrivilegeError(PermissionError):
    winerror = 1314


class WindowsSymlinkPrivilegeOSError(OSError):
    """1314 is unmapped in CPython's errmap, so native Windows raises plain OSError."""

    winerror = 1314


def _install_fake_datasets(monkeypatch, load_dataset):
    module = types.ModuleType("datasets")
    module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", module)
    monkeypatch.setattr("loggers.config.quiet_third_party_progress_bars", lambda: None)


def _isolate_hub_symlink_state(monkeypatch):
    """Keep _disable_hf_symlinks_for_process from leaking into other tests."""
    from huggingface_hub import constants, file_download

    monkeypatch.setattr(constants, "HF_HUB_DISABLE_SYMLINKS", False, raising = False)
    monkeypatch.setattr(file_download, "_are_symlinks_supported_in_dir", {})


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
    from huggingface_hub import file_download

    monkeypatch.setattr(constants, "HF_HUB_DISABLE_SYMLINKS", False, raising = False)
    monkeypatch.setattr(
        file_download,
        "_are_symlinks_supported_in_dir",
        {"dataset-cache": True},
    )

    assert cache_safe.load_dataset_cache_safe("Org/Data", split = "train") == {"loaded": True}
    assert len(calls) == 2
    assert os.environ["HF_HUB_DISABLE_SYMLINKS"] == "1"
    assert constants.HF_HUB_DISABLE_SYMLINKS is True
    assert file_download._are_symlinks_supported_in_dir == {"dataset-cache": False}


def test_pre_1_x_hub_retries_by_updating_symlink_capability_cache(monkeypatch):
    calls = []

    from huggingface_hub import constants
    from huggingface_hub import file_download

    monkeypatch.delattr(constants, "HF_HUB_DISABLE_SYMLINKS", raising = False)
    monkeypatch.setattr(
        file_download,
        "_are_symlinks_supported_in_dir",
        {"dataset-cache": True},
    )

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs.copy()))
        if len(calls) == 1:
            raise WindowsSymlinkPrivilegeError("symlink denied")
        assert file_download._are_symlinks_supported_in_dir == {"dataset-cache": False}
        return {"loaded": True}

    _install_fake_datasets(monkeypatch, load_dataset)
    monkeypatch.setattr(cache_safe, "_is_native_windows", lambda: True)
    monkeypatch.delenv("HF_HUB_DISABLE_SYMLINKS", raising = False)

    assert cache_safe.load_dataset_cache_safe("Org/Data") == {"loaded": True}
    assert len(calls) == 2
    assert "HF_HUB_DISABLE_SYMLINKS" not in vars(constants)
    assert os.environ["HF_HUB_DISABLE_SYMLINKS"] == "1"


def test_pre_1_9_hub_stops_re_probing_unknown_cache_dirs(monkeypatch, tmp_path):
    # Hub added HF_HUB_DISABLE_SYMLINKS in 1.9; before that, are_symlinks_supported
    # probes any dir missing from this mapping, which can lose the same race twice.
    # Assert the mapping itself, since which Hub reads it varies by version.
    from huggingface_hub import file_download

    monkeypatch.setattr(file_download, "_are_symlinks_supported_in_dir", {})
    monkeypatch.delenv("HF_HUB_DISABLE_SYMLINKS", raising = False)

    cache_safe._disable_hf_symlinks_for_process()

    unknown = str(tmp_path / "never-probed")
    capability = file_download._are_symlinks_supported_in_dir
    assert unknown in capability
    assert capability[unknown] is False


def test_symlink_disable_survives_an_unimportable_hub(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    monkeypatch.delenv("HF_HUB_DISABLE_SYMLINKS", raising = False)

    cache_safe._disable_hf_symlinks_for_process()

    assert os.environ["HF_HUB_DISABLE_SYMLINKS"] == "1"


def test_success_does_not_change_windows_symlink_policy(monkeypatch):
    calls = []

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return {"loaded": True}

    _install_fake_datasets(monkeypatch, load_dataset)
    monkeypatch.setattr(cache_safe, "_is_native_windows", lambda: True)
    monkeypatch.delenv("HF_HUB_DISABLE_SYMLINKS", raising = False)

    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_DISABLE_SYMLINKS", False, raising = False)

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


def test_repeated_symlink_failure_falls_back_to_studio_cache(monkeypatch, tmp_path):
    calls = []
    fallback = str(tmp_path / "fallback")

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs.copy()))
        if len(calls) == 1:
            raise WindowsSymlinkPrivilegeError("symlink denied")
        if len(calls) == 2:
            raise WindowsSymlinkPrivilegeOSError("symlink denied again")
        return {"loaded": True}

    _install_fake_datasets(monkeypatch, load_dataset)
    monkeypatch.setattr(cache_safe, "_is_native_windows", lambda: True)
    monkeypatch.setattr(cache_safe, "studio_datasets_cache", lambda: fallback)
    monkeypatch.delenv("HF_HUB_DISABLE_SYMLINKS", raising = False)
    _isolate_hub_symlink_state(monkeypatch)

    assert cache_safe.load_dataset_cache_safe("Org/Data") == {"loaded": True}
    assert calls == [
        (("Org/Data",), {}),
        (("Org/Data",), {}),
        (("Org/Data",), {"cache_dir": fallback}),
    ]


def test_unrelated_error_on_symlink_retry_is_raised(monkeypatch):
    calls = []

    class UnrelatedError(OSError):
        pass

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1:
            raise WindowsSymlinkPrivilegeError("symlink denied")
        raise UnrelatedError("unrelated")

    _install_fake_datasets(monkeypatch, load_dataset)
    monkeypatch.setattr(cache_safe, "_is_native_windows", lambda: True)
    monkeypatch.delenv("HF_HUB_DISABLE_SYMLINKS", raising = False)
    _isolate_hub_symlink_state(monkeypatch)

    with pytest.raises(UnrelatedError, match = "unrelated"):
        cache_safe.load_dataset_cache_safe("Org/Data")
    assert len(calls) == 2


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
