# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The fallback half of unsloth/disk_utils.py, used when unsloth_zoo has no disk_utils."""

import importlib.util
import pathlib
import sys

import pytest

_DISK_UTILS = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "disk_utils.py"


@pytest.fixture
def disk_utils(monkeypatch):
    # A None entry in sys.modules makes `from unsloth_zoo.disk_utils import ...` raise ImportError.
    monkeypatch.setitem(sys.modules, "unsloth_zoo", None)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.disk_utils", None)
    for key in ("UNSLOTH_IS_KAGGLE", "KAGGLE_KERNEL_RUN_TYPE"):
        monkeypatch.delenv(key, raising = False)
    spec = importlib.util.spec_from_file_location("_disk_utils_fallback", _DISK_UTILS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fallback_is_used_without_zoo(disk_utils):
    assert disk_utils.HAS_ZOO_DISK_UTILS is False
    assert disk_utils.KAGGLE_TMP == "/tmp"
    assert disk_utils.KAGGLE_WORKING == "/kaggle/working"
    for name in disk_utils.__all__:
        assert hasattr(disk_utils, name), name


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_kaggle_override_true(disk_utils, monkeypatch, value):
    monkeypatch.setenv("UNSLOTH_IS_KAGGLE", value)
    assert disk_utils.is_kaggle_environment() is True


@pytest.mark.parametrize("value", ["0", "false", "no", ""])
def test_kaggle_override_false_wins_over_kaggle_env(disk_utils, monkeypatch, value):
    monkeypatch.setenv("UNSLOTH_IS_KAGGLE", value)
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    monkeypatch.setattr(disk_utils.os.path, "isdir", lambda path: True)
    assert disk_utils.is_kaggle_environment() is False


def test_kaggle_needs_run_type_and_working_dir(disk_utils, monkeypatch):
    monkeypatch.setattr(disk_utils.os.path, "isdir", lambda path: path == "/kaggle/working")
    assert disk_utils.is_kaggle_environment() is False

    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "   ")
    assert disk_utils.is_kaggle_environment() is False

    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    assert disk_utils.is_kaggle_environment() is True

    monkeypatch.setattr(disk_utils.os.path, "isdir", lambda path: False)
    assert disk_utils.is_kaggle_environment() is False


def test_kaggle_isdir_error_is_not_kaggle(disk_utils, monkeypatch):
    def broken_isdir(path):
        raise OSError("permission denied")

    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    monkeypatch.setattr(disk_utils.os.path, "isdir", broken_isdir)
    assert disk_utils.is_kaggle_environment() is False


def test_colab_detected_from_any_colab_variable(disk_utils, monkeypatch):
    for key in list(disk_utils.os.environ):
        if key.startswith("COLAB_"):
            monkeypatch.delenv(key)
    assert disk_utils.is_colab_environment() is False
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-colab")
    assert disk_utils.is_colab_environment() is True


def test_free_bytes_walks_up_to_an_existing_parent(disk_utils, monkeypatch, tmp_path):
    import shutil

    # Assert the probed path, not two free-space readings: other writers move them on a shared disk.
    probed = []
    real_disk_usage = shutil.disk_usage
    monkeypatch.setattr(
        shutil, "disk_usage", lambda path: probed.append(path) or real_disk_usage(path)
    )
    missing = tmp_path / "not" / "created" / "yet"
    assert disk_utils.free_bytes(missing) > 0
    assert probed == [str(tmp_path)]


def test_free_bytes_returns_none_when_disk_usage_fails(disk_utils, monkeypatch, tmp_path):
    import shutil

    def broken_disk_usage(path):
        raise OSError("no such device")

    monkeypatch.setattr(shutil, "disk_usage", broken_disk_usage)
    assert disk_utils.free_bytes(tmp_path) is None


def test_logical_numel_uses_numel_and_never_raises(disk_utils):
    class Param:
        def numel(self):
            return 12

    class Broken:
        def numel(self):
            raise RuntimeError("meta tensor")

    assert disk_utils.logical_numel(Param()) == 12
    assert disk_utils.logical_numel(Broken()) == 0
    assert disk_utils.logical_numel(object()) == 0


def test_sizing_stubs_report_unmeasurable(disk_utils):
    # 0 tells callers the size is unknown, so the disk guard never blocks a save.
    assert disk_utils.model_logical_numel(object()) == 0
    assert disk_utils.model_16bit_bytes(object()) == 0
    assert disk_utils.estimate_gguf_export_bytes(object(), "q4_k_m", extra = 1) == 0


def test_kaggle_tmp_redirect_is_a_no_op(disk_utils):
    assert disk_utils.kaggle_tmp_redirect("out_dir", 10, quiet = True) == ("out_dir", None)
