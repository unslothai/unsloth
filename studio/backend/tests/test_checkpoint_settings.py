# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path

from utils import checkpoint_settings as settings


def test_colab_prefers_mounted_drive(monkeypatch, tmp_path):
    monkeypatch.setenv("COLAB_BACKEND_URL", "https://example.invalid")
    monkeypatch.delenv("UNSLOTH_OUTPUTS_DIR", raising = False)
    real_is_dir = Path.is_dir

    def fake_is_dir(path):
        if str(path) == "/content/drive/MyDrive":
            return True
        return real_is_dir(path)

    monkeypatch.setattr(Path, "is_dir", fake_is_dir)
    location = settings._detected_default()
    assert location.source == "colab"
    assert location.path == Path("/content/drive/MyDrive/unsloth_outputs")


def test_kaggle_default(monkeypatch):
    monkeypatch.delenv("UNSLOTH_OUTPUTS_DIR", raising = False)
    monkeypatch.delenv("COLAB_BACKEND_URL", raising = False)
    monkeypatch.delenv("COLAB_JUPYTER_IP", raising = False)
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    monkeypatch.setattr(Path, "is_dir", lambda path: False)
    location = settings._detected_default()
    assert location.source == "kaggle"
    assert location.path == Path("/kaggle/working/unsloth_outputs")


def test_environment_override_is_not_editable(monkeypatch, tmp_path):
    selected = tmp_path / "checkpoints"
    monkeypatch.setenv("UNSLOTH_OUTPUTS_DIR", str(selected))
    location = settings.get_checkpoint_location()
    assert location.path == selected
    assert location.source == "environment"
    assert location.editable is False
    assert location.environment_variable == "UNSLOTH_OUTPUTS_DIR"
