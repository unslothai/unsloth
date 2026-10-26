# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Voice Settings "Load into" persisted for cold STT API loads (#15280)."""

from pathlib import Path
import sys

import pytest
from fastapi import HTTPException

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import audio_device
from routes import settings as settings_routes
from storage import studio_db
from utils import stt_device_settings as sds


@pytest.fixture(autouse = True)
def isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    monkeypatch.delenv("UNSLOTH_AUDIO_DEVICE", raising = False)
    sds.forget_cached_stt_device()
    yield
    sds.forget_cached_stt_device()


def _put(device):
    return settings_routes.update_stt_device_setting(
        settings_routes.SttDevicePayload(device = device), current_subject = "owner"
    )


def test_unset_defaults_to_auto():
    response = settings_routes.get_stt_device_setting(current_subject = "owner")
    assert response.device == "auto"
    assert response.locked is False
    assert response.saved is False


def test_saving_cpu_is_what_a_device_less_cold_load_uses():
    response = _put("cpu")
    assert response.device == "cpu"
    assert audio_device.stt_device_default() == "cpu"
    assert audio_device.stt_device_forces_cpu(None)
    assert audio_device.audio_device_default() == "auto"


def test_env_locks_settings(monkeypatch):
    _put("cpu")
    monkeypatch.setenv("UNSLOTH_AUDIO_DEVICE", "gpu")
    with pytest.raises(HTTPException) as raised:
        _put("auto")
    assert raised.value.status_code == 409
    assert audio_device.stt_device_default() == "gpu"
    assert audio_device.audio_device_env_locked() is True


def test_an_explicit_auto_is_saved_so_another_browser_adopts_it():
    """Unset and a chosen Auto both read as auto, but only a choice stops a browser migrating its own."""
    assert _put("auto").saved is True
    assert settings_routes.get_stt_device_setting(current_subject = "owner").device == "auto"
    assert _put("cpu").saved is True
