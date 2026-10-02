# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The ``audio_cpp_runtime`` block of /audio/stt/status the Audio page reads before a load."""

import asyncio
import json

from core.inference import audio_cpp_server
from routes import inference


def _patch(
    monkeypatch,
    *,
    binary,
    record = None,
    espeak = False,
):
    monkeypatch.setattr(audio_cpp_server, "find_audio_cpp_server_binary", lambda: binary)
    monkeypatch.setattr(audio_cpp_server, "read_install_record", lambda _binary: record or {})
    monkeypatch.setattr(audio_cpp_server, "binary_has_espeak", lambda _binary: espeak)


def test_runtime_available_with_espeak(monkeypatch):
    _patch(
        monkeypatch,
        binary = "/opt/audio.cpp/audiocpp_server",
        record = {"backend": "cuda", "release_tag": "v0.4.0"},
        espeak = True,
    )
    assert inference._audio_cpp_runtime_status() == {
        "available": True,
        "espeak": True,
        "backend": "cuda",
        "release_tag": "v0.4.0",
    }


def test_runtime_available_without_espeak(monkeypatch):
    # A custom or upstream build: no install record, no eSpeak data beside the server.
    _patch(monkeypatch, binary = "/usr/local/bin/audiocpp_server", record = {}, espeak = False)
    assert inference._audio_cpp_runtime_status() == {
        "available": True,
        "espeak": False,
        "backend": None,
        "release_tag": None,
    }


def test_runtime_missing(monkeypatch):
    _patch(monkeypatch, binary = None, espeak = True)
    assert inference._audio_cpp_runtime_status() == {
        "available": False,
        "espeak": False,
        "backend": None,
        "release_tag": None,
    }


def test_runtime_probe_error_reports_unavailable(monkeypatch):
    def boom():
        raise OSError("permission denied")

    monkeypatch.setattr(audio_cpp_server, "find_audio_cpp_server_binary", boom)
    assert inference._audio_cpp_runtime_status()["available"] is False


def test_malformed_record_fields_are_dropped(monkeypatch):
    _patch(
        monkeypatch,
        binary = "/opt/audio.cpp/audiocpp_server",
        record = {"backend": ["cuda"], "release_tag": 7},
        espeak = False,
    )
    status = inference._audio_cpp_runtime_status()
    assert status["backend"] is None and status["release_tag"] is None


def test_stt_status_payload_includes_runtime_block(monkeypatch):
    _patch(
        monkeypatch,
        binary = "/opt/audio.cpp/audiocpp_server",
        record = {"backend": "vulkan", "release_tag": "v0.4.0"},
        espeak = False,
    )
    monkeypatch.setattr(inference.account_access, "managed_account", lambda: False)
    response = asyncio.run(inference.stt_status(model = None, current_subject = "owner"))
    body = json.loads(response.body)
    assert body["audio_cpp_runtime"] == {
        "available": True,
        "espeak": False,
        "backend": "vulkan",
        "release_tag": "v0.4.0",
    }
