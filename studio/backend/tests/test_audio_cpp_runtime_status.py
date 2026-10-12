# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The ``audio_cpp_runtime`` block of /audio/stt/status the Audio page reads before a load."""

import asyncio
import json

import pytest

from core.inference import audio_cpp_server
from routes import inference
from utils import audio_cpp_update


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
        "expected_tag": None,
        "outdated": False,
    }


def test_runtime_available_without_espeak(monkeypatch):
    # custom or upstream builds lack install records and eSpeak data beside the server.
    _patch(monkeypatch, binary = "/usr/local/bin/audiocpp_server", record = {}, espeak = False)
    assert inference._audio_cpp_runtime_status() == {
        "available": True,
        "espeak": False,
        "backend": None,
        "release_tag": None,
        "expected_tag": None,
        "outdated": False,
    }


def test_runtime_missing(monkeypatch):
    _patch(monkeypatch, binary = None, espeak = True)
    assert inference._audio_cpp_runtime_status() == {
        "available": False,
        "espeak": False,
        "backend": None,
        "release_tag": None,
        "expected_tag": None,
        "outdated": False,
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


_LADDER = [("unslothai/audio.cpp", "v0.9.0-unsloth.1"), ("0xShug0/audio.cpp", "v0.9.0")]
# setup.sh / setup.ps1 leave the managed tree alone when any of these is set.
_SETUP_SKIPS = ("AUDIOCPP_SERVER_PATH", "UNSLOTH_AUDIO_CPP_PATH", "UNSLOTH_SKIP_AUDIO_CPP_INSTALL")


def _managed(
    monkeypatch,
    tmp_path,
    record,
    ladder = _LADDER,
):
    """a Studio-managed install of ``record`` and the releases setup would install."""
    for name in _SETUP_SKIPS:
        monkeypatch.delenv(name, raising = False)
    tmp_path.mkdir(parents = True, exist_ok = True)
    (tmp_path / ".unsloth-studio-owned").touch()
    _patch(monkeypatch, binary = str(tmp_path / "audiocpp_server"), record = record, espeak = True)
    monkeypatch.setattr(audio_cpp_server, "managed_audio_cpp_dir", lambda: tmp_path)
    monkeypatch.setattr(audio_cpp_update, "_release_ladder", lambda: ladder)


def test_managed_install_of_the_pinned_release_is_current(monkeypatch, tmp_path):
    _managed(
        monkeypatch,
        tmp_path,
        {"published_repo": "unslothai/audio.cpp", "release_tag": "v0.9.0-unsloth.1"},
    )
    status = inference._audio_cpp_runtime_status()
    assert status["expected_tag"] == "v0.9.0-unsloth.1"
    assert status["outdated"] is False


def test_managed_fallback_install_is_current(monkeypatch, tmp_path):
    # setup falls back to the upstream release when the fork cannot serve the host.
    _managed(
        monkeypatch, tmp_path, {"published_repo": "0xShug0/audio.cpp", "release_tag": "v0.9.0"}
    )
    assert inference._audio_cpp_runtime_status()["outdated"] is False


def test_managed_install_of_an_old_release_is_outdated(monkeypatch, tmp_path):
    # setup keeps the old tree when the release lookup fails after a pin bump.
    _managed(
        monkeypatch,
        tmp_path,
        {"published_repo": "unslothai/audio.cpp", "release_tag": "v0.8.0-unsloth.1"},
    )
    status = inference._audio_cpp_runtime_status()
    assert status["release_tag"] == "v0.8.0-unsloth.1"
    assert status["expected_tag"] == "v0.9.0-unsloth.1"
    assert status["outdated"] is True


def test_same_tag_from_another_repo_is_outdated(monkeypatch, tmp_path):
    _managed(
        monkeypatch, tmp_path, {"published_repo": "someone/audio.cpp", "release_tag": "v0.9.0"}
    )
    assert inference._audio_cpp_runtime_status()["outdated"] is True


def test_latest_tracking_ladder_cannot_tell(monkeypatch, tmp_path):
    # UNSLOTH_AUDIO_CPP_TAG='' tracks the latest release, which no record can be checked against.
    _managed(
        monkeypatch,
        tmp_path,
        {"published_repo": "unslothai/audio.cpp", "release_tag": "v0.8.0"},
        ladder = [("unslothai/audio.cpp", None)],
    )
    status = inference._audio_cpp_runtime_status()
    assert status["expected_tag"] is None and status["outdated"] is False


def test_managed_install_without_a_tag_is_not_flagged(monkeypatch, tmp_path):
    _managed(monkeypatch, tmp_path, {})
    status = inference._audio_cpp_runtime_status()
    assert status["release_tag"] is None and status["outdated"] is False


def test_managed_path_without_ownership_marker_is_not_flagged(monkeypatch, tmp_path):
    _managed(
        monkeypatch,
        tmp_path,
        {"published_repo": "unslothai/audio.cpp", "release_tag": "v0.8.0-unsloth.1"},
    )
    (tmp_path / ".unsloth-studio-owned").unlink()
    status = inference._audio_cpp_runtime_status()
    assert status["available"] is True
    assert status["expected_tag"] is None and status["outdated"] is False


def test_user_configured_binary_is_never_outdated(monkeypatch, tmp_path):
    # AUDIOCPP_SERVER_PATH, UNSLOTH_AUDIO_CPP_PATH or PATH: setup skips it, so updating cannot help.
    _managed(
        monkeypatch,
        tmp_path / "managed",
        {"published_repo": "unslothai/audio.cpp", "release_tag": "v0.8.0-unsloth.1"},
    )
    custom = str(tmp_path / "custom" / "audiocpp_server")
    monkeypatch.setattr(audio_cpp_server, "find_audio_cpp_server_binary", lambda: custom)
    status = inference._audio_cpp_runtime_status()
    assert status["available"] is True
    assert status["expected_tag"] is None and status["outdated"] is False


@pytest.mark.parametrize(
    "name, value",
    [
        # A stale path makes discovery fall back to the managed tree.
        ("AUDIOCPP_SERVER_PATH", "/gone/audiocpp_server"),
        ("UNSLOTH_AUDIO_CPP_PATH", "<managed>"),
        ("UNSLOTH_SKIP_AUDIO_CPP_INSTALL", "1"),
    ],
)
def test_managed_binary_is_not_outdated_when_setup_skips_it(monkeypatch, tmp_path, name, value):
    # An update would not touch the tree, so its notice could never clear.
    _managed(
        monkeypatch,
        tmp_path,
        {"published_repo": "unslothai/audio.cpp", "release_tag": "v0.8.0-unsloth.1"},
    )
    monkeypatch.setenv(name, str(tmp_path) if value == "<managed>" else value)
    status = inference._audio_cpp_runtime_status()
    assert status["available"] is True
    assert status["expected_tag"] is None and status["outdated"] is False


def test_release_lookup_error_is_not_outdated(monkeypatch, tmp_path):
    _managed(
        monkeypatch,
        tmp_path,
        {"published_repo": "unslothai/audio.cpp", "release_tag": "v0.8.0-unsloth.1"},
    )

    def boom():
        raise ImportError("installer missing")

    monkeypatch.setattr(audio_cpp_update, "_release_ladder", boom)
    status = inference._audio_cpp_runtime_status()
    assert status["available"] is True and status["outdated"] is False


def test_release_ladder_comes_from_the_installer(monkeypatch):
    monkeypatch.delenv("UNSLOTH_AUDIO_CPP_REPO", raising = False)
    monkeypatch.delenv("UNSLOTH_AUDIO_CPP_TAG", raising = False)
    import install_audio_cpp_prebuilt as installer

    ladder = audio_cpp_update._release_ladder()
    assert ladder[0] == (installer.DEFAULT_REPO, installer.DEFAULT_TAG)
    assert (installer.UPSTREAM_FALLBACK_REPO, installer.UPSTREAM_FALLBACK_TAG) in ladder


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
        "expected_tag": None,
        "outdated": False,
    }
