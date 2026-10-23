# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The ``audio_cpp_runtime`` block of /audio/stt/status the Audio page reads before a load."""

import asyncio
import json
import time

import pytest

from core.inference import audio_cpp_server
from routes import inference
from utils import audio_cpp_freshness, audio_cpp_update
from utils.prebuilt import freshness_flow


def _patch(
    monkeypatch,
    *,
    binary,
    record = None,
    espeak = False,
    vendors = None,
):
    monkeypatch.setattr(audio_cpp_server, "find_audio_cpp_server_binary", lambda: binary)
    monkeypatch.setattr(audio_cpp_server, "read_install_record", lambda _binary: record or {})
    monkeypatch.setattr(audio_cpp_server, "binary_has_espeak", lambda _binary: espeak)
    # The launch backend must not depend on this host's GPUs or a CPU-only build's --version probe.
    monkeypatch.delenv("UNSLOTH_AUDIO_CPP_BACKEND", raising = False)
    monkeypatch.setattr(audio_cpp_server, "_gpu_vendors", lambda: vendors)
    monkeypatch.setattr(audio_cpp_server, "compiled_backends", lambda _binary: frozenset({"cpu"}))


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
        "launch_backend": "cuda",
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
        "launch_backend": "cpu",
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


def test_cuda_build_with_no_visible_gpu_launches_on_cpu(monkeypatch):
    # CUDA_VISIBLE_DEVICES="" hides every GPU: the build still says cuda, the launch goes to the CPU.
    _patch(
        monkeypatch,
        binary = "/opt/audio.cpp/audiocpp_server",
        record = {"backend": "cuda", "release_tag": "v0.4.0"},
        vendors = set(),
    )
    status = inference._audio_cpp_runtime_status()
    assert status["backend"] == "cuda"
    assert status["launch_backend"] == "cpu"


def test_launch_backend_error_keeps_the_runtime_available(monkeypatch):
    _patch(monkeypatch, binary = "/opt/audio.cpp/audiocpp_server", record = {"backend": "cuda"})

    def boom(_binary, _force_cpu):
        raise RuntimeError("symlink loop")

    monkeypatch.setattr(audio_cpp_server, "select_backend", boom)
    status = inference._audio_cpp_runtime_status()
    assert status["available"] is True and status["launch_backend"] is None


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


_REPO = "unslothai/audio.cpp"
_LATEST = "v0.9.1-unsloth.1"
# setup.sh / setup.ps1 leave the managed tree alone when any of these is set.
_SETUP_SKIPS = ("AUDIOCPP_SERVER_PATH", "UNSLOTH_AUDIO_CPP_PATH", "UNSLOTH_SKIP_AUDIO_CPP_INSTALL")


def _managed(
    monkeypatch,
    tmp_path,
    record,
    *,
    pin = None,
    cached = _LATEST,
):
    """an Unsloth Studio-managed install of ``record``, the configured target, and the newest release already cached."""
    for name in (*_SETUP_SKIPS, "UNSLOTH_DISABLE_UPDATE_CHECK"):
        monkeypatch.delenv(name, raising = False)
    tmp_path.mkdir(parents = True, exist_ok = True)
    (tmp_path / ".unsloth-studio-owned").touch()
    _patch(monkeypatch, binary = str(tmp_path / "audiocpp_server"), record = record, espeak = True)
    monkeypatch.setattr(audio_cpp_server, "managed_audio_cpp_dir", lambda: tmp_path)
    monkeypatch.setattr(audio_cpp_update, "_target", lambda: (_REPO, pin))
    monkeypatch.setattr(audio_cpp_freshness, "_cache_dir", lambda: tmp_path / ".freshness")
    audio_cpp_freshness.reset_caches()
    if cached is not None:
        audio_cpp_freshness._release_memo[_REPO] = (time.time(), cached)


@pytest.fixture(autouse = True)
def _forget_the_release_cache():
    yield
    audio_cpp_freshness.reset_caches()


def test_managed_install_of_the_newest_release_is_current(monkeypatch, tmp_path):
    _managed(monkeypatch, tmp_path, {"published_repo": _REPO, "release_tag": _LATEST})
    status = inference._audio_cpp_runtime_status()
    assert status["expected_tag"] == _LATEST
    assert status["outdated"] is False


def test_managed_install_newer_than_the_cached_latest_is_current(monkeypatch, tmp_path):
    # A stale cache can only under-report: the older "latest" is never offered as a downgrade.
    _managed(
        monkeypatch,
        tmp_path,
        {"published_repo": _REPO, "release_tag": _LATEST},
        cached = "v0.9.0-unsloth.1",
    )
    assert inference._audio_cpp_runtime_status()["outdated"] is False


def test_managed_upstream_install_is_outdated(monkeypatch, tmp_path):
    # The removed 0xShug0 fallback rung could leave such a record; an update installs the fork's newest.
    _managed(
        monkeypatch, tmp_path, {"published_repo": "0xShug0/audio.cpp", "release_tag": "v0.9.0"}
    )
    status = inference._audio_cpp_runtime_status()
    assert status["expected_tag"] == _LATEST
    assert status["outdated"] is True


def test_managed_install_of_an_old_release_is_outdated(monkeypatch, tmp_path):
    _managed(
        monkeypatch,
        tmp_path,
        {"published_repo": _REPO, "release_tag": "v0.9.0-unsloth.1"},
    )
    status = inference._audio_cpp_runtime_status()
    assert status["release_tag"] == "v0.9.0-unsloth.1"
    assert status["expected_tag"] == _LATEST
    assert status["outdated"] is True


def test_nothing_cached_cannot_tell(monkeypatch, tmp_path):
    _managed(
        monkeypatch,
        tmp_path,
        {"published_repo": _REPO, "release_tag": "v0.8.0-unsloth.1"},
        cached = None,
    )
    status = inference._audio_cpp_runtime_status()
    assert status["expected_tag"] is None and status["outdated"] is False


def test_the_disk_cache_answers_whatever_its_age(monkeypatch, tmp_path):
    _managed(
        monkeypatch,
        tmp_path,
        {"published_repo": _REPO, "release_tag": "v0.9.0-unsloth.1"},
        cached = None,
    )
    cache = freshness_flow.cache_path_for(_REPO, tmp_path / ".freshness")
    cache.parent.mkdir(parents = True)
    cache.write_text(json.dumps({"fetched_at": 1.0, "latest_tag": _LATEST}))
    status = inference._audio_cpp_runtime_status()
    assert status["expected_tag"] == _LATEST and status["outdated"] is True


@pytest.mark.parametrize("cached", [_LATEST, None])
def test_the_runtime_status_never_touches_the_network(monkeypatch, tmp_path, cached):
    # /audio/stt/status calls this inside an async handler.
    _managed(
        monkeypatch,
        tmp_path,
        {"published_repo": _REPO, "release_tag": "v0.9.0-unsloth.1"},
        cached = cached,
    )
    calls: list = []

    def fetch(*args, **kwargs):
        calls.append(args)
        raise AssertionError("network fetch")

    monkeypatch.setattr(freshness_flow, "_fetch_newest_published_release", fetch)
    status = inference._audio_cpp_runtime_status()
    assert status["outdated"] is (cached is not None)
    assert calls == []


def test_disabled_update_checks_are_not_outdated(monkeypatch, tmp_path):
    _managed(monkeypatch, tmp_path, {"published_repo": _REPO, "release_tag": "v0.9.0-unsloth.1"})
    monkeypatch.setenv("UNSLOTH_DISABLE_UPDATE_CHECK", "1")
    status = inference._audio_cpp_runtime_status()
    assert status["expected_tag"] is None and status["outdated"] is False


def test_a_pinned_release_is_expected_without_a_lookup(monkeypatch, tmp_path):
    # UNSLOTH_AUDIO_CPP_TAG names the release setup installs, newer or older.
    _managed(
        monkeypatch,
        tmp_path,
        {"published_repo": _REPO, "release_tag": _LATEST},
        pin = "v0.9.0-unsloth.1",
        cached = None,
    )
    status = inference._audio_cpp_runtime_status()
    assert status["expected_tag"] == "v0.9.0-unsloth.1" and status["outdated"] is True
    _managed(
        monkeypatch,
        tmp_path,
        {"published_repo": _REPO, "release_tag": "v0.9.0-unsloth.1"},
        pin = "v0.9.0-unsloth.1",
        cached = None,
    )
    assert inference._audio_cpp_runtime_status()["outdated"] is False


def test_same_tag_from_another_repo_is_outdated(monkeypatch, tmp_path):
    _managed(monkeypatch, tmp_path, {"published_repo": "someone/audio.cpp", "release_tag": _LATEST})
    assert inference._audio_cpp_runtime_status()["outdated"] is True


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

    monkeypatch.setattr(audio_cpp_update, "_target", boom)
    status = inference._audio_cpp_runtime_status()
    assert status["available"] is True and status["outdated"] is False


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
        "launch_backend": "vulkan",
        "release_tag": "v0.4.0",
        "expected_tag": None,
        "outdated": False,
    }


def test_stt_status_reads_the_runtime_off_the_event_loop(monkeypatch):
    import threading

    # A custom build's backend probe runs its --version, which must not stall other requests.
    seen = []

    def runtime_status():
        seen.append(threading.current_thread() is threading.main_thread())
        return {"available": False}

    monkeypatch.setattr(inference, "_audio_cpp_runtime_status", runtime_status)
    monkeypatch.setattr(inference.account_access, "managed_account", lambda: False)
    response = asyncio.run(inference.stt_status(model = None, current_subject = "owner"))
    assert json.loads(response.body)["audio_cpp_runtime"] == {"available": False}
    assert seen == [False]
