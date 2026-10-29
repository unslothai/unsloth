# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The audio.cpp phase of the combined in-app update: what it offers, and how it swaps the runtime."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.inference import audio_cpp_server, model_slots
from core.inference.audio_cpp_models import AUDIO_CPP_MUSIC_AUDIO_TYPE, AUDIO_CPP_TTS_AUDIO_TYPE
from utils import audio_cpp_freshness
from utils import audio_cpp_update as aupd
from utils.prebuilt import update_flow

_REPO = "unslothai/audio.cpp"
_LATEST = "v0.9.1-unsloth.1"
# The newest release's assets: the bundle the recorded one renames to, another host's, and a Windows CUDA pair.
_ASSETS = {
    f"audio-{_LATEST}-bin-ubuntu-x64-cpu-portable.tar.gz": 56_000_000,
    f"audio-{_LATEST}-bin-ubuntu-x64-vulkan.tar.gz": 70_000_000,
    f"audio-{_LATEST}-bin-windows-x64-cuda12.8.zip": 400_000_000,
    f"audio-{_LATEST}-cudart-windows-x64-cuda12.8.zip": 300_000_000,
}
_real_unload = aupd._unload_audio_cpp_models
_SETUP_SKIPS = ("AUDIOCPP_SERVER_PATH", "UNSLOTH_AUDIO_CPP_PATH", "UNSLOTH_SKIP_AUDIO_CPP_INSTALL")


def _write_record(tree, **fields):
    record = {
        "published_repo": "unslothai/audio.cpp",
        "release_tag": "v0.9.0-unsloth.1",
        "asset": "audio-v0.9.0-unsloth.1-bin-ubuntu-x64-cpu-portable.tar.gz",
        "accelerator": "cpu",
        "accelerator_request": "cpu",
        **fields,
    }
    (tree / audio_cpp_server.INSTALL_RECORD).write_text(json.dumps(record))


@pytest.fixture
def managed(monkeypatch, tmp_path):
    """An Unsloth Studio-managed install of v0.9.0-unsloth.1 whose repo's newest release is v0.9.1-unsloth.1."""
    for name in (*_SETUP_SKIPS, "UNSLOTH_DISABLE_UPDATE_CHECK"):
        monkeypatch.delenv(name, raising = False)
    tree = tmp_path / "audio.cpp"
    tree.mkdir()
    (tree / ".unsloth-studio-owned").touch()
    binary = tree / "audiocpp_server"
    binary.write_text("stub")
    _write_record(tree)
    script = tmp_path / "install_audio_cpp_prebuilt.py"
    script.write_text("stub")
    lookups: list = []
    state = SimpleNamespace(latest = _LATEST, assets = dict(_ASSETS))

    def latest(repo, *, force_refresh = False):
        lookups.append(("latest", repo, force_refresh))
        return state.latest

    def assets(repo, *, force_refresh = False):
        lookups.append(("assets", repo, force_refresh))
        return state.assets

    monkeypatch.setattr(audio_cpp_server, "managed_audio_cpp_dir", lambda: tree)
    monkeypatch.setattr(audio_cpp_server, "find_audio_cpp_server_binary", lambda: str(binary))
    monkeypatch.setattr(aupd, "_target", lambda: (_REPO, None))
    monkeypatch.setattr(aupd, "_installer_script", lambda: script)
    monkeypatch.setattr(audio_cpp_freshness, "latest_published_release", latest)
    monkeypatch.setattr(audio_cpp_freshness, "latest_release_assets", assets)
    monkeypatch.setattr(aupd.platform, "system", lambda: "Linux")
    monkeypatch.setattr(aupd.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(aupd._installer(), "detect_accelerator", lambda: "cpu")
    return SimpleNamespace(tree = tree, binary = binary, script = script, lookups = lookups, state = state)


# --- plan ---


def test_an_old_managed_runtime_is_offered_the_newest_release(managed):
    plan = aupd.chained_phase_plan()
    assert plan["update_available"] is True and plan["skip_reason"] is None
    assert plan["status"] == {
        "installed_tag": "v0.9.0-unsloth.1",
        "latest_tag": _LATEST,
        "update_size_bytes": 56_000_000,
    }
    assert plan["phase"] == {
        "install_dir": managed.tree,
        "script": managed.script,
        "accelerator": "cpu",
        "from_tag": "v0.9.0-unsloth.1",
        "to_tag": _LATEST,
        "pin_release_tag": _LATEST,
        "repo": _REPO,
    }
    assert ("latest", _REPO, False) in managed.lookups


def test_the_size_falls_back_to_the_installers_pick(managed):
    # An asset name that does not carry the installed tag still sizes the host's bundle.
    _write_record(
        managed.tree, asset = "renamed.tar.gz", accelerator = "vulkan", accelerator_request = "auto"
    )
    assert aupd.chained_phase_plan()["status"]["update_size_bytes"] == 70_000_000


def test_assets_of_another_release_never_decide_the_offer(managed):
    # After a restart the tag comes from the 24h disk cache while the asset list is fetched fresh, and a
    # newer release may have been published in between: its assets say nothing about the offered tag.
    managed.state.assets = {"audio-v0.9.2-unsloth.1-bin-windows-x64-cuda13.zip": 1}
    plan = aupd.chained_phase_plan()
    assert plan["update_available"] is True and plan["skip_reason"] is None
    assert plan["status"]["update_size_bytes"] is None
    assert plan["phase"]["pin_release_tag"] == _LATEST
    managed.state.assets = {
        "audio-v0.9.2-unsloth.1-bin-ubuntu-x64-cpu-portable.tar.gz": 1,
        **_ASSETS,
    }
    assert aupd.chained_phase_plan()["status"]["update_size_bytes"] == 56_000_000


def test_a_tag_that_extends_the_offered_one_is_another_release(managed, monkeypatch):
    # Upstream publishes v0.8.2 and later v0.8.2-audio8-perf-hotfix: the hotfix's assets start with
    # "audio-v0.8.2-" too, yet say nothing about v0.8.2.
    monkeypatch.setattr(aupd, "_target", lambda: ("0xShug0/audio.cpp", None))
    _write_record(
        managed.tree,
        published_repo = "0xShug0/audio.cpp",
        release_tag = "v0.8.1",
        asset = "audio-v0.8.1-bin-ubuntu-x64-cpu-portable.tar.gz",
    )
    managed.state.latest = "v0.8.2"
    managed.state.assets = {"audio-v0.8.2-audio8-perf-hotfix-bin-windows-x64-cpu.zip": 1}
    plan = aupd.chained_phase_plan()
    assert plan["update_available"] is True and plan["status"]["update_size_bytes"] is None
    managed.state.assets["audio-v0.8.2-bin-ubuntu-x64-cpu-portable.tar.gz"] = 5
    assert aupd.chained_phase_plan()["status"]["update_size_bytes"] == 5


def test_the_size_adds_the_cuda_runtime_zip(managed, monkeypatch):
    monkeypatch.setattr(aupd.platform, "system", lambda: "Windows")
    monkeypatch.setattr(aupd.platform, "machine", lambda: "AMD64")
    _write_record(
        managed.tree,
        asset = "audio-v0.9.0-unsloth.1-bin-windows-x64-cuda12.8.zip",
        cudart_asset = "audio-v0.9.0-unsloth.1-cudart-windows-x64-cuda12.8.zip",
        accelerator = "cuda",
        accelerator_request = "cuda",
    )
    assert aupd.chained_phase_plan()["status"]["update_size_bytes"] == 700_000_000
    # A release without the runtime zip: the installer downloads the bundle alone.
    del managed.state.assets[f"audio-{_LATEST}-cudart-windows-x64-cuda12.8.zip"]
    plan = aupd.chained_phase_plan()
    assert plan["status"]["update_size_bytes"] == 400_000_000 and plan["update_available"] is True


@pytest.mark.parametrize("arch", ["arm64", "x64"])
@pytest.mark.parametrize("reverse", [False, True])
def test_update_size_uses_the_matching_architecture_runtime_in_a_mixed_release(
    managed, monkeypatch, arch, reverse
):
    monkeypatch.setattr(aupd.platform, "system", lambda: "Windows")
    monkeypatch.setattr(aupd.platform, "machine", lambda: arch)
    monkeypatch.setattr(aupd._installer(), "torch_provides_cuda_runtime", lambda _major: False)
    _write_record(
        managed.tree,
        asset = f"audio-v0.9.0-unsloth.1-bin-windows-{arch}-cuda13.zip",
        accelerator = "cuda",
        accelerator_request = "cuda",
    )
    assets = [
        (f"audio-{_LATEST}-bin-windows-{arch}-cuda13.zip", 100_000_000),
        (f"audio-{_LATEST}-cudart-windows-x64-cuda13.zip", 300_000_000),
        (f"audio-{_LATEST}-cudart-windows-arm64-cuda13.zip", 200_000_000),
    ]
    managed.state.assets = dict(reversed(assets) if reverse else assets)
    size = aupd.chained_phase_plan()["status"]["update_size_bytes"]
    assert size == (300_000_000 if arch == "arm64" else 400_000_000)


@pytest.mark.parametrize("torch_has_runtime", [False, True])
def test_the_cuda_runtime_zip_follows_the_current_torch(managed, monkeypatch, torch_has_runtime):
    monkeypatch.setattr(aupd.platform, "system", lambda: "Windows")
    monkeypatch.setattr(aupd.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(
        aupd._installer(), "torch_provides_cuda_runtime", lambda major: torch_has_runtime
    )
    # The record says nothing about the runtime zip: what the installer does now decides the size.
    _write_record(
        managed.tree,
        asset = "audio-v0.9.0-unsloth.1-bin-windows-x64-cuda12.8.zip",
        accelerator = "cuda",
        accelerator_request = "cuda",
    )
    size = aupd.chained_phase_plan()["status"]["update_size_bytes"]
    assert size == (400_000_000 if torch_has_runtime else 700_000_000)


def test_another_repos_cuda_bundle_is_sized_like_the_installer_picks(managed, monkeypatch):
    monkeypatch.setattr(aupd.platform, "system", lambda: "Windows")
    monkeypatch.setattr(aupd.platform, "machine", lambda: "AMD64")
    seen = {}
    installer = aupd._installer()
    real = installer.resolve_release_asset
    monkeypatch.setattr(installer, "nvidia_driver_cuda_version", lambda: (12, 8))
    monkeypatch.setattr(installer, "torch_cuda_major", lambda: 12)

    def resolve(names, **kwargs):
        seen.update(kwargs)
        return real(names, **kwargs)

    monkeypatch.setattr(installer, "resolve_release_asset", resolve)
    _write_record(
        managed.tree,
        published_repo = "0xShug0/audio.cpp",
        release_tag = "v0.9.0",
        asset = "audio-v0.9.0-bin-windows-x64-cuda12.4.zip",
        accelerator = "cuda",
        accelerator_request = "cuda",
    )
    aupd.chained_phase_plan()
    assert seen["driver_cuda"] == (12, 8) and seen["prefer_cuda_major"] == 12


def test_unknown_assets_size_nothing_but_still_offer(managed):
    managed.state.assets = None
    plan = aupd.chained_phase_plan()
    assert plan["update_available"] is True
    assert plan["status"]["update_size_bytes"] is None


def test_an_automatic_install_keeps_detecting_the_host(managed):
    _write_record(managed.tree, accelerator = "cuda", accelerator_request = "auto")
    assert aupd.chained_phase_plan()["phase"]["accelerator"] == "auto"


@pytest.mark.parametrize("name", _SETUP_SKIPS)
def test_a_runtime_setup_skips_is_unmanaged(managed, monkeypatch, name):
    monkeypatch.setenv(name, "1" if name == "UNSLOTH_SKIP_AUDIO_CPP_INSTALL" else "/elsewhere")
    plan = aupd.chained_phase_plan()
    assert plan["update_available"] is False and plan["skip_reason"] == "unmanaged"
    assert managed.lookups == []


def test_a_tree_without_the_ownership_marker_is_unmanaged(managed):
    (managed.tree / ".unsloth-studio-owned").unlink()
    assert aupd.chained_phase_plan()["skip_reason"] == "unmanaged"


def test_a_binary_outside_the_managed_tree_is_unmanaged(managed, monkeypatch, tmp_path):
    custom = tmp_path / "custom" / "audiocpp_server"
    custom.parent.mkdir()
    custom.write_text("stub")
    monkeypatch.setattr(audio_cpp_server, "find_audio_cpp_server_binary", lambda: str(custom))
    assert aupd.chained_phase_plan()["skip_reason"] == "unmanaged"


def test_missing_runtime_offered_as_install(managed, monkeypatch):
    monkeypatch.setattr(audio_cpp_server, "find_audio_cpp_server_binary", lambda: None)
    plan = aupd.chained_phase_plan()
    assert plan["update_available"] is True and plan["skip_reason"] is None
    assert plan["status"] == {
        "installed_tag": None,
        "latest_tag": _LATEST,
        "update_size_bytes": None,
    }
    tags = {"from_tag": None, "to_tag": _LATEST, "pin_release_tag": _LATEST, "repo": _REPO}
    assert plan["phase"] == {
        "install_dir": managed.tree,
        "script": managed.script,
        "accelerator": None,
        **tags,
    }
    monkeypatch.setattr(aupd.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(aupd._installer(), "detect_accelerator", lambda: "metal")
    managed.state.assets = {f"audio-{_LATEST}-bin-macos-x64-metal.tar.gz": 1}
    assert aupd.chained_phase_plan()["update_available"] is True


@pytest.mark.parametrize(
    "case",
    [*_SETUP_SKIPS, "UNSLOTH_DISABLE_UPDATE_CHECK", "no_latest", "no_prebuilt", "no_installer"],
)
def test_missing_runtime_stays_not_installed_when_setup_would_skip(managed, monkeypatch, case):
    monkeypatch.setattr(audio_cpp_server, "find_audio_cpp_server_binary", lambda: None)
    if case == "no_latest":
        managed.state.latest = None
    elif case == "no_prebuilt":
        managed.state.assets = {f"audio-{_LATEST}-bin-windows-x64-cuda12.8.zip": 1}
    elif case == "no_installer":
        monkeypatch.setattr(aupd, "_installer_script", lambda: None)
    else:
        monkeypatch.setenv(case, "1")
    plan = aupd.chained_phase_plan()
    assert plan == {
        "status": None,
        "update_available": False,
        "skip_reason": "not_installed",
        "phase": None,
    }


@pytest.mark.parametrize("tag", [_LATEST, "v0.9.2-unsloth.1", "v1.0.0"])
def test_the_newest_or_a_newer_release_is_up_to_date(managed, tag):
    _write_record(managed.tree, release_tag = tag)
    plan = aupd.chained_phase_plan()
    assert plan["update_available"] is False and plan["skip_reason"] == "up_to_date"
    assert plan["status"]["latest_tag"] == _LATEST


def test_a_hotfix_published_last_is_not_offered_as_a_downgrade(managed):
    _write_record(managed.tree, release_tag = _LATEST)
    managed.state.latest = "v0.8.2-audio8-perf-hotfix-unsloth.1"
    plan = aupd.chained_phase_plan()
    assert plan["update_available"] is False and plan["skip_reason"] == "up_to_date"


def test_an_unknown_latest_offers_nothing(managed):
    managed.state.latest = None
    plan = aupd.chained_phase_plan()
    assert plan["update_available"] is False and plan["skip_reason"] == "up_to_date"
    assert plan["phase"] is None


def test_a_record_without_a_tag_is_up_to_date(managed):
    _write_record(managed.tree, release_tag = None)
    assert aupd.chained_phase_plan()["skip_reason"] == "up_to_date"


def test_an_upstream_install_is_offered_the_forks_newest(managed):
    # The removed 0xShug0 fallback rung could leave such a record; the configured repo replaces it.
    _write_record(
        managed.tree,
        published_repo = "0xShug0/audio.cpp",
        release_tag = "v0.9.1",
        asset = "audio-v0.9.1-bin-ubuntu-x64-cpu-portable.tar.gz",
    )
    plan = aupd.chained_phase_plan()
    assert plan["update_available"] is True
    assert plan["phase"]["pin_release_tag"] == _LATEST
    assert plan["phase"]["repo"] == _REPO
    assert plan["status"]["update_size_bytes"] == 56_000_000


def test_an_upstream_install_is_left_alone_while_latest_is_unknown(managed):
    _write_record(managed.tree, published_repo = "0xShug0/audio.cpp", release_tag = "v0.9.0")
    managed.state.latest = None
    assert aupd.chained_phase_plan()["skip_reason"] == "up_to_date"


def test_force_refresh_reaches_the_release_lookup(managed):
    aupd.chained_phase_plan(force_refresh = True)
    assert ("latest", _REPO, True) in managed.lookups
    assert ("assets", _REPO, True) in managed.lookups


def test_a_pinned_tag_is_compared_with_no_lookup(managed, monkeypatch):
    monkeypatch.setattr(aupd, "_target", lambda: (_REPO, "v0.9.0-unsloth.1"))
    plan = aupd.chained_phase_plan()
    assert plan["skip_reason"] == "up_to_date"
    assert plan["status"]["latest_tag"] == "v0.9.0-unsloth.1"
    monkeypatch.setattr(aupd, "_target", lambda: (_REPO, "v0.8.0-unsloth.1"))
    plan = aupd.chained_phase_plan()
    assert plan["update_available"] is True
    assert plan["phase"]["pin_release_tag"] == "v0.8.0-unsloth.1"
    assert plan["status"]["update_size_bytes"] is None
    assert managed.lookups == []


def test_disabled_update_checks_skip_the_offer_without_a_lookup(managed, monkeypatch):
    # Without this skip the union would show a card that start_update refuses.
    monkeypatch.setenv("UNSLOTH_DISABLE_UPDATE_CHECK", "1")
    assert aupd.chained_phase_plan()["skip_reason"] == "update_checks_disabled"
    assert managed.lookups == []
    monkeypatch.setattr(aupd, "_target", lambda: (_REPO, "v0.8.0-unsloth.1"))
    assert aupd.chained_phase_plan()["skip_reason"] == "update_checks_disabled"
    assert managed.lookups == []


def test_a_host_the_newest_release_does_not_cover_has_no_prebuilt(managed, monkeypatch):
    managed.state.assets = {f"audio-{_LATEST}-bin-windows-x64-cpu.zip": 1}
    assert aupd.chained_phase_plan()["skip_reason"] == "no_prebuilt"
    # The same release covers a Windows host.
    monkeypatch.setattr(aupd.platform, "system", lambda: "Windows")
    assert aupd.chained_phase_plan()["update_available"] is True


@pytest.mark.parametrize("request_, offered", [("vulkan", False), ("auto", True)])
def test_an_explicit_accelerator_is_not_offered_the_cpu_bundle(
    managed, monkeypatch, request_, offered
):
    # The installer never downgrades an explicit request, so a CPU-only release cannot satisfy it.
    managed.state.assets = {f"audio-{_LATEST}-bin-ubuntu-arm64-cpu.tar.gz": 1}
    monkeypatch.setattr(aupd.platform, "machine", lambda: "aarch64")
    _write_record(managed.tree, accelerator = "vulkan", accelerator_request = request_)
    plan = aupd.chained_phase_plan()
    assert plan["update_available"] is offered
    assert plan["skip_reason"] == (None if offered else "no_prebuilt")


def test_a_missing_installer_skips_the_offer(managed, monkeypatch):
    monkeypatch.setattr(aupd, "_installer_script", lambda: None)
    assert aupd.chained_phase_plan()["skip_reason"] == "installer_missing"


def test_a_probe_that_raises_is_a_skip(managed, monkeypatch):
    def boom():
        raise OSError("unreadable")

    monkeypatch.setattr(aupd, "_target", boom)
    plan = aupd.chained_phase_plan(force_refresh = True)
    assert plan["update_available"] is False and plan["skip_reason"] == "unavailable"


@pytest.mark.parametrize(
    "env, target",
    [
        ({}, (_REPO, None)),
        ({"UNSLOTH_AUDIO_CPP_TAG": ""}, (_REPO, None)),
        ({"UNSLOTH_AUDIO_CPP_TAG": "Latest"}, (_REPO, None)),
        ({"UNSLOTH_AUDIO_CPP_TAG": "v0.9.0-unsloth.1"}, (_REPO, "v0.9.0-unsloth.1")),
        ({"UNSLOTH_AUDIO_CPP_REPO": "someone/audio.cpp"}, ("someone/audio.cpp", None)),
    ],
)
def test_the_target_comes_from_the_installer_env(monkeypatch, env, target):
    monkeypatch.delenv("UNSLOTH_AUDIO_CPP_REPO", raising = False)
    monkeypatch.delenv("UNSLOTH_AUDIO_CPP_TAG", raising = False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert aupd._target() == target


# --- run ---


class _Sidecar:
    def __init__(self, events):
        self.events = events

    @contextmanager
    def update_maintenance(self):
        self.events.append("sidecar:enter")
        try:
            yield False
        finally:
            self.events.append("sidecar:exit")


@pytest.fixture
def run_env(managed, monkeypatch):
    """The run path with the sidecar, slot unload and installer recorded."""
    from core.inference import stt_audiocpp_sidecar

    events: list = []
    calls: dict = {}
    monkeypatch.setattr(stt_audiocpp_sidecar, "get_audio_cpp_stt_sidecar", lambda: _Sidecar(events))
    monkeypatch.setattr(aupd, "_unload_audio_cpp_models", lambda: events.append("unload") or False)

    def _installer(
        new_tag = "v0.9.1-unsloth.1",
        error = None,
        published_repo = _REPO,
    ):
        def _stream(cmd, env, *, set_progress, timeout_seconds):
            events.append("install")
            calls.update(cmd = cmd, env = env, flag = audio_cpp_server.UPDATE_IN_PROGRESS.is_set())
            if error is not None:
                raise error
            set_progress(0.5)
            _write_record(managed.tree, release_tag = new_tag, published_repo = published_repo)

        monkeypatch.setattr(update_flow, "stream_installer", _stream)

    _installer()
    phase = aupd.chained_phase_plan()["phase"]
    return SimpleNamespace(events = events, calls = calls, phase = phase, installer = _installer)


def test_the_runtime_is_swapped_while_no_audio_cpp_server_can_start(run_env):
    result = aupd.run_chained_phase(run_env.phase, lambda fraction: None)
    assert run_env.events == ["sidecar:enter", "unload", "install", "sidecar:exit"]
    assert run_env.calls["flag"] is True
    assert not audio_cpp_server.UPDATE_IN_PROGRESS.is_set()
    assert result == {
        "to_tag": "v0.9.1-unsloth.1",
        "reload_required": False,
        "message": "Updated audio.cpp to v0.9.1-unsloth.1.",
    }


def test_a_runtime_swap_drops_resolutions_made_against_the_old_tree(run_env, monkeypatch):
    from core.inference import audio_cpp_models

    forgets = []
    monkeypatch.setattr(
        audio_cpp_models,
        "forget",
        lambda *a: forgets.append((a, audio_cpp_server.UPDATE_IN_PROGRESS.is_set())),
    )
    aupd.run_chained_phase(run_env.phase, lambda fraction: None)
    # Before loads are allowed again, so none reuses an old-runtime resolution.
    assert forgets == [((), True)]
    run_env.installer(error = RuntimeError("installer timed out after 1800s"))
    with pytest.raises(RuntimeError, match = "timed out"):
        aupd.run_chained_phase(run_env.phase, lambda fraction: None)
    assert len(forgets) == 2 and not audio_cpp_server.UPDATE_IN_PROGRESS.is_set()


def test_the_installer_gets_the_managed_dir_and_recorded_accelerator(run_env, managed):
    aupd.run_chained_phase(run_env.phase, lambda fraction: None)
    cmd = run_env.calls["cmd"]
    assert cmd[1:] == [
        str(managed.script),
        "--install-dir",
        str(managed.tree),
        "--accelerator",
        "cpu",
        "--published-release-tag",
        _LATEST,
        "--keep-newer",
    ]
    assert run_env.calls["env"]["UNSLOTH_PROGRESS_PERCENT_STEP"] == "5"


@pytest.mark.parametrize("kept", [False, True])
def test_the_release_cache_is_refreshed_after_the_install(run_env, managed, monkeypatch, kept):
    events = run_env.events
    monkeypatch.setattr(
        audio_cpp_freshness,
        "reset_caches",
        lambda *, drop_disk = False: events.append(("reset", drop_disk)),
    )
    if kept:
        run_env.installer(new_tag = "v0.9.0-unsloth.1")
    managed.lookups.clear()
    aupd.run_chained_phase(run_env.phase, lambda fraction: None)
    assert events[events.index("install") + 1] == ("reset", True)
    assert managed.lookups == [("latest", _REPO, True)]


def test_a_failed_release_refresh_does_not_fail_the_install(run_env, monkeypatch):
    def offline(repo, *, force_refresh = False):
        raise OSError("offline")

    monkeypatch.setattr(audio_cpp_freshness, "latest_published_release", offline)
    result = aupd.run_chained_phase(run_env.phase, lambda fraction: None)
    assert result["to_tag"] == _LATEST


def test_the_install_drops_the_on_disk_release_cache(run_env, managed, monkeypatch, tmp_path):
    cache = tmp_path / "audio_cpp_freshness"
    monkeypatch.setattr(audio_cpp_freshness, "_cache_dir", lambda: cache)
    cache.mkdir()
    (cache / "unslothai__audio.cpp.json").write_text(
        json.dumps({"fetched_at": 1.0, "latest_tag": "v0.9.0-unsloth.1"})
    )
    aupd.run_chained_phase(run_env.phase, lambda fraction: None)
    assert not cache.exists()
    assert audio_cpp_freshness.cached_latest_release(_REPO) is None


def test_a_newer_runtime_installed_since_the_plan_is_not_downgraded(run_env, managed):
    _write_record(managed.tree, release_tag = "v0.9.2-unsloth.1")
    result = aupd.run_chained_phase(run_env.phase, lambda fraction: None)
    assert "install" not in run_env.events
    assert result["to_tag"] == "v0.9.2-unsloth.1"
    assert result["message"] == "audio.cpp is already at v0.9.2-unsloth.1."


def test_a_user_pin_still_installs_an_older_release(run_env, managed, monkeypatch):
    monkeypatch.setattr(aupd, "_target", lambda: (_REPO, "v0.9.0-unsloth.1"))
    _write_record(managed.tree, release_tag = "v0.9.2-unsloth.1")
    run_env.installer(new_tag = "v0.9.0-unsloth.1")
    phase = dict(run_env.phase, to_tag = "v0.9.0-unsloth.1", pin_release_tag = "v0.9.0-unsloth.1")
    result = aupd.run_chained_phase(phase, lambda fraction: None)
    assert "install" in run_env.events
    assert result["to_tag"] == "v0.9.0-unsloth.1"
    assert "--keep-newer" not in run_env.calls["cmd"]


def test_an_unloaded_model_asks_for_a_reload(run_env, monkeypatch):
    monkeypatch.setattr(aupd, "_unload_audio_cpp_models", lambda: True)
    result = aupd.run_chained_phase(run_env.phase, lambda fraction: None)
    assert result["reload_required"] is True
    assert (
        result["message"] == "Updated audio.cpp to v0.9.1-unsloth.1. Reload your model to use it."
    )


def test_the_update_flag_clears_when_the_install_fails(run_env):
    run_env.installer(error = RuntimeError("installer timed out after 1800s"))
    with pytest.raises(RuntimeError, match = "timed out"):
        aupd.run_chained_phase(run_env.phase, lambda fraction: None)
    assert not audio_cpp_server.UPDATE_IN_PROGRESS.is_set()
    assert run_env.events[-1] == "sidecar:exit"


def test_new_loads_are_refused_while_the_runtime_is_replaced():
    audio_cpp_server.UPDATE_IN_PROGRESS.set()
    try:
        problem = audio_cpp_server.model_runtime_problem(SimpleNamespace(unsupported = None))
    finally:
        audio_cpp_server.UPDATE_IN_PROGRESS.clear()
    assert problem == "The audio runtime is being updated. Try again in a moment."


def test_a_load_registering_after_the_unload_scan_is_refused():
    # Preflight can pass just before the flag is set; the registration is what the scan races.
    from core.inference.orchestrator import InferenceOrchestrator

    orchestrator = InferenceOrchestrator.__new__(InferenceOrchestrator)
    orchestrator.loading_models = set()
    orchestrator._managed_engine = None
    config = SimpleNamespace(identifier = "kokoro", audio_cpp = object())
    audio_cpp_server.UPDATE_IN_PROGRESS.set()
    try:
        with pytest.raises(RuntimeError, match = "being updated"):
            orchestrator.load_model(config)
    finally:
        audio_cpp_server.UPDATE_IN_PROGRESS.clear()
    assert orchestrator.loading_models == set()


def test_a_kept_tree_is_not_reported_as_updated(run_env):
    # Exit 0 also means the release lookup could not answer and the intact tree was kept.
    run_env.installer(new_tag = "v0.9.0-unsloth.1")
    result = aupd.run_chained_phase(run_env.phase, lambda fraction: None)
    assert result["message"] == (
        "audio.cpp could not be updated right now, so the existing v0.9.0-unsloth.1 "
        "install was kept. Try again later."
    )


def test_a_same_tag_from_another_repo_is_reported_as_updated(managed, run_env, monkeypatch):
    # UNSLOTH_AUDIO_CPP_REPO switched to a repo whose newest release has the installed tag.
    _write_record(
        managed.tree,
        published_repo = "0xShug0/audio.cpp",
        release_tag = _LATEST,
        asset = f"audio-{_LATEST}-bin-ubuntu-x64-cpu-portable.tar.gz",
    )
    monkeypatch.setattr(aupd, "_target", lambda: ("someone/audio.cpp", None))
    phase = aupd.chained_phase_plan()["phase"]
    run_env.installer(new_tag = _LATEST, published_repo = "someone/audio.cpp")
    result = aupd.run_chained_phase(phase, lambda fraction: None)
    assert result["message"] == f"Updated audio.cpp to {_LATEST}."


def test_a_runtime_already_at_the_offered_release_is_not_reported_as_kept(managed, run_env):
    # A setup run installed the offered release between the plan and the in-app install.
    _write_record(managed.tree, release_tag = _LATEST)
    run_env.installer(new_tag = _LATEST)
    result = aupd.run_chained_phase(run_env.phase, lambda fraction: None)
    assert result["message"] == f"Updated audio.cpp to {_LATEST}."


def test_a_kept_tree_still_asks_for_a_reload_after_an_unload(run_env, monkeypatch):
    monkeypatch.setattr(aupd, "_unload_audio_cpp_models", lambda: True)
    run_env.installer(new_tag = "v0.9.0-unsloth.1")
    result = aupd.run_chained_phase(run_env.phase, lambda fraction: None)
    assert result["reload_required"] is True
    assert result["message"].endswith("Try again later. Reload your model to use it.")


def test_a_busy_install_says_the_runtime_is_in_use(run_env):
    run_env.installer(error = update_flow.InstallerExit(3, "installer exited 3: error: busy"))
    with pytest.raises(RuntimeError, match = "in use by another Unsloth Studio or setup run"):
        aupd.run_chained_phase(run_env.phase, lambda fraction: None)


def test_a_checksum_failure_reports_the_installer_verdict(run_env):
    tail = (
        "installer exited 1: audio.cpp: downloading audio-x.tar.gz\n"
        " 50.0% (1 MiB/2 MiB)\n"
        "error: audio-x.tar.gz: sha256 mismatch (expected aa, got bb)\n"
    )
    run_env.installer(error = update_flow.InstallerExit(1, tail))
    with pytest.raises(RuntimeError) as info:
        aupd.run_chained_phase(run_env.phase, lambda fraction: None)
    assert str(info.value) == "audio-x.tar.gz: sha256 mismatch (expected aa, got bb)"


def test_a_rate_limit_gets_token_advice(run_env, monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising = False)
    monkeypatch.delenv("GITHUB_TOKEN", raising = False)
    tail = "installer exited 1: error: GitHub API is rate limiting release lookups\n"
    run_env.installer(error = update_flow.InstallerExit(1, tail))
    with pytest.raises(RuntimeError, match = "Set GH_TOKEN or GITHUB_TOKEN"):
        aupd.run_chained_phase(run_env.phase, lambda fraction: None)


def test_an_unreachable_release_fails_the_update_and_says_the_install_was_kept(run_env):
    tail = (
        "installer exited 1: audio.cpp: unslothai/audio.cpp release lookup failed (network is unreachable)\n"
        "error: no audio.cpp release lookup answered (unslothai/audio.cpp v0.9.1-unsloth.1: network is unreachable)\n"
    )
    run_env.installer(error = update_flow.InstallerExit(1, tail))
    with pytest.raises(RuntimeError) as info:
        aupd.run_chained_phase(run_env.phase, lambda fraction: None)
    assert str(info.value) == (
        "Could not reach the audio.cpp release on GitHub, so the existing install was kept. Try again later."
    )


def test_no_sidecar_coordination_fails_closed(run_env, monkeypatch):
    from core.inference import stt_audiocpp_sidecar

    def boom():
        raise ImportError("sidecar unavailable")

    monkeypatch.setattr(stt_audiocpp_sidecar, "get_audio_cpp_stt_sidecar", boom)
    with pytest.raises(RuntimeError, match = "dictation sidecar"):
        aupd.run_chained_phase(run_env.phase, lambda fraction: None)
    assert "install" not in run_env.events
    assert not audio_cpp_server.UPDATE_IN_PROGRESS.is_set()


# --- unload ---


class _Orchestrator:
    def __init__(
        self,
        models = None,
        loading = (),
    ):
        self.models = dict(models or {})
        self.loading_models = set(loading)
        self.unloaded: list = []

    def unload_model(self, name):
        self.unloaded.append(name)
        self.models.pop(name, None)
        self.loading_models.discard(name)
        return True

    def cancel_load(self, name):
        self.loading_models.discard(name)
        return True


@pytest.fixture
def slots(monkeypatch):
    from routes import inference

    dropped: list = []
    state = SimpleNamespace(main = None, dropped = dropped)
    monkeypatch.setattr(inference, "_peek_inference_backend", lambda: state.main)
    monkeypatch.setattr(model_slots, "slots", [])
    monkeypatch.setattr(model_slots, "stuck", [])
    monkeypatch.setattr(model_slots, "drop", lambda slot: dropped.append(slot))
    return state


def test_a_main_slot_audio_cpp_model_is_unloaded(slots):
    slots.main = _Orchestrator(
        {"kokoro": {"audio_type": AUDIO_CPP_TTS_AUDIO_TYPE, "is_audio": True}}
    )
    assert aupd._unload_audio_cpp_models() is True
    assert slots.main.unloaded == ["kokoro"]


def test_a_failed_unload_stops_the_update_before_the_install(run_env, slots, monkeypatch):
    # The run_env fixture stubs the unload; put the real one back to drive the orchestrator.
    monkeypatch.setattr(aupd, "_unload_audio_cpp_models", _real_unload)
    slots.main = _Orchestrator({"kokoro": {"audio_type": AUDIO_CPP_TTS_AUDIO_TYPE}})
    slots.main.unload_model = lambda name: False
    with pytest.raises(RuntimeError, match = "kokoro did not unload") as info:
        aupd.run_chained_phase(run_env.phase, lambda fraction: None)
    assert info.value.reload_required is True
    assert "install" not in run_env.events


def test_a_main_slot_chat_model_is_left_alone(slots):
    slots.main = _Orchestrator({"qwen": {"audio_type": None}})
    assert aupd._unload_audio_cpp_models() is False
    assert slots.main.unloaded == []


def test_an_audio_cpp_load_in_flight_is_cancelled(slots, monkeypatch):
    from core.inference import audio_cpp_models

    monkeypatch.setattr(audio_cpp_models, "looks_like_audio_cpp", lambda name: name == "kokoro")
    slots.main = _Orchestrator(loading = {"kokoro"})
    assert aupd._unload_audio_cpp_models() is True
    assert slots.main.unloaded == ["kokoro"]


def test_a_load_that_publishes_during_the_scan_is_still_found(monkeypatch):
    from core.inference import audio_cpp_models

    monkeypatch.setattr(audio_cpp_models, "looks_like_audio_cpp", lambda name: name == "kokoro")

    class _Publishing:
        loading_models = {"kokoro"}

        @property
        def models(self):
            # The load finishes right after this read: it leaves loading_models for models.
            self.loading_models = set()
            return {}

    assert model_slots.audio_cpp_model_names(_Publishing()) == ["kokoro"]


def test_a_kept_slot_load_is_cancelled_before_its_slot_is_dropped(slots, monkeypatch):
    from core.inference import audio_cpp_models

    monkeypatch.setattr(audio_cpp_models, "looks_like_audio_cpp", lambda name: name == "kokoro")
    chat = SimpleNamespace(orchestrator = _Orchestrator({"qwen": {"audio_type": None}}))
    loading = SimpleNamespace(orchestrator = _Orchestrator(loading = {"kokoro"}))
    model_slots.slots[:] = [chat, loading]
    assert aupd._unload_audio_cpp_models() is True
    # A marker left behind would let the load spawn its worker after the drop.
    assert loading.orchestrator.loading_models == set()
    assert slots.dropped == [loading]


def test_a_kept_audio_cpp_slot_is_dropped(slots):
    music = SimpleNamespace(
        orchestrator = _Orchestrator({"ace": {"audio_type": AUDIO_CPP_MUSIC_AUDIO_TYPE}})
    )
    chat = SimpleNamespace(orchestrator = _Orchestrator({"qwen": {"audio_type": None}}))
    model_slots.slots[:] = [music, chat]
    assert aupd._unload_audio_cpp_models() is True
    assert slots.dropped == [music]


# --- installed version (Settings > About) ---


def test_the_installed_version_is_the_records_release_tag(managed):
    assert aupd.get_installed_audio_cpp_version() == "v0.9.0-unsloth.1"


def test_no_runtime_has_no_installed_version(managed, monkeypatch):
    monkeypatch.setattr(audio_cpp_server, "find_audio_cpp_server_binary", lambda: None)
    assert aupd.get_installed_audio_cpp_version() is None


def test_a_build_without_a_record_has_no_installed_version(managed):
    (managed.tree / audio_cpp_server.INSTALL_RECORD).unlink()
    assert aupd.get_installed_audio_cpp_version() is None


def test_a_probe_that_raises_has_no_installed_version(managed, monkeypatch):
    def unreadable():
        raise OSError("unreadable")

    monkeypatch.setattr(audio_cpp_server, "find_audio_cpp_server_binary", unreadable)
    assert aupd.get_installed_audio_cpp_version() is None


def test_the_hardware_details_report_the_installed_version():
    main = (Path(aupd.__file__).resolve().parents[1] / "main.py").read_text(encoding = "utf-8")
    assert 'body["audio_cpp"] = get_installed_audio_cpp_version()' in main


def test_a_first_install_runs_and_the_runtime_is_found_after(run_env, managed, monkeypatch):
    record = managed.tree / audio_cpp_server.INSTALL_RECORD
    record.unlink()
    monkeypatch.setattr(
        audio_cpp_server,
        "find_audio_cpp_server_binary",
        lambda: str(managed.binary) if record.is_file() else None,
    )
    phase = aupd.chained_phase_plan()["phase"]
    monkeypatch.setattr(update_flow, "stream_installer", lambda *a, **k: None)
    with pytest.raises(aupd._AudioPhaseError, match = "could not be installed"):
        aupd.run_chained_phase(phase, lambda fraction: None)
    run_env.installer(new_tag = _LATEST)
    result = aupd.run_chained_phase(phase, lambda fraction: None)
    assert result["message"] == f"Installed the audio runtime {_LATEST}."
    assert "--accelerator" not in run_env.calls["cmd"]
    assert aupd.chained_phase_plan()["skip_reason"] == "up_to_date"
    assert aupd.get_installed_audio_cpp_version() == _LATEST
