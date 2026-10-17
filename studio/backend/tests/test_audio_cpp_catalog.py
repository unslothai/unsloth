# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The audio.cpp catalog builder: online, offline from the saved listing, cache-only, cached."""

import json
import re
import threading
import time

import pytest

from core.inference import audio_cpp_catalog as catalog
from core.inference import audio_cpp_files, audio_cpp_server
from core.inference import audio_cpp_models as acm
from core.inference.audio_cpp_models import AUDIO_CPP_REPO, AudioCppHeader, RepoFile
from hub.utils import state_dir

DRAMA_SPEC = {
    "family": "fish_drama3",
    "display_name": "Fish Audio Drama 3",
    "description": "Drama 3 speech generation and voice cloning.",
    "status": "experimental",
    "tasks": ["tts", "clone"],
    "languages": ["auto"],
    "ui": {"recommended_package": "q4"},
    "package_defaults": {"download": {"repo": "Jmica/Drama-3-GGUF"}},
    "packages": [
        {"id": "q4", "default": True, "format": "gguf", "files": ["drama-3-Q4_K_S.gguf"]},
        {"id": "src", "format": "safetensors", "files": ["model.safetensors"]},
    ],
}
KOKORO_SPEC = {
    "family": "kokoro_tts",
    "display_name": "Kokoro 82M",
    "status": "community",
    "tasks": ["tts"],
    "languages": ["en-us", "zh"],
    "package_defaults": {"download": {"repo": AUDIO_CPP_REPO}},
    "packages": [
        {
            "id": "kokoro_q8",
            "default": True,
            "format": "gguf",
            "target_directory": "Kokoro-82M-GGUF",
            "files": ["Kokoro-82M-GGUF/kokoro-82m-q8_0.gguf"],
        }
    ],
}
NEW_TTS = AudioCppHeader("audiocpp", "new_tts", {"family": "new_tts", "tasks": ["tts", "clone"]})
UMBRELLA = [
    RepoFile("Kokoro-82M-GGUF/kokoro-82m-q8_0.gguf", 100),
    RepoFile("Kokoro-82M-GGUF/kokoro-82m-bf16.gguf", 200),
    RepoFile("Kokoro-82M-GGUF/README.md", 5),
    RepoFile("New-TTS-GGUF/new-tts-q8_0.gguf", 300),
    RepoFile("HTDemucs-GGUF/htdemucs-f16.gguf", 400),
    RepoFile("README.md", 1),
]
DRAMA = [RepoFile("drama-3-Q4_K_S.gguf", 7000), RepoFile("drama-3-Q2_K.gguf", 5000)]


@pytest.fixture
def env(tmp_path, monkeypatch):
    hub_root = tmp_path / "hub"
    hub_root.mkdir()
    monkeypatch.setattr(acm, "_hub_cache", lambda: hub_root)
    monkeypatch.setattr(audio_cpp_files, "_hub_cache", lambda: hub_root)
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "studio-cache")
    monkeypatch.setattr(acm, "runtime_knows_family", lambda family: None)
    monkeypatch.setattr(acm, "_listing_cache", {})
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    calls = {"hub": 0, "header": 0}
    umbrella = list(UMBRELLA)

    def listing(repo_id, folder, token):
        calls["hub"] += 1
        assert token is None and folder == ""
        return list(umbrella if repo_id == AUDIO_CPP_REPO else DRAMA)

    def header(
        repo_id,
        filename,
        hf_token = None,
        *,
        size = None,
    ):
        calls["header"] += 1
        assert hf_token is False
        return NEW_TTS if filename.startswith("New-TTS-GGUF/") else None

    monkeypatch.setattr(acm, "_hub_listing", listing)
    monkeypatch.setattr(acm, "read_remote_header", header)
    runtime = tmp_path / "runtime"
    (runtime / "bin").mkdir(parents = True)
    specs = runtime / "model_specs"
    specs.mkdir()
    (specs / "fish_drama3.json").write_text(json.dumps(DRAMA_SPEC))
    (specs / "kokoro_tts.json").write_text(json.dumps(KOKORO_SPEC))
    (specs / "Bad Name.json").write_text("{not json")
    binary = str(runtime / "bin" / "audiocpp_server")
    monkeypatch.setattr(audio_cpp_server, "find_audio_cpp_server_binary", lambda: binary)
    catalog.forget_audio_catalog()
    acm.forget()
    yield {"tmp": tmp_path, "hub": hub_root, "specs": specs, "calls": calls, "umbrella": umbrella}
    catalog.forget_audio_catalog()
    acm.forget()


def _rows(payload):
    return {m.id: m for m in payload.models}


def _saved_file(env):
    return env["tmp"] / "studio-cache" / state_dir._HUB_STATE_DIRNAME / "audio_cpp_catalog.json"


def _assert_safe(env, payload):
    text = json.dumps(payload.model_dump())
    assert str(env["tmp"]) not in text
    assert not re.search(r'"(/|[A-Za-z]:\\\\)[^"]*"', text)
    for m in payload.models:
        assert not re.search(r"(?<!Unsloth )Studio", m.unsupported_reason or "")
        assert (m.unsupported_reason is None) == bool(m.workflows)


def test_online_lists_umbrella_folders_spec_families_and_community_repos(env):
    payload = catalog.audio_catalog_payload()
    assert payload.source == "hub" and payload.runtime_specs
    rows = _rows(payload)
    kokoro = rows[f"{AUDIO_CPP_REPO}/Kokoro-82M-GGUF"]
    assert kokoro.workflows == ["speak"] and kokoro.display_name == "Kokoro 82M"
    assert kokoro.languages == ["en-us", "zh"] and kokoro.status == "community"
    assert kokoro.size_bytes == 100 and kokoro.source == "umbrella"
    assert rows[f"{AUDIO_CPP_REPO}/New-TTS-GGUF"].workflows == ["speak", "clone"]
    assert rows[f"{AUDIO_CPP_REPO}/HTDemucs-GGUF"].workflows == ["separate"]
    drama = rows["Jmica/Drama-3-GGUF"]
    assert drama.source == "community" and drama.folder == "" and drama.family == "fish_drama3"
    assert drama.languages is None and drama.status == "experimental"
    assert drama.size_bytes == 7000
    assert [m.source for m in payload.models].count("community") == 1
    saved = json.loads(_saved_file(env).read_text())
    assert saved["schema"] == 1
    assert saved["folders"]["New-TTS-GGUF"]["family"] == "new_tts"
    assert "README.md" not in json.dumps(saved)
    _assert_safe(env, payload)


def test_offline_reads_the_saved_listing_without_the_hub(env, monkeypatch):
    online = _rows(catalog.audio_catalog_payload())
    catalog.forget_audio_catalog()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    env["calls"].update(hub = 0, header = 0)
    payload = catalog.audio_catalog_payload()
    assert payload.source == "saved"
    assert set(_rows(payload)) == set(online)
    assert _rows(payload)[f"{AUDIO_CPP_REPO}/New-TTS-GGUF"].workflows == ["speak", "clone"]
    assert env["calls"] == {"hub": 0, "header": 0}
    _assert_safe(env, payload)


def _kokoro_on_disk(env):
    repo = env["hub"] / ("models--" + AUDIO_CPP_REPO.replace("/", "--"))
    gguf = repo / "snapshots" / ("a" * 40) / "Kokoro-82M-GGUF" / "kokoro-82m-q8_0.gguf"
    gguf.parent.mkdir(parents = True)
    gguf.write_bytes(b"x")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("a" * 40)


def test_offline_without_a_saved_listing_lists_the_cached_folders(env, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _kokoro_on_disk(env)
    payload = catalog.audio_catalog_payload()
    assert payload.source == "cache"
    umbrella = [m for m in payload.models if m.source == "umbrella"]
    assert [m.folder for m in umbrella] == ["Kokoro-82M-GGUF"]
    assert umbrella[0].downloaded and umbrella[0].workflows == ["speak"]
    assert env["calls"] == {"hub": 0, "header": 0}
    _assert_safe(env, payload)


def test_a_saved_listing_deleted_mid_run_degrades_to_the_cache(env, monkeypatch):
    catalog.audio_catalog_payload()
    catalog.forget_audio_catalog()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _kokoro_on_disk(env)
    _saved_file(env).unlink()
    assert catalog.audio_catalog_payload().source == "cache"
    catalog.forget_audio_catalog()
    _saved_file(env).write_text('{"schema": 1, "folders": {"X": {"files": "bad"}}}')
    assert catalog.audio_catalog_payload().source == "cache"


def test_the_catalog_is_cached_and_a_degraded_answer_is_retried_sooner(env, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(catalog.time, "monotonic", lambda: clock[0])
    catalog.audio_catalog_payload()
    hub_calls = env["calls"]["hub"]
    catalog.audio_catalog_payload()
    clock[0] += 299
    catalog.audio_catalog_payload()
    assert env["calls"]["hub"] == hub_calls
    catalog.forget_audio_catalog()
    acm._listing_cache.clear()
    catalog.audio_catalog_payload()
    assert env["calls"]["hub"] > hub_calls

    def down(repo_id, *_a):
        env["calls"]["hub"] += repo_id == AUDIO_CPP_REPO
        raise OSError("Hub down")

    monkeypatch.setattr(acm, "_hub_listing", down)
    catalog.forget_audio_catalog()
    acm._listing_cache.clear()
    env["calls"]["hub"] = 0
    assert catalog.audio_catalog_payload().source == "saved"
    clock[0] += 29
    catalog.audio_catalog_payload()
    assert env["calls"]["hub"] == 1
    clock[0] += 2
    catalog.audio_catalog_payload()
    assert env["calls"]["hub"] == 2


def test_without_runtime_specs_folders_resolve_by_bounded_header_probes(env, monkeypatch):
    for f in env["specs"].iterdir():
        f.unlink()
    env["specs"].rmdir()
    monkeypatch.setattr(acm, "runtime_spec", lambda family: None)
    many = [RepoFile(f"Mystery-{i:02d}-GGUF/mystery-{i:02d}-q8_0.gguf", 10) for i in range(40)]
    env["umbrella"].extend(many)
    payload = catalog.audio_catalog_payload()
    assert payload.runtime_specs is False
    assert all(m.source == "umbrella" for m in payload.models)
    assert env["calls"]["header"] <= catalog._MAX_PROBES
    rows = _rows(payload)
    assert rows[f"{AUDIO_CPP_REPO}/Kokoro-82M-GGUF"].workflows == ["speak"]
    assert rows[f"{AUDIO_CPP_REPO}/Kokoro-82M-GGUF"].display_name == "Kokoro-82M-GGUF"
    mystery = [m for m in payload.models if m.folder.startswith("Mystery-")]
    assert len(mystery) == 40 and not mystery[0].workflows
    assert mystery[0].unsupported_reason == catalog._NO_FAMILY
    _assert_safe(env, payload)


def test_a_spec_dir_read_is_capped_and_ignores_bad_files(env, monkeypatch):
    (env["specs"] / "huge.json").write_bytes(b" " * (catalog._MAX_SPEC_BYTES + 1))
    (env["specs"] / "list.json").write_text("[]")
    (env["specs"] / "evil.json").write_text(json.dumps({"family": "../x", "packages": []}))
    index = catalog._spec_index(env["specs"])
    assert set(index) == {"fish_drama3", "kokoro_tts"}


def test_a_spec_with_an_empty_gguf_package_does_not_break_the_catalog(env):
    bad = {"family": "empty_pkg", "tasks": ["tts"], "packages": [{"format": "gguf", "files": []}]}
    (env["specs"] / "empty_pkg.json").write_text(json.dumps(bad))
    payload = catalog.audio_catalog_payload()
    assert payload.source == "hub"
    assert _rows(payload)[f"{AUDIO_CPP_REPO}/Kokoro-82M-GGUF"].workflows == ["speak"]


def test_concurrent_cold_callers_share_one_build(env, monkeypatch):
    build, builds = catalog._build, []

    def slow(*args):
        builds.append(1)
        time.sleep(0.2)
        return build(*args)

    monkeypatch.setattr(catalog, "_build", slow)
    threads = [threading.Thread(target = catalog.audio_catalog_payload) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(builds) == 1


def test_a_build_with_a_late_family_probe_is_retried_sooner(env, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(catalog.time, "monotonic", lambda: clock[0])
    # Every probe misses the deadline, so New-TTS-GGUF stays unplaced in a Hub answer.
    monkeypatch.setattr(catalog, "_bounded", lambda jobs: {})
    for f in env["specs"].iterdir():
        f.unlink()
    payload = catalog.audio_catalog_payload()
    assert payload.source == "hub"
    assert not _rows(payload)[f"{AUDIO_CPP_REPO}/New-TTS-GGUF"].workflows
    hub_calls = env["calls"]["hub"]
    acm._listing_cache.clear()
    clock[0] += catalog._DEGRADED_TTL_SECONDS + 1
    catalog.audio_catalog_payload()
    assert env["calls"]["hub"] == hub_calls + 1
