# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The old-runtime download gate, the runtime's model_specs/ folder, and its backfill."""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.inference import audio_cpp_backend, audio_cpp_files, audio_cpp_server
from core.inference import audio_cpp_models as acm
from core.inference.audio_cpp_models import AudioCppModel, AudioCppVariant, RepoFile
from utils import audio_cpp_update

COMMIT = "0097d365" + "a" * 32
REPO = "unslothai/audio.cpp"


def _model(
    folder = "Canary-180M-Flash-GGUF",
    family = "canary_asr",
    task = "asr",
) -> AudioCppModel:
    main = RepoFile(f"{folder}/{folder.lower()}-q8_0.gguf", 4)
    variant = AudioCppVariant("Q8_0", (main,), main.path)
    return AudioCppModel(
        id = f"audio-cpp/audio.cpp-gguf/{folder}",
        repo_id = "audio-cpp/audio.cpp-gguf",
        folder = folder,
        display_name = folder,
        family = family,
        task = task,
        server_task = task,
        variant = variant,
        variants = (variant,),
        default_variant = "Q8_0",
    )


def _runtime(
    monkeypatch,
    root: Path,
    *,
    record = None,
    specs = None,
    bin_dir = "",
) -> Path:
    """An audiocpp_server under ``root`` with an install record (when given) and a model_specs/ folder beside it."""
    server_dir = root / bin_dir if bin_dir else root
    server_dir.mkdir(parents = True, exist_ok = True)
    binary = server_dir / audio_cpp_server.BINARY_NAME
    binary.write_bytes(b"\x7fELF")
    if record is not None:
        (root / audio_cpp_server.INSTALL_RECORD).write_text(json.dumps(record))
    if specs is not None:
        (server_dir / "model_specs").mkdir()
        for name in specs:
            (server_dir / "model_specs" / f"{name}.json").write_text(json.dumps({"family": name}))
    monkeypatch.setattr(audio_cpp_server, "find_audio_cpp_server_binary", lambda: str(binary))
    return binary


def _missing(monkeypatch, result):
    def fake(model, *, hub_cache = None):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(audio_cpp_files, "missing_files", fake)


MESSAGE = (
    "Canary-180M-Flash-GGUF needs a newer audio runtime before its files can be downloaded. "
    "Update the audio runtime from the update banner, or run `unsloth studio update`."
)


@pytest.mark.parametrize(
    "tag, missing, expected",
    [
        ("v0.9.1-unsloth.1", [("x.gguf", 4)], MESSAGE),
        ("v0.9.0-unsloth.1", [("x.gguf", 4)], MESSAGE),
        # A complete cache still loads on an old runtime.
        ("v0.9.1-unsloth.1", [], None),
        # A cached GGUF whose companion asset is missing is repaired as before.
        ("v0.9.1-unsloth.1", [("embeddings.safetensors", 4), ("config.json", 1)], None),
        ("v0.9.1-unsloth.1", [("embeddings.safetensors", 4), ("X.GGUF", 4)], MESSAGE),
        ("v0.9.2-unsloth.1", [("x.gguf", 4)], None),
        ("v0.10.0-unsloth.1", [("x.gguf", 4)], None),
        ("garbage", [("x.gguf", 4)], None),
        ("v0.9.1-unsloth.1", OSError("unreadable cache"), None),
    ],
)
def test_the_download_gate(monkeypatch, tmp_path, tag, missing, expected):
    binary = _runtime(monkeypatch, tmp_path, record = {"release_tag": tag})
    _missing(monkeypatch, missing)
    assert audio_cpp_server.stale_runtime_download_problem(_model(), str(binary)) == expected
    assert audio_cpp_server.stale_runtime_download_problem(_model()) == expected


def test_a_custom_build_or_a_local_file_is_never_gated(monkeypatch, tmp_path):
    _missing(monkeypatch, [("x.gguf", 4)])
    _runtime(monkeypatch, tmp_path / "custom")
    assert audio_cpp_server.stale_runtime_download_problem(_model()) is None
    _runtime(monkeypatch, tmp_path / "old", record = {"release_tag": "v0.9.1-unsloth.1"})
    local = replace(_model(), local_path = str(tmp_path / "m.gguf"))
    assert audio_cpp_server.stale_runtime_download_problem(local) is None


def test_the_tag_is_checked_before_the_cache(monkeypatch, tmp_path):
    _runtime(monkeypatch, tmp_path, record = {"release_tag": "v0.9.2-unsloth.1"})
    _missing(monkeypatch, AssertionError("the cache was read on a current runtime"))
    assert audio_cpp_server.stale_runtime_download_problem(_model()) is None


def test_model_runtime_problem_refuses_a_download_on_an_old_runtime(monkeypatch, tmp_path):
    _runtime(monkeypatch, tmp_path, record = {"release_tag": "v0.9.1-unsloth.1"})
    _missing(monkeypatch, [("x.gguf", 4)])
    assert audio_cpp_server.model_runtime_problem(_model()) == MESSAGE


def test_load_is_refused_before_any_download(monkeypatch, tmp_path):
    _runtime(monkeypatch, tmp_path, record = {"release_tag": "v0.9.1-unsloth.1"})
    _missing(monkeypatch, [("x.gguf", 4)])
    downloads = []
    monkeypatch.setattr(
        audio_cpp_backend.AudioCppBackend,
        "_download_missing",
        lambda self, *a, **kw: downloads.append(a) or True,
    )
    model = _model("Chatterbox-GGUF", "chatterbox", "tts")
    config = SimpleNamespace(identifier = model.id, audio_cpp = model)
    with pytest.raises(RuntimeError, match = "needs a newer audio runtime"):
        audio_cpp_backend.AudioCppBackend().load_model(config)
    assert downloads == []


def test_the_dictation_download_is_refused_before_it_starts(monkeypatch, tmp_path):
    from core.inference import stt_audiocpp_sidecar, stt_download_worker

    _runtime(monkeypatch, tmp_path / "rt", record = {"release_tag": "v0.9.1-unsloth.1"})
    _missing(monkeypatch, [("x.gguf", 4)])
    model = _model()
    monkeypatch.setattr(stt_audiocpp_sidecar, "resolve_audio_cpp_stt_model", lambda *a, **kw: model)
    spawned = []
    monkeypatch.setattr(stt_download_worker, "spawn_download", lambda *a, **kw: spawned.append(a))
    import huggingface_hub

    def no_network(*a, **kw):
        raise AssertionError("the Hub was asked before the gate")

    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", no_network)
    state = stt_audiocpp_sidecar._AudioCppDownloadState()
    state._run(model.id, None, tmp_path / "hub")
    assert state._error == MESSAGE
    assert spawned == []


def test_a_missing_companion_is_refused_before_any_download(monkeypatch, tmp_path):
    _runtime(monkeypatch, tmp_path / "rt", record = {"release_tag": "v0.9.1-unsloth.1"})
    tts = replace(
        _model("MioTTS-GGUF", "miotts", "tts"),
        companions = (
            acm.CompanionModel("audio-cpp/audio.cpp-gguf/MioCodec-GGUF", "Q8_0", "codec_model"),
        ),
    )
    codec = _model("MioCodec-GGUF", "miocodec", "codec")
    # The TTS files are cached; only the codec still has files to fetch.
    monkeypatch.setattr(
        audio_cpp_files,
        "missing_files",
        lambda model, *, hub_cache = None: [] if model is tts else [("codec.gguf", 4)],
    )
    monkeypatch.setattr(audio_cpp_backend, "_resolve_companion", lambda *a, **kw: codec)
    downloads = []
    monkeypatch.setattr(
        audio_cpp_backend.AudioCppBackend,
        "_download_missing",
        staticmethod(lambda model, *a, **kw: downloads.append(model.id)),
    )
    assert audio_cpp_server.model_runtime_problem(tts) is None
    with pytest.raises(RuntimeError, match = "MioCodec-GGUF needs a newer audio runtime"):
        audio_cpp_backend.AudioCppBackend()._ensure_downloaded(tts, None)
    assert downloads == []


def test_a_hub_download_job_is_refused_on_an_old_runtime(monkeypatch, tmp_path):
    import asyncio

    from fastapi import HTTPException

    from hub.services.models import downloads

    _runtime(monkeypatch, tmp_path / "rt", record = {"release_tag": "v0.9.1-unsloth.1"})
    _missing(monkeypatch, [("x.gguf", 4)])
    model = replace(_model(), from_hub = True)
    monkeypatch.setattr(acm, "resolve", lambda *a, **kw: model)
    with pytest.raises(HTTPException) as refused:
        asyncio.run(downloads._audio_cpp_target(model.id, "Q8_0", persist = True))
    assert refused.value.status_code == 409 and refused.value.detail == MESSAGE
    # A cached model whose companion is missing is refused too: the job would fetch the companion.
    codec = replace(_model("MioCodec-GGUF", "miocodec", "codec"), from_hub = True)
    tts = replace(
        _model("MioTTS-GGUF", "miotts", "tts"),
        from_hub = True,
        companions = (acm.CompanionModel(codec.id, "Q8_0", "codec_model"),),
    )
    monkeypatch.setattr(acm, "resolve", lambda ident, *a, **kw: codec if ident == codec.id else tts)
    monkeypatch.setattr(
        audio_cpp_files,
        "missing_files",
        lambda m, *, hub_cache = None: [] if m is tts else [("codec.gguf", 4)],
    )
    with pytest.raises(HTTPException, match = "409"):
        asyncio.run(downloads._audio_cpp_target(tts.id, "Q8_0", persist = True))
    monkeypatch.setattr(acm, "resolve", lambda *a, **kw: model)
    _missing(monkeypatch, [("x.gguf", 4)])
    # Status and progress polls resolve the same row without the gate.
    assert asyncio.run(downloads._audio_cpp_target(model.id, "Q8_0"))[0] == model.repo_id


def test_a_download_job_outside_the_umbrella_is_gated_too(monkeypatch, tmp_path):
    import asyncio

    from fastapi import HTTPException

    from hub.services.models import downloads

    music = replace(
        _model(), id = "audio-cpp/MiniMax-Music3-GGUF", repo_id = "audio-cpp/MiniMax-Music3-GGUF"
    )
    looked_up = []
    monkeypatch.setattr(
        acm,
        "resolve",
        lambda ident, *a, **kw: looked_up.append(ident) or (music if ident == music.id else None),
    )
    _missing(monkeypatch, [("x.gguf", 4)])
    # A current runtime never looks the repo up, so ordinary GGUF downloads pay nothing.
    _runtime(monkeypatch, tmp_path / "new", record = {"release_tag": "v0.9.2-unsloth.1"})
    assert asyncio.run(downloads._audio_cpp_target(music.id, "Q8_0", persist = True)) == (
        music.id,
        "Q8_0",
    )
    assert looked_up == []
    _runtime(monkeypatch, tmp_path / "old", record = {"release_tag": "v0.9.1-unsloth.1"})
    with pytest.raises(HTTPException) as refused:
        asyncio.run(downloads._audio_cpp_target(music.id, "Q8_0", persist = True))
    assert (
        refused.value.status_code == 409 and "needs a newer audio runtime" in refused.value.detail
    )
    # A repo that is not an audio model, and status polls, pass through unchanged.
    assert asyncio.run(downloads._audio_cpp_target("org/llm-GGUF", "Q4_K_M", persist = True)) == (
        "org/llm-GGUF",
        "Q4_K_M",
    )
    assert asyncio.run(downloads._audio_cpp_target(music.id, "Q8_0")) == (music.id, "Q8_0")
    # A caller that must stay anonymous never has a repo resolved with the ambient token.
    looked_up.clear()
    assert asyncio.run(
        downloads._audio_cpp_target(music.id, "Q8_0", persist = True, stale_gate = False)
    ) == (music.id, "Q8_0")
    assert looked_up == []


def test_a_scoped_download_job_is_gated_on_the_files_it_fetches(monkeypatch, tmp_path):
    import asyncio

    import huggingface_hub
    from fastapi import HTTPException

    from hub.services.models import downloads

    aligner = _model("Qwen3-ForcedAligner-0.6B-GGUF", "qwen3_forced_aligner", "aligner")
    looked_up = []
    monkeypatch.setattr(acm, "resolve", lambda ident, *a, **kw: looked_up.append(ident) or aligner)
    cached = set()
    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda repo, name, **kw: "/cache/blob" if name in cached else None,
    )
    _missing(monkeypatch, [("x.gguf", 4)])
    files = ["Qwen3-ForcedAligner-0.6B-GGUF/aligner-q8_0.gguf", "README.md"]
    scoped = lambda repo, names: asyncio.run(  # noqa: E731
        downloads._refuse_stale_scoped_download(repo, names, None)
    )
    _runtime(monkeypatch, tmp_path / "new", record = {"release_tag": "v0.9.2-unsloth.1"})
    scoped(acm.AUDIO_CPP_REPO, files)
    assert looked_up == []
    _runtime(monkeypatch, tmp_path / "old", record = {"release_tag": "v0.9.1-unsloth.1"})
    with pytest.raises(HTTPException) as refused:
        scoped(acm.AUDIO_CPP_REPO, files)
    assert (
        refused.value.status_code == 409 and "needs a newer audio runtime" in refused.value.detail
    )
    # The umbrella's files resolve as their folder rows.
    assert looked_up == [aligner.id]
    # Files already cached, or no GGUF at all, fetch nothing the runtime loads.
    cached.add(files[0])
    scoped(acm.AUDIO_CPP_REPO, files)
    scoped("org/some-model", ["README.md", "config.json"])
    assert looked_up == [aligner.id]


def test_a_scoped_job_is_checked_at_the_quant_of_its_file(monkeypatch, tmp_path):
    import huggingface_hub

    from utils import hf_cache_settings

    folder = "Canary-180M-Flash-GGUF"
    q8, q4 = (RepoFile(f"{folder}/canary-{q}.gguf", 4) for q in ("q8_0", "q4_k_m"))
    variants = (AudioCppVariant("Q8_0", (q8,), q8.path), AudioCppVariant("Q4_K_M", (q4,), q4.path))
    base = replace(_model(), variants = variants)
    pick = lambda key: next(v for v in variants if v.key == (key or "Q8_0"))  # noqa: E731
    monkeypatch.setattr(
        acm, "resolve", lambda ident, variant = None, *a, **kw: replace(base, variant = pick(variant))
    )
    # The default Q8_0 is cached; only the Q4_K_M the job asks for is missing.
    monkeypatch.setattr(
        audio_cpp_files,
        "missing_files",
        lambda m, *, hub_cache = None: [(q4.path, 4)] if m.variant.key == "Q4_K_M" else [],
    )
    caches = []
    monkeypatch.setattr(hf_cache_settings, "active_hf_hub_cache", lambda: "/active/hub")
    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda repo, name, cache_dir = None: caches.append(cache_dir),
    )
    _runtime(monkeypatch, tmp_path / "old", record = {"release_tag": "v0.9.1-unsloth.1"})
    with pytest.raises(audio_cpp_server.AudioCppRuntimeTooOldError):
        acm.refuse_stale_download(acm.AUDIO_CPP_REPO, None, files = [q4.path])
    assert caches == ["/active/hub"]
    acm.refuse_stale_download(acm.AUDIO_CPP_REPO, None, files = [q8.path])


def test_the_timestamp_aligner_is_not_fetched_for_an_old_runtime(monkeypatch, tmp_path):
    from core.inference import stt_audiocpp_sidecar

    _runtime(monkeypatch, tmp_path / "rt", record = {"release_tag": "v0.9.1-unsloth.1"})
    _missing(monkeypatch, [("x.gguf", 4)])
    aligner = _model("Qwen3-ForcedAligner-0.6B-GGUF", "qwen3_forced_aligner", "aligner")
    monkeypatch.setattr(audio_cpp_backend, "_resolve_companion", lambda *a, **kw: aligner)
    downloads = []
    monkeypatch.setattr(
        audio_cpp_backend.AudioCppBackend,
        "_download_missing",
        staticmethod(lambda model, *a, **kw: downloads.append(model.id)),
    )
    side = stt_audiocpp_sidecar.AudioCppSttSidecar()
    with pytest.raises(
        stt_audiocpp_sidecar.SttModelNotDownloadedError, match = "needs a newer audio runtime"
    ):
        side._ensure_aligner_downloaded(_model("Qwen3-ASR-0.6B-GGUF", "qwen3_asr", "asr"))
    assert downloads == []


def test_runtime_knows_family_reads_the_folder_beside_the_server(monkeypatch, tmp_path):
    # Windows bundles keep the server and its fetched folder under bin/.
    _runtime(
        monkeypatch,
        tmp_path,
        record = {"release_tag": "v0.9.2-unsloth.1"},
        specs = ("owsm",),
        bin_dir = "bin",
    )
    assert acm.runtime_knows_family("owsm") is True
    assert acm.runtime_knows_family("crisperwhisper") is False
    assert acm.runtime_knows_family("../owsm") is None


def test_an_old_runtime_folder_hides_only_the_families_it_lacks(monkeypatch, tmp_path):
    rows = {
        "voxtral_tts": "tts",
        "kugelaudio": "tts",
        "owsm": "asr",
        "owsm_ctc": "asr",
        "crisperwhisper": "asr",
        "hviske_asr_v6": "asr",
    }
    old = [family for family in rows if family not in ("voxtral_tts", "hviske_asr_v6")]
    _runtime(monkeypatch, tmp_path, record = {"release_tag": "v0.9.1-unsloth.1"}, specs = old)
    refused = {
        family
        for family, task in rows.items()
        if acm.family_policy(family, {"tasks": [task]}).unsupported
    }
    assert refused == {"voxtral_tts", "hviske_asr_v6"}
    assert acm.family_policy("owsm", {"tasks": ["asr"]}).task == "asr"


def test_vad_families_never_reach_the_folder_check(monkeypatch, tmp_path):
    _runtime(monkeypatch, tmp_path, record = {"release_tag": "v0.9.2-unsloth.1"}, specs = ("owsm",))
    for family in ("silero_vad", "marblenet_vad"):
        assert family in acm.FAMILIES
        assert acm.family_policy(family).unsupported.startswith("Voice activity detection")


@pytest.fixture
def managed(monkeypatch, tmp_path):
    """An Unsloth Studio-owned v0.9.2 install without model_specs/, and a stub installer."""
    root = tmp_path / "audio.cpp"
    root.mkdir()
    (root / ".unsloth-studio-owned").write_text("")
    for name in (
        "AUDIOCPP_SERVER_PATH",
        "UNSLOTH_AUDIO_CPP_PATH",
        "UNSLOTH_SKIP_AUDIO_CPP_INSTALL",
    ):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(audio_cpp_server, "managed_audio_cpp_dir", lambda: root)
    record = {"release_tag": "v0.9.2-unsloth.1", "published_repo": REPO, "source_commit": COMMIT}
    binary = _runtime(monkeypatch, root, record = record)
    real_core = audio_cpp_update._installer().core
    fetches, forgets = [], []

    def fetch(repo, commit, dest_dir):
        fetches.append((repo, commit, dest_dir))
        (dest_dir / "model_specs").mkdir()
        return True

    installer = SimpleNamespace(
        core = real_core, _release_index = lambda repo, tag: {}, _fetch_model_specs = fetch
    )
    monkeypatch.setattr(audio_cpp_update, "_installer", lambda: installer)
    monkeypatch.setattr(acm, "forget", lambda *a: forgets.append(a))
    return SimpleNamespace(
        root = root,
        binary = binary,
        record = record,
        installer = installer,
        fetches = fetches,
        forgets = forgets,
    )


def _write_record(env, **changes):
    record = {k: v for k, v in {**env.record, **changes}.items() if v is not None}
    (env.root / audio_cpp_server.INSTALL_RECORD).write_text(json.dumps(record))


def test_backfill_fetches_the_recorded_commit_then_forgets(managed):
    (managed.root / ".model_specs-killed").mkdir()
    assert audio_cpp_update._backfill_model_specs() is True
    assert managed.fetches == [(REPO, COMMIT, managed.binary.parent)]
    assert managed.forgets == [()]
    assert not (managed.root / ".model_specs-killed").exists()
    # Once the folder is there, a second run does nothing.
    assert audio_cpp_update._backfill_model_specs() is False
    assert len(managed.fetches) == 1


def test_backfill_commit_falls_back_to_the_manifest_then_the_version(managed):
    manifest = "b" * 40
    _write_record(
        managed,
        source_commit = None,
        server_version = "audio.cpp v0.9.1-unsloth.1\ngit: 1234567 2026-10-07\nbuild: 20261007",
    )
    managed.installer._release_index = lambda repo, tag: {"source_commit": manifest}
    assert audio_cpp_update._backfill_model_specs() is True
    assert managed.fetches[-1][1] == manifest
    (managed.binary.parent / "model_specs").rmdir()

    def offline(repo, tag):
        raise OSError("offline")

    managed.installer._release_index = offline
    assert audio_cpp_update._backfill_model_specs() is True
    assert managed.fetches[-1][1] == "1234567"


def test_backfill_never_fetches_by_tag(managed):
    _write_record(managed, source_commit = None, server_version = "v0.9.2")
    assert audio_cpp_update._backfill_model_specs() is False
    assert managed.fetches == []


@pytest.mark.parametrize("case", ["unmanaged", "folder", "updating", "no_tag", "no_repo"])
def test_backfill_skips(managed, monkeypatch, case):
    if case == "unmanaged":
        (managed.root / ".unsloth-studio-owned").unlink()
    elif case == "folder":
        (managed.binary.parent / "model_specs").mkdir()
    elif case == "updating":
        monkeypatch.setattr(audio_cpp_server, "UPDATE_IN_PROGRESS", threading.Event())
        audio_cpp_server.UPDATE_IN_PROGRESS.set()
    elif case == "no_tag":
        _write_record(managed, release_tag = None)
    else:
        _write_record(managed, published_repo = None)
    assert audio_cpp_update._backfill_model_specs() is False
    assert managed.fetches == [] and managed.forgets == []


def test_backfill_skips_when_the_install_lock_is_busy(managed, monkeypatch):
    core = managed.installer.core

    def busy(path, *, timeout = None):
        assert timeout == 2
        raise core.BusyInstallConflict("busy")

    managed.installer.core = SimpleNamespace(
        install_lock = busy, install_lock_path = core.install_lock_path
    )
    assert audio_cpp_update._backfill_model_specs() is False
    assert managed.fetches == []


def test_a_failed_fetch_leaves_no_folder_and_raises_nothing(managed):
    managed.installer._fetch_model_specs = lambda repo, commit, dest: False
    assert audio_cpp_update._backfill_model_specs() is False
    assert not (managed.binary.parent / "model_specs").exists()
    assert list(managed.binary.parent.glob(".model_specs-*")) == []
    assert managed.forgets == []

    def broken(repo, commit, dest):
        raise RuntimeError("boom")

    managed.installer._fetch_model_specs = broken
    assert audio_cpp_update._backfill_model_specs() is False


def test_the_backfill_starts_once_and_not_when_update_checks_are_off(monkeypatch):
    calls = []
    ran = threading.Event()
    monkeypatch.setattr(audio_cpp_update, "_backfill_started", False)
    monkeypatch.setattr(
        audio_cpp_update, "_backfill_model_specs", lambda: calls.append(1) or ran.set()
    )
    monkeypatch.setenv("UNSLOTH_DISABLE_UPDATE_CHECK", "1")
    audio_cpp_update.start_model_specs_backfill()
    assert audio_cpp_update._backfill_started is False
    monkeypatch.delenv("UNSLOTH_DISABLE_UPDATE_CHECK")
    audio_cpp_update.start_model_specs_backfill()
    audio_cpp_update.start_model_specs_backfill()
    assert ran.wait(5)
    for thread in threading.enumerate():
        if thread.name == "audio-cpp-spec-backfill":
            thread.join(5)
    assert calls == [1]
