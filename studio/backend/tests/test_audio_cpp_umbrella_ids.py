# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cache actions on an audio.cpp umbrella folder row (``audio-cpp/audio.cpp-gguf/<Folder>``):
delete, delete impact, cached path and reveal act on that folder alone, and the cached GGUF
inventory lists each downloaded folder for the Audio pages."""

import asyncio
import hashlib
import os
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from core.inference import audio_cpp_files
from core.inference import audio_cpp_models as acm
from core.inference.audio_cpp_models import AUDIO_CPP_REPO

QWEN = f"{AUDIO_CPP_REPO}/Qwen3-TTS-12Hz-0.6B-Base-GGUF"
KOKORO = f"{AUDIO_CPP_REPO}/Kokoro-82M-GGUF"
ASR = f"{AUDIO_CPP_REPO}/Qwen3-ASR-0.6B-GGUF"
QWEN_FILE = "Qwen3-TTS-12Hz-0.6B-Base-GGUF/qwen3-tts-12hz-0.6b-base-q8_0.gguf"
SNAPSHOT = Path("models--audio-cpp--audio.cpp-gguf") / "snapshots" / ("a" * 40)
QWEN_KEY = "Qwen3-TTS-12Hz-0.6B-Base-GGUF/qwen3-tts-12hz-0.6b-base-q8_0"
FILES = {
    QWEN_FILE: "qwen3_tts",
    "Qwen3-ASR-0.6B-GGUF/qwen3-asr-0.6b-q8_0.gguf": "qwen3_asr",
    "Kokoro-82M-GGUF/kokoro-82m-q8_0.gguf": "kokoro_tts",
}


def _gguf(family: str, pad: int) -> bytes:
    kv = [
        ("general.architecture", 8, "audiocpp"),
        ("general.name", 8, "t"),
        ("audiocpp.model_spec.version", 4, 1),
        ("audiocpp.model_spec.family", 8, family),
    ]
    out = bytearray(struct.pack("<IIQQ", 0x46554747, 3, 0, len(kv)))
    for key, vtype, value in kv:
        k = key.encode()
        out += struct.pack("<Q", len(k)) + k + struct.pack("<I", vtype)
        if vtype == 8:
            v = value.encode()
            out += struct.pack("<Q", len(v)) + v
        else:
            out += struct.pack("<I", value)
    return bytes(out) + b"\0" * pad


def _add_snapshot(hub: Path, sha: str, files: dict, *, main: bool) -> Path:
    repo_dir = hub / ("models--" + AUDIO_CPP_REPO.replace("/", "--"))
    snap = repo_dir / "snapshots" / sha
    (repo_dir / "blobs").mkdir(parents = True, exist_ok = True)
    if main:
        (repo_dir / "refs").mkdir(exist_ok = True)
        (repo_dir / "refs" / "main").write_text(sha)
    for index, (rel, family) in enumerate(files.items()):
        data = _gguf(family, 4096 * (index + 1)) + rel.encode()
        blob = repo_dir / "blobs" / hashlib.sha256(data).hexdigest()
        blob.write_bytes(data)
        path = snap / rel
        path.parent.mkdir(parents = True, exist_ok = True)
        os.symlink(os.path.relpath(blob, path.parent), path)
    return snap


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A three-folder umbrella repo in an HF cache that is the only one any lookup can see."""
    from hub.services.models import cache_inventory, deletion
    from hub.utils import hf_cache_state, state_dir
    from utils import hf_cache_settings

    hf_home = tmp_path / "hf"
    root = hf_home / "hub"
    root.mkdir(parents = True)
    paths = hf_cache_settings.HuggingFaceCachePaths(hf_home, root, hf_home / "xet", "studio")
    monkeypatch.setattr(hf_cache_settings, "get_hf_cache_paths", lambda: paths)
    monkeypatch.setattr(hf_cache_settings, "known_hf_cache_homes", lambda: [hf_home])
    monkeypatch.setattr(hf_cache_settings, "known_hf_hub_caches", lambda: [root])
    monkeypatch.setattr(hf_cache_settings, "active_hf_hub_cache", lambda: str(root))
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda **_kwargs: [root])
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "studio-cache")
    monkeypatch.setattr(acm, "_hub_cache", lambda: root)
    monkeypatch.setattr(audio_cpp_files, "_hub_cache", lambda: root)
    monkeypatch.setattr(acm, "runtime_spec", lambda family: None)
    monkeypatch.setattr(acm, "runtime_knows_family", lambda family: None)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_CACHE", str(root))
    monkeypatch.setattr(deletion.account_access, "require_installation_owner", lambda: None)
    acm.forget()
    cache_inventory.invalidate_hf_cache_scans()
    _add_snapshot(root, "a" * 40, FILES, main = True)
    yield root
    acm.forget()
    cache_inventory.invalidate_hf_cache_scans()


def _path(repo_id: str, variant: str) -> dict:
    import routes.models as rm
    return asyncio.run(
        rm.get_cached_model_path(repo_id = repo_id, variant = variant, current_subject = "x")
    )


def _left(hub: Path) -> list[str]:
    repo_dir = hub / ("models--" + AUDIO_CPP_REPO.replace("/", "--"))
    return sorted(
        p.relative_to(s).as_posix()
        for s in (repo_dir / "snapshots").iterdir()
        for p in s.rglob("*.gguf")
    )


def _idle(
    monkeypatch,
    *,
    active = None,
    loading = (),
    stt = None,
):
    from core.inference import orchestrator, stt_audiocpp_sidecar

    side = stt_audiocpp_sidecar.AudioCppSttSidecar()
    if stt:
        side._loading, side._loading_model = True, stt
    backend = SimpleNamespace(active_model_name = active, loading_models = set(loading))
    monkeypatch.setattr(stt_audiocpp_sidecar, "get_audio_cpp_stt_sidecar", lambda: side)
    monkeypatch.setattr(orchestrator, "peek_inference_backend", lambda: backend)


def test_cached_folder_target_scopes_keys_to_the_folder(hub):
    assert acm.cached_folder_target(QWEN, "Q8_0") == (AUDIO_CPP_REPO, [QWEN_KEY])
    assert acm.cached_folder_target(QWEN, None) == (AUDIO_CPP_REPO, [QWEN_KEY])
    assert acm.cached_folder_target(f"{QWEN}:Q8_0") == (AUDIO_CPP_REPO, [QWEN_KEY])
    assert acm.cached_folder_target(QWEN, "Q4_K_M") == (AUDIO_CPP_REPO, [])
    assert acm.cached_folder_target(f"{AUDIO_CPP_REPO}/Missing-GGUF", None) == (AUDIO_CPP_REPO, [])
    # The repo itself and ordinary repos keep their own handling.
    assert acm.cached_folder_target(AUDIO_CPP_REPO, "Q8_0") is None
    assert acm.cached_folder_target("unsloth/Qwen3-0.6B-GGUF", "Q8_0") is None


@pytest.mark.parametrize("variant", ["Q8_0", None])
def test_deleting_a_folder_row_keeps_every_other_folder(hub, monkeypatch, variant):
    from hub.services.models import deletion

    _idle(monkeypatch)
    calls = []
    real = deletion._delete_cached_model_blocking
    monkeypatch.setattr(
        deletion,
        "_delete_cached_model_blocking",
        lambda repo_id, v, *a, **k: calls.append((repo_id, v)) or real(repo_id, v, *a, **k),
    )
    result = asyncio.run(deletion.delete_cached_model_response(QWEN, variant))
    assert result["repo_id"] == QWEN
    assert calls == [(AUDIO_CPP_REPO, QWEN_KEY)]
    assert _left(hub) == [
        "Kokoro-82M-GGUF/kokoro-82m-q8_0.gguf",
        "Qwen3-ASR-0.6B-GGUF/qwen3-asr-0.6b-q8_0.gguf",
    ]


def test_deleting_an_uncached_quant_of_a_folder_changes_nothing(hub, monkeypatch):
    from hub.services.models import deletion

    _idle(monkeypatch)
    with pytest.raises(HTTPException) as err:
        asyncio.run(deletion.delete_cached_model_response(QWEN, "Q4_K_M"))
    assert err.value.status_code == 404
    assert len(_left(hub)) == 3


def test_a_loaded_folder_blocks_only_its_own_delete(hub, monkeypatch):
    from hub.services.models import deletion

    _idle(monkeypatch, active = f"{KOKORO}:Q8_0")
    asyncio.run(deletion.delete_cached_model_response(QWEN, "Q8_0"))
    assert QWEN_FILE not in _left(hub)
    with pytest.raises(HTTPException) as err:
        asyncio.run(deletion.delete_cached_model_response(KOKORO, "Q8_0"))
    assert err.value.status_code == 400
    assert "Unload" in err.value.detail
    assert "Kokoro-82M-GGUF/kokoro-82m-q8_0.gguf" in _left(hub)


@pytest.mark.parametrize(
    "loaded",
    [
        {"active": QWEN},
        {"active": f"{QWEN}:Q8_0"},
        {"loading": [QWEN]},
        {"stt": ASR},
    ],
)
def test_folder_guard_matches_the_folder_however_it_was_loaded(monkeypatch, loaded):
    from hub.services.models import deletion

    _idle(monkeypatch, **loaded)
    held = loaded.get("stt") or (loaded.get("loading") or [loaded.get("active")])[0]
    held_row = acm.split_variant_ref(held)[0]
    other = KOKORO
    assert deletion._audio_cpp_blocks_delete(held_row) is True
    assert deletion._audio_cpp_blocks_delete(other) is False
    # The whole repo is still held while any of its folders is.
    assert deletion._audio_cpp_blocks_delete(AUDIO_CPP_REPO) is True


def test_folder_guard_maps_a_legacy_dictation_key_to_its_folder(monkeypatch):
    from hub.services.models import deletion

    legacy = acm.legacy_keys()[0]
    row = acm.parse_identifier(legacy).id
    _idle(monkeypatch, stt = legacy)
    assert deletion._audio_cpp_blocks_delete(row) is True
    assert deletion._audio_cpp_blocks_delete(f"{AUDIO_CPP_REPO}/Not-That-Folder-GGUF") is False


def test_delete_impact_of_a_folder_row_counts_only_its_files(hub):
    from hub.services.models import companion_cleanup

    qwen_bytes = hub / SNAPSHOT / QWEN_FILE
    size = qwen_bytes.stat().st_size
    for variant in ("Q8_0", None):
        impact = asyncio.run(companion_cleanup.delete_impact_response(QWEN, variant))
        assert impact["repo_id"] == QWEN
        assert impact["reclaimed_bytes"] == size
    whole = asyncio.run(companion_cleanup.delete_impact_response(AUDIO_CPP_REPO, None))
    assert whole["reclaimed_bytes"] > size
    with pytest.raises(HTTPException) as err:
        asyncio.run(companion_cleanup.delete_impact_response(QWEN, "Q4_K_M"))
    assert err.value.status_code == 404


def test_cached_path_and_reveal_of_a_folder_row(hub, monkeypatch):
    import routes.models as rm
    from utils.paths import path_utils

    snap = hub / SNAPSHOT
    got = _path(QWEN, "Q8_0")
    assert Path(got["path"]) == snap / QWEN_FILE
    got = _path(QWEN, "")
    assert Path(got["path"]) == snap / "Qwen3-TTS-12Hz-0.6B-Base-GGUF"
    assert got["is_dir"] is True
    revealed = []
    monkeypatch.setattr(rm.account_access, "require_installation_owner", lambda: None)
    monkeypatch.setattr(path_utils, "reveal_in_file_manager", lambda path: revealed.append(path))
    asyncio.run(rm.reveal_cached_model(repo_id = QWEN, variant = None, current_subject = "x"))
    assert revealed == [snap / "Qwen3-TTS-12Hz-0.6B-Base-GGUF"]
    with pytest.raises(HTTPException) as err:
        _path(QWEN, "Q4_K_M")
    assert err.value.status_code == 404
    # Ordinary malformed ids are still refused.
    with pytest.raises(HTTPException) as err:
        _path("a/b/c", "")
    assert err.value.status_code == 400


def test_cached_path_of_a_folder_only_in_an_older_snapshot(hub):
    kokoro_q4 = {"Kokoro-82M-GGUF/kokoro-82m-q4_0.gguf": "kokoro_tts"}
    newer = _add_snapshot(hub, "b" * 40, kokoro_q4, main = True)
    older = hub / SNAPSHOT
    acm.forget()
    got = _path(QWEN, "")
    assert Path(got["path"]) == older / "Qwen3-TTS-12Hz-0.6B-Base-GGUF"
    assert Path(got["path"]).parent != newer


def test_cached_gguf_lists_each_umbrella_folder_beside_the_repo_row(hub):
    from hub.schemas.inventory import CachedGgufResponse
    from hub.services.models import cache_inventory

    aligner = "Qwen3-ForcedAligner-0.6B-GGUF/qwen3-forced-aligner-0.6b-q8_0.gguf"
    _add_snapshot(hub, "a" * 40, {aligner: "qwen3_forced_aligner"}, main = True)
    rows = {row["repo_id"]: row for row in cache_inventory._scan_cached_gguf()}
    # A companion with no Audio page of its own is managed through the repo row only.
    assert f"{AUDIO_CPP_REPO}/Qwen3-ForcedAligner-0.6B-GGUF" not in rows
    assert AUDIO_CPP_REPO in rows
    assert not rows[AUDIO_CPP_REPO].get("umbrella_folder")
    assert {QWEN, KOKORO, ASR} <= set(rows)
    qwen, kokoro, asr = rows[QWEN], rows[KOKORO], rows[ASR]
    assert qwen["umbrella_folder"] is True
    assert qwen["load_id"] == QWEN
    assert qwen["task"] == "text-to-speech"
    assert qwen["audio_type"] == "audiocpp_tts"
    assert qwen["audio_workflows"] == ["clone"]
    assert kokoro["audio_workflows"] == ["speak"]
    assert asr["task"] == "automatic-speech-recognition"
    assert qwen["capabilities"]["can_chat"] is False
    assert qwen["inventory_id"] != rows[AUDIO_CPP_REPO]["inventory_id"]
    qwen_size = (hub / SNAPSHOT / QWEN_FILE).stat().st_size
    assert qwen["size_bytes"] == qwen_size
    folders = sum(rows[r]["size_bytes"] for r in (QWEN, KOKORO, ASR))
    assert folders <= rows[AUDIO_CPP_REPO]["size_bytes"]
    dumped = CachedGgufResponse(cached = list(rows.values())).model_dump()
    flags = {row["repo_id"]: row["umbrella_folder"] for row in dumped["cached"]}
    assert flags[QWEN] is True and flags[AUDIO_CPP_REPO] is False


@pytest.mark.parametrize("local_path", [None, QWEN, "cache"])
def test_variants_of_a_folder_row_list_whatever_local_path_the_picker_sends(hub, local_path):
    from hub.services.models import gguf_variants

    if local_path == "cache":
        local_path = str(hub / "models--audio-cpp--audio.cpp-gguf")
    answer = asyncio.run(
        gguf_variants.get_gguf_variants_answer(QWEN, offline = True, local_path = local_path)
    )
    assert [(v.quant, v.downloaded) for v in answer.response.variants] == [("Q8_0", True)]


def test_a_multi_size_folder_matches_the_hub_key_against_its_one_cached_size(hub):
    moon = f"{AUDIO_CPP_REPO}/Moonshine-Streaming-GGUF"
    key = "Moonshine-Streaming-GGUF/moonshine-streaming-tiny-q8_0"
    _add_snapshot(hub, "a" * 40, {f"{key}.gguf": "moonshine_asr"}, main = True)
    acm.forget()
    # The picker names the quant from the Hub listing; the cache holds only the tiny size.
    assert acm.cached_folder_target(moon, "tiny/Q8_0") == (AUDIO_CPP_REPO, [key])
    assert acm.cached_folder_target(moon, "Q8_0") == (AUDIO_CPP_REPO, [key])
    assert acm.cached_folder_target(moon, "small/Q8_0") == (AUDIO_CPP_REPO, [])
    assert acm.cached_folder_target(f"{moon}/tiny", None) == (AUDIO_CPP_REPO, [key])


def test_a_loaded_model_holds_the_folder_it_loads_as_a_companion(monkeypatch):
    from hub.services.models import deletion

    codec = acm.FAMILIES["miotts"].companions[0].id
    miotts = f"{AUDIO_CPP_REPO}/MioTTS-1.7B-GGUF"
    monkeypatch.setattr(
        acm,
        "resolve",
        lambda identifier, *a, **k: SimpleNamespace(companions = acm.FAMILIES["miotts"].companions)
        if identifier.startswith(miotts)
        else None,
    )
    _idle(monkeypatch, active = f"{miotts}:Q8_0")
    assert deletion._audio_cpp_blocks_delete(codec) is True
    assert deletion._audio_cpp_blocks_delete(KOKORO) is False


def test_a_finished_download_lists_its_folder_on_the_next_scan(hub):
    from hub.services.models import cache_inventory

    moon = f"{AUDIO_CPP_REPO}/Moonshine-Streaming-GGUF"
    assert moon not in {row["repo_id"] for row in cache_inventory._scan_cached_gguf()}
    key = "Moonshine-Streaming-GGUF/moonshine-streaming-tiny-q8_0"
    _add_snapshot(hub, "a" * 40, {f"{key}.gguf": "moonshine_asr"}, main = True)
    # What a completed download calls; the audio.cpp memo must not keep the old listing.
    cache_inventory.invalidate_hf_cache_scans()
    assert moon in {row["repo_id"] for row in cache_inventory._scan_cached_gguf()}


def test_a_folder_variant_delete_takes_its_own_voice_files(hub, monkeypatch):
    from hub.services.models import companion_cleanup, deletion

    _idle(monkeypatch)
    snap = hub / SNAPSHOT
    blobs = hub / "models--audio-cpp--audio.cpp-gguf" / "blobs"
    pocket = f"{AUDIO_CPP_REPO}/PocketTTS-GGUF"
    for lang in ("english", "french"):
        rel = f"PocketTTS-GGUF/{lang}/pocket-tts-{lang}-q8_0.gguf"
        _add_snapshot(hub, "a" * 40, {rel: "pocket_tts"}, main = True)
        voice = blobs / f"voice-{lang}"
        voice.write_bytes(lang.encode() * 100)
        path = snap / f"PocketTTS-GGUF/{lang}/embeddings/alba.safetensors"
        path.parent.mkdir(parents = True, exist_ok = True)
        os.symlink(os.path.relpath(voice, path.parent), path)
    acm.forget()
    english = snap / "PocketTTS-GGUF/english"
    gguf = (english / "pocket-tts-english-q8_0.gguf").stat().st_size
    voice = (english / "embeddings/alba.safetensors").stat().st_size
    impact = asyncio.run(companion_cleanup.delete_impact_response(pocket, "english/Q8_0"))
    assert impact["reclaimed_bytes"] == gguf + voice
    asyncio.run(deletion.delete_cached_model_response(pocket, "english/Q8_0"))
    assert not (english / "pocket-tts-english-q8_0.gguf").exists()
    assert not (english / "embeddings/alba.safetensors").exists()
    assert not (blobs / "voice-english").exists()
    assert (snap / "PocketTTS-GGUF/french/embeddings/alba.safetensors").exists()
    assert (snap / "PocketTTS-GGUF/french/pocket-tts-french-q8_0.gguf").exists()
    assert len(_left(hub)) == 4


def test_a_folder_delete_preview_counts_extras_its_variants_share(hub, monkeypatch):
    from hub.services.models import companion_cleanup, deletion

    _idle(monkeypatch)
    snap = hub / SNAPSHOT
    blobs = hub / "models--audio-cpp--audio.cpp-gguf" / "blobs"
    pocket = f"{AUDIO_CPP_REPO}/PocketTTS-GGUF"
    for quant in ("q8_0", "f16"):
        rel = f"PocketTTS-GGUF/english/pocket-tts-english-{quant}.gguf"
        _add_snapshot(hub, "a" * 40, {rel: "pocket_tts"}, main = True)
    voice = blobs / "voice-english"
    voice.write_bytes(b"v" * 900)
    path = snap / "PocketTTS-GGUF/english/embeddings/alba.safetensors"
    path.parent.mkdir(parents = True, exist_ok = True)
    os.symlink(os.path.relpath(voice, path.parent), path)
    acm.forget()
    ggufs = sum(p.stat().st_size for p in (snap / "PocketTTS-GGUF/english").glob("*.gguf"))
    impact = asyncio.run(companion_cleanup.delete_impact_response(pocket, None))
    assert impact["reclaimed_bytes"] == ggufs + 900
    asyncio.run(deletion.delete_cached_model_response(pocket, None))
    assert not path.exists() and not voice.exists()
    assert len(_left(hub)) == 3


@pytest.mark.parametrize(
    "peer, kept",
    [
        # english BF16 is downloading: its voices landed first, its GGUF is not listed yet.
        ("PocketTTS-GGUF/english/pocket-tts-english-bf16", True),
        # Every umbrella folder downloads on the same repo; another folder needs nothing here.
        ("Kokoro-82M-GGUF/kokoro-82m-q4_0", False),
    ],
)
def test_a_variant_delete_keeps_voice_files_only_a_downloading_sibling_needs(
    hub, monkeypatch, peer, kept
):
    from hub.services.models import companion_cleanup, deletion
    from hub.utils import download_registry

    _idle(monkeypatch)
    registry = download_registry.DownloadRegistry()
    registry.claim(
        f"{AUDIO_CPP_REPO}::{peer}",
        download_registry.TRANSPORT_HTTP,
        repo_type = "model",
        repo_id = AUDIO_CPP_REPO,
        variant = peer,
        blob_hashes = frozenset({"peer"}),
        progress_blob_hashes = frozenset({"peer"}),
    )
    monkeypatch.setattr(deletion.downloads, "registry", registry)
    snap = hub / SNAPSHOT
    blobs = hub / "models--audio-cpp--audio.cpp-gguf" / "blobs"
    pocket = f"{AUDIO_CPP_REPO}/PocketTTS-GGUF"
    rel = "PocketTTS-GGUF/english/pocket-tts-english-q8_0.gguf"
    _add_snapshot(hub, "a" * 40, {rel: "pocket_tts"}, main = True)
    voice = blobs / "voice-english"
    voice.write_bytes(b"v" * 900)
    path = snap / "PocketTTS-GGUF/english/embeddings/alba.safetensors"
    path.parent.mkdir(parents = True, exist_ok = True)
    os.symlink(os.path.relpath(voice, path.parent), path)
    acm.forget()
    gguf = (snap / rel).stat().st_size
    impact = asyncio.run(companion_cleanup.delete_impact_response(pocket, "english/Q8_0"))
    assert impact["reclaimed_bytes"] == gguf + (0 if kept else 900)
    asyncio.run(deletion.delete_cached_model_response(pocket, "english/Q8_0"))
    assert not (snap / rel).exists()
    assert path.exists() is kept and voice.exists() is kept


@pytest.mark.parametrize("folder", ["C:", "C:Qwen3-TTS-GGUF"])
def test_a_drive_like_folder_id_names_nothing(hub, monkeypatch, folder):
    from hub.services.models import companion_cleanup, deletion

    _idle(monkeypatch)
    row = f"{AUDIO_CPP_REPO}/{folder}"
    assert acm.cached_folder_target(row, None) == (AUDIO_CPP_REPO, [])
    assert acm.folder_snapshot_dir(row) is None
    for call in (
        lambda: _path(row, ""),
        lambda: asyncio.run(deletion.delete_cached_model_response(row, None)),
        lambda: asyncio.run(companion_cleanup.delete_impact_response(row, None)),
    ):
        with pytest.raises(HTTPException) as err:
            call()
        assert err.value.status_code == 404
    assert len(_left(hub)) == 3


def test_a_bare_quant_two_cached_sizes_share_names_nothing(hub, monkeypatch):
    from hub.services.models import deletion

    _idle(monkeypatch)
    moon = f"{AUDIO_CPP_REPO}/Moonshine-Streaming-GGUF"
    tiny = "Moonshine-Streaming-GGUF/moonshine-streaming-tiny-q8_0"
    small = "Moonshine-Streaming-GGUF/moonshine-streaming-small-q8_0"
    files = {f"{tiny}.gguf": "moonshine_asr", f"{small}.gguf": "moonshine_asr"}
    _add_snapshot(hub, "a" * 40, files, main = True)
    acm.forget()
    assert acm.cached_folder_target(moon, "Q8_0") == (AUDIO_CPP_REPO, [])
    assert acm.cached_folder_target(moon, "tiny/Q8_0") == (AUDIO_CPP_REPO, [tiny])
    assert acm.cached_folder_target(moon, "small/Q8_0") == (AUDIO_CPP_REPO, [small])
    assert sorted(acm.cached_folder_target(moon, None)[1]) == sorted([small, tiny])
    with pytest.raises(HTTPException) as err:
        asyncio.run(deletion.delete_cached_model_response(moon, "Q8_0"))
    assert err.value.status_code == 404
    assert len(_left(hub)) == 5


def test_a_folder_delete_reserves_every_variant_before_removing_any(hub, monkeypatch):
    from hub.services.models import deletion
    from hub.utils import download_registry

    _idle(monkeypatch)
    moon = f"{AUDIO_CPP_REPO}/Moonshine-Streaming-GGUF"
    tiny = "Moonshine-Streaming-GGUF/moonshine-streaming-tiny-q8_0"
    small = "Moonshine-Streaming-GGUF/moonshine-streaming-small-q8_0"
    files = {f"{tiny}.gguf": "moonshine_asr", f"{small}.gguf": "moonshine_asr"}
    _add_snapshot(hub, "a" * 40, files, main = True)
    acm.forget()
    registry = download_registry.DownloadRegistry()
    monkeypatch.setattr(deletion.downloads, "registry", registry)
    keys = acm.cached_folder_target(moon, None)[1]
    assert sorted(keys) == sorted([tiny, small])
    later = keys[-1]
    # A re-download of the folder's second variant is running.
    registry.claim(
        f"{AUDIO_CPP_REPO}::{later.lower()}",
        download_registry.TRANSPORT_HTTP,
        repo_type = "model",
        repo_id = AUDIO_CPP_REPO,
        variant = later,
        blob_hashes = frozenset({"later"}),
        progress_blob_hashes = frozenset({"later"}),
    )
    with pytest.raises(HTTPException) as err:
        asyncio.run(deletion.delete_cached_model_response(moon, None))
    assert err.value.status_code == 400
    assert len(_left(hub)) == 5
    assert not registry._deleting


@pytest.mark.parametrize("variant", ["/", "//", " / "])
def test_an_empty_variant_path_names_no_folder_variant(hub, monkeypatch, variant):
    from hub.services.models import deletion

    _idle(monkeypatch)
    assert acm.cached_folder_target(QWEN, variant) == (AUDIO_CPP_REPO, [])
    with pytest.raises(HTTPException) as err:
        asyncio.run(deletion.delete_cached_model_response(QWEN, variant))
    assert err.value.status_code == 404
    assert len(_left(hub)) == 3


@pytest.mark.parametrize(
    "variant, picks",
    [
        (".", []),
        ("./", []),
        ("Moonshine-Streaming-GGUF", []),
        ("gguf", []),
        ("-", []),
        ("q8_0.gguf", []),
        ("TINY/Q8_0", ["tiny"]),
        ("tiny/Q8_0/", ["tiny"]),
        ("moonshine-streaming-small-q8_0", ["small"]),
        ("moonshine-streaming-small-q8_0.gguf", ["small"]),
        ("tiny", ["tiny"]),
    ],
)
def test_a_folder_variant_names_only_what_it_says(hub, variant, picks):
    moon = f"{AUDIO_CPP_REPO}/Moonshine-Streaming-GGUF"
    keys = {
        size: f"Moonshine-Streaming-GGUF/moonshine-streaming-{size}-q8_0"
        for size in ("tiny", "small")
    }
    _add_snapshot(hub, "a" * 40, {f"{k}.gguf": "moonshine_asr" for k in keys.values()}, main = True)
    acm.forget()
    assert acm.cached_folder_target(moon, variant) == (AUDIO_CPP_REPO, [keys[p] for p in picks])


def test_cached_path_of_a_folder_skips_a_newer_snapshot_with_only_metadata(hub):
    newer = _add_snapshot(hub, "c" * 40, {}, main = True)
    (newer / "Qwen3-TTS-12Hz-0.6B-Base-GGUF").mkdir(parents = True)
    (newer / "Qwen3-TTS-12Hz-0.6B-Base-GGUF" / "README.md").write_text("card")
    acm.forget()
    got = _path(QWEN, "")
    assert Path(got["path"]) == hub / SNAPSHOT / "Qwen3-TTS-12Hz-0.6B-Base-GGUF"


def test_cached_path_of_a_folder_finds_weights_with_an_uppercase_extension(hub):
    older = _add_snapshot(hub, "0" * 40, {}, main = False)
    (older / "Upper-GGUF").mkdir(parents = True)
    (older / "Upper-GGUF" / "MODEL-Q8_0.GGUF").write_bytes(b"GGUF")
    newer = hub / SNAPSHOT
    (newer / "Upper-GGUF").mkdir()
    (newer / "Upper-GGUF" / "README.md").write_text("card")
    assert acm.folder_snapshot_dir(f"{AUDIO_CPP_REPO}/Upper-GGUF") == older / "Upper-GGUF"


def test_a_folder_delete_preview_keeps_shared_extras_a_downloading_sibling_needs(hub, monkeypatch):
    from hub.services.models import companion_cleanup, deletion
    from hub.utils import download_registry

    _idle(monkeypatch)
    snap = hub / SNAPSHOT
    blobs = hub / "models--audio-cpp--audio.cpp-gguf" / "blobs"
    pocket = f"{AUDIO_CPP_REPO}/PocketTTS-GGUF"
    for quant in ("q8_0", "f16"):
        rel = f"PocketTTS-GGUF/english/pocket-tts-english-{quant}.gguf"
        _add_snapshot(hub, "a" * 40, {rel: "pocket_tts"}, main = True)
    voice = blobs / "voice-english"
    voice.write_bytes(b"v" * 900)
    path = snap / "PocketTTS-GGUF/english/embeddings/alba.safetensors"
    path.parent.mkdir(parents = True, exist_ok = True)
    os.symlink(os.path.relpath(voice, path.parent), path)
    acm.forget()
    registry = download_registry.DownloadRegistry()
    peer = "PocketTTS-GGUF/english/pocket-tts-english-bf16"
    registry.claim(
        f"{AUDIO_CPP_REPO}::{peer}",
        download_registry.TRANSPORT_HTTP,
        repo_type = "model",
        repo_id = AUDIO_CPP_REPO,
        variant = peer,
        blob_hashes = frozenset({"peer"}),
        progress_blob_hashes = frozenset({"peer"}),
    )
    monkeypatch.setattr(deletion.downloads, "registry", registry)
    ggufs = sum(p.stat().st_size for p in (snap / "PocketTTS-GGUF/english").glob("*.gguf"))
    impact = asyncio.run(companion_cleanup.delete_impact_response(pocket, None))
    assert impact["reclaimed_bytes"] == ggufs
    asyncio.run(deletion.delete_cached_model_response(pocket, None))
    assert path.exists() and voice.exists()


def test_a_folder_delete_preview_keeps_only_the_busy_directory_s_extras(hub, monkeypatch):
    from hub.services.models import companion_cleanup, deletion
    from hub.utils import download_registry

    _idle(monkeypatch)
    snap = hub / SNAPSHOT
    blobs = hub / "models--audio-cpp--audio.cpp-gguf" / "blobs"
    pocket = f"{AUDIO_CPP_REPO}/PocketTTS-GGUF"
    voices = {}
    for lang in ("english", "french"):
        for quant in ("q8_0", "f16"):
            rel = f"PocketTTS-GGUF/{lang}/pocket-tts-{lang}-{quant}.gguf"
            _add_snapshot(hub, "a" * 40, {rel: "pocket_tts"}, main = True)
        voices[lang] = blobs / f"voice-{lang}"
        voices[lang].write_bytes(lang[:1].encode() * 900)
        path = snap / f"PocketTTS-GGUF/{lang}/embeddings/alba.safetensors"
        path.parent.mkdir(parents = True, exist_ok = True)
        os.symlink(os.path.relpath(voices[lang], path.parent), path)
    acm.forget()
    registry = download_registry.DownloadRegistry()
    peer = "PocketTTS-GGUF/english/pocket-tts-english-bf16"
    registry.claim(
        f"{AUDIO_CPP_REPO}::{peer}",
        download_registry.TRANSPORT_HTTP,
        repo_type = "model",
        repo_id = AUDIO_CPP_REPO,
        variant = peer,
        blob_hashes = frozenset({"peer"}),
        progress_blob_hashes = frozenset({"peer"}),
    )
    monkeypatch.setattr(deletion.downloads, "registry", registry)
    ggufs = sum(p.stat().st_size for p in (snap / "PocketTTS-GGUF").rglob("*.gguf"))
    impact = asyncio.run(companion_cleanup.delete_impact_response(pocket, None))
    assert impact["reclaimed_bytes"] == ggufs + 900
    asyncio.run(deletion.delete_cached_model_response(pocket, None))
    assert voices["english"].exists() and not voices["french"].exists()


def test_a_listing_read_before_a_download_finished_is_not_memoized(hub, monkeypatch):
    real = acm._scan_downloaded
    calls = []

    def scan(task):
        found = real(task)
        calls.append(task)
        if len(calls) == 1:
            # A download completes while this scan is still reading the old cache.
            acm.forget()
        return found

    monkeypatch.setattr(acm, "_scan_downloaded", scan)
    acm.forget()
    acm.downloaded_models()
    acm.downloaded_models()
    assert len(calls) == 2
    acm.downloaded_models()
    assert len(calls) == 2


def _partial(hub, monkeypatch, tmp_path, keys):
    """Download state only (cancel marker and manifest) for planner keys of the umbrella repo."""
    from hub.utils import download_manifest
    for key in keys:
        assert download_manifest.write_cancel_marker("model", AUDIO_CPP_REPO, key, hub_cache = hub)
        download_manifest.write_manifest("model", AUDIO_CPP_REPO, key, [], hub_cache = hub)


def _states(hub):
    from hub.utils import download_manifest
    return sorted(
        {
            key
            for listing in (
                download_manifest.iter_variant_manifests("model", AUDIO_CPP_REPO, hub_cache = hub),
                download_manifest.iter_variant_markers("model", AUDIO_CPP_REPO, hub_cache = hub),
            )
            for key, _ in listing
        }
    )


QWEN_F16 = "Qwen3-TTS-12Hz-0.6B-Base-GGUF/qwen3-tts-12hz-0.6b-base-f16"
KOKORO_Q4 = "Kokoro-82M-GGUF/kokoro-82m-q4_0"


def test_a_folder_delete_also_clears_a_partial_only_sibling(hub, monkeypatch, tmp_path):
    from hub.services.models import deletion

    _idle(monkeypatch)
    _partial(hub, monkeypatch, tmp_path, [QWEN_F16, KOKORO_Q4])
    assert acm.cached_folder_target(QWEN, None) == (AUDIO_CPP_REPO, [QWEN_KEY, QWEN_F16])
    asyncio.run(deletion.delete_cached_model_response(QWEN, None))
    assert QWEN_FILE not in _left(hub)
    assert _states(hub) == [KOKORO_Q4]
    assert len(_left(hub)) == 2


def test_deleting_a_partial_only_quant_of_a_folder_clears_its_state(hub, monkeypatch, tmp_path):
    from hub.services.models import deletion

    _idle(monkeypatch)
    _partial(hub, monkeypatch, tmp_path, [QWEN_F16, KOKORO_Q4])
    assert acm.cached_folder_target(QWEN, "F16") == (AUDIO_CPP_REPO, [QWEN_F16])
    assert acm.cached_folder_target(QWEN, "Q4_0") == (AUDIO_CPP_REPO, [])
    asyncio.run(deletion.delete_cached_model_response(QWEN, "F16"))
    assert _states(hub) == [KOKORO_Q4]
    assert len(_left(hub)) == 3


def test_an_ambiguous_bare_quant_never_falls_back_to_download_state(hub, monkeypatch, tmp_path):
    moon = f"{AUDIO_CPP_REPO}/Moonshine-Streaming-GGUF"
    files = {
        f"Moonshine-Streaming-GGUF/moonshine-streaming-{size}-q8_0.gguf": "moonshine_asr"
        for size in ("tiny", "small")
    }
    _add_snapshot(hub, "a" * 40, files, main = True)
    medium = "Moonshine-Streaming-GGUF/moonshine-streaming-medium-q8_0"
    _partial(hub, monkeypatch, tmp_path, [medium])
    acm.forget()
    assert acm.cached_folder_target(moon, "Q8_0") == (AUDIO_CPP_REPO, [])
    assert medium in acm.cached_folder_target(moon, None)[1]


def test_a_state_only_size_is_named_even_when_another_size_shares_its_quant(
    hub, monkeypatch, tmp_path
):
    moon = f"{AUDIO_CPP_REPO}/Moonshine-Streaming-GGUF"
    tiny = "Moonshine-Streaming-GGUF/moonshine-streaming-tiny-q8_0"
    _add_snapshot(hub, "a" * 40, {f"{tiny}.gguf": "moonshine_asr"}, main = True)
    medium = "Moonshine-Streaming-GGUF/moonshine-streaming-medium-q8_0"
    _partial(hub, monkeypatch, tmp_path, [medium])
    acm.forget()
    assert acm.cached_folder_target(moon, "medium/Q8_0") == (AUDIO_CPP_REPO, [medium])
    assert acm.cached_folder_target(moon, "tiny/Q8_0") == (AUDIO_CPP_REPO, [tiny])
    # One size on disk: a bare quant still names it, not the cancelled one.
    assert acm.cached_folder_target(moon, "Q8_0") == (AUDIO_CPP_REPO, [tiny])


def test_cached_path_of_a_folder_skips_a_newer_snapshot_with_only_an_adapter(hub):
    newer = _add_snapshot(hub, "c" * 40, {}, main = True)
    (newer / "Qwen3-TTS-12Hz-0.6B-Base-GGUF").mkdir(parents = True)
    (newer / "Qwen3-TTS-12Hz-0.6B-Base-GGUF" / "mmproj-f16.gguf").write_bytes(b"GGUF")
    acm.forget()
    assert _path(QWEN, "")["path"] == str(hub / SNAPSHOT / "Qwen3-TTS-12Hz-0.6B-Base-GGUF")


def test_a_file_name_two_directories_share_names_neither(hub):
    variants = [
        acm.AudioCppVariant(f"{size}/Q8_0", (), f"Moon-GGUF/{size}/model-Q8_0.gguf")
        for size in ("tiny", "small")
    ]
    assert acm._cached_variants_named(variants, "model-Q8_0.gguf", "Moon-GGUF") == []
    assert acm._cached_variants_named(variants, "tiny/model-Q8_0.gguf", "Moon-GGUF") == [
        variants[0]
    ]
    assert acm._cached_variants_named(variants, "small/Q8_0", "Moon-GGUF") == [variants[1]]


def test_cached_path_of_a_folder_skips_a_newer_snapshot_missing_voice_files(hub):
    blobs = hub / "models--audio-cpp--audio.cpp-gguf" / "blobs"
    rel = "PocketTTS-GGUF/english/pocket-tts-english-q8_0.gguf"
    older = _add_snapshot(hub, "0" * 40, {rel: "pocket_tts"}, main = False)
    voice = blobs / "voice-english"
    voice.write_bytes(b"v" * 900)
    path = older / "PocketTTS-GGUF/english/embeddings/alba.safetensors"
    path.parent.mkdir(parents = True, exist_ok = True)
    os.symlink(os.path.relpath(voice, path.parent), path)
    # refs/main's snapshot got the GGUF again but not its voices.
    _add_snapshot(hub, "a" * 40, {rel: "pocket_tts"}, main = True)
    acm.forget()
    assert acm.folder_snapshot_dir(f"{AUDIO_CPP_REPO}/PocketTTS-GGUF") == older / "PocketTTS-GGUF"
