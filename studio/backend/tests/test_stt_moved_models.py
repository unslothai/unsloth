# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Dictation models moved out of the active Hub cache: into a custom folder, or left behind by a Models folder change."""

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest

from core.inference import stt_ggml_sidecar, stt_mtmd_sidecar, stt_sidecar
from utils import hidden_models

_SHA = "a" * 40
_MODEL_ID = "qwen3-asr-1.7b"
_SPEC = stt_mtmd_sidecar.MTMD_STT_MODELS[_MODEL_ID]


@pytest.fixture
def caches(monkeypatch, tmp_path):
    """An empty active cache, no other known cache and no custom folder until a test registers one."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    active = tmp_path / "active"
    active.mkdir()
    known: list[Path] = [active]
    folders: list[dict] = []
    monkeypatch.setattr(stt_sidecar, "_active_hf_hub_cache", lambda: active)
    monkeypatch.setattr("utils.hf_cache_settings.known_hf_hub_caches", lambda: list(known))
    monkeypatch.setattr("storage.studio_db.list_scan_folders", lambda: list(folders))
    # Every test registers its folders before the first lookup, so only a memo from an earlier test can be stale.
    monkeypatch.setattr(stt_sidecar, "_custom_folder_dirs_memo", {}, raising = False)
    stt_mtmd_sidecar._forget_downloaded_probe()
    yield {"active": active, "known": known, "folders": folders}
    stt_mtmd_sidecar._forget_downloaded_probe()


def _hub_tree(hub: Path, repo: str, files: dict[str, bytes]) -> Path:
    repo_dir = hub / f"models--{repo.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / _SHA
    snapshot.mkdir(parents = True)
    for name, data in files.items():
        (snapshot / name).write_bytes(data)
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text(_SHA)
    return snapshot


def _asr_tree(hub: Path) -> Path:
    return _hub_tree(hub, _SPEC.repo, {_SPEC.model_file: b"GGUF", _SPEC.mmproj_file: b"GGUF"})


def _asr_pair(directory: Path, *, mmproj: bool = True) -> None:
    directory.mkdir(parents = True, exist_ok = True)
    (directory / _SPEC.model_file).write_bytes(b"GGUF")
    if mmproj:
        (directory / _SPEC.mmproj_file).write_bytes(b"GGUF")


def _paths():
    return stt_mtmd_sidecar._cached_model_paths(_MODEL_ID)


def test_cache_tree_moved_into_a_custom_folder_is_found(caches, tmp_path):
    folder = tmp_path / "custom"
    snapshot = _asr_tree(folder)
    caches["folders"].append({"path": str(folder), "recursive": False})
    assert _paths() == (str(snapshot / _SPEC.model_file), str(snapshot / _SPEC.mmproj_file))
    assert stt_mtmd_sidecar.is_model_downloaded(_MODEL_ID) is True


def test_cache_tree_under_a_registered_hf_home_is_found(caches, tmp_path):
    folder = tmp_path / "hf_home"
    snapshot = _asr_tree(folder / "hub")
    caches["folders"].append({"path": str(folder), "recursive": False})
    assert _paths()[0] == str(snapshot / _SPEC.model_file)


def test_previous_models_folder_still_answers(caches, tmp_path):
    previous = tmp_path / "previous" / "hub"
    snapshot = _asr_tree(previous)
    caches["known"].append(previous)
    assert _paths()[0] == str(snapshot / _SPEC.model_file)


def test_active_cache_wins_over_a_custom_folder_copy(caches, tmp_path):
    active_snapshot = _asr_tree(caches["active"])
    _asr_tree(tmp_path / "custom")
    caches["folders"].append({"path": str(tmp_path / "custom"), "recursive": False})
    assert _paths()[0] == str(active_snapshot / _SPEC.model_file)


@pytest.mark.parametrize("layout", ["root", "sub-folder"])
def test_loose_pair_in_a_custom_folder_is_found(caches, tmp_path, layout):
    folder = tmp_path / "custom"
    directory = folder if layout == "root" else folder / "asr"
    _asr_pair(directory)
    caches["folders"].append({"path": str(folder), "recursive": False})
    assert _paths() == (str(directory / _SPEC.model_file), str(directory / _SPEC.mmproj_file))


def test_loose_model_without_its_mmproj_is_not_downloaded(caches, tmp_path):
    _asr_pair(tmp_path / "custom", mmproj = False)
    caches["folders"].append({"path": str(tmp_path / "custom"), "recursive": False})
    assert _paths() is None
    assert stt_mtmd_sidecar.is_model_downloaded(_MODEL_ID) is False


@pytest.mark.parametrize("recursive", [True, False])
def test_deeper_loose_pair_follows_the_folder_recursive_setting(caches, tmp_path, recursive):
    folder = tmp_path / "custom"
    _asr_pair(folder / "audio" / "asr")
    caches["folders"].append({"path": str(folder), "recursive": recursive})
    assert (_paths() is not None) is recursive


def test_explicit_cache_keeps_the_single_root_download_check(caches, tmp_path):
    """The download worker verifies the files landed in the cache it wrote, not a copy elsewhere."""
    _asr_tree(tmp_path / "custom")
    _asr_pair(tmp_path / "loose")
    caches["folders"].extend(
        [
            {"path": str(tmp_path / "custom"), "recursive": False},
            {"path": str(tmp_path / "loose"), "recursive": False},
        ]
    )
    assert _paths() is not None
    assert (
        stt_mtmd_sidecar._cached_model_paths(_MODEL_ID, hub_cache = caches["active"], revision = _SHA)
        is None
    )
    assert stt_mtmd_sidecar._cached_model_paths(_MODEL_ID, hub_cache = caches["active"]) is None


def test_load_uses_the_moved_files(caches, tmp_path):
    _asr_pair(tmp_path / "custom")
    caches["folders"].append({"path": str(tmp_path / "custom"), "recursive": False})
    sidecar = stt_mtmd_sidecar.MtmdSttSidecar(keep_alive_seconds = 0)
    model, mmproj = sidecar._ensure_model_downloaded(_MODEL_ID)
    assert Path(model).parent == tmp_path / "custom"
    assert Path(mmproj).name == _SPEC.mmproj_file


def test_ggml_cache_tree_in_a_custom_folder_is_found(caches, tmp_path):
    repo = stt_ggml_sidecar.GGML_STT_REPOS["tiny"]
    filename = stt_ggml_sidecar.GGML_STT_MODELS["tiny"]
    snapshot = _hub_tree(tmp_path / "custom", repo, {filename: b"ggml"})
    caches["folders"].append({"path": str(tmp_path / "custom"), "recursive": False})
    assert stt_ggml_sidecar._cached_model_path("tiny") == str(snapshot / filename)
    assert stt_ggml_sidecar._cached_model_path("tiny", hub_cache = caches["active"]) is None


def test_transformers_whisper_in_a_previous_models_folder_is_found(caches, tmp_path):
    previous = tmp_path / "previous" / "hub"
    snapshot = _hub_tree(
        previous,
        stt_sidecar.STT_MODELS["tiny"],
        {
            "config.json": json.dumps({"model_type": "whisper"}).encode(),
            "preprocessor_config.json": b"{}",
            "tokenizer.json": b"{}",
            "model.safetensors": b"weights",
        },
    )
    assert stt_sidecar.is_model_downloaded("tiny") is False
    caches["known"].append(previous)
    assert stt_sidecar._find_complete_cached_snapshot("tiny") == snapshot.resolve()


def test_hidden_filenames_match_the_curated_mtmd_catalogue():
    expected = {
        name.lower()
        for spec in stt_mtmd_sidecar.MTMD_STT_MODELS.values()
        for name in (spec.model_file, spec.mmproj_file)
    }
    assert hidden_models._HIDDEN_STT_FILENAMES_LOWER == expected


def test_loose_asr_files_and_their_folder_are_hidden_from_chat(tmp_path):
    folder = tmp_path / "asr"
    _asr_pair(folder)
    assert hidden_models.is_hidden_model(str(folder / _SPEC.model_file)) is True
    assert hidden_models.is_hidden_model("C:\\Models\\" + _SPEC.model_file) is True
    assert hidden_models.is_hidden_model(str(folder)) is True


def test_custom_folder_walk_is_memoised_across_one_poll(caches, monkeypatch, tmp_path):
    calls = []
    caches["folders"].append({"path": str(tmp_path), "recursive": True})
    monkeypatch.setattr(
        "hub.services.models.local_inventory.nested_scan_roots",
        lambda path: calls.append(path) or [],
    )
    for _ in range(3):
        assert stt_sidecar._custom_folder_dirs() == [tmp_path]
    assert calls == [tmp_path]


def test_folder_with_another_gguf_stays_visible(tmp_path):
    folder = tmp_path / "mixed"
    _asr_pair(folder)
    (folder / "Qwen3-8B-Q4_K_M.gguf").write_bytes(b"GGUF")
    assert hidden_models.is_hidden_model(str(folder)) is False
    assert hidden_models.is_hidden_model(str(folder / "Qwen3-8B-Q4_K_M.gguf")) is False
    empty = tmp_path / "empty"
    empty.mkdir()
    assert hidden_models.is_hidden_model(str(empty)) is False


def test_folder_with_a_safetensors_model_beside_the_pair_stays_visible(tmp_path):
    folder = tmp_path / "chat-model"
    _asr_pair(folder)
    (folder / "config.json").write_text("{}")
    (folder / "model.safetensors").write_bytes(b"weights")
    assert hidden_models.is_hidden_model(str(folder)) is False


def test_payload_suffixes_match_the_model_hub_scan():
    from hub.services.models import local_inventory
    assert set(hidden_models._MODEL_PAYLOAD_SUFFIXES) == set(local_inventory._PAYLOAD_SUFFIXES)


def test_a_model_in_the_active_cache_never_reads_custom_folders(caches, monkeypatch):
    _asr_tree(caches["active"])
    reads = []
    monkeypatch.setattr("storage.studio_db.list_scan_folders", lambda: reads.append(1) or [])
    assert _paths() is not None
    assert reads == [], "a hit in the active cache read the custom folders"
    assert stt_sidecar._find_complete_cached_snapshot("tiny") is None
    assert reads == [1], "a miss reads them once"


def test_a_hung_custom_folder_is_skipped_within_the_hub_budget(caches, monkeypatch, tmp_path):
    from hub.services.models import local_inventory

    monkeypatch.setattr(local_inventory, "_CUSTOM_FOLDER_SCAN_TIMEOUT_S", 0.2)
    hung, good = tmp_path / "hung", tmp_path / "good"
    hung.mkdir()
    _asr_pair(good)
    caches["folders"].extend(
        [{"path": str(hung), "recursive": True}, {"path": str(good), "recursive": False}]
    )
    release = threading.Event()

    def nested(path):
        if path == hung:
            release.wait(10)
        return []

    monkeypatch.setattr(local_inventory, "nested_scan_roots", nested)
    try:
        started = time.monotonic()
        assert _paths()[0] == str(good / _SPEC.model_file)
        assert time.monotonic() - started < 2
        # Still stuck: the next lookup skips the folder without waiting again.
        monkeypatch.setattr(stt_sidecar, "_custom_folder_dirs_memo", {})
        started = time.monotonic()
        assert _paths()[0] == str(good / _SPEC.model_file)
        assert time.monotonic() - started < 0.15
    finally:
        release.set()
        local_inventory._custom_folder_scans[str(hung)][1].result(timeout = 5)


def test_one_accounts_custom_folder_is_not_anothers(monkeypatch, tmp_path):
    from storage import studio_db
    from utils.account_context import AccountContext, run_as

    alice = AccountContext("a" * 32, "alice")
    bob = AccountContext("b" * 32, "bob")
    active = tmp_path / "active"
    active.mkdir()
    monkeypatch.setattr(stt_sidecar, "_active_hf_hub_cache", lambda: active)
    monkeypatch.setattr("utils.hf_cache_settings.known_hf_hub_caches", lambda: [active])
    monkeypatch.setattr(stt_sidecar, "_custom_folder_dirs_memo", {}, raising = False)
    stt_mtmd_sidecar._forget_downloaded_probe()
    from utils.paths.storage_roots import workspace_root

    # A member may register only folders inside their own workspace.
    folder = run_as(alice, workspace_root) / "models"
    _asr_pair(folder)
    run_as(alice, studio_db.add_scan_folder, str(folder))
    try:
        assert run_as(alice, stt_mtmd_sidecar.is_model_downloaded, _MODEL_ID) is True
        # Both memos are warm for alice now; bob must not read them.
        assert run_as(bob, stt_mtmd_sidecar.is_model_downloaded, _MODEL_ID) is False
        assert run_as(bob, stt_mtmd_sidecar._cached_model_paths, _MODEL_ID) is None
    finally:
        stt_mtmd_sidecar._forget_downloaded_probe()


def test_stt_status_looks_up_downloads_off_the_event_loop(monkeypatch):
    import routes.inference as inference

    on_loop = []

    def probe(*_args):
        on_loop.append(threading.current_thread() is threading.main_thread())
        return False

    monkeypatch.setattr(stt_sidecar, "is_model_downloaded", probe)
    monkeypatch.setattr(stt_mtmd_sidecar, "is_model_downloaded", probe)
    monkeypatch.setattr(stt_ggml_sidecar, "_cached_model_path", lambda model_id: probe() or None)
    monkeypatch.setattr(inference, "_audio_cpp_runtime_status", lambda: {"available": False})
    monkeypatch.setattr(inference.account_access, "managed_account", lambda: False)
    response = asyncio.run(inference.stt_status(model = None, current_subject = "owner"))
    body = json.loads(response.body)
    assert body["mtmd"]["downloaded_models"] == [] and body["gguf"]["downloaded_models"] == []
    assert on_loop and not any(on_loop)
