# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Curated side models (dictation, the FLUX.3 prompt upsampler, the mask segmenter) moved out of the active Hub cache:
into a custom folder, or left behind by a Models folder change."""

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest

from core.inference import (
    diffusion_segment,
    model_cache_search,
    stt_ggml_sidecar,
    stt_mtmd_sidecar,
    stt_sidecar,
)
from core.inference import flux3_prompt_upsampler as pup
from utils import hf_cache_settings, hidden_models

# Taken before any fixture replaces it.
_KNOWN_CACHE_CANDIDATES = hf_cache_settings.known_hf_hub_cache_candidates

_SHA = "a" * 40
_MODEL_ID = "qwen3-asr-1.7b"
_SPEC = stt_mtmd_sidecar.MTMD_STT_MODELS[_MODEL_ID]


def _forget_custom_folder_memos(monkeypatch):
    for name in ("_location_rows_memo", "_location_listings_memo"):
        monkeypatch.setattr(model_cache_search, name, {})


@pytest.fixture
def caches(monkeypatch, tmp_path):
    """An empty active cache, no other known cache and no custom folder until a test registers one."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    active = tmp_path / "active"
    active.mkdir()
    known: list[Path] = [active]
    folders: list[dict] = []
    roots = {"legacy": tmp_path / "legacy" / "hub", "default": tmp_path / "default" / "hub"}
    monkeypatch.setattr(stt_sidecar, "_active_hf_hub_cache", lambda: active)
    monkeypatch.setattr(
        "utils.hf_cache_settings.known_hf_hub_cache_candidates", lambda: list(known)
    )
    monkeypatch.setattr("hub.utils.paths.legacy_hf_cache_dir", lambda: roots["legacy"])
    monkeypatch.setattr("hub.utils.paths.hf_default_cache_dir", lambda: roots["default"])
    monkeypatch.setattr("storage.studio_db.list_scan_folders", lambda: list(folders))
    # Every test registers its folders before the first lookup, so only a memo from an earlier test can be stale.
    _forget_custom_folder_memos(monkeypatch)
    stt_mtmd_sidecar._forget_downloaded_probe()
    yield {"active": active, "known": known, "folders": folders, **roots}
    stt_mtmd_sidecar._forget_downloaded_probe()


def _register(
    caches,
    folder: Path,
    recursive: bool = False,
) -> None:
    caches["folders"].append({"path": str(folder), "recursive": recursive})


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


@pytest.mark.parametrize("layout", ["folder", "registered HF_HOME"])
def test_cache_tree_moved_into_a_custom_folder_is_found(caches, tmp_path, layout):
    folder = tmp_path / "custom"
    snapshot = _asr_tree(folder if layout == "folder" else folder / "hub")
    _register(caches, folder)
    assert _paths() == (str(snapshot / _SPEC.model_file), str(snapshot / _SPEC.mmproj_file))
    assert stt_mtmd_sidecar.is_model_downloaded(_MODEL_ID) is True


@pytest.mark.parametrize("root", ["previous Models folder", "legacy", "default"])
def test_other_known_caches_are_searched_like_chat_loads(caches, tmp_path, root):
    hub = tmp_path / "previous" / "hub" if root == "previous Models folder" else caches[root]
    snapshot = _asr_tree(hub)
    if root == "previous Models folder":
        caches["known"].append(hub)
    assert _paths()[0] == str(snapshot / _SPEC.model_file)


@pytest.mark.parametrize(
    "where, recursive, found",
    [
        ("", False, True),
        ("asr", False, True),
        ("audio/asr", True, True),
        # <publisher>/<model>/: the Model Hub's LM Studio scan lists it in any custom folder.
        ("audio/asr", False, True),
        ("a/b/asr", True, True),
        ("a/b/asr", False, False),
    ],
)
def test_loose_pair_is_found_where_the_model_hub_lists_loose_ggufs(
    caches, tmp_path, where, recursive, found
):
    folder = tmp_path / "custom"
    _asr_pair(folder / where)
    _register(caches, folder, recursive)
    expected = (str(folder / where / _SPEC.model_file), str(folder / where / _SPEC.mmproj_file))
    assert _paths() == (expected if found else None)


def test_loose_model_without_its_mmproj_is_not_downloaded(caches, tmp_path):
    _asr_pair(tmp_path / "custom", mmproj = False)
    _register(caches, tmp_path / "custom")
    assert _paths() is None
    assert stt_mtmd_sidecar.is_model_downloaded(_MODEL_ID) is False


def test_active_cache_wins_and_a_hit_there_reads_no_custom_folder(caches, monkeypatch, tmp_path):
    active_snapshot = _asr_tree(caches["active"])
    _asr_tree(tmp_path / "custom")
    reads = []
    monkeypatch.setattr(
        "storage.studio_db.list_scan_folders",
        lambda: reads.append(1) or [{"path": str(tmp_path / "custom"), "recursive": False}],
    )
    assert _paths()[0] == str(active_snapshot / _SPEC.model_file)
    assert reads == []
    assert stt_sidecar._find_complete_cached_snapshot("tiny") is None
    assert reads == [1]


def test_explicit_cache_keeps_the_single_root_download_check(caches, tmp_path):
    """The download worker verifies the files landed in the cache it wrote, not a copy elsewhere."""
    _asr_tree(tmp_path / "custom")
    _asr_pair(tmp_path / "loose")
    _register(caches, tmp_path / "custom")
    _register(caches, tmp_path / "loose")
    assert _paths() is not None
    assert (
        stt_mtmd_sidecar._cached_model_paths(_MODEL_ID, hub_cache = caches["active"], revision = _SHA)
        is None
    )
    assert stt_mtmd_sidecar._cached_model_paths(_MODEL_ID, hub_cache = caches["active"]) is None


def test_load_uses_the_moved_files(caches, tmp_path):
    _asr_pair(tmp_path / "custom")
    _register(caches, tmp_path / "custom")
    model, mmproj = stt_mtmd_sidecar.MtmdSttSidecar(keep_alive_seconds = 0)._ensure_model_downloaded(
        _MODEL_ID
    )
    assert Path(model).parent == tmp_path / "custom" and Path(mmproj).name == _SPEC.mmproj_file


def test_whisper_engines_find_moved_cache_trees(caches, monkeypatch, tmp_path):
    repo, filename = (
        stt_ggml_sidecar.GGML_STT_REPOS["tiny"],
        stt_ggml_sidecar.GGML_STT_MODELS["tiny"],
    )
    ggml = _hub_tree(tmp_path / "custom", repo, {filename: b"ggml"})
    _register(caches, tmp_path / "custom")
    assert stt_ggml_sidecar._cached_model_path("tiny") == str(ggml / filename)
    assert stt_ggml_sidecar._cached_model_path("tiny", hub_cache = caches["active"]) is None

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
    # A Models folder change reaches the location list within its 2 s memo; the test does not wait for it.
    _forget_custom_folder_memos(monkeypatch)
    assert stt_sidecar._find_complete_cached_snapshot("tiny") == snapshot.resolve()


def test_hidden_names_match_the_catalogue_and_the_model_hub_scan():
    from hub.services.models import local_inventory

    expected = {
        name.lower()
        for spec in stt_mtmd_sidecar.MTMD_STT_MODELS.values()
        for name in (spec.model_file, spec.mmproj_file)
    }
    assert hidden_models._HIDDEN_STT_FILENAMES_LOWER == expected
    assert set(hidden_models._MODEL_PAYLOAD_SUFFIXES) == set(local_inventory._PAYLOAD_SUFFIXES)


def test_loose_asr_files_and_their_folder_are_hidden_from_chat(tmp_path):
    folder = tmp_path / "asr"
    _asr_pair(folder)
    assert hidden_models.is_hidden_model(str(folder / _SPEC.model_file)) is True
    assert hidden_models.is_hidden_model("C:\\Models\\" + _SPEC.model_file) is True
    assert hidden_models.is_hidden_model(str(folder)) is True
    (tmp_path / "empty").mkdir()
    assert hidden_models.is_hidden_model(str(tmp_path / "empty")) is False


@pytest.mark.parametrize("other", [("Qwen3-8B-Q4_K_M.gguf",), ("config.json", "model.safetensors")])
def test_a_folder_holding_another_model_beside_the_pair_stays_visible(tmp_path, other):
    folder = tmp_path / "mixed"
    _asr_pair(folder)
    for name in other:
        (folder / name).write_bytes(b"{}")
    assert hidden_models.is_hidden_model(str(folder)) is False
    assert hidden_models.is_hidden_model(str(folder / other[-1])) is False


def test_a_status_poll_lists_a_custom_folder_once_and_probes_only_candidates(
    caches, monkeypatch, tmp_path
):
    """Every curated model missing: one bounded listing per location per poll window, then no thread or file probe."""
    from hub.services.models import local_inventory

    folder = tmp_path / "custom"
    for index in range(30):
        (folder / f"group{index // 10}" / f"model{index}").mkdir(parents = True)
    (folder / "group0" / "model0" / _SPEC.model_file).write_bytes(b"GGUF")  # no mmproj beside it
    _register(caches, folder, recursive = True)
    bounded, probes = [], []
    real_bounded = local_inventory.run_custom_folder_bounded_sync
    monkeypatch.setattr(
        local_inventory,
        "run_custom_folder_bounded_sync",
        lambda raw, work: bounded.append(raw) or real_bounded(raw, work),
    )
    monkeypatch.setattr(stt_mtmd_sidecar, "_pair_in", lambda *args: probes.append(args))
    for _ in range(2):
        stt_mtmd_sidecar._forget_downloaded_probe()
        assert not any(stt_sidecar.is_model_downloaded(m) for m in stt_sidecar.STT_MODELS)
        assert not any(
            stt_mtmd_sidecar.is_model_downloaded(m) for m in stt_mtmd_sidecar.MTMD_STT_MODELS
        )
        assert all(
            stt_ggml_sidecar._cached_model_path(m) is None for m in stt_ggml_sidecar.GGML_STT_MODELS
        )
    assert str(folder) in bounded and len(bounded) == len(set(bounded))
    assert probes == []


@pytest.mark.parametrize("stall", ["folder walk", "loose file check", "Hub cache probe"])
def test_a_hung_custom_folder_is_skipped_within_the_hub_budget(
    caches, monkeypatch, tmp_path, stall
):
    """A folder that blocks is given up within the budget and marked slow; the next folder is still searched."""
    from hub.services.models import local_inventory
    from utils.paths.scan_folder_health import STATUS_SLOW, scan_folder_status

    monkeypatch.setattr(local_inventory, "_CUSTOM_FOLDER_SCAN_TIMEOUT_S", 0.2)
    hung, good = tmp_path / "hung", tmp_path / "good"
    hung.mkdir()
    _asr_pair(good)
    _register(caches, hung, recursive = stall == "folder walk")
    _register(caches, good)
    release = threading.Event()

    def stall_on_hung(real):
        def wrapper(first, *args, **kwargs):
            if first == hung or kwargs.get("hub_cache") == hung:
                release.wait(10)
            return real(first, *args, **kwargs)

        return wrapper

    if stall == "folder walk":
        monkeypatch.setattr(local_inventory, "nested_scan_roots", stall_on_hung(lambda path: []))
    elif stall == "loose file check":
        _asr_pair(hung)
        monkeypatch.setattr(stt_mtmd_sidecar, "_pair_in", stall_on_hung(stt_mtmd_sidecar._pair_in))
    else:
        _hub_tree(hung, _SPEC.repo, {})  # incomplete, so the lookup probes its files
        monkeypatch.setattr(
            stt_mtmd_sidecar, "_cached_file", stall_on_hung(stt_mtmd_sidecar._cached_file)
        )
    try:
        started = time.monotonic()
        assert _paths()[0] == str(good / _SPEC.model_file)
        assert time.monotonic() - started < 2
        assert scan_folder_status(str(hung)) == STATUS_SLOW
        _forget_custom_folder_memos(monkeypatch)
        started = time.monotonic()
        assert _paths()[0] == str(good / _SPEC.model_file)
        assert time.monotonic() - started < 0.15
    finally:
        release.set()
        local_inventory._custom_folder_scans[str(hung)][1].result(timeout = 5)


@pytest.mark.parametrize("stall", ["listing", "probe"])
@pytest.mark.parametrize("model_in", ["another previous Models folder", "a custom folder"])
def test_a_hung_previous_models_folder_does_not_hide_other_locations(
    caches, monkeypatch, tmp_path, stall, model_in
):
    """A previous Models folder that hangs is given up within its own budget and marked slow; the rest are searched."""
    from hub.services.models import local_inventory
    from utils.paths.scan_folder_health import STATUS_SLOW, scan_folder_status

    monkeypatch.setattr(local_inventory, "_CUSTOM_FOLDER_SCAN_TIMEOUT_S", 0.2)
    hung = tmp_path / "hung" / "hub"
    _hub_tree(hung, _SPEC.repo, {})  # incomplete, so a lookup probes its files
    caches["known"].append(hung)
    if model_in == "a custom folder":
        _asr_pair(tmp_path / "good")
        _register(caches, tmp_path / "good")
        expected = str(tmp_path / "good" / _SPEC.model_file)
    else:
        healthy = tmp_path / "healthy" / "hub"
        expected = str(_asr_tree(healthy) / _SPEC.model_file)
        caches["known"].append(healthy)
    release = threading.Event()
    if stall == "listing":
        real = model_cache_search._build_cache_listing

        def build_cache_listing(root):
            if root == hung:
                release.wait(10)
            return real(root)

        monkeypatch.setattr(model_cache_search, "_build_cache_listing", build_cache_listing)
    else:
        real_file = stt_mtmd_sidecar._cached_file

        def cached_file(
            model_id,
            filename,
            *,
            hub_cache = None,
            revision = None,
        ):
            if hub_cache == hung:
                release.wait(10)
            return real_file(model_id, filename, hub_cache = hub_cache, revision = revision)

        monkeypatch.setattr(stt_mtmd_sidecar, "_cached_file", cached_file)
    try:
        started = time.monotonic()
        assert _paths()[0] == expected
        assert time.monotonic() - started < 2
        assert scan_folder_status(str(hung)) == STATUS_SLOW
        _forget_custom_folder_memos(monkeypatch)
        started = time.monotonic()
        assert _paths()[0] == expected
        assert time.monotonic() - started < 0.15
    finally:
        release.set()
        local_inventory._custom_folder_scans[str(hung)][1].result(timeout = 5)


def test_a_dead_stored_models_folder_does_not_hide_a_healthy_one(caches, monkeypatch, tmp_path):
    """A dead stored Models folder does not stop the healthy one from answering within the budget."""
    from hub.services.models import local_inventory

    monkeypatch.setattr(local_inventory, "_CUSTOM_FOLDER_SCAN_TIMEOUT_S", 0.2)
    dead, healthy = tmp_path / "dead", tmp_path / "healthy"
    snapshot = _asr_tree(healthy / "hub")
    stored = {hf_cache_settings.CACHE_HISTORY_SETTING_KEY: [str(dead), str(healthy)]}
    monkeypatch.setattr(
        "storage.studio_db.get_app_setting", lambda key, default = None: stored.get(key, default)
    )
    monkeypatch.setattr(
        "utils.hf_cache_settings.known_hf_hub_cache_candidates", _KNOWN_CACHE_CANDIDATES
    )
    release = threading.Event()
    real_canonical, real_listing = (
        hf_cache_settings._canonical,
        model_cache_search._build_cache_listing,
    )

    def canonical(path):
        if str(path).startswith(str(dead)):
            release.wait(10)
        return real_canonical(path)

    def build_cache_listing(root):
        if str(root).startswith(str(dead)):
            release.wait(10)
        return real_listing(root)

    monkeypatch.setattr(hf_cache_settings, "_canonical", canonical)
    monkeypatch.setattr(model_cache_search, "_build_cache_listing", build_cache_listing)
    try:
        started = time.monotonic()
        assert _paths()[0] == str(snapshot / _SPEC.model_file)
        assert time.monotonic() - started < 2
    finally:
        release.set()
        for key, (_, future) in list(local_inventory._custom_folder_scans.items()):
            if str(dead) in key:
                future.result(timeout = 5)


def test_known_caches_are_listed_without_resolving_them(monkeypatch):
    """Resolving a stored Models folder on a dead mount hangs, so the location list takes the paths as written."""
    from utils import hf_cache_settings

    monkeypatch.setattr(hf_cache_settings, "_stored_cache_home_value", lambda: "/mnt/dead/models")
    monkeypatch.setattr(hf_cache_settings, "_stored_path_values", lambda key: ["/mnt/dead/old"])
    real = hf_cache_settings._canonical

    def canonical(path):
        assert "/mnt/dead" not in str(path), f"resolved {path}"
        return real(path)

    monkeypatch.setattr(hf_cache_settings, "_canonical", canonical)
    candidates = [str(p) for p in hf_cache_settings.known_hf_hub_cache_candidates()]
    assert "/mnt/dead/models/hub" in candidates and "/mnt/dead/old/hub" in candidates


def test_a_known_cache_registered_as_a_custom_folder_is_listed_both_ways(caches, tmp_path):
    """Its repos-only listing as a known cache must not stand in for its listing as a custom folder."""
    previous = tmp_path / "previous" / "hub"
    previous.mkdir(parents = True)
    _asr_pair(previous / "asr")
    caches["known"].append(previous)
    _register(caches, previous)
    assert _paths() == (
        str(previous / "asr" / _SPEC.model_file),
        str(previous / "asr" / _SPEC.mmproj_file),
    )


def test_one_accounts_custom_folder_is_not_anothers(monkeypatch, tmp_path):
    from storage import studio_db
    from utils.account_context import AccountContext, run_as
    from utils.paths.storage_roots import workspace_root

    alice, bob = AccountContext("a" * 32, "alice"), AccountContext("b" * 32, "bob")
    active = tmp_path / "active"
    active.mkdir()
    monkeypatch.setattr(stt_sidecar, "_active_hf_hub_cache", lambda: active)
    monkeypatch.setattr("utils.hf_cache_settings.known_hf_hub_cache_candidates", lambda: [active])
    monkeypatch.setattr("hub.utils.paths.legacy_hf_cache_dir", lambda: tmp_path / "legacy")
    monkeypatch.setattr("hub.utils.paths.hf_default_cache_dir", lambda: tmp_path / "default")
    _forget_custom_folder_memos(monkeypatch)
    stt_mtmd_sidecar._forget_downloaded_probe()
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


def _record_thread(seen: list, answer):
    def probe(*_args, **_kwargs):
        seen.append(threading.current_thread() is threading.main_thread())
        return answer

    return probe


def test_stt_status_looks_up_downloads_off_the_event_loop(monkeypatch):
    import routes.inference as inference

    on_loop = []
    monkeypatch.setattr(stt_sidecar, "is_model_downloaded", _record_thread(on_loop, False))
    monkeypatch.setattr(stt_mtmd_sidecar, "is_model_downloaded", _record_thread(on_loop, False))
    monkeypatch.setattr(stt_ggml_sidecar, "_cached_model_path", _record_thread(on_loop, None))
    monkeypatch.setattr(inference, "_audio_cpp_runtime_status", lambda: {"available": False})
    monkeypatch.setattr(inference.account_access, "managed_account", lambda: False)
    body = json.loads(asyncio.run(inference.stt_status(model = None, current_subject = "owner")).body)
    assert body["mtmd"]["downloaded_models"] == [] and body["gguf"]["downloaded_models"] == []
    assert on_loop and not any(on_loop)


def test_openai_translations_checks_the_model_off_the_event_loop(monkeypatch):
    import routes.inference as inference
    from fastapi import HTTPException

    on_loop = []
    monkeypatch.setattr(stt_sidecar, "can_translate", _record_thread(on_loop, False))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            inference.openai_audio_translations(
                request = None,
                file = None,
                model = None,
                response_format = "json",
                provider_id = None,
                current_subject = "owner",
            )
        )
    assert exc.value.status_code == 400 and on_loop == [False]


_UD, _Q4 = pup.UPSAMPLER_MODEL_FILES["UD-Q4_K_XL"], pup.UPSAMPLER_MODEL_FILES["Q4_K_M"]


@pytest.fixture
def upsampler(caches, monkeypatch):
    for name in (pup.ENV_MODEL_OVERRIDE, pup.ENV_MMPROJ_OVERRIDE):
        monkeypatch.delenv(name, raising = False)
    return caches


def _upsampler_tree(
    hub: Path,
    model_file: str,
    *,
    mmproj: bool = True,
) -> Path:
    files = {model_file: b"GGUF", **({pup.UPSAMPLER_MMPROJ_FILE: b"GGUF"} if mmproj else {})}
    return _hub_tree(hub, pup.UPSAMPLER_REPO, files)


def test_upsampler_moved_into_a_custom_folder_is_found(upsampler, tmp_path):
    snapshot = _upsampler_tree(tmp_path / "custom", _UD)
    _register(upsampler, tmp_path / "custom")
    assert pup.model_path() == str(snapshot / _UD)
    assert pup.mmproj_path() == str(snapshot / pup.UPSAMPLER_MMPROJ_FILE)
    status = pup.status()
    assert status["downloaded"] is True and status["current_downloaded"] is True
    # The download's own check reads only the cache it wrote.
    assert (
        pup._cached_download_paths(pup.UPSAMPLER_ID, hub_cache = upsampler["active"], revision = _SHA)
        is None
    )


def test_upsampler_current_quant_elsewhere_beats_an_older_one_in_the_active_cache(
    upsampler, tmp_path
):
    _upsampler_tree(upsampler["active"], _Q4)
    previous = tmp_path / "previous" / "hub"
    snapshot = _upsampler_tree(previous, _UD)
    upsampler["known"].append(previous)
    assert pup.model_path() == str(snapshot / _UD)
    assert pup.status()["current_downloaded"] is True


def test_upsampler_download_size_counts_only_the_active_cache(upsampler, tmp_path):
    """The download writes to the active cache, so a copy elsewhere saves no bytes."""
    previous = tmp_path / "previous" / "hub"
    _upsampler_tree(previous, _UD, mmproj = False)
    upsampler["known"].append(previous)
    status = pup.status()
    assert status["downloaded"] is True and status["current_downloaded"] is False
    assert status["current_missing_bytes"] == (
        pup.UPSAMPLER_FILE_SIZES[_UD] + pup.UPSAMPLER_FILE_SIZES[pup.UPSAMPLER_MMPROJ_FILE]
    )


def test_upsampler_loose_files_are_not_matched_but_an_override_projector_is_searched(
    upsampler, monkeypatch, tmp_path
):
    """mmproj-F16.gguf is the name many models give their projector, so only a Hub cache counts."""
    loose = tmp_path / "loose"
    loose.mkdir()
    for name in (_UD, pup.UPSAMPLER_MMPROJ_FILE):
        (loose / name).write_bytes(b"GGUF")
    _register(upsampler, loose)
    assert pup.model_path() is None

    model = tmp_path / "local.gguf"
    model.write_bytes(b"GGUF")
    monkeypatch.setenv(pup.ENV_MODEL_OVERRIDE, str(model))
    snapshot = _upsampler_tree(tmp_path / "custom", _UD)
    _register(upsampler, tmp_path / "custom")
    _forget_custom_folder_memos(monkeypatch)
    assert pup.mmproj_path() == str(snapshot / pup.UPSAMPLER_MMPROJ_FILE)


@pytest.fixture
def segmenter(caches, monkeypatch):
    for name in (diffusion_segment.ENV_PATH_OVERRIDE, diffusion_segment.ENV_VARIANT):
        monkeypatch.delenv(name, raising = False)
    return caches


def _segmenter_tree(hub: Path, variant_id: str) -> Path:
    variant = diffusion_segment.SEGMENTER_VARIANTS[variant_id]
    return _hub_tree(hub, variant.repo, {name: b"{}" for name in variant.files})


def test_segmenter_moved_into_a_custom_folder_is_found(segmenter, tmp_path):
    snapshot = _segmenter_tree(tmp_path / "custom", diffusion_segment.DEFAULT_VARIANT)
    _register(segmenter, tmp_path / "custom")
    assert diffusion_segment.model_dir() == str(snapshot)
    # The download's own check reads only the cache it wrote.
    assert (
        diffusion_segment._cached_download_paths(
            diffusion_segment.DEFAULT_VARIANT, hub_cache = segmenter["active"], revision = _SHA
        )
        is None
    )


def test_segmenter_in_a_previous_models_folder_keeps_the_default_variant(segmenter, tmp_path):
    """The default is used whenever it is downloaded; another variant only stands in when the default is not."""
    _segmenter_tree(segmenter["active"], "sam3")
    previous = tmp_path / "previous" / "hub"
    snapshot = _segmenter_tree(previous, diffusion_segment.DEFAULT_VARIANT)
    segmenter["known"].append(previous)
    variant, path = diffusion_segment._resolve_active()
    assert variant.id == diffusion_segment.DEFAULT_VARIANT and path == str(snapshot)
