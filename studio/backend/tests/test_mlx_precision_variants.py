# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MLX repos that keep each precision in its own folder (#13418): listed like GGUF quants, the
picked folder fetched and loaded alone, and a download of the repo root skipping the others."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hub.services.models import gguf_variants
from hub.utils.snapshot_filters import (
    resolve_snapshot_ignore_patterns_for_files,
    snapshot_download_size,
)

GB = 1_000_000_000
_REPO = "orcarouter/Qwen3.8-27B-Uncensored-MLX"


def _folder(folder: str, shards: list[int]) -> dict[str, int]:
    prefix = f"{folder}/" if folder else ""
    files = {
        f"{prefix}config.json": 4_000,
        f"{prefix}tokenizer_config.json": 9_000,
        f"{prefix}model.safetensors.index.json": 90_000,
    }
    for index, size in enumerate(shards, 1):
        files[f"{prefix}model-{index:05d}-of-{len(shards):05d}.safetensors"] = size
    return files


# Shape of the repo in the issue: a root copy of the 4-bit, four precision folders and a drafter.
_SIZES = {
    **_folder("", [5 * GB, 5 * GB, 6 * GB]),
    **_folder("2-bit", [5 * GB, 4 * GB]),
    **_folder("4-bit", [5 * GB, 5 * GB, 6 * GB]),
    **_folder("6-bit", [5 * GB, 5 * GB, 5 * GB, 5 * GB, 3 * GB]),
    **_folder("8-bit", [5 * GB] * 6),
    "mtp/config.json": 2_000,
    "mtp/model.safetensors": GB,
    "README.md": 10_000,
}


def _siblings(sizes = _SIZES, content = None):
    """Hub siblings whose LFS hash stands for the file's bytes: by default the root and 4-bit/ hold
    the same files, as in the issue's repo."""
    content = content or (lambda name: name.replace("4-bit/", "", 1))
    return [
        SimpleNamespace(rfilename = name, size = size, lfs = {"sha256": f"sha-{content(name)}"})
        for name, size in sizes.items()
    ]


def test_precision_folders_are_whole_checkpoints_with_a_precision_name():
    from hub.utils.snapshot_filters import precision_variant_subfolders

    folders = precision_variant_subfolders(_SIZES)
    assert sorted(folders) == ["2-bit", "4-bit", "6-bit", "8-bit"]
    for name in ("4bit", "8bit", "mlx-4bit", "2bit-lite", "bf16", "MLX-8BIT", "mxfp4"):
        assert precision_variant_subfolders(
            {f"{name}/config.json": 1, f"{name}/model.safetensors": 1}
        )
    # Components, drafters and shards are not precisions; a folder without config or weights is no checkpoint.
    for name in ("mtp", "vae", "text_encoder", "transformer", "dflash", "layers-0-24", "merged-hf"):
        assert not precision_variant_subfolders(
            {f"{name}/config.json": 1, f"{name}/model.safetensors": 1}
        )
    assert not precision_variant_subfolders({"4-bit/model.safetensors": 1})
    assert not precision_variant_subfolders({"4-bit/config.json": 1, "4-bit/model.Q4_K_M.gguf": 1})
    assert not precision_variant_subfolders(
        {"4-bit/x/config.json": 1, "4-bit/x/model.safetensors": 1}
    )


def test_root_download_skips_the_other_precisions():
    ignore = resolve_snapshot_ignore_patterns_for_files(_SIZES)
    assert {"2-bit/*", "4-bit/*", "6-bit/*", "8-bit/*"} <= set(ignore)
    assert "mtp/*" not in ignore
    # 94 GB of every precision before, the 16 GB the root load reads (plus the drafter) after.
    assert snapshot_download_size(_siblings()) < 18 * GB
    # Without a root checkpoint the folders are the model, so nothing is skipped.
    no_root = {k: v for k, v in _SIZES.items() if "/" in k}
    assert not any(p.endswith("bit/*") for p in resolve_snapshot_ignore_patterns_for_files(no_root))
    # A stray root tensor without a root config.json is not a checkpoint either.
    stray = {**no_root, "model.safetensors": GB}
    assert not any(p.endswith("bit/*") for p in resolve_snapshot_ignore_patterns_for_files(stray))


def test_frontend_size_estimate_mirrors_the_precision_folder_rule():
    from hub.utils.snapshot_filters import PRECISION_FOLDER_SOURCE
    source = (
        Path(__file__).resolve().parents[2]
        / "frontend"
        / "src"
        / "features"
        / "hub"
        / "lib"
        / "dataset-size.ts"
    ).read_text(encoding = "utf-8")
    assert f'"{PRECISION_FOLDER_SOURCE}"' in source


def _fake_hub(
    monkeypatch,
    *,
    tags = ("mlx", "safetensors"),
    library = "mlx",
    sizes = _SIZES,
    content = None,
):
    calls = []

    class _Api:
        def __init__(self, token = None):
            pass

        def model_info(
            self,
            repo_id,
            files_metadata = False,
            timeout = None,
        ):
            calls.append(repo_id)
            return SimpleNamespace(
                tags = list(tags),
                library_name = library,
                pipeline_tag = "image-text-to-text",
                siblings = _siblings(sizes, content),
            )

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)
    # Nothing on disk, so no folder reads as downloaded.
    monkeypatch.setattr("utils.utils.hf_cache_snapshot_dir_for_repo", lambda repo: None)
    return calls


def _no_gguf_rows(monkeypatch, cache_authorized = True):
    empty = gguf_variants.VariantsAnswer(
        gguf_variants.GgufVariantsResponse(repo_id = _REPO), None, cache_authorized
    )

    async def _scan(key, compute):
        return empty

    monkeypatch.setattr(gguf_variants, "_shared_variants_scan", _scan)
    return empty


def test_mlx_repo_lists_its_precision_folders_as_variants(monkeypatch):
    _no_gguf_rows(monkeypatch)
    _fake_hub(monkeypatch)
    answer = asyncio.run(gguf_variants.get_gguf_variants_answer(_REPO))
    response = answer.response
    assert [(v.quant, v.filename) for v in response.variants] == [
        ("2-bit", "2-bit/"),
        ("4-bit", "4-bit/"),
        ("6-bit", "6-bit/"),
        ("8-bit", "8-bit/"),
    ]
    sizes = {v.quant: v.size_bytes for v in response.variants}
    assert sizes["6-bit"] == 23 * GB + 103_000
    assert sizes["8-bit"] == 30 * GB + 103_000
    # The root is a copy of the 4-bit, which is what a plain load of the repo serves today.
    assert response.default_variant == "4-bit"
    assert response.has_vision is True
    assert not any(v.downloaded for v in response.variants)
    assert response.variant_format == "mlx"
    # No GGUF metadata to read a context window from.
    assert answer.cache_authorized is False


def test_non_mlx_repo_and_repo_without_folders_get_no_mlx_listing(monkeypatch):
    empty = _no_gguf_rows(monkeypatch)
    _fake_hub(monkeypatch, tags = ("safetensors",), library = "transformers")
    assert asyncio.run(gguf_variants.get_gguf_variants_answer(_REPO)) is empty
    _fake_hub(monkeypatch, sizes = _folder("", [GB]))
    assert asyncio.run(gguf_variants.get_gguf_variants_answer(_REPO)) is empty


def test_gguf_listing_is_answered_without_asking_for_mlx_folders(monkeypatch):
    row = gguf_variants.GgufVariantDetail(filename = "m-Q4_K_M.gguf", quant = "Q4_K_M", size_bytes = 1)
    gguf = gguf_variants.VariantsAnswer(
        gguf_variants.GgufVariantsResponse(repo_id = "a/b-GGUF", variants = [row]), None
    )

    async def _scan(key, compute):
        return gguf

    class _NoHub:
        def __init__(self, token = None):
            raise AssertionError("a GGUF repo must not be listed as MLX")

    monkeypatch.setattr(gguf_variants, "_shared_variants_scan", _scan)
    monkeypatch.setattr("huggingface_hub.HfApi", _NoHub)
    assert asyncio.run(gguf_variants.get_gguf_variants_answer("a/b-GGUF")) is gguf


def test_download_fetches_only_the_picked_folder(monkeypatch, tmp_path):
    from hub.utils import mlx_variants

    snapshot = tmp_path / "snapshots" / "abc"
    calls = []

    def _snapshot_download(
        repo_id,
        allow_patterns = None,
        token = None,
        cache_dir = None,
    ):
        calls.append((repo_id, allow_patterns))
        (snapshot / "6-bit").mkdir(parents = True, exist_ok = True)
        (snapshot / "6-bit" / "config.json").write_text("{}")
        return str(snapshot)

    monkeypatch.setattr(mlx_variants, "cached_precision_folder", lambda repo, folder: None)
    monkeypatch.setattr("huggingface_hub.snapshot_download", _snapshot_download)
    path = mlx_variants.download_precision_folder(_REPO, "6-bit", None)
    assert path == str(snapshot / "6-bit")
    assert calls == [(_REPO, ["6-bit/*"])]

    # A complete cached folder is reused without asking the Hub, so a repeat load is a no-op.
    monkeypatch.setattr(
        mlx_variants, "cached_precision_folder", lambda repo, folder: snapshot / folder
    )
    assert mlx_variants.download_precision_folder(_REPO, "6-bit", None) == str(snapshot / "6-bit")
    assert len(calls) == 1


def test_cached_precision_folder_needs_every_shard(monkeypatch, tmp_path):
    from hub.utils import mlx_variants

    folder = tmp_path / "8-bit"
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    index = {
        "weight_map": {
            "a": "model-00001-of-00002.safetensors",
            "b": "model-00002-of-00002.safetensors",
        }
    }
    (folder / "model.safetensors.index.json").write_text(json.dumps(index))
    (folder / "model-00001-of-00002.safetensors").write_bytes(b"x")
    monkeypatch.setattr("utils.utils.hf_cache_snapshot_dir_for_repo", lambda repo: tmp_path)
    assert mlx_variants.cached_precision_folder(_REPO, "8-bit") is None
    (folder / "model-00002-of-00002.safetensors").write_bytes(b"x")
    assert mlx_variants.cached_precision_folder(_REPO, "8-bit") == folder
    # Not a precision name, so never a lookup.
    assert mlx_variants.cached_precision_folder(_REPO, "../8-bit") is None


@pytest.mark.parametrize("variant", ["Q4_K_M", "UD-Q4_K_XL", "Q8_0", "../6-bit", "6-bit/../x"])
def test_gguf_quant_names_never_resolve_as_folders(monkeypatch, variant):
    from hub.utils import mlx_variants

    reads = []
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: reads.append(a))
    assert mlx_variants.precision_folder_config_dir(_REPO, variant, None) is None
    assert reads == []


def test_model_config_carries_the_folder_and_classifies_it(monkeypatch, tmp_path):
    from utils.models import model_config

    folder = tmp_path / "6-bit"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "vision_config": {}}))
    checked = []
    fetched = []

    def _hf_hub_download(
        repo_id,
        filename,
        token = None,
        cache_dir = None,
        **_,
    ):
        fetched.append(filename)
        if filename != "6-bit/config.json":
            raise FileNotFoundError(filename)
        return str(folder / "config.json")

    # Only the folder's config.json is fetched to classify it; its weights wait for the load.
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _hf_hub_download)
    monkeypatch.setattr("utils.utils.hf_cache_snapshot_dir_for_repo", lambda repo: None)
    monkeypatch.setattr(model_config, "detect_gguf_model_remote", lambda *a, **k: None)
    monkeypatch.setattr(
        model_config, "is_vision_model", lambda name, hf_token = None: checked.append(name) or True
    )
    monkeypatch.setattr(model_config, "detect_audio_type", lambda *a, **k: None)
    monkeypatch.setattr(model_config, "is_model_cached", lambda *a, **k: False)

    config = model_config.ModelConfig.from_identifier(_REPO, gguf_variant = "6-bit")
    assert config.identifier == _REPO
    assert config.gguf_variant == "6-bit"
    assert config.is_gguf is False
    assert checked == [str(folder)]
    assert fetched == ["6-bit/config.json"]

    checked.clear()
    plain = model_config.ModelConfig.from_identifier(_REPO)
    assert plain.gguf_variant is None
    assert checked == [_REPO]


def _cached_repo(monkeypatch, tmp_path, repo, folder):
    """``repo`` in a private Hub cache with only ``folder/`` downloaded, as a precision pick leaves it."""
    from utils import hf_cache_settings

    hub = tmp_path / "hub"
    repo_dir = hub / f"models--{repo.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / ("a" * 40)
    (snapshot / folder).mkdir(parents = True)
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text("a" * 40)
    monkeypatch.setattr(
        hf_cache_settings,
        "get_hf_cache_paths",
        lambda: hf_cache_settings.HuggingFaceCachePaths(tmp_path, hub, tmp_path / "xet", "studio"),
    )
    return snapshot / folder


def test_a_downloaded_precision_folder_loads_offline(monkeypatch, tmp_path):
    from utils.models import model_config

    folder = _cached_repo(monkeypatch, tmp_path, _REPO, "6-bit")
    (folder / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "quantization": {"bits": 6}})
    )
    (folder / "model.safetensors").write_bytes(b"x")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def _no_hub(*args, **kwargs):
        raise AssertionError("an offline load of a downloaded folder must not reach the Hub")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _no_hub)
    monkeypatch.setattr("huggingface_hub.snapshot_download", _no_hub)
    monkeypatch.setattr(model_config, "is_model_cached", lambda *a, **k: True)

    config = model_config.ModelConfig.from_identifier(_REPO, gguf_variant = "6-bit")
    assert config.identifier == _REPO
    assert config.gguf_variant == "6-bit"
    assert config.is_gguf is False

    from hub.utils.mlx_variants import download_precision_folder

    assert download_precision_folder(_REPO, "6-bit", None) == str(folder)


def test_the_download_manager_plans_a_precision_folder(monkeypatch):
    # The chat picker and the Hub stage an uncached pick through the download manager, which only
    # knew GGUF variants and exited with "No GGUF shards matching variant '6-bit'".
    from hub.workers import hf_download

    monkeypatch.setattr(
        hf_download,
        "_model_info_with_retry",
        lambda repo_id, token: SimpleNamespace(siblings = _siblings()),
    )
    plan = hf_download._gguf_variant_target_plan(_REPO, "6-bit", None)
    assert plan is not None
    assert all(name.startswith("6-bit/") for name in plan.target_filenames)
    assert plan.download_size_bytes == 23 * GB + 103_000
    assert hf_download._gguf_variant_target_plan(_REPO, "mtp", None) is None


def test_a_gguf_variant_keeps_its_own_plan():
    from hub.utils.gguf_plan import resolve_variant_plan

    siblings = _siblings({"m-Q4_K_M.gguf": 5 * GB, "m-Q8_0.gguf": 9 * GB, **_folder("4-bit", [GB])})
    plan = resolve_variant_plan(siblings, "Q4_K_M")
    assert plan.target_filenames == ("m-Q4_K_M.gguf",)


def test_a_failed_folder_probe_never_falls_back_to_the_root(monkeypatch):
    from huggingface_hub.errors import LocalEntryNotFoundError

    from hub.utils import mlx_variants

    monkeypatch.setattr("utils.utils.hf_cache_snapshot_dir_for_repo", lambda repo: None)

    def _raise(error):
        def _download(*args, **kwargs):
            raise error

        return _download

    # A Hub that answered "no such file": the repo has no such folder.
    from huggingface_hub.errors import EntryNotFoundError

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _raise(EntryNotFoundError("404")))
    assert mlx_variants.precision_folder_config_dir(_REPO, "6-bit", None) is None
    # Offline or unreachable: loading the root instead would serve another precision.
    for error in (LocalEntryNotFoundError("offline"), ConnectionError("reset")):
        monkeypatch.setattr("huggingface_hub.hf_hub_download", _raise(error))
        with pytest.raises(type(error)):
            mlx_variants.precision_folder_config_dir(_REPO, "6-bit", None)


def test_a_rejected_token_still_downloads_a_public_folder(monkeypatch, tmp_path):
    from hub.utils import mlx_variants

    tokens = []

    def _snapshot_download(
        repo_id,
        allow_patterns = None,
        token = None,
        cache_dir = None,
    ):
        tokens.append(token)
        if token == "expired":
            raise PermissionError("401 Client Error: Unauthorized")
        (tmp_path / "6-bit").mkdir(exist_ok = True)
        (tmp_path / "6-bit" / "config.json").write_text("{}")
        return str(tmp_path)

    def _retry(read, hf_token):
        try:
            return read(hf_token)
        except PermissionError:
            return read(False)

    monkeypatch.setattr("utils.utils.hf_cache_snapshot_dir_for_repo", lambda repo: None)
    monkeypatch.setattr("huggingface_hub.snapshot_download", _snapshot_download)
    monkeypatch.setattr("hub.utils.hf_tokens.call_with_anonymous_retry", _retry)
    assert mlx_variants.download_precision_folder(_REPO, "6-bit", "expired") == str(
        tmp_path / "6-bit"
    )
    assert tokens == ["expired", False]


def test_the_root_duplicate_is_found_by_content_not_size(monkeypatch):
    # 4-bit/ and 4bit-dwq/ are the same size; only the one holding the root's bytes is its copy.
    sizes = {**_SIZES, **_folder("4bit-dwq", [5 * GB, 5 * GB, 6 * GB])}
    _no_gguf_rows(monkeypatch)
    _fake_hub(monkeypatch, sizes = sizes)
    response = asyncio.run(gguf_variants.get_gguf_variants_answer(_REPO)).response
    assert response.default_variant == "4-bit"
    # Without a folder holding the root's bytes there is no default, whatever the sizes say.
    _fake_hub(monkeypatch, sizes = sizes, content = lambda name: name)
    response = asyncio.run(gguf_variants.get_gguf_variants_answer(_REPO)).response
    assert response.default_variant is None


def test_a_resumed_folder_download_rebuilds_its_plan_from_the_manifest():
    # With the Hub unreachable the worker resumes from the saved manifest; a plan with no main
    # hashes made it refuse the partial download instead of resuming it.
    from hub.utils.gguf_plan import plan_from_expected_files, resolve_variant_plan

    siblings = _siblings({"m-Q4_K_M.gguf": 5 * GB, **_SIZES})
    folder = resolve_variant_plan(siblings, "6-bit")
    resumed = plan_from_expected_files("6-bit", folder.expected_files)
    assert resumed.main_hashes == folder.main_hashes != frozenset()
    assert resumed.target_filenames == folder.target_filenames
    gguf = resolve_variant_plan(siblings, "Q4_K_M")
    assert plan_from_expected_files("Q4_K_M", gguf.expected_files) == gguf


def test_memory_estimate_prices_the_picked_folder(monkeypatch, tmp_path):
    from routes import inference as ri

    folder = _cached_repo(monkeypatch, tmp_path, _REPO, "6-bit")
    (folder / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (folder / "model.safetensors").write_bytes(b"x")
    monkeypatch.setattr(ri, "_estimate_hf_cache_roots", lambda: [str(tmp_path / "hub")])
    config = SimpleNamespace(
        path = _REPO, identifier = _REPO, is_local = False, is_lora = False, is_gguf = False
    )
    # Only 6-bit/ is on disk: the root alone is not downloaded, the pick is.
    assert ri._local_mlx_model_dir(SimpleNamespace(**vars(config), gguf_variant = None)) is None
    assert ri._local_mlx_model_dir(SimpleNamespace(**vars(config), gguf_variant = "6-bit")) == str(
        folder
    )
    # With the root checkpoint there too, the pick still prices its own folder.
    (folder.parent / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (folder.parent / "model.safetensors").write_bytes(b"x")
    assert ri._local_mlx_model_dir(SimpleNamespace(**vars(config), gguf_variant = "6-bit")) == str(
        folder
    )


@pytest.mark.parametrize("resident_folder, expected", [(None, "reuse"), ("6-bit", "load")])
def test_a_root_request_never_reuses_a_resident_precision_folder(
    monkeypatch, resident_folder, expected
):
    from hub.services.models import account_access
    from models.inference import LoadRequest
    from routes import inference

    calls = []

    def _record(step):
        def _stop(*args, **kwargs):
            calls.append(step)
            raise RuntimeError(step)

        return _stop

    backend = SimpleNamespace(
        active_model_name = _REPO,
        models = {
            _REPO: {
                "engine": "auto",
                "gguf_variant": resident_folder,
                "precision_folder": resident_folder,
            }
        },
        get_loading_model = lambda: None,
        loading_models = (),
        set_parallel_slots = _record("reuse"),
        load_model = _record("load"),
    )
    llama = SimpleNamespace(is_loaded = False, is_active = False, model_identifier = None)
    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: llama)
    monkeypatch.setattr(inference, "get_inference_backend", lambda: backend)
    monkeypatch.setattr(inference, "_peek_inference_backend", lambda: backend)
    monkeypatch.setattr(
        inference, "_resolve_model_identifier_for_request", lambda *a, **k: (_REPO, _REPO, False)
    )
    monkeypatch.setattr(account_access, "require_model_access", lambda *a, **k: None)
    # Every runtime setting matches, so only the resident folder can tell the two apart.
    for name in (
        "_mlx_runtime_settings_match",
        "_non_gguf_runtime_settings_match",
        "_resident_audio_placement_matches",
        "_resident_context_satisfies",
    ):
        monkeypatch.setattr(inference, name, lambda *a, **k: True)
    fastapi_request = SimpleNamespace(
        app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 1))
    )
    with pytest.raises(Exception):
        asyncio.run(
            inference._load_model_impl(
                LoadRequest(model_path = _REPO), fastapi_request, current_subject = "user"
            )
        )
    assert calls[:1] == [expected]


def test_a_caller_refused_the_cache_learns_nothing_from_it(monkeypatch, tmp_path):
    folder = _cached_repo(monkeypatch, tmp_path, _REPO, "6-bit")
    (folder / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (folder / "model.safetensors").write_bytes(b"x")
    monkeypatch.setattr("utils.utils.hf_cache_snapshot_dir_for_repo", lambda repo: folder.parent)
    _fake_hub(monkeypatch)
    monkeypatch.setattr("hub.utils.mlx_variants.cached_precision_folder", lambda repo, name: folder)
    _no_gguf_rows(monkeypatch)
    rows = asyncio.run(gguf_variants.get_gguf_variants_answer(_REPO)).response.variants
    assert [v.quant for v in rows if v.downloaded] == ["2-bit", "4-bit", "6-bit", "8-bit"]
    # The GGUF scan refused this caller the cache: the Hub listing stays, the on-device state goes.
    _no_gguf_rows(monkeypatch, cache_authorized = False)
    rows = asyncio.run(gguf_variants.get_gguf_variants_answer(_REPO)).response.variants
    assert rows and not any(v.downloaded for v in rows)
    # Offline, the cache is the only source, so nothing is listed.
    answer = asyncio.run(gguf_variants.get_gguf_variants_answer(_REPO, offline = True))
    assert not answer.response.variants


def test_offline_lists_a_cached_precision_folder(monkeypatch, tmp_path):
    from fastapi import HTTPException

    folder = _cached_repo(monkeypatch, tmp_path, _REPO, "6-bit")
    (folder / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (folder / "model.safetensors").write_bytes(b"x")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    answer = asyncio.run(gguf_variants.get_gguf_variants_answer(_REPO, offline = True))
    assert [(v.quant, v.downloaded) for v in answer.response.variants] == [("6-bit", True)]
    assert answer.response.variant_format == "mlx"
    # A repo with nothing usable cached still answers 404, as before.
    with pytest.raises(HTTPException) as missing:
        asyncio.run(gguf_variants.get_gguf_variants_answer("org/absent", offline = True))
    assert missing.value.status_code == 404
