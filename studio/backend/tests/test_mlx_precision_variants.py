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


def _siblings(sizes = _SIZES):
    return [SimpleNamespace(rfilename = name, size = size) for name, size in sizes.items()]


def test_precision_folders_are_whole_checkpoints_with_a_precision_name():
    from hub.utils.snapshot_filters import precision_variant_subfolders

    folders = precision_variant_subfolders(_SIZES)
    assert sorted(folders) == ["2-bit", "4-bit", "6-bit", "8-bit"]
    for name in ("4bit", "8bit", "mlx-4bit", "2bit-lite", "bf16", "MLX-8BIT", "q4"):
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
                siblings = _siblings(sizes),
            )

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)
    # Nothing on disk, so no folder reads as downloaded.
    monkeypatch.setattr("utils.utils.hf_cache_snapshot_dir_for_repo", lambda repo: None)
    return calls


def _no_gguf_rows(monkeypatch):
    empty = gguf_variants.VariantsAnswer(gguf_variants.GgufVariantsResponse(repo_id = _REPO), None)

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


@pytest.mark.parametrize("variant", ["Q4_K_M", "UD-Q4_K_XL", "../6-bit", "6-bit/../x"])
def test_gguf_quant_names_never_resolve_as_folders(monkeypatch, variant):
    from hub.utils import mlx_variants
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no Hub read")),
    )
    assert mlx_variants.precision_folder_config_dir(_REPO, variant, None) is None


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
