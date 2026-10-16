# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Mirror targets that are also a model's own artifact repo (GGUF / FP8 checkpoints at the root, the official
diffusers folders beside them): the opt-out must not rewrite them, and pipeline sizing must not count the quants."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.inference import diffusion_families as df
from core.inference.diffusion import DiffusionBackend


def test_the_new_targets_are_artifact_mirrors_and_the_old_full_mirrors_are_not():
    for repo in (
        "unsloth/Z-Image-GGUF",
        "unsloth/Qwen-Image-Edit-GGUF",
        "unsloth/Qwen-Image-Edit-2509-GGUF",
        "unsloth/Qwen-Image-Layered-GGUF",
        "unsloth/Qwen-Image-2.1-Turbo-FP8",
    ):
        assert df.mirror_hosts_artifacts(repo), repo
        assert df.mirror_hosts_artifacts(repo.lower()), repo
    for repo in (
        "unsloth/Z-Image-Turbo",
        "unsloth/FLUX.1-dev",
        "unsloth/Qwen-Image",
        "unsloth/Z-Image-Turbo-GGUF",
    ):
        assert not df.mirror_hosts_artifacts(repo), repo


def test_the_opt_out_keeps_an_artifact_repo_and_still_unmirrors_a_full_mirror(monkeypatch):
    """Under UNSLOTH_DIFFUSION_NO_MIRROR a pure mirror maps back to its upstream, but unsloth/Z-Image-GGUF holds the
    GGUF being loaded, which Tongyi-MAI/Z-Image does not have: rewriting it 404s the denoiser fetch."""
    monkeypatch.setenv("UNSLOTH_DIFFUSION_NO_MIRROR", "1")
    assert df.prefer_ungated_mirror("unsloth/Z-Image-GGUF") == "unsloth/Z-Image-GGUF"
    assert (
        df.prefer_ungated_mirror("unsloth/Qwen-Image-2.1-Turbo-FP8")
        == "unsloth/Qwen-Image-2.1-Turbo-FP8"
    )
    assert df.prefer_ungated_mirror("unsloth/Z-Image-Turbo") == "Tongyi-MAI/Z-Image-Turbo"
    assert df.prefer_ungated_mirror("Tongyi-MAI/Z-Image") == "Tongyi-MAI/Z-Image"


def _snapshot(root: Path, repo: str, files: dict[str, int]) -> None:
    repo_dir = root / f"models--{repo.replace('/', '--')}"
    snap = repo_dir / "snapshots" / "rev1"
    for name, size in files.items():
        p = snap / name
        p.parent.mkdir(parents = True, exist_ok = True)
        p.write_bytes(b"\0" * size)
    (repo_dir / "refs").mkdir(parents = True, exist_ok = True)
    (repo_dir / "refs" / "main").write_text("rev1")


def test_pipeline_sizing_of_an_artifact_repo_counts_only_the_diffusers_part(monkeypatch, tmp_path):
    _snapshot(
        tmp_path,
        "unsloth/Z-Image-GGUF",
        {
            "z-image-Q4_K_M.gguf": 5000,
            "z-image-Q8_0.gguf": 8000,
            "README.md": 10,
            "model_index.json": 7,
            "vae/diffusion_pytorch_model.safetensors": 300,
            "text_encoder/model.safetensors": 600,
        },
    )
    monkeypatch.setattr(
        DiffusionBackend,
        "_hub_cache_repo_dirs",
        staticmethod(lambda repo_id: [tmp_path / f"models--{repo_id.replace('/', '--')}"]),
    )
    assert DiffusionBackend._cache_bytes("unsloth/Z-Image-GGUF") == 13917
    assert DiffusionBackend._cache_bytes("unsloth/Z-Image-GGUF", pipeline_only = True) == 907


def test_a_direct_pipeline_pick_of_an_artifact_repo_is_still_refused():
    """The official folders inside unsloth/Qwen-Image-2.1-Turbo-FP8 are what a pick of Qwen/Qwen-Image-2.1-Turbo
    fetches; naming the FP8 repo itself as a pipeline stays refused, as it was before the folders arrived. The GGUF
    repos keep their own -GGUF refusal."""
    assert "unsloth/qwen-image-2.1-turbo-fp8" in df.prequant_only_repo_ids()
    backend = DiffusionBackend()
    with pytest.raises(ValueError, match = "pre-quantised"):
        backend.validate_load_request("unsloth/Qwen-Image-2.1-Turbo-FP8", model_kind = "pipeline")
    for repo in (
        "unsloth/Z-Image-GGUF",
        "unsloth/Qwen-Image-Edit-GGUF",
        "unsloth/Qwen-Image-Edit-2509-GGUF",
        "unsloth/Qwen-Image-Layered-GGUF",
    ):
        with pytest.raises(ValueError):
            backend.validate_load_request(repo, model_kind = "pipeline")
    # and the base pick itself still validates
    assert (
        backend.validate_load_request("Qwen/Qwen-Image-2.1-Turbo", model_kind = "pipeline").name
        == "qwen-image-2.1"
    )


def test_a_full_mirror_whose_upstream_name_ends_in_fp8_is_not_an_artifact_mirror(monkeypatch):
    """unsloth/ideogram-4-fp8 is a complete copy of ideogram-ai/ideogram-4-fp8 (the suffix is in the upstream name),
    so the opt-out still maps it back to the upstream, as for every full mirror."""
    assert not df.mirror_hosts_artifacts("unsloth/ideogram-4-fp8")
    assert df.mirror_repo("ideogram-ai/ideogram-4-fp8") == "unsloth/ideogram-4-fp8"
    monkeypatch.setenv("UNSLOTH_DIFFUSION_NO_MIRROR", "1")
    assert df.prefer_ungated_mirror("unsloth/ideogram-4-fp8") == "ideogram-ai/ideogram-4-fp8"
    # Every artifact mirror is a mirror target the table actually names.
    targets = {m.lower() for _u, m in df._MIRROR_PAIRS}
    assert df._ARTIFACT_MIRRORS <= targets


_FP8_REPO = "unsloth/Qwen-Image-2.1-Turbo-FP8"
_FP8_ROOT = {"qwen-image-2.1-turbo-fp8.safetensors": 4000, "README.md": 10}
_FOLDERS = {
    "model_index.json": 1,
    "vae/config.json": 2,
    "vae/diffusion_pytorch_model.safetensors": 300,
    "text_encoder/config.json": 2,
    "text_encoder/model.safetensors": 600,
}
_DENOISER = {
    "transformer/config.json": 2,
    "transformer/diffusion_pytorch_model.safetensors": 2000,
}


def _cache_repo(root: Path, repo: str, files: dict[str, int]):
    from types import SimpleNamespace

    _snapshot(root, repo, files)
    repo_dir = root / f"models--{repo.replace('/', '--')}"
    snap = repo_dir / "snapshots" / "rev1"
    if "model_index.json" in files:
        (snap / "model_index.json").write_text(
            '{"_class_name": "QwenImagePipeline", "transformer": ["diffusers", "QwenImageTransformer2DModel"]}'
        )
    listed = [
        SimpleNamespace(
            file_name = name.rsplit("/", 1)[-1],
            file_path = snap / name,
            blob_path = str(snap / name),
            size_on_disk = size,
        )
        for name, size in files.items()
    ]
    return SimpleNamespace(
        repo_id = repo,
        repo_type = "model",
        repo_path = repo_dir,
        revisions = [SimpleNamespace(files = listed, snapshot_path = snap, commit_hash = "rev1")],
    )


def _inventory_row(monkeypatch, tmp_path, repo: str, files: dict[str, int]) -> dict:
    from types import SimpleNamespace

    from hub.services.models import cache_inventory

    cached = _cache_repo(tmp_path, repo, files)
    monkeypatch.setattr(
        cache_inventory, "all_hf_cache_scans", lambda: [SimpleNamespace(repos = [cached])]
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan, "is_snapshot_partial", lambda *_a, **_k: False
    )
    monkeypatch.setattr(cache_inventory, "_cached_row_task", lambda *_a, **_k: "text-to-image")
    rows = [
        r
        for r in cache_inventory._scan_cached_models(active_hub_cache = tmp_path)
        if r["repo_id"] == repo
    ]
    assert len(rows) == 1
    return rows[0]


def _route_row(monkeypatch, tmp_path, repo: str, files: dict[str, int]) -> dict:
    from types import SimpleNamespace

    import routes.models as models_route

    cached = _cache_repo(tmp_path, repo, files)
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(models_route, "_cached_repo_task", lambda *_a, **_k: "text-to-image")
    monkeypatch.setattr(models_route, "_cached_repo_partial", lambda *_a, **_k: False)
    monkeypatch.setattr(models_route.account_access, "filter_model_rows", lambda rows: rows)
    rows = [
        r
        for r in models_route.cached_model_rows([SimpleNamespace(repos = [cached])])
        if r["repo_id"] == repo
    ]
    assert len(rows) == 1
    return rows[0]


@pytest.mark.parametrize("cached", ["companions", "full"])
@pytest.mark.parametrize("scan", [_inventory_row, _route_row])
def test_the_fp8_artifact_mirror_lists_as_its_checkpoint_not_as_a_pipeline(
    monkeypatch, tmp_path, scan, cached
):
    """The Turbo base's folders sit inside unsloth/Qwen-Image-2.1-Turbo-FP8, but a pipeline pick of that id is refused.
    Whatever part of them is cached, the row is the single-file checkpoint row (the shape the compatibility listing
    gave the FP8 repo before the folders arrived): visible and deletable, never offered as a pipeline load the backend
    rejects, never a companion-prefetch partial."""
    files = {**_FP8_ROOT, **_FOLDERS, **(_DENOISER if cached == "full" else {})}
    row = scan(monkeypatch, tmp_path, _FP8_REPO, files)
    assert row["single_file"] is True
    assert not row.get("partial")
    assert not row.get("companion_prefetch")
    assert row.get("artifact_kind") in (None, "unknown")
    assert row.get("load_id") in (None, row["repo_id"])
    if scan is _route_row:
        released = scan(monkeypatch, tmp_path / "released", _FP8_REPO, dict(_FP8_ROOT))
        assert {
            k: released.get(k) for k in ("task", "single_file", "partial", "artifact_kind")
        } == {k: row.get(k) for k in ("task", "single_file", "partial", "artifact_kind")}


@pytest.mark.parametrize("cached", ["companions", "full"])
def test_a_gguf_artifact_mirror_lists_only_its_gguf_row_which_counts_the_folders(
    monkeypatch, tmp_path, cached
):
    """A bf16 pick of Tongyi-MAI/Z-Image caches the base's folders inside unsloth/Z-Image-GGUF. Deletes are whole-repo
    or per-quant, so a second, non-GGUF row for those folders would delete the quants with them, and a pipeline pick
    of the -GGUF id is refused. The folders fold into the GGUF row instead, whose size counts them."""
    from types import SimpleNamespace

    from hub.services.models import cache_inventory

    repo = "unsloth/Z-Image-GGUF"
    files = {"z-image-Q4_K_M.gguf": 5000, **_FOLDERS, **(_DENOISER if cached == "full" else {})}
    cached_repo = _cache_repo(tmp_path, repo, files)
    monkeypatch.setattr(
        cache_inventory, "all_hf_cache_scans", lambda: [SimpleNamespace(repos = [cached_repo])]
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan, "is_snapshot_partial", lambda *_a, **_k: False
    )
    monkeypatch.setattr(cache_inventory, "_cached_row_task", lambda *_a, **_k: "text-to-image")
    assert [
        r
        for r in cache_inventory._scan_cached_models(active_hub_cache = tmp_path)
        if r["repo_id"] == repo
    ] == []
    gguf_rows = [
        r
        for r in cache_inventory._scan_cached_gguf(active_hub_cache = tmp_path)
        if r["repo_id"] == repo
    ]
    assert len(gguf_rows) == 1
    # model_index.json is rewritten with real JSON, so count every other file exactly and that one as present.
    assert gguf_rows[0]["size_bytes"] > sum(v for k, v in files.items() if k != "model_index.json")


@pytest.mark.parametrize("cached", ["companions", "full"])
def test_on_device_lists_an_artifact_mirror_as_its_checkpoint_only(tmp_path, cached):
    """The On Device scan applies the same rule: the GGUF repo shows only its GGUF row (folders counted), and the FP8
    repo's row is never a loadable or resumable pipeline. A snapshot with no folders is left as released builds
    listed it."""
    from hub.services.models import local_inventory

    extra = {**_FOLDERS, **(_DENOISER if cached == "full" else {})}
    gguf_files = {"z-image-Q4_K_M.gguf": 5000, **extra}
    _cache_repo(tmp_path / "gguf", "unsloth/Z-Image-GGUF", gguf_files)
    rows = local_inventory._scan_hf_cache(tmp_path / "gguf")
    assert [r.model_format for r in rows] == ["gguf"]
    assert rows[0].size_bytes > sum(v for k, v in gguf_files.items() if k != "model_index.json")

    _cache_repo(tmp_path / "fp8", _FP8_REPO, {**_FP8_ROOT, **extra})
    rows = local_inventory._scan_hf_cache(tmp_path / "fp8")
    assert rows and all(r.partial and r.companion_prefetch for r in rows)

    _cache_repo(tmp_path / "released", _FP8_REPO, dict(_FP8_ROOT))
    _cache_repo(tmp_path / "released-other", "unsloth/Some-Model-FP8", dict(_FP8_ROOT))
    released = local_inventory._scan_hf_cache(tmp_path / "released")
    baseline = local_inventory._scan_hf_cache(tmp_path / "released-other")
    assert [(r.model_format, r.partial, r.companion_prefetch) for r in released] == [
        (r.model_format, r.partial, r.companion_prefetch) for r in baseline
    ]


def test_load_progress_through_an_artifact_mirror_counts_only_the_pipeline_part(monkeypatch, tmp_path):
    """A bf16 Turbo load fetches only the diffusers folders of unsloth/Qwen-Image-2.1-Turbo-FP8, so expected_bytes
    counts only those. FP8 / INT8 checkpoints already cached at the repo root must not count toward progress, while
    the blob downloading right now must."""
    from core.inference.diffusion import _LoadingState

    pipeline = {
        "model_index.json": 7,
        "vae/diffusion_pytorch_model.safetensors": 300,
        "text_encoder/model.safetensors": 600,
    }
    # Sample renders and the root checkpoint are cached too; neither is a pipeline file the plan counted.
    _snapshot(
        tmp_path,
        _FP8_REPO,
        {
            **pipeline,
            "qwen-image-2.1-turbo-fp8.safetensors": 50_000,
            "samples/fp8_1.png": 20_000,
            "assets/preview.png": 20_000,
        },
    )
    blobs = tmp_path / f"models--{_FP8_REPO.replace('/', '--')}" / "blobs"
    blobs.mkdir(parents = True, exist_ok = True)
    (blobs / "abc.incomplete").write_bytes(b"\0" * 1_000)
    monkeypatch.setattr(
        DiffusionBackend,
        "_hub_cache_repo_dirs",
        staticmethod(lambda repo_id: [tmp_path / f"models--{repo_id.replace('/', '--')}"]),
    )
    backend = DiffusionBackend()
    base = "Qwen/Qwen-Image-2.1-Turbo"
    backend._loading = _LoadingState(
        repo_id = base, base_repo = base, fetch_repo = _FP8_REPO, expected_bytes = 10_000
    )
    progress = backend.load_progress()
    assert progress["phase"] == "downloading"
    assert progress["bytes_downloaded"] == sum(pipeline.values()) + 1_000


def test_on_device_keeps_a_gguf_cached_in_an_older_revision_than_the_folders(tmp_path):
    """An install that cached a quant before the diffusers folders were added, then a base pick that fetched those
    folders from the newer revision: the newest snapshot holds only folders, the quant sits in the older one. The GGUF
    row must survive, with the folders counted, rather than leave only a partial companion row."""
    from hub.services.models import local_inventory

    repo_dir = tmp_path / "models--unsloth--Z-Image-GGUF"
    old = repo_dir / "snapshots" / "rev-old"
    new = repo_dir / "snapshots" / "rev-new"
    (old).mkdir(parents = True)
    (old / "z-image-Q4_K_M.gguf").write_bytes(b"\0" * 5_000)
    for name, size in {**_FOLDERS, **_DENOISER}.items():
        p = new / name
        p.parent.mkdir(parents = True, exist_ok = True)
        p.write_bytes(b"\0" * size)
    (new / "model_index.json").write_text('{"_class_name": "ZImagePipeline"}')
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text("rev-new")
    import os

    os.utime(old, (1, 1))

    rows = local_inventory._scan_hf_cache(tmp_path)
    assert [r.model_format for r in rows] == ["gguf"]
    assert rows[0].size_bytes > 5_000 + sum(v for k, v in _DENOISER.items())
    assert not rows[0].partial


@pytest.mark.parametrize("repo", sorted(df._ARTIFACT_MIRRORS))
def test_every_artifact_mirror_is_refused_as_a_pipeline_pick(repo):
    """Its folders are its base's pipeline, reached by picking the base. An explicit pipeline pick of the repo id
    itself is refused, naming that base, for the -GGUF mirrors as for the FP8 one."""
    assert repo in df.prequant_only_repo_ids()
    with pytest.raises(ValueError, match = "pre-quantised") as excinfo:
        DiffusionBackend().validate_load_request(repo, model_kind = "pipeline")
    assert df.canonical_base(repo).lower() in str(excinfo.value).lower()
