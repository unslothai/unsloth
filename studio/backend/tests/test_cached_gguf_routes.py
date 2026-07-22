# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

# Keep this test runnable without optional logging deps.
if "structlog" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

import routes.models as models_route
from hub.services.models import gguf_variants as GV


def _repo(
    repo_id: str,
    files: list[SimpleNamespace],
    repo_path: Path,
    *,
    revisions: list[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        repo_id = repo_id,
        repo_type = "model",
        repo_path = repo_path,
        revisions = revisions or [SimpleNamespace(files = files)],
    )


def _file(
    name: str,
    size_on_disk: int,
    *,
    blob_path: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        file_name = name,
        size_on_disk = size_on_disk,
        blob_path = blob_path,
    )


def test_iter_gguf_paths_matches_extension_case_insensitively(tmp_path):
    nested = tmp_path / "snapshots" / "rev"
    nested.mkdir(parents = True)
    lower = nested / "Q4_K_M.gguf"
    upper = nested / "Q8_0.GGUF"
    other = nested / "README.md"
    lower.write_text("a")
    upper.write_text("b")
    other.write_text("c")

    result = sorted(path.name for path in models_route._iter_gguf_paths(tmp_path))

    assert result == ["Q4_K_M.gguf", "Q8_0.GGUF"]


def test_legacy_hf_scan_uses_snapshot_path_for_inactive_cache(tmp_path):
    repo = tmp_path / "models--Org--Model"
    snapshot = repo / "snapshots" / "revision"
    snapshot.mkdir(parents = True)

    [row] = models_route._scan_hf_cache(tmp_path, active_cache = False)

    assert row.model_id == "Org/Model"
    assert row.id == str(snapshot.resolve())
    assert row.path == str(snapshot.resolve())


def test_collect_local_models_scans_previous_cache(monkeypatch, tmp_path):
    active = tmp_path / "active"
    previous = tmp_path / "previous"
    active.mkdir()
    snapshot = previous / "models--Org--Previous" / "snapshots" / "revision"
    snapshot.mkdir(parents = True)

    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: active)
    monkeypatch.setattr("utils.paths.legacy_hf_cache_dir", lambda: tmp_path / "legacy")
    monkeypatch.setattr("utils.paths.hf_default_cache_dir", lambda: tmp_path / "default")
    monkeypatch.setattr("utils.paths.lmstudio_model_dirs", lambda: [])
    monkeypatch.setattr("utils.hf_cache_settings.known_hf_hub_caches", lambda: [active, previous])
    monkeypatch.setattr("storage.studio_db.list_scan_folders", lambda: [])

    rows = models_route.collect_local_models(tmp_path / "models")

    previous_row = next(row for row in rows if row.model_id == "Org/Previous")
    assert previous_row.id == str(snapshot.resolve())


def test_collect_local_models_prefers_complete_previous_copy(monkeypatch, tmp_path):
    active = tmp_path / "active"
    previous = tmp_path / "previous"
    active_partial = active / "models--Org--Model" / "blobs" / "abc.incomplete"
    active_partial.parent.mkdir(parents = True)
    active_partial.write_bytes(b"partial")
    snapshot = previous / "models--Org--Model" / "snapshots" / "revision"
    snapshot.mkdir(parents = True)
    (snapshot / "model.safetensors").write_bytes(b"complete")

    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: active)
    monkeypatch.setattr("utils.paths.legacy_hf_cache_dir", lambda: tmp_path / "legacy")
    monkeypatch.setattr("utils.paths.hf_default_cache_dir", lambda: tmp_path / "default")
    monkeypatch.setattr("utils.paths.lmstudio_model_dirs", lambda: [])
    monkeypatch.setattr(
        "utils.hf_cache_settings.known_hf_hub_caches",
        lambda: [active, previous],
    )
    monkeypatch.setattr("storage.studio_db.list_scan_folders", lambda: [])

    rows = models_route.collect_local_models(tmp_path / "models")

    [row] = [row for row in rows if row.model_id == "Org/Model"]
    assert row.id == str(snapshot.resolve())
    assert row.partial is False
    assert row.active_cache is False


def test_list_cached_gguf_includes_non_suffix_repo_when_cache_contains_gguf(monkeypatch, tmp_path):
    repo = _repo(
        "HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive",
        [_file("Q4_K_M.gguf", 5_000), _file("README.md", 10)],
        tmp_path / "models--HauhauCS--Gemma",
    )
    scan = SimpleNamespace(repos = [repo])

    monkeypatch.setattr(models_route, "_all_hf_cache_scans", lambda: [scan])

    result = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))

    assert result["cached"] == [
        {
            "repo_id": "HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive",
            "size_bytes": 5_000,
            "cache_path": str(repo.repo_path),
            "has_vision": False,
        }
    ]


def test_list_cached_gguf_matches_extension_case_insensitively(monkeypatch, tmp_path):
    repo = _repo(
        "Org/Model-Without-Suffix",
        [_file("Q8_0.GGUF", 7_000)],
        tmp_path / "models--Org--Model-Without-Suffix",
    )
    scan = SimpleNamespace(repos = [repo])

    monkeypatch.setattr(models_route, "_all_hf_cache_scans", lambda: [scan])

    result = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))

    assert result["cached"] == [
        {
            "repo_id": "Org/Model-Without-Suffix",
            "size_bytes": 7_000,
            "cache_path": str(repo.repo_path),
            "has_vision": False,
        }
    ]


def test_is_hidden_model_hides_validation_probe_everywhere():
    """Every picker (model list, local, cached GGUF, cached models) gates on
    _is_hidden_model, so hiding the probe here hides it in the search menu too.
    Cover both forms callers pass: the reconstructed repo id and the on-disk
    snapshot path."""
    assert models_route._is_hidden_model("ggml-org/models")
    assert models_route._is_hidden_model("ggml-org/models/tinyllamas/stories260K.gguf")
    assert models_route._is_hidden_model(
        None, "/hf/models--ggml-org--models/snapshots/abc/tinyllamas/stories260K.gguf"
    )
    # A Windows-style snapshot path must match too, even on a POSIX interpreter
    # (the filename check splits on both separators).
    assert models_route._is_hidden_model(
        r"C:\Users\u\.cache\huggingface\hub\models--ggml-org--models\snapshots\abc\tinyllamas\stories260K.gguf"
    )
    assert not models_route._is_hidden_model("unsloth/gemma-3-270m-it-GGUF")
    # The exact-filename needle must not hide a real repo that merely
    # references stories260K in its name.
    assert not models_route._is_hidden_model("user/stories260K-finetune-GGUF")


def test_is_hidden_model_matches_repo_ids_exactly(monkeypatch):
    """A custom embedder with a generic basename is hidden by EXACT repo-id
    match only, so unrelated cached repos that merely contain the basename stay
    visible. Regression: substring basename matching hid real chat models like
    ``user/model-chat`` from the On Device inventory."""
    from core.rag import config as rag_config

    monkeypatch.setattr(rag_config, "effective_embedding_model", lambda: "org/model")
    monkeypatch.setattr(rag_config, "effective_gguf_repo", lambda: "org/model-GGUF")

    # The exact embedder repo and its GGUF companion are hidden.
    assert models_route._is_hidden_model("org/model")
    assert models_route._is_hidden_model("org/model-GGUF")
    # Unrelated repos that merely contain "model" must NOT be hidden.
    assert not models_route._is_hidden_model("user/model-chat")
    assert not models_route._is_hidden_model("org/model-instruct")
    assert not models_route._is_hidden_model("acme/remodelled-chat")
    # The validation probe stays hidden regardless of embedder config.
    assert models_route._is_hidden_model("ggml-org/models")


def test_is_hidden_model_matches_repo_derived_local_paths(monkeypatch):
    """Match exact repo-derived cache and LM Studio paths."""
    from core.rag import config as rag_config

    monkeypatch.setattr(rag_config, "effective_embedding_model", lambda: "org/model")
    monkeypatch.setattr(rag_config, "effective_gguf_repo", lambda: "org/model-GGUF")

    assert models_route._is_hidden_model(
        "/cache/models--org--model/snapshots/abc/model.safetensors"
    )
    assert models_route._is_hidden_model(
        r"C:\Users\u\.cache\huggingface\hub\models--org--model-GGUF\snapshots\abc"
    )
    assert models_route._is_hidden_model("/lm-studio/org/model-GGUF/model-Q8_0.gguf")
    assert not models_route._is_hidden_model("/lm-studio/user/model-chat/model-Q8_0.gguf")
    assert not models_route._is_hidden_model("/cache/models--org--model-instruct")


def test_is_hidden_model_prefers_existing_relative_path(monkeypatch, tmp_path):
    """Prefer an existing relative path over repo-id syntax."""
    from core.rag import config as rag_config

    embedder = tmp_path / "models" / "embedder"
    embedder.mkdir(parents = True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rag_config, "effective_embedding_model", lambda: "models/embedder")
    monkeypatch.setattr(rag_config, "effective_gguf_repo", lambda: "org/embedder-GGUF")

    assert models_route._is_hidden_model(str(embedder))


def test_is_hidden_model_keeps_stale_default_embedder_hidden(monkeypatch):
    """Keep default embedders hidden after a settings change."""
    from core.rag import config as rag_config

    monkeypatch.setattr(rag_config, "effective_embedding_model", lambda: "org/custom")
    monkeypatch.setattr(rag_config, "effective_gguf_repo", lambda: "org/custom-GGUF")

    assert models_route._is_hidden_model("unsloth/bge-small-en-v1.5")
    assert models_route._is_hidden_model("unsloth/bge-small-en-v1.5-GGUF")
    assert models_route._is_hidden_model("/models/bge-small-en-v1.5")
    assert models_route._is_hidden_model("/models/bge-small-en-v1.5-F16.gguf")
    assert models_route._is_hidden_model(r"C:\models\bge-small-en-v1.5-Q8_0.gguf")
    # Repo IDs still use exact matching, and similar local basenames must have
    # a real separator after the static default name.
    assert not models_route._is_hidden_model("user/bge-small-en-v1.5-chat")
    assert not models_route._is_hidden_model("/models/bge-small-en-v1.50")


def test_is_hidden_model_keeps_env_default_hidden_after_override(monkeypatch):
    """A persisted override must not expose the deployment's env default."""
    from core.rag import config as rag_config

    monkeypatch.delenv("RAG_EMBED_GGUF_REPO", raising = False)
    monkeypatch.setattr(rag_config, "EMBEDDING_MODEL", "org/env-default")
    monkeypatch.setattr(rag_config, "effective_embedding_model", lambda: "org/custom")
    monkeypatch.setattr(rag_config, "effective_gguf_repo", lambda: "org/custom-GGUF")

    assert models_route._is_hidden_model("org/env-default")
    assert models_route._is_hidden_model("org/env-default-GGUF")
    assert models_route._is_hidden_model("org/custom")
    assert models_route._is_hidden_model("org/custom-GGUF")
    assert not models_route._is_hidden_model("org/env-default-chat")


def test_hidden_models_importable_without_heavy_model_stack():
    """The hub cache scanner imports ``is_hidden_model`` at module scope, so it
    must not drag in ``utils/models/__init__`` (the model-config + checkpoint
    stack). Verify in a clean interpreter that importing the helper touches
    neither ``utils.models`` nor those heavy submodules, and still classifies
    the probe."""
    import os
    import subprocess
    import textwrap

    backend = Path(__file__).resolve().parents[1]
    code = textwrap.dedent(
        """
        import sys

        class _Blocker:
            _blocked = (
                "utils.models",
                "utils.models.model_config",
                "utils.models.checkpoints",
            )

            def find_spec(self, name, path=None, target=None):
                if name in self._blocked:
                    raise ImportError("blocked heavy import: " + name)
                return None

        sys.meta_path.insert(0, _Blocker())
        from utils.hidden_models import is_hidden_model

        loaded = sorted(m for m in sys.modules if m.startswith("utils.models"))
        assert not loaded, loaded
        assert is_hidden_model("ggml-org/models") is True
        assert is_hidden_model("unsloth/gemma-3-270m-it-GGUF") is False
        print("HIDDEN_MODELS_IMPORT_OK")
        """
    )
    env = dict(os.environ, PYTHONPATH = str(backend))
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output = True,
        text = True,
        env = env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "HIDDEN_MODELS_IMPORT_OK" in proc.stdout


def test_list_cached_gguf_hides_llama_validation_probe(monkeypatch, tmp_path):
    """The ggml-org/models / stories260K install validation probe can land in
    the HF cache as a side effect of installing the prebuilt llama-server.
    It is not a chat model (it sorts smallest and would be auto-selected), so
    pickers must hide it while keeping real cached models."""
    probe = _repo(
        "ggml-org/models",
        [_file("tinyllamas/stories260K.gguf", 1_000)],
        tmp_path / "models--ggml-org--models",
    )
    real = _repo(
        "unsloth/gemma-3-270m-it-GGUF",
        [_file("gemma-3-270m-it-UD-Q4_K_XL.gguf", 200_000)],
        tmp_path / "models--unsloth--gemma-3-270m-it-GGUF",
    )
    monkeypatch.setattr(
        models_route, "_all_hf_cache_scans", lambda: [SimpleNamespace(repos = [probe, real])]
    )

    result = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))

    repo_ids = [c["repo_id"] for c in result["cached"]]
    assert "ggml-org/models" not in repo_ids
    assert "unsloth/gemma-3-270m-it-GGUF" in repo_ids


def test_list_cached_gguf_skips_repos_without_positive_gguf_size(monkeypatch, tmp_path):
    missing = _repo(
        "Org/ReadmeOnly",
        [_file("README.md", 10)],
        tmp_path / "models--Org--ReadmeOnly",
    )
    zero = _repo(
        "Org/ZeroSize",
        [_file("Q4_K_M.gguf", 0)],
        tmp_path / "models--Org--ZeroSize",
    )
    scan = SimpleNamespace(repos = [missing, zero])

    monkeypatch.setattr(models_route, "_all_hf_cache_scans", lambda: [scan])

    result = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))

    assert result["cached"] == []


def test_list_cached_gguf_keeps_largest_duplicate_repo_across_scans(monkeypatch, tmp_path):
    smaller = _repo(
        "Org/Dupe",
        [_file("Q4_K_M.gguf", 2_000)],
        tmp_path / "models--Org--Dupe-a",
    )
    larger = _repo(
        "org/dupe",
        [_file("Q4_K_M.gguf", 5_000), _file("Q6_K.gguf", 1_000)],
        tmp_path / "models--Org--Dupe-b",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [
            SimpleNamespace(repos = [smaller]),
            SimpleNamespace(repos = [larger]),
        ],
    )

    result = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))

    assert result["cached"] == [
        {
            "repo_id": "org/dupe",
            "size_bytes": 6_000,
            "cache_path": str(larger.repo_path),
            "has_vision": False,
        }
    ]


def test_list_cached_gguf_dedupes_shared_blobs_across_revisions(monkeypatch, tmp_path):
    shared = "blobs/shared-q4"
    repo = _repo(
        "Org/SharedBlobRepo",
        [],
        tmp_path / "models--Org--SharedBlobRepo",
        revisions = [
            SimpleNamespace(files = [_file("Q4_K_M.gguf", 5_000, blob_path = shared)]),
            SimpleNamespace(files = [_file("Q4_K_M.gguf", 5_000, blob_path = shared)]),
        ],
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [repo])],
    )

    result = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))

    assert result["cached"] == [
        {
            "repo_id": "Org/SharedBlobRepo",
            "size_bytes": 5_000,
            "cache_path": str(repo.repo_path),
            "has_vision": False,
        }
    ]


def test_list_cached_models_skips_non_suffix_repo_when_gguf_files_exist(monkeypatch, tmp_path):
    mixed = _repo(
        "Org/MixedRepo",
        [
            _file("Q4_K_M.gguf", 5_000),
            _file("model.safetensors", 10_000),
        ],
        tmp_path / "models--Org--MixedRepo",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [mixed])],
    )

    result = asyncio.run(models_route.list_cached_models(current_subject = "test-user"))

    assert result["cached"] == []


def test_list_cached_gguf_includes_mixed_repo_with_gguf_and_safetensors(monkeypatch, tmp_path):
    """Mixed repo still surfaces in cached-gguf as a GGUF download."""
    mixed = _repo(
        "Org/MixedRepo",
        [
            _file("Q4_K_M.gguf", 5_000),
            _file("model.safetensors", 10_000),
        ],
        tmp_path / "models--Org--MixedRepo",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [mixed])],
    )

    result = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))

    assert result["cached"] == [
        {
            "repo_id": "Org/MixedRepo",
            "size_bytes": 5_000,
            "cache_path": str(mixed.repo_path),
            "has_vision": False,
        }
    ]


def test_list_cached_gguf_handles_none_size_on_disk(monkeypatch, tmp_path):
    """``size_on_disk = None`` (partial download) is treated as zero, not a
    TypeError from ``sum()`` that wipes the response."""
    partial = _repo(
        "Org/PartialDownload",
        [_file("Q4_K_M.gguf", None), _file("Q6_K.gguf", 5_000)],
        tmp_path / "models--Org--PartialDownload",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [partial])],
    )

    result = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))

    assert result["cached"] == [
        {
            "repo_id": "Org/PartialDownload",
            "size_bytes": 5_000,
            "cache_path": str(partial.repo_path),
            "has_vision": False,
        }
    ]


def test_list_cached_gguf_skips_malformed_repo_without_wiping_response(monkeypatch, tmp_path):
    """One repo raising during classification must not poison the response."""

    class _ExplodingRepo:
        repo_id = "Org/Broken"
        repo_type = "model"
        repo_path = tmp_path / "models--Org--Broken"

        @property
        def revisions(self):
            raise RuntimeError("boom")

    healthy = _repo(
        "Org/Healthy",
        [_file("Q4_K_M.gguf", 5_000)],
        tmp_path / "models--Org--Healthy",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [_ExplodingRepo(), healthy])],
    )

    result = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))

    assert result["cached"] == [
        {
            "repo_id": "Org/Healthy",
            "size_bytes": 5_000,
            "cache_path": str(healthy.repo_path),
            "has_vision": False,
        }
    ]


def test_list_cached_gguf_skips_repo_with_only_mmproj_gguf(monkeypatch, tmp_path):
    """A repo whose only ``.gguf`` is an mmproj vision adapter is not a GGUF
    repo: mmproj is filtered out, leaving zero variants."""
    mmproj_only = _repo(
        "Org/MmprojOnly",
        [
            _file("mmproj-Q8_0.gguf", 5_000),
            _file("model.safetensors", 10_000),
        ],
        tmp_path / "models--Org--MmprojOnly",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [mmproj_only])],
    )

    result = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))

    assert result["cached"] == []


def test_list_cached_models_includes_repo_with_only_mmproj_gguf(monkeypatch, tmp_path):
    """A safetensors repo with an auxiliary mmproj adapter still surfaces in
    cached-models as a normal model."""
    mmproj_aux = _repo(
        "Org/MmprojAux",
        [
            _file("mmproj-Q8_0.gguf", 5_000),
            _file("model.safetensors", 10_000),
        ],
        tmp_path / "models--Org--MmprojAux",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [mmproj_aux])],
    )

    result = asyncio.run(models_route.list_cached_models(current_subject = "test-user"))

    assert result["cached"] == [{"repo_id": "Org/MmprojAux", "size_bytes": 15_000}]


def test_list_cached_gguf_includes_vision_repo_with_main_gguf_and_mmproj(monkeypatch, tmp_path):
    """A vision GGUF repo (main weight + mmproj) is a GGUF repo; reported size
    is the main weight only, since mmproj is filtered at classification."""
    vision_repo = _repo(
        "Org/VisionGguf",
        [
            _file("Q4_K_M.gguf", 5_000),
            _file("mmproj-Q8_0.gguf", 1_000),
        ],
        tmp_path / "models--Org--VisionGguf",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [vision_repo])],
    )

    result = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))

    assert result["cached"] == [
        {
            "repo_id": "Org/VisionGguf",
            "size_bytes": 5_000,
            "cache_path": str(vision_repo.repo_path),
            "has_vision": True,
        }
    ]


def _gfile(name: str, size: int, mtime: float) -> SimpleNamespace:
    """A cached file carrying a Hugging Face ``blob_last_modified`` timestamp."""
    return SimpleNamespace(
        file_name = name,
        size_on_disk = size,
        blob_path = None,
        blob_last_modified = mtime,
    )


def test_all_hf_cache_scans_uses_shared_inventory(monkeypatch, tmp_path):
    from hub.utils import inventory_scan

    active = SimpleNamespace(
        repos = [_repo("Org/Active", [_file("Q4_K_M.gguf", 5_000)], tmp_path / "active")]
    )

    monkeypatch.setattr(inventory_scan, "all_hf_cache_scans", lambda: [active])

    scans = models_route._all_hf_cache_scans()
    assert scans == [active]

    # End-to-end: the endpoint still returns the active cache's repo.
    monkeypatch.setattr(models_route, "_all_hf_cache_scans", lambda: [active])
    result = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))
    assert result["cached"] == [
        {
            "repo_id": "Org/Active",
            "size_bytes": 5_000,
            "cache_path": str(tmp_path / "active"),
            "has_vision": False,
        }
    ]


def test_list_cached_gguf_sorts_newest_first_grouping_by_latest_quant(monkeypatch, tmp_path):
    """Downloaded is ordered newest-first, and a multi-quant repo is placed by
    its most recently downloaded quant (``last_modified`` = newest quant)."""
    older = _repo(
        "Org/Older",
        [_gfile("Older-Q4_K_M.gguf", 5_000, 1_000.0)],
        tmp_path / "models--Org--Older",
    )
    newer = _repo(
        "Org/Newer",
        [
            _gfile("Newer-Q4_K_M.gguf", 5_000, 2_000.0),
            _gfile("Newer-Q8_0.gguf", 9_000, 3_000.0),  # newest quant in the repo
        ],
        tmp_path / "models--Org--Newer",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [older, newer])],
    )

    result = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))

    assert [c["repo_id"] for c in result["cached"]] == ["Org/Newer", "Org/Older"]
    assert result["cached"][0]["last_modified"] == 3_000.0
    assert result["cached"][1]["last_modified"] == 1_000.0


def test_list_cached_gguf_dedupe_keeps_newest_timestamp(monkeypatch, tmp_path):
    """Same repo in two caches with equal size keeps the newest last_modified,
    regardless of scan order."""
    older = _repo("org/dupe", [_gfile("dupe-Q4_K_M.gguf", 5_000, 1_000.0)], tmp_path / "a")
    newer = _repo("org/dupe", [_gfile("dupe-Q4_K_M.gguf", 5_000, 9_000.0)], tmp_path / "b")
    for scans in ([older, newer], [newer, older]):  # both orders
        monkeypatch.setattr(
            models_route,
            "_all_hf_cache_scans",
            lambda s = scans: [SimpleNamespace(repos = [s[0]]), SimpleNamespace(repos = [s[1]])],
        )
        result = asyncio.run(models_route.list_cached_gguf(current_subject = "t"))
        assert len(result["cached"]) == 1
        assert result["cached"][0]["last_modified"] == 9_000.0


def test_gguf_variants_mmproj_does_not_mark_quant_downloaded(monkeypatch, tmp_path):
    """The per-quant 'downloaded' flag is driven by the real weight file in a
    single snapshot; an mmproj vision adapter (matching a quant label) must
    not make that quant appear downloaded."""
    variants = [
        SimpleNamespace(
            filename = "model-Q4_K_M.gguf",
            quant = "Q4_K_M",
            display_label = None,
            size_bytes = 10_000,
        ),
        SimpleNamespace(
            filename = "model-F16.gguf",
            quant = "F16",
            display_label = None,
            size_bytes = 20_000,
        ),
    ]
    monkeypatch.setattr(
        GV,
        "list_gguf_variants",
        lambda repo_id, hf_token = None: (variants, True, []),
    )
    monkeypatch.setattr(
        GV,
        "_local_main_gguf_blobs_by_quant",
        lambda _repo_id, repo_cache_dir = None: {},
    )

    snap = tmp_path / "models--org--repo" / "snapshots" / "rev"
    snap.mkdir(parents = True)
    (snap / "model-Q4_K_M.gguf").write_bytes(b"x" * 10_000)  # real weight, fully present
    (snap / "mmproj-F16.gguf").write_bytes(b"y" * 20_000)  # mmproj adapter, label "F16"
    monkeypatch.setattr(GV, "iter_hf_cache_snapshots", lambda _repo_id, root = None: [snap])

    result = asyncio.run(
        models_route.get_gguf_variants(
            repo_id = "org/repo", hf_token = None, current_subject = "test-user"
        )
    )

    flags = {v.quant: v.downloaded for v in result.variants}
    assert flags["Q4_K_M"] is True
    assert flags["F16"] is False


def test_gguf_variants_route_scopes_local_probe_to_selected_cache(monkeypatch, tmp_path):
    snapshot = tmp_path / "inactive" / "models--org--repo" / "snapshots" / "rev"
    snapshot.mkdir(parents = True)
    calls = []

    async def scoped_variants(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return SimpleNamespace(
            repo_id = repo_id,
            variants = [],
            has_vision = False,
            default_variant = None,
        )

    context_calls = []
    monkeypatch.setattr(GV, "get_gguf_variants_response", scoped_variants)
    monkeypatch.setattr(
        models_route,
        "_read_native_context_length",
        lambda model, *, is_local: context_calls.append((model, is_local)) or 8192,
    )

    result = asyncio.run(
        models_route.get_gguf_variants(
            repo_id = "org/repo",
            prefer_local_cache = True,
            local_path = str(snapshot),
            hf_token = None,
            current_subject = "test-user",
        )
    )

    assert calls == [
        (
            "org/repo",
            {
                "prefer_local_cache": True,
                "local_path": str(snapshot),
                "hf_token": None,
            },
        )
    ]
    assert context_calls == [(str(snapshot), True)]
    assert result.context_length == 8192


def test_gguf_variants_ignore_big_endian_siblings(monkeypatch, tmp_path):
    siblings = [
        SimpleNamespace(rfilename = "model-Q4_K_M-be.gguf", size = 100),
        SimpleNamespace(rfilename = "model-Q4_K_M.gguf", size = 10),
    ]
    monkeypatch.setattr(
        GV,
        "list_gguf_variants",
        lambda repo_id, hf_token = None: (
            [
                SimpleNamespace(
                    filename = "model-Q4_K_M.gguf",
                    quant = "Q4_K_M",
                    display_label = None,
                    size_bytes = 10,
                )
            ],
            False,
            siblings,
        ),
    )
    monkeypatch.setattr(
        GV,
        "_local_main_gguf_blobs_by_quant",
        lambda _repo_id, repo_cache_dir = None: {},
    )

    snap = tmp_path / "models--org--repo" / "snapshots" / "rev"
    snap.mkdir(parents = True)
    (snap / "model-Q4_K_M.gguf").write_bytes(b"x" * 10)
    monkeypatch.setattr(GV, "iter_hf_cache_snapshots", lambda _repo_id, root = None: [snap])

    result = asyncio.run(
        models_route.get_gguf_variants(
            repo_id = "org/repo", hf_token = None, current_subject = "test-user"
        )
    )

    assert [(v.quant, v.filename, v.size_bytes, v.downloaded) for v in result.variants] == [
        ("Q4_K_M", "model-Q4_K_M.gguf", 10, True)
    ]


def test_gguf_variants_cached_big_endian_does_not_satisfy_variant(monkeypatch, tmp_path):
    variants = [
        SimpleNamespace(
            filename = "model-Q4_K_M.gguf",
            quant = "Q4_K_M",
            display_label = None,
            size_bytes = 10,
        ),
    ]
    monkeypatch.setattr(
        GV,
        "list_gguf_variants",
        lambda repo_id, hf_token = None: (variants, False, []),
    )
    monkeypatch.setattr(
        GV,
        "_local_main_gguf_blobs_by_quant",
        lambda _repo_id, repo_cache_dir = None: {},
    )

    snap = tmp_path / "models--org--repo" / "snapshots" / "rev"
    snap.mkdir(parents = True)
    (snap / "model-Q4_K_M-be.gguf").write_bytes(b"x" * 10)
    monkeypatch.setattr(GV, "iter_hf_cache_snapshots", lambda _repo_id, root = None: [snap])

    result = asyncio.run(
        models_route.get_gguf_variants(
            repo_id = "org/repo", hf_token = None, current_subject = "test-user"
        )
    )

    assert result.variants[0].downloaded is False


def test_legacy_gguf_progress_delegates_to_shared_service(monkeypatch):
    calls = []

    async def shared(repo_id, *, variant, expected_bytes, hf_token):
        calls.append((repo_id, variant, expected_bytes, hf_token))
        return {"downloaded_bytes": 10, "expected_bytes": 20, "progress": 0.5}

    monkeypatch.setattr(
        "hub.services.models.downloads.get_gguf_download_progress_response",
        shared,
    )

    result = asyncio.run(
        models_route.get_gguf_download_progress(
            repo_id = "org/repo",
            variant = "Q4_K_M",
            expected_bytes = 20,
            hf_token = "token",
            current_subject = "test-user",
        )
    )

    assert result["progress"] == 0.5
    assert calls == [("org/repo", "Q4_K_M", 20, "token")]


def test_legacy_model_progress_delegates_to_shared_service(monkeypatch):
    calls = []

    async def shared(repo_id, *, hf_token):
        calls.append((repo_id, hf_token))
        return {"downloaded_bytes": 10, "expected_bytes": 20, "progress": 0.5}

    monkeypatch.setattr(
        "hub.services.models.downloads.get_download_progress_response",
        shared,
    )

    result = asyncio.run(
        models_route.get_download_progress(
            repo_id = "org/repo",
            hf_token = "token",
            current_subject = "test-user",
        )
    )

    assert result["progress"] == 0.5
    assert calls == [("org/repo", "token")]


def test_legacy_delete_delegates_to_shared_service(monkeypatch):
    calls = []

    async def shared(repo_id, variant, hf_token, cache_path = None):
        calls.append((repo_id, variant, hf_token, cache_path))
        return {"status": "deleted", "repo_id": repo_id}

    monkeypatch.setattr(
        "hub.services.models.deletion.delete_cached_model_response",
        shared,
    )

    result = asyncio.run(
        models_route.delete_cached_model(
            repo_id = "org/repo",
            variant = None,
            cache_path = "/data/hf/hub",
            hf_token = "token",
            current_subject = "test-user",
        )
    )

    assert result == {"status": "deleted", "repo_id": "org/repo"}
    assert calls == [("org/repo", None, "token", "/data/hf/hub")]
