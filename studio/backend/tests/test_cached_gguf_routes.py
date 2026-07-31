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


def test_list_cached_gguf_reports_snapshot_load_id_for_inactive_cache(monkeypatch, tmp_path):
    """Only a repo outside the active cache needs a snapshot load_id."""
    active = tmp_path / "active"
    snapshot = tmp_path / "legacy" / "models--Org--Away" / "snapshots" / "rev"
    snapshot.mkdir(parents = True)
    (snapshot / "Q4_K_M.gguf").write_bytes(b"\0")
    away = _repo(
        "Org/Away",
        [],
        tmp_path / "legacy" / "models--Org--Away",
        revisions = [
            SimpleNamespace(files = [_file("Q4_K_M.gguf", 5_000)], snapshot_path = snapshot),
        ],
    )
    here = _repo("Org/Here", [_file("Q4_K_M.gguf", 6_000)], active / "models--Org--Here")

    monkeypatch.setattr(
        models_route, "_all_hf_cache_scans", lambda: [SimpleNamespace(repos = [away, here])]
    )
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: active)

    rows = {
        c["repo_id"]: c
        for c in asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))["cached"]
    }

    assert rows["Org/Away"]["load_id"] == str(snapshot)
    assert "load_id" not in rows["Org/Here"]


def test_list_cached_gguf_load_id_follows_snapshot_dir_mtime(monkeypatch, tmp_path):
    """Pick the snapshot variant discovery reads: newest directory, not newest blob."""
    import os

    active = tmp_path / "active"
    repo_dir = tmp_path / "legacy" / "models--Org--Multi"
    older, newer = repo_dir / "snapshots" / "rev-a", repo_dir / "snapshots" / "rev-b"
    for path in (older, newer):
        path.mkdir(parents = True)
    (older / "Q4_K_M.gguf").write_bytes(b"\0")
    (newer / "Q8_0.gguf").write_bytes(b"\0")
    os.utime(older, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))

    repo = _repo(
        "Org/Multi",
        [],
        repo_dir,
        revisions = [
            # The older directory holds the newer blob, which is what diverges.
            SimpleNamespace(
                files = [_file("Q4_K_M.gguf", 5_000, blob_path = "b1")], snapshot_path = older
            ),
            SimpleNamespace(files = [_file("Q8_0.gguf", 6_000, blob_path = "b2")], snapshot_path = newer),
        ],
    )

    monkeypatch.setattr(
        models_route, "_all_hf_cache_scans", lambda: [SimpleNamespace(repos = [repo])]
    )
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: active)
    monkeypatch.setattr(
        models_route, "_blob_mtime", lambda f: 9_000 if f.blob_path == "b1" else 1.0
    )

    rows = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))["cached"]

    assert rows[0]["load_id"] == str(newer)


def test_list_cached_gguf_load_id_skips_partial_split_snapshot(monkeypatch, tmp_path):
    """A half-downloaded split quant must not beat an older snapshot that can load."""
    import os

    active = tmp_path / "active"
    repo_dir = tmp_path / "legacy" / "models--Org--Split"
    older, newer = repo_dir / "snapshots" / "rev-a", repo_dir / "snapshots" / "rev-b"
    for path in (older, newer):
        path.mkdir(parents = True)
    (older / "Model-Q8_0.gguf").write_bytes(b"\0")
    # Only part 1 of 3 landed before the download was interrupted.
    (newer / "Model-Q4_K_M-00001-of-00003.gguf").write_bytes(b"\0")
    os.utime(older, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))

    repo = _repo(
        "Org/Split",
        [],
        repo_dir,
        revisions = [
            SimpleNamespace(files = [_file("Model-Q8_0.gguf", 5_000)], snapshot_path = older),
            SimpleNamespace(
                files = [_file("Model-Q4_K_M-00001-of-00003.gguf", 6_000)], snapshot_path = newer
            ),
        ],
    )

    monkeypatch.setattr(
        models_route, "_all_hf_cache_scans", lambda: [SimpleNamespace(repos = [repo])]
    )
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: active)

    rows = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))["cached"]

    assert rows[0]["load_id"] == str(older)


def test_list_cached_gguf_omits_load_id_when_no_snapshot_is_complete(monkeypatch, tmp_path):
    """With only a half-downloaded split quant, fall back to the repo id, not a path."""
    active = tmp_path / "active"
    repo_dir = tmp_path / "legacy" / "models--Org--Torn"
    snapshot = repo_dir / "snapshots" / "rev"
    snapshot.mkdir(parents = True)
    (snapshot / "Model-Q4_K_M-00001-of-00003.gguf").write_bytes(b"\0")

    repo = _repo(
        "Org/Torn",
        [],
        repo_dir,
        revisions = [
            SimpleNamespace(
                files = [_file("Model-Q4_K_M-00001-of-00003.gguf", 6_000)], snapshot_path = snapshot
            ),
        ],
    )

    monkeypatch.setattr(
        models_route, "_all_hf_cache_scans", lambda: [SimpleNamespace(repos = [repo])]
    )
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: active)

    rows = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))["cached"]

    assert "load_id" not in rows[0]


def test_list_cached_gguf_skips_snapshot_with_one_incomplete_variant(monkeypatch, tmp_path):
    """A good quant beside a half-downloaded one is still not a safe load target."""
    import os

    active = tmp_path / "active"
    repo_dir = tmp_path / "legacy" / "models--Org--Mixed"
    older, newer = repo_dir / "snapshots" / "rev-a", repo_dir / "snapshots" / "rev-b"
    for path in (older, newer):
        path.mkdir(parents = True)
    (older / "Model-Q8_0.gguf").write_bytes(b"\0")
    # rev-b has a complete Q8_0 AND a half-downloaded split Q4_K_M. The picker
    # enumerates the whole directory, so it would offer the broken one.
    (newer / "Model-Q8_0.gguf").write_bytes(b"\0")
    (newer / "Model-Q4_K_M-00001-of-00003.gguf").write_bytes(b"\0")
    os.utime(older, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))

    repo = _repo(
        "Org/Mixed",
        [],
        repo_dir,
        revisions = [
            SimpleNamespace(files = [_file("Model-Q8_0.gguf", 5_000)], snapshot_path = older),
            SimpleNamespace(
                files = [
                    _file("Model-Q8_0.gguf", 5_000),
                    _file("Model-Q4_K_M-00001-of-00003.gguf", 6_000),
                ],
                snapshot_path = newer,
            ),
        ],
    )

    monkeypatch.setattr(
        models_route, "_all_hf_cache_scans", lambda: [SimpleNamespace(repos = [repo])]
    )
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: active)

    rows = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))["cached"]

    assert rows[0]["load_id"] == str(older)


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
            "task": None,
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
            "task": None,
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


def test_is_hidden_model_hides_dictation_models(tmp_path):
    assert models_route._is_hidden_model("unsloth/whisper-tiny")
    assert models_route._is_hidden_model("unsloth/whisper-base")
    assert models_route._is_hidden_model("unsloth/whisper-small")
    assert models_route._is_hidden_model("unsloth/whisper-large-v3-turbo")
    assert models_route._is_hidden_model(
        "/hf/models--unsloth--whisper-large-v3/snapshots/abc/model.safetensors"
    )
    assert not models_route._is_hidden_model("user/whisper-finetune")
    assert not models_route._is_hidden_model(
        "C:\\cache\\models--unsloth--whisper-small-finetune\\model.safetensors"
    )
    custom = tmp_path / "custom-whisper"
    custom.mkdir()
    (custom / "config.json").write_text(
        '{"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]}'
    )
    (custom / "model.safetensors").write_bytes(b"weights")
    assert models_route._is_hidden_model(
        "user/custom-checkpoint",
        str(custom / "model.safetensors"),
    )
    named_only = tmp_path / "whisper-finetune"
    named_only.mkdir()
    (named_only / "config.json").write_text('{"model_type": "llama"}')
    assert not models_route._is_hidden_model("user/whisper-finetune", str(named_only))


def test_list_cached_models_hides_custom_whisper_by_config(monkeypatch, tmp_path):
    # Regression: the legacy /cached-models picker must pass the snapshot path so
    # the config check hides a custom (non-curated) Whisper checkpoint; a bare
    # repo id cannot ("user/whisper-finetune" is not in the curated set).
    repo_path = tmp_path / "models--user--whisper-finetune"
    snap = repo_path / "snapshots" / "abc"
    snap.mkdir(parents = True)
    (snap / "config.json").write_text(
        '{"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]}'
    )
    (snap / "model.safetensors").write_bytes(b"weights")

    captured: list = []
    real_hidden = models_route._is_hidden_model

    def spy(*values):
        captured.append(values)
        return real_hidden(*values)

    monkeypatch.setattr(models_route, "_is_hidden_model", spy)
    repo = _repo(
        "user/whisper-finetune",
        [SimpleNamespace(file_name = "model.safetensors", size_on_disk = 10)],
        repo_path,
    )
    monkeypatch.setattr(
        models_route, "_all_hf_cache_scans", lambda: [SimpleNamespace(repos = [repo])]
    )

    result = asyncio.run(
        models_route.list_cached_models(current_subject = "test-user", hf_token = None)
    )
    # The route passed the snapshot path (not just the repo id) ...
    assert any(str(repo_path) in values for values in captured)
    # ... so the custom Whisper checkpoint is hidden from the chat picker.
    assert result["cached"] == []


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
            "task": None,
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
            "task": None,
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


def test_list_cached_models_prefers_complete_over_larger_partial(monkeypatch, tmp_path):
    # The same repo cached in two roots: a LARGER but PARTIAL copy must not shadow a SMALLER but COMPLETE one, or the picker hides a usable model.
    complete = _repo(
        "Org/Dup",
        [_file("model.safetensors", 10_000)],
        tmp_path / "root_a" / "models--Org--Dup",
    )
    partial = _repo(
        "Org/Dup",
        [_file("model.safetensors", 15_000)],
        tmp_path / "root_b" / "models--Org--Dup",
    )

    # The larger copy (root_b) is the partial one; the smaller (root_a) is complete.
    monkeypatch.setattr(
        models_route,
        "_cached_repo_partial",
        lambda repo_id, repo_cache_dir = None: "root_b" in str(repo_cache_dir),
    )
    monkeypatch.setattr(models_route, "_cached_repo_task", lambda repo_info: None)
    # List the partial (larger) FIRST, so the old size-only rule would have picked it.
    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [partial, complete])],
    )

    result = asyncio.run(models_route.list_cached_models(current_subject = "test-user"))

    assert len(result["cached"]) == 1
    row = result["cached"][0]
    assert row["repo_id"] == "Org/Dup"
    # The COMPLETE (smaller) copy won.
    assert row.get("partial") is not True
    assert row["size_bytes"] == 10_000


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
            "task": None,
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
            "task": None,
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
            "task": None,
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

    assert result["cached"] == [{"repo_id": "Org/MmprojAux", "size_bytes": 15_000, "task": None}]


def test_list_cached_models_tags_diffusers_pipeline_as_text_to_image(monkeypatch, tmp_path):
    """A cached diffusers pipeline repo (model_index.json present) is tagged
    text-to-image so the chat picker hides it, while a plain checkpoint isn't."""
    diffusion = _repo(
        "Tongyi-MAI/Z-Image-Turbo",
        [
            _file("model_index.json", 1_000),
            _file("text_encoder/model.safetensors", 9_000),
            _file("transformer/diffusion_pytorch_model.safetensors", 9_000),
        ],
        tmp_path / "models--Tongyi-MAI--Z-Image-Turbo",
    )
    checkpoint = _repo(
        "unsloth/Llama-3.2-1B-Instruct",
        [_file("config.json", 1_000), _file("model.safetensors", 9_000)],
        tmp_path / "models--unsloth--Llama-3.2-1B-Instruct",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [diffusion, checkpoint])],
    )

    result = asyncio.run(models_route.list_cached_models(current_subject = "test-user"))
    by_repo = {c["repo_id"]: c["task"] for c in result["cached"]}
    assert by_repo == {
        "Tongyi-MAI/Z-Image-Turbo": "text-to-image",
        "unsloth/Llama-3.2-1B-Instruct": None,
    }


def test_list_cached_models_marks_companion_only_pipeline_partial(monkeypatch, tmp_path):
    """A companion-only prefetch (VAE / text-encoder / model_index.json but no transformer) carries
    a root model_index.json yet is not a loadable pipeline, so it must be marked partial. A sibling
    repo that DOES ship its transformer shards stays complete."""
    companion_only = _repo(
        "black-forest-labs/FLUX.1-dev",
        [
            _file("model_index.json", 1_000),
            _file("vae/diffusion_pytorch_model.safetensors", 9_000),
            _file("text_encoder/model.safetensors", 9_000),
        ],
        tmp_path / "models--black-forest-labs--FLUX.1-dev",
    )
    complete = _repo(
        "Tongyi-MAI/Z-Image-Turbo",
        [
            _file("model_index.json", 1_000),
            _file("text_encoder/model.safetensors", 9_000),
            _file("transformer/diffusion_pytorch_model.safetensors", 9_000),
        ],
        tmp_path / "models--Tongyi-MAI--Z-Image-Turbo",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [companion_only, complete])],
    )

    result = asyncio.run(models_route.list_cached_models(current_subject = "test-user"))
    by_repo = {c["repo_id"]: c for c in result["cached"]}
    assert by_repo["black-forest-labs/FLUX.1-dev"].get("partial") is True
    assert by_repo["Tongyi-MAI/Z-Image-Turbo"].get("partial") is None


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
            "task": None,
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
            "task": None,
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

    async def shared(
        repo_id,
        variant,
        hf_token,
        cache_path = None,
    ):
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


def test_arch_to_task_hides_unsupported_diffusion_from_chat():
    assert models_route._arch_to_task("flux") == "text-to-image"
    assert models_route._arch_to_task("z_image") == "text-to-image"
    assert models_route._arch_to_task("qwen_image") == "text-to-image"
    assert models_route._arch_to_task("llama") == "text-generation"
    assert models_route._arch_to_task(None) is None
    # Known-but-unsupported diffusion archs get a task that is neither chat nor a loadable image task, so both pickers skip them.
    for arch in ("sdxl", "sd1", "sd3", "lumina2", "hidream", "cosmos", "hyvid"):
        task = models_route._arch_to_task(arch)
        assert task == models_route._UNSUPPORTED_DIFFUSION_TASK
        assert task not in ("text-generation", "text-to-image")
    # A video arch with a REGISTERED VideoFamily surfaces with the Video-picker task.
    assert models_route._arch_to_task("ltxv") == models_route._VIDEO_GEN_TASK
    assert models_route._arch_to_task("ltxv") not in ("text-generation", "text-to-image")
    # A video arch that does not resolve from the bare arch alone ("wan" covers TI2V-5B and the A14B MoE) stays unsupported.
    assert models_route._arch_to_task("wan") == models_route._UNSUPPORTED_DIFFUSION_TASK
    assert models_route._arch_to_task("wan") not in ("text-generation", "text-to-image")
    # With a repo/file name hint the loadable TI2V-5B resolves to Video while the A14B MoE stays unsupported, matching the loader.
    assert (
        models_route._arch_to_task("wan", ("QuantStack/Wan2.2-TI2V-5B-GGUF",))
        == models_route._VIDEO_GEN_TASK
    )
    assert (
        models_route._arch_to_task("wan", (None, "Wan2.2-TI2V-5B-Q4_K_M.gguf"))
        == models_route._VIDEO_GEN_TASK
    )
    assert (
        models_route._arch_to_task("wan", ("QuantStack/Wan2.2-T2V-A14B-GGUF",))
        == models_route._UNSUPPORTED_DIFFUSION_TASK
    )
    # Drift guard: every diffusion arch llama.cpp rejects as a chat model must classify here as some non-chat task.
    from core.inference.llama_cpp import LlamaCppBackend

    classified = (
        models_route._DIFFUSION_GGUF_ARCHS
        | models_route._UNSUPPORTED_DIFFUSION_GGUF_ARCHS
        | models_route._AMBIGUOUS_DIFFUSION_GGUF_ARCHS
        | models_route._VIDEO_GGUF_ARCHS
    )
    missing = {a for a in LlamaCppBackend._DIFFUSION_ARCHES if a.lower() not in classified}
    assert not missing, f"diffusion archs would still show in chat: {missing}"


def test_arch_to_task_resolves_z_image_gguf_tagged_lumina2():
    # Z-Image's DiT is a Lumina2 derivative, so both Z-Image GGUF repos declare general.architecture = "lumina2". Reading
    # the arch alone tagged the whole line unsupported and hid it, even though validate_load_request loads it happily.
    for repo, fname in (
        ("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q4_K_M.gguf"),
        ("unsloth/Z-Image-GGUF", "z-image-Q8_0.gguf"),
    ):
        assert models_route._arch_to_task("lumina2", (repo, fname)) == "text-to-image"
        # The filename alone carries the family for a bare local .gguf pick.
        assert models_route._arch_to_task("lumina2", (None, fname)) == "text-to-image"
    # An unrecognised repo on the shared arch stays hidden rather than being guessed loadable.
    assert (
        models_route._arch_to_task("lumina2", ("someone/mystery-gguf", "model-Q4_K.gguf"))
        == models_route._UNSUPPORTED_DIFFUSION_TASK
    )


def test_arch_to_task_agrees_with_the_loader_on_ambiguous_archs():
    # The picker and the loader must not disagree: whatever _arch_to_task advertises as loadable, validate_load_request
    # must accept, and whatever it hides must be rejected. Otherwise the Images list hides a working model or offers a 400.
    from core.inference.diffusion import DiffusionBackend
    from core.inference.diffusion_families import _FAMILIES

    backend = DiffusionBackend.__new__(DiffusionBackend)  # validation touches no state
    for fam in _FAMILIES:
        repo = f"unsloth/{fam.name}-GGUF"
        fname = f"{fam.name}-Q4_K_M.gguf"
        task = models_route._arch_to_task("lumina2", (repo, fname))
        try:
            backend.validate_load_request(repo, gguf_filename = fname, model_kind = "gguf")
            loader_accepts = True
        except (ValueError, FileNotFoundError):
            loader_accepts = False
        assert (
            task == "text-to-image"
        ) == loader_accepts, f"{fam.name}: picker task={task} but loader accepts={loader_accepts}"


def _clear_chat_delete_guards(monkeypatch):
    """Report chat + orchestrator idle so only the Images / Video guards can refuse a delete."""
    import core.inference as core_inference
    import routes.inference as routes_inference

    monkeypatch.setattr(
        routes_inference,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(
            is_active = False,
            is_loaded = False,
            model_identifier = None,
            hf_variant = None,
        ),
    )
    monkeypatch.setattr(
        core_inference,
        "get_inference_backend",
        lambda: SimpleNamespace(active_model_name = None),
    )


def _idle_video_backend():
    return SimpleNamespace(
        status = lambda: {"loaded": False, "repo_id": None},
        loading_repo_ids = lambda: (),
    )


def _idle_diffusion_engine():
    return SimpleNamespace(
        status = lambda: {"loaded": False, "repo_id": None},
        loaded_repo_ids = lambda: (),
        loading_repo_ids = lambda: (),
    )


def test_delete_cached_refuses_diffusion_loaded_repo(monkeypatch):
    # The cached-delete guard refuses deleting a repo the Images backend has loaded, so its GGUF cannot vanish from under a live pipeline.
    from fastapi import HTTPException
    from hub.services.models import deletion
    import core.inference.diffusion_engine_router as der
    import core.inference.video as video_mod

    _clear_chat_delete_guards(monkeypatch)
    monkeypatch.setattr(
        der,
        "get_active_diffusion_engine",
        lambda: SimpleNamespace(
            status = lambda: {"loaded": True, "repo_id": "org/Z-Image-GGUF"},
            loaded_repo_ids = lambda: (),
            loading_repo_ids = lambda: (),
        ),
    )
    monkeypatch.setattr(video_mod, "get_video_backend", _idle_video_backend)

    try:
        asyncio.run(deletion.delete_cached_model_response("org/Z-Image-GGUF"))
        assert False, "expected HTTPException refusing the delete"
    except HTTPException as e:
        assert e.status_code == 400
        assert "Unload the model before deleting" in e.detail


def test_delete_cached_refuses_video_loaded_repo(monkeypatch):
    # Same for the Video backend, which shares the On-Device GGUF delete UI with chat/Images.
    from fastapi import HTTPException
    from hub.services.models import deletion
    import core.inference.diffusion_engine_router as der
    import core.inference.video as video_mod

    _clear_chat_delete_guards(monkeypatch)
    monkeypatch.setattr(der, "get_active_diffusion_engine", _idle_diffusion_engine)
    monkeypatch.setattr(
        video_mod,
        "get_video_backend",
        lambda: SimpleNamespace(
            status = lambda: {"loaded": True, "repo_id": "unsloth/LTX-2.3-GGUF"},
            loading_repo_ids = lambda: (),
        ),
    )

    try:
        asyncio.run(deletion.delete_cached_model_response("unsloth/LTX-2.3-GGUF"))
        assert False, "expected HTTPException refusing the delete"
    except HTTPException as e:
        assert e.status_code == 400
        assert "Unload the model before deleting" in e.detail


def test_delete_cached_refuses_loaded_native_companion_repo(monkeypatch):
    # The native sd.cpp one-shot engine re-reads its companion VAE / text-encoder files every generation, so deleting a
    # companion repo while a FLUX GGUF is loaded must be refused. The repo_id does not match, so the guard needs loaded_repo_ids().
    from fastapi import HTTPException
    from hub.services.models import deletion
    import core.inference.diffusion_engine_router as der
    import core.inference.video as video_mod

    _clear_chat_delete_guards(monkeypatch)
    monkeypatch.setattr(
        der,
        "get_active_diffusion_engine",
        lambda: SimpleNamespace(
            status = lambda: {"loaded": True, "repo_id": "unsloth/FLUX.1-dev-GGUF"},
            loaded_repo_ids = lambda: (
                "unsloth/FLUX.1-dev-GGUF",
                "black-forest-labs/FLUX.1-dev",
                "comfyanonymous/flux_text_encoders",
            ),
            loading_repo_ids = lambda: (),
        ),
    )
    monkeypatch.setattr(video_mod, "get_video_backend", _idle_video_backend)

    try:
        asyncio.run(deletion.delete_cached_model_response("comfyanonymous/flux_text_encoders"))
        assert False, "expected HTTPException refusing the in-use companion delete"
    except HTTPException as e:
        assert e.status_code == 400
        assert "Unload the model before deleting" in e.detail


def test_delete_cached_refuses_repo_a_diffusion_load_is_downloading(monkeypatch):
    # status().loaded is still False while a background Images load downloads the repo, so loading_repo_ids() must refuse the delete.
    from fastapi import HTTPException
    from hub.services.models import deletion
    import core.inference.diffusion_engine_router as der
    import core.inference.video as video_mod

    _clear_chat_delete_guards(monkeypatch)
    monkeypatch.setattr(
        der,
        "get_active_diffusion_engine",
        lambda: SimpleNamespace(
            status = lambda: {"loaded": False, "repo_id": None},
            loaded_repo_ids = lambda: (),
            loading_repo_ids = lambda: ("unsloth/Qwen-Image-2512-GGUF",),
        ),
    )
    monkeypatch.setattr(video_mod, "get_video_backend", _idle_video_backend)

    try:
        asyncio.run(deletion.delete_cached_model_response("unsloth/Qwen-Image-2512-GGUF"))
        assert False, "expected HTTPException refusing the delete mid-download"
    except HTTPException as e:
        assert e.status_code == 400
        assert "An Images model load is using this repo" in e.detail


def test_delete_cached_allows_sibling_of_loaded_diffusion_repo(monkeypatch):
    # A loaded Images repo must not block deleting a different cached repo sharing a name prefix; the guard is `/`-boundary aware.
    from fastapi import HTTPException
    from hub.services.models import deletion
    import core.inference.diffusion_engine_router as der
    import core.inference.video as video_mod

    _clear_chat_delete_guards(monkeypatch)
    monkeypatch.setattr(
        der,
        "get_active_diffusion_engine",
        lambda: SimpleNamespace(
            status = lambda: {"loaded": True, "repo_id": "Qwen/Qwen-Image-2512"},
            loaded_repo_ids = lambda: (),
            loading_repo_ids = lambda: (),
        ),
    )
    monkeypatch.setattr(video_mod, "get_video_backend", _idle_video_backend)
    # Stub the destructive stage: this test is about the guard boundary, not the cache walk.
    monkeypatch.setattr(
        deletion,
        "_delete_cached_model_blocking",
        lambda repo_id, variant, hf_token, cache_path = None: {
            "status": "deleted",
            "repo_id": repo_id,
        },
    )

    # The sibling repo clears every guard and reaches the delete.
    result = asyncio.run(deletion.delete_cached_model_response("Qwen/Qwen-Image"))
    assert result == {"status": "deleted", "repo_id": "Qwen/Qwen-Image"}

    # The loaded repo itself is still refused (exact match).
    try:
        asyncio.run(deletion.delete_cached_model_response("Qwen/Qwen-Image-2512"))
        assert False, "expected HTTPException refusing delete of the loaded repo"
    except HTTPException as e:
        assert e.status_code == 400
        assert "Unload the model before deleting" in e.detail


def test_cached_repo_partial_scopes_probe_to_snapshot_dir(monkeypatch):
    # The partial probe must be scoped to the snapshot row being listed: unscoped, a stale .incomplete copy in one cache
    # root would flag a complete copy in another as partial and hide the usable model.
    import hub.utils.inventory_scan as scan

    calls = []

    def _fake(
        repo_type,
        repo_id,
        repo_cache_dir = None,
    ):
        calls.append((repo_type, repo_id, repo_cache_dir))
        return False

    monkeypatch.setattr(scan, "is_snapshot_partial", _fake)
    snapshot_dir = Path("/root_a/models--Org--Repo/snapshots/abc")
    assert models_route._cached_repo_partial("Org/Repo", snapshot_dir) is False
    assert calls == [("model", "Org/Repo", snapshot_dir)]

    monkeypatch.setattr(scan, "is_snapshot_partial", lambda *a, **k: True)
    assert models_route._cached_repo_partial("Org/Repo", snapshot_dir) is True

    # A probe error is swallowed (never hides a usable repo over a scan glitch).
    def _boom(*a, **k):
        raise RuntimeError("scan glitch")

    monkeypatch.setattr(scan, "is_snapshot_partial", _boom)
    assert models_route._cached_repo_partial("Org/Repo", snapshot_dir) is False


def test_repo_has_pipeline_index_requires_root_model_index(tmp_path):
    # Only a ROOT model_index.json makes a repo pipeline-loadable, so a nested subdir one must NOT clear the single_file
    # flag. CachedFileInfo.file_name is the basename, so the helper scopes by snapshot path.
    snap = tmp_path / "snapshots" / "abc"
    nested = SimpleNamespace(
        file_name = "model_index.json",
        file_path = snap / "prior" / "model_index.json",
    )
    repo_nested = SimpleNamespace(
        repo_id = "unsloth/nested-index",
        revisions = [SimpleNamespace(files = [nested], snapshot_path = snap)],
    )
    assert models_route._repo_has_pipeline_index(repo_nested) is False

    root = SimpleNamespace(
        file_name = "model_index.json",
        file_path = snap / "model_index.json",
    )
    repo_root = SimpleNamespace(
        repo_id = "unsloth/root-index",
        revisions = [SimpleNamespace(files = [root], snapshot_path = snap)],
    )
    assert models_route._repo_has_pipeline_index(repo_root) is True


def test_pipeline_scans_read_the_snapshot_the_loader_will_open(tmp_path):
    # A repo cached twice -- an older complete snapshot plus a newer companion-only one, the shape a GGUF load leaves when
    # it prefetches the base repo and skips the transformer -- must be judged on the snapshot from_pretrained resolves, the
    # newest by mtime. Scanning every revision let the OLD transformer satisfy completeness, so the row read as on-device.
    import os

    import hub.utils.inventory_scan as scan

    repo_dir = tmp_path / "models--Org--Repo"
    old_snap = repo_dir / "snapshots" / "old"
    new_snap = repo_dir / "snapshots" / "new"
    for d in (old_snap / "transformer", new_snap / "vae"):
        d.mkdir(parents = True)
    (old_snap / "model_index.json").write_text("{}", encoding = "utf-8")
    (new_snap / "model_index.json").write_text("{}", encoding = "utf-8")
    # Make "new" unambiguously newer than "old" for the mtime rule both this and the loader use.
    os.utime(old_snap, (1_000_000, 1_000_000))
    os.utime(new_snap, (2_000_000, 2_000_000))

    def _rev(snap, files):
        return SimpleNamespace(
            snapshot_path = snap,
            last_modified = float(snap.stat().st_mtime),
            files = [SimpleNamespace(file_name = Path(f).name, file_path = snap / f) for f in files],
        )

    info = SimpleNamespace(
        repo_id = "Org/Repo",
        repo_path = repo_dir,
        revisions = [
            _rev(old_snap, ["model_index.json", "transformer/diffusion_pytorch_model.safetensors"]),
            _rev(new_snap, ["model_index.json", "vae/diffusion_pytorch_model.safetensors"]),
        ],
    )
    assert scan.repo_has_pipeline_index(info) is True
    assert scan.repo_pipeline_missing_denoiser(info) is True

    # The reverse cache (the complete snapshot is the newer one) still reports complete.
    os.utime(old_snap, (3_000_000, 3_000_000))
    info.revisions = [
        _rev(old_snap, ["model_index.json", "transformer/diffusion_pytorch_model.safetensors"]),
        _rev(new_snap, ["model_index.json", "vae/diffusion_pytorch_model.safetensors"]),
    ]
    assert scan.repo_pipeline_missing_denoiser(info) is False


def test_list_cached_models_flags_single_file_diffusion_repos(monkeypatch, tmp_path):
    # A diffusion-tagged repo with NO top-level model_index.json is a single-file checkpoint (single_file=True); a full pipeline or chat repo carries no flag.
    single = _repo(
        "unsloth/Qwen-Image-fp8-single",
        [_file("qwen-image-fp8.safetensors", 10_000)],
        tmp_path / "models--unsloth--Qwen-Image-fp8-single",
    )
    pipeline = _repo(
        "unsloth/Qwen-Image-pipeline",
        [_file("model_index.json", 10), _file("transformer/model.safetensors", 10_000)],
        tmp_path / "models--unsloth--Qwen-Image-pipeline",
    )
    chat = _repo(
        "Org/ChatRepo",
        [_file("model.safetensors", 10_000)],
        tmp_path / "models--Org--ChatRepo",
    )

    monkeypatch.setattr(
        models_route,
        "_cached_repo_task",
        lambda repo_info: ("text-to-image" if "Qwen-Image" in repo_info.repo_id else None),
    )
    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [single, pipeline, chat])],
    )

    result = asyncio.run(models_route.list_cached_models(current_subject = "test-user"))

    rows = {r["repo_id"]: r for r in result["cached"]}
    assert rows["unsloth/Qwen-Image-fp8-single"].get("single_file") is True
    assert "single_file" not in rows["unsloth/Qwen-Image-pipeline"]
    assert "single_file" not in rows["Org/ChatRepo"]


def _pipeline_repo(repo_id: str, tmp_path: Path) -> SimpleNamespace:
    return _repo(
        repo_id,
        [
            _file("model_index.json", 1_000),
            _file("transformer/diffusion_pytorch_model.safetensors", 5_000_000),
        ],
        tmp_path / f"models--{repo_id.replace('/', '--')}",
    )


def test_cached_repo_task_gates_an_image_pipeline_on_the_load_path_trust_rule(tmp_path):
    """Every advertised row must be loadable. A cached community pipeline has a model_index.json
    like any other, so tagging it text-to-image put a row in the Images picker that the loader's
    trust check refuses -- the pick 400s. Gate the tag on the same rule."""
    assert models_route._cached_repo_task(_pipeline_repo("unsloth/Qwen-Image", tmp_path)) == (
        "text-to-image"
    )
    assert (
        models_route._cached_repo_task(_pipeline_repo("someone/their-sdxl-mix", tmp_path)) is None
    )


def test_cached_repo_task_hides_an_untrusted_video_repo_instead_of_listing_it_under_images(
    monkeypatch, tmp_path
):
    """A detected video pipeline that fails the video trust rule used to fall through to the image
    fallback and show up in the Images picker, where it is just as unloadable."""
    import core.inference.video as video_mod

    repo = _pipeline_repo("someone/their-ltx-fork", tmp_path)
    monkeypatch.setattr(
        "core.inference.video_families.detect_video_family",
        lambda repo_id: object(),
    )
    monkeypatch.setattr(video_mod, "_is_trusted_video_repo", lambda repo_id: False)
    assert models_route._cached_repo_task(repo) is None

    monkeypatch.setattr(video_mod, "_is_trusted_video_repo", lambda repo_id: True)
    assert models_route._cached_repo_task(repo) == models_route._VIDEO_GEN_TASK


def test_hub_cached_rows_carry_the_task_the_pickers_filter_on(monkeypatch, tmp_path):
    """The picker's On Device rows come from the /api/hub inventory, not the models API. Without a
    task on those rows the Images and Video pickers filtered every one of them out, and the chat
    picker's diffusion routing (which reads the same field) never fired."""
    from hub.schemas.inventory import CachedGgufRepo, CachedModelRepo
    from hub.services.models import cache_inventory

    assert "task" in CachedGgufRepo.model_fields
    assert "task" in CachedModelRepo.model_fields

    repo = _pipeline_repo("unsloth/Qwen-Image", tmp_path)
    monkeypatch.setattr(
        "routes.models._cached_repo_task", lambda repo_info: "text-to-image", raising = True
    )
    assert cache_inventory._cached_row_task(repo, gguf = False) == "text-to-image"
    monkeypatch.setattr(
        "routes.models._repo_gguf_task", lambda repo_info: "text-generation", raising = True
    )
    assert cache_inventory._cached_row_task(repo, gguf = True) == "text-generation"


def test_hub_cached_row_task_never_hides_a_row_when_classification_fails(monkeypatch, tmp_path):
    # Best-effort, like the models API: a classifier that raises leaves the row untagged rather than dropping it.
    from hub.services.models import cache_inventory

    def _boom(repo_info):
        raise RuntimeError("unreadable")

    monkeypatch.setattr("routes.models._cached_repo_task", _boom, raising = True)
    assert cache_inventory._cached_row_task(_pipeline_repo("a/b", tmp_path), gguf = False) is None


def test_hub_local_rows_are_tagged_with_their_task():
    """/api/hub/local feeds the same pickers, and its rows were untagged too."""
    import inspect

    from hub.schemas.inventory import LocalModelInfo
    from hub.services.models import local_inventory

    assert "task" in LocalModelInfo.model_fields
    src = inspect.getsource(local_inventory.list_local_models_response)
    assert "_local_model_task" in src
    assert 'model_copy(update = {"task"' in src


def test_pipeline_class_guard_fires_before_any_download():
    # The 0.39-only families used to die with a bare AttributeError deep in the load, after the checkpoint was fetched, on
    # the older diffusers packaging still allows on Python 3.9. Validation refuses first, naming the version and the fix.
    import pytest

    from core.inference.diffusion_families import _FAMILIES, assert_pipeline_class_available

    # Present -> no raise (every shipped family resolves on a current diffusers).
    import diffusers

    for fam in _FAMILIES:
        assert_pipeline_class_available(fam.pipeline_class, fam.name)

    stub = types.SimpleNamespace(__version__ = "0.37.0")
    real = sys.modules.get("diffusers")
    sys.modules["diffusers"] = stub
    try:
        # ValueError, like every other unloadable-pick refusal: RuntimeError reached /images/load's 409 and escaped download-plan as a 500.
        with pytest.raises(ValueError) as excinfo:
            assert_pipeline_class_available("ZImagePipeline", "z-image")
    finally:
        if real is not None:
            sys.modules["diffusers"] = real
        else:  # pragma: no cover
            del sys.modules["diffusers"]
    msg = str(excinfo.value)
    assert "z-image" in msg and "ZImagePipeline" in msg
    assert "0.39" in msg and "0.37.0" in msg
    assert "3.10" in msg  # names the Python floor that carries a new enough diffusers
    assert diffusers is not None


def test_cached_pipeline_needs_a_detectable_image_family(monkeypatch):
    # A top-level model_index.json only proves the repo is a diffusers pipeline. An unsloth-hosted pipeline of a class this
    # backend cannot assemble cleared the trust gate, was advertised to the picker, then failed validate_load_request.
    # Both gates now, mirroring the video branch above.
    monkeypatch.setattr(models_route, "_repo_has_pipeline_index", lambda info: True)

    def _task(repo_id):
        return models_route._cached_repo_task(SimpleNamespace(repo_id = repo_id, repo_path = "/x"))

    # Trusted AND a detected family -> claimed by Images.
    assert _task("unsloth/Z-Image-Turbo") == "text-to-image"
    assert _task("unsloth/FLUX.1-dev") == "text-to-image"
    # Trusted but no image family the loader can detect -> not advertised.
    assert _task("unsloth/some-unsupported-pipeline") is None
    # Untrusted keeps its existing refusal.
    assert _task("someone/random-diffusers-pipeline") is None


def test_cached_repo_task_agrees_with_the_image_loader(monkeypatch):
    # Same invariant as the GGUF arch test: whatever the picker advertises as loadable, validate_load_request must accept.
    from core.inference.diffusion import DiffusionBackend

    monkeypatch.setattr(models_route, "_repo_has_pipeline_index", lambda info: True)
    backend = DiffusionBackend.__new__(DiffusionBackend)
    for repo_id in (
        "unsloth/Z-Image-Turbo",
        "unsloth/FLUX.1-dev",
        "unsloth/some-unsupported-pipeline",
        "unsloth/stable-audio-open-1.0",
    ):
        task = models_route._cached_repo_task(SimpleNamespace(repo_id = repo_id, repo_path = "/x"))
        try:
            backend.validate_load_request(repo_id)
            loader_accepts = True
        except (ValueError, FileNotFoundError, RuntimeError):
            loader_accepts = False
        assert (
            task == "text-to-image"
        ) == loader_accepts, f"{repo_id}: picker task={task} but loader accepts={loader_accepts}"


def test_cached_picker_hides_a_family_this_diffusers_cannot_build(monkeypatch):
    # The newer families exist only from diffusers 0.39, which cannot be installed on Python 3.9 at all, so advertising one
    # there is a pick that can only fail; the picker applies the same availability check validate_load_request does.
    import types

    import routes.models as models_module
    from core.inference.diffusion_families import detect_family, family_pipeline_available

    fam = detect_family("unsloth/Z-Image-Turbo")
    assert fam is not None
    # Present in this environment's diffusers, so the row is offered.
    assert family_pipeline_available(fam) is True

    monkeypatch.setattr(models_module, "_repo_is_diffusers", lambda info: True)
    monkeypatch.setattr("core.inference.diffusion._is_trusted_diffusion_repo", lambda repo_id: True)
    info = types.SimpleNamespace(repo_id = "unsloth/Z-Image-Turbo")
    assert models_module._cached_repo_task(info) == "text-to-image"

    # An older diffusers without the pipeline class hides the row instead.
    monkeypatch.setattr(
        "core.inference.diffusion_families.family_pipeline_available", lambda f: False
    )
    assert models_module._cached_repo_task(info) is None


def test_family_pipeline_available_fails_open_without_diffusers(monkeypatch):
    # No diffusers at all is a different problem the load path reports properly; a listing must not hide every image model over it.
    import sys

    from core.inference.diffusion_families import detect_family, family_pipeline_available

    monkeypatch.setitem(sys.modules, "diffusers", None)
    assert family_pipeline_available(detect_family("unsloth/Z-Image-Turbo")) is True
    assert family_pipeline_available(None) is False


# ── the unbuildable-family gate on the GGUF paths (both engines) ─────────────


def _pretend_old_diffusers(monkeypatch, *, engine):
    """An environment whose diffusers has none of the newer pipeline classes, on a host whose GGUF
    loads route to ``engine``.

    0.36.0 is the real ceiling for a Python 3.9 host (0.37.0 already declares requires-python
    >=3.10), and it ships no Flux2KleinPipeline. Only the diffusers module and the engine prediction
    are substituted: the availability check, the picker and validate_load_request are the real code.
    """
    import core.inference.diffusion_engine_router as router

    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace(__version__ = "0.36.0"))
    monkeypatch.setattr(router, "predict_engine", lambda fam, **kwargs: engine)


def test_gguf_picker_hides_a_family_no_engine_here_can_build(monkeypatch):
    # The gate landed on the cached-repo picker only, so the GGUF repos -- the ones the Images picker actually offers for
    # these families -- still showed as text-to-image on a diffusers too old to build them, and every pick died.
    from core.inference.sd_cpp_engine import ENGINE_DIFFUSERS

    _pretend_old_diffusers(monkeypatch, engine = ENGINE_DIFFUSERS)

    # The flat diffusion-arch branch (FLUX.2) and the ambiguous one (Z-Image ships as "lumina2").
    assert (
        models_route._arch_to_task("flux2", ("unsloth/FLUX.2-klein-4B-GGUF",))
        == models_route._UNSUPPORTED_DIFFUSION_TASK
    )
    assert (
        models_route._arch_to_task("lumina2", ("unsloth/Z-Image-Turbo-GGUF",))
        == models_route._UNSUPPORTED_DIFFUSION_TASK
    )
    # Neither chat nor Images: the row is hidden, not moved to the picker that would also fail.
    assert models_route._arch_to_task("flux2", ("unsloth/FLUX.2-klein-4B-GGUF",)) not in (
        "text-generation",
        "text-to-image",
    )


def test_gguf_picker_keeps_a_family_the_native_engine_serves(monkeypatch):
    # The opposite mistake: on a CPU/MPS or force-native host sd.cpp loads the GGUF and never instantiates a diffusers
    # class, so hiding the row over a missing class would withhold a model that loads fine.
    from core.inference.sd_cpp_engine import ENGINE_SD_CPP

    _pretend_old_diffusers(monkeypatch, engine = ENGINE_SD_CPP)

    assert models_route._arch_to_task("flux2", ("unsloth/FLUX.2-klein-4B-GGUF",)) == "text-to-image"
    assert models_route._arch_to_task("lumina2", ("unsloth/Z-Image-Turbo-GGUF",)) == "text-to-image"


def test_the_loader_demands_the_diffusers_class_only_when_diffusers_loads_it(monkeypatch):
    # Same predicate on the load path: refuse a too-old diffusers before the download, and never when sd.cpp serves the GGUF.
    import pytest

    from core.inference.diffusion import DiffusionBackend
    from core.inference.sd_cpp_engine import ENGINE_DIFFUSERS, ENGINE_SD_CPP

    backend = DiffusionBackend.__new__(DiffusionBackend)

    _pretend_old_diffusers(monkeypatch, engine = ENGINE_SD_CPP)
    fam = backend.validate_load_request(
        "unsloth/FLUX.2-klein-4B-GGUF",
        gguf_filename = "flux2-klein-4b-Q4_0.gguf",
        model_kind = "gguf",
    )
    assert fam.name == "flux.2-klein"

    _pretend_old_diffusers(monkeypatch, engine = ENGINE_DIFFUSERS)
    with pytest.raises(ValueError) as excinfo:
        backend.validate_load_request(
            "unsloth/FLUX.2-klein-4B-GGUF",
            gguf_filename = "flux2-klein-4b-Q4_0.gguf",
            model_kind = "gguf",
        )
    # ValueError, not RuntimeError: /images/load maps RuntimeError to 409 and /images/download-plan catches only
    # (ValueError, FileNotFoundError), so the message escaped as a bare 500.
    assert "Flux2KleinPipeline" in str(excinfo.value)


def test_the_video_picker_hides_a_family_this_diffusers_cannot_build(monkeypatch):
    # Same gap on the video branches: LTX-2's pipeline class is 0.39-only too, and video has no native engine to fall back
    # to, so the load asserts it unconditionally (video.py -> assert_pipeline_class_available).
    monkeypatch.setattr(models_route, "_repo_is_diffusers", lambda info: True)
    info = SimpleNamespace(repo_id = "Lightricks/LTX-2", repo_path = "/x")
    # Offered on this environment's diffusers ...
    assert models_route._arch_to_task("ltxv") == models_route._VIDEO_GEN_TASK
    assert models_route._cached_repo_task(info) == models_route._VIDEO_GEN_TASK

    # ... and hidden on one that has no LTX2Pipeline.
    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace(__version__ = "0.36.0"))
    assert models_route._arch_to_task("ltxv") == models_route._UNSUPPORTED_DIFFUSION_TASK
    assert models_route._cached_repo_task(info) is None


def test_every_shipped_video_family_resolves_on_this_diffusers():
    # Drift guard: the video picker now hides a family whose pipeline class the installed diffusers lacks, so a stale class
    # name in the table would hide a working model rather than just fail late.
    from core.inference.diffusion_families import family_pipeline_available
    from core.inference.video_families import _FAMILIES as _VIDEO_FAMILIES
    for fam in _VIDEO_FAMILIES:
        assert family_pipeline_available(
            fam
        ), f"{fam.name}: {fam.pipeline_class} is not in diffusers"


def test_the_gguf_picker_and_the_image_loader_agree_on_an_old_diffusers(monkeypatch):
    # The invariant test_cached_repo_task_agrees_with_the_image_loader states for cached repos, applied to the GGUF path on
    # both host kinds: whatever the picker advertises must be accepted, and whatever it hides must be refused.
    from core.inference.diffusion import DiffusionBackend
    from core.inference.sd_cpp_engine import ENGINE_DIFFUSERS, ENGINE_SD_CPP

    backend = DiffusionBackend.__new__(DiffusionBackend)
    picks = (
        ("flux2", "unsloth/FLUX.2-klein-4B-GGUF", "flux2-klein-4b-Q4_0.gguf"),
        ("lumina2", "unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q8_0.gguf"),
    )
    for engine in (ENGINE_DIFFUSERS, ENGINE_SD_CPP):
        _pretend_old_diffusers(monkeypatch, engine = engine)
        for arch, repo_id, filename in picks:
            task = models_route._arch_to_task(arch, (repo_id, filename))
            try:
                backend.validate_load_request(repo_id, gguf_filename = filename, model_kind = "gguf")
                loader_accepts = True
            except (ValueError, FileNotFoundError, RuntimeError):
                loader_accepts = False
            assert (
                (task == "text-to-image") == loader_accepts
            ), f"{repo_id} on {engine}: picker task={task} but loader accepts={loader_accepts}"
