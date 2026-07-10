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
    assert not models_route._is_hidden_model("unsloth/gemma-3-270m-it-GGUF")
    # The exact-filename needle must not hide a real repo that merely
    # references stories260K in its name.
    assert not models_route._is_hidden_model("user/stories260K-finetune-GGUF")


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
    # The same repo cached in two roots: a LARGER but PARTIAL copy must not shadow a SMALLER but
    # COMPLETE one. The picker drops partial rows, so picking the partial winner by size alone would
    # make a usable model vanish from On Device. Completeness wins over size.
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
    # The COMPLETE (smaller) copy won: it is not flagged partial and carries its 10_000 size.
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
        [_file("model_index.json", 1_000), _file("text_encoder/model.safetensors", 9_000)],
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


def test_all_hf_cache_scans_survives_inaccessible_aux_cache(monkeypatch, tmp_path):
    """An unreadable auxiliary cache (e.g. an inaccessible
    ``~/.cache/huggingface/hub``) must be skipped, not abort the scan.
    Regression guard for ``extra.is_dir()`` raising and wiping the response.
    """
    import huggingface_hub
    import utils.paths as paths_mod

    active = SimpleNamespace(
        repos = [_repo("Org/Active", [_file("Q4_K_M.gguf", 5_000)], tmp_path / "active")]
    )

    def _fake_scan(cache_dir = None):
        if cache_dir is None:
            return active
        raise AssertionError("auxiliary scan should have been skipped")

    class _Boom:
        def is_dir(self):
            raise PermissionError(13, "Permission denied")

        def resolve(self):
            raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", _fake_scan)
    monkeypatch.setattr(paths_mod, "legacy_hf_cache_dir", lambda: _Boom())
    monkeypatch.setattr(paths_mod, "hf_default_cache_dir", lambda: _Boom())

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
    monkeypatch.setattr(GV, "_local_main_gguf_blobs_by_quant", lambda _repo_id: {})

    snap = tmp_path / "models--org--repo" / "snapshots" / "rev"
    snap.mkdir(parents = True)
    (snap / "model-Q4_K_M.gguf").write_bytes(b"x" * 10_000)  # real weight, fully present
    (snap / "mmproj-F16.gguf").write_bytes(b"y" * 20_000)  # mmproj adapter, label "F16"
    monkeypatch.setattr(GV, "iter_hf_cache_snapshots", lambda _repo_id: [snap])

    result = asyncio.run(
        models_route.get_gguf_variants(
            repo_id = "org/repo", hf_token = None, current_subject = "test-user"
        )
    )

    flags = {v.quant: v.downloaded for v in result.variants}
    assert flags["Q4_K_M"] is True
    assert flags["F16"] is False


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
    monkeypatch.setattr(GV, "_local_main_gguf_blobs_by_quant", lambda _repo_id: {})

    snap = tmp_path / "models--org--repo" / "snapshots" / "rev"
    snap.mkdir(parents = True)
    (snap / "model-Q4_K_M.gguf").write_bytes(b"x" * 10)
    monkeypatch.setattr(GV, "iter_hf_cache_snapshots", lambda _repo_id: [snap])

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
    monkeypatch.setattr(GV, "_local_main_gguf_blobs_by_quant", lambda _repo_id: {})

    snap = tmp_path / "models--org--repo" / "snapshots" / "rev"
    snap.mkdir(parents = True)
    (snap / "model-Q4_K_M-be.gguf").write_bytes(b"x" * 10)
    monkeypatch.setattr(GV, "iter_hf_cache_snapshots", lambda _repo_id: [snap])

    result = asyncio.run(
        models_route.get_gguf_variants(
            repo_id = "org/repo", hf_token = None, current_subject = "test-user"
        )
    )

    assert result.variants[0].downloaded is False


def test_gguf_download_progress_excludes_mmproj(monkeypatch, tmp_path):
    """A cached mmproj adapter must not count toward a same-label main
    variant's download progress (mmproj-F16 vs an F16 weight)."""
    import huggingface_hub.constants as hf_constants

    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    snap = tmp_path / "models--org--repo" / "snapshots" / "rev"
    snap.mkdir(parents = True)
    (snap / "mmproj-F16.gguf").write_bytes(b"y" * 20_000)  # only the adapter on disk

    result = asyncio.run(
        models_route.get_gguf_download_progress(
            repo_id = "org/repo",
            variant = "F16",
            expected_bytes = 20_000,
            current_subject = "test-user",
        )
    )

    assert result["downloaded_bytes"] == 0
    assert result["progress"] == 0


def test_gguf_download_progress_excludes_big_endian_sibling(monkeypatch, tmp_path):
    import huggingface_hub.constants as hf_constants

    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    snap = tmp_path / "models--org--repo" / "snapshots" / "rev"
    snap.mkdir(parents = True)
    (snap / "model-Q4_K_M-be.gguf").write_bytes(b"y" * 20_000)

    result = asyncio.run(
        models_route.get_gguf_download_progress(
            repo_id = "org/repo",
            variant = "Q4_K_M",
            expected_bytes = 20_000,
            current_subject = "test-user",
        )
    )

    assert result["downloaded_bytes"] == 0
    assert result["progress"] == 0


def test_gguf_download_progress_counts_quant_subdir(monkeypatch, tmp_path):
    import huggingface_hub.constants as hf_constants

    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    snap = tmp_path / "models--org--repo" / "snapshots" / "rev" / "Q4_K_M"
    snap.mkdir(parents = True)
    (snap / "foo.gguf").write_bytes(b"x" * 20_000)

    result = asyncio.run(
        models_route.get_gguf_download_progress(
            repo_id = "org/repo",
            variant = "Q4_K_M",
            expected_bytes = 20_000,
            current_subject = "test-user",
        )
    )

    assert result["downloaded_bytes"] == 20_000
    assert result["progress"] == 1.0


def test_arch_to_task_hides_unsupported_diffusion_from_chat():
    # Loadable diffusion archs -> the Images-picker task.
    assert models_route._arch_to_task("flux") == "text-to-image"
    assert models_route._arch_to_task("z_image") == "text-to-image"
    assert models_route._arch_to_task("qwen_image") == "text-to-image"
    # A real LLM arch stays a chat model; None passes through.
    assert models_route._arch_to_task("llama") == "text-generation"
    assert models_route._arch_to_task(None) is None
    # Known-but-unsupported diffusion archs get a task that is NEITHER chat
    # ("text-generation") NOR a loadable image task ("text-to-image"), so the chat
    # picker hides them (they'd die in llama.cpp) and the Images picker leaves them
    # out (they'd 400 in validate_load).
    for arch in ("sdxl", "sd1", "sd3", "lumina2", "hidream", "cosmos", "hyvid"):
        task = models_route._arch_to_task(arch)
        assert task == models_route._UNSUPPORTED_DIFFUSION_TASK
        assert task not in ("text-generation", "text-to-image")
    # A video arch with a REGISTERED VideoFamily surfaces with the Video-picker task (unsloth
    # LTX-2.x GGUFs ship general.architecture "ltxv" and ltx-2 is registered).
    assert models_route._arch_to_task("ltxv") == models_route._VIDEO_GEN_TASK
    assert models_route._arch_to_task("ltxv") not in ("text-generation", "text-to-image")
    # A video arch that does not resolve from the bare arch alone ("wan" is ambiguous -- it
    # covers both the loadable single-DiT TI2V-5B and the A14B MoE whose single file the loader
    # refuses) stays unsupported when no repo/file name is available to disambiguate, rather than
    # surfacing a GGUF that might 400 on load.
    assert models_route._arch_to_task("wan") == models_route._UNSUPPORTED_DIFFUSION_TASK
    assert models_route._arch_to_task("wan") not in ("text-generation", "text-to-image")
    # With a repo/file name hint, the loadable TI2V-5B Wan GGUF resolves to the Video task (so it
    # surfaces in the Video On-Device picker), while the A14B MoE (single file refused by the
    # loader) stays in the unsupported bucket -- matching the loader's own name-aware detection.
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
    # Drift guard: every diffusion arch llama.cpp rejects as a chat model must be
    # classified here as some non-chat task (image, video, or unsupported).
    from core.inference.llama_cpp import LlamaCppBackend

    classified = (
        models_route._DIFFUSION_GGUF_ARCHS
        | models_route._UNSUPPORTED_DIFFUSION_GGUF_ARCHS
        | models_route._VIDEO_GGUF_ARCHS
    )
    missing = {a for a in LlamaCppBackend._DIFFUSION_ARCHES if a.lower() not in classified}
    assert not missing, f"diffusion archs would still show in chat: {missing}"


def test_delete_cached_refuses_diffusion_loaded_repo(monkeypatch):
    # The cached-delete guard refuses deleting a repo the diffusion (Images)
    # backend has loaded, mirroring the chat guard, so its GGUF can't be removed
    # from under a live pipeline.
    from fastapi import HTTPException
    import core.inference.diffusion as diffusion_mod
    import routes.inference as routes_inference

    # Chat and orchestrator report nothing loaded; only diffusion holds the repo.
    # delete_cached_model resolves get_inference_backend from the models module
    # namespace, so patch it there (not on core.inference) to isolate that guard.
    monkeypatch.setattr(
        routes_inference,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(is_loaded = False, model_identifier = None),
    )
    monkeypatch.setattr(
        models_route,
        "get_inference_backend",
        lambda: SimpleNamespace(active_model_name = None),
    )
    monkeypatch.setattr(
        diffusion_mod,
        "get_diffusion_backend",
        lambda: SimpleNamespace(status = lambda: {"loaded": True, "repo_id": "org/Z-Image-GGUF"}),
    )

    try:
        asyncio.run(
            models_route.delete_cached_model(
                repo_id = "org/Z-Image-GGUF",
                variant = None,
                current_subject = "u",
            )
        )
        assert False, "expected HTTPException refusing the delete"
    except HTTPException as e:
        assert e.status_code == 400
        assert "Unload the model before deleting" in e.detail


def test_delete_cached_refuses_video_loaded_repo(monkeypatch):
    # The cached-delete guard must refuse deleting a repo the Video backend has loaded (it
    # shares the On-Device GGUF delete UI with chat/Images), so its GGUF can't be removed from
    # under a live video pipeline -- the same invariant the three sibling guards enforce.
    from fastapi import HTTPException
    import core.inference.diffusion_engine_router as der
    import core.inference.video as video_mod
    import routes.inference as routes_inference

    # Chat, orchestrator, and Images all report nothing loaded; only Video holds the repo.
    monkeypatch.setattr(
        routes_inference,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(is_loaded = False, model_identifier = None),
    )
    monkeypatch.setattr(
        models_route,
        "get_inference_backend",
        lambda: SimpleNamespace(active_model_name = None),
    )
    monkeypatch.setattr(
        der,
        "get_active_diffusion_engine",
        lambda: SimpleNamespace(
            status = lambda: {"loaded": False, "repo_id": None},
            loading_repo_ids = lambda: (),
        ),
    )
    monkeypatch.setattr(
        video_mod,
        "get_video_backend",
        lambda: SimpleNamespace(status = lambda: {"loaded": True, "repo_id": "unsloth/LTX-2.3-GGUF"}),
    )

    try:
        asyncio.run(
            models_route.delete_cached_model(
                repo_id = "unsloth/LTX-2.3-GGUF",
                variant = None,
                current_subject = "u",
            )
        )
        assert False, "expected HTTPException refusing the delete"
    except HTTPException as e:
        assert e.status_code == 400
        assert "Unload the model before deleting" in e.detail


def test_delete_cached_refuses_loaded_native_companion_repo(monkeypatch):
    # The native sd.cpp one-shot engine re-reads its companion VAE / text-encoder files from the
    # HF cache on every generation, so deleting a companion repo (comfyanonymous/flux_text_encoders)
    # while a FLUX GGUF is loaded must be refused. The loaded main repo_id does not match the
    # companion, so the guard relies on loaded_repo_ids() to cover the committed companions.
    from fastapi import HTTPException
    import core.inference.diffusion_engine_router as der
    import core.inference.video as video_mod
    import routes.inference as routes_inference

    monkeypatch.setattr(
        routes_inference,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(is_loaded = False, model_identifier = None),
    )
    monkeypatch.setattr(
        models_route,
        "get_inference_backend",
        lambda: SimpleNamespace(active_model_name = None),
    )
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
    monkeypatch.setattr(
        video_mod,
        "get_video_backend",
        lambda: SimpleNamespace(
            status = lambda: {"loaded": False, "repo_id": None},
            loading_repo_ids = lambda: (),
        ),
    )

    try:
        asyncio.run(
            models_route.delete_cached_model(
                repo_id = "comfyanonymous/flux_text_encoders",
                variant = None,
                current_subject = "u",
            )
        )
        assert False, "expected HTTPException refusing the in-use companion delete"
    except HTTPException as e:
        assert e.status_code == 400
        assert "Unload the model before deleting" in e.detail


def test_delete_cached_allows_sibling_of_loaded_diffusion_repo(monkeypatch):
    # A loaded Images repo must not block deleting a DIFFERENT cached repo that merely shares a
    # name prefix. Qwen/Qwen-Image and Qwen/Qwen-Image-2512 are both real catalog artifacts, so
    # with Qwen-Image-2512 loaded, deleting the sibling Qwen-Image is a supported operation. The
    # guard is `/`-boundary aware, so it refuses only the loaded repo (or a file within it), not
    # a prefix sibling.
    from fastapi import HTTPException
    import core.inference.diffusion_engine_router as der
    import core.inference.video as video_mod
    import routes.inference as routes_inference

    monkeypatch.setattr(
        routes_inference,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(is_loaded = False, model_identifier = None),
    )
    monkeypatch.setattr(
        models_route,
        "get_inference_backend",
        lambda: SimpleNamespace(active_model_name = None),
    )
    monkeypatch.setattr(
        der,
        "get_active_diffusion_engine",
        lambda: SimpleNamespace(
            status = lambda: {"loaded": True, "repo_id": "Qwen/Qwen-Image-2512"},
            loading_repo_ids = lambda: (),
        ),
    )
    monkeypatch.setattr(
        video_mod,
        "get_video_backend",
        lambda: SimpleNamespace(
            status = lambda: {"loaded": False, "repo_id": None},
            loading_repo_ids = lambda: (),
        ),
    )
    # Nothing cached, so a delete that clears the guards reaches the not-found path.
    monkeypatch.setattr(models_route, "_all_hf_cache_scans", lambda: [])

    # The sibling repo clears every guard and reaches the cache lookup, which 404s -- it is NOT
    # refused with the 400 "Unload" the un-delimited prefix match would have produced.
    try:
        asyncio.run(
            models_route.delete_cached_model(
                repo_id = "Qwen/Qwen-Image",
                variant = None,
                current_subject = "u",
            )
        )
        assert False, "expected HTTPException (404 not-found) after clearing the guard"
    except HTTPException as e:
        assert e.status_code == 404, f"sibling delete wrongly blocked: {e.status_code} {e.detail}"

    # The loaded repo itself is still refused (exact match).
    try:
        asyncio.run(
            models_route.delete_cached_model(
                repo_id = "Qwen/Qwen-Image-2512",
                variant = None,
                current_subject = "u",
            )
        )
        assert False, "expected HTTPException refusing delete of the loaded repo"
    except HTTPException as e:
        assert e.status_code == 400
        assert "Unload the model before deleting" in e.detail


def test_cached_repo_partial_scopes_probe_to_snapshot_dir(monkeypatch):
    # The partial probe must be scoped to the snapshot row being listed. Unscoped, the scan
    # spans every HF cache root, so a stale .incomplete copy in one root would flag a complete
    # copy living in another root as partial and hide the usable model from the picker. Verify
    # _cached_repo_partial forwards the snapshot dir (matching the sibling inventory paths).
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

    # When that specific snapshot is partial, the row is flagged.
    monkeypatch.setattr(scan, "is_snapshot_partial", lambda *a, **k: True)
    assert models_route._cached_repo_partial("Org/Repo", snapshot_dir) is True

    # A probe error is swallowed (never hides a usable repo over a scan glitch).
    def _boom(*a, **k):
        raise RuntimeError("scan glitch")

    monkeypatch.setattr(scan, "is_snapshot_partial", _boom)
    assert models_route._cached_repo_partial("Org/Repo", snapshot_dir) is False


def test_repo_has_pipeline_index_requires_root_model_index(tmp_path):
    # Only a ROOT model_index.json makes a repo pipeline-loadable: from_pretrained
    # reads the repo root, so a nested subdir/model_index.json must NOT clear the
    # single_file flag. CachedFileInfo.file_name is the basename, so the helper has
    # to scope by file_path/snapshot_path -- a name-only match would claim both.
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


def test_list_cached_models_flags_single_file_diffusion_repos(monkeypatch, tmp_path):
    # A diffusion-tagged repo with NO top-level model_index.json is a single-file
    # checkpoint: the task pickers must not offer it as a pipeline load (from_pretrained
    # fails on it), so the row carries single_file=True. A full pipeline repo (has
    # model_index.json) and a chat repo (task None) carry no flag.
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
