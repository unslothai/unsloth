# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

# Keep this test runnable in lightweight environments where optional logging
# deps are not installed.
if "structlog" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

import pytest

import routes.models as models_route


@pytest.fixture(autouse = True)
def _stub_classify(monkeypatch):
    """Keep the disk-scan list endpoints offline: ``_classify_cached_repos``
    hits the live HF API, which makes these unit tests flake on a slow or
    sandboxed network. Stub it to a no-op so each row keeps the disk-derived
    fields the assertions actually pin."""

    async def _noop(repos, hf_token):
        return None

    monkeypatch.setattr(models_route, "_classify_cached_repos", _noop)


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
    file_path: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        file_name = name,
        size_on_disk = size_on_disk,
        blob_path = blob_path,
        file_path = file_path,
    )


def _mark_repo_partial(repo_path: Path) -> None:
    blobs = repo_path / "blobs"
    blobs.mkdir(parents = True, exist_ok = True)
    (blobs / "stale.incomplete").write_bytes(b"x")


def _legacy_cached(rows: list[dict]) -> list[dict]:
    contract_fields = {
        "inventory_id",
        "load_id",
        "model_format",
        "runtime",
        "format_variant",
        "capabilities",
    }
    return [
        {key: value for key, value in row.items() if key not in contract_fields}
        for row in rows
    ]


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


def test_list_cached_gguf_includes_non_suffix_repo_when_cache_contains_gguf(
    monkeypatch, tmp_path
):
    repo = _repo(
        "HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive",
        [_file("Q4_K_M.gguf", 5_000), _file("README.md", 10)],
        tmp_path / "models--HauhauCS--Gemma",
    )
    scan = SimpleNamespace(repos = [repo])

    monkeypatch.setattr(models_route, "_all_hf_cache_scans", lambda: [scan])

    result = asyncio.run(models_route.list_cached_gguf(hf_token = None, current_subject = "test-user"))

    assert _legacy_cached(result["cached"]) == [
        {
            "repo_id": "HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive",
            "size_bytes": 5_000,
            "cache_path": str(repo.repo_path),
            "partial": False,
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

    result = asyncio.run(models_route.list_cached_gguf(hf_token = None, current_subject = "test-user"))

    assert _legacy_cached(result["cached"]) == [
        {
            "repo_id": "Org/Model-Without-Suffix",
            "size_bytes": 7_000,
            "cache_path": str(repo.repo_path),
            "partial": False,
        }
    ]


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

    result = asyncio.run(models_route.list_cached_gguf(hf_token = None, current_subject = "test-user"))

    assert _legacy_cached(result["cached"]) == []


def test_list_cached_gguf_keeps_largest_duplicate_repo_across_scans(
    monkeypatch, tmp_path
):
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

    result = asyncio.run(models_route.list_cached_gguf(hf_token = None, current_subject = "test-user"))

    assert _legacy_cached(result["cached"]) == [
        {
            "repo_id": "org/dupe",
            "size_bytes": 6_000,
            "cache_path": str(larger.repo_path),
            "partial": False,
        }
    ]


def test_list_cached_gguf_prefers_complete_duplicate_over_larger_partial(
    monkeypatch, tmp_path
):
    partial_path = tmp_path / "root-a" / "models--Org--Dupe"
    complete_path = tmp_path / "root-b" / "models--Org--Dupe"
    _mark_repo_partial(partial_path)
    complete_path.mkdir(parents = True)
    partial = _repo(
        "Org/Dupe",
        [_file("Q4_K_M.gguf", 8_000)],
        partial_path,
    )
    complete = _repo(
        "org/dupe",
        [_file("Q4_K_M.gguf", 5_000)],
        complete_path,
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [
            SimpleNamespace(repos = [partial]),
            SimpleNamespace(repos = [complete]),
        ],
    )

    result = asyncio.run(models_route.list_cached_gguf(hf_token = None, current_subject = "test-user"))

    assert _legacy_cached(result["cached"]) == [
        {
            "repo_id": "org/dupe",
            "size_bytes": 5_000,
            "cache_path": str(complete_path),
            "partial": False,
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

    result = asyncio.run(models_route.list_cached_gguf(hf_token = None, current_subject = "test-user"))

    assert _legacy_cached(result["cached"]) == [
        {
            "repo_id": "Org/SharedBlobRepo",
            "size_bytes": 5_000,
            "cache_path": str(repo.repo_path),
            "partial": False,
        }
    ]


def test_list_cached_models_includes_non_gguf_weights_when_gguf_files_exist(
    monkeypatch, tmp_path
):
    mixed = _repo(
        "Org/MixedRepo",
        [
            _file("Q4_K_M.gguf", 5_000),
            _file("config.json", 100),
            _file("model.safetensors", 10_000),
        ],
        tmp_path / "models--Org--MixedRepo",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [mixed])],
    )

    result = asyncio.run(
        models_route.list_cached_models(
            hf_token = None,
            current_subject = "test-user",
        )
    )

    assert _legacy_cached(result["cached"]) == [
        {
            "repo_id": "Org/MixedRepo",
            "size_bytes": 10_000,
            "cache_path": str(mixed.repo_path),
            "partial": False,
        }
    ]


def test_list_cached_models_includes_standard_safetensors_without_config(
    monkeypatch, tmp_path
):
    cached = _repo(
        "Org/SafetensorsOnly",
        [_file("model-00001-of-00002.safetensors", 10_000)],
        tmp_path / "models--Org--SafetensorsOnly",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [cached])],
    )

    result = asyncio.run(
        models_route.list_cached_models(
            hf_token = None,
            current_subject = "test-user",
        )
    )

    assert _legacy_cached(result["cached"]) == [
        {
            "repo_id": "Org/SafetensorsOnly",
            "size_bytes": 10_000,
            "cache_path": str(cached.repo_path),
            "partial": False,
        }
    ]
    row = result["cached"][0]
    assert row["model_format"] == "safetensors"
    assert row["runtime"] == "transformers"
    assert row["capabilities"]["can_train"] is True
    assert row["capabilities"]["can_chat"] is True


def test_list_cached_models_skips_aux_safetensors_without_config(
    monkeypatch, tmp_path
):
    cached = _repo(
        "Org/AuxWeights",
        [_file("diffusion_pytorch_model.safetensors", 10_000)],
        tmp_path / "models--Org--AuxWeights",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [cached])],
    )

    result = asyncio.run(
        models_route.list_cached_models(
            hf_token = None,
            current_subject = "test-user",
        )
    )

    assert _legacy_cached(result["cached"]) == []


def test_list_cached_models_prefers_complete_duplicate_over_larger_partial(
    monkeypatch, tmp_path
):
    partial_path = tmp_path / "root-a" / "models--Org--Model"
    complete_path = tmp_path / "root-b" / "models--Org--Model"
    _mark_repo_partial(partial_path)
    complete_path.mkdir(parents = True)
    partial = _repo(
        "Org/Model",
        [_file("config.json", 100), _file("model.safetensors", 12_000)],
        partial_path,
    )
    complete = _repo(
        "org/model",
        [_file("config.json", 100), _file("model.safetensors", 7_000)],
        complete_path,
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [
            SimpleNamespace(repos = [partial]),
            SimpleNamespace(repos = [complete]),
        ],
    )

    result = asyncio.run(models_route.list_cached_models(hf_token = None, current_subject = "test-user"))

    assert _legacy_cached(result["cached"]) == [
        {
            "repo_id": "org/model",
            "size_bytes": 7_000,
            "cache_path": str(complete_path),
            "partial": False,
        }
    ]


def test_list_cached_models_skips_repo_with_only_main_gguf(monkeypatch, tmp_path):
    gguf_only = _repo(
        "Org/GgufOnly",
        [_file("Q4_K_M.gguf", 5_000)],
        tmp_path / "models--Org--GgufOnly",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [gguf_only])],
    )

    result = asyncio.run(models_route.list_cached_models(hf_token = None, current_subject = "test-user"))

    assert _legacy_cached(result["cached"]) == []


def test_list_cached_models_skips_large_tokenizer_config_only_repo(
    monkeypatch, tmp_path
):
    tokenizer_only = _repo(
        "Org/TokenizerOnly",
        [
            _file("config.json", 100),
            _file("tokenizer.json", 2_000_000),
            _file("tokenizer.model", 1_000_000),
        ],
        tmp_path / "models--Org--TokenizerOnly",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [tokenizer_only])],
    )

    result = asyncio.run(models_route.list_cached_models(hf_token = None, current_subject = "test-user"))

    assert _legacy_cached(result["cached"]) == []


def test_list_cached_models_skips_adapter_config_without_weights(
    monkeypatch, tmp_path
):
    adapter_config_only = _repo(
        "Org/AdapterConfigOnly",
        [
            _file("adapter_config.json", 500),
            _file("README.md", 1_500_000),
        ],
        tmp_path / "models--Org--AdapterConfigOnly",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [adapter_config_only])],
    )

    result = asyncio.run(models_route.list_cached_models(hf_token = None, current_subject = "test-user"))

    assert _legacy_cached(result["cached"]) == []


def test_list_cached_models_includes_complete_adapter_repo(monkeypatch, tmp_path):
    adapter = _repo(
        "Org/Adapter",
        [
            _file("adapter_config.json", 500),
            _file("adapter_model.safetensors", 8_000),
        ],
        tmp_path / "models--Org--Adapter",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [adapter])],
    )

    result = asyncio.run(models_route.list_cached_models(hf_token = None, current_subject = "test-user"))

    assert _legacy_cached(result["cached"]) == [
        {
            "repo_id": "Org/Adapter",
            "size_bytes": 8_000,
            "cache_path": str(adapter.repo_path),
            "partial": False,
        }
    ]
    row = result["cached"][0]
    assert row["model_format"] == "adapter"
    assert row["capabilities"]["can_train"] is False
    assert row["capabilities"]["can_chat"] is True


def test_list_cached_models_includes_complete_checkpoint_repo(monkeypatch, tmp_path):
    checkpoint = _repo(
        "Org/Checkpoint",
        [
            _file("config.json", 500),
            _file("pytorch_model.bin", 8_000),
        ],
        tmp_path / "models--Org--Checkpoint",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [checkpoint])],
    )

    result = asyncio.run(models_route.list_cached_models(hf_token = None, current_subject = "test-user"))

    assert _legacy_cached(result["cached"]) == [
        {
            "repo_id": "Org/Checkpoint",
            "size_bytes": 8_000,
            "cache_path": str(checkpoint.repo_path),
            "partial": False,
        }
    ]
    row = result["cached"][0]
    assert row["model_format"] == "checkpoint"
    assert row["capabilities"]["can_train"] is True


def test_list_cached_gguf_includes_mixed_repo_with_gguf_and_safetensors(
    monkeypatch, tmp_path
):
    """Mirror of the _skips_ test: the mixed repo should still surface in
    cached-gguf so the picker can show it as a GGUF download."""
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

    result = asyncio.run(models_route.list_cached_gguf(hf_token = None, current_subject = "test-user"))

    assert _legacy_cached(result["cached"]) == [
        {
            "repo_id": "Org/MixedRepo",
            "size_bytes": 5_000,
            "cache_path": str(mixed.repo_path),
            "partial": False,
        }
    ]
    row = result["cached"][0]
    assert row["inventory_id"] == "cache:gguf:Org%2FMixedRepo"
    assert row["load_id"] == "Org/MixedRepo"
    assert row["model_format"] == "gguf"
    assert row["runtime"] == "llama_cpp"
    assert row["capabilities"]["can_train"] is False
    assert row["capabilities"]["requires_variant"] is True


def test_cached_mixed_repo_has_distinct_gguf_and_safetensors_rows(
    monkeypatch, tmp_path
):
    mixed = _repo(
        "Org/MixedRepo",
        [
            _file("Q4_K_M.gguf", 5_000),
            _file("config.json", 100),
            _file("model.safetensors", 10_000),
        ],
        tmp_path / "models--Org--MixedRepo",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [mixed])],
    )

    gguf = asyncio.run(
        models_route.list_cached_gguf(hf_token = None, current_subject = "test-user")
    )["cached"][0]
    safetensors = asyncio.run(
        models_route.list_cached_models(hf_token = None, current_subject = "test-user")
    )["cached"][0]

    assert gguf["repo_id"] == safetensors["repo_id"] == "Org/MixedRepo"
    assert gguf["load_id"] == safetensors["load_id"] == "Org/MixedRepo"
    assert gguf["model_format"] == "gguf"
    assert safetensors["model_format"] == "safetensors"
    assert gguf["inventory_id"] != safetensors["inventory_id"]
    assert safetensors["capabilities"]["can_train"] is True


def test_list_cached_gguf_handles_none_size_on_disk(monkeypatch, tmp_path):
    """A partial/interrupted GGUF download has ``size_on_disk = None``. The
    route must treat the unknown bytes as zero instead of raising TypeError
    out of ``sum()`` and wiping the entire response."""
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

    result = asyncio.run(models_route.list_cached_gguf(hf_token = None, current_subject = "test-user"))

    assert _legacy_cached(result["cached"]) == [
        {
            "repo_id": "Org/PartialDownload",
            "size_bytes": 5_000,
            "cache_path": str(partial.repo_path),
            "partial": False,
        }
    ]


def test_list_cached_gguf_skips_malformed_repo_without_wiping_response(
    monkeypatch, tmp_path
):
    """One repo raising during classification must not poison the response
    for every other repo in the scan."""

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

    result = asyncio.run(models_route.list_cached_gguf(hf_token = None, current_subject = "test-user"))

    assert _legacy_cached(result["cached"]) == [
        {
            "repo_id": "Org/Healthy",
            "size_bytes": 5_000,
            "cache_path": str(healthy.repo_path),
            "partial": False,
        }
    ]


def test_list_cached_gguf_skips_repo_with_only_mmproj_gguf(monkeypatch, tmp_path):
    """A repo whose only ``.gguf`` artifact is an mmproj vision adapter
    must not be classified as a GGUF repo: the variant selector filters
    mmproj out and the picker would otherwise show zero variants."""
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

    result = asyncio.run(models_route.list_cached_gguf(hf_token = None, current_subject = "test-user"))

    assert _legacy_cached(result["cached"]) == []


def test_list_cached_models_includes_repo_with_only_mmproj_gguf(monkeypatch, tmp_path):
    """Mirror of the cached-gguf skip: a safetensors repo with an
    auxiliary mmproj vision adapter must still surface in cached-models
    so the user can load it as a normal model."""
    mmproj_aux = _repo(
        "Org/MmprojAux",
        [
            _file("mmproj-Q8_0.gguf", 5_000),
            _file("config.json", 100),
            _file("model.safetensors", 10_000),
        ],
        tmp_path / "models--Org--MmprojAux",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [mmproj_aux])],
    )

    result = asyncio.run(models_route.list_cached_models(hf_token = None, current_subject = "test-user"))

    assert _legacy_cached(result["cached"]) == [
        {
            "repo_id": "Org/MmprojAux",
            "size_bytes": 10_000,
            "cache_path": str(mmproj_aux.repo_path),
            "partial": False,
        }
    ]


def test_list_cached_gguf_includes_vision_repo_with_main_gguf_and_mmproj(
    monkeypatch, tmp_path
):
    """A vision-capable GGUF repo (main weight + mmproj adapter) is still
    a GGUF repo. The reported size is the main weight size; mmproj is
    excluded from the GGUF-size accounting because it is filtered out at
    classification time."""
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

    result = asyncio.run(models_route.list_cached_gguf(hf_token = None, current_subject = "test-user"))

    assert _legacy_cached(result["cached"]) == [
        {
            "repo_id": "Org/VisionGguf",
            "size_bytes": 5_000,
            "cache_path": str(vision_repo.repo_path),
            "partial": False,
        }
    ]


def test_gguf_variant_requirements_include_shared_mmproj():
    siblings = [
        SimpleNamespace(
            rfilename = "model-Q4_K_M.gguf",
            size = 4_000,
            lfs = SimpleNamespace(sha256 = "main-q4"),
        ),
        SimpleNamespace(
            rfilename = "model-Q8_0.gguf",
            size = 8_000,
            lfs = SimpleNamespace(sha256 = "main-q8"),
        ),
        SimpleNamespace(
            rfilename = "mmproj-F16.gguf",
            size = 1_000,
            lfs = SimpleNamespace(sha256 = "projector"),
        ),
    ]

    requirements = models_route._build_gguf_variant_requirements(siblings)

    q4 = requirements["q4_k_m"]
    assert q4.main_filenames == frozenset({"model-Q4_K_M.gguf"})
    assert q4.required_filenames == frozenset(
        {"model-Q4_K_M.gguf", "mmproj-F16.gguf"}
    )
    assert q4.main_hashes == frozenset({"main-q4"})
    assert q4.required_hashes == frozenset({"main-q4", "projector"})
    assert q4.mmproj_filenames == frozenset({"mmproj-F16.gguf"})
    assert q4.mmproj_hashes == frozenset({"projector"})
    assert q4.main_size_bytes == 4_000
    assert q4.download_size_bytes == 5_000


def test_preferred_mmproj_picks_f16_not_bf16():
    """`mmproj-BF16` lexically contains ``bf16`` which is a superstring of
    ``f16``. The earlier substring check picked BF16 first whenever it
    appeared before F16 in the API sibling order, which then poisoned the
    downloaded-state check for caches that only ship mmproj-F16."""
    siblings = [
        SimpleNamespace(
            rfilename = "mmproj-BF16.gguf",
            size = 2_000,
            lfs = SimpleNamespace(sha256 = "proj-bf16"),
        ),
        SimpleNamespace(
            rfilename = "mmproj-F16.gguf",
            size = 1_000,
            lfs = SimpleNamespace(sha256 = "proj-f16"),
        ),
        SimpleNamespace(
            rfilename = "mmproj-F32.gguf",
            size = 4_000,
            lfs = SimpleNamespace(sha256 = "proj-f32"),
        ),
    ]

    preferred = models_route._preferred_mmproj_sibling(siblings)

    assert preferred is not None
    assert preferred.rfilename == "mmproj-F16.gguf"


def test_preferred_mmproj_falls_back_when_no_f16():
    siblings = [
        SimpleNamespace(
            rfilename = "mmproj-BF16.gguf",
            size = 2_000,
            lfs = SimpleNamespace(sha256 = "proj-bf16"),
        ),
        SimpleNamespace(
            rfilename = "mmproj-Q8_0.gguf",
            size = 1_500,
            lfs = SimpleNamespace(sha256 = "proj-q8"),
        ),
    ]

    preferred = models_route._preferred_mmproj_sibling(siblings)

    assert preferred is not None
    # Falls back to the first candidate when there's no F16 to prefer.
    assert preferred.rfilename == "mmproj-BF16.gguf"


def test_gguf_variant_requirements_collect_every_mmproj_candidate():
    siblings = [
        SimpleNamespace(
            rfilename = "model-Q4_K_M.gguf",
            size = 4_000,
            lfs = SimpleNamespace(sha256 = "main-q4"),
        ),
        SimpleNamespace(
            rfilename = "mmproj-BF16.gguf",
            size = 2_000,
            lfs = SimpleNamespace(sha256 = "proj-bf16"),
        ),
        SimpleNamespace(
            rfilename = "mmproj-F16.gguf",
            size = 1_000,
            lfs = SimpleNamespace(sha256 = "proj-f16"),
        ),
        SimpleNamespace(
            rfilename = "mmproj-F32.gguf",
            size = 4_000,
            lfs = SimpleNamespace(sha256 = "proj-f32"),
        ),
    ]

    requirements = models_route._build_gguf_variant_requirements(siblings)
    q4 = requirements["q4_k_m"]

    # All mmproj precisions are valid satisfiers of the vision requirement.
    assert q4.mmproj_filenames == frozenset(
        {"mmproj-BF16.gguf", "mmproj-F16.gguf", "mmproj-F32.gguf"},
    )
    assert q4.mmproj_hashes == frozenset({"proj-bf16", "proj-f16", "proj-f32"})
    # The download bundle still uses the canonical preferred mmproj (F16).
    assert q4.required_filenames == frozenset(
        {"model-Q4_K_M.gguf", "mmproj-F16.gguf"},
    )
    assert q4.required_hashes == frozenset({"main-q4", "proj-f16"})


def test_gguf_variants_downloaded_accepts_any_cached_mmproj(
    monkeypatch, tmp_path
):
    """A repo whose remote sibling list ranks mmproj-BF16 first but whose
    on-disk cache ships only mmproj-F16 must still report the variant as
    downloaded. Regression for the substring-matched preferred-mmproj bug."""
    repo_root = tmp_path / "models--Org--VisionGguf"
    blobs_dir = repo_root / "blobs"
    snap_dir = repo_root / "snapshots" / "rev"
    blobs_dir.mkdir(parents = True)
    snap_dir.mkdir(parents = True)

    main_blob = blobs_dir / "main-q3"
    main_blob.write_bytes(b"x" * 4_000)
    main_snap = snap_dir / "model-Q3_K_M.gguf"
    main_snap.symlink_to(main_blob)

    mmproj_blob = blobs_dir / "mmproj-f16"
    mmproj_blob.write_bytes(b"y" * 1_000)
    mmproj_snap = snap_dir / "mmproj-F16.gguf"
    mmproj_snap.symlink_to(mmproj_blob)

    monkeypatch.setattr(
        models_route.hf_cache_scan,
        "iter_repo_cache_dirs",
        lambda repo_type, repo_id: iter([repo_root])
        if repo_type == "model" and repo_id == "Org/VisionGguf"
        else iter([]),
    )
    monkeypatch.setattr(
        models_route.hf_cache_scan,
        "incomplete_blob_hashes",
        lambda repo_type, repo_id: set(),
    )

    siblings = [
        SimpleNamespace(
            rfilename = "model-Q3_K_M.gguf",
            size = 4_000,
            lfs = SimpleNamespace(sha256 = "main-q3"),
        ),
        SimpleNamespace(
            rfilename = "mmproj-BF16.gguf",
            size = 2_000,
            lfs = SimpleNamespace(sha256 = "proj-bf16"),
        ),
        SimpleNamespace(
            rfilename = "mmproj-F16.gguf",
            size = 1_000,
            lfs = SimpleNamespace(sha256 = "proj-f16"),
        ),
    ]

    def _fake_list_gguf_variants(repo_id, hf_token = None):
        from utils.models.model_config import GgufVariantInfo
        return [
            GgufVariantInfo(
                filename = "model-Q3_K_M.gguf",
                quant = "Q3_K_M",
                size_bytes = 4_000,
            ),
        ], True

    monkeypatch.setattr(models_route, "list_gguf_variants", _fake_list_gguf_variants)
    monkeypatch.setattr(
        models_route,
        "_gguf_all_variant_requirements",
        lambda repo_id, hf_token: models_route._build_gguf_variant_requirements(
            siblings,
        ),
    )

    response = asyncio.run(
        models_route.get_gguf_variants(
            repo_id = "Org/VisionGguf",
            hf_token = None,
            current_subject = "test-user",
        ),
    )

    assert len(response.variants) == 1
    q3 = response.variants[0]
    assert q3.quant == "Q3_K_M"
    assert q3.downloaded is True
    assert q3.partial is False


def test_gguf_variants_uses_selected_cache_path_without_remote_fallback(
    monkeypatch, tmp_path
):
    repo_root = tmp_path / "models--Org--Local-GGUF"
    snap_dir = repo_root / "snapshots" / "rev"
    snap_dir.mkdir(parents = True)
    (snap_dir / "model-Q4_K_M.gguf").write_bytes(b"x" * 4_000)

    def _remote_called(*_args, **_kwargs):
        raise AssertionError("remote GGUF variant lookup should not run")

    monkeypatch.setattr(models_route, "list_gguf_variants", _remote_called)

    response = asyncio.run(
        models_route.get_gguf_variants(
            repo_id = "Org/Local-GGUF",
            prefer_local_cache = False,
            offline = False,
            local_path = str(repo_root),
            hf_token = None,
            current_subject = "test-user",
        ),
    )

    assert response.repo_id == "Org/Local-GGUF"
    assert response.default_variant == "Q4_K_M"
    assert [(v.quant, v.downloaded, v.size_bytes) for v in response.variants] == [
        ("Q4_K_M", True, 4_000)
    ]


def test_gguf_variants_ignores_stale_local_path_while_online(
    monkeypatch, tmp_path
):
    stale_repo_root = tmp_path / "models--Org--Stale-GGUF"

    def _fake_list_gguf_variants(repo_id, hf_token = None):
        from utils.models.model_config import GgufVariantInfo

        return [
            GgufVariantInfo(
                filename = "model-Q4_K_M.gguf",
                quant = "Q4_K_M",
                size_bytes = 4_000,
            ),
        ], False

    monkeypatch.setattr(models_route, "list_gguf_variants", _fake_list_gguf_variants)
    monkeypatch.setattr(models_route, "_gguf_all_variant_requirements", lambda *_a: {})
    monkeypatch.setattr(
        models_route.hf_cache_scan,
        "iter_repo_cache_dirs",
        lambda *_a: [],
    )
    monkeypatch.setattr(
        models_route.hf_cache_scan,
        "incomplete_blob_hashes",
        lambda *_a: set(),
    )

    response = asyncio.run(
        models_route.get_gguf_variants(
            repo_id = "Org/Stale-GGUF",
            prefer_local_cache = False,
            offline = False,
            local_path = str(stale_repo_root),
            hf_token = None,
            current_subject = "test-user",
        ),
    )

    assert response.repo_id == "Org/Stale-GGUF"
    assert [(v.quant, v.downloaded, v.size_bytes) for v in response.variants] == [
        ("Q4_K_M", False, 4_000)
    ]


def test_gguf_variants_merges_remote_catalog_with_selected_cache_path(
    monkeypatch, tmp_path
):
    repo_root = tmp_path / "models--Org--Mixed-GGUF"
    snap_dir = repo_root / "snapshots" / "rev"
    snap_dir.mkdir(parents = True)
    (snap_dir / "model-Q4_K_M.gguf").write_bytes(b"x" * 4_000)

    def _fake_list_gguf_variants(repo_id, hf_token = None):
        from utils.models.model_config import GgufVariantInfo

        return [
            GgufVariantInfo(
                filename = "model-Q4_K_M.gguf",
                quant = "Q4_K_M",
                size_bytes = 4_000,
            ),
            GgufVariantInfo(
                filename = "model-Q8_0.gguf",
                quant = "Q8_0",
                size_bytes = 8_000,
            ),
        ], False

    monkeypatch.setattr(models_route, "list_gguf_variants", _fake_list_gguf_variants)
    monkeypatch.setattr(models_route, "_gguf_all_variant_requirements", lambda *_a: {})
    monkeypatch.setattr(
        models_route.hf_cache_scan,
        "incomplete_blob_hashes",
        lambda *_a: set(),
    )

    response = asyncio.run(
        models_route.get_gguf_variants(
            repo_id = "Org/Mixed-GGUF",
            prefer_local_cache = False,
            offline = False,
            local_path = str(repo_root),
            hf_token = None,
            current_subject = "test-user",
        ),
    )

    assert response.repo_id == "Org/Mixed-GGUF"
    assert [(v.quant, v.downloaded, v.size_bytes) for v in response.variants] == [
        ("Q4_K_M", True, 4_000),
        ("Q8_0", False, 8_000),
    ]


def test_cached_variant_delete_keeps_shared_mmproj_for_same_quant(
    monkeypatch, tmp_path
):
    repo_root = tmp_path / "models--Org--Vision"
    blob_dir = repo_root / "blobs"
    snap_dir = repo_root / "snapshots" / "rev"
    blob_dir.mkdir(parents = True)
    snap_dir.mkdir(parents = True)
    main_blob = blob_dir / "main-f16"
    main_snap = snap_dir / "model-F16.gguf"
    mmproj_blob = blob_dir / "mmproj-f16"
    mmproj_snap = snap_dir / "mmproj-F16.gguf"
    main_blob.write_bytes(b"main")
    main_snap.write_bytes(b"main")
    mmproj_blob.write_bytes(b"proj")
    mmproj_snap.write_bytes(b"proj")

    repo = _repo(
        "Org/Vision",
        [
            _file(
                "model-F16.gguf",
                main_blob.stat().st_size,
                blob_path = str(main_blob),
                file_path = str(main_snap),
            ),
            _file(
                "mmproj-F16.gguf",
                mmproj_blob.stat().st_size,
                blob_path = str(mmproj_blob),
                file_path = str(mmproj_snap),
            ),
        ],
        repo_root,
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [repo])],
    )
    monkeypatch.setattr(
        models_route,
        "_delete_variant_incomplete_blobs",
        lambda repo_id, variant, hf_token: 0,
    )

    result = models_route._delete_cached_model_blocking("Org/Vision", "F16", None)

    assert result == {"status": "deleted", "repo_id": "Org/Vision", "variant": "F16"}
    assert not main_blob.exists()
    assert not main_snap.exists()
    assert mmproj_blob.exists()
    assert mmproj_snap.exists()


def test_cached_model_delete_falls_back_to_repo_dir_purge(monkeypatch, tmp_path):
    repo_root = tmp_path / "models--Org--Partial"
    snapshot = repo_root / "snapshots" / "rev"
    blobs = repo_root / "blobs"
    snapshot.mkdir(parents = True)
    blobs.mkdir()
    (snapshot / "config.json").write_text("{}")
    (blobs / "config").write_text("{}")

    monkeypatch.setattr(models_route, "_all_hf_cache_scans", lambda: [])
    monkeypatch.setattr(
        models_route.hf_cache_scan,
        "_hf_cache_roots",
        lambda: [tmp_path],
    )

    result = models_route._delete_cached_model_blocking("Org/Partial", None, None)

    assert result == {"status": "deleted", "repo_id": "Org/Partial"}
    assert not repo_root.exists()


def test_loaded_repo_variant_delete_guard_allows_inactive_quant():
    assert (
        models_route._loaded_repo_variant_blocks_delete(
            "org/model-gguf",
            "Org/Model-GGUF",
            "Q8_0",
            "Q4_K_M",
        )
        is False
    )
    assert (
        models_route._loaded_repo_variant_blocks_delete(
            "org/model-gguf",
            "Org/Model-GGUF",
            "Q4_K_M",
            "Q4_K_M",
        )
        is True
    )
    assert (
        models_route._loaded_repo_variant_blocks_delete(
            "org/model-gguf",
            "Org/Model-GGUF",
            "Q8_0",
            None,
        )
        is True
    )
