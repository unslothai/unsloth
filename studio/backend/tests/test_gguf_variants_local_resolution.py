# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Load-path parity for /api/models/gguf-variants local resolution."""

import asyncio
import os

import pytest

from hub.services.models.gguf_variants import get_gguf_variants_response


def _variants(repo_id: str, **kwargs):
    return asyncio.run(get_gguf_variants_response(repo_id, **kwargs))


@pytest.fixture()
def in_tmp_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_markerless_relative_dir_resolves_locally(in_tmp_cwd):
    gguf_dir = in_tmp_cwd / "models" / "qwen"
    gguf_dir.mkdir(parents = True)
    (gguf_dir / "qwen-Q4_K_M.gguf").write_bytes(b"GGUF")

    response = _variants("models/qwen")
    assert [v.quant for v in response.variants] == ["Q4_K_M"]


def test_direct_gguf_file_is_a_loadable_variant(in_tmp_cwd):
    gguf = in_tmp_cwd / "foo-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")

    response = _variants(os.fspath(gguf))
    assert [v.filename for v in response.variants] == ["foo-Q4_K_M.gguf"]
    assert response.variants[0].quant == "Q4_K_M"
    # The file is the model: the shard scan cannot walk it, so its empty answer
    # must not mark the only row partial (the picker disables partial local rows).
    assert response.variants[0].downloaded is True
    assert response.variants[0].partial is False


def test_markerless_relative_gguf_file_resolves_locally(in_tmp_cwd):
    (in_tmp_cwd / "models").mkdir()
    (in_tmp_cwd / "models" / "foo.gguf").write_bytes(b"GGUF")

    response = _variants("models/foo.gguf")
    assert [v.filename for v in response.variants] == ["foo.gguf"]


def test_nonexistent_local_syntax_path_still_returns_empty(in_tmp_cwd):
    response = _variants(os.fspath(in_tmp_cwd / "missing-dir"))
    assert response.variants == []


def test_direct_gguf_file_in_marked_dir_still_lists_siblings(in_tmp_cwd):
    # The load path resolves a .gguf in a marked directory to the whole
    # directory, so the listing keeps sibling quants and the vision flag.
    (in_tmp_cwd / "config.json").write_text("{}")
    (in_tmp_cwd / "model-Q4_K_M.gguf").write_bytes(b"GGUF")
    (in_tmp_cwd / "model-Q8_0.gguf").write_bytes(b"GGUF" * 2)
    (in_tmp_cwd / "mmproj-F16.gguf").write_bytes(b"GGUF")

    response = _variants(os.fspath(in_tmp_cwd / "model-Q4_K_M.gguf"))
    assert sorted(v.quant for v in response.variants) == ["Q4_K_M", "Q8_0"]
    assert response.has_vision is True
    # A marked parent is still scanned for completeness, so whole quants stay ready.
    assert all(v.downloaded for v in response.variants)


@pytest.mark.parametrize(
    "relpath",
    [
        "mmproj-F16.gguf",
        "mtp-model-Q4_K_M.gguf",
        "MTP/model-Q8_0-MTP.gguf",
        "dspark/dspark-model-Q8_0.gguf",
        "stories260K-be.gguf",
    ],
)
def test_direct_auxiliary_gguf_file_is_not_a_variant(in_tmp_cwd, relpath):
    # detect_gguf_model refuses the companions and big-endian builds, so a row
    # for one would offer a load that cannot happen.
    from utils.models.model_config import detect_gguf_model

    target = in_tmp_cwd / relpath
    target.parent.mkdir(parents = True, exist_ok = True)
    target.write_bytes(b"GGUF")

    assert detect_gguf_model(os.fspath(target)) is None
    assert _variants(os.fspath(target)).variants == []


def test_direct_gguf_file_quant_round_trips_through_the_load_path(in_tmp_cwd):
    # Clients echo the selected quant back as gguf_variant, so the quant this
    # endpoint advertises has to resolve for the same identifier -- otherwise
    # the file loads without a variant and fails with the one just offered.
    from utils.models.model_config import ModelConfig

    gguf = in_tmp_cwd / "foo-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")

    quant = _variants(os.fspath(gguf)).variants[0].quant
    config = ModelConfig.from_identifier(os.fspath(gguf), gguf_variant = quant)
    assert config is not None and config.is_gguf
    assert config.gguf_file == os.fspath(gguf)
    # A quant the file is not stays unresolved rather than loading other weights.
    assert ModelConfig.from_identifier(os.fspath(gguf), gguf_variant = "Q8_0").is_gguf is False


def test_direct_gguf_file_quant_round_trips_case_insensitively(in_tmp_cwd):
    # llama.cpp matches a quant label case-insensitively, and so does the CLI's
    # pre-load gate, so a typed lowercase --gguf-variant has to resolve to the
    # same file here; resolving nothing loads no GGUF and evicts the resident
    # model on the transformers path before failing.
    from utils.models.model_config import ModelConfig

    gguf = in_tmp_cwd / "foo-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")

    config = ModelConfig.from_identifier(os.fspath(gguf), gguf_variant = "q4_k_m")
    assert config is not None and config.is_gguf
    assert config.gguf_file == os.fspath(gguf)
    # A directory of the same weights answers the same spelling.
    marked = in_tmp_cwd / "marked"
    marked.mkdir()
    (marked / "config.json").write_text("{}")
    (marked / "foo-Q4_K_M.gguf").write_bytes(b"GGUF")
    dir_config = ModelConfig.from_identifier(os.fspath(marked), gguf_variant = "q4_k_m")
    assert dir_config is not None and dir_config.is_gguf


def test_direct_gguf_bpw_label_round_trips_through_the_load_path(in_tmp_cwd):
    # The hub-side extractor drops the bpw modifier, so the advertised quant is
    # the shorter label; echoing it back must still resolve this same file.
    from utils.models.model_config import ModelConfig

    gguf = in_tmp_cwd / "model-IQ4_XS-3.53bpw.gguf"
    gguf.write_bytes(b"GGUF")

    quant = _variants(os.fspath(gguf)).variants[0].quant
    config = ModelConfig.from_identifier(os.fspath(gguf), gguf_variant = quant)
    assert config is not None and config.is_gguf
    assert config.gguf_file == os.fspath(gguf)


def test_torn_direct_split_is_not_offered_as_downloaded(in_tmp_cwd):
    # llama.cpp resolves a split's siblings from the main shard's directory, so
    # a lone shard is a load that fails after the teardown. The directory scan
    # marks a torn quant partial; the direct-file fallback must not call it ready.
    shard = in_tmp_cwd / "model-Q4_K_M-00001-of-00002.gguf"
    shard.write_bytes(b"GGUF")

    row = _variants(os.fspath(shard)).variants[0]
    assert row.quant == "Q4_K_M"
    assert row.downloaded is False and row.partial is True

    # The whole set beside it is ready, and an unsplit file is untouched.
    (in_tmp_cwd / "model-Q4_K_M-00002-of-00002.gguf").write_bytes(b"GGUF")
    whole = _variants(os.fspath(shard)).variants[0]
    assert whole.downloaded is True and whole.partial is False


def test_stray_over_indexed_shard_does_not_complete_a_split(in_tmp_cwd):
    # Completeness is the declared index set, not a file count: a stray
    # 00003-of-00002 must not stand in for the missing shard 2.
    shard = in_tmp_cwd / "model-Q4_K_M-00001-of-00002.gguf"
    shard.write_bytes(b"GGUF")
    (in_tmp_cwd / "model-Q4_K_M-00003-of-00002.gguf").write_bytes(b"GGUF")

    row = _variants(os.fspath(shard)).variants[0]
    assert row.downloaded is False and row.partial is True


def test_zero_byte_split_sibling_does_not_complete_a_split(in_tmp_cwd):
    # The directory scan marks a torn split partial when a sibling is an empty
    # interrupted copy; the name alone must not count as the shard.
    shard = in_tmp_cwd / "model-Q4_K_M-00001-of-00002.gguf"
    shard.write_bytes(b"GGUF")
    (in_tmp_cwd / "model-Q4_K_M-00002-of-00002.gguf").write_bytes(b"")

    row = _variants(os.fspath(shard)).variants[0]
    assert row.downloaded is False and row.partial is True


def test_zero_byte_direct_gguf_is_partial(in_tmp_cwd):
    # The directory scan treats an empty gguf as incomplete (an interrupted
    # copy), so the direct-file fallback must not call the same bytes ready.
    empty = in_tmp_cwd / "foo-Q4_K_M.gguf"
    empty.write_bytes(b"")

    row = _variants(os.fspath(empty)).variants[0]
    assert row.downloaded is False and row.partial is True


def test_local_dir_answer_ignores_the_hub_cache_of_the_same_name(in_tmp_cwd, monkeypatch):
    # A repo-shaped id that exists as a directory is resolved existence-first by
    # the load, so this answer describes that directory. An empty leftover
    # <quant>/ folder in the HF cache of the identically named repo must not add
    # a row to it: the CLI's attach gate reads any row as "this is a GGUF model"
    # and the load then evicts the resident model for a directory with none.
    from types import SimpleNamespace

    hub_cache = in_tmp_cwd / "hub"
    (hub_cache / "models--unsloth--foo" / "snapshots" / "rev" / "Q4_K_M").mkdir(parents = True)
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(
            hub_cache = hub_cache,
            hf_home = in_tmp_cwd,
            source = "studio",
            cache_home = in_tmp_cwd,
        ),
    )
    (in_tmp_cwd / "unsloth" / "foo").mkdir(parents = True)
    (in_tmp_cwd / "unsloth" / "foo" / "config.json").write_text("{}")

    from utils.models.model_config import detect_gguf_model

    assert detect_gguf_model("unsloth/foo") is None
    assert _variants("unsloth/foo").variants == []
