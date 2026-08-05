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
