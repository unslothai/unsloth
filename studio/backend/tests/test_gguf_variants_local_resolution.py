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
