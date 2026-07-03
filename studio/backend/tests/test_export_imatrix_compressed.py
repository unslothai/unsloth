# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the GGUF imatrix option and compressed-tensors merged export wiring.

Schema checks use the real Pydantic models; the cross-layer threading is verified with ast so it
runs on CPU with no GPU, no model, and no llama.cpp.
"""

import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

from models.export import ExportGGUFRequest, ExportMergedModelRequest

_BACKEND = Path(__file__).resolve().parent.parent


def _src(rel):
    return (_BACKEND / rel).read_text(encoding = "utf-8")


def _func_src(rel, name):
    src = _src(rel)
    node = next(
        n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return ast.get_source_segment(src, node)


# -- schema -------------------------------------------------------------------------------------


def test_gguf_request_imatrix_defaults_and_set():
    assert ExportGGUFRequest(save_directory = "/tmp/x").imatrix is False
    assert ExportGGUFRequest(save_directory = "/tmp/x").imatrix_path is None
    r = ExportGGUFRequest(save_directory = "/tmp/x", imatrix = True, imatrix_path = "/i.dat")
    assert r.imatrix is True and r.imatrix_path == "/i.dat"


def test_merged_request_accepts_compressed_formats():
    for fmt in ("16-bit (FP16)", "FP8 (compressed-tensors)", "NVFP4 (compressed-tensors)"):
        assert ExportMergedModelRequest(save_directory = "/tmp/x", format_type = fmt).format_type == fmt


def test_merged_request_rejects_unknown_format():
    with pytest.raises(ValidationError):
        ExportMergedModelRequest(save_directory = "/tmp/x", format_type = "bogus")


# -- threading (ast) ----------------------------------------------------------------------------


def test_export_gguf_threads_imatrix_to_save_and_push():
    # imatrix_file must reach both save_pretrained_gguf and push_to_hub_gguf, but only via the
    # conditional **imatrix_kw so a no-imatrix export never sends an unsupported keyword.
    g = _func_src("core/export/export.py", "export_gguf")
    assert g.count("**imatrix_kw") >= 2
    assert 'imatrix_kw = {"imatrix_file": imatrix_file} if imatrix_file is not None else {}' in g
    # Unconditional pass-through (the old wiring) must be gone.
    assert "imatrix_file = imatrix_file" not in g


def test_export_gguf_guards_unsupported_imatrix_build():
    # An older unsloth without imatrix_file support gets a clean error, not a TypeError.
    g = _func_src("core/export/export.py", "export_gguf")
    assert "_supports_kwarg(" in g and '"imatrix_file"' in g


def test_export_merged_guards_unsupported_compressed_build():
    m = _func_src("core/export/export.py", "export_merged_model")
    assert "_compressed_export_supported()" in m


def test_supports_kwarg_helper():
    # exec just the helper source so the test stays free of export.py's heavy import chain.
    ns = {}
    exec(_func_src("core/export/export.py", "_supports_kwarg"), ns)
    supports = ns["_supports_kwarg"]

    def has_it(a, imatrix_file = None):
        pass

    def lacks_it(a):
        pass

    def via_kwargs(a, **kw):
        pass

    assert supports(has_it, "imatrix_file") is True
    assert supports(lacks_it, "imatrix_file") is False
    assert supports(via_kwargs, "imatrix_file") is True


def test_orchestrator_and_worker_pass_imatrix():
    assert "imatrix_file" in _func_src("core/export/orchestrator.py", "export_gguf")
    assert 'imatrix_file = cmd.get("imatrix_file")' in _src("core/export/worker.py")


def test_route_resolves_imatrix_file():
    assert "request.imatrix_path or (True if request.imatrix else None)" in _src("routes/export.py")


def test_export_merged_maps_compressed_to_save_method():
    m = _func_src("core/export/export.py", "export_merged_model")
    assert "is_compressed" in m and '"fp8"' in m and '"nvfp4"' in m


def test_compressed_hub_push_uploads_local_dir_without_recompressing():
    # A compressed Hub push must upload the already-built output_path, not re-run compression
    # via push_to_hub_merged (which would compress a second time).
    m = _func_src("core/export/export.py", "export_merged_model")
    assert "elif is_compressed and output_path and Path(output_path).is_dir():" in m
    assert "hf_api.upload_folder(" in m and "folder_path = output_path" in m
