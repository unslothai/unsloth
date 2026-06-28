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
    # imatrix_file must reach both save_pretrained_gguf and push_to_hub_gguf.
    assert (
        _func_src("core/export/export.py", "export_gguf").count("imatrix_file = imatrix_file") >= 2
    )


def test_orchestrator_and_worker_pass_imatrix():
    assert "imatrix_file" in _func_src("core/export/orchestrator.py", "export_gguf")
    assert 'imatrix_file = cmd.get("imatrix_file")' in _src("core/export/worker.py")


def test_route_resolves_imatrix_file():
    assert "request.imatrix_path or (True if request.imatrix else None)" in _src("routes/export.py")


def test_export_merged_maps_compressed_to_save_method():
    m = _func_src("core/export/export.py", "export_merged_model")
    assert "is_compressed" in m and '"fp8"' in m and '"nvfp4"' in m
