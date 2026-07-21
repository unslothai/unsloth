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
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return ast.get_source_segment(src, node)


# -- schema -------------------------------------------------------------------------------------


def test_gguf_request_imatrix_defaults_and_set():
    assert ExportGGUFRequest(save_directory = "/tmp/x").imatrix is False
    assert ExportGGUFRequest(save_directory = "/tmp/x").imatrix_path is None
    r = ExportGGUFRequest(save_directory = "/tmp/x", imatrix = True, imatrix_path = "/i.dat")
    assert r.imatrix is True and r.imatrix_path == "/i.dat"


def test_merged_request_accepts_compressed_formats():
    for fmt in (
        "16-bit (FP16)",
        "FP8 (compressed-tensors)",
        "NVFP4 (compressed-tensors)",
    ):
        assert (
            ExportMergedModelRequest(
                save_directory = "/tmp/x", format_type = fmt
            ).format_type
            == fmt
        )


def test_merged_request_rejects_unknown_format():
    with pytest.raises(ValidationError):
        ExportMergedModelRequest(save_directory = "/tmp/x", format_type = "bogus")


# -- threading (ast) ----------------------------------------------------------------------------


def test_export_gguf_threads_imatrix_to_save_and_push():
    # imatrix_file must reach both save paths, but only via the conditional **imatrix_kw.
    g = _func_src("core/export/export.py", "export_gguf")
    assert g.count("**imatrix_kw") >= 2
    assert (
        'imatrix_kw = {"imatrix_file": imatrix_file} if imatrix_file is not None else {}'
        in g
    )
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
    assert "request.imatrix_path or (True if request.imatrix else None)" in _src(
        "routes/export.py"
    )


def test_export_merged_maps_compressed_to_save_method():
    m = _func_src("core/export/export.py", "export_merged_model")
    assert "is_compressed" in m and '"fp8"' in m and '"nvfp4"' in m


def test_compressed_hub_push_uploads_local_dir_without_recompressing():
    # A compressed / torchao Hub push must upload the built output_path, not re-quantize.
    m = _func_src("core/export/export.py", "export_merged_model")
    assert (
        "elif (is_compressed or is_torchao) and output_path and Path(output_path).is_dir():"
        in m
    )
    assert "hf_api.upload_folder(" in m and "folder_path = output_path" in m


# -- torchao portable FP8/INT8 (device-agnostic, no NVIDIA GPU) ---------------------------------


def test_merged_request_accepts_torchao_aliases():
    # Portable torchao aliases pass through compressed_method (validated in the backend registry).
    for alias in ("torchao_fp8", "torchao_int8"):
        r = ExportMergedModelRequest(save_directory = "/tmp/x", compressed_method = alias)
        assert r.compressed_method == alias


def test_export_merged_routes_torchao_and_skips_nvidia_guard():
    m = _func_src("core/export/export.py", "export_merged_model")
    # torchao is classified separately and its suffix comes from the torchao normalizer.
    assert "_normalize_torchao_method(compressed_alias)" in m
    assert "is_torchao = torchao_info is not None" in m
    assert "is_compressed = compressed_alias is not None and not is_torchao" in m
    # The NVIDIA guard applies to compressed-tensors only, not torchao.
    assert "_has_nvidia_gpu()" in m
    # torchao routes through save_method just like compressed.
    assert "elif is_compressed or is_torchao:" in m


def test_export_merged_nvidia_guard_present():
    m = _func_src("core/export/export.py", "export_merged_model")
    assert "requires an NVIDIA GPU" in m


def test_has_nvidia_gpu_helper_reads_hardware_module():
    h = _func_src("core/export/export.py", "_has_nvidia_gpu")
    assert "DeviceType.CUDA" in h and "IS_ROCM" in h


def test_export_merged_relaxes_is_peft_guard():
    # Non-PEFT (Local/HF base) models can now export merged; the old hard block must be gone.
    m = _func_src("core/export/export.py", "export_merged_model")
    assert "Use 'Export Base Model' instead." not in m


def test_unsloth_save_has_torchao_registry_and_path():
    # Read unsloth/save.py as text (not import) so this runs in the CPU suite without unsloth.
    save_py = (_BACKEND.parent.parent / "unsloth" / "save.py").read_text(
        encoding = "utf-8"
    )
    assert "def _normalize_torchao_method" in save_py
    assert "def _unsloth_save_torchao" in save_py
    assert "TORCHAO_EXPORT_SCHEMES = {" in save_py
    # torchao aliases must map to (scheme, suffix) so the backend routes to the torchao path.
    assert '"torchao_fp8": ("fp8", "torchao-fp8")' in save_py
    assert '"torchao_int8": ("int8", "torchao-int8")' in save_py


# -- GGUF multi-quant list ----------------------------------------------------------------------


def test_gguf_request_accepts_list_of_quants():
    r = ExportGGUFRequest(
        save_directory = "/tmp/x", quantization_method = ["Q4_K_M", "Q8_0"]
    )
    assert r.quantization_method == ["Q4_K_M", "Q8_0"]
    r2 = ExportGGUFRequest(save_directory = "/tmp/x", quantization_method = "Q4_K_M")
    assert r2.quantization_method == "Q4_K_M"


def test_export_gguf_normalizes_quant_list():
    g = _func_src("core/export/export.py", "export_gguf")
    assert "isinstance(quantization_method, (list, tuple))" in g
    assert "quant_methods" in g


# -- GGUF LoRA adapter export -------------------------------------------------------------------


def test_lora_request_has_gguf_fields():
    from models.export import ExportLoRAAdapterRequest

    r = ExportLoRAAdapterRequest(save_directory = "/tmp/x")
    assert r.gguf is False and r.gguf_outtype == "q8_0"
    r2 = ExportLoRAAdapterRequest(
        save_directory = "/tmp/x", gguf = True, gguf_outtype = "q8_0"
    )
    assert r2.gguf is True and r2.gguf_outtype == "q8_0"


def test_lora_request_rejects_bad_outtype():
    from models.export import ExportLoRAAdapterRequest
    with pytest.raises(ValidationError):
        ExportLoRAAdapterRequest(save_directory = "/tmp/x", gguf_outtype = "q3_k")


def test_export_lora_wires_gguf_save_method():
    la = _func_src("core/export/export.py", "export_lora_adapter")
    assert 'save_method = "lora"' in la
    assert "quantization_method = outtype" in la


def test_orchestrator_and_worker_pass_lora_gguf():
    o = _func_src("core/export/orchestrator.py", "export_lora_adapter")
    assert '"gguf": gguf' in o and '"gguf_outtype": gguf_outtype' in o
    w = _src("core/export/worker.py")
    assert 'gguf = cmd.get("gguf", False)' in w
    assert 'gguf_outtype = cmd.get("gguf_outtype", "q8_0")' in w


def test_route_passes_lora_gguf():
    r = _src("routes/export.py")
    assert "gguf = request.gguf" in r and "gguf_outtype = request.gguf_outtype" in r


# -- compressed_method ("all formats" dropdown) -------------------------------------------------


def test_merged_request_accepts_compressed_method():
    # Defaults to None; any scheme alias is accepted (validation happens in the backend registry).
    assert ExportMergedModelRequest(save_directory = "/tmp/x").compressed_method is None
    for alias in (
        "fp8",
        "fp8_static",
        "w8a8",
        "w8a16",
        "w4a16",
        "mxfp4",
        "mxfp8",
        "nvfp4",
    ):
        r = ExportMergedModelRequest(save_directory = "/tmp/x", compressed_method = alias)
        assert r.compressed_method == alias


def test_export_merged_resolves_alias_via_registry():
    # The scheme + suffix must come from unsloth.save's registry normalizer, not a hardcoded dict.
    m = _func_src("core/export/export.py", "export_merged_model")
    assert "compressed_method" in m
    assert "_normalize_compressed_method(compressed_alias)" in m
    assert (
        "compressed_alias = compressed_method or _LABEL_TO_ALIAS.get(format_type)" in m
    )
    assert "compressed_suffix" in m and 'f"{save_directory}-{compressed_suffix}"' in m


def test_orchestrator_and_worker_pass_compressed_method():
    o = _func_src("core/export/orchestrator.py", "export_merged_model")
    assert "compressed_method" in o and '"compressed_method": compressed_method' in o
    assert 'compressed_method = cmd.get("compressed_method")' in _src(
        "core/export/worker.py"
    )


def test_route_passes_compressed_method():
    assert "compressed_method = request.compressed_method" in _src("routes/export.py")
