# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
EXPORT = _BACKEND_DIR / "core" / "export" / "export.py"


def _load_gguf_upload_helpers():
    from typing import List, Optional

    source = EXPORT.read_text()
    namespace: dict = {"List": List, "Optional": Optional, "os": __import__("os")}
    lines = source.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("def _find_export_artifact"))
    end = next(
        i for i, line in enumerate(lines) if line.startswith("def _upload_gguf_directory_to_hub")
    )
    exec("\n".join(lines[start:end]), namespace)
    return namespace


def _load_quant_helpers():
    from typing import List, Optional, Union

    source = EXPORT.read_text()
    namespace: dict = {
        "List": List,
        "Optional": Optional,
        "Union": Union,
        "os": __import__("os"),
    }
    lines = source.splitlines()
    start = next(
        i for i, line in enumerate(lines) if line.startswith("def _normalize_gguf_quant_methods")
    )
    end = next(i for i, line in enumerate(lines) if line.startswith("def _precheck_hub_or_fail"))
    exec("\n".join(lines[start:end]), namespace)
    return namespace


def test_export_gguf_supports_batch_quantization_and_upload_helper():
    source = EXPORT.read_text()
    assert "quantization_methods: Optional[List[str]] = None" in source
    assert "quantization_method = quant_methods" in source
    assert "_upload_gguf_directory_to_hub" in source
    assert "_gguf_path_in_repo" in source
    assert "_upload_model_folder_to_hub" in source


def test_gguf_path_in_repo_normalizes_temp_prefix():
    namespace = _load_gguf_upload_helpers()
    name = namespace["_gguf_path_in_repo"](
        "/exports/unsloth_gguf_abc123.Q4_K_M.gguf",
        model_name = "my-model",
        save_directory = "/exports",
    )
    assert name == "my-model.Q4_K_M.gguf"


def test_gguf_path_in_repo_replaces_save_dir_basename():
    namespace = _load_gguf_upload_helpers()
    name = namespace["_gguf_path_in_repo"](
        "/exports/checkpoint-100/checkpoint-100.Q8_0.gguf",
        model_name = "my-model",
        save_directory = "/exports/checkpoint-100",
    )
    assert name == "my-model.Q8_0.gguf"


def test_build_gguf_hub_readme_includes_vlm_tag():
    namespace = _load_gguf_upload_helpers()
    readme = namespace["_build_gguf_hub_readme"](
        repo_id = "alice/my-vlm",
        upload_names = ["my-vlm.Q4_K_M.gguf", "my-vlm-mmproj.Q4_K_M.gguf"],
        is_vlm = True,
        has_modelfile = True,
    )
    assert "vision-language-model" in readme
    assert "Ollama Note for Vision Models" in readme


def test_normalize_single_gguf_quant_method():
    ns = _load_quant_helpers()
    normalized = ns["_normalize_gguf_quant_methods"](
        quantization_method="Q4_K_M",
        quantization_methods=None,
    )
    assert normalized == ["q4_k_m"]


def test_export_gguf_request_schema_accepts_batch_quants():
    models_path = _BACKEND_DIR / "models" / "export.py"
    source = models_path.read_text()
    assert "quantization_methods: Optional[List[str]]" in source
    assert "class HubPrecheckRequest" in source
