# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import ast
import os
from pathlib import Path

import pytest

_backend_root = Path(__file__).resolve().parent.parent
_models_src = _backend_root / "routes" / "models.py"


def _load_coerce_scan_folder_path():
    """Load the real helper without importing the full FastAPI route module."""
    tree = ast.parse(_models_src.read_text())
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_coerce_scan_folder_path"
    )
    module = ast.Module(body = [fn], type_ignores = [])
    ast.fix_missing_locations(module)
    ns: dict = {"Path": Path, "os": os}
    exec(compile(module, f"<extracted {_models_src}>", "exec"), ns)
    return ns["_coerce_scan_folder_path"]


coerce_scan_folder_path = _load_coerce_scan_folder_path()


def test_existing_directory_is_returned_as_normalized_directory(tmp_path):
    assert Path(coerce_scan_folder_path(str(tmp_path))) == tmp_path.resolve()


@pytest.mark.parametrize("filename", ["model.gguf", "model.safetensors", "model.bin"])
def test_model_weight_file_paths_register_their_parent_directory(tmp_path, filename):
    model_file = tmp_path / filename
    model_file.write_bytes(b"x")

    assert Path(coerce_scan_folder_path(str(model_file))) == tmp_path.resolve()


def test_non_weight_file_path_is_rejected(tmp_path):
    text_file = tmp_path / "notes.txt"
    text_file.write_text("not a model", encoding = "utf-8")

    with pytest.raises(ValueError, match = "Path must be a folder or model weight file"):
        coerce_scan_folder_path(str(text_file))


def test_missing_path_is_normalized_for_storage_validation(tmp_path):
    missing = tmp_path / "missing-models"

    assert coerce_scan_folder_path(str(missing)) == os.path.realpath(str(missing))


def test_backslash_wsl_style_directory_path_falls_back_to_slashes(tmp_path):
    model_dir = tmp_path / "train" / "llama3_8B"
    model_dir.mkdir(parents = True)
    pasted_path = str(model_dir).replace("/", "\\")

    assert Path(coerce_scan_folder_path(pasted_path)) == model_dir.resolve()


def test_backslash_wsl_style_model_file_path_falls_back_to_slashes(tmp_path):
    model_dir = tmp_path / "train" / "llama3_8B"
    model_dir.mkdir(parents = True)
    model_file = model_dir / "model.Q4_K_M.gguf"
    model_file.write_bytes(b"x")
    pasted_path = str(model_file).replace("/", "\\")

    assert Path(coerce_scan_folder_path(pasted_path)) == model_dir.resolve()


@pytest.mark.skipif(os.sep != "/", reason = "POSIX-only path spelling")
def test_existing_posix_path_with_backslash_is_not_rewritten(tmp_path):
    model_dir = tmp_path / "model\\name"
    model_dir.mkdir()

    assert Path(coerce_scan_folder_path(str(model_dir))) == model_dir.resolve()


def test_empty_path_is_rejected():
    with pytest.raises(ValueError, match = "Path cannot be empty"):
        coerce_scan_folder_path("   ")


def test_null_byte_path_is_rejected():
    with pytest.raises(ValueError, match = "Path cannot contain null bytes"):
        coerce_scan_folder_path("model\x00.gguf")
