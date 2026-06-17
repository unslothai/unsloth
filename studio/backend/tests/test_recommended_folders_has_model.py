# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Regression test for /recommended-folders suggesting empty scaffolds.

The endpoint used to surface any well-known dir that merely existed, so a
freshly installed LM Studio or Ollama (empty ``models`` dir) showed up as a
"Recommended" chip with no models behind it. ``_dir_has_downloaded_model``
now gates each candidate on real weights: a GGUF/safetensors file anywhere in
the tree, or a non-empty Ollama ``manifests/`` beside ``blobs/``.

``routes.models`` pulls the full backend dep tree, so we extract the real
helper (and its ``_safe_is_dir`` dependency) from the source via AST and run
the shipped code in isolation, mirroring
``test_recommended_folders_permission.py``.

Run:
    python -m pytest studio/backend/tests/test_recommended_folders_has_model.py -v
"""

import ast
import os
from pathlib import Path

_backend_root = Path(__file__).resolve().parent.parent
_models_src = _backend_root / "routes" / "models.py"


def _load_has_downloaded_model():
    """Return the real ``_dir_has_downloaded_model`` (plus its
    ``_safe_is_dir`` dependency) without importing the heavy module."""
    tree = ast.parse(_models_src.read_text())
    wanted = {"_safe_is_dir", "_dir_has_downloaded_model"}
    fns = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert {f.name for f in fns} == wanted, "helpers missing from source"
    module = ast.Module(body = fns, type_ignores = [])
    ns: dict = {"Path": Path, "os": os}
    exec(compile(module, f"<extracted {_models_src}>", "exec"), ns)
    return ns["_dir_has_downloaded_model"]


has_downloaded_model = _load_has_downloaded_model()


def test_empty_scaffold_is_false(tmp_path):
    empty = tmp_path / "lmstudio" / "models"
    empty.mkdir(parents = True)
    assert has_downloaded_model(empty) is False


def test_lmstudio_gguf_is_true(tmp_path):
    # models/publisher/repo/file.gguf (LM Studio's nested layout).
    repo = tmp_path / "models" / "bartowski" / "Qwen3-4B-GGUF"
    repo.mkdir(parents = True)
    (repo / "q4.gguf").write_bytes(b"x")
    assert has_downloaded_model(tmp_path / "models") is True


def test_safetensors_is_true(tmp_path):
    repo = tmp_path / "models" / "repo"
    repo.mkdir(parents = True)
    (repo / "model.safetensors").write_bytes(b"x")
    assert has_downloaded_model(tmp_path / "models") is True


def test_ollama_empty_scaffold_is_false(tmp_path):
    models = tmp_path / "ollama" / "models"
    (models / "manifests").mkdir(parents = True)
    (models / "blobs").mkdir()
    assert has_downloaded_model(models) is False


def test_ollama_with_manifest_is_true(tmp_path):
    models = tmp_path / "ollama" / "models"
    manifest = models / "manifests" / "registry.ollama.ai" / "library" / "llama3"
    manifest.mkdir(parents = True)
    (manifest / "latest").write_text("{}")
    (models / "blobs").mkdir()
    (models / "blobs" / "sha256-abc").write_bytes(b"x")
    assert has_downloaded_model(models) is True


def test_non_model_files_is_false(tmp_path):
    junk = tmp_path / "junk"
    junk.mkdir()
    (junk / "readme.txt").write_text("hi")
    assert has_downloaded_model(junk) is False
