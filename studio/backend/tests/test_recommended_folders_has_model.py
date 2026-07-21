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
import json
import os
from pathlib import Path

_backend_root = Path(__file__).resolve().parent.parent
_models_src = _backend_root / "routes" / "models.py"


def _load_has_downloaded_model():
    """Return the real ``_dir_has_downloaded_model`` (plus its ``_safe_is_dir``
    and ``_is_weight_bin`` deps, and the ``_WEIGHT_BIN_PREFIXES`` constant the
    latter reads) without importing the heavy module."""
    tree = ast.parse(_models_src.read_text())
    wanted = {"_safe_is_dir", "_dir_has_downloaded_model", "_is_weight_bin"}
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_WEIGHT_BIN_PREFIXES"
            for t in node.targets
        ):
            body.append(node)
    got = {n.name for n in body if isinstance(n, ast.FunctionDef)}
    assert got == wanted, f"helpers missing from source: {wanted - got}"
    module = ast.Module(body = body, type_ignores = [])
    ns: dict = {"Path": Path, "os": os, "json": json}
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
    # A real manifest references its weights via an image.model layer; the
    # referenced blob must exist on disk for the model to be loadable.
    (manifest / "latest").write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "mediaType": "application/vnd.ollama.image.model",
                        "digest": "sha256:abc",
                    }
                ]
            }
        )
    )
    (models / "blobs").mkdir()
    (models / "blobs" / "sha256-abc").write_bytes(b"x")
    assert has_downloaded_model(models) is True


def test_ollama_manifest_without_blob_is_false(tmp_path):
    # A failed/pruned pull leaves the manifest behind but its model blob is
    # gone: the chip must not lead to an empty picker.
    models = tmp_path / "ollama" / "models"
    manifest = models / "manifests" / "registry.ollama.ai" / "library" / "llama3"
    manifest.mkdir(parents = True)
    (manifest / "latest").write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "mediaType": "application/vnd.ollama.image.model",
                        "digest": "sha256:missing",
                    }
                ]
            }
        )
    )
    (models / "blobs").mkdir()  # empty: the referenced blob never landed
    assert has_downloaded_model(models) is False


def test_non_model_files_is_false(tmp_path):
    junk = tmp_path / "junk"
    junk.mkdir()
    (junk / "readme.txt").write_text("hi")
    assert has_downloaded_model(junk) is False


def test_pytorch_bin_weights_are_true(tmp_path):
    # A folder whose only weights are PyTorch .bin checkpoints (which the local
    # scanner accepts) should still earn a Recommended chip.
    repo = tmp_path / "models" / "repo"
    repo.mkdir(parents = True)
    (repo / "config.json").write_text("{}")
    (repo / "pytorch_model.bin").write_bytes(b"x")
    assert has_downloaded_model(tmp_path / "models") is True


def test_non_weight_bin_is_false(tmp_path):
    # A stray .bin that is not a weight file (e.g. tokenizer.bin) must not count.
    repo = tmp_path / "models" / "repo"
    repo.mkdir(parents = True)
    (repo / "tokenizer.bin").write_bytes(b"x")
    assert has_downloaded_model(tmp_path / "models") is False


def test_hidden_subtree_does_not_starve_the_budget(tmp_path):
    # A real model dir that also holds a huge hidden subtree (e.g. a .git or
    # .cache). The hidden entries must not exhaust max_entries before the walk
    # reaches the actual weights, which would falsely report "no model".
    models = tmp_path / "models"
    git = models / ".git" / "objects"
    git.mkdir(parents = True)
    for i in range(50):
        (git / f"obj{i}").write_bytes(b"x")
    repo = models / "repo"
    repo.mkdir()
    (repo / "model.safetensors").write_bytes(b"x")
    assert has_downloaded_model(models, max_entries = 10) is True
