# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import hashlib
import json
import ast
import os
from pathlib import Path
from urllib.parse import quote
from typing import List, Optional


_backend_root = Path(__file__).resolve().parent.parent
_models_src = _backend_root / "routes" / "models.py"


class _DummyLogger:
    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


class _LocalModelInfo:
    def __init__(
        self,
        *,
        id: str,
        display_name: str,
        path: str,
        source: str,
        model_id: str | None = None,
        updated_at: float | None = None,
        partial: bool = False,
    ):
        self.id = id
        self.display_name = display_name
        self.path = path
        self.source = source
        self.model_id = model_id
        self.updated_at = updated_at
        self.partial = partial


def _load_ollama_helpers():
    """Load the real scanner without importing the full FastAPI route package."""
    tree = ast.parse(_models_src.read_text())
    wanted_functions = {
        "is_ollama_manifest_ref",
        "_ollama_manifest_ref",
        "_safe_is_file",
        "_make_ollama_blob_link",
        "_ollama_model_info_from_manifest",
        "_scan_ollama_dir",
    }
    body = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_OLLAMA_MANIFEST_REF_PREFIX"
                for target in node.targets
            )
        )
        or (isinstance(node, ast.FunctionDef) and node.name in wanted_functions)
    ]
    module = ast.Module(body = body, type_ignores = [])
    ast.fix_missing_locations(module)
    ns: dict = {
        "Path": Path,
        "Optional": Optional,
        "List": List,
        "LocalModelInfo": _LocalModelInfo,
        "hashlib": hashlib,
        "json": json,
        "logger": _DummyLogger(),
        "os": os,
        "quote": quote,
        "uuid": __import__("uuid"),
        "_ollama_links_dir": lambda _dir: None,
    }
    exec(compile(module, f"<extracted {_models_src}>", "exec"), ns)
    return ns


_ollama_helpers = _load_ollama_helpers()
scan_ollama_dir = _ollama_helpers["_scan_ollama_dir"]


def _write_ollama_model(root: Path, rel_manifest: str, digest: str) -> None:
    tag_file = root / "manifests" / rel_manifest
    tag_file.parent.mkdir(parents = True, exist_ok = True)
    tag_file.write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "layers": [
                    {
                        "mediaType": "application/vnd.ollama.image.model",
                        "digest": digest,
                        "size": 4,
                    }
                ],
            }
        ),
        encoding = "utf-8",
    )
    blobs = root / "blobs"
    blobs.mkdir(parents = True, exist_ok = True)
    (blobs / digest.replace(":", "-")).write_bytes(b"GGUF")


def _manifest_hash(rel_manifest: str) -> str:
    return hashlib.sha256(rel_manifest.encode()).hexdigest()[:10]


def test_scan_ollama_dir_discovers_models(monkeypatch, tmp_path):
    ollama_dir = tmp_path / "ollama" / "models"
    _write_ollama_model(
        ollama_dir,
        "registry.ollama.ai/library/qwen3.5/0.8b",
        "sha256:111111",
    )
    _write_ollama_model(
        ollama_dir,
        "hf.co/unsloth/gemma-3-12b-it-GGUF/Q4_K_M",
        "sha256:222222",
    )
    monkeypatch.setitem(
        scan_ollama_dir.__globals__,
        "_ollama_links_dir",
        lambda _dir: (_ for _ in ()).throw(AssertionError("scan created links")),
    )

    models = scan_ollama_dir(ollama_dir)

    assert {m.source for m in models} == {"ollama"}
    assert {m.model_id for m in models} == {
        "ollama/qwen3.5:0.8b",
        "ollama/hf.co/unsloth/gemma-3-12b-it-GGUF:Q4_K_M",
    }
    assert all(m.id.startswith("ollama-manifest:") for m in models)
    assert all(Path(m.path).is_file() for m in models)
    assert not (ollama_dir / ".studio_links").exists()


def test_scan_ollama_dir_materializes_links_only_when_requested(
    monkeypatch,
    tmp_path,
):
    ollama_dir = tmp_path / "ollama" / "models"
    _write_ollama_model(
        ollama_dir,
        "registry.ollama.ai/library/qwen3.5/0.8b",
        "sha256:111111",
    )
    links_root = tmp_path / "links"
    monkeypatch.setitem(
        scan_ollama_dir.__globals__,
        "_ollama_links_dir",
        lambda _dir: links_root,
    )

    models = scan_ollama_dir(ollama_dir, materialize_links = True)

    assert len(models) == 1
    assert models[0].id == models[0].path
    assert Path(models[0].path).suffix == ".gguf"
    assert Path(models[0].path).is_file()


def test_scan_ollama_dir_skips_one_failed_link_without_aborting(
    monkeypatch,
    tmp_path,
):
    ollama_dir = tmp_path / "ollama" / "models"
    blocked_rel = "registry.ollama.ai/library/blocked/latest"
    ok_rel = "registry.ollama.ai/library/ok/latest"
    _write_ollama_model(ollama_dir, blocked_rel, "sha256:111111")
    _write_ollama_model(ollama_dir, ok_rel, "sha256:222222")

    links_root = tmp_path / "links"
    links_root.mkdir()
    (links_root / _manifest_hash(blocked_rel)).write_text(
        "not a directory",
        encoding = "utf-8",
    )
    monkeypatch.setitem(
        scan_ollama_dir.__globals__,
        "_ollama_links_dir",
        lambda _dir: links_root,
    )

    models = scan_ollama_dir(ollama_dir, materialize_links = True)

    assert [m.model_id for m in models] == ["ollama/ok:latest"]
    assert Path(models[0].path).is_file()
