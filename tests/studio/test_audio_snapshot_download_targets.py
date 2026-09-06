# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The audio paths must fetch Spark-TTS into the shared HF cache, never a relative dir.

``snapshot_download(repo, local_dir = "Spark-TTS-0.5B")`` resolves against the process
CWD. Under the desktop shell that is ``studio/src-tauri``, so ~1.5 GB of model landed
inside the Tauri crate, the dev watcher rebuilt on every downloaded file and killed the
backend mid-load. It also bypasses hf_cache_settings, so the copy is invisible to the
inventory and re-downloads once per CWD -- while the cached-start path (``local_files_only``)
was already reading the cache.

Parsed rather than string-matched: reformatting this call should not fail the suite,
only reintroducing a download target outside the cache should.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest


def _find_repo_root() -> Path | None:
    env = os.environ.get("UNSLOTH_REPO_ROOT")
    if env:
        p = Path(env).resolve()
        if (p / "studio" / "backend").is_dir():
            return p
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "studio" / "backend").is_dir():
            return parent
    return None


_REPO_ROOT = _find_repo_root()
if _REPO_ROOT is None:
    pytest.skip(
        "Could not locate studio/backend. Set UNSLOTH_REPO_ROOT or run from "
        "the repository checkout.",
        allow_module_level = True,
    )

# Every module that fetches a Spark-TTS repo: TTS inference, the GGUF BiCodec decoder and the trainer.
_AUDIO_SOURCES = (
    "studio/backend/core/inference/inference.py",
    "studio/backend/core/inference/llama_cpp.py",
    "studio/backend/core/training/trainer.py",
)


def _snapshot_download_calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "snapshot_download")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "snapshot_download")
        )
    ]


def _local_dir_arg(call: ast.Call) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == "local_dir":
            return kw.value
    return None


def _assignments(tree: ast.Module) -> dict[str, ast.expr]:
    out: dict[str, ast.expr] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node.value
    return out


def _is_anchored(
    expr: ast.expr,
    assigned: dict[str, ast.expr],
    depth: int = 0,
) -> bool:
    """True when the target is built from an explicit base rather than the CWD."""
    if depth > 3:
        return False
    if isinstance(expr, ast.Name):
        source = assigned.get(expr.id)
        return source is not None and _is_anchored(source, assigned, depth + 1)
    if isinstance(expr, ast.Call):
        func = expr.func
        # os.path.join(base, ...) / Path(base) / str(base) anchor to something named.
        if isinstance(func, ast.Attribute):
            # ...but a repo id sliced into a bare name does not.
            return func.attr not in ("split", "rsplit")
        return True
    return False


@pytest.mark.parametrize("rel", _AUDIO_SOURCES)
def test_audio_snapshot_downloads_target_the_hf_cache(rel: str) -> None:
    path = _REPO_ROOT / rel
    assert path.is_file(), f"missing {rel}"

    tree = ast.parse(path.read_text(encoding = "utf-8"))
    assigned = _assignments(tree)

    offenders = []
    for call in _snapshot_download_calls(path):
        local_dir = _local_dir_arg(call)
        if local_dir is None:
            continue
        if not _is_anchored(local_dir, assigned):
            offenders.append(local_dir.lineno)

    assert not offenders, (
        f"{rel} passes a CWD-relative local_dir to snapshot_download at line(s) "
        f"{offenders}. Drop local_dir so the repo lands in the shared HF cache."
    )
