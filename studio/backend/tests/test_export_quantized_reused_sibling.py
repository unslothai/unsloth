# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""A quantized (torchao / compressed-tensors) export writes the "<dir>-<suffix>" sibling and pushes it; when that
sibling already holds an earlier export, only the files this save wrote may go to the Hub."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "export_hub_push_cases", Path(__file__).with_name("test_export_hub_push.py")
)
_CASES = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CASES)


def _push(
    tmp_path,
    monkeypatch,
    before = None,
):
    calls: list[str] = []
    seen: dict = {}
    backend = _CASES._non_mlx_backend(
        monkeypatch, "test_export_quantized_reused_backend", calls, seen
    )
    if before is not None:
        before(backend)
    success, message, output_path = backend.export_merged_model(
        str(tmp_path / "export"),
        push_to_hub = True,
        repo_id = "model",
        hf_token = "hf_fake",
        compressed_method = "torchao_fp8",
    )
    assert success is True, message
    return backend, calls, seen, output_path


def test_reused_quantized_sibling_uploads_only_this_saves_files(tmp_path, monkeypatch):
    sibling = tmp_path / "export-torchao-fp8"
    sibling.mkdir()
    (sibling / "model-00002-of-00002.safetensors").write_bytes(b"an earlier private export")
    (sibling / "notes.txt").write_text("private")
    _CASES._age(sibling)

    backend, calls, seen, output_path = _push(tmp_path, monkeypatch)

    assert seen["folder"] == output_path
    assert seen["uploaded"] == ["model.safetensors"]
    assert (sibling / "notes.txt").read_text() == "private"


def test_untracked_reused_quantized_sibling_is_rebuilt_in_a_fresh_folder(tmp_path, monkeypatch):
    sibling = tmp_path / "export-torchao-fp8"
    sibling.mkdir()
    (sibling / "notes.txt").write_text("private")
    # Written moments ago: a rewrite cannot be told apart from it, so the sibling is not uploaded.

    sidecars: list[tuple] = []

    def spy(backend):
        backend._package_drafter_merged = lambda folder, repo_id, **kw: sidecars.append(
            ("drafter", folder)
        )
        backend._sync_reap_report = lambda folder: sidecars.append(("reap", folder))

    backend, calls, seen, output_path = _push(tmp_path, monkeypatch, spy)

    # Rebuilt in a fresh staging folder and uploaded from there, so the Studio sidecars
    # (drafter, REAP report) go up with it; the reused sibling's own files never do.
    assert backend.current_model.merges == ["torchao_fp8", "torchao_fp8"]
    assert "upload_folder" in calls
    assert seen["folder"] != output_path
    assert seen["uploaded"] == ["model.safetensors"]
    assert (sibling / "notes.txt").read_text() == "private"
    assert ("drafter", seen["folder"]) in sidecars and ("reap", seen["folder"]) in sidecars


def test_fresh_quantized_sibling_is_uploaded_as_before(tmp_path, monkeypatch):
    backend, calls, seen, output_path = _push(tmp_path, monkeypatch)

    assert backend.current_model.merges == ["torchao_fp8"]
    assert "model.safetensors" in seen["uploaded"]
    assert not os.path.exists(tmp_path / "export-torchao-fp8" / "notes.txt")
