# SPDX-License-Identifier: Apache-2.0
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""#9179: slim whisper pairing falls back to an external --with-llama-cpp-dir
runtime when the managed-installer metadata is absent."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_studio = Path(__file__).resolve().parent.parent.parent
if str(_studio) not in sys.path:
    sys.path.insert(0, str(_studio))

iwp = importlib.import_module("install_whisper_prebuilt")


class _Proc:
    def __init__(
        self,
        out: str = "",
        err: str = "",
    ):
        self.stdout = out
        self.stderr = err


@pytest.mark.p1
def test_fallback_parses_llama_server_version(monkeypatch, tmp_path):
    server = tmp_path / "llama-server"
    server.write_text("#!/bin/sh\n", encoding = "utf-8")

    monkeypatch.setenv("LLAMA_SERVER_PATH", str(server))
    monkeypatch.setattr(
        iwp.subprocess,
        "run",
        lambda *a, **k: _Proc(err = "llama-server\nversion: 10360 (87da1a2)\nbuilt with ...\n"),
    )

    runtime = iwp._external_llama_runtime_fallback()
    assert runtime is not None
    bin_dir, tag, profile = runtime
    assert tag == "b10360"
    assert bin_dir == server.resolve().parent
    assert profile == ""


@pytest.mark.p1
def test_fallback_rejects_unknown_source_build_version(monkeypatch, tmp_path):
    server = tmp_path / "llama-server"
    server.write_text("#!/bin/sh\n", encoding = "utf-8")

    monkeypatch.setenv("LLAMA_SERVER_PATH", str(server))
    monkeypatch.setattr(iwp.subprocess, "run", lambda *a, **k: _Proc(err = "version: 1 (no tags)"))

    assert iwp._external_llama_runtime_fallback() is None


@pytest.mark.p1
def test_fallback_none_without_a_llama_server(monkeypatch):
    monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
    monkeypatch.setattr(iwp.shutil, "which", lambda name: None)

    assert iwp._external_llama_runtime_fallback() is None


@pytest.mark.p1
def test_slim_pairing_uses_fallback_when_no_managed_install(monkeypatch, tmp_path):
    artifact = {
        "asset": "whisper-linux-x64-slim.tar.gz",
        "requires_llama_tag": "b10360",
        "requires_ggml_sonames": ["libggml.so", "libggml-base.so"],
    }
    for soname in artifact["requires_ggml_sonames"]:
        (tmp_path / soname).write_bytes(b"")
    (tmp_path / "libggml-cuda.so").write_bytes(b"")

    server = tmp_path / "llama-server"
    server.write_text("#!/bin/sh\n", encoding = "utf-8")

    monkeypatch.setattr(iwp, "installed_llama_runtime", lambda *a, **k: None)
    monkeypatch.setenv("LLAMA_SERVER_PATH", str(server))
    monkeypatch.setattr(
        iwp.subprocess,
        "run",
        lambda *a, **k: _Proc(err = "version: 10360 (87da1a2)"),
    )

    host = MagicMock()
    host.is_macos = False
    host.is_windows = False
    host.whisper_os = "linux"

    monkeypatch.setattr(iwp, "installed_llama_ggml_tree", lambda *a, **k: None)

    pairing = iwp.slim_pairing_for_artifact(artifact, host, "cuda")
    assert pairing is not None
    bin_dir, tag = pairing
    assert tag == "b10360"
    assert bin_dir == server.resolve().parent


@pytest.mark.p1
def test_slim_pairing_still_refuses_without_any_runtime(monkeypatch):
    artifact = {
        "asset": "whisper-linux-x64-slim.tar.gz",
        "requires_llama_tag": "b10360",
        "requires_ggml_sonames": ["libggml.so"],
    }
    monkeypatch.setattr(iwp, "installed_llama_runtime", lambda *a, **k: None)
    monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
    monkeypatch.setattr(iwp.shutil, "which", lambda name: None)

    host = MagicMock()
    host.is_macos = False
    host.is_windows = False
    host.whisper_os = "linux"

    assert iwp.slim_pairing_for_artifact(artifact, host, "cuda") is None
