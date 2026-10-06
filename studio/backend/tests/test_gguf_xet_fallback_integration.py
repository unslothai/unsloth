# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Integration: GGUF Chat-Mode downloads route through the Xet->HTTP helper,
preserving cancellation and the best-effort companion contract. No GPU, no
network, no real subprocess (the helper is patched).
"""

from __future__ import annotations

import sys
import threading
import types as _types
from pathlib import Path
from unittest.mock import patch

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Heavy-dep stubbing; prefer the real structlog so a bare stub never leaks to
# later modules that log at import time.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
try:
    import structlog  # noqa: F401
except ImportError:
    sys.modules["structlog"] = _types.ModuleType("structlog")
try:
    import httpx  # noqa: F401
except ImportError:
    _httpx_stub = _types.ModuleType("httpx")
    for _exc in (
        "ConnectError",
        "TimeoutException",
        "ReadTimeout",
        "ReadError",
        "RemoteProtocolError",
        "CloseError",
        "HTTPError",
        "RequestError",
        "HTTPStatusError",
    ):
        setattr(_httpx_stub, _exc, type(_exc, (Exception,), {}))
    _httpx_stub.Response = type("Response", (), {})
    _httpx_stub.Request = type("Request", (), {})
    _httpx_stub.Timeout = type("Timeout", (), {"__init__": lambda self, *a, **k: None})
    _httpx_stub.Client = type(
        "Client",
        (),
        {
            "__init__": lambda self, **k: None,
            "__enter__": lambda self: self,
            "__exit__": lambda self, *a: None,
        },
    )
    sys.modules.setdefault("httpx", _httpx_stub)

from huggingface_hub import constants as hf_constants

from core.inference.llama_cpp import LlamaCppBackend
from utils.hf_xet_fallback import DownloadStallError

REPO = "unsloth/vision-GGUF"


@pytest.fixture
def hf_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    return tmp_path


def _build_cache(
    root: Path,
    repo_id: str,
    files: dict[str, int],
    sha: str = "a" * 40,
) -> Path:
    repo_dir = root / f"models--{repo_id.replace('/', '--')}"
    (repo_dir / "blobs").mkdir(parents = True, exist_ok = True)
    snap = repo_dir / "snapshots" / sha
    snap.mkdir(parents = True, exist_ok = True)
    for rel, size in files.items():
        (snap / rel).write_bytes(b"\0" * size)
    return snap


def test_companion_routes_through_helper(hf_cache):
    _build_cache(hf_cache, REPO, {"mmproj-vision-F16.gguf": 1})
    backend = LlamaCppBackend()
    captured = {}

    def fake_helper(
        repo_id,
        filename,
        token = None,
        **kwargs,
    ):
        captured["filename"] = filename
        captured["cancel_event"] = kwargs.get("cancel_event")
        return f"/fake/{filename}"

    with (
        patch("huggingface_hub.list_repo_files", lambda *a, **k: ["mmproj-vision-F16.gguf"]),
        patch("core.inference.llama_cpp.hf_hub_download_with_xet_fallback", fake_helper),
    ):
        out = backend._download_mmproj(hf_repo = REPO, hf_token = None)

    assert out == "/fake/mmproj-vision-F16.gguf"
    # _cancel_event must be threaded through so /unload can abort the download.
    assert captured["cancel_event"] is backend._cancel_event


def test_companion_swallows_terminal_stall_to_none(hf_cache):
    _build_cache(hf_cache, REPO, {"mmproj-vision-F16.gguf": 1})
    backend = LlamaCppBackend()

    def stalling_helper(
        repo_id,
        filename,
        token = None,
        **kwargs,
    ):
        raise DownloadStallError("both transports stalled")

    with (
        patch("huggingface_hub.list_repo_files", lambda *a, **k: ["mmproj-vision-F16.gguf"]),
        patch("core.inference.llama_cpp.hf_hub_download_with_xet_fallback", stalling_helper),
    ):
        out = backend._download_mmproj(hf_repo = REPO, hf_token = None)

    assert out is None, "a companion download is best-effort; a terminal stall must not raise"


def test_companion_cancelled_skips_download(hf_cache):
    _build_cache(hf_cache, REPO, {"mmproj-vision-F16.gguf": 1})
    backend = LlamaCppBackend()
    backend._cancel_event.set()
    called = {"n": 0}

    def helper(
        repo_id,
        filename,
        token = None,
        **kwargs,
    ):
        called["n"] += 1
        return "/should-not-happen"

    with (
        patch("huggingface_hub.list_repo_files", lambda *a, **k: ["mmproj-vision-F16.gguf"]),
        patch("core.inference.llama_cpp.hf_hub_download_with_xet_fallback", helper),
    ):
        out = backend._download_mmproj(hf_repo = REPO, hf_token = None)

    assert out is None
    assert called["n"] == 0, "a cancelled load must not start a companion download"
