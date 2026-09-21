# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A loaded GGUF reports image input only when its projector has a vision tower.

An mmproj is attached for audio input too (ultravox, Voxtral, Qwen3-ASR), so reporting
``_is_vision`` as image support offers an image button the model cannot honour and sends
the image to llama-server instead of returning the typed 400.
"""

from __future__ import annotations

import inspect
import sys
import types as _types
from pathlib import Path

import pytest


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


def _stub_modules_ctx():
    """Stub only the heavy deps llama_cpp imports that are not already available."""
    from unittest.mock import patch

    _loggers_stub = _types.ModuleType("loggers")
    _loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
    _structlog_stub = _types.ModuleType("structlog")
    _structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
    _httpx_stub = _types.ModuleType("httpx")
    for _exc in ("ConnectError", "TimeoutException", "ReadTimeout", "ReadError"):
        setattr(_httpx_stub, _exc, type(_exc, (Exception,), {}))
    _httpx_stub.Timeout = type("T", (), {"__init__": lambda s, *a, **k: None})
    _httpx_stub.Client = type(
        "C",
        (),
        {
            "__init__": lambda s, **kw: None,
            "__enter__": lambda s: s,
            "__exit__": lambda s, *a: None,
        },
    )
    overrides = {
        name: stub
        for name, stub in (
            ("loggers", _loggers_stub),
            ("structlog", _structlog_stub),
            ("httpx", _httpx_stub),
        )
        if name not in sys.modules
    }
    return patch.dict(sys.modules, overrides)


def _backend():
    with _stub_modules_ctx():
        from core.inference.llama_cpp import LlamaCppBackend
    return LlamaCppBackend()


@pytest.mark.parametrize(
    "accepts_image, expected",
    [(True, True), (False, False)],
)
def test_projector_modality_decides_reported_image_input(accepts_image, expected):
    backend = _backend()
    backend._is_vision = True  # a projector is attached, which is what the launch asks
    backend._mmproj_accepts_image = accepts_image
    assert backend.is_vision is expected


def test_a_model_without_a_projector_takes_no_image():
    backend = _backend()
    backend._is_vision = False
    backend._mmproj_accepts_image = True  # the default for "nothing was read"
    assert backend.is_vision is False


def test_the_load_reads_both_capabilities_from_the_projector_it_attaches():
    """The read cannot be reached without spawning llama-server, so pin it in the source:
    both flags must come from one call on the same probed path, or the pair can describe
    two files."""
    with _stub_modules_ctx():
        from core.inference.llama_cpp import LlamaCppBackend
    src = inspect.getsource(LlamaCppBackend.load_model)
    assert "has_audio, accepts_image = mmproj_capabilities(_mmproj_probe)" in src
    assert "self._mmproj_has_audio = has_audio" in src
    assert "self._mmproj_accepts_image = accepts_image" in src
