# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Tests for GGUF routing in detect_gguf_model.

Regression test: on Windows a .gguf file can briefly appear inaccessible
during llama-server teardown, making is_file() return False and routing
the model to the transformers backend instead of llama-server.
"""

import sys
import os
import types
from pathlib import Path
from unittest.mock import patch

# Stub structlog before importing backend modules (as in other suite tests)
if "structlog" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _):
            return lambda *a, **k: None

    sys.modules["structlog"] = types.SimpleNamespace(
        get_logger = lambda *a, **k: _DummyLogger(),
        BoundLogger = _DummyLogger,
    )

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.models.model_config import detect_gguf_model


def test_detects_gguf_file_normally(tmp_path):
    """Normal case: .gguf file exists and is accessible."""
    gguf = tmp_path / "gpt-oss-20b-MXFP4.gguf"
    gguf.write_bytes(b"")
    result = detect_gguf_model(str(gguf))
    assert result is not None
    assert result.endswith("gpt-oss-20b-MXFP4.gguf")


def test_detects_gguf_when_stat_raises_oserror(tmp_path):
    """
    Regression: on Windows is_file()/exists() call stat(), which raises OSError
    in the brief lock window after llama-server is killed. detect_gguf_model must
    still route to llama-server by file extension alone.
    """
    gguf = tmp_path / "gpt-oss-20b-MXFP4.gguf"
    gguf.write_bytes(b"")

    original_stat = Path.stat

    def flaky_stat(self, **kwargs):
        if self.suffix.lower() == ".gguf":
            raise OSError("file temporarily inaccessible (Windows lock window)")
        return original_stat(self, **kwargs)

    with patch.object(Path, "stat", flaky_stat):
        result = detect_gguf_model(str(gguf))

    assert result is not None, (
        "detect_gguf_model returned None when stat() raised OSError. "
        "This causes the model to fall through to the transformers backend."
    )


def test_does_not_detect_mmproj_as_main_model(tmp_path):
    """mmproj files must never be returned as the primary model."""
    mmproj = tmp_path / "mmproj-model-f16.gguf"
    mmproj.write_bytes(b"")
    result = detect_gguf_model(str(mmproj))
    assert result is None


def test_detects_gguf_in_directory(tmp_path):
    """Directory containing a .gguf file is resolved to that file."""
    gguf = tmp_path / "model-Q4_K_M.gguf"
    gguf.write_bytes(b"")
    result = detect_gguf_model(str(tmp_path))
    assert result is not None
    assert result.endswith("model-Q4_K_M.gguf")


def test_directory_auto_detect_ignores_big_endian_sibling(tmp_path):
    be = tmp_path / "model-Q4_K_M-be.gguf"
    be.write_bytes(b"x" * 100)
    target = tmp_path / "model-Q4_K_M.gguf"
    target.write_bytes(b"y" * 10)

    result = detect_gguf_model(str(tmp_path))
    assert result == str(target.resolve())


def test_direct_big_endian_file_is_not_detected(tmp_path):
    gguf = tmp_path / "model-Q4_K_M-be.gguf"
    gguf.write_bytes(b"")

    assert detect_gguf_model(str(gguf)) is None


def test_directory_named_like_gguf_scans_inside(tmp_path):
    """A directory named *.gguf resolves the real .gguf inside, not itself."""
    gguf_dir = tmp_path / "mymodel.gguf"
    gguf_dir.mkdir()
    inner = gguf_dir / "model-Q4_K_M.gguf"
    inner.write_bytes(b"")
    result = detect_gguf_model(str(gguf_dir))
    assert result is not None
    assert result.endswith("model-Q4_K_M.gguf")


def test_returns_none_for_non_gguf_path(tmp_path):
    """Non-.gguf paths with no .gguf files inside return None."""
    result = detect_gguf_model(str(tmp_path))
    assert result is None
