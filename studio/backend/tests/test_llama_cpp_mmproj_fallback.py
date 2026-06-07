# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the llama-server mmproj text-only fallback.

A GGUF vision model is launched with ``--mmproj <projector>``. When the
installed llama.cpp prebuilt is older than the model's projector format,
llama-server aborts at startup with ``clip.cpp:NNNN: Unknown projector
type`` (exit -6). load_model now retries once WITHOUT ``--mmproj`` so the
base model still loads text-only, warns the user to update llama.cpp, and
marks the session non-vision. These tests pin the two decision helpers:
``_is_projector_incompatibility`` (when to retry) and ``_strip_mmproj_args``
(how the retry argv is built). Unrelated failures must NOT trigger a retry.
"""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Match the stubbing pattern in sibling tests so the module imports in a
# lightweight env without fastapi.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger(
    "structlog"
)
sys.modules.setdefault("structlog", _structlog_stub)
if not hasattr(sys.modules["structlog"], "get_logger"):
    sys.modules["structlog"].get_logger = _structlog_stub.get_logger

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

_detect = LlamaCppBackend._is_projector_incompatibility
_strip = LlamaCppBackend._strip_mmproj_args

# Real abort captured loading gemma-4 on a 3-day-old prebuilt (build b9496).
_GEMMA4_OLD_LLAMACPP_OUT = (
    "srv    load_model: loading model 'gemma-4-E2B-it-UD-Q4_K_XL.gguf'\n"
    "/build_work/src/llama.cpp-b9496/tools/mtmd/clip.cpp:4391: "
    "Unknown projector type\n"
    "libggml-base.so.0(ggml_abort+0x152)\n"
    "libmtmd.so.0(clip_n_mmproj_embd)\n"
)
# Unrelated failures that must keep their own handling (no projector retry).
_OOM_OUT = (
    "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 12000.00 MiB on "
    "device 0: cudaMalloc failed: out of memory"
)
_BAD_ARCH_OUT = (
    "llama_model_load: error loading model: unknown model architecture: "
    "'qwen_image'"
)
_PORT_OUT = "srv start: failed to bind: address already in use"
_MISSING_OUT = "error: failed to open GGUF file: no such file or directory"
# A healthy startup log that merely mentions the projector must not match.
_HEALTHY_VISION_OUT = (
    "Using mmproj for vision: /cache/mmproj-F16.gguf\n"
    "clip_model_loader: loaded meta data with 20 key-value pairs\n"
    "srv  update_slots: all slots are idle"
)


class TestProjectorIncompatibilityDetector:
    def test_gemma4_on_old_llamacpp_triggers_retry(self):
        # Headline case: a 3-day-old llama.cpp aborts on Gemma-4's projector.
        assert _detect(_GEMMA4_OLD_LLAMACPP_OUT) is True

    @pytest.mark.parametrize(
        "out",
        [
            "clip.cpp:4391: Unknown projector type",
            "error: unsupported projector type for this model",
            "llama_mmproj: unsupported mmproj file version",
            "clip.cpp: projector type 'gemma4' is not supported",
        ],
    )
    def test_projector_format_errors_match(self, out):
        assert _detect(out) is True

    def test_case_insensitive(self):
        assert _detect("UNKNOWN PROJECTOR TYPE") is True

    @pytest.mark.parametrize(
        "out",
        [
            _OOM_OUT,
            _BAD_ARCH_OUT,
            _PORT_OUT,
            _MISSING_OUT,
            _HEALTHY_VISION_OUT,
            "",
            # bare multimodal words without a failure term must not match
            "loading clip model",
            "mmproj file resolved from cache",
        ],
    )
    def test_unrelated_failures_do_not_retry(self, out):
        assert _detect(out) is False


# A realistic vision launch argv (mirrors the live "Starting llama-server"
# command), projector pair at the end.
_VISION_CMD = [
    "/home/u/.unsloth/llama.cpp/build/bin/llama-server",
    "-m", "/cache/gemma-4-E2B-it-UD-Q4_K_XL.gguf",
    "--port", "55473",
    "-c", "131072",
    "--parallel", "1",
    "--flash-attn", "on",
    "--no-context-shift",
    "-ngl", "-1",
    "--threads", "-1",
    "--jinja",
    "--spec-default",
    "--mmproj", "/cache/mmproj-F16.gguf",
]


class TestStripMmprojArgs:
    def test_removes_mmproj_pair(self):
        stripped = _strip(_VISION_CMD)
        assert "--mmproj" not in stripped
        assert "/cache/mmproj-F16.gguf" not in stripped

    def test_preserves_every_text_flag(self):
        stripped = _strip(_VISION_CMD)
        for flag in (
            "-m", "/cache/gemma-4-E2B-it-UD-Q4_K_XL.gguf",
            "--port", "55473", "-c", "131072", "-ngl", "-1",
            "--jinja", "--spec-default", "--flash-attn", "on",
        ):
            assert flag in stripped
        # Exactly the two projector tokens are dropped.
        assert len(stripped) == len(_VISION_CMD) - 2

    def test_strips_mmproj_in_the_middle(self):
        cmd = ["llama-server", "--mmproj", "/p/mm.gguf", "-c", "4096", "--jinja"]
        assert _strip(cmd) == ["llama-server", "-c", "4096", "--jinja"]

    def test_noop_when_no_mmproj(self):
        cmd = ["llama-server", "-m", "/p/model.gguf", "-c", "4096", "--jinja"]
        assert _strip(cmd) == cmd

    def test_returns_new_list(self):
        cmd = ["llama-server", "--mmproj", "/p/mm.gguf"]
        out = _strip(cmd)
        assert out is not cmd
        assert cmd[-1] == "/p/mm.gguf"  # input untouched


class TestRetryContract:
    """The two helpers compose into the load_model retry decision."""

    def test_gemma4_failure_yields_valid_text_only_command(self):
        # Old-llama.cpp projector abort -> retry, and the retry argv is a
        # valid text-only launch (model + context kept, projector gone).
        assert _detect(_GEMMA4_OLD_LLAMACPP_OUT) is True
        retry_cmd = _strip(_VISION_CMD)
        assert "--mmproj" not in retry_cmd
        assert "-m" in retry_cmd and "--jinja" in retry_cmd

    def test_oom_does_not_retry_text_only(self):
        # An OOM with --mmproj present must NOT be treated as a projector
        # problem: load_model errors out instead of dropping vision.
        assert _detect(_OOM_OUT) is False
