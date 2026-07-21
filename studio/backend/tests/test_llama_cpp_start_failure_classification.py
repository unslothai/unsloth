# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for LlamaCppBackend._classify_llama_start_failure.

When llama-server exits before becoming healthy, load_model turns its
captured stdout/stderr into a user-facing reason. A diffusion/image GGUF
(FLUX, Qwen-Image, ...) is a valid file with plenty of memory, so the
generic "invalid file or out of memory" message is misleading (issue
#5842). These tests pin the classification.
"""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Match sibling tests' stubbing so the module imports in a lightweight
# env without fastapi.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
# Give the structlog stub a real get_logger: a bare ModuleType poisons
# sys.modules for later tests that call structlog.get_logger at import time.
_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger(
    "structlog"
)
sys.modules.setdefault("structlog", _structlog_stub)
if not hasattr(sys.modules["structlog"], "get_logger"):
    sys.modules["structlog"].get_logger = _structlog_stub.get_logger

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

_classify = LlamaCppBackend._classify_llama_start_failure

# Real llama-server failure lines (lower-cased downstream anyway).
_QWEN_IMAGE_OUT = (
    "load_model: loading model 'qwen-image-edit-2511-Q4_K_M.gguf'\n"
    "llama_model_load: error loading model: unknown model architecture: 'qwen_image'\n"
    "llama_model_load_from_file_impl: failed to load model"
)
_OOM_OUT = (
    "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 12000.00 MiB on "
    "device 0: cudaMalloc failed: out of memory"
)


class TestDiffusionArchitectures:
    def test_qwen_image_routes_to_images_page(self):
        msg = _classify(_QWEN_IMAGE_OUT, "/models/qwen-image.gguf", "local/qwen-image")
        assert "diffusion" in msg.lower()
        assert "Images page" in msg
        assert "qwen_image" in msg
        # Must NOT keep blaming memory / file validity.
        assert "out of memory" not in msg.lower()
        assert "enough memory" not in msg.lower()

    # Parametrize over the production set so new arches are auto-covered.
    @pytest.mark.parametrize("arch", sorted(LlamaCppBackend._DIFFUSION_ARCHES))
    def test_every_diffusion_arch_is_recognised(self, arch):
        out = f"error loading model: unknown model architecture: '{arch}'"
        msg = _classify(out, f"/models/{arch}.gguf", f"local/{arch}")
        assert "diffusion" in msg.lower()
        assert "Images page" in msg
        assert arch in msg


class TestUnsupportedNonDiffusionArchitecture:
    def test_unknown_llm_arch_says_unsupported_not_oom(self):
        out = "error loading model: unknown model architecture: 'some_new_llm'"
        msg = _classify(out, "/models/x.gguf", "local/x")
        assert "some_new_llm" in msg
        assert "architecture" in msg.lower()
        # Specific, not the misleading memory message.
        assert "enough memory" not in msg.lower()
        assert "diffusion" not in msg.lower()

    # Exact match: a chat arch merely containing a diffusion token (wan,
    # sd1, flux, ...) must not be routed to the Images page.
    @pytest.mark.parametrize(
        "arch",
        [
            "taiwan",  # contains "wan"
            "swan_llm",  # contains "wan"
            "fluxion",  # contains "flux"
            "sd1234",  # contains "sd1"
            "sd3_chat",  # contains "sd3"
            "aura2_text",  # contains "aura"
            "cosmos_reason",  # contains "cosmos"
            "qwen_image_text",  # contains "qwen_image"
        ],
    )
    def test_arch_containing_diffusion_token_is_not_misrouted(self, arch):
        out = f"error loading model: unknown model architecture: '{arch}'"
        msg = _classify(out, f"/models/{arch}.gguf", f"local/{arch}")
        assert arch in msg
        assert "does not support" in msg.lower()
        assert "diffusion" not in msg.lower()
        assert "Images page" not in msg


class TestOllamaAndFallback:
    _OLLAMA_GGUF = (
        f"/home/u/.ollama{__import__('os').sep}ollama_links"
        f"{__import__('os').sep}m.gguf"
    )

    def test_ollama_compat_message_still_works(self):
        out = "llama_model_load: error loading model: key not found"
        msg = _classify(out, self._OLLAMA_GGUF, "ollama/llama3")
        assert "Ollama" in msg

    def test_ollama_unknown_arch_keeps_ollama_guidance(self):
        # Ollama + non-diffusion unknown arch keeps the Ollama hint, not the
        # generic llama.cpp "unsupported" message.
        out = "error loading model: unknown model architecture: 'some_new_llm'"
        msg = _classify(out, self._OLLAMA_GGUF, "ollama/some-new")
        assert "Ollama" in msg
        assert "directly through Ollama" in msg
        assert "does not support" not in msg.lower()

    def test_ollama_diffusion_arch_still_routes_to_images(self):
        # Diffusion routing wins over the Ollama hint.
        out = "error loading model: unknown model architecture: 'flux'"
        msg = _classify(out, self._OLLAMA_GGUF, "ollama/flux")
        assert "diffusion" in msg.lower()
        assert "Images page" in msg

    def test_generic_oom_keeps_memory_message(self):
        msg = _classify(_OOM_OUT, "/models/big.gguf", "local/big")
        assert "enough memory" in msg.lower()
        assert "diffusion" not in msg.lower()

    def test_empty_output_is_safe(self):
        msg = _classify("", None, None)
        assert "llama-server failed to start" in msg

    def test_health_timeout_names_probe_not_generic(self):
        # A live server that never returns 200 on /health must name the probe and
        # proxy/context causes, not blame a bad GGUF (#5740).
        msg = _classify(
            "llama-server health check timed out after 600.0s",
            "/models/x.gguf",
            "local/x",
        )
        assert "/health" in msg
        assert "NO_PROXY" in msg
        assert "GGUF file is valid" not in msg


class TestOsKillReturncode:
    """SIGKILL (-9) with no diagnostic output is the OOM killer and gets a named,
    actionable message; SIGTERM (-15) is also unload/cancel/supervisor stop, so it
    stays neutral; a recognized output still wins; a hard fault (-11) keeps the
    generic fallback."""

    def test_sigkill_with_no_output_names_oom(self):
        msg = _classify("", "/models/big-bf16.gguf", "local/big", -9)
        assert "signal 9" in msg
        assert "out of memory" in msg.lower()
        assert ".wslconfig" in msg
        assert "GGUF file is valid" not in msg

    def test_sigterm_is_neutral_not_oom(self):
        msg = _classify("", "/models/big-bf16.gguf", "local/big", -15)
        assert "signal 15" in msg
        assert "terminated" in msg.lower()
        assert "out of memory" not in msg.lower()

    def test_specific_output_wins_over_os_kill_code(self):
        msg = _classify(
            _QWEN_IMAGE_OUT, "/models/qwen-image.gguf", "local/qwen-image", -9
        )
        assert "diffusion" in msg.lower()
        assert "out of memory" not in msg.lower()

    def test_signal_crash_code_keeps_generic_message(self):
        # -11 is handled by the retry ladder; if it reaches here with no output
        # it gets the generic fallback, not the OOM message.
        msg = _classify("", "/models/x.gguf", "local/x", -11)
        assert "GGUF file is valid" in msg
        assert "out of memory" not in msg.lower()
