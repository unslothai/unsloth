# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for LlamaCppBackend._classify_llama_start_failure.

When llama-server exits before becoming healthy, load_model turns its
captured stdout/stderr into a user-facing reason. A diffusion / image
GGUF (FLUX, Qwen-Image, ...) is a valid file with plenty of memory, so
the generic "invalid file or out of memory" message is actively
misleading (issue #5842). These tests pin the classification.
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
sys.modules.setdefault("structlog", _types.ModuleType("structlog"))

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

    @pytest.mark.parametrize(
        "arch",
        ["flux", "sd1", "sdxl", "sd3", "aura", "hidream", "cosmos", "ltxv", "hyvid", "wan"],
    )
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


class TestOllamaAndFallback:
    def test_ollama_compat_message_still_works(self):
        out = "llama_model_load: error loading model: key not found"
        gguf = f"/home/u/.ollama{__import__('os').sep}ollama_links{__import__('os').sep}m.gguf"
        msg = _classify(out, gguf, "ollama/llama3")
        assert "Ollama" in msg

    def test_generic_oom_keeps_memory_message(self):
        msg = _classify(_OOM_OUT, "/models/big.gguf", "local/big")
        assert "enough memory" in msg.lower()
        assert "diffusion" not in msg.lower()

    def test_empty_output_is_safe(self):
        msg = _classify("", None, None)
        assert "llama-server failed to start" in msg
