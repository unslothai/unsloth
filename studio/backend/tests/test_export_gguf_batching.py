# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Unit tests for the GGUF export quantization_method normalization helper.

Tests the pure helper function `normalize_gguf_quantization_method` in
isolation — no FastAPI app, no TestClient, no route invocation, no stubs.
The helper owns the Union[str, List[str]] coercion, case normalization,
deduplication, and empty-list rejection that `routes/export.py` relies on.

Corresponds to the design spec at
`.claude/specs/2026-04-14-studio-batch-gguf-exports-design.md`.

No GPU, no network, no heavy imports — runs in milliseconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Match conftest.py's sys.path handling so flat imports from the backend
# root (e.g. `from models.export import ...`) resolve.
_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from models.export import (
    ExportGGUFRequest,
    normalize_gguf_quantization_method,
)


class TestNormalizeGGUFQuantizationMethod:
    """Covers the five normalization contracts the route handler relies on."""

    def test_string_input_wraps_and_lowercases(self):
        """A single string becomes a single-element lowercase list."""
        assert normalize_gguf_quantization_method("Q4_K_M") == ["q4_k_m"]

    def test_list_input_lowercases_each_element(self):
        """A list of strings is lowercased element-wise, order preserved."""
        assert normalize_gguf_quantization_method(["Q4_K_M", "BF16"]) == [
            "q4_k_m",
            "bf16",
        ]

    def test_duplicate_formats_deduped_preserving_first_seen_order(self):
        """Dedup runs after lowercasing; first occurrence wins the position."""
        result = normalize_gguf_quantization_method(
            ["Q4_K_M", "q4_k_m", "Q8_0", "Q4_K_M"],
        )
        assert result == ["q4_k_m", "q8_0"]

    def test_empty_list_raises_valueerror(self):
        """Empty input is rejected so the route can map it to HTTP 400."""
        with pytest.raises(ValueError, match = "at least one"):
            normalize_gguf_quantization_method([])

    def test_single_element_list_passes_through(self):
        """Single-element list is a valid input (equivalent to the string form)."""
        assert normalize_gguf_quantization_method(["BF16"]) == ["bf16"]


class TestExportGGUFRequestDefault:
    """Integration between Pydantic default and the normalization helper."""

    def test_pydantic_default_normalizes_to_lowercase_list(self):
        """When the caller omits quantization_method, the default string flows
        through the helper and becomes a one-element lowercase list."""
        req = ExportGGUFRequest(save_directory = "/tmp/x")
        assert normalize_gguf_quantization_method(req.quantization_method) == [
            "q4_k_m",
        ]

    def test_pydantic_accepts_string_input(self):
        """Legacy scripted callers can still send a single-format string."""
        req = ExportGGUFRequest(
            save_directory = "/tmp/x",
            quantization_method = "Q4_K_M",
        )
        assert normalize_gguf_quantization_method(req.quantization_method) == [
            "q4_k_m",
        ]

    def test_pydantic_accepts_list_input(self):
        """New callers send the list form."""
        req = ExportGGUFRequest(
            save_directory = "/tmp/x",
            quantization_method = ["Q4_K_M", "BF16"],
        )
        assert normalize_gguf_quantization_method(req.quantization_method) == [
            "q4_k_m",
            "bf16",
        ]
