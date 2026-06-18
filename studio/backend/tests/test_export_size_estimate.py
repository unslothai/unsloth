# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the Export page model-size estimate endpoint (GET /api/models/export-size).

The Export GGUF quant picker scales its per-quant size estimates from this
endpoint's ``fp16_bytes`` instead of a hardcoded, model-independent constant.
The endpoint must never raise (a size hint must not break the Export page) and
must degrade to nulls when the size can't be determined.
"""

import asyncio
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

_BACKEND_ROOT = Path(__file__).resolve().parent.parent

# Real Qwen3.6-35B-A3B counts (the model from the reported issue): 35.95B params
# -> ~67 GiB bf16. The old UI wrongly showed Q8 ~8.2 GB for this model.
_QWEN35_PARAMS = 35_951_822_704
_QWEN35_FP16_BYTES = _QWEN35_PARAMS * 2  # ~71.9e9 bytes (~67 GiB)


def _load_route_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, _BACKEND_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestExportSizeEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models_route = _load_route_module(
            "models_route_module_for_export_size_test",
            "routes/models.py",
        )

    def setUp(self):
        # Each test starts with a cold cache so memoization doesn't leak.
        self.models_route._EXPORT_SIZE_CACHE.clear()

    def _call(self, model: str = "unsloth/Qwen3.6-35B-A3B"):
        # Keep id resolution deterministic and offline (no HF cache lookup).
        with (
            patch.object(self.models_route, "is_local_path", return_value = False),
            patch.object(self.models_route, "resolve_cached_repo_id_case", side_effect = lambda m: m),
        ):
            return asyncio.run(
                self.models_route.get_export_size(
                    model = model, hf_token = None, current_subject = "test-user"
                )
            )

    def test_known_model_returns_bytes_and_params(self):
        with patch(
            "utils.hardware.hardware.estimate_fp16_model_size_bytes",
            return_value = (_QWEN35_FP16_BYTES, "safetensors"),
        ):
            resp = self._call()
        self.assertEqual(resp.fp16_bytes, _QWEN35_FP16_BYTES)
        self.assertEqual(resp.total_params, _QWEN35_PARAMS)
        self.assertEqual(resp.source, "safetensors")
        self.assertEqual(resp.model, "unsloth/Qwen3.6-35B-A3B")

    def test_moe_via_config_fallback(self):
        # Local/uncached MoE: the sizer's config path counts experts and returns
        # the same ~67 GiB; the endpoint surfaces it with source "config".
        with patch(
            "utils.hardware.hardware.estimate_fp16_model_size_bytes",
            return_value = (67 * (1024**3), "config"),
        ):
            resp = self._call()
        self.assertEqual(resp.fp16_bytes, 67 * (1024**3))
        self.assertEqual(resp.total_params, (67 * (1024**3)) // 2)
        self.assertEqual(resp.source, "config")

    def test_unknown_size_returns_nulls_not_error(self):
        # Offline / gated / unresolved: sizer returns (None, "unavailable").
        with patch(
            "utils.hardware.hardware.estimate_fp16_model_size_bytes",
            return_value = (None, "unavailable"),
        ):
            resp = self._call()
        self.assertIsNone(resp.fp16_bytes)
        self.assertIsNone(resp.total_params)
        self.assertEqual(resp.source, "unavailable")

    def test_zero_size_treated_as_unknown(self):
        with patch(
            "utils.hardware.hardware.estimate_fp16_model_size_bytes",
            return_value = (0, "safetensors"),
        ):
            resp = self._call()
        self.assertIsNone(resp.fp16_bytes)
        self.assertIsNone(resp.total_params)

    def test_sizer_exception_is_swallowed(self):
        # A size hint must never break the Export page -> nulls, no raise.
        with patch(
            "utils.hardware.hardware.estimate_fp16_model_size_bytes",
            side_effect = RuntimeError("boom"),
        ):
            resp = self._call()
        self.assertIsNone(resp.fp16_bytes)
        self.assertEqual(resp.source, "unavailable")

    def test_result_is_memoized_per_model(self):
        with patch(
            "utils.hardware.hardware.estimate_fp16_model_size_bytes",
            return_value = (_QWEN35_FP16_BYTES, "safetensors"),
        ) as mock_sizer:
            first = self._call()
            second = self._call()
        self.assertEqual(first.fp16_bytes, second.fp16_bytes)
        # Second identical call is served from cache (sizer called once).
        self.assertEqual(mock_sizer.call_count, 1)

    def test_failures_are_not_cached(self):
        # A transient failure must NOT poison the cache: once metadata (or a
        # token) becomes available, a later call for the same model recovers.
        with patch(
            "utils.hardware.hardware.estimate_fp16_model_size_bytes",
            side_effect = [(None, "unavailable"), (_QWEN35_FP16_BYTES, "safetensors")],
        ) as mock_sizer:
            first = self._call()
            second = self._call()
        self.assertIsNone(first.fp16_bytes)
        self.assertEqual(second.fp16_bytes, _QWEN35_FP16_BYTES)
        # The failed first call was re-attempted (not served from cache).
        self.assertEqual(mock_sizer.call_count, 2)

    def test_token_is_forwarded_to_sizer(self):
        # The X-HF-Token header value must reach the sizer for gated repos.
        with (
            patch.object(self.models_route, "is_local_path", return_value = False),
            patch.object(self.models_route, "resolve_cached_repo_id_case", side_effect = lambda m: m),
            patch(
                "utils.hardware.hardware.estimate_fp16_model_size_bytes",
                return_value = (_QWEN35_FP16_BYTES, "safetensors"),
            ) as mock_sizer,
        ):
            asyncio.run(
                self.models_route.get_export_size(
                    model = "unsloth/Private",
                    hf_token = "secret-token",
                    current_subject = "test-user",
                )
            )
        self.assertEqual(mock_sizer.call_args.kwargs.get("hf_token"), "secret-token")

    def test_arbitrary_local_path_is_not_scanned(self):
        # An authenticated caller must not be able to make the sizer rglob an
        # arbitrary directory; unsafe local paths return unavailable, unscanned.
        with (
            patch.object(self.models_route, "is_local_path", return_value = True),
            patch.object(self.models_route, "_is_sizable_local_path", return_value = False),
            patch("utils.hardware.hardware.estimate_fp16_model_size_bytes") as mock_sizer,
        ):
            resp = asyncio.run(
                self.models_route.get_export_size(
                    model = "/etc", hf_token = None, current_subject = "test-user"
                )
            )
        self.assertIsNone(resp.fp16_bytes)
        self.assertEqual(resp.source, "unavailable")
        mock_sizer.assert_not_called()

    def test_sizable_local_path_is_sized(self):
        # A local path under an allowed Studio root is sized normally.
        with (
            patch.object(self.models_route, "is_local_path", return_value = True),
            patch.object(self.models_route, "_is_sizable_local_path", return_value = True),
            patch(
                "utils.hardware.hardware.estimate_fp16_model_size_bytes",
                return_value = (_QWEN35_FP16_BYTES, "local"),
            ),
        ):
            resp = asyncio.run(
                self.models_route.get_export_size(
                    model = "/root/.unsloth/studio/outputs/run",
                    hf_token = None,
                    current_subject = "test-user",
                )
            )
        self.assertEqual(resp.fp16_bytes, _QWEN35_FP16_BYTES)
        self.assertEqual(resp.source, "local")


if __name__ == "__main__":
    unittest.main()
