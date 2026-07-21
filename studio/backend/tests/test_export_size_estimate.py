# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for GET /api/models/export-size (the Export page size estimate).

The endpoint must never raise and must degrade to nulls when size is unknown.
"""

import asyncio
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_BACKEND_ROOT = Path(__file__).resolve().parent.parent

# Real Qwen3.6-35B-A3B: 35.95B params -> ~67 GiB bf16 (UI wrongly showed Q8 ~8.2 GB).
_QWEN35_PARAMS = 35_951_822_704
_QWEN35_FP16_BYTES = _QWEN35_PARAMS * 2


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
        self.models_route._EXPORT_SIZE_CACHE.clear()

    def _call(self, model: str = "unsloth/Qwen3.6-35B-A3B"):
        with (
            patch.object(self.models_route, "is_local_path", return_value = False),
            patch.object(
                self.models_route,
                "resolve_cached_repo_id_case",
                side_effect = lambda m: m,
            ),
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
        # MoE sized via the sizer's config path -> source "config".
        with patch(
            "utils.hardware.hardware.estimate_fp16_model_size_bytes",
            return_value = (67 * (1024**3), "config"),
        ):
            resp = self._call()
        self.assertEqual(resp.fp16_bytes, 67 * (1024**3))
        self.assertEqual(resp.total_params, (67 * (1024**3)) // 2)
        self.assertEqual(resp.source, "config")

    def test_unknown_size_returns_nulls_not_error(self):
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
        self.assertEqual(mock_sizer.call_count, 1)

    def test_failures_are_not_cached(self):
        # A transient failure must not poison the cache; a later call recovers.
        with patch(
            "utils.hardware.hardware.estimate_fp16_model_size_bytes",
            side_effect = [(None, "unavailable"), (_QWEN35_FP16_BYTES, "safetensors")],
        ) as mock_sizer:
            first = self._call()
            second = self._call()
        self.assertIsNone(first.fp16_bytes)
        self.assertEqual(second.fp16_bytes, _QWEN35_FP16_BYTES)
        self.assertEqual(mock_sizer.call_count, 2)

    def test_token_is_forwarded_to_sizer(self):
        with (
            patch.object(self.models_route, "is_local_path", return_value = False),
            patch.object(
                self.models_route,
                "resolve_cached_repo_id_case",
                side_effect = lambda m: m,
            ),
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
        # Unsafe local paths must not be scanned -> unavailable.
        with (
            patch.object(self.models_route, "is_local_path", return_value = True),
            patch.object(
                self.models_route, "_is_sizable_local_path", return_value = False
            ),
            patch(
                "utils.hardware.hardware.estimate_fp16_model_size_bytes"
            ) as mock_sizer,
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
        with (
            patch.object(self.models_route, "is_local_path", return_value = True),
            patch.object(
                self.models_route, "_is_sizable_local_path", return_value = True
            ),
            patch(
                "utils.hardware.hardware._resolve_model_identifier_for_gpu_estimate",
                side_effect = lambda m, **_kw: m,
            ),
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

    def test_local_adapter_base_escaping_roots_is_rejected(self):
        # A local adapter under a root whose resolved base points outside the
        # roots (e.g. "/") must not be sized: the resolved base is re-validated.
        adapter = "/root/.unsloth/studio/outputs/adapter"
        with (
            patch.object(self.models_route, "is_local_path", return_value = True),
            patch.object(
                self.models_route,
                "_is_sizable_local_path",
                side_effect = lambda p: p == adapter,
            ),
            patch(
                "utils.hardware.hardware._resolve_model_identifier_for_gpu_estimate",
                return_value = "/",
            ),
            patch(
                "utils.hardware.hardware.estimate_fp16_model_size_bytes"
            ) as mock_sizer,
        ):
            resp = asyncio.run(
                self.models_route.get_export_size(
                    model = adapter, hf_token = None, current_subject = "test-user"
                )
            )
        self.assertIsNone(resp.fp16_bytes)
        self.assertEqual(resp.source, "unavailable")
        mock_sizer.assert_not_called()

    def test_is_sizable_local_path_containment(self):
        # Only paths under a trusted root are sizable; '..' can't escape.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            inside = root / "run-1"
            inside.mkdir(parents = True)
            with (
                patch("utils.paths.studio_root", return_value = root),
                patch("utils.paths.outputs_root", return_value = root),
                patch("utils.paths.exports_root", return_value = root),
                patch("utils.paths.storage_roots.cache_root", return_value = root),
            ):
                is_sizable = self.models_route._is_sizable_local_path
                self.assertTrue(is_sizable(str(inside)))
                self.assertTrue(is_sizable(str(root)))
                self.assertFalse(is_sizable(str(root / "missing")))
                self.assertFalse(is_sizable("/etc"))
                self.assertFalse(is_sizable(str(root / ".." / "etc")))
                # A symlink inside a root pointing outside it cannot escape.
                escape = root / "escape"
                os.symlink(tmp, escape)
                self.assertFalse(is_sizable(str(escape)))

    def test_local_weight_size_skips_nested_checkpoints(self):
        # A run dir's intermediate checkpoint-*/global_step* snapshots must not
        # be counted; only the model files at the root are summed.
        from utils.hardware.hardware import _get_local_weight_size_bytes
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "model.safetensors").write_bytes(b"\0" * 1000)
            for sub, size in (("checkpoint-60", 5000), ("global_step10", 7000)):
                d = run / sub
                d.mkdir()
                (d / "model.safetensors").write_bytes(b"\0" * size)
            self.assertEqual(_get_local_weight_size_bytes(str(run)), 1000)


if __name__ == "__main__":
    unittest.main()
