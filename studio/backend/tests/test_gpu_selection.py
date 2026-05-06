# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import importlib.util
import os
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from core.training.training import TrainingBackend
from models.inference import LoadRequest
from models.training import TrainingStartRequest
from utils.hardware import (
    apply_gpu_ids,
    DeviceType,
    auto_select_gpu_ids,
    estimate_required_model_memory_gb,
    get_backend_visible_gpu_info,
    get_device_map,
    get_offloaded_device_map_entries,
    get_parent_visible_gpu_ids,
    get_visible_gpu_utilization,
    prepare_gpu_selection,
    resolve_requested_gpu_ids,
)
import utils.hardware.hardware as _hw_module

_BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _load_route_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, _BACKEND_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _GpuCacheResetMixin:
    """Reset module-level GPU caches between tests to prevent state leaks."""

    def tearDown(self):
        _hw_module._physical_gpu_count = None
        _hw_module._visible_gpu_count = None


class TestResolveRequestedGpuIds(_GpuCacheResetMixin, unittest.TestCase):
    def test_parent_visibility_defaults_to_physical_enumeration(self):
        with (
            patch.dict(os.environ, {}, clear = True),
            patch("utils.hardware.hardware.get_physical_gpu_count", return_value = 4),
        ):
            self.assertEqual(get_parent_visible_gpu_ids(), [0, 1, 2, 3])
            self.assertEqual(resolve_requested_gpu_ids(None), [0, 1, 2, 3])

    def test_parent_visibility_uses_cuda_visible_devices(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,3"}, clear = True):
            self.assertEqual(get_parent_visible_gpu_ids(), [1, 3])
            self.assertEqual(resolve_requested_gpu_ids(None), [1, 3])

    def test_parent_visibility_uses_empty_numeric_ids_for_uuid_masks(self):
        with (
            patch.dict(
                os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-aaa,GPU-bbb"}, clear = True
            ),
            patch("utils.hardware.hardware.get_physical_gpu_count", return_value = 8),
        ):
            self.assertEqual(get_parent_visible_gpu_ids(), [])

    def test_invalid_requests_raise_clear_value_errors(self):
        cases = [
            ([1, 1], "duplicate GPU IDs"),
            ([-1], "Rejected IDs: [-1]"),
            ([99], "Rejected IDs: [99]"),
            ([0], "outside the parent-visible set [1, 3]"),
        ]
        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,3"}, clear = True),
            patch("utils.hardware.hardware.get_physical_gpu_count", return_value = 8),
        ):
            for gpu_ids, message in cases:
                with self.subTest(gpu_ids = gpu_ids):
                    with self.assertRaisesRegex(ValueError, re.escape(message)):
                        resolve_requested_gpu_ids(gpu_ids)

    def test_explicit_ids_must_be_physical_not_relative(self):
        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,3"}, clear = True),
            patch("utils.hardware.hardware.get_physical_gpu_count", return_value = 8),
        ):
            self.assertEqual(resolve_requested_gpu_ids([1, 3]), [1, 3])

    def test_explicit_ids_are_rejected_for_uuid_parent_visibility(self):
        with (
            patch.dict(
                os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-aaa,GPU-bbb"}, clear = True
            ),
            patch("utils.hardware.hardware.get_physical_gpu_count", return_value = 8),
        ):
            with self.assertRaisesRegex(
                ValueError, "unsupported when CUDA_VISIBLE_DEVICES uses UUID/MIG"
            ):
                resolve_requested_gpu_ids([1])

    def test_empty_list_is_treated_as_auto(self):
        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,3"}, clear = True),
            patch("utils.hardware.hardware.get_physical_gpu_count", return_value = 8),
        ):
            self.assertEqual(resolve_requested_gpu_ids([]), [1, 3])

    def test_apply_gpu_ids_only_updates_cuda_visible_devices(self):
        with patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "1,3", "TEST_PARENT_ENV": "keep-me"},
            clear = True,
        ):
            apply_gpu_ids([5, 6])

            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "5,6")
            self.assertEqual(os.environ["TEST_PARENT_ENV"], "keep-me")


class TestVisibleGpuUtilization(_GpuCacheResetMixin, unittest.TestCase):
    def test_visible_gpu_utilization_filters_to_parent_visible_ids(self):
        smi_output = "\n".join(
            [
                "0, 10, 30, 1000, 10000, 50, 100",
                "1, 20, 40, 2000, 10000, 60, 120",
                "3, 30, 50, 3000, 10000, 70, 140",
            ]
        )

        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,3"}, clear = True),
            patch("utils.hardware.hardware.get_device", return_value = DeviceType.CUDA),
            patch("utils.hardware.nvidia.subprocess.run") as mock_run,
        ):
            mock_run.return_value = SimpleNamespace(
                returncode = 0,
                stdout = smi_output,
            )
            result = get_visible_gpu_utilization()

        self.assertTrue(result["available"])
        self.assertEqual(result["parent_visible_gpu_ids"], [1, 3])
        self.assertEqual(result["index_kind"], "physical")
        self.assertEqual([device["index"] for device in result["devices"]], [1, 3])
        self.assertEqual(result["devices"][0]["visible_ordinal"], 0)
        self.assertEqual(result["devices"][1]["visible_ordinal"], 1)
        self.assertEqual(result["devices"][0]["gpu_utilization_pct"], 20.0)
        self.assertEqual(result["devices"][1]["power_utilization_pct"], 50.0)

    def test_backend_visible_gpu_info_preserves_physical_indices(self):
        smi_output = "\n".join(
            [
                "0, GPU Zero, 10000",
                "1, GPU One, 20000",
                "3, GPU Three, 30000",
            ]
        )

        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,3"}, clear = True),
            patch("utils.hardware.hardware.get_device", return_value = DeviceType.CUDA),
            patch("utils.hardware.nvidia.subprocess.run") as mock_run,
        ):
            mock_run.return_value = SimpleNamespace(
                returncode = 0,
                stdout = smi_output,
            )
            result = get_backend_visible_gpu_info()

        self.assertTrue(result["available"])
        self.assertEqual(result["parent_visible_gpu_ids"], [1, 3])
        self.assertEqual(result["index_kind"], "physical")
        self.assertEqual([device["index"] for device in result["devices"]], [1, 3])
        self.assertEqual(result["devices"][0]["visible_ordinal"], 0)
        self.assertEqual(result["devices"][1]["visible_ordinal"], 1)
        self.assertEqual(result["devices"][0]["name"], "GPU One")
        self.assertAlmostEqual(result["devices"][1]["memory_total_gb"], 29.3, places = 1)

    def test_uuid_parent_visibility_falls_back_to_torch(self):
        """UUID/MIG masks should fall through nvidia to torch fallback and
        still report visible devices using relative ordinals."""
        fake_torch_devices = [
            {
                "index": 0,
                "visible_ordinal": 0,
                "name": "GPU-A",
                "total_gb": 24.0,
                "used_gb": 2.0,
            },
            {
                "index": 1,
                "visible_ordinal": 1,
                "name": "GPU-B",
                "total_gb": 24.0,
                "used_gb": 3.0,
            },
        ]
        with (
            patch.dict(
                os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-aaa,GPU-bbb"}, clear = True
            ),
            patch("utils.hardware.hardware.get_device", return_value = DeviceType.CUDA),
            patch(
                "utils.hardware.hardware._torch_get_physical_gpu_count", return_value = 2
            ),
            patch(
                "utils.hardware.hardware._torch_get_per_device_info",
                return_value = fake_torch_devices,
            ),
        ):
            result = get_backend_visible_gpu_info()

        self.assertTrue(result["available"])
        self.assertEqual(result["parent_visible_gpu_ids"], [])
        self.assertEqual(len(result["devices"]), 2)
        self.assertEqual(result["index_kind"], "relative")

    def test_mlx_visible_gpu_info_is_best_effort_relative(self):
        with (
            patch("utils.hardware.hardware.get_device", return_value = DeviceType.MLX),
            patch(
                "utils.hardware.hardware.get_gpu_memory_info",
                return_value = {
                    "available": True,
                    "device_name": "Apple Silicon",
                    "total_gb": 64.0,
                    "allocated_gb": 8.0,
                    "utilization_pct": 12.5,
                },
            ),
        ):
            result = get_backend_visible_gpu_info()

        self.assertTrue(result["available"])
        self.assertEqual(result["index_kind"], "relative")
        self.assertEqual(result["devices"][0]["index"], 0)
        self.assertEqual(result["devices"][0]["visible_ordinal"], 0)


class TestGpuAutoSelection(_GpuCacheResetMixin, unittest.TestCase):
    def test_get_device_map_uses_explicit_gpu_selection(self):
        with patch("utils.hardware.hardware.get_device", return_value = DeviceType.CUDA):
            self.assertEqual(get_device_map(None), "sequential")
            self.assertEqual(get_device_map([0]), "sequential")
            self.assertEqual(get_device_map([0, 1]), "balanced")

    def test_get_device_map_uses_all_inherited_visible_gpus_for_uuid_masks(self):
        with (
            patch.dict(
                os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-aaa,GPU-bbb"}, clear = True
            ),
            patch("utils.hardware.hardware.get_device", return_value = DeviceType.CUDA),
        ):
            self.assertEqual(get_device_map(None), "balanced")

    def test_get_offloaded_device_map_entries_returns_only_cpu_and_disk(self):
        model = SimpleNamespace(
            hf_device_map = {
                "model.embed_tokens": 0,
                "model.layers.0": 1,
                "model.layers.1": "cpu",
                "lm_head": "disk",
            }
        )

        self.assertEqual(
            get_offloaded_device_map_entries(model),
            {
                "model.layers.1": "cpu",
                "lm_head": "disk",
            },
        )

    def test_get_offloaded_device_map_entries_handles_models_without_device_map(self):
        self.assertEqual(get_offloaded_device_map_entries(SimpleNamespace()), {})

    def test_estimate_required_memory_formulas(self):
        eight_gb = 8 * (1024**3)

        with patch(
            "utils.hardware.hardware.estimate_fp16_model_size_bytes",
            return_value = (eight_gb, "config"),
        ):
            # FP16 inference: 8GB * 1.3 = 10.4GB
            required_gb, metadata = estimate_required_model_memory_gb(
                "unsloth/test",
                load_in_4bit = False,
            )
            self.assertAlmostEqual(required_gb, 10.4, places = 3)
            self.assertEqual(metadata["model_size_source"], "config")

            # 4bit inference: base_4bit = 8/3.2 = 2.5GB
            # required = 2.5 + max(2.5*0.3, 2.0) = 2.5 + 2.0 = 4.5GB
            required_gb, _ = estimate_required_model_memory_gb(
                "unsloth/test",
                load_in_4bit = True,
            )
            self.assertAlmostEqual(required_gb, 4.5, places = 2)

            # Full FT fallback: model_size * 3.5 + overhead
            required_gb, metadata = estimate_required_model_memory_gb(
                "unsloth/test", training_type = "Full Finetuning"
            )
            self.assertEqual(metadata.get("estimation_mode"), "fallback")
            self.assertGreater(required_gb, 25.0)
            self.assertLess(required_gb, 40.0)

            # LoRA fp16 fallback: model_size + lora_overhead + activations + overhead
            required_gb, metadata = estimate_required_model_memory_gb(
                "unsloth/test",
                training_type = "LoRA/QLoRA",
                load_in_4bit = False,
            )
            self.assertEqual(metadata.get("estimation_mode"), "fallback")
            self.assertGreater(required_gb, 8.0)
            self.assertLess(required_gb, 15.0)

            # QLoRA 4-bit fallback: compressed weights + lora overhead + activations + overhead
            required_gb, metadata = estimate_required_model_memory_gb(
                "unsloth/test",
                training_type = "LoRA/QLoRA",
                load_in_4bit = True,
            )
            self.assertEqual(metadata.get("estimation_mode"), "fallback")
            self.assertGreater(required_gb, 3.0)
            self.assertLess(required_gb, 8.0)

        # Larger model: 16GB fp16
        sixteen_gb = 16 * (1024**3)
        with patch(
            "utils.hardware.hardware.estimate_fp16_model_size_bytes",
            return_value = (sixteen_gb, "config"),
        ):
            required_gb, _ = estimate_required_model_memory_gb(
                "unsloth/test",
                training_type = "LoRA/QLoRA",
                load_in_4bit = True,
            )
            # QLoRA for 16GB model should be < 12 GB
            self.assertGreater(required_gb, 5.0)
            self.assertLess(required_gb, 12.0)

    def test_estimate_fp16_model_size_bytes_uses_vllm_fallback_last(self):
        config = object()
        with (
            patch(
                "utils.hardware.hardware._resolve_model_identifier_for_gpu_estimate",
                return_value = "unsloth/test",
            ),
            patch(
                "utils.hardware.hardware._get_hf_safetensors_total_params",
                return_value = None,
            ),
            patch(
                "utils.hardware.hardware._load_config_for_gpu_estimate",
                return_value = config,
            ),
            patch(
                "utils.hardware.hardware._estimate_fp16_model_size_bytes_from_config",
                return_value = None,
            ),
            patch(
                "utils.hardware.hardware._get_local_weight_size_bytes",
                return_value = None,
            ),
            patch(
                "utils.hardware.hardware._estimate_fp16_model_size_bytes_from_vllm_utils",
                return_value = 1234,
            ),
        ):
            model_size_bytes, source = _hw_module.estimate_fp16_model_size_bytes(
                "unsloth/test"
            )

        self.assertEqual(model_size_bytes, 1234)
        self.assertEqual(source, "vllm_utils")

    def test_auto_select_gpu_ids_chooses_smallest_fitting_subset(self):
        fake_devices = {
            "devices": [
                {"index": 0, "vram_total_gb": 16.0, "vram_used_gb": 4.0},
                {"index": 1, "vram_total_gb": 16.0, "vram_used_gb": 6.0},
                {"index": 2, "vram_total_gb": 16.0, "vram_used_gb": 7.0},
            ]
        }

        with (
            patch("utils.hardware.hardware.get_device", return_value = DeviceType.CUDA),
            patch(
                "utils.hardware.hardware.estimate_required_model_memory_gb",
                return_value = (
                    14.0,
                    {"required_gb": 14.0, "model_size_source": "config"},
                ),
            ),
            patch(
                "utils.hardware.hardware.get_visible_gpu_utilization",
                return_value = fake_devices,
            ),
        ):
            selected, metadata = auto_select_gpu_ids("unsloth/test")

        self.assertEqual(selected, [0, 1])
        self.assertEqual(metadata["selection_mode"], "auto")
        # First GPU full (12GB) + second GPU with overhead (10*0.85=8.5) = 20.5GB
        self.assertAlmostEqual(metadata["usable_gb"], 20.5, places = 3)

    def test_auto_select_gpu_ids_falls_back_to_all_visible(self):
        fake_devices = {
            "devices": [
                {"index": 0, "vram_total_gb": 12.0, "vram_used_gb": 2.0},
                {"index": 1, "vram_total_gb": 12.0, "vram_used_gb": 2.0},
            ]
        }

        with (
            patch("utils.hardware.hardware.get_device", return_value = DeviceType.CUDA),
            patch(
                "utils.hardware.hardware.estimate_required_model_memory_gb",
                return_value = (
                    30.0,
                    {"required_gb": 30.0, "model_size_source": "config"},
                ),
            ),
            patch(
                "utils.hardware.hardware.get_visible_gpu_utilization",
                return_value = fake_devices,
            ),
        ):
            selected, metadata = auto_select_gpu_ids("unsloth/test")

        self.assertEqual(selected, [0, 1])
        self.assertEqual(metadata["selection_mode"], "fallback_all")
        # First GPU full (10GB) + second GPU with overhead (10*0.85=8.5) = 18.5GB
        self.assertAlmostEqual(metadata["usable_gb"], 18.5, places = 3)

    def test_prepare_gpu_selection_preserves_explicit_ids_without_auto_selection(self):
        with (
            patch(
                "utils.hardware.hardware.resolve_requested_gpu_ids",
                return_value = [2, 3],
            ),
            patch("utils.hardware.hardware.auto_select_gpu_ids") as mock_auto_select,
        ):
            selected, metadata = prepare_gpu_selection(
                [2, 3],
                model_name = "unsloth/test",
            )

        self.assertEqual(selected, [2, 3])
        self.assertEqual(metadata["selection_mode"], "explicit")
        mock_auto_select.assert_not_called()

    def test_prepare_gpu_selection_treats_empty_list_as_auto(self):
        with patch(
            "utils.hardware.hardware.auto_select_gpu_ids",
            return_value = ([0, 1], {"selection_mode": "auto"}),
        ) as mock_auto_select:
            selected, metadata = prepare_gpu_selection(
                [],
                model_name = "unsloth/test",
            )

        self.assertEqual(selected, [0, 1])
        self.assertEqual(metadata["selection_mode"], "auto")
        mock_auto_select.assert_called_once()

    def test_prepare_gpu_selection_preserves_uuid_parent_visibility_in_auto_mode(self):
        with (
            patch.dict(
                os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-aaa,GPU-bbb"}, clear = True
            ),
            patch(
                "utils.hardware.hardware.estimate_required_model_memory_gb",
                return_value = (
                    14.0,
                    {"required_gb": 14.0, "model_size_source": "config"},
                ),
            ),
        ):
            selected, metadata = prepare_gpu_selection(
                None,
                model_name = "unsloth/test",
            )

        self.assertIsNone(selected)
        self.assertEqual(metadata["selection_mode"], "inherit_parent_visible")
        self.assertIsNone(metadata["selected_gpu_ids"])


class TestPreSpawnGpuResolution(_GpuCacheResetMixin, unittest.TestCase):
    def test_training_backend_resolves_explicit_gpu_ids_before_spawn(self):
        backend = TrainingBackend()

        class DummyProcess:
            pid = 12345

            def start(self):
                return None

        class DummyThread:
            def start(self):
                return None

        dummy_queue = object()

        with (
            patch(
                "core.training.training.prepare_gpu_selection",
                return_value = ([1, 2], {"selection_mode": "explicit"}),
            ),
            patch(
                "core.training.training._CTX.Queue",
                side_effect = [dummy_queue, dummy_queue],
            ),
            patch(
                "core.training.training._CTX.Process", return_value = DummyProcess()
            ) as mock_process,
            patch(
                "core.training.training.threading.Thread", return_value = DummyThread()
            ),
        ):
            backend.start_training(
                job_id = "test-job-1",
                model_name = "unsloth/test",
                training_type = "LoRA/QLoRA",
                gpu_ids = [1, 2],
            )

        config = mock_process.call_args.kwargs["kwargs"]["config"]
        self.assertEqual(config["gpu_ids"], [1, 2])
        self.assertEqual(config["resolved_gpu_ids"], [1, 2])
        self.assertEqual(config["gpu_selection"]["selection_mode"], "explicit")

    def test_training_backend_auto_selects_gpu_ids_when_omitted(self):
        backend = TrainingBackend()

        class DummyProcess:
            pid = 12345

            def start(self):
                return None

        class DummyThread:
            def start(self):
                return None

        dummy_queue = object()

        with (
            patch(
                "core.training.training.prepare_gpu_selection",
                return_value = ([0, 1], {"selection_mode": "auto"}),
            ),
            patch(
                "core.training.training._CTX.Queue",
                side_effect = [dummy_queue, dummy_queue],
            ),
            patch(
                "core.training.training._CTX.Process", return_value = DummyProcess()
            ) as mock_process,
            patch(
                "core.training.training.threading.Thread", return_value = DummyThread()
            ),
        ):
            backend.start_training(
                job_id = "test-job-2",
                model_name = "unsloth/test",
                training_type = "LoRA/QLoRA",
                gpu_ids = None,
            )

        config = mock_process.call_args.kwargs["kwargs"]["config"]
        self.assertIsNone(config["gpu_ids"])
        self.assertEqual(config["resolved_gpu_ids"], [0, 1])
        self.assertEqual(config["gpu_selection"]["selection_mode"], "auto")

    def test_training_backend_preserves_uuid_parent_visibility_in_auto_mode(self):
        backend = TrainingBackend()

        class DummyProcess:
            pid = 12345

            def start(self):
                return None

        class DummyThread:
            def start(self):
                return None

        dummy_queue = object()

        with (
            patch.dict(
                os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-aaa,GPU-bbb"}, clear = True
            ),
            patch(
                "core.training.training._CTX.Queue",
                side_effect = [dummy_queue, dummy_queue],
            ),
            patch(
                "core.training.training._CTX.Process", return_value = DummyProcess()
            ) as mock_process,
            patch(
                "core.training.training.threading.Thread", return_value = DummyThread()
            ),
            patch(
                "utils.hardware.hardware.estimate_required_model_memory_gb",
                return_value = (
                    14.0,
                    {"required_gb": 14.0, "model_size_source": "config"},
                ),
            ),
        ):
            backend.start_training(
                job_id = "test-job-uuid-auto",
                model_name = "unsloth/test",
                training_type = "LoRA/QLoRA",
                gpu_ids = None,
            )

        config = mock_process.call_args.kwargs["kwargs"]["config"]
        self.assertIsNone(config["resolved_gpu_ids"])
        self.assertEqual(
            config["gpu_selection"]["selection_mode"], "inherit_parent_visible"
        )

    def test_inference_orchestrator_resolves_explicit_gpu_ids_before_spawn(self):
        class DummyThread:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                return None

        with patch("core.inference.orchestrator.threading.Thread", DummyThread):
            from core.inference.orchestrator import InferenceOrchestrator

            orchestrator = InferenceOrchestrator()

        config = SimpleNamespace(identifier = "unsloth/test", gguf_variant = None)

        with (
            patch(
                "core.inference.orchestrator.prepare_gpu_selection",
                return_value = ([1], {"selection_mode": "explicit"}),
            ),
            patch.object(orchestrator, "_ensure_subprocess_alive", return_value = False),
            patch.object(orchestrator, "_spawn_subprocess") as mock_spawn,
            patch.object(
                orchestrator,
                "_wait_response",
                return_value = {"success": True, "model_info": {}},
            ),
            patch(
                "utils.transformers_version.needs_transformers_5", return_value = False
            ),
        ):
            self.assertTrue(orchestrator.load_model(config = config, gpu_ids = [1]))

        sub_config = mock_spawn.call_args.args[0]
        self.assertEqual(sub_config["gpu_ids"], [1])
        self.assertEqual(sub_config["resolved_gpu_ids"], [1])
        self.assertEqual(sub_config["gpu_selection"]["selection_mode"], "explicit")

    def test_inference_orchestrator_auto_selects_gpu_ids_when_omitted(self):
        class DummyThread:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                return None

        with patch("core.inference.orchestrator.threading.Thread", DummyThread):
            from core.inference.orchestrator import InferenceOrchestrator

            orchestrator = InferenceOrchestrator()

        config = SimpleNamespace(identifier = "unsloth/test", gguf_variant = None)

        with (
            patch(
                "core.inference.orchestrator.prepare_gpu_selection",
                return_value = ([0], {"selection_mode": "auto"}),
            ),
            patch.object(orchestrator, "_ensure_subprocess_alive", return_value = False),
            patch.object(orchestrator, "_spawn_subprocess") as mock_spawn,
            patch.object(
                orchestrator,
                "_wait_response",
                return_value = {"success": True, "model_info": {}},
            ),
            patch(
                "utils.transformers_version.needs_transformers_5", return_value = False
            ),
        ):
            self.assertTrue(orchestrator.load_model(config = config, gpu_ids = None))

        sub_config = mock_spawn.call_args.args[0]
        self.assertIsNone(sub_config["gpu_ids"])
        self.assertEqual(sub_config["resolved_gpu_ids"], [0])
        self.assertEqual(sub_config["gpu_selection"]["selection_mode"], "auto")


class TestRouteErrors(unittest.TestCase):
    def test_prepare_gpu_selection_rejects_gpu_ids_on_non_cuda_backend(self):
        with patch("utils.hardware.hardware.get_device", return_value = DeviceType.CPU):
            with self.assertRaises(ValueError) as exc_info:
                prepare_gpu_selection([0], model_name = "unsloth/test")

        self.assertIn("only supported on CUDA devices", str(exc_info.exception))

    def test_inference_route_rejects_gpu_ids_for_gguf(self):
        inference_route = _load_route_module(
            "inference_route_module_for_gguf_gpu_ids_test",
            "routes/inference.py",
        )
        request = LoadRequest(model_path = "unsloth/test.gguf", gpu_ids = [0, 1])
        model_config = SimpleNamespace(
            is_gguf = True,
            is_lora = False,
            gguf_hf_repo = None,
            gguf_file = "/tmp/test.gguf",
            gguf_mmproj_file = None,
            gguf_variant = None,
            identifier = "unsloth/test.gguf",
            display_name = "unsloth/test.gguf",
            is_vision = False,
            is_audio = False,
            audio_type = None,
            has_audio_input = False,
        )

        with patch.object(
            inference_route.ModelConfig,
            "from_identifier",
            return_value = model_config,
        ):
            with self.assertRaises(HTTPException) as exc_info:
                asyncio.run(
                    inference_route.load_model(
                        request,
                        SimpleNamespace(
                            app = SimpleNamespace(
                                state = SimpleNamespace(llama_parallel_slots = 1),
                            ),
                        ),
                        current_subject = "test-user",
                    )
                )

        self.assertEqual(exc_info.exception.status_code, 400)
        self.assertIn("GGUF", exc_info.exception.detail)

    def test_training_route_returns_400_for_invalid_gpu_ids(self):
        training_route = _load_route_module(
            "training_route_module_for_test",
            "routes/training.py",
        )
        request = TrainingStartRequest(
            model_name = "unsloth/test",
            training_type = "LoRA/QLoRA",
            format_type = "alpaca",
            gpu_ids = [99],
        )

        class DummyBackend:
            current_job_id = None

            def is_training_active(self):
                return False

            def start_training(self, **kwargs):
                raise ValueError("Invalid gpu_ids [99]")

        with (
            patch.object(
                training_route, "get_training_backend", return_value = DummyBackend()
            ),
            patch(
                "core.inference.get_inference_backend",
                return_value = SimpleNamespace(active_model_name = None),
            ),
            patch(
                "core.export.get_export_backend",
                return_value = SimpleNamespace(current_checkpoint = None),
            ),
        ):
            with self.assertRaises(HTTPException) as exc_info:
                asyncio.run(
                    training_route.start_training(request, current_subject = "test-user")
                )

        self.assertEqual(exc_info.exception.status_code, 400)
        self.assertIn("gpu_ids [99]", exc_info.exception.detail)

    def test_training_route_returns_400_for_uuid_parent_visibility_gpu_ids(self):
        training_route = _load_route_module(
            "training_route_module_for_uuid_parent_visibility_test",
            "routes/training.py",
        )
        request = TrainingStartRequest(
            model_name = "unsloth/test",
            training_type = "LoRA/QLoRA",
            format_type = "alpaca",
            gpu_ids = [1],
        )

        class DummyBackend:
            current_job_id = None

            def is_training_active(self):
                return False

            def start_training(self, **kwargs):
                raise ValueError(
                    "Invalid gpu_ids [1]: explicit physical GPU IDs are unsupported when CUDA_VISIBLE_DEVICES uses UUID/MIG entries"
                )

        with (
            patch.object(
                training_route, "get_training_backend", return_value = DummyBackend()
            ),
            patch(
                "core.inference.get_inference_backend",
                return_value = SimpleNamespace(active_model_name = None),
            ),
            patch(
                "core.export.get_export_backend",
                return_value = SimpleNamespace(current_checkpoint = None),
            ),
        ):
            with self.assertRaises(HTTPException) as exc_info:
                asyncio.run(
                    training_route.start_training(request, current_subject = "test-user")
                )

        self.assertEqual(exc_info.exception.status_code, 400)
        self.assertIn("UUID/MIG", exc_info.exception.detail)

    def test_inference_route_returns_400_for_invalid_gpu_ids(self):
        inference_route = _load_route_module(
            "inference_route_module_for_test",
            "routes/inference.py",
        )
        request = LoadRequest(model_path = "unsloth/test", gpu_ids = [99])
        model_config = SimpleNamespace(
            is_gguf = False,
            is_lora = False,
            path = None,
            identifier = "unsloth/test",
            display_name = "unsloth/test",
            is_vision = False,
            is_audio = False,
            audio_type = None,
            has_audio_input = False,
        )

        class DummyInferenceBackend:
            active_model_name = None
            models = {}

            def load_model(self, **kwargs):
                raise ValueError("Invalid gpu_ids [99]")

        with (
            patch.object(
                inference_route.ModelConfig,
                "from_identifier",
                return_value = model_config,
            ),
            patch.object(
                inference_route,
                "get_inference_backend",
                return_value = DummyInferenceBackend(),
            ),
            patch.object(
                inference_route,
                "get_llama_cpp_backend",
                return_value = SimpleNamespace(is_loaded = False),
            ),
            patch(
                "core.export.get_export_backend",
                return_value = SimpleNamespace(current_checkpoint = None),
            ),
        ):
            with self.assertRaises(HTTPException) as exc_info:
                asyncio.run(
                    inference_route.load_model(
                        request,
                        SimpleNamespace(
                            app = SimpleNamespace(
                                state = SimpleNamespace(llama_parallel_slots = 1),
                            ),
                        ),
                        current_subject = "test-user",
                    )
                )

        self.assertEqual(exc_info.exception.status_code, 400)
        self.assertIn("gpu_ids [99]", exc_info.exception.detail)

    def test_inference_route_returns_400_for_uuid_parent_visibility_gpu_ids(self):
        inference_route = _load_route_module(
            "inference_route_module_for_uuid_parent_visibility_test",
            "routes/inference.py",
        )
        request = LoadRequest(model_path = "unsloth/test", gpu_ids = [1])
        model_config = SimpleNamespace(
            is_gguf = False,
            is_lora = False,
            path = None,
            identifier = "unsloth/test",
            display_name = "unsloth/test",
            is_vision = False,
            is_audio = False,
            audio_type = None,
            has_audio_input = False,
        )

        class DummyInferenceBackend:
            active_model_name = None
            models = {}

            def load_model(self, **kwargs):
                raise ValueError(
                    "Invalid gpu_ids [1]: explicit physical GPU IDs are unsupported when CUDA_VISIBLE_DEVICES uses UUID/MIG entries"
                )

        with (
            patch.object(
                inference_route.ModelConfig,
                "from_identifier",
                return_value = model_config,
            ),
            patch.object(
                inference_route,
                "get_inference_backend",
                return_value = DummyInferenceBackend(),
            ),
            patch.object(
                inference_route,
                "get_llama_cpp_backend",
                return_value = SimpleNamespace(is_loaded = False),
            ),
            patch(
                "core.export.get_export_backend",
                return_value = SimpleNamespace(current_checkpoint = None),
            ),
        ):
            with self.assertRaises(HTTPException) as exc_info:
                asyncio.run(
                    inference_route.load_model(
                        request,
                        SimpleNamespace(
                            app = SimpleNamespace(
                                state = SimpleNamespace(llama_parallel_slots = 1),
                            ),
                        ),
                        current_subject = "test-user",
                    )
                )

        self.assertEqual(exc_info.exception.status_code, 400)
        self.assertIn("UUID/MIG", exc_info.exception.detail)


class TestRaiseIfOffloaded(unittest.TestCase):
    def test_no_offload_is_noop(self):
        from utils.hardware import raise_if_offloaded

        model = SimpleNamespace(hf_device_map = {"model.embed_tokens": 0, "lm_head": 1})
        raise_if_offloaded(model, "balanced", "Test")

    def test_cpu_offload_raises(self):
        from utils.hardware import raise_if_offloaded

        model = SimpleNamespace(
            hf_device_map = {"model.layers.0": 0, "model.layers.1": "cpu"}
        )
        with self.assertRaisesRegex(ValueError, "offloaded"):
            raise_if_offloaded(model, "balanced", "Test")

    def test_no_device_map_attr_is_noop(self):
        from utils.hardware import raise_if_offloaded

        raise_if_offloaded(SimpleNamespace(), "sequential", "Test")


class TestMinGpuVram(unittest.TestCase):
    def test_min_gpu_vram_decreases_with_more_gpus(self):
        from utils.hardware.vram_estimation import (
            ModelArchConfig,
            TrainingVramConfig,
            estimate_training_vram,
        )

        arch = ModelArchConfig(
            hidden_size = 4096,
            num_hidden_layers = 32,
            num_attention_heads = 32,
            num_key_value_heads = 8,
            intermediate_size = 14336,
            vocab_size = 128256,
            tie_word_embeddings = False,
        )
        config = TrainingVramConfig(
            training_method = "qlora",
            load_in_4bit = True,
        )
        breakdown = estimate_training_vram(arch, config)
        v1 = breakdown.min_gpu_vram(1)
        v2 = breakdown.min_gpu_vram(2)
        v4 = breakdown.min_gpu_vram(4)
        self.assertGreater(v1, v2)
        self.assertGreater(v2, v4)
        self.assertGreater(v4, 0)

    def test_total_equals_min_gpu_vram_1(self):
        from utils.hardware.vram_estimation import (
            ModelArchConfig,
            TrainingVramConfig,
            estimate_training_vram,
        )

        arch = ModelArchConfig(
            hidden_size = 4096,
            num_hidden_layers = 32,
            num_attention_heads = 32,
            num_key_value_heads = 8,
            intermediate_size = 14336,
            vocab_size = 128256,
            tie_word_embeddings = False,
        )
        config = TrainingVramConfig(
            training_method = "qlora",
            load_in_4bit = True,
        )
        breakdown = estimate_training_vram(arch, config)
        self.assertEqual(breakdown.total, breakdown.min_gpu_vram(1))


class TestPerGpuFitGuardAllCounts(unittest.TestCase):
    def test_training_estimate_resolves_attention_without_raising(self):
        with (
            patch("utils.hardware.hardware.get_device", return_value = DeviceType.CUDA),
            patch(
                "utils.hardware.hardware.estimate_fp16_model_size_bytes",
                return_value = (8 * (1024**3), "config"),
            ),
            patch(
                "utils.hardware.hardware._resolve_model_identifier_for_gpu_estimate",
                return_value = "unsloth/test",
            ),
            patch(
                "utils.hardware.hardware._load_config_for_gpu_estimate",
                return_value = SimpleNamespace(
                    hidden_size = 4096,
                    num_hidden_layers = 32,
                    num_attention_heads = 32,
                    num_key_value_heads = 8,
                    intermediate_size = 14336,
                    vocab_size = 128256,
                    tie_word_embeddings = False,
                ),
            ),
            patch(
                "utils.hardware.hardware._determine_attention_impl_for_gpu_estimate",
                return_value = "eager",
            ),
            patch("utils.hardware.hardware.get_visible_gpu_count", return_value = 1),
        ):
            _, metadata = estimate_required_model_memory_gb(
                "unsloth/test",
                training_type = "LoRA/QLoRA",
                load_in_4bit = True,
            )

        self.assertEqual(metadata.get("estimation_mode"), "detailed")
        self.assertEqual(metadata.get("attention_implementation"), "eager")

    def test_training_estimate_falls_back_when_attention_resolution_fails(self):
        with (
            patch("utils.hardware.hardware.get_device", return_value = DeviceType.CUDA),
            patch(
                "utils.hardware.hardware.estimate_fp16_model_size_bytes",
                return_value = (8 * (1024**3), "config"),
            ),
            patch(
                "utils.hardware.hardware._resolve_model_identifier_for_gpu_estimate",
                return_value = "unsloth/test",
            ),
            patch(
                "utils.hardware.hardware._load_config_for_gpu_estimate",
                return_value = SimpleNamespace(
                    hidden_size = 4096,
                    num_hidden_layers = 32,
                    num_attention_heads = 32,
                    num_key_value_heads = 8,
                    intermediate_size = 14336,
                    vocab_size = 128256,
                    tie_word_embeddings = False,
                ),
            ),
            patch(
                "utils.hardware.hardware._determine_attention_impl_for_gpu_estimate",
                side_effect = RuntimeError("attention unavailable"),
            ),
            patch("utils.hardware.hardware.get_visible_gpu_count", return_value = 1),
        ):
            _, metadata = estimate_required_model_memory_gb(
                "unsloth/test",
                training_type = "LoRA/QLoRA",
                load_in_4bit = True,
            )

        self.assertEqual(metadata.get("estimation_mode"), "detailed")
        self.assertEqual(
            metadata.get("attention_implementation"),
            "eager",
        )

    def test_attention_resolver_does_not_mutate_loaded_config(self):
        from utils.hardware import hardware as hardware_module

        config = SimpleNamespace(
            hidden_size = 1024,
            num_hidden_layers = 2,
            num_attention_heads = 8,
            num_key_value_heads = 8,
            intermediate_size = 2048,
            vocab_size = 1024,
            tie_word_embeddings = True,
        )

        def _stub_resolver(model_class, cfg):
            cfg._attn_implementation = "eager"
            return "eager"

        with patch(
            "unsloth.models._utils.resolve_attention_implementation",
            side_effect = _stub_resolver,
        ):
            hardware_module._determine_attention_impl_for_gpu_estimate(config)

        self.assertFalse(hasattr(config, "_attn_implementation"))

    def test_attention_resolver_handles_missing_model_mapping(self):
        from utils.hardware import hardware as hardware_module

        config = SimpleNamespace(
            hidden_size = 1024,
            num_hidden_layers = 2,
            num_attention_heads = 8,
            num_key_value_heads = 8,
            intermediate_size = 2048,
            vocab_size = 1024,
            tie_word_embeddings = True,
        )
        captured = {}

        def _stub_resolver(model_class, cfg):
            captured["model_class"] = model_class
            return "eager"

        from transformers import AutoModel, AutoModelForCausalLM

        with (
            patch.object(AutoModelForCausalLM, "_model_mapping", new = None),
            patch.object(AutoModel, "_model_mapping", new = None),
            patch(
                "unsloth.models._utils.resolve_attention_implementation",
                side_effect = _stub_resolver,
            ),
        ):
            result = hardware_module._determine_attention_impl_for_gpu_estimate(config)

        self.assertEqual(result, "eager")
        self.assertIsNone(captured["model_class"])

    def test_attention_resolver_does_not_mutate_nested_text_config(self):
        from utils.hardware import hardware as hardware_module

        text_config = SimpleNamespace(
            hidden_size = 1024,
            num_hidden_layers = 2,
            num_attention_heads = 8,
            num_key_value_heads = 8,
            intermediate_size = 2048,
            vocab_size = 1024,
            tie_word_embeddings = True,
        )
        config = SimpleNamespace(
            hidden_size = 1024,
            num_hidden_layers = 2,
            num_attention_heads = 8,
            num_key_value_heads = 8,
            intermediate_size = 2048,
            vocab_size = 1024,
            tie_word_embeddings = True,
            text_config = text_config,
        )

        def _stub_resolver(model_class, cfg):
            cfg._attn_implementation = "eager"
            inner = getattr(cfg, "text_config", None)
            if inner is not None:
                inner._attn_implementation = "eager"
            return "eager"

        with patch(
            "unsloth.models._utils.resolve_attention_implementation",
            side_effect = _stub_resolver,
        ):
            hardware_module._determine_attention_impl_for_gpu_estimate(config)

        self.assertFalse(hasattr(config, "_attn_implementation"))
        self.assertFalse(hasattr(text_config, "_attn_implementation"))

    def test_min_per_gpu_generated_for_all_visible_counts(self):
        with (
            patch("utils.hardware.hardware.get_device", return_value = DeviceType.CUDA),
            patch(
                "utils.hardware.hardware.estimate_fp16_model_size_bytes",
                return_value = (8 * (1024**3), "config"),
            ),
            patch(
                "utils.hardware.hardware._resolve_model_identifier_for_gpu_estimate",
                return_value = "unsloth/test",
            ),
            patch(
                "utils.hardware.hardware._load_config_for_gpu_estimate",
                return_value = SimpleNamespace(
                    hidden_size = 4096,
                    num_hidden_layers = 32,
                    num_attention_heads = 32,
                    num_key_value_heads = 8,
                    intermediate_size = 14336,
                    vocab_size = 128256,
                    tie_word_embeddings = False,
                ),
            ),
            patch("utils.hardware.hardware.get_visible_gpu_count", return_value = 6),
        ):
            _, metadata = estimate_required_model_memory_gb(
                "unsloth/test",
                training_type = "LoRA/QLoRA",
                load_in_4bit = True,
            )

        self.assertEqual(metadata.get("estimation_mode"), "detailed")
        breakdown = metadata["vram_breakdown"]
        for n in range(1, 7):
            self.assertIn(f"min_per_gpu_{n}", breakdown)


class TestAutoSelectWithNoneRequired(_GpuCacheResetMixin, unittest.TestCase):
    def test_auto_select_falls_back_when_estimate_unavailable(self):
        with (
            patch("utils.hardware.hardware.get_device", return_value = DeviceType.CUDA),
            patch(
                "utils.hardware.hardware.estimate_required_model_memory_gb",
                return_value = (None, {"model_size_source": "unavailable"}),
            ),
            patch(
                "utils.hardware.hardware._get_parent_visible_gpu_spec",
                return_value = {
                    "raw": "0,1",
                    "numeric_ids": [0, 1],
                    "supports_explicit_gpu_ids": True,
                },
            ),
            patch(
                "utils.hardware.hardware.get_parent_visible_gpu_ids",
                return_value = [0, 1],
            ),
        ):
            selected, metadata = auto_select_gpu_ids("unsloth/test")

        self.assertEqual(selected, [0, 1])
        self.assertEqual(metadata["selection_mode"], "fallback_all")


class TestXpuRejection(_GpuCacheResetMixin, unittest.TestCase):
    def test_auto_select_returns_non_cuda_for_xpu(self):
        with patch("utils.hardware.hardware.get_device", return_value = DeviceType.XPU):
            selected, metadata = auto_select_gpu_ids("unsloth/test")

        self.assertIsNone(selected)
        self.assertEqual(metadata["selection_mode"], "non_cuda")

    def test_prepare_gpu_selection_rejects_explicit_ids_on_xpu(self):
        with patch("utils.hardware.hardware.get_device", return_value = DeviceType.XPU):
            with self.assertRaisesRegex(ValueError, "only supported on CUDA"):
                prepare_gpu_selection([0], model_name = "unsloth/test")


class TestEstimateFp16ModelSizeBytesPrefersLocalWeights(unittest.TestCase):
    def _run(
        self,
        model_path,
        *,
        config_bytes,
        local_bytes,
        safetensors_params = None,
        config = object(),
    ):
        from utils.hardware import hardware as hardware_module

        with (
            patch.object(
                hardware_module,
                "_resolve_model_identifier_for_gpu_estimate",
                return_value = model_path,
            ),
            patch.object(
                hardware_module,
                "_get_hf_safetensors_total_params",
                return_value = safetensors_params,
            ),
            patch.object(
                hardware_module,
                "_load_config_for_gpu_estimate",
                return_value = config,
            ),
            patch.object(
                hardware_module,
                "_estimate_fp16_model_size_bytes_from_config",
                return_value = config_bytes,
            ),
            patch.object(
                hardware_module,
                "_get_local_weight_size_bytes",
                return_value = local_bytes,
            ),
        ):
            return hardware_module.estimate_fp16_model_size_bytes(model_path)

    def test_local_weight_bytes_preferred_when_larger_than_config(self):
        bytes_, src = self._run(
            "/local/vlm",
            config_bytes = 2 * (1 << 30),
            local_bytes = 20 * (1 << 30),
        )
        self.assertEqual(bytes_, 20 * (1 << 30))
        self.assertEqual(src, "weight_bytes")

    def test_config_bytes_preferred_when_larger_than_local(self):
        bytes_, src = self._run(
            "/local/text-only",
            config_bytes = 20 * (1 << 30),
            local_bytes = 2 * (1 << 30),
        )
        self.assertEqual(bytes_, 20 * (1 << 30))
        self.assertEqual(src, "config")

    def test_config_bytes_returned_when_no_local_weights(self):
        bytes_, src = self._run(
            "/local/no-weights",
            config_bytes = 5 * (1 << 30),
            local_bytes = None,
        )
        self.assertEqual(bytes_, 5 * (1 << 30))
        self.assertEqual(src, "config")

    def test_local_bytes_returned_when_config_resolution_fails(self):
        bytes_, src = self._run(
            "/local/no-config",
            config_bytes = None,
            local_bytes = 7 * (1 << 30),
            config = None,
        )
        self.assertEqual(bytes_, 7 * (1 << 30))
        self.assertEqual(src, "weight_bytes")

    def test_equal_local_and_config_keeps_config_label(self):
        # why: tie-breaker is "local must be strictly larger" so an exact
        # match keeps the config-derived path.
        same = 8 * (1 << 30)
        bytes_, src = self._run(
            "/local/equal",
            config_bytes = same,
            local_bytes = same,
        )
        self.assertEqual(bytes_, same)
        self.assertEqual(src, "config")

    def test_remote_safetensors_path_unaffected_by_local_weights(self):
        from utils.hardware import hardware as hardware_module

        with (
            patch.object(
                hardware_module,
                "_resolve_model_identifier_for_gpu_estimate",
                return_value = "owner/repo",
            ),
            patch.object(
                hardware_module,
                "_get_hf_safetensors_total_params",
                return_value = 1_000_000_000,
            ),
            patch.object(
                hardware_module,
                "_load_config_for_gpu_estimate",
            ) as mock_load,
            patch.object(
                hardware_module,
                "_get_local_weight_size_bytes",
            ) as mock_local,
        ):
            bytes_, src = hardware_module.estimate_fp16_model_size_bytes("owner/repo")
            self.assertEqual(bytes_, 2 * 1_000_000_000)
            self.assertEqual(src, "safetensors")
            mock_load.assert_not_called()
            mock_local.assert_not_called()
