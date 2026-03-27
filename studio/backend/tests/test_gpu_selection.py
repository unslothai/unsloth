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
    DeviceType,
    get_parent_visible_gpu_ids,
    get_visible_gpu_utilization,
    resolve_requested_gpu_ids,
)

_BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _load_route_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, _BACKEND_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestResolveRequestedGpuIds(unittest.TestCase):
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

    def test_invalid_requests_raise_clear_value_errors(self):
        cases = [
            ([], "non-empty list"),
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


class TestVisibleGpuUtilization(unittest.TestCase):
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
            patch(
                "subprocess.run",
            ) as mock_run,
        ):
            mock_run.return_value = SimpleNamespace(
                returncode = 0,
                stdout = smi_output,
            )
            result = get_visible_gpu_utilization()

        self.assertTrue(result["available"])
        self.assertEqual(result["parent_visible_gpu_ids"], [1, 3])
        self.assertEqual([device["index"] for device in result["devices"]], [1, 3])
        self.assertEqual(result["devices"][0]["gpu_utilization_pct"], 20.0)
        self.assertEqual(result["devices"][1]["power_utilization_pct"], 50.0)


class TestPreSpawnGpuResolution(unittest.TestCase):
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
                "core.training.training.resolve_requested_gpu_ids", return_value = [1, 2]
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

    def test_training_backend_preserves_none_for_inherited_visibility(self):
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
                "core.training.training.resolve_requested_gpu_ids", return_value = [0, 1]
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
        self.assertIsNone(config["resolved_gpu_ids"])

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
                "core.inference.orchestrator.resolve_requested_gpu_ids",
                return_value = [1],
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


class TestRouteErrors(unittest.TestCase):
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
                    inference_route.load_model(request, current_subject = "test-user")
                )

        self.assertEqual(exc_info.exception.status_code, 400)
        self.assertIn("gpu_ids [99]", exc_info.exception.detail)
