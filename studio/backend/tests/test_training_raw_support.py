# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

from datasets import Dataset

from core.training.training import TrainingBackend
from models.training import TrainingStartRequest
from utils.datasets import format_dataset, format_and_template_dataset
from utils.datasets.raw_text import prepare_raw_text_dataset

_BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _load_route_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, _BACKEND_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTrainingRawSupport(unittest.TestCase):
    def test_training_backend_preserves_cpt_4bit_and_embedding_lr(self):
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
                return_value = ([0], {"selection_mode": "auto"}),
            ),
            patch(
                "core.training.training._CTX.Queue",
                side_effect = [dummy_queue, dummy_queue],
            ),
            patch(
                "core.training.training._CTX.Process", return_value = DummyProcess()
            ) as mock_process,
            patch(
                "core.training.training.threading.Thread",
                return_value = DummyThread(),
            ),
        ):
            backend.start_training(
                job_id = "test-cpt-raw",
                model_name = "unsloth/test-bnb-4bit",
                training_type = "Continued Pretraining",
                format_type = "raw",
                load_in_4bit = True,
                embedding_learning_rate = 1e-5,
            )

        config = mock_process.call_args.kwargs["kwargs"]["config"]
        self.assertTrue(config["load_in_4bit"])
        self.assertEqual(config["embedding_learning_rate"], 1e-5)

    def test_training_route_forwards_embedding_learning_rate(self):
        training_route = _load_route_module(
            "training_route_module_raw_support",
            "routes/training.py",
        )
        captured: dict = {}

        class DummyBackend:
            current_job_id = None

            def is_training_active(self):
                return False

            def start_training(self, **kwargs):
                captured.update(kwargs)
                return True

        request = TrainingStartRequest(
            model_name = "unsloth/test-bnb-4bit",
            training_type = "Continued Pretraining",
            format_type = "raw",
            load_in_4bit = True,
            embedding_learning_rate = 1e-5,
        )

        with (
            patch.object(
                training_route,
                "get_training_backend",
                return_value = DummyBackend(),
            ),
            patch.object(training_route, "load_model_defaults", return_value = {}),
            patch(
                "core.inference.get_inference_backend",
                return_value = type(
                    "InferenceBackend",
                    (),
                    {"active_model_name": None},
                )(),
            ),
            patch(
                "core.export.get_export_backend",
                return_value = type(
                    "ExportBackend",
                    (),
                    {"current_checkpoint": None},
                )(),
            ),
        ):
            response = asyncio.run(
                training_route.start_training(request, current_subject = "test-user")
            )

        self.assertEqual(response.status, "queued")
        self.assertEqual(captured["embedding_learning_rate"], 1e-5)
        self.assertTrue(captured["load_in_4bit"])

    def test_format_dataset_supports_raw_text(self):
        dataset = Dataset.from_dict(
            {
                "body": ["hello", "world"],
                "title": ["a", "b"],
                "id": [1, 2],
            }
        )

        result = format_dataset(dataset, format_type = "raw")

        self.assertEqual(result["final_format"], "raw_text")
        self.assertIn("text", result["dataset"].column_names)
        self.assertEqual(result["dataset"][0]["text"], "hello")
        self.assertFalse(result["requires_manual_mapping"])

    def test_format_and_template_dataset_supports_raw_text_without_template(self):
        dataset = Dataset.from_dict({"body": ["hello raw world"]})

        result = format_and_template_dataset(
            dataset,
            model_name = "unsloth/test",
            tokenizer = None,
            format_type = "raw",
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["final_format"], "raw_text")
        self.assertEqual(result["dataset"][0]["text"], "hello raw world")

    def test_prepare_raw_text_dataset_drops_null_rows_before_appending_eos(self):
        dataset = Dataset.from_dict({"text": ["hello", None, "world"]})

        result = prepare_raw_text_dataset(
            dataset,
            mode_label = "CPT",
            split_name = "train",
            eos_token = "<eos>",
            append_eos = True,
        )

        self.assertEqual(len(result.dataset), 2)
        self.assertEqual(result.dataset[0]["text"], "hello<eos>")
        self.assertEqual(result.dataset[1]["text"], "world<eos>")
        self.assertTrue(
            any(
                "null or non-string 'text' values" in notice.message
                for notice in result.notices
            )
        )


if __name__ == "__main__":
    unittest.main()
