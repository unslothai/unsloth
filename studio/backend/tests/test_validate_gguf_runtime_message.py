# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/api/inference/validate and /load must surface an actionable "install the runtime"
message when a GGUF model's llama-server is missing, not a generic error."""

import asyncio
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

from core.inference.llama_cpp import LlamaServerNotFoundError
from models.inference import LoadRequest, ValidateModelRequest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _load_route_module(name: str, relative_path: str):
    # Load routes/inference.py under a standalone name (mirrors test_gpu_selection).
    spec = importlib.util.spec_from_file_location(name, _BACKEND_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_GGUF_MSG = (
    "This is a GGUF model, but the llama.cpp runtime (llama-server) is not "
    "installed. Run `unsloth studio setup` to download the prebuilt runtime, "
    "then try again. (Advanced: set LLAMA_SERVER_PATH to an existing binary.)"
)


class TestValidateGgufRuntimeMessage(unittest.TestCase):
    def _validate(self, route, model_path, side_effect):
        request = ValidateModelRequest(model_path = model_path)
        with (
            patch.object(
                route,
                "_resolve_model_identifier_for_request",
                return_value = (model_path, model_path, False),
            ),
            patch.object(route.ModelConfig, "from_identifier", side_effect = side_effect),
        ):
            with self.assertRaises(HTTPException) as exc:
                asyncio.run(route.validate_model(request, current_subject = "test-user"))
        return exc.exception

    def test_missing_llama_server_returns_actionable_message(self):
        route = _load_route_module("inf_route_runtime_msg_1", "routes/inference.py")
        err = self._validate(route, "unsloth/Qwen3-1.7B-GGUF", LlamaServerNotFoundError(_GGUF_MSG))
        self.assertEqual(err.status_code, 400)
        self.assertIn("unsloth studio setup", err.detail)
        self.assertIn("llama.cpp runtime", err.detail)
        self.assertNotEqual(err.detail, "Invalid model")

    def test_other_errors_still_generic_invalid_model(self):
        route = _load_route_module("inf_route_runtime_msg_2", "routes/inference.py")
        err = self._validate(route, "not/a-real-model", RuntimeError("totally different failure"))
        self.assertEqual(err.status_code, 400)
        self.assertEqual(err.detail, "Invalid model")


class TestLoadGgufRuntimeMessage(unittest.TestCase):
    """/api/inference/load surfaces the same message (not a 500) when the runtime is missing."""

    def _load(self, route, model_path, side_effect):
        request = LoadRequest(model_path = model_path)
        backend = MagicMock(active_model_name = None)  # no resident model -> reach from_identifier
        with (
            patch.object(
                route,
                "_resolve_model_identifier_for_request",
                return_value = (model_path, model_path, False),
            ),
            patch.object(route, "resolve_effective_chat_template_override", return_value = None),
            patch.object(route, "get_inference_backend", return_value = backend),
            patch.object(route, "get_llama_cpp_backend", return_value = MagicMock()),
            patch.object(route.ModelConfig, "from_identifier", side_effect = side_effect),
        ):
            with self.assertRaises(HTTPException) as exc:
                asyncio.run(route.load_model(request, MagicMock(), current_subject = "test-user"))
        return exc.exception

    def test_missing_llama_server_returns_actionable_message(self):
        route = _load_route_module("inf_route_load_runtime_msg_1", "routes/inference.py")
        err = self._load(route, "unsloth/Qwen3-1.7B-GGUF", LlamaServerNotFoundError(_GGUF_MSG))
        self.assertEqual(err.status_code, 400)
        self.assertIn("unsloth studio setup", err.detail)
        self.assertIn("llama.cpp runtime", err.detail)

    def test_other_load_errors_still_500(self):
        route = _load_route_module("inf_route_load_runtime_msg_2", "routes/inference.py")
        err = self._load(route, "unsloth/some-model", RuntimeError("totally different failure"))
        self.assertEqual(err.status_code, 500)


if __name__ == "__main__":
    unittest.main()
