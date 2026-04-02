# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for llama.cpp server arg handling."""

from __future__ import annotations

import asyncio
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from models.inference import LoadRequest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _load_route_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, _BACKEND_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestLlamaCppServerArgs(unittest.TestCase):
    def test_user_args_override_builtin_flags(self):
        from core.inference.llama_cpp import LlamaCppBackend

        backend = LlamaCppBackend()
        cmd = ["llama-server", "--flash-attn", "on", "--parallel", "1"]
        backend._remove_conflicting_server_args(
            cmd,
            {
                "flash-attn": "off",
                "threads": 8,
                "port": 7777,
                "model": "blocked",
            },
        )
        cmd.extend(
            backend._build_server_args(
                {"flash-attn": "off", "threads": 8, "port": 7777, "model": "blocked"}
            )
        )

        self.assertNotIn("on", cmd)
        self.assertIn("--flash-attn", cmd)
        self.assertIn("off", cmd)
        self.assertIn("--threads", cmd)
        self.assertIn("8", cmd)
        self.assertNotIn("blocked", cmd)

    def test_build_server_args_blocks_dangerous_flags(self):
        from core.inference.llama_cpp import LlamaCppBackend

        backend = LlamaCppBackend()
        args = backend._build_server_args(
            {
                "host": "0.0.0.0",
                "api-key": "secret",
                "threads": 4,
                "flag": True,
                "disabled": False,
            }
        )

        self.assertNotIn("0.0.0.0", args)
        self.assertNotIn("secret", args)
        self.assertIn("--threads", args)
        self.assertIn("4", args)
        self.assertIn("--flag", args)
        self.assertNotIn("--disabled", args)


class TestServerArgsRoute(unittest.TestCase):
    def test_get_server_args_returns_400_when_not_loaded(self):
        inference_route = _load_route_module(
            "server_args_route_missing", "routes/inference.py"
        )

        with patch.object(
            inference_route,
            "get_llama_cpp_backend",
            return_value = SimpleNamespace(is_loaded = False),
        ):
            with self.assertRaises(HTTPException) as exc_info:
                asyncio.run(
                    inference_route.get_server_args(current_subject = "test-user")
                )

        self.assertEqual(exc_info.exception.status_code, 400)

    def test_get_server_args_returns_used_args_when_loaded(self):
        inference_route = _load_route_module(
            "server_args_route_loaded", "routes/inference.py"
        )

        with patch.object(
            inference_route,
            "get_llama_cpp_backend",
            return_value = SimpleNamespace(
                is_loaded = True, server_args_used = ["llama-server", "--threads", "8"]
            ),
        ):
            result = asyncio.run(
                inference_route.get_server_args(current_subject = "test-user")
            )

        self.assertEqual(result, {"server_args": ["llama-server", "--threads", "8"]})

    def test_load_route_merges_model_and_request_server_args(self):
        inference_route = _load_route_module(
            "server_args_route_merge", "routes/inference.py"
        )

        request = LoadRequest(
            model_path = "unsloth/test",
            server_args = {"flash-attn": "off", "threads": 8},
        )
        model_config = SimpleNamespace(
            is_gguf = True,
            gguf_hf_repo = None,
            gguf_file = "/tmp/test.gguf",
            gguf_mmproj_file = None,
            gguf_variant = None,
            identifier = "unsloth/test",
            display_name = "unsloth/test",
            is_vision = False,
        )

        class DummyLlamaBackend:
            is_loaded = False
            hf_variant = None
            model_identifier = None
            _is_vision = False
            _is_audio = False
            _audio_type = None
            context_length = 4096
            max_context_length = 4096
            supports_reasoning = False
            reasoning_always_on = False
            supports_tools = False
            cache_type_kv = None
            chat_template = None
            server_args_used = ["llama-server", "--flash-attn", "off", "--threads", "8"]

            def load_model(self, **kwargs):
                self.kwargs = kwargs
                return True

            def detect_audio_type(self):
                return None

            def unload_model(self):
                return True

        dummy_llama = DummyLlamaBackend()

        with (
            patch.object(
                inference_route.ModelConfig,
                "from_identifier",
                return_value = model_config,
            ),
            patch.object(
                inference_route,
                "get_llama_cpp_backend",
                return_value = dummy_llama,
            ),
            patch.object(
                inference_route,
                "get_inference_backend",
                return_value = SimpleNamespace(active_model_name = None),
            ),
            patch(
                "core.export.get_export_backend",
                return_value = SimpleNamespace(current_checkpoint = None),
            ),
            patch.object(inference_route, "load_inference_config", return_value = {}),
            patch.object(
                inference_route,
                "load_model_defaults",
                return_value = {"server_args": {"flash-attn": "on"}},
            ),
        ):
            result = asyncio.run(
                inference_route.load_model(request, current_subject = "test-user")
            )

        self.assertEqual(
            dummy_llama.kwargs["server_args"], {"flash-attn": "off", "threads": 8}
        )
        self.assertEqual(
            result.server_args_used,
            ["llama-server", "--flash-attn", "off", "--threads", "8"],
        )
