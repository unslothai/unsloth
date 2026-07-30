# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/api/inference/validate must report an INCONCLUSIVE diffusion check as such.

`_classify_diffusion_gguf` is a tri-state: True (diffusion), False (header read,
ordinary), None (nothing to read, and no family in the name).

The response used to collapse None into `is_diffusion = False`, so a caller could not
tell "ordinary GGUF" from "unknown". The staged-metadata preflight picks a GPU-layer
split from that answer, and /load may then apply it to a diffusion runner: an inherited
0 CPU-masks it, another count repartitions or OOMs it.
"""

import asyncio
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from models.inference import ValidateModelRequest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _load_route_module(name: str):
    spec = importlib.util.spec_from_file_location(name, _BACKEND_ROOT / "routes/inference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def _noop_gpu_ids(_config, gpu_ids, **_kwargs):
    return gpu_ids, False


class TestValidateReportsDiffusionUnknown(unittest.TestCase):
    def _validate(
        self,
        route,
        *,
        diffusion_kind,
        is_gguf = True,
    ):
        # Mirrors the real staged-metadata preflight; also skips the training guard.
        request = ValidateModelRequest(
            model_path = "someone/repacked-gguf",
            include_context_length = True,
        )
        config = SimpleNamespace(
            identifier = "someone/repacked-gguf",
            display_name = "repacked-gguf",
            is_gguf = is_gguf,
            is_lora = False,
            is_vision = False,
            gguf_file = None,
        )
        with (
            patch.object(
                route,
                "_resolve_model_identifier_for_request",
                return_value = ("someone/repacked-gguf", "someone/repacked-gguf", False),
            ),
            patch.object(route.ModelConfig, "from_identifier", return_value = config),
            patch.object(route, "_resolve_inherited_extra_args", return_value = None),
            patch.object(route, "_classify_diffusion_gguf", return_value = diffusion_kind),
            patch.object(route, "_resolve_gguf_gpu_ids_for_request", new = _noop_gpu_ids),
            patch.object(route, "_effective_load_in_4bit", return_value = True),
        ):
            return asyncio.run(route.validate_model(request, current_subject = "test-user"))

    def test_unclassifiable_gguf_is_reported_unknown_not_ordinary(self):
        """The bug: None must not look identical to a confirmed ordinary GGUF."""
        route = _load_route_module("inf_route_diffusion_unknown_1")
        resp = self._validate(route, diffusion_kind = None)
        self.assertFalse(resp.is_diffusion)
        self.assertTrue(
            resp.diffusion_unknown,
            "an unreadable/undownloaded GGUF with no family in its name is UNKNOWN; "
            "reporting it as a plain non-diffusion GGUF lets a caller inherit a "
            "GPU-layer split that /load will apply to a diffusion runner",
        )

    def test_confirmed_ordinary_gguf_is_not_unknown(self):
        route = _load_route_module("inf_route_diffusion_unknown_2")
        resp = self._validate(route, diffusion_kind = False)
        self.assertFalse(resp.is_diffusion)
        self.assertFalse(resp.diffusion_unknown)

    def test_confirmed_diffusion_gguf_is_not_unknown(self):
        route = _load_route_module("inf_route_diffusion_unknown_3")
        resp = self._validate(route, diffusion_kind = True)
        self.assertTrue(resp.is_diffusion)
        self.assertFalse(resp.diffusion_unknown)

    def test_non_gguf_is_never_unknown(self):
        """A transformers model is definitively not a diffusion GGUF."""
        route = _load_route_module("inf_route_diffusion_unknown_4")
        resp = self._validate(route, diffusion_kind = False, is_gguf = False)
        self.assertFalse(resp.is_diffusion)
        self.assertFalse(resp.diffusion_unknown)

    def test_flag_defaults_off_so_an_old_client_reads_the_same_response(self):
        """Additive field: absent/False keeps the pre-#7575 meaning of is_diffusion."""
        from models.inference import ValidateModelResponse

        resp = ValidateModelResponse(valid = True, message = "ok")
        self.assertFalse(resp.diffusion_unknown)


if __name__ == "__main__":
    unittest.main()
