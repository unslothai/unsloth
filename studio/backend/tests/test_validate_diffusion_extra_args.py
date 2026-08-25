# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/api/inference/validate must ignore pass-through arguments for a diffusion GGUF.

/load already drops them: the visual runner builds its own command and appends none of
them. /validate is the call that approves the load, and it reads a --ctx-size out of the
same list to size the estimate, so leaving them in place approves a load against a
command that will never carry them. The caller cannot decide this itself either, since
its staged metadata is inconclusive for a GGUF it has not finished downloading, which is
why the drop belongs after the authoritative classification rather than before it.
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


class TestValidateDropsDiffusionExtraArgs(unittest.TestCase):
    def _validate(self, route, *, diffusion_kind):
        seen: list = []

        def _capture(*_args, **kwargs):
            seen.append(kwargs.get("llama_extra_args"))

        request = ValidateModelRequest(
            model_path = "someone/diffusion-gguf",
            llama_extra_args = ["--ctx-size", "8192"],
        )
        config = SimpleNamespace(
            identifier = "someone/diffusion-gguf",
            display_name = "diffusion-gguf",
            is_gguf = True,
            is_lora = False,
            is_vision = False,
            gguf_file = None,
        )
        with (
            patch.object(
                route,
                "_resolve_model_identifier_for_request",
                return_value = ("someone/diffusion-gguf", "someone/diffusion-gguf", False),
            ),
            patch.object(route.ModelConfig, "from_identifier", return_value = config),
            patch.object(
                route,
                "_resolve_inherited_extra_args",
                return_value = ["--ctx-size", "8192"],
            ),
            patch.object(route, "_classify_diffusion_gguf", return_value = diffusion_kind),
            patch.object(route, "_resolve_gguf_gpu_ids_for_request", new = _noop_gpu_ids),
            patch.object(route, "_effective_load_in_4bit", return_value = True),
            patch.object(route, "_guard_chat_load_against_training", new = _capture),
        ):
            asyncio.run(route.validate_model(request, current_subject = "test-user"))
        return seen

    def test_a_diffusion_gguf_is_estimated_without_them(self):
        route = _load_route_module("inf_route_diffusion_extra_args_1")
        self.assertEqual(self._validate(route, diffusion_kind = True), [[]])

    def test_an_ordinary_gguf_still_estimates_with_them(self):
        # The drop is narrow on purpose: this is the path the editor exists for, and
        # a --ctx-size here has to reach the estimate that approves the load.
        route = _load_route_module("inf_route_diffusion_extra_args_2")
        self.assertEqual(
            self._validate(route, diffusion_kind = False),
            [["--ctx-size", "8192"]],
        )

    def test_an_inconclusive_gguf_keeps_them(self):
        # None is "nothing to read yet", not "diffusion". Dropping on it would strip a
        # working override from an ordinary model whose header has not arrived.
        route = _load_route_module("inf_route_diffusion_extra_args_3")
        self.assertEqual(
            self._validate(route, diffusion_kind = None),
            [["--ctx-size", "8192"]],
        )


class TestValidateJudgesTheListBeforeRewritingIt(unittest.TestCase):
    """The manual translation reads -ngl out of the extras and strips it, so it has to
    run AFTER the list has been validated: otherwise a spelling /load refuses is
    parsed and removed before validation sees it, and the switch is approved for a
    load that answers 400."""

    def _validate(
        self,
        route,
        *,
        extra_args,
        manual = True,
    ):
        request = ValidateModelRequest(
            model_path = "someone/gguf",
            llama_extra_args = extra_args,
            **({"gpu_memory_mode": "manual", "gpu_layers": 0} if manual else {}),
        )
        config = SimpleNamespace(
            identifier = "someone/gguf",
            display_name = "gguf",
            is_gguf = True,
            is_lora = False,
            is_vision = False,
            gguf_file = None,
        )
        with (
            patch.object(
                route,
                "_resolve_model_identifier_for_request",
                return_value = ("someone/gguf", "someone/gguf", False),
            ),
            patch.object(route.ModelConfig, "from_identifier", return_value = config),
            patch.object(route, "_resolve_inherited_extra_args", return_value = list(extra_args)),
            patch.object(route, "_classify_diffusion_gguf", return_value = False),
            patch.object(route, "_resolve_gguf_gpu_ids_for_request", new = _noop_gpu_ids),
            patch.object(route, "_effective_load_in_4bit", return_value = True),
            patch.object(route, "_guard_chat_load_against_training", new = lambda *a, **k: None),
        ):
            return asyncio.run(route.validate_model(request, current_subject = "test-user"))

    def test_an_attached_offload_spelling_is_refused_not_translated(self):
        # llama.cpp looks the whole token up in its option map, so "--gpu-layers=20"
        # is an argument it has never heard of; /load refuses the list. Translating
        # first read the 20, stripped the token, and approved the switch.
        from fastapi import HTTPException

        route = _load_route_module("inf_route_validate_order_1")
        with self.assertRaises(HTTPException) as caught:
            self._validate(route, extra_args = ["--gpu-layers=20"])
        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn("two separate arguments", str(caught.exception.detail))

    def test_a_malformed_layer_count_is_a_refusal_not_a_crash(self):
        # parse_gpu_layers_override raises on a non-integer, and it used to run
        # before the try that turns a bad list into a 400, so this was a 500.
        from fastapi import HTTPException

        route = _load_route_module("inf_route_validate_order_2")
        with self.assertRaises(HTTPException) as caught:
            self._validate(route, extra_args = ["-ngl", "bad"])
        self.assertEqual(caught.exception.status_code, 400)

    def test_a_well_formed_list_still_passes(self):
        route = _load_route_module("inf_route_validate_order_3")
        response = self._validate(route, extra_args = ["-ngl", "20"])
        self.assertTrue(getattr(response, "valid", True))


class TestValidateTranslatesManualNgl(unittest.TestCase):
    """Manual GPU memory owns the offload flags, and /load turns an explicit -ngl into
    the first-class field before stripping them. /validate has to do the same, or the
    call that APPROVES the switch is judging a different command than the one that runs:
    gpu_layers 0 with "-ngl 20" was approved as a load that places nothing on any device
    and cannot compete with training for VRAM, and then launched twenty layers on it."""

    def _validate(
        self,
        route,
        *,
        gpu_layers,
        extra_args,
        diffusion_kind = True,
    ):
        seen: list = []

        def _capture(_config, request, **kwargs):
            seen.append((request.gpu_layers, kwargs.get("llama_extra_args")))

        request = ValidateModelRequest(
            model_path = "someone/diffusion-gguf",
            llama_extra_args = extra_args,
            gpu_memory_mode = "manual",
            gpu_layers = gpu_layers,
        )
        config = SimpleNamespace(
            identifier = "someone/diffusion-gguf",
            display_name = "diffusion-gguf",
            is_gguf = True,
            is_lora = False,
            is_vision = False,
            gguf_file = None,
        )
        with (
            patch.object(
                route,
                "_resolve_model_identifier_for_request",
                return_value = ("someone/diffusion-gguf", "someone/diffusion-gguf", False),
            ),
            patch.object(route.ModelConfig, "from_identifier", return_value = config),
            patch.object(route, "_resolve_inherited_extra_args", return_value = list(extra_args)),
            patch.object(route, "_classify_diffusion_gguf", return_value = diffusion_kind),
            patch.object(route, "_resolve_gguf_gpu_ids_for_request", new = _noop_gpu_ids),
            patch.object(route, "_effective_load_in_4bit", return_value = True),
            patch.object(route, "_guard_chat_load_against_training", new = _capture),
        ):
            asyncio.run(route.validate_model(request, current_subject = "test-user"))
        return seen

    def test_an_explicit_layer_count_reaches_the_guard(self):
        route = _load_route_module("inf_route_manual_ngl_1")
        seen = self._validate(route, gpu_layers = 0, extra_args = ["-ngl", "20"])
        # The layer count the load will really run, and the raw flag stripped out of
        # the list exactly as /load strips it once it owns the field.
        self.assertEqual(seen, [(20, [])])

    def test_a_zero_layer_override_is_read_the_same_way(self):
        # The inverse pairing: asked for 20, overridden to 0. Judged as the CPU-only
        # load it is, rather than refused for VRAM it never takes.
        route = _load_route_module("inf_route_manual_ngl_2")
        seen = self._validate(route, gpu_layers = 20, extra_args = ["-ngl", "0"])
        self.assertEqual(seen, [(0, [])])

    def test_auto_mode_leaves_the_flag_alone(self):
        # Only manual mode owns these. In Auto the flag is a pass-through the loader
        # honours, so translating it here would invent a first-class value /load never set.
        route = _load_route_module("inf_route_manual_ngl_3")
        seen: list = []

        def _capture(_config, request, **kwargs):
            seen.append((request.gpu_layers, kwargs.get("llama_extra_args")))

        request = ValidateModelRequest(
            model_path = "someone/gguf",
            llama_extra_args = ["-ngl", "20"],
        )
        config = SimpleNamespace(
            identifier = "someone/gguf",
            display_name = "gguf",
            is_gguf = True,
            is_lora = False,
            is_vision = False,
            gguf_file = None,
        )
        with (
            patch.object(
                route,
                "_resolve_model_identifier_for_request",
                return_value = ("someone/gguf", "someone/gguf", False),
            ),
            patch.object(route.ModelConfig, "from_identifier", return_value = config),
            patch.object(route, "_resolve_inherited_extra_args", return_value = ["-ngl", "20"]),
            patch.object(route, "_classify_diffusion_gguf", return_value = False),
            patch.object(route, "_resolve_gguf_gpu_ids_for_request", new = _noop_gpu_ids),
            patch.object(route, "_effective_load_in_4bit", return_value = True),
            patch.object(route, "_guard_chat_load_against_training", new = _capture),
        ):
            asyncio.run(route.validate_model(request, current_subject = "test-user"))
        self.assertEqual(seen, [(request.gpu_layers, ["-ngl", "20"])])


if __name__ == "__main__":
    unittest.main()


class TestValidateRefusesWhatLoadWouldRefuse(unittest.TestCase):
    """The picker unloads the running model once /validate approves the switch, so a
    list /load would answer 400 on has to be refused here instead: a refusal leaves
    the current model alone, a failed switch does not."""

    def _validate(
        self,
        route,
        *,
        extra_args,
        n_parallel = None,
        diffusion_kind = False,
    ):
        request = ValidateModelRequest(
            model_path = "someone/gguf",
            llama_extra_args = extra_args,
            n_parallel = n_parallel,
        )
        config = SimpleNamespace(
            identifier = "someone/gguf",
            display_name = "gguf",
            is_gguf = True,
            is_lora = False,
            is_vision = False,
            gguf_file = None,
        )
        with (
            patch.object(
                route,
                "_resolve_model_identifier_for_request",
                return_value = ("someone/gguf", "someone/gguf", False),
            ),
            patch.object(route.ModelConfig, "from_identifier", return_value = config),
            patch.object(route, "_resolve_inherited_extra_args", return_value = extra_args),
            patch.object(route, "_classify_diffusion_gguf", return_value = diffusion_kind),
            patch.object(route, "_resolve_gguf_gpu_ids_for_request", new = _noop_gpu_ids),
            patch.object(route, "_effective_load_in_4bit", return_value = True),
            patch.object(route, "_effective_parallel_slots", side_effect = lambda n, **_: n),
            patch.object(route, "_guard_chat_load_against_training", new = lambda *a, **k: None),
        ):
            return asyncio.run(route.validate_model(request, current_subject = "test-user"))

    def test_a_denied_flag_is_refused_before_the_switch(self):
        route = _load_route_module("inf_route_validate_denies_1")
        with self.assertRaises(Exception) as caught:
            self._validate(route, extra_args = ["--agent"])
        self.assertEqual(getattr(caught.exception, "status_code", None), 400)
        self.assertIn("managed by Unsloth Studio", str(caught.exception.detail))

    def test_a_batch_below_the_slot_floor_is_refused_before_the_switch(self):
        route = _load_route_module("inf_route_validate_denies_2")
        with self.assertRaises(Exception) as caught:
            self._validate(route, extra_args = ["-b", "2"], n_parallel = 4)
        self.assertEqual(getattr(caught.exception, "status_code", None), 400)
        self.assertIn("aborts on --batch-size", str(caught.exception.detail))

    def test_a_list_the_load_would_accept_still_passes(self):
        route = _load_route_module("inf_route_validate_denies_3")
        resp = self._validate(route, extra_args = ["--numa", "distribute"], n_parallel = 4)
        self.assertTrue(resp.is_gguf)


class TestEmbeddingSlotClampInTheBatchFloor(unittest.TestCase):
    """--embedding caps the batch at the micro-batch and llama-server aborts when that
    is below the slot count, so load_model reduces the slots to it before launching.
    A floor sized from the pre-clamp count refuses a command the launcher would run."""

    def _clamped(
        self,
        route,
        *,
        is_embedding,
        extra_args,
        slots = 4,
        **kwargs,
    ):
        config = SimpleNamespace(identifier = "someone/embed-gguf", gguf_file = None)
        with patch.object(route, "_is_embedding_gguf", return_value = is_embedding):
            return route._embedding_clamped_slots(
                config,
                slots,
                extra_args = extra_args,
                n_batch = kwargs.get("n_batch"),
                n_ubatch = kwargs.get("n_ubatch"),
                n_ctx = kwargs.get("n_ctx"),
            )

    def test_the_slots_follow_the_micro_batch_down(self):
        route = _load_route_module("inf_route_embed_clamp_1")
        self.assertEqual(
            self._clamped(route, is_embedding = True, extra_args = ["-b", "2", "-ub", "2"]),
            2,
        )

    def test_a_chat_gguf_keeps_the_slots_it_asked_for(self):
        route = _load_route_module("inf_route_embed_clamp_2")
        self.assertEqual(
            self._clamped(route, is_embedding = False, extra_args = ["-b", "2", "-ub", "2"]),
            4,
        )

    def test_defaults_clamp_nothing(self):
        # Nothing overrides the batch, so the launch runs llama.cpp's own 2048 and the
        # micro-batch is nowhere near the slot count.
        route = _load_route_module("inf_route_embed_clamp_3")
        self.assertEqual(
            self._clamped(route, is_embedding = True, extra_args = ["--numa", "distribute"]),
            4,
        )

    def test_the_clamp_floors_at_one_slot(self):
        # "-b 0" resolves to a zero micro-batch, and --parallel 0 is rejected at arg
        # parse, which is the floor load_model applies too.
        route = _load_route_module("inf_route_embed_clamp_4")
        self.assertEqual(
            self._clamped(route, is_embedding = True, extra_args = ["-b", "0", "-ub", "0"]),
            1,
        )

    def test_an_unreadable_header_leaves_the_refusal_alone(self):
        # _is_embedding_gguf answers False for a GGUF that is not on this disk yet, so
        # nothing is relaxed on a guess.
        route = _load_route_module("inf_route_embed_clamp_5")
        config = SimpleNamespace(identifier = "someone/gguf", gguf_file = None, gguf_hf_repo = None)
        self.assertFalse(route._is_embedding_gguf(config))

    def test_an_uncached_embedding_identifier_stays_fail_closed(self):
        route = _load_route_module("inf_route_embed_clamp_6")
        config = SimpleNamespace(
            identifier = "Qwen/Qwen3-Embedding-4B-GGUF",
            gguf_file = None,
            gguf_hf_repo = "Qwen/Qwen3-Embedding-4B-GGUF",
            gguf_variant = "Q4_K_M",
        )
        with patch.object(route, "_local_gguf_main_path", return_value = None):
            self.assertFalse(route._is_embedding_gguf(config))


class TestValidateAllowsTheEmbeddingClampedBatch(TestValidateRefusesWhatLoadWouldRefuse):
    """The preflight has to allow exactly what the load allows, or the picker refuses a
    switch the load it gates would have completed."""

    def test_an_embedding_gguf_may_batch_at_its_micro_batch(self):
        route = _load_route_module("inf_route_validate_embed_1")
        with patch.object(route, "_is_embedding_gguf", return_value = True):
            resp = self._validate(
                route,
                extra_args = ["-b", "2", "-ub", "2"],
                n_parallel = 4,
            )
        self.assertTrue(resp.is_gguf)

    def test_a_chat_gguf_is_still_refused(self):
        route = _load_route_module("inf_route_validate_embed_2")
        with (
            patch.object(route, "_is_embedding_gguf", return_value = False),
            self.assertRaises(Exception) as caught,
        ):
            self._validate(route, extra_args = ["-b", "2", "-ub", "2"], n_parallel = 4)
        self.assertEqual(getattr(caught.exception, "status_code", None), 400)
        self.assertIn("aborts on --batch-size", str(caught.exception.detail))

    def test_an_embedding_gguf_below_its_own_floor_is_still_refused(self):
        # The clamp floors at one slot, and llama-server aborts on a batch of 1 at any
        # slot count, so this is not a refusal the clamp may lift.
        route = _load_route_module("inf_route_validate_embed_3")
        with (
            patch.object(route, "_is_embedding_gguf", return_value = True),
            self.assertRaises(Exception) as caught,
        ):
            self._validate(route, extra_args = ["-b", "1", "-ub", "1"], n_parallel = 4)
        self.assertEqual(getattr(caught.exception, "status_code", None), 400)
