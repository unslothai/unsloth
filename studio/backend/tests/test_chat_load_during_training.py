# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Loading a NEW chat model while training runs: can_load_chat_during_training
(VRAM fit check), _guard_chat_load_against_training and _effective_load_in_4bit
(409 + sizing wiring). The guard sizes the same effective load the backend will
perform (HF auto reuses the loader's selector, HF explicit applies a per-GPU
floor, GGUF sizes from on-disk weights, LoRA 4-bit->16-bit flips resolved first)
and leaves non-training/external loads untouched."""

import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from utils.hardware import DeviceType
import utils.hardware.hardware as _hw_module

_BACKEND_ROOT = Path(__file__).resolve().parent.parent

# Load training_vram.py standalone (avoids the heavy routes/__init__.py); its
# lazy hardware imports still resolve against the patched utils.hardware names.
_spec = importlib.util.spec_from_file_location(
    "training_vram_load_test", _BACKEND_ROOT / "routes" / "training_vram.py"
)
tv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tv)


class _GpuCacheResetMixin:
    def tearDown(self):
        _hw_module._physical_gpu_count = None
        _hw_module._visible_gpu_count = None


def _devices(*free_specs):
    """Build a device list from (index, total, used) tuples."""
    return [
        {"index": i, "vram_total_gb": total, "vram_used_gb": used}
        for (i, total, used) in free_specs
    ]


# ── can_load_chat_during_training: HF auto (reuses auto_select_gpu_ids) ───────


class TestCanLoadAutoHF(_GpuCacheResetMixin, unittest.TestCase):
    def _run(self, *, selection_mode, required, usable):
        meta = {"selection_mode": selection_mode, "required_gb": required, "usable_gb": usable}
        with (
            patch("utils.hardware.get_device", return_value = DeviceType.CUDA),
            patch("utils.hardware.auto_select_gpu_ids", return_value = ([0], meta)) as auto_mock,
        ):
            ok, info = tv.can_load_chat_during_training(
                model_name = "unsloth/Qwen3-1.7B",
                hf_token = None,
                load_in_4bit = True,
                max_seq_length = 0,
                requested_gpu_ids = None,
                is_gguf = False,
            )
        return ok, info, auto_mock

    def test_fits_with_margin(self):
        # free 60 >= 8*1.15+4 = 13.2
        ok, info, auto_mock = self._run(selection_mode = "auto", required = 8.0, usable = 60.0)
        self.assertTrue(ok)
        self.assertEqual(info["mode"], "auto")
        self.assertAlmostEqual(info["needed_gb"], 13.2, places = 3)
        auto_mock.assert_called_once()  # mirrors the loader's own selection

    def test_too_tight_refuses(self):
        # free 10 < 8*1.15+4 = 13.2 -> refuse even though raw 10 > 8
        ok, _, _ = self._run(selection_mode = "auto", required = 8.0, usable = 10.0)
        self.assertFalse(ok)

    def test_fallback_all_refuses(self):
        # Selector couldn't confirm placement -> default-deny to protect training.
        ok, info = self._run(selection_mode = "fallback_all", required = 8.0, usable = 999.0)[:2]
        self.assertFalse(ok)


# ── can_load_chat_during_training: HF explicit (per-GPU floor) ────────────────


class TestCanLoadExplicitHF(_GpuCacheResetMixin, unittest.TestCase):
    def _run(
        self,
        *,
        required,
        devices,
        gpu_ids,
        resolved = None,
        resolve_side_effect = None,
    ):
        resolve_kwargs = (
            {"side_effect": resolve_side_effect}
            if resolve_side_effect
            else {"return_value": resolved if resolved is not None else gpu_ids}
        )
        with (
            patch("utils.hardware.get_device", return_value = DeviceType.CUDA),
            patch("utils.hardware.estimate_required_model_memory_gb", return_value = (required, {})),
            patch("utils.hardware.get_visible_gpu_utilization", return_value = {"devices": devices}),
            patch("utils.hardware.resolve_requested_gpu_ids", **resolve_kwargs),
            patch("utils.hardware.auto_select_gpu_ids") as auto_mock,
        ):
            ok, info = tv.can_load_chat_during_training(
                model_name = "m",
                hf_token = None,
                load_in_4bit = True,
                max_seq_length = 0,
                requested_gpu_ids = gpu_ids,
                is_gguf = False,
            )
        return ok, info, auto_mock

    def test_single_gpu_fits(self):
        ok, info, auto_mock = self._run(required = 8.0, devices = _devices((0, 80, 20)), gpu_ids = [0])
        self.assertTrue(ok)
        self.assertEqual(info["mode"], "explicit")
        auto_mock.assert_not_called()  # explicit never calls the auto selector

    def test_per_gpu_floor_blocks_uneven_split(self):
        # free [45, 10]; aggregate 45 + 10*0.85 = 53.5 >= needed 27, but the 10 GB
        # GPU is below the even-share floor 27/2 = 13.5 -> refuse (would OOM it).
        ok, info, _ = self._run(
            required = 20.0, devices = _devices((0, 80, 35), (1, 80, 70)), gpu_ids = [0, 1]
        )
        self.assertFalse(ok)
        self.assertAlmostEqual(info["min_free_gb"], 10.0, places = 3)

    def test_per_gpu_floor_passes_when_even(self):
        # free [30, 30]; both clear the 13.5 even-share floor -> allow.
        ok, _, _ = self._run(
            required = 20.0, devices = _devices((0, 80, 50), (1, 80, 50)), gpu_ids = [0, 1]
        )
        self.assertTrue(ok)

    def test_missing_gpu_counts_as_zero(self):
        ok, _, _ = self._run(required = 5.0, devices = _devices((0, 80, 5)), gpu_ids = [3], resolved = [3])
        self.assertFalse(ok)

    def test_invalid_ids_does_not_block(self):
        ok, info, _ = self._run(
            required = 5.0,
            devices = [],
            gpu_ids = [99],
            resolve_side_effect = ValueError("Invalid gpu_ids [99]"),
        )
        self.assertTrue(ok)
        self.assertEqual(info["reason"], "invalid_gpu_ids")


# ── can_load_chat_during_training: GGUF (sized from on-disk weights) ──────────


class TestCanLoadGGUF(_GpuCacheResetMixin, unittest.TestCase):
    # Conservative upper bound: main_gb (weights + KV, offload-scaled)
    # DISTRIBUTES across the allowed GPUs; companion_gb lands whole on ONE
    # device (min-free check); single_device (diffusion / manual split) needs
    # the whole footprint on one device. SAFETY_MARGIN 1.15, KEEP_FLOOR 4.0,
    # multi-GPU aggregate = best + 0.85 * rest.
    def _run(
        self,
        *,
        devices,
        main_gb = None,
        companion_gb = 0.0,
        single_device = False,
        split_max_share = None,
        gpu_ids = None,
    ):
        with (
            patch("utils.hardware.get_device", return_value = DeviceType.CUDA),
            patch("utils.hardware.get_visible_gpu_utilization", return_value = {"devices": devices}),
            patch("utils.hardware.resolve_requested_gpu_ids", return_value = list(gpu_ids or [])),
            patch("utils.hardware.auto_select_gpu_ids") as auto_mock,
        ):
            ok, info = tv.can_load_chat_during_training(
                model_name = "unsloth/gemma-GGUF",
                hf_token = None,
                load_in_4bit = True,
                max_seq_length = 0,
                requested_gpu_ids = gpu_ids,
                is_gguf = True,
                gguf_main_gb = main_gb,
                gguf_companion_gb = companion_gb,
                gguf_single_device = single_device,
                gguf_split_max_share = split_max_share,
            )
        return ok, info, auto_mock

    def test_main_fits_aggregate(self):
        # free [45, 10]; main 20 -> needed 27, aggregate 45 + 10*0.85 = 53.5 -> fit.
        ok, info, auto_mock = self._run(devices = _devices((0, 80, 35), (1, 80, 70)), main_gb = 20.0)
        self.assertTrue(ok)
        self.assertEqual(info["mode"], "gguf")
        auto_mock.assert_not_called()  # GGUF never uses the HF auto selector

    def test_main_over_aggregate_refuses(self):
        # free [10, 10]; main 30 -> needed 38.5, aggregate 18.5 -> refuse.
        ok, _, _ = self._run(devices = _devices((0, 80, 70), (1, 80, 70)), main_gb = 30.0)
        self.assertFalse(ok)

    def test_companion_checked_against_min_free_device(self):
        # Main 20 fits the aggregate, but a companion lands whole on one device.
        # min free = 10; companion 2 -> 2.3 <= 10 (allow); companion 10 -> 11.5 > 10.
        ok_small, _, _ = self._run(
            devices = _devices((0, 80, 35), (1, 80, 70)), main_gb = 20.0, companion_gb = 2.0
        )
        ok_big, _, _ = self._run(
            devices = _devices((0, 80, 35), (1, 80, 70)), main_gb = 20.0, companion_gb = 10.0
        )
        self.assertTrue(ok_small)
        self.assertFalse(ok_big)

    def test_companion_check_is_order_independent(self):
        # min-free is the same whichever GPU is full, so no first/physical-id
        # guessing: the tight GPU first or last gives the same verdict.
        first_full, _, _ = self._run(
            devices = _devices((0, 80, 78), (1, 80, 35)), main_gb = 5.0, companion_gb = 8.0
        )
        last_full, _, _ = self._run(
            devices = _devices((0, 80, 35), (1, 80, 78)), main_gb = 5.0, companion_gb = 8.0
        )
        self.assertEqual(first_full, last_full)
        self.assertFalse(first_full)  # min free 2 < 8*1.15

    def test_single_device_requires_one_gpu_holds_all(self):
        # Diffusion / manual split: no distribution. main 20 + companion 2 ->
        # needed 29.3. free [45, 10] aggregate is huge but min free 10 -> refuse;
        # [45, 45] -> allow.
        refuse, _, _ = self._run(
            devices = _devices((0, 80, 35), (1, 80, 70)),
            main_gb = 20.0,
            companion_gb = 2.0,
            single_device = True,
        )
        allow, _, _ = self._run(
            devices = _devices((0, 80, 35), (1, 80, 35)),
            main_gb = 20.0,
            companion_gb = 2.0,
            single_device = True,
        )
        self.assertFalse(refuse)
        self.assertTrue(allow)

    def test_manual_split_checks_max_share_not_whole_model(self):
        # An even split [1,1] concentrates half the main model on one card, so
        # the tightest device needs main/2 (not the whole model, which would
        # over-refuse a valid even split, nor just the aggregate). main 20,
        # share 0.5 -> per-device 10; free [45, 25] min 25 >= 11.5 -> allow;
        # free [45, 8] min 8 < 11.5 -> refuse.
        allow, _, _ = self._run(
            devices = _devices((0, 80, 35), (1, 80, 55)), main_gb = 20.0, split_max_share = 0.5
        )
        refuse, _, _ = self._run(
            devices = _devices((0, 80, 35), (1, 80, 72)), main_gb = 20.0, split_max_share = 0.5
        )
        self.assertTrue(allow)
        self.assertFalse(refuse)

    def test_zero_estimate_is_cpu_only(self):
        # Deliberate zero-offload (gpu_layers=0, no companions): holds no VRAM,
        # so near-full GPUs (free 1 GB < the 4 GB floor) must not 409 the load.
        ok, info, _ = self._run(devices = _devices((0, 80, 79)), main_gb = 0.0)
        self.assertTrue(ok)
        self.assertEqual(info["reason"], "cpu_only")

    def test_zero_main_still_checks_companion(self):
        # gpu_layers=0 with a vision projector: main 0 but the companion still
        # lands on the GPU. free 2 < 3*1.15 -> refuse; free 4 -> allow.
        refuse, _, _ = self._run(devices = _devices((0, 80, 78)), main_gb = 0.0, companion_gb = 3.0)
        allow, _, _ = self._run(devices = _devices((0, 80, 76)), main_gb = 0.0, companion_gb = 3.0)
        self.assertFalse(refuse)
        self.assertTrue(allow)

    def test_explicit_pick_scopes_to_picked_free(self):
        # gpu_ids=[0] considers only GPU 0's free (5), not the free GPU 1.
        ok, info, _ = self._run(
            devices = _devices((0, 80, 75), (1, 80, 0)), main_gb = 10.0, gpu_ids = [0]
        )
        self.assertFalse(ok)
        self.assertEqual(info["mode"], "explicit")

    def test_estimate_unavailable_refuses(self):
        # main_gb None (the estimator couldn't size it) -> default-deny.
        ok, info, _ = self._run(devices = _devices((0, 80, 0)), main_gb = None)
        self.assertFalse(ok)
        self.assertEqual(info["reason"], "estimate_unavailable")


# ── can_load_chat_during_training: device-independent paths ──────────────────


class TestCanLoadMisc(_GpuCacheResetMixin, unittest.TestCase):
    def test_non_cuda_allows(self):
        with patch("utils.hardware.get_device", return_value = DeviceType.MLX):
            ok, info = tv.can_load_chat_during_training(
                model_name = "m",
                hf_token = None,
                load_in_4bit = True,
                max_seq_length = 0,
                requested_gpu_ids = None,
            )
        self.assertTrue(ok)
        self.assertEqual(info["mode"], "non_cuda")

    def test_no_visible_gpus_refuses(self):
        # GGUF with an empty device list -> no candidate GPU -> default-deny.
        with (
            patch("utils.hardware.get_device", return_value = DeviceType.CUDA),
            patch("utils.hardware.get_visible_gpu_utilization", return_value = {"devices": []}),
            patch("utils.hardware.auto_select_gpu_ids"),
        ):
            ok, info = tv.can_load_chat_during_training(
                model_name = "m",
                hf_token = None,
                load_in_4bit = True,
                max_seq_length = 0,
                requested_gpu_ids = None,
                is_gguf = True,
                gguf_main_gb = 8.0,
            )
        self.assertFalse(ok)
        self.assertEqual(info["reason"], "no_visible_gpus")

    def test_probe_exception_refuses(self):
        with patch("utils.hardware.get_device", side_effect = RuntimeError("boom")):
            ok, info = tv.can_load_chat_during_training(
                model_name = "m",
                hf_token = None,
                load_in_4bit = True,
                max_seq_length = 0,
                requested_gpu_ids = None,
            )
        self.assertFalse(ok)
        self.assertEqual(info["reason"], "probe_error")


# ── _guard_chat_load_against_training + _effective_load_in_4bit (route) ───────


def _load_inference_route():
    spec = importlib.util.spec_from_file_location(
        "inference_route_chatload_test", _BACKEND_ROOT / "routes" / "inference.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub_guard_deps(
    *,
    training_active,
    decision,
    captured = None,
):
    """Inject the guard's two lazy imports (get_training_backend, can_load_chat_
    during_training); `captured` records the can_load kwargs for assertions."""
    core_training = types.ModuleType("core.training")
    if isinstance(training_active, Exception):

        def _raise():
            raise training_active

        core_training.get_training_backend = _raise
    else:
        core_training.get_training_backend = lambda: SimpleNamespace(
            is_training_active = lambda: training_active
        )

    def _can_load(**kwargs):
        if captured is not None:
            captured.append(kwargs)
        return decision

    tv_stub = types.ModuleType("routes.training_vram")
    tv_stub.can_load_chat_during_training = _can_load
    return patch.dict(
        sys.modules, {"core.training": core_training, "routes.training_vram": tv_stub}
    )


class TestChatLoadGuardRoute(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.route = _load_inference_route()

    def _guard(
        self,
        *,
        config = None,
        captured = None,
        training_active,
        decision,
    ):
        config = config or SimpleNamespace(is_gguf = False, is_lora = False, path = None)
        with _stub_guard_deps(
            training_active = training_active, decision = decision, captured = captured
        ):
            self.route._guard_chat_load_against_training(
                config,
                model_identifier = "unsloth/Qwen3-1.7B",
                hf_token = None,
                load_in_4bit = True,
                max_seq_length = 0,
                requested_gpu_ids = None,
            )

    def test_noop_when_training_inactive(self):
        self._guard(training_active = False, decision = (False, {}))  # must not raise

    def test_noop_when_training_state_unknown(self):
        self._guard(training_active = RuntimeError("no backend"), decision = (False, {}))

    def test_allows_when_fits(self):
        self._guard(training_active = True, decision = (True, {"mode": "auto"}))

    def test_refuses_with_headroom_number(self):
        info = {"required_gb": 30.0, "usable_gb": 6.0, "needed_gb": 39.0, "mode": "auto"}
        with self.assertRaises(HTTPException) as exc:
            self._guard(training_active = True, decision = (False, info))
        self.assertEqual(exc.exception.status_code, 409)
        self.assertIn("39 GB", exc.exception.detail)  # reports needed_gb, not required_gb 30
        self.assertNotIn("30 GB", exc.exception.detail)
        self.assertIn("including safety headroom", exc.exception.detail)
        self.assertNotIn("chat is disabled", exc.exception.detail.lower())

    def test_refuses_generic_when_unsizable(self):
        with self.assertRaises(HTTPException) as exc:
            self._guard(training_active = True, decision = (False, {"reason": "estimate_unavailable"}))
        self.assertEqual(exc.exception.status_code, 409)
        self.assertIn("could not be verified", exc.exception.detail)

    def test_gguf_config_passes_main_and_companion_parts(self):
        captured = []
        config = SimpleNamespace(is_gguf = True, identifier = "org/repo-GGUF")
        # The estimator returns (main, companion) parts: main distributes, the
        # companion lands on one device, so the guard receives them separately.
        with patch.object(self.route, "_estimate_gguf_required_gb", return_value = (12.5, 1.5)):
            self._guard(
                config = config,
                captured = captured,
                training_active = True,
                decision = (True, {}),
            )
        self.assertEqual(captured[0]["is_gguf"], True)
        self.assertEqual(captured[0]["gguf_main_gb"], 12.5)
        self.assertEqual(captured[0]["gguf_companion_gb"], 1.5)
        self.assertFalse(captured[0]["gguf_single_device"])

    def test_gguf_diffusion_marks_single_device(self):
        captured = []
        config = SimpleNamespace(is_gguf = True, identifier = "unsloth/DiffusionGemma-GGUF")
        with patch.object(self.route, "_estimate_gguf_required_gb", return_value = (12.5, 0.0)):
            self._guard(config = config, captured = captured, training_active = True, decision = (True, {}))
        self.assertTrue(captured[0]["gguf_single_device"])


class TestEffectiveLoadIn4bit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.route = _load_inference_route()

    def _write_adapter(self, tmpdir, payload):
        import json
        (Path(tmpdir) / "adapter_config.json").write_text(json.dumps(payload))

    def test_non_lora_returns_request(self):
        cfg = SimpleNamespace(is_lora = False, path = None, base_model = None)
        self.assertTrue(self.route._effective_load_in_4bit(cfg, True))

    def test_lora_method_flips_to_16bit(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self._write_adapter(d, {"unsloth_training_method": "lora"})
            cfg = SimpleNamespace(is_lora = True, path = d, base_model = "x")
            # requested 4-bit, but a 'lora' adapter loads 16-bit
            self.assertFalse(self.route._effective_load_in_4bit(cfg, True))

    def test_qlora_method_keeps_4bit(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self._write_adapter(d, {"unsloth_training_method": "qlora"})
            cfg = SimpleNamespace(is_lora = True, path = d, base_model = "x")
            self.assertTrue(self.route._effective_load_in_4bit(cfg, True))

    def test_no_method_non_bnb_base_flips_to_16bit(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self._write_adapter(d, {})
            cfg = SimpleNamespace(is_lora = True, path = d, base_model = "meta/Llama-3-8B")
            self.assertFalse(self.route._effective_load_in_4bit(cfg, True))

    def test_malformed_adapter_config_returns_request(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "adapter_config.json").write_text("[1, 2, 3]")  # not a dict
            cfg = SimpleNamespace(is_lora = True, path = d, base_model = "x")
            self.assertTrue(self.route._effective_load_in_4bit(cfg, True))  # no crash


# ── validate_model integration (early refusal, real settings) ────────────────


class TestValidateRefusesDuringTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.route = _load_inference_route()

    def _validate(
        self,
        *,
        training_active,
        decision,
        captured = None,
        load_in_4bit = True,
    ):
        from models.inference import ValidateModelRequest

        request = ValidateModelRequest(
            model_path = "unsloth/Qwen3-1.7B", load_in_4bit = load_in_4bit, max_seq_length = 4096
        )
        cfg = SimpleNamespace(
            identifier = "unsloth/Qwen3-1.7B",
            display_name = "Qwen3-1.7B",
            is_gguf = False,
            is_lora = False,
            is_vision = False,
            path = None,
            base_model = None,
        )
        with (
            patch.object(
                self.route,
                "_resolve_model_identifier_for_request",
                return_value = ("unsloth/Qwen3-1.7B", "unsloth/Qwen3-1.7B", False),
            ),
            patch.object(self.route.ModelConfig, "from_identifier", return_value = cfg),
            patch.object(self.route, "load_inference_config", return_value = {}),
            _stub_guard_deps(training_active = training_active, decision = decision, captured = captured),
        ):
            return asyncio.run(self.route.validate_model(request, current_subject = "test-user"))

    def test_ok_when_training_inactive(self):
        resp = self._validate(training_active = False, decision = (False, {}))
        self.assertTrue(resp.valid)

    def test_refuses_when_wont_fit(self):
        info = {"required_gb": 40.0, "usable_gb": 5.0, "needed_gb": 50.0}
        with self.assertRaises(HTTPException) as exc:
            self._validate(training_active = True, decision = (False, info))
        self.assertEqual(exc.exception.status_code, 409)
        self.assertIn("training is running", exc.exception.detail)

    def test_passes_real_load_settings_to_guard(self):
        # validate must size with the request's settings, not hardcoded defaults.
        captured = []
        self._validate(
            training_active = True, decision = (True, {}), captured = captured, load_in_4bit = False
        )
        self.assertEqual(captured[0]["load_in_4bit"], False)
        self.assertEqual(captured[0]["max_seq_length"], 4096)

    def test_validates_gguf_gpu_ids_before_guard(self):
        # gpu_ids is now SUPPORTED for GGUF (the GPU picker); /validate must
        # mirror /load by validating the pick before the VRAM guard, so a bad
        # pick is a clean 400 (not the removed "not supported for GGUF") and the
        # guard is never reached. Patch the validator so the test is
        # deterministic regardless of the host's GPU env.
        import utils.hardware.hardware as hardware_mod
        from models.inference import ValidateModelRequest

        request = ValidateModelRequest(model_path = "x.gguf", gpu_ids = [0])
        cfg = SimpleNamespace(
            identifier = "x.gguf",
            display_name = "x",
            is_gguf = True,
            is_lora = False,
            is_vision = False,
            path = None,
            base_model = None,
        )
        captured = []
        with (
            patch.object(
                self.route,
                "_resolve_model_identifier_for_request",
                return_value = ("x.gguf", "x.gguf", False),
            ),
            patch.object(self.route.ModelConfig, "from_identifier", return_value = cfg),
            patch.object(self.route, "load_inference_config", return_value = {}),
            patch.object(
                hardware_mod,
                "resolve_requested_gpu_ids",
                side_effect = ValueError("Invalid gpu_ids [0]: rejected by test"),
            ),
            _stub_guard_deps(training_active = True, decision = (True, {}), captured = captured),
        ):
            with self.assertRaises(HTTPException) as exc:
                asyncio.run(self.route.validate_model(request, current_subject = "u"))
        self.assertEqual(exc.exception.status_code, 400)
        self.assertIn("gpu_ids", exc.exception.detail.lower())
        self.assertNotIn("not supported", exc.exception.detail.lower())
        self.assertEqual(captured, [])  # guard never reached


# ── _estimate_gguf_required_gb (sizes the same weights the loader loads) ──────


class TestEstimateGgufRequiredGb(unittest.TestCase):
    """The conservative upper bound: (main_gb, companion_gb). main is weights +
    KV scaled by the manual offload fraction (gpu_layers=0 -> 0, no header);
    companions (mmproj + any MTP/draft model) are always charged in full so the
    guard never under-estimates. An unsizable size returns None (default-deny)."""

    @classmethod
    def setUpClass(cls):
        cls.route = _load_inference_route()

    def _local_gguf_cfg(self, path):
        return SimpleNamespace(
            gguf_file = str(path),
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_hf_repo = None,
            gguf_variant = None,
        )

    def test_local_sums_split_shards_no_companion(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "model-00001-of-00002.gguf").write_bytes(b"x" * 1000)
            (p / "model-00002-of-00002.gguf").write_bytes(b"y" * 2000)
            cfg = self._local_gguf_cfg(p / "model-00001-of-00002.gguf")
            with patch.object(self.route, "_estimate_gguf_kv_gb", return_value = 0.0):
                main_gb, companion_gb = self.route._estimate_gguf_required_gb(cfg)
        self.assertAlmostEqual(main_gb, 3000 / (1024**3), places = 9)  # both shards
        self.assertEqual(companion_gb, 0.0)

    def test_local_adds_kv_to_main(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.gguf"
            p.write_bytes(b"x" * 1000)
            cfg = self._local_gguf_cfg(p)
            with patch.object(self.route, "_estimate_gguf_kv_gb", return_value = 2.0):
                main_gb, companion_gb = self.route._estimate_gguf_required_gb(
                    cfg, max_seq_length = 8192
                )
        self.assertAlmostEqual(main_gb, 1000 / (1024**3) + 2.0, places = 6)
        self.assertEqual(companion_gb, 0.0)

    def test_remote_threads_token_and_charges_companions(self):
        import utils.models.model_config as mc

        cfg = SimpleNamespace(
            gguf_file = None,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_hf_repo = "org/repo",
            gguf_variant = "Q4_K_M",
        )
        variant = SimpleNamespace(quant = "Q4_K_M", size_bytes = 10 * 1024**3)
        captured = {}

        def fake_list(repo, hf_token = None):
            captured["token"] = hf_token
            return ([variant], True)  # has_vision -> include mmproj

        with (
            patch.object(mc, "list_gguf_variants", fake_list),
            patch.object(
                self.route, "_remote_gguf_companion_bytes", return_value = 2 * 1024**3
            ) as comp,
        ):
            main_gb, companion_gb = self.route._estimate_gguf_required_gb(cfg, hf_token = "tok")
        self.assertEqual(captured["token"], "tok")  # token threaded for gated repos
        self.assertAlmostEqual(main_gb, 10.0, places = 6)  # full remote (not downloaded)
        self.assertAlmostEqual(companion_gb, 2.0, places = 6)
        self.assertTrue(comp.call_args.kwargs["include_mmproj"])

    def test_remote_unknown_variant_returns_none(self):
        import utils.models.model_config as mc
        cfg = SimpleNamespace(
            gguf_file = None,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_hf_repo = "org/repo",
            gguf_variant = "Q8_0",
        )
        with patch.object(
            mc,
            "list_gguf_variants",
            return_value = ([SimpleNamespace(quant = "Q4_K_M", size_bytes = 1)], False),
        ):
            self.assertIsNone(self.route._estimate_gguf_required_gb(cfg))

    def test_manual_scales_main_by_gpu_layer_fraction(self):
        # Manual offload keeps only gpu_layers/total of the main model (weights +
        # KV) on the GPU; auto (the default) never scales.
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.gguf"
            p.write_bytes(b"x" * 1000)
            cfg = self._local_gguf_cfg(p)
            with (
                patch.object(self.route, "_estimate_gguf_kv_gb", return_value = 2.0),
                patch.object(self.route, "_manual_gpu_layer_fraction", return_value = 0.25),
            ):
                resident = 1000 / (1024**3) + 2.0
                manual, _ = self.route._estimate_gguf_required_gb(
                    cfg, gpu_memory_mode = "manual", gpu_layers = 8
                )
                auto, _ = self.route._estimate_gguf_required_gb(cfg)
        self.assertAlmostEqual(manual, resident * 0.25, places = 6)
        self.assertAlmostEqual(auto, resident, places = 6)

    def test_manual_keeps_full_main_when_layer_count_unreadable(self):
        # Fraction None (can't read layers) -> no scale; the full estimate stands
        # so training can't be OOM'd.
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.gguf"
            p.write_bytes(b"x" * 1000)
            cfg = self._local_gguf_cfg(p)
            with (
                patch.object(self.route, "_estimate_gguf_kv_gb", return_value = 0.0),
                patch.object(self.route, "_manual_gpu_layer_fraction", return_value = None),
            ):
                main_gb, _ = self.route._estimate_gguf_required_gb(
                    cfg, gpu_memory_mode = "manual", gpu_layers = 4
                )
        self.assertAlmostEqual(main_gb, 1000 / (1024**3), places = 9)

    def test_zero_offload_zeroes_main_keeps_companions(self):
        # gpu_layers=0: main is CPU-only (0, no header read), but every declared
        # and extras companion is still charged in full -- they land on the GPU.
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            main = Path(d) / "model.gguf"
            main.write_bytes(b"x" * 1000)
            mtp = Path(d) / "mtp-model.gguf"
            mtp.write_bytes(b"m" * 3000)
            draft = Path(d) / "draft.gguf"
            draft.write_bytes(b"y" * 500)
            cfg = self._local_gguf_cfg(main)
            cfg.gguf_mtp_file = str(mtp)
            main_gb, companion_gb = self.route._estimate_gguf_required_gb(
                cfg,
                llama_extra_args = ["--model-draft", str(draft)],
                gpu_memory_mode = "manual",
                gpu_layers = 0,
            )
        self.assertEqual(main_gb, 0.0)  # CPU-only main
        # config MTP (3000) + extras drafter (500), both charged in full.
        self.assertAlmostEqual(companion_gb, 3500 / (1024**3), places = 9)

    def test_companions_charged_in_full_regardless_of_mode(self):
        # A projector / drafter is GPU-resident no matter the spec mode or CPU
        # flags -- the conservative bound charges it, over-refusing at worst.
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            main = Path(d) / "model.gguf"
            main.write_bytes(b"x" * 1000)
            proj = Path(d) / "mmproj.gguf"
            proj.write_bytes(b"p" * 600)
            cfg = self._local_gguf_cfg(main)
            cfg.gguf_mmproj_file = str(proj)
            with (
                patch.object(self.route, "_estimate_gguf_kv_gb", return_value = 0.0),
                patch.object(self.route, "_manual_gpu_layer_fraction", return_value = 0.0),
            ):
                # --no-mmproj / spec off do NOT change the charge (upper bound).
                _, companion_off = self.route._estimate_gguf_required_gb(
                    cfg,
                    llama_extra_args = ["--no-mmproj"],
                    gpu_memory_mode = "manual",
                    gpu_layers = 0,
                )
        self.assertAlmostEqual(companion_off, 600 / (1024**3), places = 9)

    def test_unsizable_extras_drafter_denies(self):
        # An HF-repo drafter can't be sized pre-download -> None (default-deny).
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.gguf"
            p.write_bytes(b"x" * 1000)
            cfg = self._local_gguf_cfg(p)
            self.assertIsNone(
                self.route._estimate_gguf_required_gb(
                    cfg, llama_extra_args = ["-hfd", "org/draft-repo"]
                )
            )

    def test_remote_zero_layer_credited_before_download(self):
        # Zero offload needs no header: an uncached remote GGUF at gpu_layers=0
        # zeroes main; companions stay charged.
        import utils.models.model_config as mc

        cfg = SimpleNamespace(
            gguf_file = None,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_hf_repo = "org/repo",
            gguf_variant = "Q4_K_M",
        )
        variant = SimpleNamespace(quant = "Q4_K_M", size_bytes = 10 * 1024**3)
        with (
            patch.object(mc, "list_gguf_variants", return_value = ([variant], True)),
            patch.object(self.route, "_remote_gguf_companion_bytes", return_value = 1024**3),
        ):
            main_gb, companion_gb = self.route._estimate_gguf_required_gb(
                cfg, gpu_memory_mode = "manual", gpu_layers = 0
            )
        self.assertEqual(main_gb, 0.0)
        self.assertAlmostEqual(companion_gb, 1.0, places = 6)

    def test_manual_scales_cached_hf_repo_by_gpu_layer_fraction(self):
        # A cached repo id is local in all but name; scale its main from the
        # cached header. A not-yet-cached repo keeps the full remote size.
        import hub.utils.gguf as gguf_mod
        import utils.models.model_config as mc

        cfg = SimpleNamespace(
            gguf_file = None,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_hf_repo = "org/repo",
            gguf_variant = "Q4_K_M",
        )
        variant = SimpleNamespace(quant = "Q4_K_M", size_bytes = 10 * 1024**3)
        with (
            patch.object(mc, "list_gguf_variants", return_value = ([variant], False)),
            patch.object(self.route, "_remote_gguf_companion_bytes", return_value = 0),
            patch.object(
                self.route.LlamaCppBackend, "_get_gguf_size_bytes", return_value = 8 * 1024**3
            ),
            patch.object(self.route, "_estimate_gguf_kv_gb", return_value = 2.0),
            patch.object(self.route, "_manual_gpu_layer_fraction", return_value = 0.25),
        ):
            with patch.object(gguf_mod, "resolve_local_gguf_path", return_value = "/cache/m.gguf"):
                cached, _ = self.route._estimate_gguf_required_gb(
                    cfg, gpu_memory_mode = "manual", gpu_layers = 8
                )
            with patch.object(gguf_mod, "resolve_local_gguf_path", return_value = None):
                not_cached, _ = self.route._estimate_gguf_required_gb(
                    cfg, gpu_memory_mode = "manual", gpu_layers = 8
                )
            auto, _ = self.route._estimate_gguf_required_gb(cfg)  # auto never scales
        self.assertAlmostEqual(cached, (8.0 + 2.0) * 0.25, places = 6)  # cached header, scaled
        self.assertAlmostEqual(not_cached, 10.0, places = 6)  # no local header -> full remote
        self.assertAlmostEqual(auto, 10.0, places = 6)

    def test_is_diffusion_gguf_by_name(self):
        # Name match catches DiffusionGemma pre-download (no header needed); a
        # plainly-named dense model with no local file is not diffusion.
        diff = SimpleNamespace(
            identifier = "unsloth/DiffusionGemma-4B-GGUF",
            gguf_hf_repo = None,
            gguf_file = None,
            gguf_variant = None,
        )
        dense = SimpleNamespace(
            identifier = "unsloth/gemma-4-E2B-it-GGUF",
            gguf_hf_repo = None,
            gguf_file = None,
            gguf_variant = None,
        )
        self.assertTrue(self.route._is_diffusion_gguf(diff))
        self.assertFalse(self.route._is_diffusion_gguf(dense))

    def test_is_diffusion_gguf_by_header_when_name_lacks_it(self):
        # A locally-renamed DiffusionGemma file (generic path, no "diffusion" in
        # the name) must still be caught via the on-disk header, or the guard
        # would size it multi-GPU and OOM the single device it runs on.
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.gguf"
            p.write_bytes(b"x" * 100)
            cfg = SimpleNamespace(
                identifier = "/models/model.gguf",
                gguf_hf_repo = None,
                gguf_file = str(p),
                gguf_variant = None,
            )
            with (
                patch(
                    "utils.models.gguf_metadata.read_gguf_general_metadata",
                    return_value = {"general.architecture": "diffusion-gemma"},
                ),
                patch("utils.models.gguf_metadata.gguf_header_has_key", return_value = False),
            ):
                self.assertTrue(self.route._is_diffusion_gguf(cfg))
            with (
                patch(
                    "utils.models.gguf_metadata.read_gguf_general_metadata",
                    return_value = {"general.architecture": "gemma3"},
                ),
                patch("utils.models.gguf_metadata.gguf_header_has_key", return_value = True),
            ):
                self.assertTrue(self.route._is_diffusion_gguf(cfg))  # canvas marker only
            with (
                patch(
                    "utils.models.gguf_metadata.read_gguf_general_metadata",
                    return_value = {"general.architecture": "gemma3"},
                ),
                patch("utils.models.gguf_metadata.gguf_header_has_key", return_value = False),
            ):
                self.assertFalse(self.route._is_diffusion_gguf(cfg))  # genuine dense

    def test_validate_shaped_request_inherits_same_model_extras(self):
        # /validate has no llama_extra_args field, so a same-model reload must
        # inherit the loaded extras for its guard just like /load -- else
        # validate passes a smaller estimate, the frontend unloads, and the
        # follow-up /load 409s.
        backend = SimpleNamespace(
            extra_args = ["-c", "32768"],
            extra_args_source = ("owner/repo", "q4_k_m"),
        )
        cfg = SimpleNamespace(is_gguf = True, gguf_variant = "Q4_K_M")
        request = SimpleNamespace(gguf_variant = "Q4_K_M", gpu_memory_mode = "auto")
        with patch.object(self.route, "get_llama_cpp_backend", return_value = backend):
            inherited = self.route._resolve_inherited_extra_args(request, cfg, "owner/repo", None)
            cross = self.route._resolve_inherited_extra_args(request, cfg, "other/model", None)
        self.assertEqual(inherited, ["-c", "32768"])
        self.assertEqual(cross, [])

    def test_manual_gpu_layer_fraction_clamps_and_reads_layers(self):
        # _manual_gpu_layer_fraction imports read_gguf_staged_dims lazily, so
        # patch it at its source module.
        import utils.models.gguf_metadata as gm

        frac = self.route._manual_gpu_layer_fraction
        with patch.object(gm, "read_gguf_staged_dims", return_value = {"layer_count": 32}):
            self.assertAlmostEqual(frac("x.gguf", 8), 0.25, places = 9)
            self.assertEqual(frac("x.gguf", 0), 0.0)  # all on CPU
            self.assertEqual(frac("x.gguf", 999), 1.0)  # clamps above layer count
        with patch.object(gm, "read_gguf_staged_dims", return_value = None):
            self.assertIsNone(frac("x.gguf", 8))
        with patch.object(gm, "read_gguf_staged_dims", return_value = {"layer_count": None}):
            self.assertIsNone(frac("x.gguf", 8))

    def test_kv_helper_graceful_on_non_gguf(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "not-a.gguf"
            p.write_bytes(b"not a gguf")
            self.assertEqual(self.route._estimate_gguf_kv_gb(str(p), 4096), 0.0)

    def test_kv_sizes_at_larger_of_max_seq_len_and_ctx_override(self):
        # KV sized at the larger of max_seq_length and --ctx-size, else native.
        seen = {}

        class _FakeBackend:
            _context_length = 2048

            def _read_gguf_metadata(self, path):
                pass

            def _can_estimate_kv(self):
                return True

            def _estimate_kv_cache_bytes(
                self,
                ctx,
                n_parallel = 1,
            ):
                seen["ctx"] = ctx
                seen["n_parallel"] = n_parallel
                return ctx * n_parallel * (1024**2)  # 1 MiB per ctx unit per slot

        with patch.object(self.route, "LlamaCppBackend", _FakeBackend):
            r = self.route
            self.assertAlmostEqual(
                r._estimate_gguf_kv_gb("m", 4096, ["--ctx-size", "131072"]), 128.0
            )
            self.assertEqual(seen["ctx"], 131072)
            self.assertEqual(seen["n_parallel"], 1)  # default single slot
            self.assertAlmostEqual(r._estimate_gguf_kv_gb("m", 4096, ["--ctx-size", "1024"]), 4.0)
            self.assertEqual(seen["ctx"], 4096)
            self.assertAlmostEqual(r._estimate_gguf_kv_gb("m", 0, None), 2.0)
            self.assertEqual(seen["ctx"], 2048)
            self.assertAlmostEqual(r._estimate_gguf_kv_gb("m", 4096, ["--ctx-size", "oops"]), 4.0)
            self.assertAlmostEqual(r._estimate_gguf_kv_gb("m", 4096, None, 4), 16.0)
            self.assertEqual(seen["n_parallel"], 4)


# ── load_model integration: authoritative 409, and no unload before refusal ──


class TestLoadModelGuardIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.route = _load_inference_route()

    def test_refusal_409_and_no_unload(self):
        import contextlib
        from unittest.mock import MagicMock
        from models.inference import LoadRequest

        inf = SimpleNamespace(active_model_name = None)
        inf.unload_model = MagicMock()
        inf._shutdown_subprocess = MagicMock()
        llama = SimpleNamespace(is_loaded = False, model_identifier = None, hf_variant = None)
        llama.unload_model = MagicMock()
        cfg = SimpleNamespace(is_gguf = False, is_lora = False, path = None, base_model = None)
        request = LoadRequest(model_path = "unsloth/Qwen3-1.7B")
        info = {"required_gb": 40.0, "usable_gb": 5.0, "needed_gb": 50.0, "mode": "auto"}

        with (
            patch.object(self.route, "validate_extra_args", return_value = None),
            patch.object(
                self.route,
                "_resolve_model_identifier_for_request",
                return_value = ("unsloth/Qwen3-1.7B", "unsloth/Qwen3-1.7B", False),
            ),
            patch.object(self.route, "resolve_effective_chat_template_override", return_value = None),
            patch.object(self.route, "get_inference_backend", return_value = inf),
            patch.object(self.route, "get_llama_cpp_backend", return_value = llama),
            patch.object(self.route, "_hf_offline_if_dns_dead", lambda: contextlib.nullcontext()),
            patch.object(self.route.ModelConfig, "from_identifier", return_value = cfg),
            _stub_guard_deps(training_active = True, decision = (False, info)),
        ):
            with self.assertRaises(HTTPException) as exc:
                asyncio.run(
                    self.route.load_model(request, fastapi_request = MagicMock(), current_subject = "u")
                )

        self.assertEqual(exc.exception.status_code, 409)
        # Guard runs before the unload step, so a refused load tears down nothing.
        inf.unload_model.assert_not_called()
        inf._shutdown_subprocess.assert_not_called()
        llama.unload_model.assert_not_called()


# ── split-mode vs tensor-split strip predicates (manual ratio decoupling) ─────


class TestStripSplitPredicates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.route = _load_inference_route()

    def _req(self, **kw):
        from models.inference import LoadRequest
        return LoadRequest(model_path = "unsloth/Qwen3-1.7B", **kw)

    def test_should_strip_tensor_split_for_manual_explicit_offload(self):
        # Manual explicit offload (gpu_layers >= 0) owns the split, so strip an
        # inherited --tensor-split whether the ratio is set OR cleared -- else the
        # cleared case silently keeps the stale inherited ratio while status
        # reports None.
        self.assertTrue(
            self.route._should_strip_tensor_split(
                self._req(gpu_memory_mode = "manual", gpu_layers = 8, tensor_split = [2, 1])
            )
        )
        self.assertTrue(  # manual, ratio cleared -> still strip the stale inherited ratio
            self.route._should_strip_tensor_split(self._req(gpu_memory_mode = "manual", gpu_layers = 8))
        )
        self.assertFalse(  # manual + Auto layers (gpu_layers < 0): --fit owns placement
            self.route._should_strip_tensor_split(
                self._req(gpu_memory_mode = "manual", gpu_layers = -1)
            )
        )
        self.assertFalse(self.route._should_strip_tensor_split(self._req()))  # auto

    def test_split_mode_strip_decouples_from_manual_ratio(self):
        # fix #5: a manual ratio must NOT strip --split-mode (the user's row/none
        # survives), but the Tensor Parallelism toggle still owns the whole group.
        manual_ratio = self._req(gpu_memory_mode = "manual", gpu_layers = 8, tensor_split = [2, 1])
        self.assertFalse(self.route._should_strip_split_mode(manual_ratio, []))
        self.assertTrue(self.route._should_strip_split_mode(self._req(tensor_parallel = True), []))


if __name__ == "__main__":
    unittest.main()
