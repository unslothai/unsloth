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
    def _run(
        self,
        *,
        devices,
        required_override = None,
        estimate = None,
        gpu_ids = None,
        tensor_split = None,
    ):
        with (
            patch("utils.hardware.get_device", return_value = DeviceType.CUDA),
            patch("utils.hardware.estimate_required_model_memory_gb", return_value = (estimate, {})),
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
                required_override_gb = required_override,
                tensor_split = tensor_split,
            )
        return ok, info, auto_mock

    def test_override_fits(self):
        ok, info, auto_mock = self._run(devices = _devices((0, 80, 20)), required_override = 10.0)
        self.assertTrue(ok)
        self.assertEqual(info["mode"], "gguf")
        auto_mock.assert_not_called()  # GGUF never uses the HF auto selector

    def test_no_per_gpu_floor_for_gguf(self):
        # free [45, 10], override 20 -> needed 27, aggregate 53.5 >= 27. GGUF self-
        # places, so the per-GPU floor that would block HF doesn't apply -> allow.
        ok, _, _ = self._run(devices = _devices((0, 80, 35), (1, 80, 70)), required_override = 20.0)
        self.assertTrue(ok)

    def test_no_per_gpu_floor_for_gguf_explicit_pick(self):
        # Same free [45, 10] shape, but with the GPUs pinned via gpu_ids. llama.cpp
        # still self-places by free VRAM within the allowed set, so the HF
        # balanced-shard floor (27/2 = 13.5 > 10) must not fire -> allow.
        ok, info, _ = self._run(
            devices = _devices((0, 80, 35), (1, 80, 70)),
            required_override = 20.0,
            gpu_ids = [0, 1],
        )
        self.assertTrue(ok)
        self.assertEqual(info["mode"], "explicit")

    def test_manual_even_split_reenables_per_gpu_check(self):
        # A manual --tensor-split overrides free-VRAM placement: [1, 1] puts half
        # (13.5 of needed 27) on the 10 GB-free GPU even though the aggregate
        # fits -> refuse, would OOM training.
        ok, _, _ = self._run(
            devices = _devices((0, 80, 35), (1, 80, 70)),
            required_override = 20.0,
            tensor_split = [1, 1],
        )
        self.assertFalse(ok)

    def test_manual_weighted_split_that_fits_passes(self):
        # [4, 1] charges each GPU its actual share (21.6 / 5.4 of needed 27);
        # both fit their free VRAM (45 / 10) -> allow.
        ok, _, _ = self._run(
            devices = _devices((0, 80, 35), (1, 80, 70)),
            required_override = 20.0,
            tensor_split = [4, 1],
        )
        self.assertTrue(ok)

    def test_manual_split_with_explicit_pick_checks_shares(self):
        # The split check also covers picks routed through the explicit branch.
        ok, info, _ = self._run(
            devices = _devices((0, 80, 35), (1, 80, 70)),
            required_override = 20.0,
            gpu_ids = [0, 1],
            tensor_split = [1, 1],
        )
        self.assertFalse(ok)
        self.assertEqual(info["mode"], "explicit")

    def test_manual_split_maps_shares_to_sorted_pick(self):
        # llama-server pins CUDA to the SORTED pick, so with gpu_ids [1, 0] the
        # split [1, 4] puts the big share on GPU 1 (10 GB free -> needs 21.6).
        # Zipping in request order would wrongly allow it.
        ok, _, _ = self._run(
            devices = _devices((0, 80, 35), (1, 80, 70)),
            required_override = 20.0,
            gpu_ids = [1, 0],
            tensor_split = [1, 4],
        )
        self.assertFalse(ok)

    def test_manual_split_length_mismatch_falls_back_to_even_floor(self):
        # A split that doesn't match the GPU count aborts llama-server at launch;
        # meanwhile the guard keeps the conservative even-share floor.
        ok, _, _ = self._run(
            devices = _devices((0, 80, 35), (1, 80, 70)),
            required_override = 20.0,
            tensor_split = [1, 1, 1],
        )
        self.assertFalse(ok)

    def test_zero_estimate_bypasses_floor(self):
        # Deliberate zero-offload (manual gpu_layers=0, no companions): estimate
        # 0 means no VRAM held, so even near-full GPUs (free 1 GB < the 4 GB
        # safety floor) must not 409 the load.
        ok, info, _ = self._run(devices = _devices((0, 80, 79)), required_override = 0.0)
        self.assertTrue(ok)
        self.assertEqual(info["reason"], "cpu_only")

    def test_zero_estimate_bypasses_floor_with_explicit_pick(self):
        ok, info, _ = self._run(
            devices = _devices((0, 80, 79), (1, 80, 79)),
            required_override = 0.0,
            gpu_ids = [0, 1],
        )
        self.assertTrue(ok)
        self.assertEqual(info["mode"], "explicit")

    def test_estimate_unavailable_refuses(self):
        # No override and the estimator can't size it -> default-deny.
        ok, info, _ = self._run(devices = _devices((0, 80, 0)), required_override = None, estimate = None)
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
                required_override_gb = 8.0,
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

    def test_gguf_config_passes_is_gguf_and_override(self):
        captured = []
        config = SimpleNamespace(is_gguf = True)
        with patch.object(self.route, "_estimate_gguf_required_gb", return_value = 12.5):
            self._guard(
                config = config,
                captured = captured,
                training_active = True,
                decision = (True, {}),
            )
        self.assertEqual(captured[0]["is_gguf"], True)
        self.assertEqual(captured[0]["required_override_gb"], 12.5)


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
    @classmethod
    def setUpClass(cls):
        cls.route = _load_inference_route()

    def test_local_sums_split_shards(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "model-00001-of-00002.gguf").write_bytes(b"x" * 1000)
            (p / "model-00002-of-00002.gguf").write_bytes(b"y" * 2000)
            cfg = SimpleNamespace(
                gguf_file = str(p / "model-00001-of-00002.gguf"),
                gguf_mmproj_file = None,
                gguf_mtp_file = None,
                gguf_hf_repo = None,
                gguf_variant = None,
            )
            gb = self.route._estimate_gguf_required_gb(cfg)
        self.assertAlmostEqual(gb, 3000 / (1024**3), places = 9)  # both shards

    def test_remote_threads_token_and_adds_companions(self):
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
            gb = self.route._estimate_gguf_required_gb(cfg, hf_token = "tok")
        self.assertEqual(captured["token"], "tok")  # token threaded for gated repos
        self.assertAlmostEqual(gb, 12.0, places = 6)  # 10 GB variant + 2 GB companions
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

    def test_local_adds_kv_cache(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.gguf"
            p.write_bytes(b"x" * 1000)
            cfg = SimpleNamespace(
                gguf_file = str(p),
                gguf_mmproj_file = None,
                gguf_mtp_file = None,
                gguf_hf_repo = None,
                gguf_variant = None,
            )
            with patch.object(self.route, "_estimate_gguf_kv_gb", return_value = 2.0):
                gb = self.route._estimate_gguf_required_gb(cfg, max_seq_length = 8192)
        self.assertAlmostEqual(gb, 1000 / (1024**3) + 2.0, places = 6)  # weights + KV

    def _local_gguf_cfg(self, path):
        return SimpleNamespace(
            gguf_file = str(path),
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_hf_repo = None,
            gguf_variant = None,
        )

    def test_manual_scales_estimate_by_gpu_layer_fraction(self):
        # Manual offload keeps only gpu_layers/total of the model (weights + KV)
        # on the GPU; the guard estimate scales down so a CPU-heavy pick isn't
        # over-blocked. auto mode (the default) must not scale.
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
                manual = self.route._estimate_gguf_required_gb(
                    cfg, gpu_memory_mode = "manual", gpu_layers = 8
                )
                auto = self.route._estimate_gguf_required_gb(cfg)
        self.assertAlmostEqual(manual, resident * 0.25, places = 6)
        self.assertAlmostEqual(auto, resident, places = 6)  # default never scales

    def test_manual_keeps_full_estimate_when_layer_count_unreadable(self):
        # _manual_gpu_layer_fraction returns None (can't read layers) -> no scale,
        # the conservative full estimate stands so training can't be OOM'd.
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.gguf"
            p.write_bytes(b"x" * 1000)
            cfg = self._local_gguf_cfg(p)
            with (
                patch.object(self.route, "_estimate_gguf_kv_gb", return_value = 0.0),
                patch.object(self.route, "_manual_gpu_layer_fraction", return_value = None),
            ):
                gb = self.route._estimate_gguf_required_gb(
                    cfg, gpu_memory_mode = "manual", gpu_layers = 4
                )
        self.assertAlmostEqual(gb, 1000 / (1024**3), places = 9)

    def test_extras_drafter_charged_in_full(self):
        # An explicit pass-through drafter is a GPU companion too: charge its
        # file size unscaled, even when the manual fraction zeroes the main
        # model -- else a gpu_layers=0 load with a drafter takes the guard's
        # CPU-only bypass while the drafter still lands on the GPU.
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.gguf"
            p.write_bytes(b"x" * 1000)
            draft = Path(d) / "draft.gguf"
            draft.write_bytes(b"y" * 500)
            cfg = self._local_gguf_cfg(p)
            with (
                patch.object(self.route, "_estimate_gguf_kv_gb", return_value = 0.0),
                patch.object(self.route, "_manual_gpu_layer_fraction", return_value = 0.0),
            ):
                gb = self.route._estimate_gguf_required_gb(
                    cfg,
                    llama_extra_args = ["--model-draft", str(draft)],
                    gpu_memory_mode = "manual",
                    gpu_layers = 0,
                )
        self.assertAlmostEqual(gb, 500 / (1024**3), places = 9)

    def test_unsizable_extras_drafter_denies(self):
        # An HF-repo drafter can't be sized pre-download: return None so the
        # guard default-denies instead of under-estimating.
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.gguf"
            p.write_bytes(b"x" * 1000)
            cfg = self._local_gguf_cfg(p)
            gb = self.route._estimate_gguf_required_gb(
                cfg, llama_extra_args = ["-hfd", "org/draft-repo"]
            )
        self.assertIsNone(gb)

    def test_mtp_companion_skipped_when_spec_mode_never_emits_it(self):
        # "off"/"ngram" never launch the separate drafter, so it must not count
        # toward the estimate (an unused drafter would push a CPU-only load over
        # the guard floor). Default/auto keeps charging it (may emit).
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.gguf"
            p.write_bytes(b"x" * 1000)
            mtp = Path(d) / "mtp-draft.gguf"
            mtp.write_bytes(b"y" * 400)
            cfg = SimpleNamespace(
                gguf_file = str(p),
                gguf_mmproj_file = None,
                gguf_mtp_file = str(mtp),
                gguf_hf_repo = None,
                gguf_variant = None,
            )
            with (
                patch.object(self.route, "_estimate_gguf_kv_gb", return_value = 0.0),
                patch.object(self.route, "_manual_gpu_layer_fraction", return_value = 0.0),
            ):
                kwargs = dict(gpu_memory_mode = "manual", gpu_layers = 0)
                off = self.route._estimate_gguf_required_gb(cfg, speculative_type = "off", **kwargs)
                ngram = self.route._estimate_gguf_required_gb(
                    cfg, speculative_type = "ngram", **kwargs
                )
                auto = self.route._estimate_gguf_required_gb(cfg, **kwargs)
                extras_own_spec = self.route._estimate_gguf_required_gb(
                    cfg, llama_extra_args = ["--spec-type", "ngram-mod"], **kwargs
                )
        self.assertEqual(off, 0.0)
        self.assertEqual(ngram, 0.0)
        self.assertAlmostEqual(auto, 400 / (1024**3), places = 9)
        self.assertEqual(extras_own_spec, 0.0)

    def test_diffusion_pick_collapses_to_lowest_gpu(self):
        # The diffusion runner uses a single device (the lowest of the pick), so
        # the guard must size that GPU only -- aggregating [0, 1] could pass on
        # GPU 1's free VRAM and OOM GPU 0, the one actually used.
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.gguf"
            p.write_bytes(b"x" * 1000)
            cfg = self._local_gguf_cfg(p)
            with patch(
                "utils.models.gguf_metadata.read_gguf_general_metadata",
                return_value = {"general.architecture": "diffusion-gemma"},
            ):
                collapsed = self.route._diffusion_guard_gpu_ids(cfg, [1, 0])
            with patch(
                "utils.models.gguf_metadata.read_gguf_general_metadata",
                return_value = {"general.architecture": "gemma3"},
            ):
                dense = self.route._diffusion_guard_gpu_ids(cfg, [1, 0])
                # Canvas-only block diffusion (arch doesn't say "diffusion")
                # must collapse too, mirroring load_model's OR canvas_seen.
                with patch(
                    "utils.models.gguf_metadata.gguf_header_has_key",
                    return_value = True,
                ):
                    canvas_only = self.route._diffusion_guard_gpu_ids(cfg, [1, 0])
            with patch(
                "utils.models.gguf_metadata.read_gguf_general_metadata",
                return_value = {"general.architecture": "diffusion-gemma"},
            ):
                # Unpinned: the runner falls back to DG_GPU (else GPU 0), so the
                # guard sizes that single device, not the whole visible pool.
                with patch.dict("os.environ", {"DG_GPU": "1"}):
                    unpinned_dg = self.route._diffusion_guard_gpu_ids(cfg, None)
                unpinned = self.route._diffusion_guard_gpu_ids(cfg, None)
        self.assertEqual(collapsed, [0])
        self.assertEqual(dense, [1, 0])
        self.assertEqual(canvas_only, [0])
        self.assertEqual(unpinned_dg, [1])
        self.assertEqual(unpinned, [0])
        # A single pick passes through without a header read; a non-diffusion
        # unpinned load keeps the aggregate (file here isn't a real GGUF).
        self.assertEqual(self.route._diffusion_guard_gpu_ids(cfg, [1]), [1])
        self.assertIsNone(self.route._diffusion_guard_gpu_ids(cfg, None))

    def test_cpu_forced_drafter_not_charged(self):
        # --spec-draft-ngl 0 keeps the drafter off the GPU (the loader's own
        # budget skips it too), so the guard must not charge it or deny an HF
        # form -- else a valid CPU-only load is rejected for VRAM never used.
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.gguf"
            p.write_bytes(b"x" * 1000)
            draft = Path(d) / "draft.gguf"
            draft.write_bytes(b"y" * 500)
            cfg = self._local_gguf_cfg(p)
            with (
                patch.object(self.route, "_estimate_gguf_kv_gb", return_value = 0.0),
                patch.object(self.route, "_manual_gpu_layer_fraction", return_value = 0.0),
            ):
                kwargs = dict(gpu_memory_mode = "manual", gpu_layers = 0)
                local_cpu = self.route._estimate_gguf_required_gb(
                    cfg,
                    llama_extra_args = [
                        "--model-draft",
                        str(draft),
                        "--spec-draft-ngl",
                        "0",
                    ],
                    **kwargs,
                )
                hf_cpu = self.route._estimate_gguf_required_gb(
                    cfg,
                    llama_extra_args = ["-hfd", "org/draft", "--spec-draft-ngl", "0"],
                    **kwargs,
                )
        self.assertEqual(local_cpu, 0.0)
        self.assertEqual(hf_cpu, 0.0)

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

    def test_manual_charges_companions_in_full_not_scaled_by_gpu_layers(self):
        # A companion (mmproj / separate MTP drafter) is GPU-resident regardless of
        # the main --gpu-layers, so it must not be scaled by the fraction. At
        # gpu_layers=0 the old code scaled the whole sum to ~0 GB (an OOM risk); now
        # only the main term zeros out.
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            main = Path(d) / "model.gguf"
            main.write_bytes(b"x" * 1000)
            drafter = Path(d) / "mtp-model.gguf"
            drafter.write_bytes(b"y" * 3000)
            cfg = self._local_gguf_cfg(main)
            cfg.gguf_mtp_file = str(drafter)
            with (
                patch.object(self.route, "_estimate_gguf_kv_gb", return_value = 5.0),
                patch.object(self.route, "_manual_gpu_layer_fraction", return_value = 0.0),
            ):
                gb = self.route._estimate_gguf_required_gb(
                    cfg, gpu_memory_mode = "manual", gpu_layers = 0
                )
        # main (weights + KV) scaled to 0; the 3000-byte companion stands in full.
        self.assertAlmostEqual(gb, 3000 / (1024**3), places = 9)
        self.assertGreater(gb, 0.0)  # regression guard: not silently ~0

    def test_manual_scales_cached_hf_repo_by_gpu_layer_fraction(self):
        # A repo-id GGUF that's already in the HF cache is local in all but name
        # (ModelConfig leaves gguf_file unset for repo ids). The manual offload
        # fraction must be credited from the cached header, not left at the full
        # remote size -- else a CPU-heavy load is over-blocked during training.
        # A not-yet-cached repo (resolve returns None) keeps the full size.
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
                cached = self.route._estimate_gguf_required_gb(
                    cfg, gpu_memory_mode = "manual", gpu_layers = 8
                )
            with patch.object(gguf_mod, "resolve_local_gguf_path", return_value = None):
                not_cached = self.route._estimate_gguf_required_gb(
                    cfg, gpu_memory_mode = "manual", gpu_layers = 8
                )
            auto = self.route._estimate_gguf_required_gb(cfg)  # auto never scales
        self.assertAlmostEqual(cached, (8.0 + 2.0) * 0.25, places = 6)  # cached header, scaled
        self.assertAlmostEqual(not_cached, 10.0, places = 6)  # no local header -> full remote
        self.assertAlmostEqual(auto, 10.0, places = 6)  # default never scales

    def test_manual_gpu_layer_fraction_clamps_and_reads_layers(self):
        # _manual_gpu_layer_fraction imports read_gguf_staged_dims lazily, so
        # patch it at its source module.
        import utils.models.gguf_metadata as gm

        frac = self.route._manual_gpu_layer_fraction
        with patch.object(gm, "read_gguf_staged_dims", return_value = {"layer_count": 32}):
            self.assertAlmostEqual(frac("x.gguf", 8), 0.25, places = 9)
            self.assertEqual(frac("x.gguf", 0), 0.0)  # all on CPU
            self.assertEqual(frac("x.gguf", 999), 1.0)  # clamps above layer count
        # Unreadable / non-GGUF (None dims, or a None/zero layer_count) -> None so
        # the caller keeps the full estimate.
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
            # --ctx-size override above max_seq_length -> override wins
            self.assertAlmostEqual(
                r._estimate_gguf_kv_gb("m", 4096, ["--ctx-size", "131072"]), 128.0
            )
            self.assertEqual(seen["ctx"], 131072)
            self.assertEqual(seen["n_parallel"], 1)  # default single slot
            # override below max_seq_length -> larger (max_seq_length) wins
            self.assertAlmostEqual(r._estimate_gguf_kv_gb("m", 4096, ["--ctx-size", "1024"]), 4.0)
            self.assertEqual(seen["ctx"], 4096)
            # no override, no max_seq_length -> native context fallback
            self.assertAlmostEqual(r._estimate_gguf_kv_gb("m", 0, None), 2.0)
            self.assertEqual(seen["ctx"], 2048)
            # malformed extras are ignored (fall back to max_seq_length)
            self.assertAlmostEqual(r._estimate_gguf_kv_gb("m", 4096, ["--ctx-size", "oops"]), 4.0)
            # --parallel slots scale the cache the same way the launcher does
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
