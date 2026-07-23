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
        single_device_gpu = None,
        gpu_ids = None,
    ):
        with (
            patch("utils.hardware.get_device", return_value = DeviceType.CUDA),
            patch("utils.hardware.estimate_required_model_memory_gb", return_value = (estimate, {})),
            patch("utils.hardware.get_visible_gpu_utilization", return_value = {"devices": devices}),
            patch("utils.hardware.resolve_requested_gpu_ids", return_value = gpu_ids),
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
                single_device_gpu = single_device_gpu,
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

    def test_no_per_gpu_floor_for_gguf_with_explicit_gpu_ids(self):
        # gpu_ids narrows llama.cpp's candidate pool but does not turn its
        # self-placement into HF device_map="balanced". The uneven selected
        # pair therefore keeps the aggregate GGUF check without an even-share
        # floor on the nearly-full card.
        ok, info, _ = self._run(
            devices = _devices((0, 80, 35), (1, 80, 70), (2, 80, 0)),
            required_override = 20.0,
            gpu_ids = [0, 1],
        )
        self.assertTrue(ok)
        self.assertEqual(info["mode"], "gguf")

    def test_single_device_uses_selected_gpu(self):
        # The model needs 27 GB with headroom. GPU 0 has 45 GB free, while an
        # unrelated training-heavy GPU 1 has only 10 GB free.
        ok, info, _ = self._run(
            devices = _devices((0, 80, 35), (1, 80, 70)),
            required_override = 20.0,
            single_device_gpu = "0",
        )
        self.assertTrue(ok)
        self.assertEqual(info["usable_gb"], 45.0)

        blocked, blocked_info, _ = self._run(
            devices = _devices((0, 80, 35), (1, 80, 70)),
            required_override = 20.0,
            single_device_gpu = "1",
        )
        self.assertFalse(blocked)
        self.assertEqual(blocked_info["usable_gb"], 10.0)

    def test_single_device_unresolved_token_sizes_against_worst_device(self):
        # A non-numeric device token (a CUDA UUID / MIG handle) can't map to a
        # free-VRAM index. The runner still drives ONE device, so size against the
        # worst-case visible device (min free), not the aggregate pool: one GPU
        # with 80 GB free vs a 20 GB model -> allow.
        ok, info, _ = self._run(
            devices = _devices((0, 80, 0)),
            required_override = 20.0,
            single_device_gpu = "GPU-uuid",
        )
        self.assertTrue(ok)
        self.assertEqual(info["mode"], "single_device")
        self.assertNotIn("reason", info)

    def test_single_device_unresolved_token_refuses_when_worst_device_full(self):
        # Same UUID fallback, worst-case device nearly full (2 GB for a 20 GB
        # model) -> refuse (default-deny), not on an unresolved-token technicality.
        ok, info, _ = self._run(
            devices = _devices((0, 80, 78)),
            required_override = 20.0,
            single_device_gpu = "GPU-uuid",
        )
        self.assertFalse(ok)
        self.assertNotEqual(info.get("reason"), "unresolved_gpu_id")

    def test_single_device_unresolved_token_uses_min_free_not_aggregate(self):
        # The single-device runner uses ONE device but we can't tell which from a
        # UUID token. Sizing against the aggregate pool would let a 20 GB model
        # "fit" 160 GB of pooled free VRAM while landing on a 2 GB card and OOMing
        # training. Min-free (2 GB) is the safe worst case -> refuse.
        ok, info, _ = self._run(
            devices = _devices((0, 80, 78), (1, 80, 0), (2, 80, 0)),
            required_override = 20.0,
            single_device_gpu = "GPU-uuid",
        )
        self.assertFalse(ok)
        self.assertEqual(info["mode"], "single_device")

    def test_single_device_cpu_token_allows(self):
        # An empty device token = a CPU-only single-device runner (CPU diffusion
        # GGUF): it uses no GPU VRAM, so it never threatens training -> allow
        # regardless of how full the GPUs are.
        ok, info, _ = self._run(
            devices = _devices((0, 80, 78)),
            required_override = 20.0,
            single_device_gpu = "",
        )
        self.assertTrue(ok)
        self.assertEqual(info["reason"], "cpu_only")

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
        gpu_memory_mode = "auto",
        requested_gpu_ids = None,
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
                requested_gpu_ids = requested_gpu_ids,
                gpu_memory_mode = gpu_memory_mode,
            )

    def test_noop_when_training_inactive(self):
        self._guard(training_active = False, decision = (False, {}))  # must not raise

    def test_noop_when_training_state_unknown(self):
        self._guard(training_active = RuntimeError("no backend"), decision = (False, {}))

    def test_allows_when_fits(self):
        self._guard(training_active = True, decision = (True, {"mode": "auto"}))

    def test_diffusion_detection_uses_name_before_download(self):
        config = SimpleNamespace(
            identifier = "unsloth/DiffusionGemma-GGUF",
            gguf_hf_repo = "unsloth/DiffusionGemma-GGUF",
            gguf_file = None,
        )
        self.assertTrue(self.route._classify_diffusion_gguf(config))

    def test_uncached_gguf_classification_remains_unknown(self):
        config = SimpleNamespace(
            identifier = "owner/renamed-model",
            gguf_hf_repo = "owner/renamed-model",
            gguf_variant = "Q4_K_M",
            gguf_file = None,
        )
        self.assertIsNone(self.route._classify_diffusion_gguf(config))

    def test_diffusion_detection_reuses_loader_metadata_probe(self):
        import tempfile

        seen = []

        class _Probe:
            is_diffusion = False
            _architecture = None

            def _read_gguf_metadata(self, path):
                seen.append(path)
                self.is_diffusion = True

        with tempfile.TemporaryDirectory() as d:
            model = Path(d) / "renamed.gguf"
            model.write_bytes(b"GGUF")
            config = SimpleNamespace(identifier = "local", gguf_file = str(model))
            with patch.object(self.route, "LlamaCppBackend", _Probe):
                self.assertTrue(self.route._classify_diffusion_gguf(config))
        self.assertEqual(seen, [str(model)])

    def test_local_chat_gguf_classification_is_definitive(self):
        import tempfile
        class _Probe:
            is_diffusion = False
            _architecture = "llama"

            def _read_gguf_metadata(self, _path):
                pass

        with tempfile.TemporaryDirectory() as d:
            model = Path(d) / "renamed.gguf"
            model.write_bytes(b"GGUF")
            config = SimpleNamespace(identifier = "local", gguf_file = str(model))
            with patch.object(self.route, "LlamaCppBackend", _Probe):
                self.assertFalse(self.route._classify_diffusion_gguf(config))

    def test_manual_known_normal_gguf_bypasses_training_estimate(self):
        captured = []
        config = SimpleNamespace(is_gguf = True)
        with patch.object(self.route, "_classify_diffusion_gguf", return_value = False):
            self._guard(
                config = config,
                captured = captured,
                training_active = True,
                decision = (False, {"reason": "must not run"}),
                gpu_memory_mode = "manual",
            )
        self.assertEqual(captured, [])

    def test_manual_unknown_gguf_keeps_single_device_training_guard(self):
        captured = []
        config = SimpleNamespace(is_gguf = True)
        with (
            patch.object(self.route, "_classify_diffusion_gguf", return_value = None),
            patch.object(self.route, "_estimate_gguf_required_gb", return_value = 12.5),
            patch.object(
                self.route.LlamaCppBackend,
                "_diffusion_gpu_arg",
                return_value = "2",
            ),
        ):
            self._guard(
                config = config,
                captured = captured,
                training_active = True,
                decision = (True, {"mode": "single_device"}),
                gpu_memory_mode = "manual",
            )
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["single_device_gpu"], "2")

    def test_manual_diffusion_uses_single_device_guard(self):
        captured = []
        config = SimpleNamespace(is_gguf = True)
        with (
            patch.object(self.route, "_classify_diffusion_gguf", return_value = True),
            patch.object(self.route, "_estimate_gguf_required_gb", return_value = 12.5),
        ):
            self._guard(
                config = config,
                captured = captured,
                training_active = True,
                decision = (True, {"mode": "gguf"}),
                gpu_memory_mode = "manual",
                requested_gpu_ids = [3, 1],
            )
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["single_device_gpu"], "1")
        self.assertEqual(captured[0]["requested_gpu_ids"], [3, 1])

    def test_unpinned_diffusion_uses_runner_default_gpu(self):
        captured = []
        config = SimpleNamespace(is_gguf = True)
        with (
            patch.object(self.route, "_classify_diffusion_gguf", return_value = True),
            patch.object(self.route, "_estimate_gguf_required_gb", return_value = 12.5),
            patch.object(
                self.route.LlamaCppBackend,
                "_effective_gpu_count",
                return_value = 2,
            ),
            patch.object(
                self.route.LlamaCppBackend,
                "_diffusion_gpu_arg",
                return_value = "3",
            ) as gpu_arg,
        ):
            self._guard(
                config = config,
                captured = captured,
                training_active = True,
                decision = (True, {"mode": "single_device"}),
                gpu_memory_mode = "manual",
            )
        gpu_arg.assert_called_once_with(None, cpu_only = False)
        self.assertEqual(captured[0]["single_device_gpu"], "3")

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

    def test_validate_forwards_manual_gpu_memory_mode_to_guard(self):
        from models.inference import ValidateModelRequest

        request = ValidateModelRequest(
            model_path = "unsloth/model-GGUF",
            gguf_variant = "Q4_K_M",
            gpu_memory_mode = "manual",
        )
        cfg = SimpleNamespace(
            identifier = "unsloth/model-GGUF",
            display_name = "model-GGUF",
            is_gguf = True,
            is_lora = False,
            is_vision = False,
            path = None,
            base_model = None,
        )
        captured = {}
        with (
            patch.object(
                self.route,
                "_resolve_model_identifier_for_request",
                return_value = ("unsloth/model-GGUF", "unsloth/model-GGUF", False),
            ),
            patch.object(self.route.ModelConfig, "from_identifier", return_value = cfg),
            patch.object(self.route, "load_inference_config", return_value = {}),
            patch.object(
                self.route,
                "_guard_chat_load_against_training",
                lambda config, **kw: captured.update(kw),
            ),
        ):
            asyncio.run(self.route.validate_model(request, current_subject = "u"))
        self.assertEqual(captured.get("gpu_memory_mode"), "manual")

    def test_validate_forwards_inherited_extras_and_parallel_to_guard(self):
        # Regression: /load resolves inherited same-model extras and passes the
        # real slot count to the guard; validate must do the same, else it sizes
        # a smaller estimate (no inherited -c/--model-draft, n_parallel=1) and
        # /load then 409s after the frontend has already unloaded.
        from models.inference import ValidateModelRequest

        request = ValidateModelRequest(model_path = "unsloth/Qwen3-1.7B", max_seq_length = 4096)
        cfg = SimpleNamespace(
            identifier = "unsloth/Qwen3-1.7B",
            display_name = "Qwen3-1.7B",
            is_gguf = False,
            is_lora = False,
            is_vision = False,
            path = None,
            base_model = None,
        )
        captured = {}
        with (
            patch.object(
                self.route,
                "_resolve_model_identifier_for_request",
                return_value = ("unsloth/Qwen3-1.7B", "unsloth/Qwen3-1.7B", False),
            ),
            patch.object(self.route.ModelConfig, "from_identifier", return_value = cfg),
            patch.object(self.route, "load_inference_config", return_value = {}),
            patch.object(self.route, "_resolve_inherited_extra_args", return_value = ["-c", "32768"]),
            patch.object(
                self.route,
                "_guard_chat_load_against_training",
                lambda config, **kw: captured.update(kw),
            ),
        ):
            asyncio.run(self.route.validate_model(request, current_subject = "u"))
        self.assertEqual(captured.get("llama_extra_args"), ["-c", "32768"])
        self.assertIn("n_parallel", captured)

    def test_validate_uses_effective_inherited_mmproj_state(self):
        from models.inference import ValidateModelRequest

        request = ValidateModelRequest(
            model_path = "unsloth/model-GGUF",
            gguf_variant = "Q4_K_M",
            load_mmproj = True,
        )
        cfg = SimpleNamespace(
            identifier = "unsloth/model-GGUF",
            display_name = "model-GGUF",
            is_gguf = True,
            is_lora = False,
            is_vision = True,
            path = None,
            base_model = None,
        )
        captured = {}
        with (
            patch.object(
                self.route,
                "_resolve_model_identifier_for_request",
                return_value = ("unsloth/model-GGUF", "unsloth/model-GGUF", False),
            ),
            patch.object(self.route.ModelConfig, "from_identifier", return_value = cfg),
            patch.object(self.route, "load_inference_config", return_value = {}),
            patch.object(
                self.route,
                "_resolve_inherited_extra_args",
                return_value = ["--no-mmproj"],
            ),
            patch.object(
                self.route,
                "_guard_chat_load_against_training",
                lambda config, **kw: captured.update(kw),
            ),
        ):
            asyncio.run(self.route.validate_model(request, current_subject = "u"))
        self.assertFalse(captured["include_mmproj"])

    def test_metadata_probe_skips_training_guard(self):
        # A header-only probe (include_context_length) allocates no VRAM, so the
        # training guard must not run -- else the staging GPU-layers / MoE sliders
        # it feeds are hidden exactly when a during-training user needs them.
        from models.inference import ValidateModelRequest

        request = ValidateModelRequest(
            model_path = "unsloth/Qwen3-1.7B",
            max_seq_length = 4096,
            include_context_length = True,
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
        guard_called = []
        with (
            patch.object(
                self.route,
                "_resolve_model_identifier_for_request",
                return_value = ("unsloth/Qwen3-1.7B", "unsloth/Qwen3-1.7B", False),
            ),
            patch.object(self.route.ModelConfig, "from_identifier", return_value = cfg),
            patch.object(self.route, "load_inference_config", return_value = {}),
            patch.object(
                self.route,
                "_guard_chat_load_against_training",
                lambda *a, **kw: guard_called.append(True),
            ),
        ):
            asyncio.run(self.route.validate_model(request, current_subject = "u"))
        self.assertEqual(guard_called, [])

    def _validate_gguf_template(
        self,
        *,
        template,
        canonical_path = "/picked/model.gguf",
    ):
        # Drive validate_model for a native lease-backed GGUF template probe and
        # capture what the embedded-template reader was called with.
        from models.inference import ValidateModelRequest

        request = ValidateModelRequest(
            model_path = "model.gguf",
            gguf_variant = "Q4_K_M",
            native_path_lease = "signed-lease",
            include_chat_template = True,
        )
        cfg = SimpleNamespace(
            identifier = canonical_path,
            display_name = "model.gguf",
            is_gguf = True,
            is_lora = False,
            is_vision = False,
            gguf_file = canonical_path,
            path = None,
            base_model = None,
        )
        import utils.models.gguf_metadata as gguf_meta

        seen = {}

        def _fake_read(path):
            seen["path"] = path
            return template

        guard_called = []
        with (
            patch.object(
                self.route,
                "_resolve_model_identifier_for_request",
                return_value = (canonical_path, "model.gguf", True),
            ),
            patch.object(self.route.ModelConfig, "from_identifier", return_value = cfg),
            patch.object(self.route, "load_inference_config", return_value = {}),
            patch.object(gguf_meta, "read_gguf_chat_template", _fake_read),
            patch.object(
                self.route,
                "_guard_chat_load_against_training",
                lambda *a, **kw: guard_called.append(True),
            ),
        ):
            resp = asyncio.run(self.route.validate_model(request, current_subject = "u"))
        return resp, seen, guard_called

    def test_include_chat_template_reads_leased_gguf_embedded_template(self):
        # The picker chat-template GET has no lease plumbing, so a native picked
        # GGUF surfaces its default template through this lease-aware probe: the
        # embedded template is read from the granted canonical path and returned.
        resp, seen, _ = self._validate_gguf_template(template = "{{ messages }}")
        self.assertEqual(resp.chat_template, "{{ messages }}")
        # Read strictly the leased file's own embedded template, never a sibling
        # sidecar: the grant authorizes just this one path.
        self.assertEqual(seen["path"], "/picked/model.gguf")

    def test_include_chat_template_skips_training_guard(self):
        # A template-only probe allocates no VRAM, so like include_context_length
        # it must not be refused by the training guard.
        _, _, guard_called = self._validate_gguf_template(template = "{{ messages }}")
        self.assertEqual(guard_called, [])

    def test_include_chat_template_over_cap_is_dropped(self):
        from picker.schemas import MAX_CHAT_TEMPLATE_BYTES
        resp, _, _ = self._validate_gguf_template(template = "a" * (MAX_CHAT_TEMPLATE_BYTES + 1))
        self.assertIsNone(resp.chat_template)


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

    def test_local_excludes_projector_but_keeps_other_companions(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            model = p / "model.gguf"
            projector = p / "mmproj.gguf"
            drafter = p / "mtp.gguf"
            model.write_bytes(b"m" * 1000)
            projector.write_bytes(b"p" * 2000)
            drafter.write_bytes(b"d" * 3000)
            cfg = SimpleNamespace(
                gguf_file = str(model),
                gguf_mmproj_file = str(projector),
                gguf_mtp_file = str(drafter),
                gguf_hf_repo = None,
                gguf_variant = None,
            )
            gb = self.route._estimate_gguf_required_gb(
                cfg,
                include_mmproj = False,
            )
        self.assertAlmostEqual(gb, 4000 / (1024**3), places = 9)

    def test_remote_excludes_projector_from_companion_query(self):
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
            patch.object(
                self.route,
                "_remote_gguf_companion_bytes",
                return_value = 1 * 1024**3,
            ) as comp,
        ):
            gb = self.route._estimate_gguf_required_gb(
                cfg,
                include_mmproj = False,
            )
        self.assertAlmostEqual(gb, 11.0, places = 6)
        self.assertFalse(comp.call_args.kwargs["include_mmproj"])

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
        cfg = SimpleNamespace(
            is_gguf = False,
            is_lora = False,
            path = None,
            base_model = None,
            identifier = "unsloth/Qwen3-1.7B",
        )
        request = LoadRequest(model_path = "unsloth/Qwen3-1.7B")
        info = {"required_gb": 40.0, "usable_gb": 5.0, "needed_gb": 50.0, "mode": "auto"}

        with (
            # Pin the latest-sidecar tier check so the guard path stays offline.
            patch("utils.transformers_version.latest_tier_active_for", return_value = False),
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

    def test_load_guard_uses_effective_inherited_mmproj_state(self):
        import contextlib
        from unittest.mock import MagicMock
        from models.inference import LoadRequest

        inf = SimpleNamespace(active_model_name = None)
        llama = SimpleNamespace(is_loaded = False, model_identifier = None, hf_variant = None)
        cfg = SimpleNamespace(
            is_gguf = False,
            is_lora = False,
            path = None,
            base_model = None,
            identifier = "unsloth/Qwen3-1.7B",
        )
        request = LoadRequest(
            model_path = "unsloth/Qwen3-1.7B",
            load_mmproj = True,
        )
        captured = {}

        def stop_after_guard(config, **kwargs):
            captured.update(kwargs)
            raise HTTPException(status_code = 409, detail = "stop")

        with (
            patch("utils.transformers_version.latest_tier_active_for", return_value = False),
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
            patch.object(
                self.route,
                "_resolve_inherited_extra_args",
                return_value = ["--no-mmproj"],
            ),
            patch.object(
                self.route,
                "_guard_chat_load_against_training",
                side_effect = stop_after_guard,
            ),
        ):
            with self.assertRaises(HTTPException):
                asyncio.run(
                    self.route.load_model(
                        request,
                        fastapi_request = MagicMock(),
                        current_subject = "u",
                    )
                )
        self.assertFalse(captured["include_mmproj"])


if __name__ == "__main__":
    unittest.main()
