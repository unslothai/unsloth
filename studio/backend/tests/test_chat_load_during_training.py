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
        # Aggregate capacity passes, but the 10 GB shard fails its 13.5 GB floor.
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
        gpu_ids_are_vulkan_ordinals = False,
        vulkan_free_vram_gb = None,
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
                gpu_ids_are_vulkan_ordinals = gpu_ids_are_vulkan_ordinals,
                vulkan_free_vram_gb = vulkan_free_vram_gb,
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
        # llama.cpp self-placement uses aggregate capacity within the pinned pool.
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

    def test_vulkan_pin_takes_precedence_over_unknown_diffusion_fallback(self):
        # Use the Vulkan probe, not unrelated CUDA telemetry or a speculative
        # CUDA diffusion device.
        ok, info, _ = self._run(
            devices = _devices((0, 80, 0)),
            required_override = 20.0,
            single_device_gpu = "0",
            gpu_ids = [0],
            gpu_ids_are_vulkan_ordinals = True,
            vulkan_free_vram_gb = {0: 2.0, 1: 80.0},
        )
        self.assertFalse(ok)
        self.assertEqual(info["mode"], "gguf_vulkan")
        self.assertEqual(info["usable_gb"], 2.0)

    def test_vulkan_multi_gpu_guard_counts_requested_devices(self):
        # Vulkan ordinals select the same entries reported by the Vulkan probe.
        ok, info, _ = self._run(
            devices = _devices((0, 80, 0)),
            required_override = 10.0,
            gpu_ids = [0, 1],
            gpu_ids_are_vulkan_ordinals = True,
            vulkan_free_vram_gb = {0: 10.0, 1: 10.0, 2: 80.0},
        )
        self.assertTrue(ok)
        self.assertEqual(info["mode"], "gguf_vulkan")
        self.assertEqual(info["usable_gb"], 18.5)

    def test_vulkan_uneven_pool_uses_fitting_subset(self):
        # The loader may choose the 20 GB device alone from this candidate pool.
        ok, info, _ = self._run(
            devices = _devices((0, 80, 0), (1, 80, 0)),
            required_override = 10.0,
            gpu_ids = [0, 1],
            gpu_ids_are_vulkan_ordinals = True,
            vulkan_free_vram_gb = {0: 20.0, 1: 2.0},
        )
        self.assertTrue(ok)
        self.assertNotIn("per_gpu_needed_gb", info)

    def test_vulkan_uneven_pool_is_order_independent(self):
        ok, _, _ = self._run(
            devices = _devices((0, 80, 0), (1, 80, 0)),
            required_override = 10.0,
            gpu_ids = [1, 0],
            gpu_ids_are_vulkan_ordinals = True,
            vulkan_free_vram_gb = {0: 20.0, 1: 2.0},
        )
        self.assertTrue(ok)

    def test_vulkan_pool_with_insufficient_aggregate_refuses(self):
        ok, _, _ = self._run(
            devices = _devices((0, 80, 0), (1, 80, 0)),
            required_override = 10.0,
            gpu_ids = [0, 1],
            gpu_ids_are_vulkan_ordinals = True,
            vulkan_free_vram_gb = {0: 8.0, 1: 2.0},
        )
        self.assertFalse(ok)

    def test_vulkan_auto_uses_full_vulkan_pool(self):
        ok, info, _ = self._run(
            devices = _devices((0, 80, 79)),
            required_override = 20.0,
            vulkan_free_vram_gb = {0: 30.0, 1: 30.0},
        )
        self.assertTrue(ok)
        self.assertEqual(info["mode"], "gguf_vulkan")
        self.assertEqual(info["usable_gb"], 55.5)

    def test_single_device_unresolved_token_sizes_against_worst_device(self):
        # Unknown single-device tokens use the least-free visible GPU.
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
        # Aggregate capacity cannot justify an unresolved single-device load.
        ok, info, _ = self._run(
            devices = _devices((0, 80, 78), (1, 80, 0), (2, 80, 0)),
            required_override = 20.0,
            single_device_gpu = "GPU-uuid",
        )
        self.assertFalse(ok)
        self.assertEqual(info["mode"], "single_device")

    def test_single_device_cpu_token_allows(self):
        # An empty device token is CPU-only and consumes no VRAM.
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
    def test_non_accelerator_allows(self):
        with patch("utils.hardware.get_device", return_value = DeviceType.MLX):
            ok, info = tv.can_load_chat_during_training(
                model_name = "m",
                hf_token = None,
                load_in_4bit = True,
                max_seq_length = 0,
                requested_gpu_ids = None,
            )
        self.assertTrue(ok)
        self.assertEqual(info["mode"], "non_accelerator")

    def test_xpu_overcommit_is_refused(self):
        # XPU must NOT get the blanket non-accelerator allow: an oversized
        # chat model during resident training is refused, like CUDA.
        with (
            patch("utils.hardware.get_device", return_value = DeviceType.XPU),
            patch(
                "utils.hardware.auto_select_gpu_ids",
                return_value = (
                    None,
                    {"selection_mode": "auto", "required_gb": 50.0, "usable_gb": 4.0},
                ),
            ),
        ):
            ok, info = tv.can_load_chat_during_training(
                model_name = "m",
                hf_token = None,
                load_in_4bit = True,
                max_seq_length = 0,
                requested_gpu_ids = None,
            )
        self.assertFalse(ok)
        self.assertNotEqual(info.get("mode"), "non_accelerator")

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
        gpu_ids_are_vulkan_ordinals = False,
        llama_extra_args = None,
        cache_type_kv = None,
        tensor_parallel = False,
        gpu_layers = -1,
    ):
        config = config or SimpleNamespace(is_gguf = False, is_lora = False, path = None)
        placement = self.route._LoadPlacement(
            requested_gpu_ids,
            requested_gpu_ids,
            gpu_ids_are_vulkan_ordinals,
            self.route._classify_diffusion_gguf(config) if config.is_gguf else False,
        )
        with _stub_guard_deps(
            training_active = training_active, decision = decision, captured = captured
        ):
            request = SimpleNamespace(
                model_path = "unsloth/Qwen3-1.7B",
                hf_token = None,
                max_seq_length = 0,
                cache_type_kv = cache_type_kv,
                tensor_parallel = tensor_parallel,
                gpu_memory_mode = gpu_memory_mode,
                gpu_layers = gpu_layers,
            )
            self.route._guard_chat_load_against_training(
                config,
                request,
                load_in_4bit = True,
                placement = placement,
                llama_extra_args = llama_extra_args,
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
        with patch.object(self.route, "_classify_diffusion_gguf", return_value = False) as classify:
            self._guard(
                config = config,
                captured = captured,
                training_active = True,
                decision = (False, {"reason": "must not run"}),
                gpu_memory_mode = "manual",
                requested_gpu_ids = [1, 3],
            )
        classify.assert_called_once_with(config)
        self.assertEqual(captured, [])

    def test_manual_diffusion_keeps_single_device_training_guard(self):
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
        ):
            self._guard(
                config = config,
                captured = captured,
                training_active = True,
                decision = (True, {"mode": "single_device"}),
                gpu_memory_mode = "manual",
                requested_gpu_ids = [3, 1],
            )
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["single_device_gpu"], "1")
        self.assertEqual(captured[0]["requested_gpu_ids"], [3, 1])

    def _guard_zero_layer(self, *, diffusion_kind, captured):
        """Drive the guard with a manual zero-layer split and a shim that has --ngl."""
        config = SimpleNamespace(is_gguf = True)
        backend = SimpleNamespace(diffusion_split_supported = lambda: True)
        with (
            patch.object(self.route, "_classify_diffusion_gguf", return_value = diffusion_kind),
            patch.object(self.route, "get_llama_cpp_backend", return_value = backend),
            patch.object(self.route, "_estimate_gguf_required_gb", return_value = 12.5),
            patch.object(self.route.LlamaCppBackend, "_effective_gpu_count", return_value = 2),
        ):
            self._guard(
                config = config,
                captured = captured,
                training_active = True,
                decision = (False, {"reason": "must not run"}),
                gpu_memory_mode = "manual",
                gpu_layers = 0,
            )

    def test_zero_layer_diffusion_split_bypasses_the_training_guard(self):
        """A confirmed DiffusionGemma at ngl 0 places no layers, so it is not refused."""
        captured = []
        self._guard_zero_layer(diffusion_kind = True, captured = captured)
        self.assertEqual(captured, [])

    def test_zero_layer_unclassified_gguf_still_hits_the_training_guard(self):
        """An unreadable header is not a diffusion promise: --gpu-layers 0 on an
        ordinary GGUF can still hold VRAM, so the estimate must run."""
        captured = []
        with self.assertRaises(HTTPException) as ctx:
            self._guard_zero_layer(diffusion_kind = None, captured = captured)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(len(captured), 1)

    def test_zero_layer_unclassified_gguf_is_not_sized_as_cpu_only(self):
        """Reaching the guard is not enough: it must not be handed a CPU-only token.

        can_load_chat_during_training short-circuits an EMPTY single_device_gpu to
        "cpu_only" and always returns True, so an unclassified GGUF passed through with
        force_cpu would be allowed during training on an assumption that only holds for
        confirmed diffusion. The sibling test above stubs can_load, so it cannot see this.
        """
        captured = []
        with self.assertRaises(HTTPException):
            self._guard_zero_layer(diffusion_kind = None, captured = captured)
        self.assertEqual(len(captured), 1)
        token = captured[0].get("single_device_gpu")
        self.assertTrue(
            token is None or str(token).strip() != "",
            "an unclassified zero-layer GGUF must not be budgeted as CPU-only; "
            f"single_device_gpu was {token!r}, which training_vram reads as cpu_only",
        )

    def test_unclassified_gguf_on_vulkan_build_budgets_as_ordinals(self):
        # Unknown GGUFs still use the Vulkan ordinal namespace selected by the build.
        captured = []
        config = SimpleNamespace(is_gguf = True)
        with (
            patch.object(self.route, "_classify_diffusion_gguf", return_value = None),
            patch.object(self.route, "_estimate_gguf_required_gb", return_value = 12.5),
            patch.object(
                self.route.LlamaCppBackend,
                "_get_gpu_memory",
                return_value = [(0, 2048, 8192), (1, 4096, 8192)],
            ),
        ):
            self._guard(
                config = config,
                captured = captured,
                training_active = True,
                decision = (True, {"mode": "gguf_vulkan"}),
                requested_gpu_ids = [0, 1],
                gpu_ids_are_vulkan_ordinals = True,
            )
        self.assertEqual(len(captured), 1)
        self.assertTrue(captured[0]["gpu_ids_are_vulkan_ordinals"])
        self.assertEqual(captured[0]["vulkan_free_vram_gb"], {0: 2.0, 1: 4.0})
        self.assertIsNone(captured[0]["single_device_gpu"])

    def test_auto_vulkan_uses_vulkan_probe_for_training_budget(self):
        captured = []
        config = SimpleNamespace(is_gguf = True)
        with (
            patch.object(self.route, "_classify_diffusion_gguf", return_value = False),
            patch.object(self.route, "_estimate_gguf_required_gb", return_value = 12.5),
            patch.object(
                self.route.LlamaCppBackend,
                "_find_llama_server_binary",
                return_value = "/tmp/llama-server",
            ),
            patch.object(self.route.LlamaCppBackend, "_is_vulkan_backend", return_value = True),
            patch.object(
                self.route.LlamaCppBackend,
                "_get_gpu_memory",
                return_value = [(0, 3072, 0), (1, 2048, 8192)],
            ),
        ):
            self._guard(
                config = config,
                captured = captured,
                training_active = True,
                decision = (True, {"mode": "gguf_vulkan"}),
            )
        self.assertEqual(captured[0]["vulkan_free_vram_gb"], {1: 2.0})

    def test_unclassified_gguf_budget_depends_on_the_pin(self):
        # An uncached remote GGUF that neither the header nor the name can classify
        # may still turn out to be the CUDA-only diffusion runner, so the Vulkan
        # free-VRAM map cannot stand in for it. The pin decides what that means:
        cases = (
            # No gpu_ids -> no ordinal to mis-map, so fall back to the torch view
            # (None) rather than an empty map, which would read as "no free VRAM
            # anywhere" and 409 every such load during training.
            (None, "single_device", None),
            # An explicit pin is the opposite case: the ordinal belongs to exactly
            # one of the two device namespaces and neither can stand in for the
            # other, so the guard must fail closed with an empty map.
            ([1], "gguf_vulkan", {}),
        )
        for requested_gpu_ids, mode, expected in cases:
            with self.subTest(requested_gpu_ids = requested_gpu_ids):
                captured = []
                config = SimpleNamespace(is_gguf = True)
                with (
                    patch.object(self.route, "_classify_diffusion_gguf", return_value = None),
                    patch.object(self.route, "_estimate_gguf_required_gb", return_value = 12.5),
                    patch.object(
                        self.route.LlamaCppBackend,
                        "_find_llama_server_binary",
                        return_value = "/tmp/llama-server",
                    ),
                    patch.object(
                        self.route.LlamaCppBackend, "_is_vulkan_backend", return_value = True
                    ),
                ):
                    self._guard(
                        config = config,
                        captured = captured,
                        training_active = True,
                        decision = (True, {"mode": mode}),
                        requested_gpu_ids = requested_gpu_ids,
                    )
                self.assertEqual(len(captured), 1)
                self.assertEqual(captured[0]["vulkan_free_vram_gb"], expected)

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
        config = SimpleNamespace(identifier = "unsloth/Canonical-Repo", is_gguf = True)
        with patch.object(self.route, "_estimate_gguf_required_gb", return_value = 12.5):
            self._guard(
                config = config,
                captured = captured,
                training_active = True,
                decision = (True, {}),
            )
        self.assertEqual(captured[0]["is_gguf"], True)
        self.assertEqual(captured[0]["required_override_gb"], 12.5)
        self.assertEqual(captured[0]["model_name"], config.identifier)

    def test_vulkan_gguf_estimate_keeps_tensor_cache_coercion(self):
        config = SimpleNamespace(is_gguf = True)
        estimate_kwargs = {}
        with (
            patch.object(
                self.route,
                "_estimate_gguf_required_gb",
                side_effect = lambda *args, **kwargs: estimate_kwargs.update(kwargs) or 12.5,
            ),
            patch.object(
                self.route.LlamaCppBackend,
                "_effective_gpu_count",
                return_value = 0,
            ),
            patch.object(
                self.route.LlamaCppBackend,
                "_find_llama_server_binary",
                return_value = "/fake/llama-server",
            ),
            patch.object(self.route.LlamaCppBackend, "_is_vulkan_backend", return_value = True),
        ):
            self._guard(
                config = config,
                training_active = True,
                decision = (True, {}),
                llama_extra_args = ["--split-mode", "tensor"],
                cache_type_kv = "q4_0",
            )
        self.assertEqual(estimate_kwargs["cache_type_kv"], "q4_0")
        self.assertTrue(estimate_kwargs["tensor_parallel"])


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
                lambda config, request, **kw: captured.update(request = request, **kw),
            ),
        ):
            asyncio.run(self.route.validate_model(request, current_subject = "u"))
        self.assertEqual(captured["request"].gpu_memory_mode, "manual")

    def test_validate_forwards_inherited_extras_and_parallel_to_guard(self):
        # Validate and load must size the same inherited command.
        from models.inference import ValidateModelRequest

        request = ValidateModelRequest(
            model_path = "unsloth/Qwen3-1.7B",
            max_seq_length = 4096,
            cache_type_kv = "f32",
            tensor_parallel = True,
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
                lambda config, request, **kw: captured.update(request = request, **kw),
            ),
        ):
            asyncio.run(self.route.validate_model(request, current_subject = "u"))
        self.assertEqual(captured.get("llama_extra_args"), ["-c", "32768"])
        self.assertIn("n_parallel", captured)
        self.assertEqual(captured["request"].cache_type_kv, "f32")
        self.assertTrue(captured["request"].tensor_parallel)

    def test_metadata_probe_skips_training_guard(self):
        # Header-only probes allocate no VRAM.
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

    def test_metadata_probe_reports_diffusion(self):
        from models.inference import ValidateModelRequest

        request = ValidateModelRequest(
            model_path = "unsloth/DiffusionGemma-GGUF",
            include_context_length = True,
        )
        cfg = SimpleNamespace(
            identifier = "unsloth/DiffusionGemma-GGUF",
            display_name = "DiffusionGemma-GGUF",
            is_gguf = True,
            is_lora = False,
            is_vision = False,
            gguf_file = None,
            gguf_hf_repo = "unsloth/DiffusionGemma-GGUF",
            gguf_variant = None,
            path = None,
            base_model = None,
        )
        with (
            patch.object(
                self.route,
                "_resolve_model_identifier_for_request",
                return_value = (
                    "unsloth/DiffusionGemma-GGUF",
                    "unsloth/DiffusionGemma-GGUF",
                    False,
                ),
            ),
            patch.object(self.route.ModelConfig, "from_identifier", return_value = cfg),
            patch.object(self.route, "load_inference_config", return_value = {}),
        ):
            response = asyncio.run(self.route.validate_model(request, current_subject = "u"))
        self.assertTrue(response.is_diffusion)

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

    @staticmethod
    def _dspark_capable(supported = True):
        """The sizing gate asks the binary whether it can run draft-dspark, so the
        probe must be stubbed or these assertions track the host's llama.cpp."""
        from core.inference.llama_cpp import LlamaCppBackend
        return patch.object(
            LlamaCppBackend,
            "probe_server_capabilities",
            classmethod(lambda cls, binary = None: {"supports_dspark": supported}),
        )

    def test_local_dspark_sidecar_is_only_counted_when_requested(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            target = p / "model.gguf"
            sidecar = p / "dspark-model-Q8_0.gguf"
            target.write_bytes(b"x" * 2000)
            sidecar.write_bytes(b"y" * 3000)
            cfg = SimpleNamespace(
                gguf_file = str(target),
                gguf_mmproj_file = None,
                gguf_mtp_file = None,
                gguf_dspark_file = str(sidecar),
                gguf_hf_repo = None,
                gguf_variant = None,
            )
            with (
                patch.object(self.route, "_estimate_gguf_kv_gb", return_value = 0.0),
                self._dspark_capable(),
            ):
                off_gb = self.route._estimate_gguf_required_gb(
                    cfg,
                    speculative_type = "off",
                )
                dspark_gb = self.route._estimate_gguf_required_gb(
                    cfg,
                    speculative_type = "dspark",
                )
                extras_gb = self.route._estimate_gguf_required_gb(
                    cfg,
                    speculative_type = "off",
                    llama_extra_args = ["--spec-type", "draft-dspark"],
                )
        self.assertAlmostEqual(off_gb, 2000 / (1024**3), places = 9)
        self.assertAlmostEqual(dspark_gb, 5000 / (1024**3), places = 9)
        self.assertAlmostEqual(extras_gb, 5000 / (1024**3), places = 9)

    def test_forced_dspark_on_an_incapable_binary_charges_no_drafter_at_all(self):
        """The loader's DSpark branch falls back to --spec-default, which loads no
        drafter, so charging the MTP one would refuse a load that fits. Auto is
        different: it falls through to the MTP branch and keeps that charge."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            target = p / "model.gguf"
            mtp = p / "mtp-model.gguf"
            target.write_bytes(b"x" * 2000)
            mtp.write_bytes(b"y" * 3000)
            cfg = SimpleNamespace(
                gguf_file = str(target),
                gguf_mmproj_file = None,
                gguf_mtp_file = str(mtp),
                gguf_dspark_file = None,
                gguf_hf_repo = None,
                gguf_variant = None,
            )
            with (
                patch.object(self.route, "_estimate_gguf_kv_gb", return_value = 0.0),
                self._dspark_capable(False),
            ):
                forced = self.route._estimate_gguf_required_gb(cfg, speculative_type = "dspark")
                auto = self.route._estimate_gguf_required_gb(cfg, speculative_type = "auto")
        self.assertAlmostEqual(forced, 2000 / (1024**3), places = 9)
        self.assertAlmostEqual(auto, 5000 / (1024**3), places = 9)

    def test_validate_request_carries_the_mode_the_load_will_use(self):
        """The estimate is mode-dependent, so /validate must be told the mode or
        its verdict disagrees with the /load that follows it: a user with
        speculative decoding off and a sidecar on disk would be refused at the
        preflight for a load that would have been admitted."""
        from models.inference import ValidateModelRequest

        req = ValidateModelRequest(
            model_path = "unsloth/DeepSeek-V4-Flash-0731-GGUF",
            speculative_type = "off",
            spec_draft_n_max = 3,
        )
        self.assertEqual(req.speculative_type, "off")
        self.assertEqual(req.spec_draft_n_max, 3)
        # Omitted stays None rather than defaulting to a mode, so the estimate
        # keeps its previous behaviour for callers that do not send it.
        self.assertIsNone(ValidateModelRequest(model_path = "org/repo").speculative_type)

    def test_dspark_sidecar_is_not_charged_to_a_binary_that_cannot_run_it(self):
        """The loader skips the ~11 GB fetch when llama.cpp has no usable
        draft-dspark, so charging it here would refuse a load that never opens it
        and would evict nothing."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            target = p / "model.gguf"
            sidecar = p / "dspark-model-Q8_0.gguf"
            target.write_bytes(b"x" * 2000)
            sidecar.write_bytes(b"y" * 3000)
            cfg = SimpleNamespace(
                gguf_file = str(target),
                gguf_mmproj_file = None,
                gguf_mtp_file = None,
                gguf_dspark_file = str(sidecar),
                gguf_hf_repo = None,
                gguf_variant = None,
            )
            with patch.object(self.route, "_estimate_gguf_kv_gb", return_value = 0.0):
                with self._dspark_capable(False):
                    incapable = self.route._estimate_gguf_required_gb(
                        cfg, speculative_type = "dspark"
                    )
                with self._dspark_capable(True):
                    capable = self.route._estimate_gguf_required_gb(cfg, speculative_type = "dspark")
        self.assertAlmostEqual(incapable, 2000 / (1024**3), places = 9)
        self.assertAlmostEqual(capable, 5000 / (1024**3), places = 9)

    def test_split_dspark_sidecar_counts_every_shard(self):
        """Discovery hands back shard 1, so sizing it with stat() alone would let the
        guard admit a load that evicts the training run it exists to protect."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            target = p / "model.gguf"
            target.write_bytes(b"x" * 2000)
            shard1 = p / "dspark-model-Q8_0-00001-of-00002.gguf"
            shard1.write_bytes(b"y" * 3000)
            (p / "dspark-model-Q8_0-00002-of-00002.gguf").write_bytes(b"z" * 4000)
            cfg = SimpleNamespace(
                gguf_file = str(target),
                gguf_mmproj_file = None,
                gguf_mtp_file = None,
                gguf_dspark_file = str(shard1),
                gguf_hf_repo = None,
                gguf_variant = None,
            )
            with patch.object(self.route, "_estimate_gguf_kv_gb", return_value = 0.0):
                with self._dspark_capable():
                    gb = self.route._estimate_gguf_required_gb(cfg, speculative_type = "dspark")
        self.assertAlmostEqual(gb, 9000 / (1024**3), places = 9)  # 2000 + 3000 + 4000

    @staticmethod
    def _dflash_capable(supported = True):
        """Same shape as _dspark_capable: the DFlash sizing gate asks the binary
        whether it can run draft-dflash."""
        from core.inference.llama_cpp import LlamaCppBackend
        return patch.object(
            LlamaCppBackend,
            "probe_server_capabilities",
            classmethod(lambda cls, binary = None: {"supports_dflash": supported}),
        )

    def test_extra_args_drafter_is_charged_once_when_it_is_the_local_sidecar(self):
        """--model-draft usually names the very sidecar discovery already found,
        and charging it on both paths billed a 1.5 GiB drafter as 3 GiB, so the
        guard refused an inference load that fits. Identity is the resolved path,
        so a symlink or another spelling of the same file dedupes too. A drafter
        somewhere else is charged instead of the discovered sidecar, not on top of
        it: the loader ranks the extras path ahead of Studio's and the launch
        appends the caller's flags last, so only one --model-draft is resident.
        """
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            target = p / "model.gguf"
            sidecar = p / "dflash-kquant.gguf"
            target.write_bytes(b"x" * 2000)
            sidecar.write_bytes(b"y" * 3000)
            link = p / "linked-dflash.gguf"
            os.symlink(sidecar, link)
            elsewhere = p / "other" / "dflash-elsewhere.gguf"
            elsewhere.parent.mkdir()
            elsewhere.write_bytes(b"z" * 4000)
            cfg = SimpleNamespace(
                gguf_file = str(target),
                gguf_mmproj_file = None,
                gguf_mtp_file = None,
                gguf_dspark_file = None,
                gguf_dflash_file = str(sidecar),
                gguf_hf_repo = None,
                gguf_variant = None,
            )
            with (
                patch.object(self.route, "_estimate_gguf_kv_gb", return_value = 0.0),
                self._dflash_capable(),
            ):
                plain = self.route._estimate_gguf_required_gb(cfg, speculative_type = "dflash")
                same = self.route._estimate_gguf_required_gb(
                    cfg,
                    speculative_type = "dflash",
                    llama_extra_args = ["--model-draft", str(sidecar)],
                )
                through_link = self.route._estimate_gguf_required_gb(
                    cfg,
                    speculative_type = "dflash",
                    llama_extra_args = ["--model-draft", str(link)],
                )
                separate = self.route._estimate_gguf_required_gb(
                    cfg,
                    speculative_type = "dflash",
                    llama_extra_args = ["--model-draft", str(elsewhere)],
                )
        self.assertAlmostEqual(plain, 5000 / (1024**3), places = 9)
        self.assertAlmostEqual(same, 5000 / (1024**3), places = 9)  # not 8000
        self.assertAlmostEqual(through_link, 5000 / (1024**3), places = 9)
        # 2000 target + 4000 override; the 3000 sidecar loses to it and is not charged.
        self.assertAlmostEqual(separate, 6000 / (1024**3), places = 9)

    def test_extras_owning_spec_type_charge_only_their_own_drafter(self):
        """--spec-type in the extras ends _build_speculative_flags before discovery's
        sidecar is emitted, so llama-server opens the extras' --model-draft alone and
        charging the configured one too billed two drafters for the one that loads."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            target = p / "model.gguf"
            sidecar = p / "dflash-kquant.gguf"
            custom = p / "custom-dflash.gguf"
            target.write_bytes(b"x" * 2000)
            sidecar.write_bytes(b"y" * 3000)
            custom.write_bytes(b"z" * 4000)
            cfg = SimpleNamespace(
                gguf_file = str(target),
                gguf_mmproj_file = None,
                gguf_mtp_file = None,
                gguf_dspark_file = None,
                gguf_dflash_file = str(sidecar),
                gguf_hf_repo = None,
                gguf_variant = None,
            )
            with (
                patch.object(self.route, "_estimate_gguf_kv_gb", return_value = 0.0),
                self._dflash_capable(),
            ):
                owned = self.route._estimate_gguf_required_gb(
                    cfg,
                    speculative_type = "auto",
                    llama_extra_args = [
                        "--spec-type",
                        "draft-dflash",
                        "--model-draft",
                        str(custom),
                    ],
                )
        # 2000 weights + 4000 for the drafter that actually launches, not 9000.
        self.assertAlmostEqual(owned, 6000 / (1024**3), places = 9)

    def test_remote_weights_stay_in_the_estimate_beside_a_local_extra_args_drafter(self):
        """A remote repo has no local main weight, so a local --model-draft was
        the only thing making the local branch fire: it returned ~1.5 GiB and
        skipped the listing that prices the target model entirely. The drafter is
        a companion, not evidence of local weights, so it is added to whichever
        branch produces the estimate."""
        import tempfile

        import utils.models.model_config as mc

        cfg = SimpleNamespace(
            gguf_file = None,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_dspark_file = None,
            gguf_dflash_file = None,
            gguf_hf_repo = "org/repo",
            gguf_variant = "Q4_K_M",
        )
        variant = SimpleNamespace(quant = "Q4_K_M", size_bytes = 10 * 1024**3)
        with tempfile.TemporaryDirectory() as d:
            drafter = Path(d) / "dflash-kquant.gguf"
            drafter.write_bytes(b"y" * 3000)
            with (
                patch.object(mc, "list_gguf_variants", return_value = ([variant], False)),
                patch.object(self.route, "_remote_gguf_companion_bytes", return_value = 0),
                self._dflash_capable(),
            ):
                gb = self.route._estimate_gguf_required_gb(
                    cfg,
                    speculative_type = "dflash",
                    llama_extra_args = ["--model-draft", str(drafter)],
                )
        # The 10 GB target weights, not just the drafter beside them.
        self.assertAlmostEqual(gb, 10.0 + 3000 / (1024**3), places = 9)

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
            self._dspark_capable(),
        ):
            gb = self.route._estimate_gguf_required_gb(
                cfg,
                hf_token = "tok",
                speculative_type = "dspark",
            )
        self.assertEqual(captured["token"], "tok")  # token threaded for gated repos
        self.assertAlmostEqual(gb, 12.0, places = 6)  # 10 GB variant + 2 GB companions
        self.assertTrue(comp.call_args.kwargs["include_mmproj"])
        self.assertFalse(comp.call_args.kwargs["include_mtp"])
        self.assertTrue(comp.call_args.kwargs["include_dspark"])

    def test_remote_companions_choose_preferred_dspark_sidecar(self):
        siblings = [
            SimpleNamespace(rfilename = "mtp-model.gguf", size = 100),
            SimpleNamespace(rfilename = "dspark/dspark-model-BF16.gguf", size = 300),
            SimpleNamespace(rfilename = "dspark/dspark-model-Q8_0.gguf", size = 200),
            SimpleNamespace(rfilename = "dflash-model-Q8_0.gguf", size = 400),
        ]
        with patch(
            "huggingface_hub.model_info",
            return_value = SimpleNamespace(siblings = siblings),
        ):
            total = self.route._remote_gguf_companion_bytes(
                "org/repo",
                hf_token = "tok",
                include_mmproj = False,
                include_dspark = True,
            )
        self.assertEqual(total, 300)  # root MTP plus the preferred Q8_0 DSpark file
        with patch(
            "huggingface_hub.model_info",
            return_value = SimpleNamespace(siblings = siblings),
        ):
            dspark_only = self.route._remote_gguf_companion_bytes(
                "org/repo",
                hf_token = "tok",
                include_mmproj = False,
                include_mtp = False,
                include_dspark = True,
            )
        self.assertEqual(dspark_only, 200)

    def test_auto_charges_only_dspark_when_a_repo_publishes_both_sidecars(self):
        """The loader stands down on the DFlash fetch once DSpark has resolved
        under Auto, so those bytes are never resident. Charging both is not the
        safe over-estimate it is for an unlisted repo -- the listing has answered
        by then -- it is a 409 for a load that fits."""
        both = [
            SimpleNamespace(rfilename = "dspark/dspark-model-Q8_0.gguf", size = 200),
            SimpleNamespace(rfilename = "dflash-kquant.gguf", size = 400),
        ]

        def _companion_bytes(siblings, **kwargs):
            with patch(
                "huggingface_hub.model_info",
                return_value = SimpleNamespace(siblings = siblings),
            ):
                return self.route._remote_gguf_companion_bytes(
                    "org/repo",
                    hf_token = None,
                    include_mmproj = False,
                    include_mtp = False,
                    **kwargs,
                )

        self.assertEqual(
            _companion_bytes(both, include_dspark = True, include_dflash = True, dspark_first = True),
            200,
        )
        # Only one kind published: Auto still charges whichever the repo has.
        self.assertEqual(
            _companion_bytes(
                [both[1]], include_dspark = True, include_dflash = True, dspark_first = True
            ),
            400,
        )
        # An explicit DFlash request is not the Auto race and still pays for it.
        self.assertEqual(_companion_bytes(both, include_dflash = True), 400)

    def test_extras_owning_the_spec_type_are_not_charged_the_repo_sidecar(self):
        """A caller who sets --spec-type ends _build_speculative_flags before any
        mode branch, so no repository sidecar of any kind is fetched or launched.
        Only the drafter their --model-draft names becomes resident, and that is
        charged separately; billing the repo's on top is a 409 for a load that fits."""
        import utils.models.model_config as mc

        cfg = SimpleNamespace(
            gguf_file = None,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_dspark_file = None,
            gguf_dflash_file = None,
            gguf_hf_repo = "org/repo",
            gguf_variant = "Q4_K_M",
        )
        variant = SimpleNamespace(quant = "Q4_K_M", size_bytes = 1024**3)
        import tempfile

        _tmp = tempfile.TemporaryDirectory()
        self.addCleanup(_tmp.cleanup)
        # A real file: the suppression is only sound for a drafter _extras_bytes
        # can actually charge, and that charge is gated on Path(...).is_file().
        _draft = Path(_tmp.name) / "d.gguf"
        _draft.write_bytes(b"x" * 512)
        # A repo that DOES ship a sidecar, so the assertion is about the resulting
        # number and not about which flags were passed: a stub returning 0 whatever
        # it is asked would pass even if the suppression stopped working.
        _sidecar = 7 * 1024**3

        def _companions(repo, **kw):
            return (
                _sidecar
                if (kw["include_mtp"] or kw["include_dspark"] or kw["include_dflash"])
                else 0
            )

        def _estimate(extras):
            with (
                patch.object(
                    mc, "list_gguf_variants", lambda repo, hf_token = None: ([variant], False)
                ),
                patch.object(self.route, "_remote_gguf_companion_bytes", _companions),
                self._dflash_capable(),
            ):
                return self.route._estimate_gguf_required_gb(
                    cfg, speculative_type = "auto", llama_extra_args = extras
                )

        # Same load, minus the private drafter: Auto fetches the repo's sidecar and
        # pays for it, which is the charge the suppression has to remove.
        baseline = _estimate([])
        self.assertGreater(baseline, _sidecar / (1024**3))
        for extras in (
            ["--spec-type", "draft-dflash", "--model-draft", str(_draft)],
            ["--spec-type", "draft-dspark", "--model-draft", str(_draft)],
        ):
            # The repo sidecar gone, their own 512-byte drafter charged in its place.
            self.assertAlmostEqual(
                _estimate(extras),
                baseline - _sidecar / (1024**3) + 512 / (1024**3),
                places = 9,
                msg = extras,
            )

    def test_a_remote_extras_drafter_is_sized_from_its_own_repository(self):
        """--spec-draft-hf/-hfd names a SEPARATE repo, which llama-server downloads
        and loads. The target repository's companion scan cannot see it, so a target
        that ships no sidecar of its own left the drafter charged nowhere and let the
        guard admit a multi-GB overcommit beside a running training job. Bounded by
        the largest whole shard set, the only answer a listing can give: which file
        the fetch lands on is not knowable, and a split set is resident in full."""
        import utils.models.model_config as mc
        from core.inference.llama_cpp import LlamaCppBackend

        cfg = SimpleNamespace(
            gguf_file = None,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_dspark_file = None,
            gguf_dflash_file = None,
            gguf_hf_repo = "org/repo",
            gguf_variant = "Q4_K_M",
        )
        variant = SimpleNamespace(quant = "Q4_K_M", size_bytes = 1024**3)
        siblings = [
            SimpleNamespace(rfilename = "drafter-Q4_K_M-00001-of-00002.gguf", size = 3 * 1024**3 // 2),
            SimpleNamespace(rfilename = "drafter-Q4_K_M-00002-of-00002.gguf", size = 3 * 1024**3 // 2),
            SimpleNamespace(rfilename = "drafter-Q8_0.gguf", size = 2 * 1024**3),
            # Mid-upload: the fetch refuses a short set, so it must not set the bound.
            SimpleNamespace(rfilename = "drafter-F16-00001-of-00002.gguf", size = 5 * 1024**3),
            SimpleNamespace(rfilename = "notes.md", size = 9 * 1024**3),
        ]

        def _estimate(extras):
            with (
                patch.object(
                    mc, "list_gguf_variants", lambda repo, hf_token = None: ([variant], False)
                ),
                # The exact hole: the target repo has no sidecar to be charged for.
                patch.object(self.route, "_remote_gguf_companion_bytes", return_value = 0),
                patch(
                    "huggingface_hub.model_info",
                    return_value = SimpleNamespace(siblings = siblings),
                ),
                patch.object(
                    LlamaCppBackend,
                    "probe_server_capabilities",
                    classmethod(
                        lambda cls, binary = None: {
                            "supports_dspark": True,
                            "supports_dflash": True,
                        }
                    ),
                ),
            ):
                return self.route._estimate_gguf_required_gb(
                    cfg, speculative_type = "auto", llama_extra_args = extras
                )

        for own, remote in (
            (["--spec-type", "draft-dspark"], ["--spec-draft-hf", "org/drafter"]),
            (["--spec-type", "draft-dflash"], ["-hfd", "org/drafter"]),
        ):
            # Everything else identical, so the difference IS the drafter charge.
            self.assertAlmostEqual(
                _estimate(own + remote),
                _estimate(own) + 3 * 1024**3 / (1024**3),
                places = 9,
                msg = remote,
            )
        # The :quant tag is llama.cpp's own narrowing, so the bound follows it down
        # rather than charging the repo's largest family for a small drafter.
        self.assertAlmostEqual(
            _estimate(["--spec-type", "draft-dflash", "-hfd", "org/drafter:Q8_0"]),
            _estimate(["--spec-type", "draft-dflash"]) + 2 * 1024**3 / (1024**3),
            places = 9,
        )

    def test_an_unreadable_remote_drafter_repo_still_pays_a_flat_reserve(self):
        """No network, a gated repo or a malformed id leaves the drafter unsized,
        and this guard protects a running training job: charging zero for a
        download the launch is certainly going to make is the one answer that
        admits the overcommit."""
        import utils.models.model_config as mc
        from core.inference.llama_cpp import LlamaCppBackend

        cfg = SimpleNamespace(
            gguf_file = None,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_dspark_file = None,
            gguf_dflash_file = None,
            gguf_hf_repo = "org/repo",
            gguf_variant = "Q4_K_M",
        )
        variant = SimpleNamespace(quant = "Q4_K_M", size_bytes = 1024**3)
        reserve = self.route._REMOTE_DRAFTER_RESERVE_BYTES
        self.assertGreater(reserve, 0)

        def _estimate(extras, listing):
            with (
                patch.object(
                    mc, "list_gguf_variants", lambda repo, hf_token = None: ([variant], False)
                ),
                patch.object(self.route, "_remote_gguf_companion_bytes", return_value = 0),
                patch("huggingface_hub.model_info", **listing),
                patch.object(
                    LlamaCppBackend,
                    "probe_server_capabilities",
                    classmethod(
                        lambda cls, binary = None: {
                            "supports_dspark": True,
                            "supports_dflash": True,
                        }
                    ),
                ),
            ):
                return self.route._estimate_gguf_required_gb(
                    cfg, speculative_type = "auto", llama_extra_args = extras
                )

        _raises = {"side_effect": OSError("gated repo")}
        _empty = {"return_value": SimpleNamespace(siblings = [])}
        base = _estimate(["--spec-type", "draft-dflash"], _raises)
        for extras, listing in (
            # Unreadable listing, a repo that lists no GGUF, and an id no listing
            # could ever answer for: all of them mean "unsized", not "free".
            (["--spec-type", "draft-dflash", "-hfd", "org/drafter"], _raises),
            (["--spec-type", "draft-dflash", "-hfd", "org/drafter"], _empty),
            (["--spec-type", "draft-dflash", "-hfd", "not-a-repo-id"], _empty),
        ):
            self.assertAlmostEqual(
                _estimate(extras, listing),
                base + reserve / (1024**3),
                places = 9,
                msg = extras,
            )

    def test_a_priced_remote_extras_drafter_is_not_charged_twice(self):
        """--spec-draft-hf names the drafter that actually loads, and it is now
        priced from its own listing, so the target repository's sidecar must NOT
        be charged as well: _build_speculative_flags returns before Studio emits
        that sidecar, so it never becomes resident and billing it 409s a load
        that fits. (Before the remote repo was priced this test asserted the
        opposite, which was the safe reading while the drafter was charged
        nowhere at all.)"""
        import utils.models.model_config as mc
        from core.inference.llama_cpp import LlamaCppBackend

        cfg = SimpleNamespace(
            gguf_file = None,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_dspark_file = None,
            gguf_dflash_file = None,
            gguf_hf_repo = "org/repo",
            gguf_variant = "Q4_K_M",
        )
        variant = SimpleNamespace(quant = "Q4_K_M", size_bytes = 1024**3)
        for extras, kind in (
            (["--spec-type", "draft-dspark", "--spec-draft-hf", "org/drafter"], "include_dspark"),
            (["--spec-type", "draft-dflash", "-hfd", "org/drafter"], "include_dflash"),
        ):
            with (
                patch.object(
                    mc, "list_gguf_variants", lambda repo, hf_token = None: ([variant], False)
                ),
                patch.object(self.route, "_remote_gguf_companion_bytes", return_value = 0) as comp,
                patch.object(
                    LlamaCppBackend,
                    "probe_server_capabilities",
                    classmethod(
                        lambda cls, binary = None: {
                            "supports_dspark": True,
                            "supports_dflash": True,
                        }
                    ),
                ),
            ):
                self.route._estimate_gguf_required_gb(
                    cfg, speculative_type = "auto", llama_extra_args = extras
                )
                self.assertFalse(comp.call_args.kwargs[kind], extras)
                self.assertFalse(comp.call_args.kwargs["include_mtp"], extras)

    def test_the_unreadable_drafter_reserve_covers_the_largest_drafter_class(self):
        """The fallback is only reached when the listing cannot be read, and
        llama-server can still open the repo from its local HF cache, so the
        number has to cover what it might find. A DSpark sidecar is about 11 GB
        (llama_cpp._emit_dspark says so where it warns that --fit skips it), and
        --spec-draft-hf can name any repo, so a typical-drafter figure here
        underprices the load the guard is protecting a training run from."""
        self.assertGreaterEqual(self.route._REMOTE_DRAFTER_RESERVE_BYTES, 11 * 1024**3)

    def test_an_unlistable_remote_drafter_is_measured_from_the_local_cache(self):
        """An unreadable listing is exactly the case where the repo is already
        cached, which is what lets llama-server open it offline, and --spec-draft-hf
        takes any repo, so a class-based constant can undercount a 30 GB drafter by
        a lot. Measure what is on disk instead, with the same whole-shard-set bound
        the listing path uses."""
        cached = SimpleNamespace(
            repo_id = "org/drafter",
            revisions = [
                SimpleNamespace(
                    refs = {"main"},
                    last_modified = 2.0,
                    files = [
                        SimpleNamespace(
                            file_name = "big-00001-of-00002.gguf", size_on_disk = 16 * 1024**3
                        ),
                        SimpleNamespace(
                            file_name = "big-00002-of-00002.gguf", size_on_disk = 14 * 1024**3
                        ),
                        SimpleNamespace(file_name = "notes.txt", size_on_disk = 10),
                    ],
                ),
                # A stale snapshot still on disk. llama-server resolves the cached
                # ref it was asked for, so this relic must not become the bound.
                SimpleNamespace(
                    refs = set(),
                    last_modified = 1.0,
                    files = [SimpleNamespace(file_name = "old-F16.gguf", size_on_disk = 60 * 1024**3)],
                ),
            ],
        )
        with (
            patch("huggingface_hub.model_info", side_effect = OSError("no network")),
            patch(
                "huggingface_hub.scan_cache_dir",
                return_value = SimpleNamespace(repos = [cached]),
            ),
        ):
            charged = self.route._remote_drafter_repo_bytes("org/drafter", hf_token = None)
        # The whole 30 GB set that is actually resident, not the flat reserve.
        self.assertEqual(charged, 30 * 1024**3)

    def test_a_local_model_draft_that_is_not_on_disk_is_not_priced_as_a_repo(self):
        """--model-draft takes a path, --spec-draft-hf takes a repo id. A path that
        does not exist is a drafter llama-server will not load, so it costs nothing.
        Charging it the unreadable-repo reserve 409s the chat load over 12 GiB that
        a typo, not a download, put in the extras."""
        import utils.models.model_config as mc

        cfg = SimpleNamespace(
            gguf_file = None,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_dspark_file = None,
            gguf_dflash_file = None,
            gguf_hf_repo = "org/repo",
            gguf_variant = "Q4_K_M",
        )
        variant = SimpleNamespace(quant = "Q4_K_M", size_bytes = 4 * 1024**3)
        with (
            patch.object(mc, "list_gguf_variants", lambda repo, hf_token = None: ([variant], False)),
            patch.object(self.route, "_remote_gguf_companion_bytes", return_value = 0),
            patch.object(self.route, "_estimate_gguf_kv_gb", return_value = 0.0),
            patch("huggingface_hub.model_info", side_effect = AssertionError("priced as a repo")),
        ):
            charged = self.route._estimate_gguf_required_gb(
                cfg,
                speculative_type = "auto",
                llama_extra_args = ["--model-draft", "/nope/does-not-exist.gguf"],
            )
        self.assertAlmostEqual(charged, 4.0, places = 6)

    def test_only_the_winning_draft_flag_decides_repo_or_path(self):
        """Draft flags are last-wins in llama-server, so a repo id followed by a
        --model-draft leaves the path as the drafter. Asking "does any remote flag
        appear" prices that path as a repository and charges the 12 GiB reserve for
        a drafter the launch cannot open."""
        import utils.models.model_config as mc

        cfg = SimpleNamespace(
            gguf_file = None,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_dspark_file = None,
            gguf_dflash_file = None,
            gguf_hf_repo = "org/repo",
            gguf_variant = "Q4_K_M",
        )
        variant = SimpleNamespace(quant = "Q4_K_M", size_bytes = 4 * 1024**3)
        with (
            patch.object(mc, "list_gguf_variants", lambda repo, hf_token = None: ([variant], False)),
            patch.object(self.route, "_remote_gguf_companion_bytes", return_value = 0),
            patch.object(self.route, "_estimate_gguf_kv_gb", return_value = 0.0),
            patch("huggingface_hub.model_info", side_effect = AssertionError("priced as a repo")),
        ):
            charged = self.route._estimate_gguf_required_gb(
                cfg,
                speculative_type = "auto",
                llama_extra_args = [
                    "--spec-draft-hf",
                    "org/drafter",
                    "--model-draft",
                    "/nope/does-not-exist.gguf",
                ],
            )
        self.assertAlmostEqual(charged, 4.0, places = 6)

    def test_the_cached_drafter_scan_reads_the_cache_studio_is_pointed_at(self):
        """A user who moved the Hugging Face cache launches llama-server against the
        new one. Scanning huggingface_hub's import-time default finds nothing there,
        and the reserve that replaces the measurement can undercount a large cached
        drafter beside a running training job."""
        seen = {}
        cached = SimpleNamespace(
            repo_id = "org/drafter",
            revisions = [
                SimpleNamespace(
                    refs = {"main"},
                    last_modified = 2.0,
                    files = [
                        SimpleNamespace(file_name = "drafter-Q4_K_M.gguf", size_on_disk = 7 * 1024**3)
                    ],
                )
            ],
        )

        def fake_scan(cache_dir = None, **kwargs):
            seen["cache_dir"] = cache_dir
            return SimpleNamespace(repos = [cached])

        with (
            patch("huggingface_hub.model_info", side_effect = OSError("no network")),
            patch("huggingface_hub.scan_cache_dir", side_effect = fake_scan),
            patch("utils.hf_cache_settings.active_hf_hub_cache", return_value = "/elsewhere/hub"),
        ):
            charged = self.route._remote_drafter_repo_bytes("org/drafter", hf_token = None)
        self.assertEqual(seen["cache_dir"], "/elsewhere/hub")
        self.assertEqual(charged, 7 * 1024**3)

    def test_underscore_spelled_draft_flags_classify_as_remote(self):
        """llama.cpp accepts --spec_draft_hf as well as --spec-draft-hf. The value
        parser normalises the spelling, so a classifier that compares raw tokens
        calls the repo a local path, charges nothing, and lets the guard admit a
        load whose drafter is multiple GB."""
        import utils.models.model_config as mc

        cfg = SimpleNamespace(
            gguf_file = None,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_dspark_file = None,
            gguf_dflash_file = None,
            gguf_hf_repo = "org/repo",
            gguf_variant = "Q4_K_M",
        )
        variant = SimpleNamespace(quant = "Q4_K_M", size_bytes = 4 * 1024**3)
        with (
            patch.object(mc, "list_gguf_variants", lambda repo, hf_token = None: ([variant], False)),
            patch.object(self.route, "_remote_gguf_companion_bytes", return_value = 0),
            patch.object(self.route, "_estimate_gguf_kv_gb", return_value = 0.0),
            patch.object(self.route, "_remote_drafter_repo_bytes", return_value = 6 * 1024**3),
        ):
            charged = self.route._estimate_gguf_required_gb(
                cfg,
                speculative_type = "auto",
                llama_extra_args = ["--spec_draft_hf", "org/drafter"],
            )
        self.assertAlmostEqual(charged, 10.0, places = 6)

    def test_a_listing_that_carries_no_sizes_still_pays_the_reserve(self):
        """A complete family whose listing omits sizes is not a free drafter, it is
        an unmeasured one: something loads and the guard does not know how big. The
        zero answer belongs to the case where every family is an incomplete split
        and the fetch can load none of them."""
        sizeless = SimpleNamespace(
            siblings = [SimpleNamespace(rfilename = "drafter-Q4_K_M.gguf", size = None)]
        )
        with (
            patch("huggingface_hub.model_info", return_value = sizeless),
            patch("huggingface_hub.scan_cache_dir", return_value = SimpleNamespace(repos = [])),
        ):
            charged = self.route._remote_drafter_repo_bytes("org/drafter", hf_token = None)
        self.assertEqual(charged, self.route._REMOTE_DRAFTER_RESERVE_BYTES)

    def test_extras_naming_their_own_drafter_do_not_also_pay_for_a_dflash_sidecar(self):
        """A drafter in the extras is the drafter the loader launches: the Auto
        promotion that would have discovered the repo's DFlash sidecar never runs.
        Billing both charges the training job for weights only one of them loads."""
        import utils.models.model_config as mc

        cfg = SimpleNamespace(
            gguf_file = None,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_dspark_file = None,
            gguf_dflash_file = None,
            gguf_hf_repo = "org/repo",
            gguf_variant = "Q4_K_M",
        )
        variant = SimpleNamespace(
            filename = "model-Q4_K_M.gguf", quant = "Q4_K_M", size_bytes = 10 * 1024**3
        )
        siblings = [SimpleNamespace(rfilename = "dflash-kquant.gguf", size = 2 * 1024**3)]
        with (
            patch.object(mc, "list_gguf_variants", lambda repo, hf_token = None: ([variant], False)),
            patch(
                "huggingface_hub.model_info",
                return_value = SimpleNamespace(siblings = siblings),
            ),
            patch.object(self.route, "_remote_drafter_repo_bytes", return_value = 3 * 1024**3),
            patch.object(self.route, "_estimate_gguf_kv_gb", return_value = 0.0),
            self._dflash_capable(),
        ):
            charged = self.route._estimate_gguf_required_gb(
                cfg,
                speculative_type = "auto",
                llama_extra_args = ["--spec-draft-hf", "org/drafter"],
            )
        # The extras drafter, not the extras drafter plus the sidecar Auto no
        # longer reaches.
        self.assertAlmostEqual(charged, 13.0, places = 6)

    def test_a_remote_drafter_that_is_neither_listable_nor_cached_pays_the_reserve(self):
        """Nothing to measure and nothing to download over the Hub that just
        refused the listing, so the reserve is a cushion rather than a bound."""
        with (
            patch("huggingface_hub.model_info", side_effect = OSError("no network")),
            patch("huggingface_hub.scan_cache_dir", return_value = SimpleNamespace(repos = [])),
        ):
            charged = self.route._remote_drafter_repo_bytes("org/drafter", hf_token = None)
        self.assertEqual(charged, self.route._REMOTE_DRAFTER_RESERVE_BYTES)

    def test_a_cached_drafter_is_narrowed_by_the_quant_tag(self):
        """llama.cpp's :quant is its own narrowing, so a repo holding several
        quants must not be charged its F16 for a Q4_K_M request. Same rule the
        listing path applies, and the two have to agree."""
        cached = SimpleNamespace(
            repo_id = "org/drafter",
            revisions = [
                SimpleNamespace(
                    files = [
                        SimpleNamespace(file_name = "drafter-F16.gguf", size_on_disk = 20 * 1024**3),
                        SimpleNamespace(file_name = "drafter-Q4_K_M.gguf", size_on_disk = 3 * 1024**3),
                    ]
                )
            ],
        )
        with (
            patch("huggingface_hub.model_info", side_effect = OSError("no network")),
            patch(
                "huggingface_hub.scan_cache_dir",
                return_value = SimpleNamespace(repos = [cached]),
            ),
        ):
            charged = self.route._remote_drafter_repo_bytes("org/drafter:Q4_K_M", hf_token = None)
        self.assertEqual(charged, 3 * 1024**3)

    def test_a_cpu_offloaded_extras_drafter_is_not_charged_vram(self):
        """-ngld 0 keeps the drafter in host memory. Charging it against the
        training job's VRAM is not a conservative estimate, it is the wrong
        resource, and it 409s a load that takes no VRAM for the drafter at all."""
        import utils.models.model_config as mc

        cfg = SimpleNamespace(
            gguf_file = None,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_dspark_file = None,
            gguf_dflash_file = None,
            gguf_hf_repo = "org/repo",
            gguf_variant = "Q4_K_M",
        )
        variant = SimpleNamespace(quant = "Q4_K_M", size_bytes = 1024**3)
        extras = ["--spec-type", "draft-dspark", "--spec-draft-hf", "org/drafter"]
        with (
            patch.object(mc, "list_gguf_variants", lambda repo, hf_token = None: ([variant], False)),
            patch.object(self.route, "_remote_gguf_companion_bytes", return_value = 0),
            patch.object(self.route, "_remote_drafter_repo_bytes", return_value = 8 * 1024**3),
            patch.object(self.route, "_estimate_gguf_kv_gb", return_value = 0.0),
        ):
            on_gpu = self.route._estimate_gguf_required_gb(
                cfg, speculative_type = "auto", llama_extra_args = extras
            )
            on_cpu = self.route._estimate_gguf_required_gb(
                cfg,
                speculative_type = "auto",
                llama_extra_args = [*extras, "--spec-draft-ngl", "0"],
            )
        self.assertAlmostEqual(on_gpu - on_cpu, 8.0, places = 6)

    def test_a_bare_model_draft_overrides_the_repository_sidecar(self):
        """No --spec-type needed for the override to win: the loader ranks the
        extras draft path ahead of Studio's and the launch appends the caller's
        flags last, so exactly one --model-draft is resident. Charging the repo's
        sidecar as well 409s a load that fits."""
        import tempfile

        import utils.models.model_config as mc

        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "model.gguf"
            sidecar = Path(d) / "dspark-model-Q8_0.gguf"
            custom = Path(d) / "elsewhere.gguf"
            target.write_bytes(b"t" * 2000)
            sidecar.write_bytes(b"s" * 3000)
            custom.write_bytes(b"c" * 4000)
            cfg = SimpleNamespace(
                gguf_file = str(target),
                gguf_mmproj_file = None,
                gguf_mtp_file = None,
                gguf_dspark_file = str(sidecar),
                gguf_dflash_file = None,
                gguf_hf_repo = None,
                gguf_variant = None,
            )
            with (
                patch.object(self.route, "_estimate_gguf_kv_gb", return_value = 0.0),
                self._dspark_capable(),
            ):
                charged = self.route._estimate_gguf_required_gb(
                    cfg,
                    speculative_type = "auto",
                    llama_extra_args = ["--model-draft", str(custom)],
                )
        # 2000 target + 4000 override, not the 3000 sidecar it displaces.
        self.assertAlmostEqual(charged, 6000 / (1024**3), places = 9)

    def test_a_cpu_pinned_discovered_sidecar_is_not_charged_vram(self):
        """-ngld 0 applies to whichever separate drafter launches, including one
        Studio resolved itself with no draft path in the extras at all. It is then
        host-resident, so charging it against the training job's VRAM 409s a load
        that takes none."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "model.gguf"
            sidecar = Path(d) / "dspark-model-Q8_0.gguf"
            target.write_bytes(b"t" * 2000)
            sidecar.write_bytes(b"s" * 3000)
            cfg = SimpleNamespace(
                gguf_file = str(target),
                gguf_mmproj_file = None,
                gguf_mtp_file = None,
                gguf_dspark_file = str(sidecar),
                gguf_dflash_file = None,
                gguf_hf_repo = None,
                gguf_variant = None,
            )
            with (
                patch.object(self.route, "_estimate_gguf_kv_gb", return_value = 0.0),
                self._dspark_capable(),
            ):
                on_gpu = self.route._estimate_gguf_required_gb(cfg, speculative_type = "auto")
                on_cpu = self.route._estimate_gguf_required_gb(
                    cfg,
                    speculative_type = "auto",
                    llama_extra_args = ["--spec-draft-ngl", "0"],
                )
        self.assertAlmostEqual(on_gpu, 5000 / (1024**3), places = 9)
        self.assertAlmostEqual(on_cpu, 2000 / (1024**3), places = 9)

    def test_a_drafter_repo_with_only_a_partial_set_is_charged_nothing(self):
        """The bound is deliberately zero when every family is an incomplete split:
        the fetch refuses all of them, so no draft weights become resident. Turning
        that into the unreadable-listing reserve 409s a load for VRAM nothing takes."""
        partial = SimpleNamespace(
            siblings = [SimpleNamespace(rfilename = "drafter-00001-of-00002.gguf", size = 4 * 1024**3)]
        )
        with patch("huggingface_hub.model_info", return_value = partial):
            charged = self.route._remote_drafter_repo_bytes("org/drafter", hf_token = None)
        self.assertEqual(charged, 0)

    def test_a_partial_dflash_shard_set_is_not_charged(self):
        """The fetch refuses a family whose encoded shard count is short, so a
        listing caught mid-publication must not be billed for its listed half:
        the DSpark path already filters on this, and the two have to agree."""
        partial = [
            SimpleNamespace(rfilename = "dflash-kquant-00001-of-00002.gguf", size = 400),
        ]
        whole = partial + [
            SimpleNamespace(rfilename = "dflash-kquant-00002-of-00002.gguf", size = 300),
        ]

        def _companion_bytes(siblings):
            with patch(
                "huggingface_hub.model_info",
                return_value = SimpleNamespace(siblings = siblings),
            ):
                return self.route._remote_gguf_companion_bytes(
                    "org/repo",
                    hf_token = None,
                    include_mmproj = False,
                    include_mtp = False,
                    include_dflash = True,
                )

        self.assertEqual(_companion_bytes(partial), 0)
        self.assertEqual(_companion_bytes(whole), 700)

    def test_auto_tells_the_companion_sizing_that_dspark_comes_first(self):
        """The remote branch is where both kinds can be asked for at once, so it
        is the caller that has to pass the loader's Auto rule down."""
        import utils.models.model_config as mc

        cfg = SimpleNamespace(
            gguf_file = None,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_dspark_file = None,
            gguf_dflash_file = None,
            gguf_hf_repo = "org/repo",
            gguf_variant = "Q4_K_M",
        )
        variant = SimpleNamespace(quant = "Q4_K_M", size_bytes = 1024**3)
        with (
            patch.object(mc, "list_gguf_variants", lambda repo, hf_token = None: ([variant], False)),
            patch.object(self.route, "_remote_gguf_companion_bytes", return_value = 0) as comp,
            self._dspark_capable(),
        ):
            self.route._estimate_gguf_required_gb(cfg, speculative_type = "auto")
            self.assertTrue(comp.call_args.kwargs["dspark_first"])
            self.route._estimate_gguf_required_gb(cfg, speculative_type = "dflash")
            self.assertFalse(comp.call_args.kwargs["dspark_first"])

    _MULTI_FAMILY_SIBLINGS = [
        SimpleNamespace(rfilename = "model-A-Q4_K_M.gguf", size = 10 * 1024**3),
        SimpleNamespace(rfilename = "model-B-Q4_K_M.gguf", size = 10 * 1024**3),
        # Named after model A and higher precision, so the name-only key ranks it
        # first for every weight in the repo.
        SimpleNamespace(rfilename = "dflash-model-A-Q8_0.gguf", size = 1024**3),
        SimpleNamespace(rfilename = "dflash-kquant.gguf", size = 4 * 1024**3),
    ]

    def test_remote_dflash_sizing_bounds_every_candidate_the_fallback_can_reach(self):
        """_download_dflash reads a candidate's header only after paying for the
        bytes, and a rejection falls through to the next name in the ranking, so
        the file that lands can be any candidate -- including one LARGER than the
        best-ranked pick. Sizing the first-ranked entry alone under-charged model
        A by 3 GiB and admitted a load that then exhausts VRAM beside a running
        training job. Headers are unreadable from a listing, so the bound has to
        cover the whole reachable set."""
        with patch(
            "huggingface_hub.model_info",
            return_value = SimpleNamespace(siblings = self._MULTI_FAMILY_SIBLINGS),
        ):
            total = self.route._remote_gguf_companion_bytes(
                "org/repo",
                hf_token = None,
                include_mmproj = False,
                include_mtp = False,
                include_dflash = True,
            )
        # 4 GiB, the largest reachable candidate, for either weight in the repo:
        # model A's own 1 GiB sidecar is merely the one tried FIRST.
        self.assertEqual(total, 4 * 1024**3)

    def test_remote_dflash_sizing_totals_every_shard_of_a_split_sidecar(self):
        """A split sidecar is picked as its first shard, and the download then
        fetches every sibling; llama-server keeps the whole set resident. Sizing
        one shard budgeted a two-shard 2 GiB sidecar at 1 GiB and let it lose the
        comparison to a smaller single-file candidate, which is the direction that
        admits a load and then exhausts VRAM beside a running training job."""
        siblings = [
            SimpleNamespace(rfilename = "model-Q4_K_M.gguf", size = 10 * 1024**3),
            SimpleNamespace(rfilename = "dflash-split-00001-of-00002.gguf", size = 1024**3),
            SimpleNamespace(rfilename = "dflash-split-00002-of-00002.gguf", size = 1024**3),
            # Bigger than either shard, smaller than the set they form.
            SimpleNamespace(rfilename = "dflash-kquant.gguf", size = 3 * 1024**3 // 2),
        ]
        with patch(
            "huggingface_hub.model_info",
            return_value = SimpleNamespace(siblings = siblings),
        ):
            total = self.route._remote_gguf_companion_bytes(
                "org/repo",
                hf_token = None,
                include_mmproj = False,
                include_mtp = False,
                include_dflash = True,
            )
        # The set totals 2 GiB and is the largest thing the fallback can land on.
        self.assertEqual(total, 2 * 1024**3)

    def test_remote_dflash_sizing_charges_a_split_set_once(self):
        """The other half: every shard is a listed dflash- name, so a rule that
        totalled the candidates rather than taking the safe maximum across shard
        SETS would double-charge this repo and 409 a load that fits."""
        siblings = [
            SimpleNamespace(rfilename = "model-Q4_K_M.gguf", size = 10 * 1024**3),
            SimpleNamespace(rfilename = "dflash-split-00001-of-00002.gguf", size = 1024**3),
            SimpleNamespace(rfilename = "dflash-split-00002-of-00002.gguf", size = 1024**3),
        ]
        with patch(
            "huggingface_hub.model_info",
            return_value = SimpleNamespace(siblings = siblings),
        ):
            total = self.route._remote_gguf_companion_bytes(
                "org/repo",
                hf_token = None,
                include_mmproj = False,
                include_mtp = False,
                include_dflash = True,
            )
        self.assertEqual(total, 2 * 1024**3)

    def test_remote_dflash_sizing_ignores_a_nested_dflash_named_weight(self):
        """The picker is root level only, so a quants/dflash-*.gguf is an ordinary
        weight there and can never be fetched as the drafter. Charging it made the
        bound track a file the load cannot reach."""
        siblings = [
            SimpleNamespace(rfilename = "model-Q4_K_M.gguf", size = 10 * 1024**3),
            SimpleNamespace(rfilename = "dflash-kquant.gguf", size = 1024**3),
            SimpleNamespace(rfilename = "quants/dflash-model-Q8_0.gguf", size = 9 * 1024**3),
        ]
        with patch(
            "huggingface_hub.model_info",
            return_value = SimpleNamespace(siblings = siblings),
        ):
            total = self.route._remote_gguf_companion_bytes(
                "org/repo",
                hf_token = None,
                include_mmproj = False,
                include_mtp = False,
                include_dflash = True,
            )
        self.assertEqual(total, 1024**3)

    def test_remote_estimate_bounds_the_dflash_fallback_end_to_end(self):
        """End to end: the guard's own estimate has to carry the same bound, or
        the multi-family repo above is under-charged by the whole difference."""
        import utils.models.model_config as mc

        cfg = SimpleNamespace(
            gguf_file = None,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_dspark_file = None,
            gguf_dflash_file = None,
            gguf_hf_repo = "org/repo",
            gguf_variant = "Q4_K_M",
        )
        variant = SimpleNamespace(
            filename = "model-A-Q4_K_M.gguf", quant = "Q4_K_M", size_bytes = 10 * 1024**3
        )
        with (
            patch.object(mc, "list_gguf_variants", lambda repo, hf_token = None: ([variant], False)),
            patch(
                "huggingface_hub.model_info",
                return_value = SimpleNamespace(siblings = self._MULTI_FAMILY_SIBLINGS),
            ),
            self._dflash_capable(),
        ):
            gb = self.route._estimate_gguf_required_gb(cfg, speculative_type = "dflash")
        # 10 GiB of weights plus the 4 GiB the fallback can still land on, not the
        # 1 GiB candidate that merely goes first.
        self.assertAlmostEqual(gb, 14.0, places = 6)

    def test_auto_does_not_charge_dflash_when_extra_args_own_speculation(self):
        """Extra args setting --spec-type stop the loader's Auto promotion, so the
        sidecar is never opened. Charging it anyway refused a chat load with 409 for
        ~1.5 GiB nothing would load. Extra args asking for draft-dflash still pay."""
        import utils.models.model_config as mc

        cfg = SimpleNamespace(
            gguf_file = None,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_dspark_file = None,
            gguf_dflash_file = None,
            gguf_hf_repo = "org/repo",
            gguf_variant = "Q4_K_M",
        )
        variant = SimpleNamespace(
            filename = "model-Q4_K_M.gguf", quant = "Q4_K_M", size_bytes = 10 * 1024**3
        )
        siblings = [SimpleNamespace(rfilename = "dflash-kquant.gguf", size = 2 * 1024**3)]
        with (
            patch.object(mc, "list_gguf_variants", lambda repo, hf_token = None: ([variant], False)),
            patch(
                "huggingface_hub.model_info",
                return_value = SimpleNamespace(siblings = siblings),
            ),
            self._dflash_capable(),
        ):
            owned = self.route._estimate_gguf_required_gb(
                cfg,
                speculative_type = "auto",
                llama_extra_args = ["--spec-type", "ngram-mod"],
            )
            asked = self.route._estimate_gguf_required_gb(
                cfg,
                speculative_type = "auto",
                llama_extra_args = ["--spec-type", "draft-dflash"],
            )
            # Same for the forced mode: _build_speculative_flags returns before any
            # mode branch when extra args own --spec-type, so dflash never emits.
            forced = self.route._estimate_gguf_required_gb(
                cfg,
                speculative_type = "dflash",
                llama_extra_args = ["--spec-type", "ngram-mod"],
            )
        self.assertAlmostEqual(owned, 10.0, places = 6)
        self.assertAlmostEqual(asked, 12.0, places = 6)
        self.assertAlmostEqual(forced, 10.0, places = 6)

    def test_remote_dflash_sizing_drops_a_candidate_too_big_to_be_a_drafter(self):
        """The fetch refuses an oversized root dflash-*.gguf, so charging for it is a
        409 for bytes that will never be resident."""
        siblings = [
            SimpleNamespace(rfilename = "model-Q4_K_M.gguf", size = 10 * 1024**3),
            SimpleNamespace(rfilename = "dflash-model-BF16.gguf", size = 40 * 1024**3),
            SimpleNamespace(rfilename = "dflash-kquant.gguf", size = 1024**3),
        ]
        with patch(
            "huggingface_hub.model_info",
            return_value = SimpleNamespace(siblings = siblings),
        ):
            charged = self.route._remote_gguf_companion_bytes(
                "org/repo",
                hf_token = None,
                include_mmproj = False,
                include_mtp = False,
                include_dflash = True,
                weight_bytes = 10 * 1024**3,
            )
        self.assertEqual(charged, 1024**3)

    def test_remote_dspark_sizing_totals_every_shard_of_a_split_sidecar(self):
        """llama-server maps every shard, so pricing the one the ranking picked
        halved a two-shard sidecar and let the guard admit a load that evicts
        the training run it protects."""
        siblings = [
            SimpleNamespace(rfilename = "dspark/dspark-00001-of-00002.gguf", size = 5 * 1024**3),
            SimpleNamespace(rfilename = "dspark/dspark-00002-of-00002.gguf", size = 5 * 1024**3),
        ]
        with patch(
            "huggingface_hub.model_info",
            return_value = SimpleNamespace(siblings = siblings),
        ):
            charged = self.route._remote_gguf_companion_bytes(
                "org/repo",
                hf_token = None,
                include_mmproj = False,
                include_mtp = False,
                include_dspark = True,
            )
        self.assertEqual(charged, 10 * 1024**3)

    def test_auto_budgets_dflash_when_the_dspark_set_is_incomplete(self):
        """A listing missing a DSpark shard is not a load this can end up on: the
        fetch refuses it and falls through to DFlash, which can be the larger of
        the two, so granting first refusal on the listing under-charged."""
        siblings = [
            SimpleNamespace(rfilename = "dspark/dspark-00001-of-00002.gguf", size = 1024**3),
            SimpleNamespace(rfilename = "dflash-kquant.gguf", size = 4 * 1024**3),
        ]
        with patch(
            "huggingface_hub.model_info",
            return_value = SimpleNamespace(siblings = siblings),
        ):
            charged = self.route._remote_gguf_companion_bytes(
                "org/repo",
                hf_token = None,
                include_mmproj = False,
                include_mtp = False,
                include_dspark = True,
                include_dflash = True,
                dspark_first = True,
            )
        self.assertEqual(charged, 4 * 1024**3)

    # ── Auto charges ONE drafter, the one the promotion leaves resident ──

    def _auto_companion_bytes(self, siblings):
        with patch(
            "huggingface_hub.model_info",
            return_value = SimpleNamespace(siblings = siblings),
        ):
            return self.route._remote_gguf_companion_bytes(
                "org/repo",
                hf_token = None,
                include_mmproj = False,
                include_mtp = True,
                include_dspark = True,
                include_dflash = True,
                dspark_first = True,
            )

    def test_auto_does_not_charge_the_mtp_drafter_dflash_replaces(self):
        """Under Auto the caller asks for MTP and DFlash together, but the loader
        promotes DFlash and overwrites mtp_draft_path with it, so the two are
        never resident at once. Charging the sum was a 409 for a load that fits.

        The DFlash sidecar is the larger of the two here, so the bound is its
        size alone -- the MTP bytes are not added on top."""
        siblings = [
            SimpleNamespace(rfilename = "mtp-model.gguf", size = 1024**3),
            SimpleNamespace(rfilename = "dflash-kquant.gguf", size = 3 * 1024**3),
        ]
        self.assertEqual(self._auto_companion_bytes(siblings), 3 * 1024**3)

    def test_auto_keeps_the_mtp_charge_when_the_dflash_candidates_may_all_fail(self):
        """The other half of the same rule: every DFlash candidate can still be
        turned away on its header, and the load then keeps the MTP drafter it has
        already fetched. That outcome is genuinely unknown from a listing, so the
        larger of the two is charged -- here the MTP one."""
        siblings = [
            SimpleNamespace(rfilename = "mtp-model.gguf", size = 5 * 1024**3),
            SimpleNamespace(rfilename = "dflash-kquant.gguf", size = 1024**3),
        ]
        self.assertEqual(self._auto_companion_bytes(siblings), 5 * 1024**3)

    def test_auto_charges_the_largest_reachable_dflash_against_the_mtp_drafter(self):
        """Items 2 and 5 together, which is the only way they are coherent: the
        DFlash side of the comparison is the whole reachable candidate set (4
        GiB), not the first-ranked pick (1 GiB), and it is compared against the
        MTP drafter rather than added to it. Fixing only one of the two lands on
        the wrong number from either side: summing the first-ranked pick charges
        3 GiB, and comparing against the first-ranked pick charges 2 GiB."""
        siblings = [
            SimpleNamespace(rfilename = "mtp-model.gguf", size = 2 * 1024**3),
            *self._MULTI_FAMILY_SIBLINGS,
        ]
        self.assertEqual(self._auto_companion_bytes(siblings), 4 * 1024**3)

    def test_auto_charges_dspark_alone_over_both_of_the_others(self):
        """DSpark takes first refusal in the promotion and has no post-fetch
        rejection, so a listed sidecar settles the load: the DFlash fetch stands
        down and mtp_draft_path is replaced. Neither of the other two is
        resident."""
        siblings = [
            SimpleNamespace(rfilename = "mtp-model.gguf", size = 1024**3),
            SimpleNamespace(rfilename = "dspark/dspark-model-Q8_0.gguf", size = 2 * 1024**3),
            SimpleNamespace(rfilename = "dflash-kquant.gguf", size = 3 * 1024**3),
        ]
        self.assertEqual(self._auto_companion_bytes(siblings), 2 * 1024**3)

    def test_auto_still_charges_the_mtp_drafter_when_the_repo_ships_no_sidecar(self):
        """Positive control: with nothing to promote, Auto launches the MTP
        drafter and it keeps its charge."""
        siblings = [
            SimpleNamespace(rfilename = "mtp-model.gguf", size = 1024**3),
            SimpleNamespace(rfilename = "model-Q4_K_M.gguf", size = 10 * 1024**3),
        ]
        self.assertEqual(self._auto_companion_bytes(siblings), 1024**3)

    def test_an_explicit_request_is_not_the_auto_race(self):
        """dspark_first off means the caller already narrowed the kinds to the one
        it asked for, so nothing here may drop a charge it passed in."""
        siblings = [
            SimpleNamespace(rfilename = "mtp-model.gguf", size = 1024**3),
            SimpleNamespace(rfilename = "dflash-kquant.gguf", size = 3 * 1024**3),
        ]
        with patch(
            "huggingface_hub.model_info",
            return_value = SimpleNamespace(siblings = siblings),
        ):
            total = self.route._remote_gguf_companion_bytes(
                "org/repo",
                hf_token = None,
                include_mmproj = False,
                include_mtp = True,
                include_dflash = True,
            )
        self.assertEqual(total, 4 * 1024**3)

    def test_native_drafter_accept_applies_the_lease_before_the_scan_reads(self):
        """The load route's boundary, in the shape ModelConfig.from_identifier
        takes. Discovery runs inside from_identifier and opens a DFlash
        candidate's header, so a dflash-*.gguf symlinked out of the granted
        directory was read before the validated rescan could reject it, and no
        later rejection takes a read back."""
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            leased = Path(d) / "leased"
            leased.mkdir()
            outside = Path(d) / "outside"
            outside.mkdir()
            weight = leased / "model-Q4_K_M.gguf"
            weight.write_bytes(b"x")
            inside = leased / "dflash-kquant.gguf"
            inside.write_bytes(b"y")
            target = outside / "dflash-escape.gguf"
            target.write_bytes(b"z")
            escape = leased / "dflash-escape.gguf"
            os.symlink(target, escape)

            accept = self.route._native_drafter_accept
            self.assertTrue(accept(str(inside), str(weight), "dflash", str(leased)))
            self.assertFalse(accept(str(target.resolve()), str(weight), "dflash", str(leased)))

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
            _TENSOR_PARALLEL_KV_TYPES = frozenset({"f16", "bf16", "f32"})
            supports_kv_unified = True

            def _read_gguf_metadata(self, path):
                pass

            def _can_estimate_kv(self):
                return True

            @classmethod
            def probe_server_capabilities(cls):
                return {"supports_kv_unified": cls.supports_kv_unified}

            def _estimate_kv_cache_bytes(
                self,
                ctx,
                cache_type = None,
                n_parallel = 1,
                swa_full = False,
                kv_unified = False,
                n_ubatch = None,
                flash_attn = True,
            ):
                seen["ctx"] = ctx
                seen["cache_type"] = cache_type
                seen["n_parallel"] = n_parallel
                seen["swa_full"] = swa_full
                seen["kv_unified"] = kv_unified
                seen["n_ubatch"] = n_ubatch
                seen["flash_attn"] = flash_attn
                return ctx * n_parallel * (1024**2)  # 1 MiB per ctx unit per slot

            _PIPELINE_PER_DEVICE_OVERHEAD_MIB = 0

            # zeroed: this test pins the kv sizing, not the compute buffers
            def _estimate_compute_buffer_bytes(
                self,
                *,
                n_ubatch = None,
                n_parallel = 1,
                per_device_tensor = False,
            ):
                seen["compute_n_ubatch"] = n_ubatch
                return 0

            def _compute_buffer_ctx_bytes(
                self,
                n_ctx,
                n_ubatch = None,
                cache_type_kv = None,
                *,
                layer_split = False,
            ):
                return 0

        with patch.object(self.route, "LlamaCppBackend", _FakeBackend):
            r = self.route
            # --ctx-size override above max_seq_length -> override wins
            self.assertAlmostEqual(
                r._estimate_gguf_kv_gb("m", 4096, ["--ctx-size", "131072"]), 128.0
            )
            self.assertEqual(seen["ctx"], 131072)
            self.assertEqual(seen["n_parallel"], 1)  # default single slot
            self.assertFalse(seen["swa_full"])
            self.assertFalse(seen["flash_attn"])
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
            self.assertTrue(seen["kv_unified"])
            # User extras are appended after Studio's managed default.
            r._estimate_gguf_kv_gb("m", 4096, ["--no-kv-unified"], 4)
            self.assertFalse(seen["kv_unified"])
            # An older binary without the flag keeps separate KV streams.
            _FakeBackend.supports_kv_unified = False
            r._estimate_gguf_kv_gb("m", 4096, None, 4)
            self.assertFalse(seen["kv_unified"])
            r._estimate_gguf_kv_gb("m", 4096, None, 1, "f32")
            self.assertEqual(seen["cache_type"], "f32")
            r._estimate_gguf_kv_gb("m", 4096, ["--cache-type-v", "f32"])
            self.assertEqual(seen["cache_type"], "f32")
            with patch.dict(self.route.os.environ, {"LLAMA_ARG_CACHE_TYPE_K": "f32"}):
                r._estimate_gguf_kv_gb("m", 4096)
            self.assertEqual(seen["cache_type"], "f32")
            with patch.dict(
                self.route.os.environ,
                {
                    "LLAMA_ARG_CACHE_TYPE_K": "q4_0",
                    "LLAMA_ARG_CACHE_TYPE_V": "q4_0",
                },
            ):
                r._estimate_gguf_kv_gb("m", 4096)
            self.assertEqual(seen["cache_type"], "q4_0")
            r._estimate_gguf_kv_gb(
                "m",
                4096,
                ["--cache-type-k", "q4_0", "--cache-type-v", "q4_0"],
                tensor_parallel = True,
            )
            self.assertEqual(seen["cache_type"], "f16")
            r._estimate_gguf_kv_gb(
                "m",
                4096,
                ["--cache-type-k", "f32", "--cache-type-v", "q4_0"],
                tensor_parallel = True,
            )
            self.assertEqual(seen["cache_type"], "f32")
            # Full SWA mode follows the same pass-through args as the launcher.
            r._estimate_gguf_kv_gb("m", 4096, ["--swa_full"])
            self.assertTrue(seen["swa_full"])
            r._estimate_gguf_kv_gb("m", 4096, ["--kv_unified", "--ubatch_size", "256"])
            self.assertTrue(seen["kv_unified"])
            self.assertEqual(seen["n_ubatch"], 256)


# ── load_model integration: authoritative 409, and no unload before refusal ──


class TestLoadModelGuardIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.route = _load_inference_route()

    def test_cuda_gpu_ids_allow_matching_draft_device(self):
        self.route._reject_draft_device_with_gpu_ids(
            [1],
            ["--spec-draft-device", "CUDA0"],
            gpu_ids_are_vulkan_ordinals = False,
        )

    def test_vulkan_gpu_ids_reject_underscore_draft_device_alias(self):
        with self.assertRaises(HTTPException) as exc:
            self.route._reject_draft_device_with_gpu_ids(
                [0],
                ["--spec_draft_device", "Vulkan1"],
                gpu_ids_are_vulkan_ordinals = True,
            )

        self.assertEqual(exc.exception.status_code, 400)
        self.assertIn("Vulkan1", exc.exception.detail)

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
            patch.object(
                self.route, "_hf_offline_if_unreachable", lambda: contextlib.nullcontext()
            ),
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

    def test_gguf_inherited_draft_device_rejected_under_vulkan_gpu_ids(self):
        # Check effective inherited extras, not only the raw request.
        import contextlib
        from unittest.mock import MagicMock
        from models.inference import LoadRequest

        inf = SimpleNamespace(active_model_name = None)
        inf.unload_model = MagicMock()
        inf._shutdown_subprocess = MagicMock()
        llama = SimpleNamespace(
            is_loaded = True,
            model_identifier = "x.gguf",
            hf_variant = None,
            hf_repo = None,
            gguf_path = None,
            extra_args = ["--spec-draft-device", "CUDA1"],
            extra_args_source = ("x.gguf", None),
            last_load_intent = None,
            layer_preserves_tensor_intent = False,
            is_vulkan_build = lambda: True,
            adopt_load_intent_if_matched = lambda intent: False,
        )
        llama.unload_model = MagicMock()
        cfg = SimpleNamespace(
            is_gguf = True,
            is_lora = False,
            is_vision = False,
            path = None,
            base_model = None,
            identifier = "x.gguf",
            display_name = "x",
            gguf_variant = None,
        )
        request = LoadRequest(model_path = "x.gguf", gpu_ids = [0], max_seq_length = 4096)
        captured = []
        with (
            patch("utils.transformers_version.latest_tier_active_for", return_value = False),
            patch.object(
                self.route,
                "validate_extra_args",
                side_effect = lambda args: list(args) if args else None,
            ),
            patch.object(
                self.route,
                "_resolve_model_identifier_for_request",
                return_value = ("x.gguf", "x.gguf", False),
            ),
            patch.object(self.route, "resolve_effective_chat_template_override", return_value = None),
            patch.object(
                self.route,
                "_resolve_gguf_gpu_ids_for_request",
                return_value = ([0], True),
            ),
            patch.object(self.route, "get_inference_backend", return_value = inf),
            patch.object(self.route, "get_llama_cpp_backend", return_value = llama),
            patch.object(
                self.route, "_hf_offline_if_unreachable", lambda: contextlib.nullcontext()
            ),
            patch.object(self.route.ModelConfig, "from_identifier", return_value = cfg),
            _stub_guard_deps(training_active = True, decision = (True, {}), captured = captured),
        ):
            with self.assertRaises(HTTPException) as exc:
                asyncio.run(
                    self.route.load_model(request, fastapi_request = MagicMock(), current_subject = "u")
                )

        self.assertEqual(exc.exception.status_code, 400)
        self.assertIn("draft-model device", exc.exception.detail)
        self.assertIn("set it to none", exc.exception.detail)
        self.assertEqual(captured, [])
        inf.unload_model.assert_not_called()
        llama.unload_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
