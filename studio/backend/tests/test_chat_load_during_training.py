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
                llama_extra_args = llama_extra_args,
                cache_type_kv = cache_type_kv,
                tensor_parallel = tensor_parallel,
                gpu_memory_mode = gpu_memory_mode,
                gpu_ids_are_vulkan_ordinals = gpu_ids_are_vulkan_ordinals,
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
                lambda config, **kw: captured.update(kw),
            ),
        ):
            asyncio.run(self.route.validate_model(request, current_subject = "u"))
        self.assertEqual(captured.get("gpu_memory_mode"), "manual")

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
                lambda config, **kw: captured.update(kw),
            ),
        ):
            asyncio.run(self.route.validate_model(request, current_subject = "u"))
        self.assertEqual(captured.get("llama_extra_args"), ["-c", "32768"])
        self.assertIn("n_parallel", captured)
        self.assertEqual(captured.get("cache_type_kv"), "f32")
        self.assertTrue(captured.get("tensor_parallel"))

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

    def test_remote_mmproj_opt_out_keeps_mtp_and_honors_last_wins(self):
        import utils.models.model_config as mc

        cfg = SimpleNamespace(
            gguf_file = None,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_hf_repo = "org/repo",
            gguf_variant = "Q4_K_M",
            has_audio_input = True,
        )
        variant = SimpleNamespace(quant = "Q4_K_M", size_bytes = 10 * 1024**3)

        def companion_bytes(_repo, *, hf_token, include_mmproj):
            # One GiB of MTP is always loaded; the second GiB is mmproj.
            return (1 + int(include_mmproj)) * 1024**3

        with (
            patch.object(mc, "list_gguf_variants", return_value = ([variant], False)),
            patch.object(self.route, "_remote_gguf_companion_bytes", side_effect = companion_bytes) as comp,
        ):
            disabled_gb = self.route._estimate_gguf_required_gb(
                cfg, llama_extra_args = ["--mmproj-auto", "--no-mmproj-auto"]
            )
            self.assertAlmostEqual(disabled_gb, 11.0, places = 6)
            self.assertFalse(comp.call_args.kwargs["include_mmproj"])

            enabled_gb = self.route._estimate_gguf_required_gb(
                cfg, llama_extra_args = ["--no-mmproj", "--mmproj-auto"]
            )
            self.assertAlmostEqual(enabled_gb, 12.0, places = 6)
            self.assertTrue(comp.call_args.kwargs["include_mmproj"])

    def test_local_mmproj_opt_out_keeps_mtp_weights_and_kv(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            main = p / "model.gguf"
            mmproj = p / "mmproj-model.gguf"
            mtp = p / "mtp-model.gguf"
            main.write_bytes(b"w" * 1000)
            mmproj.write_bytes(b"p" * 2000)
            mtp.write_bytes(b"m" * 3000)
            cfg = SimpleNamespace(
                gguf_file = str(main),
                gguf_mmproj_file = str(mmproj),
                gguf_mtp_file = str(mtp),
                gguf_hf_repo = None,
                gguf_variant = None,
            )

            with patch.object(self.route, "_estimate_gguf_kv_gb", return_value = 2.0):
                disabled_gb = self.route._estimate_gguf_required_gb(
                    cfg, llama_extra_args = ["--no-mmproj"]
                )
                enabled_gb = self.route._estimate_gguf_required_gb(
                    cfg, llama_extra_args = ["--no-mmproj", "--mmproj-auto"]
                )

        self.assertAlmostEqual(disabled_gb, (1000 + 3000) / (1024**3) + 2.0, places = 9)
        self.assertAlmostEqual(enabled_gb, (1000 + 2000 + 3000) / (1024**3) + 2.0, places = 9)

    def test_cached_remote_projector_does_not_hide_main_download(self):
        import tempfile
        import utils.models.model_config as mc

        with tempfile.TemporaryDirectory() as d:
            mmproj = Path(d) / "mmproj.gguf"
            mmproj.write_bytes(b"x" * 1000)
            cfg = SimpleNamespace(
                gguf_file = None,
                gguf_mmproj_file = str(mmproj),
                gguf_mtp_file = None,
                gguf_hf_repo = "org/audio-repo",
                gguf_variant = "Q4_K_M",
                has_audio_input = True,
            )
            variant = SimpleNamespace(quant = "Q4_K_M", size_bytes = 10 * 1024**3)
            with (
                patch.object(mc, "list_gguf_variants", return_value = ([variant], False)),
                patch.object(self.route, "_remote_gguf_companion_bytes", return_value = 1000) as comp,
            ):
                gb = self.route._estimate_gguf_required_gb(cfg)

        self.assertAlmostEqual(gb, (10 * 1024**3 + 1000) / (1024**3), places = 9)
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
            extra_args = ["--spec-draft-device", "CUDA1"],
            extra_args_source = ("x.gguf", None),
            is_vulkan_build = lambda: True,
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
            patch.object(self.route, "_request_matches_loaded_settings", return_value = False),
            patch.object(
                self.route,
                "_resolve_gguf_gpu_ids_for_request",
                return_value = ([0], True),
            ),
            patch.object(self.route, "get_inference_backend", return_value = inf),
            patch.object(self.route, "get_llama_cpp_backend", return_value = llama),
            patch.object(self.route, "_hf_offline_if_dns_dead", lambda: contextlib.nullcontext()),
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
