# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Tests for routes/training_vram.py -- the VRAM-aware decision to keep or unload a
resident chat model when a training run starts.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from utils.hardware import DeviceType
import utils.hardware.hardware as _hw_module

# Load training_vram.py standalone so importing it doesn't pull the heavy
# routes/__init__.py; its lazy backend imports still resolve via sys.modules stubs.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "training_vram_under_test", _BACKEND_ROOT / "routes" / "training_vram.py"
)
tv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tv)


class _GpuCacheResetMixin:
    """Reset module-level GPU caches between tests to prevent state leaks."""

    def tearDown(self):
        _hw_module._physical_gpu_count = None
        _hw_module._visible_gpu_count = None


def _fake_inference_backend(
    *,
    active = None,
    loading = None,
    alive = False,
):
    inf = SimpleNamespace(
        active_model_name = active,
        loading_models = set(loading or []),
        models = {},
    )
    inf._ensure_subprocess_alive = lambda: alive
    inf._shutdown_subprocess = MagicMock()
    # Bind the MagicMock so assertions can reach it.
    inf._shutdown_subprocess_mock = inf._shutdown_subprocess
    return inf


def _fake_llama_backend(
    *,
    active = False,
    identifier = "model.gguf",
    gpu_offload = None,
    loaded = None,
):
    # A healthy active server is loaded; pass loaded=False for a mid-start one.
    is_loaded = active if loaded is None else loaded
    llama = SimpleNamespace(
        is_active = active,
        is_loaded = is_loaded,
        model_identifier = identifier,
        _gpu_offload_active = gpu_offload,
    )
    llama.unload_model = MagicMock()
    return llama


def _patch_backends(inf, llama):
    """Stub core.inference + routes.inference modules so the lazy imports inside
    training_vram resolve to fakes (avoids importing torch-heavy backends)."""
    core_inf = types.ModuleType("core.inference")
    core_inf.get_inference_backend = lambda: inf
    routes_inf = types.ModuleType("routes.inference")
    routes_inf.get_llama_cpp_backend = lambda: llama
    return patch.dict(
        sys.modules, {"core.inference": core_inf, "routes.inference": routes_inf}
    )


# ── summarize_resident_chat ──────────────────────────────────────────────────


class TestSummarizeResidentChat(_GpuCacheResetMixin, unittest.TestCase):
    def test_nothing_resident(self):
        with _patch_backends(
            _fake_inference_backend(), _fake_llama_backend(active = False)
        ):
            self.assertEqual(
                tv.summarize_resident_chat(),
                {"hf": None, "gguf": None, "loading": False, "any": False},
            )

    def test_hf_resident_via_active_model(self):
        with _patch_backends(
            _fake_inference_backend(active = "unsloth/Qwen3-4B"),
            _fake_llama_backend(active = False),
        ):
            out = tv.summarize_resident_chat()
        self.assertEqual(out["hf"], "unsloth/Qwen3-4B")
        self.assertFalse(out["loading"])
        self.assertTrue(out["any"])

    def test_hf_resident_while_still_loading(self):
        # Mid-load: no active model yet but VRAM is held -> flag in-flight.
        with _patch_backends(
            _fake_inference_backend(active = None, loading = ["unsloth/Qwen3-4B"]),
            _fake_llama_backend(active = False),
        ):
            out = tv.summarize_resident_chat()
        self.assertEqual(out["hf"], "unsloth/Qwen3-4B")
        self.assertTrue(out["loading"])
        self.assertTrue(out["any"])

    def test_replacement_hf_load_is_in_flight(self):
        # Swap: new model loading while old still active -> unsafe to keep.
        with _patch_backends(
            _fake_inference_backend(active = "unsloth/old", loading = ["unsloth/new"]),
            _fake_llama_backend(active = False),
        ):
            out = tv.summarize_resident_chat()
        self.assertEqual(out["hf"], "unsloth/old")
        self.assertTrue(out["loading"])

    def test_cpu_only_gguf_is_not_a_vram_resident(self):
        # A llama-server confirmed to run entirely on CPU holds no VRAM.
        with _patch_backends(
            _fake_inference_backend(),
            _fake_llama_backend(active = True, identifier = "cpu.gguf", gpu_offload = False),
        ):
            out = tv.summarize_resident_chat()
        self.assertIsNone(out["gguf"])
        self.assertFalse(out["any"])

    def test_mid_start_gguf_is_in_flight(self):
        # Active but not yet healthy -- still allocating, so unsafe to size.
        with _patch_backends(
            _fake_inference_backend(),
            _fake_llama_backend(active = True, loaded = False, identifier = "starting.gguf"),
        ):
            out = tv.summarize_resident_chat()
        self.assertEqual(out["gguf"], "starting.gguf")
        self.assertTrue(out["loading"])

    def test_bare_alive_subprocess_without_model_is_not_resident(self):
        # Bare-alive subprocess (no model, only CUDA context) must NOT count.
        with _patch_backends(
            _fake_inference_backend(active = None, alive = True),
            _fake_llama_backend(active = False),
        ):
            out = tv.summarize_resident_chat()
        self.assertIsNone(out["hf"])
        self.assertFalse(out["any"])

    def test_gguf_resident(self):
        with _patch_backends(
            _fake_inference_backend(),
            _fake_llama_backend(active = True, identifier = "gemma.gguf"),
        ):
            out = tv.summarize_resident_chat()
        self.assertEqual(out["gguf"], "gemma.gguf")
        self.assertFalse(out["loading"])  # healthy/loaded -> safe to size
        self.assertTrue(out["any"])

    def test_one_backend_raising_does_not_break_the_other(self):
        bad_inf = SimpleNamespace()  # missing attributes -> AttributeError
        with _patch_backends(bad_inf, _fake_llama_backend(active = True)):
            out = tv.summarize_resident_chat()
        self.assertIsNone(out["hf"])
        self.assertTrue(out["any"])  # GGUF still detected


# ── can_keep_during_training (auto mode) ─────────────────────────────────────


_BASE_KW = dict(
    model_name = "unsloth/Qwen3-4B",
    hf_token = None,
    training_type = "LoRA/QLoRA",
    load_in_4bit = True,
    batch_size = 2,
    max_seq_length = 2048,
    lora_rank = 16,
    target_modules = None,
    gradient_checkpointing = "unsloth",
    optimizer = "adamw_8bit",
    gpu_ids = None,
)


class TestCanKeepAuto(_GpuCacheResetMixin, unittest.TestCase):
    def _run(
        self,
        auto_return,
        *,
        device = DeviceType.CUDA,
        **overrides,
    ):
        kw = {**_BASE_KW, **overrides}
        with (
            patch("utils.hardware.get_device", return_value = device),
            patch(
                "utils.hardware.auto_select_gpu_ids", return_value = auto_return
            ) as auto_mock,
        ):
            keep, info = tv.can_keep_chat_during_training(**kw)
        return keep, info, auto_mock

    def test_keep_when_abundant(self):
        # required 10 -> threshold 10*1.15+4 = 15.5; usable 30 >> 15.5.
        meta = {"selection_mode": "auto", "required_gb": 10.0, "usable_gb": 30.0}
        keep, info, _ = self._run(([1], meta))
        self.assertTrue(keep)
        self.assertEqual(info["mode"], "auto")

    def test_unload_when_within_margin(self):
        # required 10 -> threshold 15.5; usable 15.0 fits raw but not the margin.
        meta = {"selection_mode": "auto", "required_gb": 10.0, "usable_gb": 15.0}
        keep, _, _ = self._run(([0], meta))
        self.assertFalse(keep)

    def test_unload_on_fallback_all(self):
        meta = {
            "selection_mode": "fallback_all",
            "required_gb": 10.0,
            "usable_gb": 100.0,
        }
        keep, _, _ = self._run(([0, 1], meta))
        self.assertFalse(keep)

    def test_unload_when_estimate_unavailable(self):
        meta = {"selection_mode": "auto", "required_gb": None, "usable_gb": None}
        keep, _, _ = self._run((None, meta))
        self.assertFalse(keep)

    def test_unload_on_non_cuda(self):
        keep, info, auto_mock = self._run(([0], {}), device = DeviceType.CPU)
        self.assertFalse(keep)
        self.assertEqual(info["mode"], "non_cuda")
        auto_mock.assert_not_called()

    def test_full_finetuning_forces_16bit_in_estimate(self):
        meta = {"selection_mode": "auto", "required_gb": 10.0, "usable_gb": 30.0}
        _keep, _info, auto_mock = self._run(
            ([0], meta), training_type = "Full Finetuning", load_in_4bit = True
        )
        self.assertFalse(auto_mock.call_args.kwargs["load_in_4bit"])

    def test_hf_token_forwarded(self):
        meta = {"selection_mode": "auto", "required_gb": 10.0, "usable_gb": 30.0}
        _keep, _info, auto_mock = self._run(([0], meta), hf_token = "hf_secret")
        self.assertEqual(auto_mock.call_args.kwargs["hf_token"], "hf_secret")

    def test_probe_exception_defaults_to_unload(self):
        kw = {**_BASE_KW}
        with (
            patch("utils.hardware.get_device", return_value = DeviceType.CUDA),
            patch(
                "utils.hardware.auto_select_gpu_ids", side_effect = RuntimeError("boom")
            ),
        ):
            keep, info = tv.can_keep_chat_during_training(**kw)
        self.assertFalse(keep)
        self.assertEqual(info["reason"], "probe_error")


# ── can_keep_during_training (explicit GPU mode) ─────────────────────────────


class TestCanKeepExplicit(_GpuCacheResetMixin, unittest.TestCase):
    def _run(
        self,
        *,
        required,
        devices,
        resolved,
        gpu_ids,
        est_meta = None,
        resolve_side_effect = None,
    ):
        kw = {**_BASE_KW, "gpu_ids": gpu_ids}
        resolve_kwargs = (
            {"side_effect": resolve_side_effect}
            if resolve_side_effect
            else {"return_value": resolved}
        )
        with (
            patch("utils.hardware.get_device", return_value = DeviceType.CUDA),
            patch(
                "utils.hardware.estimate_required_model_memory_gb",
                return_value = (required, est_meta or {}),
            ),
            patch(
                "utils.hardware.get_visible_gpu_utilization",
                return_value = {"devices": devices},
            ),
            patch("utils.hardware.resolve_requested_gpu_ids", **resolve_kwargs),
            patch("utils.hardware.auto_select_gpu_ids") as auto_mock,
        ):
            keep, info = tv.can_keep_chat_during_training(**kw)
        return keep, info, auto_mock

    def test_keep_when_chosen_gpu_has_room(self):
        devices = [{"index": 0, "vram_total_gb": 80.0, "vram_used_gb": 20.0}]
        keep, info, auto_mock = self._run(
            required = 30.0, devices = devices, resolved = [0], gpu_ids = [0]
        )
        # free 60 >= 30*1.15+4 = 38.5
        self.assertTrue(keep)
        self.assertEqual(info["mode"], "explicit")
        auto_mock.assert_not_called()  # explicit mode never calls the selector

    def test_unload_when_chosen_gpu_too_tight(self):
        devices = [{"index": 0, "vram_total_gb": 24.0, "vram_used_gb": 20.0}]
        keep, _, _ = self._run(
            required = 10.0, devices = devices, resolved = [0], gpu_ids = [0]
        )
        # free 4 < 10*1.15+4 = 15.5
        self.assertFalse(keep)

    def test_multi_gpu_overhead_applied(self):
        # frees [20, 10]; without overhead usable=30, with overhead=20+10*0.85=28.5.
        # required 22 -> threshold 22*1.15+4 = 29.3. 28.5 < 29.3 -> unload, proving
        # the 0.85 overhead was applied (raw 30 would have kept).
        devices = [
            {"index": 0, "vram_total_gb": 24.0, "vram_used_gb": 4.0},
            {"index": 1, "vram_total_gb": 24.0, "vram_used_gb": 14.0},
        ]
        keep, info, _ = self._run(
            required = 22.0, devices = devices, resolved = [0, 1], gpu_ids = [0, 1]
        )
        self.assertFalse(keep)
        self.assertAlmostEqual(info["usable_gb"], 28.5, places = 3)

    def test_requested_gpu_missing_from_devices_counts_as_zero(self):
        devices = [{"index": 0, "vram_total_gb": 80.0, "vram_used_gb": 5.0}]
        # resolved [3] is absent -> free 0 -> unload.
        keep, _, _ = self._run(required = 5.0, devices = devices, resolved = [3], gpu_ids = [3])
        self.assertFalse(keep)

    def test_unload_when_estimate_none(self):
        with (
            patch("utils.hardware.get_device", return_value = DeviceType.CUDA),
            patch(
                "utils.hardware.estimate_required_model_memory_gb",
                return_value = (None, {}),
            ),
            patch("utils.hardware.resolve_requested_gpu_ids", return_value = [0]),
        ):
            keep, info = tv.can_keep_chat_during_training(
                **{**_BASE_KW, "gpu_ids": [0]}
            )
        self.assertFalse(keep)
        self.assertEqual(info["reason"], "estimate_unavailable")

    def test_per_gpu_floor_blocks_uneven_explicit_split(self):
        # free [45, 10]: aggregate 53.5 >= 50 passes, but GPU1's 10 < per-GPU
        # floor 25 -> unload (the tight GPU would OOM).
        devices = [
            {"index": 0, "vram_total_gb": 80.0, "vram_used_gb": 35.0},  # 45 free
            {"index": 1, "vram_total_gb": 80.0, "vram_used_gb": 70.0},  # 10 free
        ]
        keep, info, _ = self._run(
            required = 40.0,
            devices = devices,
            resolved = [0, 1],
            gpu_ids = [0, 1],
            est_meta = {"vram_breakdown": {"min_per_gpu_2": 25.0}},
        )
        self.assertFalse(keep)
        self.assertAlmostEqual(info["min_free_gb"], 10.0, places = 3)

    def test_per_gpu_floor_passes_when_even(self):
        # Same aggregate, but both GPUs clear the 25 GB per-GPU floor -> keep.
        devices = [
            {"index": 0, "vram_total_gb": 80.0, "vram_used_gb": 45.0},  # 35 free
            {"index": 1, "vram_total_gb": 80.0, "vram_used_gb": 50.0},  # 30 free
        ]
        keep, _, _ = self._run(
            required = 40.0,
            devices = devices,
            resolved = [0, 1],
            gpu_ids = [0, 1],
            est_meta = {"vram_breakdown": {"min_per_gpu_2": 25.0}},
        )
        self.assertTrue(keep)

    def test_invalid_gpu_ids_keeps_chat_instead_of_unloading(self):
        # resolve raising -> request will 400 before training, so leave chat alone.
        keep, info, _ = self._run(
            required = 5.0,
            devices = [],
            resolved = None,
            gpu_ids = [99],
            resolve_side_effect = ValueError("Invalid gpu_ids [99]"),
        )
        self.assertTrue(keep)
        self.assertEqual(info["reason"], "invalid_gpu_ids")


# ── free_chat_models_for_training ────────────────────────────────────────────


class TestFreeChatModels(_GpuCacheResetMixin, unittest.TestCase):
    def test_unloads_both_backends(self):
        inf = _fake_inference_backend(active = "unsloth/Qwen3-4B")
        llama = _fake_llama_backend(active = True, identifier = "gemma.gguf")
        with _patch_backends(inf, llama):
            freed = tv.free_chat_models_for_training(reason = "test")
        inf._shutdown_subprocess.assert_called_once()
        llama.unload_model.assert_called_once()
        self.assertIn("hf:unsloth/Qwen3-4B", freed)
        self.assertIn("gguf:gemma.gguf", freed)
        # State cleared so a later resident check is accurate.
        self.assertIsNone(inf.active_model_name)
        self.assertEqual(inf.models, {})
        self.assertEqual(inf.loading_models, set())

    def test_unloads_gguf_only(self):
        inf = _fake_inference_backend()  # nothing resident
        llama = _fake_llama_backend(active = True, identifier = "gemma.gguf")
        with _patch_backends(inf, llama):
            freed = tv.free_chat_models_for_training(reason = "test")
        inf._shutdown_subprocess.assert_not_called()
        llama.unload_model.assert_called_once()
        self.assertEqual(freed, ["gguf:gemma.gguf"])

    def test_leaves_cpu_only_gguf_alone(self):
        # Killing a CPU-only llama-server cannot reclaim VRAM, so don't.
        inf = _fake_inference_backend()
        llama = _fake_llama_backend(
            active = True, identifier = "cpu.gguf", gpu_offload = False
        )
        with _patch_backends(inf, llama):
            freed = tv.free_chat_models_for_training(reason = "test")
        llama.unload_model.assert_not_called()
        self.assertEqual(freed, [])

    def test_unloads_inflight_hf_load(self):
        inf = _fake_inference_backend(active = None, loading = ["unsloth/Qwen3-4B"])
        llama = _fake_llama_backend(active = False)
        with _patch_backends(inf, llama):
            freed = tv.free_chat_models_for_training(reason = "test")
        inf._shutdown_subprocess.assert_called_once()
        self.assertEqual(freed, ["hf:unsloth/Qwen3-4B"])

    def test_nothing_to_free(self):
        inf = _fake_inference_backend()
        llama = _fake_llama_backend(active = False)
        with _patch_backends(inf, llama):
            freed = tv.free_chat_models_for_training(reason = "test")
        self.assertEqual(freed, [])

    def test_hf_failure_still_unloads_gguf(self):
        bad_inf = SimpleNamespace()  # AttributeError on access
        llama = _fake_llama_backend(active = True, identifier = "gemma.gguf")
        with _patch_backends(bad_inf, llama):
            freed = tv.free_chat_models_for_training(reason = "test")
        llama.unload_model.assert_called_once()
        self.assertEqual(freed, ["gguf:gemma.gguf"])


if __name__ == "__main__":
    unittest.main()
