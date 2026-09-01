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

# Load training_vram.py standalone so importing it does not pull the heavy routes/__init__.py.
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
    inf._shutdown_subprocess_mock = inf._shutdown_subprocess
    return inf


def _fake_llama_backend(
    *,
    active = False,
    identifier = "model.gguf",
    gpu_offload = None,
    loaded = None,
):
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
    return patch.dict(sys.modules, {"core.inference": core_inf, "routes.inference": routes_inf})


def _fake_stt_sidecar(
    *,
    model = None,
    device = None,
    loading = False,
):
    sidecar = SimpleNamespace(
        loaded_model = model,
        device = device,
        is_loading = lambda: loading,
    )
    sidecar.cancel_pending_load = MagicMock(return_value = loading)
    sidecar.wait_for_load_to_settle = MagicMock()
    sidecar.unload = MagicMock()
    return sidecar


def _fake_ggml_sidecar(
    *,
    model = None,
    device = None,
    loading = False,
):
    ggml = SimpleNamespace(
        loaded_model = model,
        device = device,
        is_loading = lambda: loading,
    )
    ggml.cancel_pending_load = MagicMock(return_value = loading)
    ggml.wait_for_load_to_settle = MagicMock()
    ggml.unload = MagicMock()
    return ggml


def _patch_stt(sidecar):
    stt_module = types.ModuleType("core.inference.stt_sidecar")
    stt_module.get_stt_sidecar = lambda: sidecar
    # A fresh import of the GGUF sidecar pulls names from the fake module, so fake it too and pin ordering.
    ggml_module = types.ModuleType("core.inference.stt_ggml_sidecar")
    empty_ggml = _fake_ggml_sidecar()
    ggml_module.get_ggml_stt_sidecar = lambda: empty_ggml
    return patch.dict(
        sys.modules,
        {
            "core.inference.stt_sidecar": stt_module,
            "core.inference.stt_ggml_sidecar": ggml_module,
        },
    )


def _patch_ggml_stt(sidecar):
    ggml_module = types.ModuleType("core.inference.stt_ggml_sidecar")
    ggml_module.get_ggml_stt_sidecar = lambda: sidecar
    return patch.dict(sys.modules, {"core.inference.stt_ggml_sidecar": ggml_module})




class TestSummarizeResidentChat(_GpuCacheResetMixin, unittest.TestCase):
    def test_nothing_resident(self):
        with _patch_backends(_fake_inference_backend(), _fake_llama_backend(active = False)):
            self.assertEqual(
                tv.summarize_resident_chat(),
                {"hf": None, "gguf": None, "loading": False, "any": False},
            )

    def test_hf_resident_via_active_model(self):
        with _patch_backends(
            _fake_inference_backend(active = "unsloth/Qwen3-4B"), _fake_llama_backend(active = False)
        ):
            out = tv.summarize_resident_chat()
        self.assertEqual(out["hf"], "unsloth/Qwen3-4B")
        self.assertFalse(out["loading"])
        self.assertTrue(out["any"])

    def test_hf_resident_while_still_loading(self):
        with _patch_backends(
            _fake_inference_backend(active = None, loading = ["unsloth/Qwen3-4B"]),
            _fake_llama_backend(active = False),
        ):
            out = tv.summarize_resident_chat()
        self.assertEqual(out["hf"], "unsloth/Qwen3-4B")
        self.assertTrue(out["loading"])
        self.assertTrue(out["any"])

    def test_replacement_hf_load_is_in_flight(self):
        with _patch_backends(
            _fake_inference_backend(active = "unsloth/old", loading = ["unsloth/new"]),
            _fake_llama_backend(active = False),
        ):
            out = tv.summarize_resident_chat()
        self.assertEqual(out["hf"], "unsloth/old")
        self.assertTrue(out["loading"])

    def test_cpu_only_gguf_is_not_a_vram_resident(self):
        with _patch_backends(
            _fake_inference_backend(),
            _fake_llama_backend(active = True, identifier = "cpu.gguf", gpu_offload = False),
        ):
            out = tv.summarize_resident_chat()
        self.assertIsNone(out["gguf"])
        self.assertFalse(out["any"])

    def test_mid_start_gguf_is_in_flight(self):
        with _patch_backends(
            _fake_inference_backend(),
            _fake_llama_backend(active = True, loaded = False, identifier = "starting.gguf"),
        ):
            out = tv.summarize_resident_chat()
        self.assertEqual(out["gguf"], "starting.gguf")
        self.assertTrue(out["loading"])

    def test_bare_alive_subprocess_without_model_is_not_resident(self):
        with _patch_backends(
            _fake_inference_backend(active = None, alive = True), _fake_llama_backend(active = False)
        ):
            out = tv.summarize_resident_chat()
        self.assertIsNone(out["hf"])
        self.assertFalse(out["any"])

    def test_gguf_resident(self):
        with _patch_backends(
            _fake_inference_backend(), _fake_llama_backend(active = True, identifier = "gemma.gguf")
        ):
            out = tv.summarize_resident_chat()
        self.assertEqual(out["gguf"], "gemma.gguf")
        self.assertFalse(out["loading"])
        self.assertTrue(out["any"])

    def test_one_backend_raising_does_not_break_the_other(self):
        bad_inf = SimpleNamespace()
        with _patch_backends(bad_inf, _fake_llama_backend(active = True)):
            out = tv.summarize_resident_chat()
        self.assertIsNone(out["hf"])
        self.assertTrue(out["any"])


class TestSummarizeResidentStt(_GpuCacheResetMixin, unittest.TestCase):
    def test_reports_resident_model(self):
        sidecar = _fake_stt_sidecar(model = "small", device = "cuda")
        with _patch_stt(sidecar):
            out = tv.summarize_resident_stt()
        self.assertEqual(out["model"], "small")
        self.assertEqual(out["device"], "cuda")
        self.assertTrue(out["any"])
        self.assertFalse(out["loading"])

    def test_reports_inflight_load(self):
        sidecar = _fake_stt_sidecar(loading = True)
        with _patch_stt(sidecar):
            out = tv.summarize_resident_stt()
        self.assertTrue(out["any"])
        self.assertTrue(out["loading"])

    def test_reports_empty_sidecar(self):
        with _patch_stt(_fake_stt_sidecar()):
            out = tv.summarize_resident_stt()
        self.assertFalse(out["any"])

    def test_reports_resident_gguf_when_transformers_idle(self):
        ggml = _fake_ggml_sidecar(model = "small", device = "whisper.cpp")
        with _patch_stt(_fake_stt_sidecar()), _patch_ggml_stt(ggml):
            out = tv.summarize_resident_stt()
        self.assertEqual(out["model"], "small")
        self.assertEqual(out["device"], "whisper.cpp")
        self.assertTrue(out["any"])

    def test_resident_transformers_does_not_mask_loading_gguf(self):
        # A CPU-resident Transformers model holds no VRAM, but a GGUF whisper-server binding its backend does.
        sidecar = _fake_stt_sidecar(model = "small", device = "cpu")
        ggml = _fake_ggml_sidecar(loading = True)
        with _patch_stt(sidecar), _patch_ggml_stt(ggml):
            out = tv.summarize_resident_stt()
        self.assertEqual(out["model"], "small")
        self.assertTrue(out["loading"])
        self.assertTrue(out["any"])




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
            patch("utils.hardware.auto_select_gpu_ids", return_value = auto_return) as auto_mock,
        ):
            keep, info = tv.can_keep_chat_during_training(**kw)
        return keep, info, auto_mock

    def test_keep_when_abundant(self):
        meta = {"selection_mode": "auto", "required_gb": 10.0, "usable_gb": 30.0}
        keep, info, _ = self._run(([1], meta))
        self.assertTrue(keep)
        self.assertEqual(info["mode"], "auto")

    def test_unload_when_within_margin(self):
        meta = {"selection_mode": "auto", "required_gb": 10.0, "usable_gb": 15.0}
        keep, _, _ = self._run(([0], meta))
        self.assertFalse(keep)

    def test_unload_on_fallback_all(self):
        meta = {"selection_mode": "fallback_all", "required_gb": 10.0, "usable_gb": 100.0}
        keep, _, _ = self._run(([0, 1], meta))
        self.assertFalse(keep)

    def test_unload_when_estimate_unavailable(self):
        meta = {"selection_mode": "auto", "required_gb": None, "usable_gb": None}
        keep, _, _ = self._run((None, meta))
        self.assertFalse(keep)

    def test_unload_on_non_accelerator(self):
        keep, info, auto_mock = self._run(([0], {}), device = DeviceType.CPU)
        self.assertFalse(keep)
        self.assertEqual(info["mode"], "non_accelerator")
        auto_mock.assert_not_called()

    def test_xpu_gets_sized_like_cuda(self):
        # XPU is a first-class training backend: the keep-guard must size it, not unload it as non-accelerator.
        meta = {"selection_mode": "auto", "required_gb": 10.0, "usable_gb": 30.0}
        keep, info, auto_mock = self._run(([0], meta), device = DeviceType.XPU)
        self.assertTrue(keep)
        self.assertNotEqual(info.get("mode"), "non_accelerator")
        auto_mock.assert_called_once()

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
            patch("utils.hardware.auto_select_gpu_ids", side_effect = RuntimeError("boom")),
        ):
            keep, info = tv.can_keep_chat_during_training(**kw)
        self.assertFalse(keep)
        self.assertEqual(info["reason"], "probe_error")




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
        keep, info, auto_mock = self._run(required = 30.0, devices = devices, resolved = [0], gpu_ids = [0])
        self.assertTrue(keep)
        self.assertEqual(info["mode"], "explicit")
        auto_mock.assert_not_called()

    def test_unload_when_chosen_gpu_too_tight(self):
        devices = [{"index": 0, "vram_total_gb": 24.0, "vram_used_gb": 20.0}]
        keep, _, _ = self._run(required = 10.0, devices = devices, resolved = [0], gpu_ids = [0])
        self.assertFalse(keep)

    def test_multi_gpu_overhead_applied(self):
        # frees [20, 10] gives usable 28.5 at 0.85 against a 29.3 threshold, so this unloads: the overhead applied.
        devices = [
            {"index": 0, "vram_total_gb": 24.0, "vram_used_gb": 4.0},
            {"index": 1, "vram_total_gb": 24.0, "vram_used_gb": 14.0},
        ]
        keep, info, _ = self._run(required = 22.0, devices = devices, resolved = [0, 1], gpu_ids = [0, 1])
        self.assertFalse(keep)
        self.assertAlmostEqual(info["usable_gb"], 28.5, places = 3)

    def test_requested_gpu_missing_from_devices_counts_as_zero(self):
        devices = [{"index": 0, "vram_total_gb": 80.0, "vram_used_gb": 5.0}]
        keep, _, _ = self._run(required = 5.0, devices = devices, resolved = [3], gpu_ids = [3])
        self.assertFalse(keep)

    def test_unload_when_estimate_none(self):
        with (
            patch("utils.hardware.get_device", return_value = DeviceType.CUDA),
            patch("utils.hardware.estimate_required_model_memory_gb", return_value = (None, {})),
            patch("utils.hardware.resolve_requested_gpu_ids", return_value = [0]),
        ):
            keep, info = tv.can_keep_chat_during_training(**{**_BASE_KW, "gpu_ids": [0]})
        self.assertFalse(keep)
        self.assertEqual(info["reason"], "estimate_unavailable")

    def test_per_gpu_floor_blocks_uneven_explicit_split(self):
        # The aggregate passes, but GPU1's 10 is below the per-GPU floor and would OOM.
        devices = [
            {"index": 0, "vram_total_gb": 80.0, "vram_used_gb": 35.0},
            {"index": 1, "vram_total_gb": 80.0, "vram_used_gb": 70.0},
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
        devices = [
            {"index": 0, "vram_total_gb": 80.0, "vram_used_gb": 45.0},
            {"index": 1, "vram_total_gb": 80.0, "vram_used_gb": 50.0},
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
        keep, info, _ = self._run(
            required = 5.0,
            devices = [],
            resolved = None,
            gpu_ids = [99],
            resolve_side_effect = ValueError("Invalid gpu_ids [99]"),
        )
        self.assertTrue(keep)
        self.assertEqual(info["reason"], "invalid_gpu_ids")




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
        self.assertIsNone(inf.active_model_name)
        self.assertEqual(inf.models, {})
        self.assertEqual(inf.loading_models, set())

    def test_unloads_gguf_only(self):
        inf = _fake_inference_backend()
        llama = _fake_llama_backend(active = True, identifier = "gemma.gguf")
        with _patch_backends(inf, llama):
            freed = tv.free_chat_models_for_training(reason = "test")
        inf._shutdown_subprocess.assert_not_called()
        llama.unload_model.assert_called_once()
        self.assertEqual(freed, ["gguf:gemma.gguf"])

    def test_leaves_cpu_only_gguf_alone(self):
        inf = _fake_inference_backend()
        llama = _fake_llama_backend(active = True, identifier = "cpu.gguf", gpu_offload = False)
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
        bad_inf = SimpleNamespace()
        llama = _fake_llama_backend(active = True, identifier = "gemma.gguf")
        with _patch_backends(bad_inf, llama):
            freed = tv.free_chat_models_for_training(reason = "test")
        llama.unload_model.assert_called_once()
        self.assertEqual(freed, ["gguf:gemma.gguf"])


class TestFreeSttModel(_GpuCacheResetMixin, unittest.TestCase):
    def test_unloads_resident_model(self):
        sidecar = _fake_stt_sidecar(model = "small", device = "cuda")
        with _patch_stt(sidecar):
            freed = tv.free_stt_model_for_training(reason = "test")
        sidecar.unload.assert_called_once()
        self.assertEqual(freed, ["stt:small"])

    def test_cancels_inflight_load_and_waits_to_settle(self):
        sidecar = _fake_stt_sidecar(loading = True)
        with _patch_stt(sidecar):
            freed = tv.free_stt_model_for_training(reason = "test")
        sidecar.cancel_pending_load.assert_called_once()
        # The cancelled loader may still hold VRAM, so wait for it to release.
        sidecar.wait_for_load_to_settle.assert_called_once()
        sidecar.unload.assert_not_called()
        self.assertEqual(freed, ["stt:loading"])

    def test_cancels_inflight_load_then_unloads_settled_model(self):
        # A load that finished before seeing the cancel leaves a resident model, which must be unloaded.
        sidecar = _fake_stt_sidecar(model = "small", loading = True)
        with _patch_stt(sidecar):
            freed = tv.free_stt_model_for_training(reason = "test")
        sidecar.cancel_pending_load.assert_called_once()
        sidecar.wait_for_load_to_settle.assert_called_once()
        sidecar.unload.assert_called_once()
        self.assertEqual(freed, ["stt:loading"])

    def test_cancelled_load_still_unloads_gguf_sidecar(self):
        # Cancelling a Transformers load must not skip the GGUF sidecar: both engines can hold memory at once.
        sidecar = _fake_stt_sidecar(loading = True)
        ggml = _fake_ggml_sidecar(model = "small")
        with _patch_stt(sidecar), _patch_ggml_stt(ggml):
            freed = tv.free_stt_model_for_training(reason = "test")
        sidecar.cancel_pending_load.assert_called_once()
        ggml.unload.assert_called_once()
        self.assertEqual(freed, ["stt:loading", "stt:small"])

    def test_leaves_empty_sidecar_alone(self):
        sidecar = _fake_stt_sidecar()
        with _patch_stt(sidecar):
            freed = tv.free_stt_model_for_training(reason = "test")
        sidecar.unload.assert_not_called()
        self.assertEqual(freed, [])

    def test_cancels_inflight_gguf_load_and_waits_to_settle(self):
        # A GGUF whisper-server still starting has no loaded_model, so the coordinator cancels and waits.
        sidecar = _fake_stt_sidecar()
        ggml = _fake_ggml_sidecar(loading = True)
        with _patch_stt(sidecar), _patch_ggml_stt(ggml):
            freed = tv.free_stt_model_for_training(reason = "test")
        ggml.cancel_pending_load.assert_called_once()
        ggml.wait_for_load_to_settle.assert_called_once()
        ggml.unload.assert_not_called()
        self.assertEqual(freed, ["stt:gguf-loading"])


class TestCoordinateModels(_GpuCacheResetMixin, unittest.TestCase):
    def _run(self, chat, stt, keep_results):
        keep = MagicMock(side_effect = keep_results)
        with (
            patch.object(tv, "summarize_resident_chat", return_value = chat),
            patch.object(tv, "summarize_resident_stt", return_value = stt),
            patch.object(
                tv,
                "free_stt_model_for_training",
                return_value = ["stt:small"],
            ) as free_stt,
            patch.object(
                tv,
                "free_chat_models_for_training",
                return_value = ["hf:chat"],
            ) as free_chat,
        ):
            freed = tv.coordinate_models_for_training(keep)
        return freed, keep, free_stt, free_chat

    def test_keeps_everything_when_training_fits(self):
        chat = {"any": True, "loading": False}
        stt = {"any": True, "loading": False}
        freed, keep, free_stt, free_chat = self._run(
            chat,
            stt,
            [(True, {"usable_gb": 40, "required_gb": 10})],
        )
        self.assertEqual(freed, [])
        keep.assert_called_once()
        free_stt.assert_not_called()
        free_chat.assert_not_called()

    def test_frees_stt_before_chat(self):
        chat = {"any": True, "loading": False}
        stt = {"any": True, "loading": False}
        freed, keep, free_stt, free_chat = self._run(
            chat,
            stt,
            [
                (False, {"usable_gb": 8, "required_gb": 10}),
                (True, {"usable_gb": 12, "required_gb": 10}),
            ],
        )
        self.assertEqual(freed, ["stt:small"])
        self.assertEqual(keep.call_count, 2)
        free_stt.assert_called_once()
        free_chat.assert_not_called()

    def test_frees_chat_when_stt_is_not_enough(self):
        chat = {"any": True, "loading": False}
        stt = {"any": True, "loading": False}
        freed, keep, free_stt, free_chat = self._run(
            chat,
            stt,
            [
                (False, {"usable_gb": 8, "required_gb": 10}),
                (False, {"usable_gb": 9, "required_gb": 10}),
            ],
        )
        self.assertEqual(freed, ["stt:small", "hf:chat"])
        self.assertEqual(keep.call_count, 2)
        free_stt.assert_called_once()
        free_chat.assert_called_once()

    def test_frees_loading_models_without_probe(self):
        chat = {"any": True, "loading": True}
        stt = {"any": True, "loading": True}
        freed, keep, free_stt, free_chat = self._run(chat, stt, [])
        self.assertEqual(freed, ["stt:small", "hf:chat"])
        keep.assert_not_called()
        free_stt.assert_called_once()
        free_chat.assert_called_once()

    def test_cancels_loading_stt_without_probe(self):
        chat = {"any": False, "loading": False}
        stt = {"any": True, "loading": True}
        freed, keep, free_stt, free_chat = self._run(chat, stt, [])
        self.assertEqual(freed, ["stt:small"])
        keep.assert_not_called()
        free_stt.assert_called_once()
        free_chat.assert_not_called()


if __name__ == "__main__":
    unittest.main()
