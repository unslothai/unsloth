# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Export checkpoint loading must shard across every visible GPU (#7053): the
``device_map="sequential"`` loader default stacks the whole model on GPU0 and OOMs
while the other GPUs sit empty. The loader now passes whichever sharding map
``get_device_map`` resolves (``"unsloth"`` on CUDA, ``"balanced"`` elsewhere), but only
on a real multi-GPU host, so single-GPU, CPU and MLX are untouched."""

from __future__ import annotations

import contextlib
import sys
import types
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

# Reuse the absolute-paths stub harness: loads core/export/export.py without torch/unsloth.
from test_export_absolute_paths import (  # noqa: E402
    _install_export_backend_stubs,
    _load_module,
)


def _export_mod(monkeypatch):
    _install_export_backend_stubs(monkeypatch)
    return _load_module("test_core_export_backend_device_map", "core/export/export.py", monkeypatch)


def _stub_hardware(monkeypatch, visible, device_map):
    hw = sys.modules["utils.hardware"]
    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: visible, raising = False)
    monkeypatch.setattr(hw, "get_device_map", lambda ids: device_map, raising = False)


# ── _multi_gpu_device_map_kwargs ──


def test_multi_gpu_host_gets_balanced(monkeypatch):
    mod = _export_mod(monkeypatch)
    monkeypatch.setattr(mod, "_IS_MLX", False)
    _stub_hardware(monkeypatch, [0, 1, 2], "balanced")
    assert mod._multi_gpu_device_map_kwargs() == {"device_map": "balanced"}


def test_multi_gpu_cuda_host_passes_the_unsloth_planner_through(monkeypatch):
    """CUDA multi-GPU now resolves to "unsloth". A whitelist of just "balanced"
    dropped the map and left the export on the loader default, which is the
    single-GPU stacking this file exists to prevent."""
    mod = _export_mod(monkeypatch)
    monkeypatch.setattr(mod, "_IS_MLX", False)
    _stub_hardware(monkeypatch, [0, 1, 2], "unsloth_balanced")
    assert mod._multi_gpu_device_map_kwargs() == {"device_map": "unsloth_balanced"}


def test_single_gpu_host_keeps_loader_default(monkeypatch):
    mod = _export_mod(monkeypatch)
    monkeypatch.setattr(mod, "_IS_MLX", False)
    _stub_hardware(monkeypatch, [0], "sequential")
    assert mod._multi_gpu_device_map_kwargs() == {}


def test_non_balanced_resolution_keeps_loader_default(monkeypatch):
    # >1 visible id but a non-CUDA device resolves to "sequential": pass nothing.
    mod = _export_mod(monkeypatch)
    monkeypatch.setattr(mod, "_IS_MLX", False)
    _stub_hardware(monkeypatch, [0, 1], "sequential")
    assert mod._multi_gpu_device_map_kwargs() == {}


def test_uuid_mig_mask_falls_back_to_count_detection(monkeypatch):
    # UUID/MIG masks resolve to no numeric ids, but get_device_map(None) still sees >1 GPU.
    mod = _export_mod(monkeypatch)
    monkeypatch.setattr(mod, "_IS_MLX", False)
    hw = sys.modules["utils.hardware"]
    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: [], raising = False)
    monkeypatch.setattr(
        hw,
        "get_device_map",
        lambda ids: "balanced" if ids is None else "sequential",
        raising = False,
    )
    assert mod._multi_gpu_device_map_kwargs() == {"device_map": "balanced"}


def test_no_visible_gpus_keeps_loader_default(monkeypatch):
    # Empty mask / CPU host: get_device_map(None) resolves "sequential" -> {}.
    mod = _export_mod(monkeypatch)
    monkeypatch.setattr(mod, "_IS_MLX", False)
    hw = sys.modules["utils.hardware"]
    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: [], raising = False)
    monkeypatch.setattr(hw, "get_device_map", lambda ids: "sequential", raising = False)
    assert mod._multi_gpu_device_map_kwargs() == {}


def test_mlx_host_keeps_loader_default(monkeypatch):
    mod = _export_mod(monkeypatch)
    # The stubs set _IS_MLX = True; even a multi-GPU view must yield no device_map.
    _stub_hardware(monkeypatch, [0, 1], "balanced")
    assert mod._multi_gpu_device_map_kwargs() == {}


def test_hardware_probe_failure_keeps_loader_default(monkeypatch):
    mod = _export_mod(monkeypatch)
    monkeypatch.setattr(mod, "_IS_MLX", False)
    hw = sys.modules["utils.hardware"]

    def _boom():
        raise RuntimeError("no GPUs")

    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", _boom, raising = False)
    assert mod._multi_gpu_device_map_kwargs() == {}


# ── load_checkpoint forwards the kwargs to from_pretrained ──


class _RecordingLoader:
    calls: list[dict] = []

    @classmethod
    def from_pretrained(cls, **kwargs):
        cls.calls.append(kwargs)
        return types.SimpleNamespace(), types.SimpleNamespace()


def _load_text_checkpoint(monkeypatch, tmp_path, device_map_kwargs):
    mod = _export_mod(monkeypatch)
    _RecordingLoader.calls = []
    monkeypatch.setattr(mod, "FastLanguageModel", _RecordingLoader)
    monkeypatch.setattr(mod, "detect_audio_type", lambda *a, **k: None)
    monkeypatch.setattr(mod, "is_vision_model", lambda *a, **k: False)
    monkeypatch.setattr(mod, "_hf_offline", lambda *a, **k: False)
    monkeypatch.setattr(mod, "_offline_window_if", lambda flag: contextlib.nullcontext())
    monkeypatch.setattr(mod, "_multi_gpu_device_map_kwargs", lambda: device_map_kwargs)

    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    backend = mod.ExportBackend.__new__(mod.ExportBackend)
    backend.cleanup_memory = lambda: None
    ok, message = backend.load_checkpoint(str(checkpoint))
    assert ok, message
    assert len(_RecordingLoader.calls) == 1
    return _RecordingLoader.calls[0]


def test_load_checkpoint_forwards_balanced_device_map(monkeypatch, tmp_path):
    kwargs = _load_text_checkpoint(monkeypatch, tmp_path, {"device_map": "balanced"})
    assert kwargs["device_map"] == "balanced"


def test_load_checkpoint_omits_device_map_on_single_gpu(monkeypatch, tmp_path):
    kwargs = _load_text_checkpoint(monkeypatch, tmp_path, {})
    assert "device_map" not in kwargs  # loader default (sequential) untouched


def test_load_checkpoint_repairs_legacy_cache_identity_without_rewriting_adapter(
    monkeypatch, tmp_path
):
    mod = _export_mod(monkeypatch)
    snapshot = (
        tmp_path
        / "cache"
        / "models--unsloth--Llama-3.2-1B-Instruct"
        / "snapshots"
        / "0123456789abcdef"
    )
    snapshot.mkdir(parents = True)
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    adapter_path = checkpoint / "adapter_config.json"
    original_adapter = '{"base_model_name_or_path":"' + str(snapshot) + '","r":16}\n'
    adapter_path.write_text(original_adapter, encoding = "utf-8")

    class _LegacyAdapterLoader:
        @classmethod
        def from_pretrained(cls, **_kwargs):
            model = types.SimpleNamespace(
                config = types.SimpleNamespace(_name_or_path = str(checkpoint)),
                peft_config = {
                    "default": types.SimpleNamespace(base_model_name_or_path = str(snapshot))
                },
            )
            return model, types.SimpleNamespace()

    monkeypatch.setattr(mod, "FastLanguageModel", _LegacyAdapterLoader)
    monkeypatch.setattr(mod, "get_base_model_from_lora", lambda _path: str(snapshot))
    monkeypatch.setattr(mod, "detect_audio_type", lambda *a, **k: None)
    monkeypatch.setattr(mod, "is_vision_model", lambda *a, **k: False)
    monkeypatch.setattr(mod, "_hf_offline", lambda *a, **k: False)
    monkeypatch.setattr(mod, "_offline_window_if", lambda flag: contextlib.nullcontext())
    monkeypatch.setattr(mod, "_multi_gpu_device_map_kwargs", lambda: {})

    backend = mod.ExportBackend.__new__(mod.ExportBackend)
    backend.cleanup_memory = lambda: None
    ok, message = backend.load_checkpoint(str(checkpoint))

    assert ok, message
    assert (
        backend.current_model.peft_config["default"].base_model_name_or_path
        == "unsloth/Llama-3.2-1B-Instruct"
    )
    assert backend.current_model.config._name_or_path == str(checkpoint)
    assert adapter_path.read_text(encoding = "utf-8") == original_adapter


# ── a load that succeeds but offloads to CPU/disk ──


def test_cpu_offloaded_modules_counts_cpu_and_disk(monkeypatch):
    mod = _export_mod(monkeypatch)
    model = types.SimpleNamespace(hf_device_map = {"a": 0, "b": "cpu", "c": 1, "d": "disk"})
    assert mod._cpu_offloaded_modules(model) == 2


def test_cpu_offloaded_modules_ignores_gpu_only_and_missing_maps(monkeypatch):
    mod = _export_mod(monkeypatch)
    assert mod._cpu_offloaded_modules(types.SimpleNamespace(hf_device_map = {"a": 0})) == 0
    assert mod._cpu_offloaded_modules(types.SimpleNamespace(hf_device_map = None)) == 0
    assert mod._cpu_offloaded_modules(types.SimpleNamespace()) == 0


class _SpillThenCleanLoader:
    """First call offloads to CPU (bf16 accepts it silently), second is clean."""

    calls: list[dict] = []

    @classmethod
    def from_pretrained(cls, **kwargs):
        cls.calls.append(kwargs)
        device_map = {"model.layers.0": 0} if len(cls.calls) > 1 else {"model.layers.0": "cpu"}
        return types.SimpleNamespace(hf_device_map = device_map), types.SimpleNamespace()


def _run_spill_loader(monkeypatch, tmp_path, device_map_kwargs):
    mod = _export_mod(monkeypatch)
    _SpillThenCleanLoader.calls = []
    monkeypatch.setattr(mod, "FastLanguageModel", _SpillThenCleanLoader)
    monkeypatch.setattr(mod, "detect_audio_type", lambda *a, **k: None)
    monkeypatch.setattr(mod, "is_vision_model", lambda *a, **k: False)
    monkeypatch.setattr(mod, "_hf_offline", lambda *a, **k: False)
    monkeypatch.setattr(mod, "_offline_window_if", lambda flag: contextlib.nullcontext())
    monkeypatch.setattr(mod, "_multi_gpu_device_map_kwargs", lambda: device_map_kwargs)

    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    backend = mod.ExportBackend.__new__(mod.ExportBackend)
    backend.cleanup_memory = lambda: None
    ok, message = backend.load_checkpoint(str(checkpoint))
    return ok, message, _SpillThenCleanLoader.calls


def test_successful_load_that_offloads_to_cpu_retries_single_device(monkeypatch, tmp_path):
    # Nothing raises, so only hf_device_map catches it; otherwise the params stay on meta and kill the export.
    ok, message, calls = _run_spill_loader(monkeypatch, tmp_path, {"device_map": "balanced"})
    assert ok, message
    assert len(calls) == 2
    assert calls[0]["device_map"] == "balanced"
    # Named, not omitted: an omitted map is unsloth's marked default, which is upgraded
    # back to the planner rather than being the single-device load this retry wants.
    assert calls[1]["device_map"] == "sequential"


def test_single_gpu_offload_is_left_alone(monkeypatch, tmp_path):
    # No multi-GPU map was requested, so there is nothing to retry on.
    ok, message, calls = _run_spill_loader(monkeypatch, tmp_path, {})
    assert ok, message
    assert len(calls) == 1


def test_retry_result_is_kept_even_if_it_also_offloads(monkeypatch, tmp_path):
    # The retry runs with _device_map_override set, so it must never recurse again.
    mod = _export_mod(monkeypatch)

    class _AlwaysSpills:
        calls: list[dict] = []

        @classmethod
        def from_pretrained(cls, **kwargs):
            cls.calls.append(kwargs)
            return types.SimpleNamespace(hf_device_map = {"a": "cpu"}), types.SimpleNamespace()

    monkeypatch.setattr(mod, "FastLanguageModel", _AlwaysSpills)
    monkeypatch.setattr(mod, "detect_audio_type", lambda *a, **k: None)
    monkeypatch.setattr(mod, "is_vision_model", lambda *a, **k: False)
    monkeypatch.setattr(mod, "_hf_offline", lambda *a, **k: False)
    monkeypatch.setattr(mod, "_offline_window_if", lambda flag: contextlib.nullcontext())
    monkeypatch.setattr(
        mod,
        "_multi_gpu_device_map_kwargs",
        lambda: {"device_map": "balanced"},
    )

    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    backend = mod.ExportBackend.__new__(mod.ExportBackend)
    backend.cleanup_memory = lambda: None
    ok, message = backend.load_checkpoint(str(checkpoint))
    assert ok, message
    assert len(_AlwaysSpills.calls) == 2


# ── the "unsloth" planner refusing to place the model ──


def test_is_device_map_infeasible_matches_by_class_name(monkeypatch):
    mod = _export_mod(monkeypatch)

    class DeviceMapInfeasible(RuntimeError):
        pass

    assert mod._is_device_map_infeasible(DeviceMapInfeasible("needs 7.5 GiB free on cuda:0"))
    assert not mod._is_device_map_infeasible(RuntimeError("needs 7.5 GiB free on cuda:0"))


def test_planner_refusal_retries_on_the_single_device_loader(monkeypatch, tmp_path):
    """The planner budgets from memory read before this process opens a context, so a
    training or chat job holding the other cards can make it refuse a model the old
    single-device load still fits. That was a working export before the switch."""
    mod = _export_mod(monkeypatch)

    class DeviceMapInfeasible(RuntimeError):
        pass

    class _RefusesThenLoads:
        calls: list[dict] = []

        @classmethod
        def from_pretrained(cls, **kwargs):
            cls.calls.append(kwargs)
            if len(cls.calls) == 1:
                raise DeviceMapInfeasible("2 x 8 GiB is not enough for the head")
            return types.SimpleNamespace(
                hf_device_map = {"model.layers.0": 0}
            ), types.SimpleNamespace()

    monkeypatch.setattr(mod, "FastLanguageModel", _RefusesThenLoads)
    monkeypatch.setattr(mod, "detect_audio_type", lambda *a, **k: None)
    monkeypatch.setattr(mod, "is_vision_model", lambda *a, **k: False)
    monkeypatch.setattr(mod, "_hf_offline", lambda *a, **k: False)
    monkeypatch.setattr(mod, "_offline_window_if", lambda flag: contextlib.nullcontext())
    monkeypatch.setattr(
        mod, "_multi_gpu_device_map_kwargs", lambda: {"device_map": "unsloth_balanced"}
    )

    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    backend = mod.ExportBackend.__new__(mod.ExportBackend)
    backend.cleanup_memory = lambda: None
    ok, message = backend.load_checkpoint(str(checkpoint))

    assert ok, message
    assert len(_RefusesThenLoads.calls) == 2
    assert _RefusesThenLoads.calls[0]["device_map"] == "unsloth_balanced"
    assert _RefusesThenLoads.calls[1]["device_map"] == "sequential"


def test_an_unrelated_error_is_still_reported_rather_than_retried(monkeypatch, tmp_path):
    # The retry is for placement, not for every failure: a broken checkpoint must not
    # be loaded twice and reported against the second attempt.
    mod = _export_mod(monkeypatch)

    class _AlwaysBroken:
        calls: list[dict] = []

        @classmethod
        def from_pretrained(cls, **kwargs):
            cls.calls.append(kwargs)
            raise ValueError("adapter_config.json is not valid JSON")

    monkeypatch.setattr(mod, "FastLanguageModel", _AlwaysBroken)
    monkeypatch.setattr(mod, "detect_audio_type", lambda *a, **k: None)
    monkeypatch.setattr(mod, "is_vision_model", lambda *a, **k: False)
    monkeypatch.setattr(mod, "_hf_offline", lambda *a, **k: False)
    monkeypatch.setattr(mod, "_offline_window_if", lambda flag: contextlib.nullcontext())
    monkeypatch.setattr(
        mod, "_multi_gpu_device_map_kwargs", lambda: {"device_map": "unsloth_balanced"}
    )

    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    backend = mod.ExportBackend.__new__(mod.ExportBackend)
    backend.cleanup_memory = lambda: None
    ok, message = backend.load_checkpoint(str(checkpoint))

    assert not ok
    assert len(_AlwaysBroken.calls) == 1
    assert "not valid JSON" in message


def test_the_retry_names_sequential_rather_than_omitting_the_device_map(monkeypatch, tmp_path):
    """An omitted device_map is not the loader default any more.

    unsloth's signature default is `DEFAULT_DEVICE_MAP`, a marked "sequential" that
    `requested_device_map` upgrades back to the planner unless UNSLOTH_AUTO_DEVICE_MAP=0.
    So `_device_map_override = {}` would re-run the very placement that just failed and
    report the same error, and the retry would look like a fallback while being none.
    """
    mod = _export_mod(monkeypatch)

    class DeviceMapInfeasible(RuntimeError):
        pass

    class _RefusesAnyPlan:
        calls: list[dict] = []

        @classmethod
        def from_pretrained(cls, **kwargs):
            cls.calls.append(kwargs)
            # Stands in for the planner: it refuses whenever it is the one asked to place.
            if (
                kwargs.get("device_map") in ("unsloth", "unsloth_balanced", None)
                or "device_map" not in kwargs
            ):
                raise DeviceMapInfeasible("needs 7.57 GiB free on cuda:0, has 4.10 GiB")
            return types.SimpleNamespace(
                hf_device_map = {"model.layers.0": 0}
            ), types.SimpleNamespace()

    monkeypatch.setattr(mod, "FastLanguageModel", _RefusesAnyPlan)
    monkeypatch.setattr(mod, "detect_audio_type", lambda *a, **k: None)
    monkeypatch.setattr(mod, "is_vision_model", lambda *a, **k: False)
    monkeypatch.setattr(mod, "_hf_offline", lambda *a, **k: False)
    monkeypatch.setattr(mod, "_offline_window_if", lambda flag: contextlib.nullcontext())
    monkeypatch.setattr(
        mod, "_multi_gpu_device_map_kwargs", lambda: {"device_map": "unsloth_balanced"}
    )

    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    backend = mod.ExportBackend.__new__(mod.ExportBackend)
    backend.cleanup_memory = lambda: None
    ok, message = backend.load_checkpoint(str(checkpoint))

    assert ok, message
    assert len(_RefusesAnyPlan.calls) == 2
    assert _RefusesAnyPlan.calls[0]["device_map"] == "unsloth_balanced"
    assert _RefusesAnyPlan.calls[1]["device_map"] == "sequential"
