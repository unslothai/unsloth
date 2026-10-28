# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""sd.cpp graph-cut VRAM budget and the H3 CUDA compute-capability fallback.

The committed H3 flags keep ``--max-vram -1`` (sd.cpp's own free-VRAM reading) as a placeholder; each process launch
replaces it with an explicit budget read by Studio right before the spawn, and a device OOM under that budget is
retried once at a smaller one. Hermetic: no binary, no GPU, no socket."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

from core.inference import sd_cpp_args as args_mod
from core.inference import sd_cpp_backend
from core.inference import sd_cpp_server as srv
from core.inference import video as video_mod
from core.inference import video_minimax_h3 as h3
from core.inference.sd_cpp_args import SdCppModelFiles, SdCppVideoGenParams
from core.inference.video import VideoBackend

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

GiB = 1 << 30

# The failure the offload audit recorded (H3 Q4_K, 12 GB card, another process moved VRAM mid-render).
_AUDIT_OOM = (
    "sd-server generation failed: generation_failed: generate_video returned no results\n"
    "Last output:\n"
    "[E] ggml: ggml_backend_cuda_buffer_type_alloc_buffer: allocating 210.08 MiB on device 0: "
    "cudaMalloc failed: out of memory\n"
    "[E] ggml: alloc_tensor_range: failed to allocate CUDA0 buffer of size 220288512\n"
    "[E] minimax_h3 failed to capture graph cut tensor: ggml_runner_cut:minimax_h3.blocks.7|hidden_states|\n"
    "[E] minimax_h3 segment 11/54 (minimax_h3.blocks.7) failed during output caching\n"
)

_FILES = SdCppModelFiles(
    diffusion_model = "/m/h3.gguf",
    vae = "/m/vae.safetensors",
    audio_vae = "/m/audio_vae.safetensors",
    llm = "/m/te.gguf",
)
_COMMITTED = ("--offload-to-cpu", "--diffusion-fa", "--max-vram", "-1", "--stream-layers")


@pytest.fixture(autouse = True)
def _no_kill_switch(monkeypatch):
    monkeypatch.delenv(args_mod.GRAPH_CUT_EXPLICIT_BUDGET_ENV, raising = False)


def test_budget_is_free_minus_reserve_rounded_down():
    assert args_mod.graph_cut_budget_gib(None) is None
    assert args_mod.graph_cut_budget_gib(int(8.0 * GiB)) == 6.0
    assert args_mod.graph_cut_budget_gib(int(7.999 * GiB)) == 5.99
    # Under the floor after the reserve: keep sd.cpp's own reading.
    assert args_mod.graph_cut_budget_gib(int(2.9 * GiB)) is None
    assert args_mod.graph_cut_budget_gib(int(8 * GiB), scale = 0.5) == 3.0


def test_only_the_auto_placeholder_is_replaced():
    assert args_mod.with_max_vram_budget(_COMMITTED, 6.25)[3] == "6.25"
    per_device = args_mod.with_max_vram_budget(_COMMITTED, {"cuda0": 6.0, "cuda1": 12.5})
    assert per_device[3] == "cuda0=6.00,cuda1=12.50"
    # An explicit value (the user's, or an earlier launch's) is never widened or rewritten.
    explicit = ["--max-vram", "4"]
    assert args_mod.with_max_vram_budget(explicit, 20.0) == explicit
    assert args_mod.with_max_vram_budget(["--max-vram", "cuda0=4"], 20.0) == [
        "--max-vram",
        "cuda0=4",
    ]
    assert args_mod.with_max_vram_budget(["--offload-to-cpu"], 6.0) == ["--offload-to-cpu"]
    assert args_mod.with_max_vram_budget(_COMMITTED, None) == list(_COMMITTED)
    assert args_mod.with_max_vram_budget(_COMMITTED, {}) == list(_COMMITTED)


def test_explicit_budget_round_trips_and_scales():
    assert args_mod.explicit_max_vram_gib(_COMMITTED) is None
    assert args_mod.explicit_max_vram_gib(["--max-vram", "6.25"]) == 6.25
    assert args_mod.explicit_max_vram_gib(["--max-vram", "0"]) is None
    assert args_mod.explicit_max_vram_gib(["--max-vram", "cuda0=6,cuda1=12.5"]) == {
        "cuda0": 6.0,
        "cuda1": 12.5,
    }
    assert args_mod.explicit_max_vram_gib(["--max-vram", "garbage"]) is None
    assert args_mod.scaled_vram_budget(8.0, 0.75) == 6.0
    assert args_mod.scaled_vram_budget({"cuda0": 8.0, "cuda1": 4.0}, 0.75) == {
        "cuda0": 6.0,
        "cuda1": 3.0,
    }
    # Below the floor a smaller budget is not a working one.
    assert args_mod.scaled_vram_budget(1.2, 0.75) is None
    assert args_mod.scaled_vram_budget({"cuda0": 8.0, "cuda1": 1.2}, 0.75) is None


def test_device_oom_text():
    assert args_mod.is_device_oom_text(_AUDIT_OOM)
    assert args_mod.is_device_oom_text(
        "ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory"
    )
    assert args_mod.is_device_oom_text("hipMalloc failed: out of memory")
    assert not args_mod.is_device_oom_text(
        "sd-cli exited 1. Last output:\n[E] unknown argument --foo"
    )
    assert not args_mod.is_device_oom_text(None)


def test_a_host_allocation_failure_is_not_a_device_oom():
    # A smaller --max-vram keeps more on the host: it cannot cure exhausted RAM.
    for text in (
        "sd-server generation failed\n[E] host malloc failed: out of memory",
        "terminate called after throwing an instance of 'std::bad_alloc'",
        "[E] vk::Device::allocateMemory: ErrorOutOfHostMemory out_of_host_memory",
        "load failed: out of memory",
    ):
        assert not args_mod.is_device_oom_text(text), text
    # The graph-cut executor's own segment capacity failures are device ones.
    assert args_mod.is_device_oom_text(
        "[E] minimax_h3 segment 3/54 (minimax_h3.blocks.1) failed during workspace capacity check"
    )
    assert args_mod.is_device_oom_text("CUDA error: out of memory\n  current device: 0")


class _Rt:
    def __init__(self, free):
        self.calls = 0
        self._free = free

        def probe():
            self.calls += 1
            if isinstance(self._free, Exception):
                raise self._free
            return self._free

        self.vram_free_probe = probe


def test_launch_flags_read_the_card_at_launch():
    rt = _Rt(int(10 * GiB))
    out = video_mod._h3_budgeted_launch_flags(rt, _COMMITTED)
    assert out[3] == "8.00" and rt.calls == 1
    # A retry cap is a ceiling, never a floor.
    assert video_mod._h3_budgeted_launch_flags(rt, _COMMITTED, cap = 5.5)[3] == "5.50"
    assert video_mod._h3_budgeted_launch_flags(rt, _COMMITTED, cap = 50.0)[3] == "8.00"
    rt = _Rt({"cuda0": int(10 * GiB), "cuda1": int(16 * GiB)})
    assert video_mod._h3_budgeted_launch_flags(rt, _COMMITTED)[3] == "cuda0=8.00,cuda1=14.00"


@pytest.mark.parametrize("free", [None, int(2 * GiB), RuntimeError("nvidia-smi timed out")])
def test_unknown_or_tiny_reading_keeps_the_placeholder(free):
    rt = _Rt(free)
    assert video_mod._h3_budgeted_launch_flags(rt, _COMMITTED) == list(_COMMITTED)


@pytest.mark.parametrize("free", [None, RuntimeError("nvidia-smi timed out")])
def test_an_unreadable_card_still_retries_below_the_budget_that_failed(free):
    assert video_mod._h3_budgeted_launch_flags(_Rt(free), _COMMITTED, cap = 4.5)[3] == "4.50"


def test_a_known_small_reading_never_takes_a_larger_retry_cap():
    # 2 GiB free now: a 4.5 / 6 GiB cap from the earlier launch would exceed the card; sd.cpp's own -1 does not.
    for cap in (4.5, 6.0, {"cuda0": 6.0}):
        assert video_mod._h3_budgeted_launch_flags(_Rt(int(2 * GiB)), _COMMITTED, cap = cap) == list(
            _COMMITTED
        )
    rt = _Rt({"cuda0": int(10 * GiB), "cuda1": int(2 * GiB)})
    assert video_mod._h3_budgeted_launch_flags(
        rt, _COMMITTED, cap = {"cuda0": 6.0, "cuda1": 6.0}
    ) == list(_COMMITTED)


def test_a_uma_pinned_weight_reserve_is_never_replaced():
    """#13786: a negative --max-vram other than the -1 placeholder is sd.cpp's reading less the GTT-pinned weights; an
    explicit free-minus-2-GiB budget in its place would hand those weights' memory back to the graph."""
    uma = ["--offload-to-cpu", "--diffusion-fa", "--max-vram", "-17.05", "--stream-layers"]
    rt = _Rt(int(22 * GiB))
    assert video_mod._h3_budgeted_launch_flags(rt, uma) == uma
    assert video_mod._h3_budgeted_launch_flags(rt, uma, cap = 6.0) == uma
    assert rt.calls == 0
    assert args_mod.with_max_vram_budget(uma, 8.0) == uma
    assert args_mod.explicit_max_vram_gib(uma) is None
    assert video_mod._h3_retry_vram_cap(uma, _AUDIT_OOM) is None
    assert args_mod.max_vram_is_placeholder("-1") and not args_mod.max_vram_is_placeholder("-17.05")


def test_kill_switch_and_no_probe_keep_the_placeholder(monkeypatch):
    class _NoProbe:
        pass

    assert video_mod._h3_budgeted_launch_flags(_NoProbe(), _COMMITTED) == list(_COMMITTED)
    monkeypatch.setenv(args_mod.GRAPH_CUT_EXPLICIT_BUDGET_ENV, "0")
    rt = _Rt(int(10 * GiB))
    assert video_mod._h3_budgeted_launch_flags(rt, _COMMITTED) == list(_COMMITTED)
    assert rt.calls == 0


def test_flags_without_graph_cut_are_untouched():
    rt = _Rt(int(10 * GiB))
    flags = ["--offload-to-cpu", "--diffusion-fa"]
    assert video_mod._h3_budgeted_launch_flags(rt, flags) == flags
    assert rt.calls == 0


def test_retry_cap():
    launched = ["--offload-to-cpu", "--max-vram", "8.00"]
    assert video_mod._h3_retry_vram_cap(launched, _AUDIT_OOM) == 6.0
    assert video_mod._h3_retry_vram_cap(launched, "sd-cli exited 1: bad flag") is None
    assert (
        video_mod._h3_retry_vram_cap(list(_COMMITTED), _AUDIT_OOM) is None
    )  # sd.cpp's own reading: no retry
    assert video_mod._h3_retry_vram_cap(None, _AUDIT_OOM) is None


def test_slot_launches_with_the_budget_but_reuses_on_the_placeholder(monkeypatch):
    started = []

    class _Srv:
        def __init__(self, binary):
            self.alive = False

        def start(
            self,
            files,
            *,
            offload = None,
            threads = None,
            env = None,
            extra_args = None,
        ):
            started.append(list(offload))
            self.alive = True

        def is_alive(self):
            return self.alive

        def stop(self):
            self.alive = False

    monkeypatch.setattr(srv, "SdCppServer", _Srv)
    monkeypatch.setattr("core.inference.sd_cpp_engine.is_managed_binary", lambda b: False)
    slot = h3.H3NativeServerSlot("/x/sd-server", _FILES, _COMMITTED)
    budgets = iter(["8.00", "7.00"])

    def transform(flags):
        out = list(flags)
        out[out.index("--max-vram") + 1] = next(budgets)
        return out

    first = slot.get(_COMMITTED, launch_transform = transform)
    # Same committed flags, a different launch-time budget: the live server is reused, not respawned.
    assert slot.get(_COMMITTED, launch_transform = transform) is first
    assert started == [
        ["--offload-to-cpu", "--diffusion-fa", "--max-vram", "8.00", "--stream-layers"]
    ]
    assert slot.launched_flags[3] == "8.00"
    assert slot.alive_signature()[2] == _COMMITTED
    slot.stop()


class _Server:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.alive = True

    def is_alive(self):
        return self.alive

    def vid_gen(
        self,
        payload,
        *,
        on_step = None,
        cancel_event = None,
    ):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _Slot:
    def __init__(self, server):
        self.server = server
        self.disabled_reason = None
        self.launched_flags = None
        self.launches = []
        self.stops = []

    def get(
        self,
        flags = None,
        env = None,
        *,
        cancel_event = None,
        pre_spawn = None,
        launch_transform = None,
    ):
        if pre_spawn is not None:
            flags = list(pre_spawn())
        self.launched_flags = tuple(launch_transform(tuple(flags)))
        self.launches.append(self.launched_flags)
        return self.server

    def begin_render(self):
        pass

    def end_render(self):
        pass

    def stop(self, reason = "stopped"):
        self.stops.append(reason)

    def disable(self, reason):
        self.disabled_reason = reason

    def release(self, reason):
        return True

    def alive_signature(self):
        return None


class _ServerRuntime:
    def __init__(self, slot, free):
        self.server_slot = slot
        self.offload_flags = _COMMITTED
        self.env = ()
        self.vram_free_probe = lambda: free


def _params():
    return SdCppVideoGenParams(
        prompt = "p", width = 864, height = 480, num_frames = 39, fps = 24, steps = 4, cfg_scale = 1.0, seed = 1
    )


def _server_render(slot, free, tmp_path):
    return VideoBackend._h3_native_server_render(
        None,
        _ServerRuntime(slot, free),
        _params(),
        output_path = tmp_path / "out.avi",
        on_log = lambda line: None,
        cancel = threading.Event(),
    )


def test_server_render_retries_once_smaller_on_device_oom(tmp_path, monkeypatch):
    monkeypatch.delenv(h3.H3_NATIVE_SERVER_ENV, raising = False)
    slot = _Slot(_Server([RuntimeError(_AUDIT_OOM), b"AVI"]))
    out = _server_render(slot, int(10 * GiB), tmp_path)
    assert out.read_bytes() == b"AVI"
    budgets = [launch[launch.index("--max-vram") + 1] for launch in slot.launches]
    assert budgets == ["8.00", "6.00"]
    assert len(slot.stops) == 1 and "smaller VRAM budget" in slot.stops[0]


def test_server_render_retries_only_once(tmp_path, monkeypatch):
    monkeypatch.delenv(h3.H3_NATIVE_SERVER_ENV, raising = False)
    slot = _Slot(_Server([RuntimeError(_AUDIT_OOM), RuntimeError(_AUDIT_OOM)]))
    with pytest.raises(RuntimeError, match = "out of memory"):
        _server_render(slot, int(10 * GiB), tmp_path)
    assert len(slot.launches) == 2


def test_server_render_does_not_fall_back_to_sd_cli_after_the_retry_dies(tmp_path, monkeypatch):
    monkeypatch.delenv(h3.H3_NATIVE_SERVER_ENV, raising = False)
    server = _Server([RuntimeError(_AUDIT_OOM), RuntimeError(_AUDIT_OOM)])
    vid_gen = server.vid_gen

    def dies_on_retry(*args, **kwargs):
        if len(server.outcomes) == 1:
            server.alive = False
        return vid_gen(*args, **kwargs)

    server.vid_gen = dies_on_retry
    slot = _Slot(server)
    with pytest.raises(RuntimeError, match = "out of memory"):
        _server_render(slot, int(10 * GiB), tmp_path)
    assert len(slot.launches) == 2
    assert slot.disabled_reason is None


def test_server_render_does_not_fall_back_when_the_retry_start_runs_out_of_memory(
    tmp_path, monkeypatch
):
    monkeypatch.delenv(h3.H3_NATIVE_SERVER_ENV, raising = False)
    slot = _Slot(_Server([RuntimeError(_AUDIT_OOM)]))
    get = slot.get

    def oom_on_second_start(*args, **kwargs):
        server = get(*args, **kwargs)
        if len(slot.launches) == 2:
            raise RuntimeError(_AUDIT_OOM)
        return server

    slot.get = oom_on_second_start
    with pytest.raises(RuntimeError, match = "out of memory"):
        _server_render(slot, int(10 * GiB), tmp_path)
    assert len(slot.launches) == 2
    assert slot.disabled_reason is None


def test_server_render_does_not_restart_over_a_pending_release(tmp_path, monkeypatch):
    monkeypatch.delenv(h3.H3_NATIVE_SERVER_ENV, raising = False)
    slot = _Slot(_Server([RuntimeError(_AUDIT_OOM)]))
    slot.release_pending = "another model is loading"
    with pytest.raises(RuntimeError, match = "out of memory"):
        _server_render(slot, int(10 * GiB), tmp_path)
    assert len(slot.launches) == 1


def test_a_server_start_that_runs_out_of_memory_retries_once_smaller_on_the_server(
    tmp_path, monkeypatch
):
    monkeypatch.delenv(h3.H3_NATIVE_SERVER_ENV, raising = False)
    slot = _Slot(_Server([b"AVI"]))
    get = slot.get

    def oom_on_first_start(*args, **kwargs):
        server = get(*args, **kwargs)
        slot.attempted_flags = slot.launched_flags
        if len(slot.launches) == 1:
            slot.launched_flags = None  # a failed start never records launched flags
            raise RuntimeError(_AUDIT_OOM)
        return server

    slot.get = oom_on_first_start
    out = _server_render(slot, int(10 * GiB), tmp_path)
    assert out.read_bytes() == b"AVI"
    budgets = [launch[launch.index("--max-vram") + 1] for launch in slot.launches]
    assert budgets == ["8.00", "6.00"]
    assert slot.disabled_reason is None  # the server path is kept, not given up for the session


def test_restart_for_retry_keeps_a_release_that_landed_before_the_stop(monkeypatch):
    stopped = []

    class _Srv:
        def __init__(self, binary):
            self.alive = False

        def start(
            self,
            files,
            *,
            offload = None,
            threads = None,
            env = None,
            extra_args = None,
        ):
            self.alive = True

        def is_alive(self):
            return self.alive

        def stop(self):
            stopped.append(1)
            self.alive = False

    monkeypatch.setattr(srv, "SdCppServer", _Srv)
    monkeypatch.setattr("core.inference.sd_cpp_engine.is_managed_binary", lambda b: False)
    slot = h3.H3NativeServerSlot("/x/sd-server", _FILES, _COMMITTED)
    slot.get(_COMMITTED)
    slot.begin_render()
    assert slot.release("another model needs the memory") is True  # busy: left pending
    assert slot.restart_for_retry("device out of memory") is False
    assert slot.release_pending == "another model needs the memory" and stopped == []
    assert (
        slot.end_render() == "another model needs the memory"
    )  # the release is honoured at the end
    assert stopped == [1]
    # With nothing pending the retry stops the server in the same locked step.
    slot.get(_COMMITTED)
    assert slot.restart_for_retry("device out of memory") is True and slot.release_pending is None
    # A release landing in the gap between the retry's stop and its respawn still stops the fresh server at the end.
    slot.begin_render()
    assert slot.release("another load") is True and slot.release_pending == "another load"
    slot.get(_COMMITTED)
    assert slot.end_render() == "another load" and stopped == [1, 1, 1]
    # A release left with no server to stop does not outlive the render (it would refuse the next retry).
    slot.begin_render()
    slot.release("late")
    assert slot.end_render() is None and slot.release_pending is None


_H3_FLAGS = ("--offload-to-cpu", "--diffusion-fa", "--max-vram", "-1", "--stream-layers")


def _h3_telemetry(monkeypatch, *, rocm, smi):
    """48 GiB card with 8 used: amd-smi / nvidia-smi rows when ``smi``, else only torch can read it."""
    from utils.hardware import hardware

    touched = []
    monkeypatch.setattr(hardware, "get_device", lambda: hardware.DeviceType.CUDA)
    monkeypatch.setattr(hardware, "IS_ROCM", rocm)
    monkeypatch.setattr(
        hardware, "_get_parent_visible_gpu_spec", lambda: {"raw": "0", "numeric_ids": [0]}
    )
    monkeypatch.setattr(hardware, "get_parent_visible_gpu_ids", lambda: [0])
    row = {"index": 0, "visible_ordinal": 0, "vram_total_gb": 48.0, "vram_used_gb": 8.0}
    monkeypatch.setattr(
        hardware,
        "_smi_visible_utilization",
        lambda spec: {"available": True, "devices": [dict(row)], "index_kind": "physical"}
        if smi
        else None,
    )
    monkeypatch.setattr(
        hardware,
        "_reconcile_rocm_unified_memory",
        lambda *_a: touched.append("reconcile"),
    )

    def per_device(idx):
        touched.append("torch")
        return [{"index": 0, "visible_ordinal": 0, "total_gb": 48.0, "used_gb": 8.0}]

    monkeypatch.setattr(hardware, "_torch_get_per_device_info", per_device)
    monkeypatch.setattr(hardware, "_overlay_system_wide_vram", lambda devices: None)
    # No KFD topology, so only amd-smi + torch can answer.
    monkeypatch.setattr(hardware, "_rocm_kfd_gpu_pci_ids", lambda: [])
    return touched


def test_h3_card_probes_read_nothing_on_rocm_without_a_context(monkeypatch):
    touched = _h3_telemetry(monkeypatch, rocm = True, smi = True)
    assert video_mod._h3_card_memory_bytes("cuda", 0, allow_context = False) == (None, None)
    assert video_mod._h3_visible_card_count("cuda") == 0
    assert touched == []


_ROCM_MASKS = (
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)


def _rocm_linux_host(
    monkeypatch,
    cards,
    boards = None,
):
    """``cards`` = [(pci, gfx target, (used_gb, total_gb) or None)] in KFD order; torch and amd-smi are tripwires."""
    from utils.hardware import amd, hardware

    for name in _ROCM_MASKS:
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(hardware, "get_device", lambda: hardware.DeviceType.CUDA)
    monkeypatch.setattr(hardware, "IS_ROCM", True)
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hardware, "_rocm_kfd_gpu_pci_ids", lambda: [c[0] for c in cards])
    monkeypatch.setattr(amd, "amd_kfd_gpu_gfx_targets", lambda *a, **k: [c[1] for c in cards])
    monkeypatch.setattr(
        hardware,
        "_rocm_linux_sysfs_vram_by_pci_gb",
        lambda: {c[0]: c[2] for c in cards if c[2] is not None},
    )
    monkeypatch.setattr(
        hardware,
        "_rocm_linux_amdgpu_cards",
        lambda: [
            (bdf, i, f"/sys/class/drm/card{i}/device")
            for i, bdf in enumerate(boards or dict.fromkeys(c[0] for c in cards))
        ],
    )

    def _no_torch(*a, **k):
        pytest.fail("the no-context probe touched torch")

    import torch

    for name in ("mem_get_info", "get_device_properties", "init", "_lazy_init", "device_count"):
        monkeypatch.setattr(torch.cuda, name, _no_torch)
    for name in (
        "_torch_get_device_module",
        "_torch_get_per_device_info",
        "_torch_get_device_inventory",
        "_torch_get_physical_gpu_count",
        "_smi_visible_utilization",
    ):
        monkeypatch.setattr(hardware, name, _no_torch)


def test_h3_card_probes_read_rocm_sysfs_without_a_context(monkeypatch):
    _rocm_linux_host(monkeypatch, [("0000:03:00.0", "gfx1100", (4.0, 24.0))])
    assert video_mod._h3_card_memory_bytes("cuda", 0, allow_context = False) == (20 * GiB, 24 * GiB)
    assert video_mod._h3_visible_card_count("cuda") == 1
    assert (
        video_mod._h3_card_free_bytes("cuda", 0, single_card_only = True, allow_context = False)
        == 20 * GiB
    )


def test_h3_card_probes_on_masked_rocm_read_nothing(monkeypatch):
    _rocm_linux_host(monkeypatch, [("0000:03:00.0", "gfx1100", (4.0, 24.0))])
    assert video_mod._h3_card_memory_bytes("cuda", 0, allow_context = False) == (20 * GiB, 24 * GiB)
    for name in _ROCM_MASKS:
        monkeypatch.setenv(name, "0")
        assert video_mod._h3_card_memory_bytes("cuda", 0, allow_context = False) == (None, None), name
        assert video_mod._h3_visible_card_count("cuda") == 0, name
        monkeypatch.delenv(name)


def test_h3_card_probes_never_report_an_apu_carve_out(monkeypatch):
    _rocm_linux_host(
        monkeypatch,
        [
            ("0000:03:00.0", "gfx1100", (4.0, 24.0)),
            ("0000:c5:00.0", "gfx1151", (1.0, 0.5)),
            ("0000:0b:00.0", "gfx942", (1.0, 192.0)),
        ],
    )
    assert video_mod._h3_card_memory_bytes("cuda", 0, allow_context = False) == (20 * GiB, 24 * GiB)
    assert video_mod._h3_card_memory_bytes("cuda", 1, allow_context = False) == (None, None)
    assert video_mod._h3_card_memory_bytes("cuda", 2, allow_context = False) == (None, None)
    assert video_mod._h3_visible_card_count("cuda") == 3
    # A partitioned card (two ordinals on one PCI address) reads nothing.
    _rocm_linux_host(
        monkeypatch,
        [("0000:03:00.0", "gfx950", (4.0, 24.0)), ("0000:03:00.0", "gfx950", (4.0, 24.0))],
    )
    assert video_mod._h3_card_memory_bytes("cuda", 0, allow_context = False) == (None, None)
    assert video_mod._h3_visible_card_count("cuda") == 0


def test_h3_card_probes_never_report_a_partition_the_whole_board(monkeypatch):
    # MI355X DPX: partition 1 decodes to 03:00.1 (no amdgpu device); partition 0 must not read the whole board.
    _rocm_linux_host(
        monkeypatch,
        [("0000:03:00.0", "gfx950", (8.0, 288.0)), ("0000:03:00.1", "gfx950", None)],
        boards = ["0000:03:00.0"],
    )
    assert video_mod._h3_card_memory_bytes("cuda", 0, allow_context = False) == (None, None)
    assert video_mod._h3_visible_card_count("cuda") == 0
    # The same board unpartitioned (SPX) still reads.
    _rocm_linux_host(monkeypatch, [("0000:03:00.0", "gfx950", (8.0, 288.0))])
    assert video_mod._h3_card_memory_bytes("cuda", 0, allow_context = False) == (280 * GiB, 288 * GiB)


def test_h3_card_probes_on_cuda_still_read_nvidia_smi(monkeypatch):
    from utils.hardware import hardware

    monkeypatch.setattr(hardware, "get_device", lambda: hardware.DeviceType.CUDA)
    monkeypatch.setattr(hardware, "IS_ROCM", False)
    monkeypatch.setattr(
        hardware,
        "_get_parent_visible_gpu_spec",
        lambda: {"raw": None, "numeric_ids": None, "supports_explicit_gpu_ids": True},
    )
    monkeypatch.setattr(hardware, "_reconcile_cuda_integrated_memory", lambda *a, **k: None)
    monkeypatch.setattr(
        hardware,
        "_smi_visible_utilization",
        lambda spec: {
            "available": True,
            "devices": [
                {"index": 0, "visible_ordinal": 0, "vram_used_gb": 2.0, "vram_total_gb": 16.0}
            ],
            "index_kind": "relative",
            "parent_visible_gpu_ids": [],
        },
    )
    monkeypatch.setattr(
        hardware,
        "_rocm_context_free_visible_devices",
        lambda: pytest.fail("ROCm sysfs read on a CUDA host"),
        raising = False,
    )
    assert video_mod._h3_card_memory_bytes("cuda", 0, allow_context = False) == (14 * GiB, 16 * GiB)
    assert video_mod._h3_visible_card_count("cuda") == 1


def test_h3_residency_reads_fall_back_to_torch_on_rocm(monkeypatch):
    # The residency and pressure reads may create a context, as they did before the launch budget.
    _h3_telemetry(monkeypatch, rocm = True, smi = True)
    assert video_mod._h3_card_memory_bytes("cuda", 0) == (40 * GiB, 48 * GiB)


@pytest.mark.parametrize(
    "rocm, smi", [(True, True), (False, False)], ids = ["rocm_amd_smi", "nvidia_no_smi"]
)
def test_h3_auto_runs_resident_when_the_card_fits(monkeypatch, rocm, smi):
    """48 GiB total, 40 free, 39 needed: Auto drops the streaming flags, as it did before the launch budget."""
    _h3_telemetry(monkeypatch, rocm = rocm, smi = smi)
    flags, resident = h3.h3_native_render_flags(
        _H3_FLAGS,
        memory_mode = "auto",
        free_bytes = video_mod._h3_card_free_bytes("cuda", 0),
        need_bytes = 39 * GiB,
        env = {},
    )
    assert resident is True
    assert "--offload-to-cpu" not in flags and "--stream-layers" not in flags


def test_a_slot_start_records_what_it_attempted_even_when_it_fails(monkeypatch):
    class _Srv:
        def __init__(self, binary):
            pass

        def start(
            self,
            files,
            *,
            offload = None,
            threads = None,
            env = None,
            extra_args = None,
        ):
            raise RuntimeError(_AUDIT_OOM)

        def is_alive(self):
            return False

        def stop(self):
            pass

    monkeypatch.setattr(srv, "SdCppServer", _Srv)
    monkeypatch.setattr("core.inference.sd_cpp_engine.is_managed_binary", lambda b: False)
    slot = h3.H3NativeServerSlot("/x/sd-server", _FILES, _COMMITTED)
    with pytest.raises(RuntimeError):
        slot.get(_COMMITTED, launch_transform = lambda f: [*f[:3], "8.00", *f[4:]])
    assert slot.launched_flags is None and slot.attempted_flags[3] == "8.00"


def test_server_render_does_not_retry_other_failures_or_auto_budget(tmp_path, monkeypatch):
    monkeypatch.delenv(h3.H3_NATIVE_SERVER_ENV, raising = False)
    slot = _Slot(_Server([RuntimeError("sd-server generation failed: bad prompt")]))
    with pytest.raises(RuntimeError, match = "bad prompt"):
        _server_render(slot, int(10 * GiB), tmp_path)
    assert len(slot.launches) == 1
    # Unreadable card: "-1" as before, and an OOM under it is not retried here (nothing smaller to try).
    slot = _Slot(_Server([RuntimeError(_AUDIT_OOM)]))
    with pytest.raises(RuntimeError, match = "out of memory"):
        _server_render(slot, None, tmp_path)
    assert [launch[3] for launch in slot.launches] == ["-1"]


def test_one_shot_retries_once_smaller_on_device_oom():
    rt = _Rt(int(12 * GiB))
    seen = []

    def run(flags):
        seen.append(flags[flags.index("--max-vram") + 1])
        if len(seen) == 1:
            raise RuntimeError("sd-cli exited 1. Last output:\n" + _AUDIT_OOM)
        return "/out.avi"

    assert (
        video_mod._h3_one_shot_with_vram_retry(rt, run, _COMMITTED, threading.Event()) == "/out.avi"
    )
    assert seen == ["10.00", "7.50"]

    def bad(flags):
        raise RuntimeError("sd-cli exited 1. Last output:\n[E] unknown argument")

    with pytest.raises(RuntimeError, match = "unknown argument"):
        video_mod._h3_one_shot_with_vram_retry(rt, bad, _COMMITTED, threading.Event())


def _load(monkeypatch, tmp_path, **kw):
    from test_video_backend import _SAGE_HELP, _load_h3_native_offload
    return _load_h3_native_offload(monkeypatch, tmp_path, help_text = _SAGE_HELP, **kw)


def test_load_commits_the_placeholder_and_a_probe(monkeypatch, tmp_path):
    from test_video_backend import _GRAPH_CUT_HELP, _load_h3_native_offload

    reads = []
    monkeypatch.setattr(
        video_mod,
        "_h3_card_free_bytes",
        lambda device, ordinal, **kw: reads.append((device, ordinal)) or int(9 * GiB),
    )
    state, offload = _load_h3_native_offload(monkeypatch, tmp_path, help_text = _GRAPH_CUT_HELP)
    assert offload[offload.index("--max-vram") + 1] == "-1"
    probe = state.pipe.vram_free_probe
    assert probe is not None
    assert (
        video_mod._h3_budgeted_launch_flags(state.pipe, offload)[offload.index("--max-vram") + 1]
        == "7.00"
    )
    assert reads and reads[-1][0] == "cuda"


@pytest.mark.parametrize(
    "torch_cc, reason",
    [
        ((10, 0), "sd.cpp BF16 cuBLAS matmuls"),
        ((8, 9), "sd.cpp BF16 cuBLAS matmuls"),
        ((7, 5), "sd.cpp exact kernels"),
    ],
)
def test_unreadable_listing_falls_back_to_torch_cc(monkeypatch, tmp_path, torch_cc, reason):
    """--list-devices unreadable (timed out / non-zero exit): nothing is cached, the verdict keeps the GPU, and the
    capability comes from torch instead of silently routing a sm80+ card to the exact kernels."""
    monkeypatch.delenv("UNSLOTH_H3_SAGE_ATTN", raising = False)
    monkeypatch.delenv("GGML_CUDA_QUANT_CUBLAS_MIN_BATCH", raising = False)
    monkeypatch.setattr(sd_cpp_backend, "_LAST_DEVICE_LISTING", {})
    monkeypatch.setattr(sd_cpp_backend, "_remember_device_listing", lambda *_a: None)
    asked = []
    monkeypatch.setattr(
        sd_cpp_backend,
        "torch_cuda_compute_capability",
        lambda ordinal: asked.append(ordinal) or torch_cc,
    )
    state, _offload = _load(monkeypatch, tmp_path, devices = "", installed_accelerator = "cuda")
    assert asked, "torch was not consulted"
    assert state.resolved["speed_mode"]["reason"] == reason
    assert (dict(state.pipe.env).get("GGML_CUDA_QUANT_CUBLAS_MIN_BATCH") is not None) == (
        torch_cc >= (8, 0)
    )


def test_base_behaviour_without_fallback_is_exact_kernels(monkeypatch, tmp_path):
    """The defect itself: the same unreadable listing with torch giving no answer routes to the exact kernels."""
    monkeypatch.delenv("UNSLOTH_H3_SAGE_ATTN", raising = False)
    monkeypatch.delenv("GGML_CUDA_QUANT_CUBLAS_MIN_BATCH", raising = False)
    monkeypatch.setattr(sd_cpp_backend, "_LAST_DEVICE_LISTING", {})
    monkeypatch.setattr(sd_cpp_backend, "_remember_device_listing", lambda *_a: None)
    monkeypatch.setattr(sd_cpp_backend, "torch_cuda_compute_capability", lambda ordinal: None)
    state, _offload = _load(monkeypatch, tmp_path, devices = "", installed_accelerator = "cuda")
    assert state.resolved["speed_mode"]["reason"] == "sd.cpp exact kernels"


def test_torch_is_not_consulted_when_the_listing_was_read(monkeypatch, tmp_path):
    from test_video_backend import _cuda_devices

    monkeypatch.delenv("GGML_CUDA_QUANT_CUBLAS_MIN_BATCH", raising = False)
    monkeypatch.setattr(sd_cpp_backend, "_LAST_DEVICE_LISTING", {})
    monkeypatch.setattr(
        sd_cpp_backend,
        "torch_cuda_compute_capability",
        lambda ordinal: pytest.fail("torch consulted"),
    )
    state, _ = _load(
        monkeypatch, tmp_path, devices = _cuda_devices("7.5"), installed_accelerator = "cuda"
    )
    assert state.resolved["speed_mode"]["reason"] == "sd.cpp exact kernels"


def test_cc_fallback_rules(monkeypatch):
    monkeypatch.setattr(sd_cpp_backend, "_LAST_DEVICE_LISTING", {})
    monkeypatch.setattr(
        sd_cpp_backend,
        "torch_cuda_compute_capability",
        lambda ordinal: (9, 0) if ordinal in (0, 1) else None,
    )
    kw = dict(binary = "/x/sd-cli", native_device = "cuda", listed_accelerator = True)
    fb = video_mod._h3_cuda_cc_fallback
    assert fb((8, 6), accelerator_names = ("cuda",), device_name = "CUDA0", **kw) == (8, 6)
    assert fb(None, accelerator_names = ("cuda",), device_name = "CUDA1", **kw) == (9, 0)
    assert fb(None, accelerator_names = ("cuda",), device_name = None, **kw) == (9, 0)
    # Not a CUDA build, not a CUDA device name, a CPU target: unknown stays unknown.
    assert fb(None, accelerator_names = ("vulkan",), device_name = None, **kw) is None
    assert fb(None, accelerator_names = ("rocm",), device_name = "ROCm0", **kw) is None
    assert fb(None, accelerator_names = ("cuda",), device_name = "Vulkan0", **kw) is None
    assert (
        fb(
            None,
            binary = "/x/sd-cli",
            native_device = "cpu",
            listed_accelerator = True,
            accelerator_names = ("cuda",),
            device_name = None,
        )
        is None
    )


def test_cc_fallback_reads_the_card_sd_cpp_runs_on_and_only_an_owned_cuda_build(monkeypatch):
    monkeypatch.setattr(sd_cpp_backend, "_LAST_DEVICE_LISTING", {})
    asked = []
    monkeypatch.setattr(
        sd_cpp_backend,
        "torch_cuda_compute_capability",
        lambda ordinal: asked.append(ordinal) or {0: (7, 5), 1: (8, 9)}.get(ordinal),
    )
    kw = dict(binary = "/x/sd-cli", native_device = "cuda", listed_accelerator = True)
    fb = video_mod._h3_cuda_cc_fallback
    # No device name means no --backend pin: sd.cpp renders on CUDA0, so that card is read (not the lowest across cards).
    assert fb(None, accelerator_names = ("cuda",), device_name = None, **kw) == (7, 5)
    assert asked == [0]
    assert fb(None, accelerator_names = ("cuda",), device_name = "CUDA1", **kw) == (8, 9)
    assert asked == [0, 1]
    # No install record (an SD_CLI_PATH build): its accelerator is unknown, so no CUDA route from torch.
    assert fb(None, accelerator_names = (None,), device_name = None, **kw) is None
    # The requested ordinal is not an input: it names a card sd.cpp only uses through a resolved pin.
    import inspect

    assert "native_ordinal" not in inspect.signature(fb).parameters


def test_a_stalled_torch_capability_reading_is_bounded(monkeypatch):
    import time

    monkeypatch.setattr(
        sd_cpp_backend, "_torch_cuda_compute_capability", lambda ordinal: time.sleep(5) or (9, 0)
    )
    t0 = time.monotonic()
    assert sd_cpp_backend.torch_cuda_compute_capability(0, timeout_s = 0.2) is None
    assert time.monotonic() - t0 < 2.0
    monkeypatch.setattr(sd_cpp_backend, "_torch_cuda_compute_capability", lambda ordinal: (9, 0))
    assert sd_cpp_backend.torch_cuda_compute_capability(0) == (9, 0)


def test_an_unpinned_render_on_several_cards_takes_no_scalar_budget(monkeypatch):
    monkeypatch.setattr(
        video_mod,
        "_h3_card_memory_bytes",
        lambda device, ordinal, **_k: (int(9 * GiB), int(24 * GiB)),
    )
    monkeypatch.setattr(video_mod, "_h3_visible_card_count", lambda device: 2)
    assert video_mod._h3_card_free_bytes("cuda", None, single_card_only = True) is None
    monkeypatch.setattr(video_mod, "_h3_visible_card_count", lambda device: 1)
    assert video_mod._h3_card_free_bytes("cuda", None, single_card_only = True) == int(9 * GiB)
    monkeypatch.setattr(video_mod, "_h3_visible_card_count", lambda device: 2)
    assert video_mod._h3_card_free_bytes("cuda", 1) == int(9 * GiB)


def test_torch_cc_helper_is_none_off_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(torch.version, "cuda", None, raising = False)
    assert sd_cpp_backend.torch_cuda_compute_capability(0) is None
    monkeypatch.setattr(torch.version, "cuda", "13.0", raising = False)
    monkeypatch.setattr(torch.version, "hip", "7.0", raising = False)
    assert sd_cpp_backend.torch_cuda_compute_capability(0) is None


def test_h3_card_probes_never_fall_back_to_a_torch_context(monkeypatch):
    from utils.hardware import hardware

    monkeypatch.setattr(hardware, "get_device", lambda: hardware.DeviceType.CUDA)
    monkeypatch.setattr(hardware, "_smi_visible_utilization", lambda spec: None)
    monkeypatch.setattr(hardware, "IS_ROCM", False)
    monkeypatch.setattr(hardware, "get_parent_visible_gpu_ids", lambda: [0])
    touched = []
    monkeypatch.setattr(
        hardware, "_torch_get_per_device_info", lambda idx: touched.append(idx) or []
    )
    assert video_mod._h3_visible_card_count("cuda") == 0
    assert video_mod._h3_card_memory_bytes("cuda", 0, allow_context = False) == (None, None)
    assert video_mod._h3_card_free_bytes("cuda", 0, allow_context = False) is None
    assert touched == []
