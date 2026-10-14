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


# -- pure helpers -------------------------------------------------------------------------------------------------


def test_budget_is_free_minus_reserve_rounded_down():
    assert args_mod.graph_cut_budget_gib(None) is None
    assert args_mod.graph_cut_budget_gib(int(8.0 * GiB)) == 6.0
    # Rounded down to 1/100 GiB: never above the plan.
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


# -- launch flags -------------------------------------------------------------------------------------------------


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
    # Per device: each card's own reading.
    rt = _Rt({"cuda0": int(10 * GiB), "cuda1": int(16 * GiB)})
    assert video_mod._h3_budgeted_launch_flags(rt, _COMMITTED)[3] == "cuda0=8.00,cuda1=14.00"


@pytest.mark.parametrize("free", [None, int(2 * GiB), RuntimeError("nvidia-smi timed out")])
def test_unknown_or_tiny_reading_keeps_the_placeholder(free):
    rt = _Rt(free)
    assert video_mod._h3_budgeted_launch_flags(rt, _COMMITTED) == list(_COMMITTED)
    # ... but a retry still runs below the budget that failed.
    assert video_mod._h3_budgeted_launch_flags(rt, _COMMITTED, cap = 4.5)[3] == "4.50"


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


# -- the real server slot: budget in argv, not in the signature ---------------------------------------------------


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


# -- render paths: one retry at a smaller budget on a device OOM --------------------------------------------------


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


def test_server_render_does_not_fall_back_when_the_retry_start_runs_out_of_memory(tmp_path, monkeypatch):
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


# -- load: the committed flags keep the placeholder and the runtime carries the probe ----------------------------


def _load(monkeypatch, tmp_path, **kw):
    from test_video_backend import _SAGE_HELP, _load_h3_native_offload
    return _load_h3_native_offload(monkeypatch, tmp_path, help_text = _SAGE_HELP, **kw)


def test_load_commits_the_placeholder_and_a_probe(monkeypatch, tmp_path):
    from test_video_backend import _GRAPH_CUT_HELP, _load_h3_native_offload

    reads = []
    monkeypatch.setattr(
        video_mod,
        "_h3_card_free_bytes",
        lambda device, ordinal: reads.append((device, ordinal)) or int(9 * GiB),
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


# -- CUDA compute capability when --list-devices could not be read ------------------------------------------------


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
        lambda ordinal: (9, 0) if ordinal in (None, 1) else None,
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


def test_torch_cc_helper_is_none_off_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(torch.version, "cuda", None, raising = False)
    assert sd_cpp_backend.torch_cuda_compute_capability(0) is None
    monkeypatch.setattr(torch.version, "cuda", "13.0", raising = False)
    monkeypatch.setattr(torch.version, "hip", "7.0", raising = False)
    assert sd_cpp_backend.torch_cuda_compute_capability(0) is None
