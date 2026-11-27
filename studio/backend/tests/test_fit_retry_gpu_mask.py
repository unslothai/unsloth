# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The --fit on retry after a failed full-offload estimate may use every card Auto could pick."""

from __future__ import annotations

import os
import struct
import subprocess

import pytest

from core.inference import llama_cpp
from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend

_GIB = 1024**3
_OOM = "allocating 16152.03 MiB on device 0: cudaMalloc failed: out of memory"


def _fit_mode(cmd):
    return cmd[cmd.index("--fit") + 1] if "--fit" in cmd else None


def _load(
    monkeypatch,
    tmp_path,
    *,
    returncodes,
    pick = ([0], False),
    gpu_ids = None,
    extra_args = None,
    sm_gate_error = None,
    parent_env = None,
    outputs = None,
    rows = None,
    parent_mask = None,
):
    """Two CUDA cards; Auto pins ``pick`` with --fit off. Returns the (argv, env) of each spawn."""

    def _gguf_string(value: str) -> bytes:
        return struct.pack("<Q", len(value)) + value.encode()

    metadata = _gguf_string("general.architecture") + struct.pack("<I", 8) + _gguf_string("llama")
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 1) + metadata)

    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising = False)
    if parent_mask is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", parent_mask)
        monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setattr(LlamaCppBackend, "_GPU_IDS_ARE_PCI_INDICES", True)
    rows = rows or [(0, 46 * _GIB, 48 * _GIB), (1, 24 * _GIB, 24 * _GIB)]
    backend = LlamaCppBackend()
    backend._get_gpu_memory = lambda _binary = None, **_kw: list(rows)
    backend._get_gpu_free_memory = lambda _binary = None, **_kw: [(i, f) for i, f, _t in rows]
    backend._read_gguf_metadata = lambda _path: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _path: 1024
    backend._mmproj_vram_bytes = lambda _path: 0
    backend._resolve_launch_mmproj_path = lambda **_kwargs: None
    backend._apu_ram_shortfall_message = lambda *_args, **_kwargs: None
    backend._amd_apu_wants_unified_memory = lambda *_args, **_kwargs: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._select_gpus = lambda *_a, **_k: pick
    backend._host_torch_is_rocm = lambda: False
    backend._cuda_sm_gate_error = lambda _binary, ids = None: (
        sm_gate_error if ids is not None and 1 in list(ids) else None
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _binary = None: False)
    )
    monkeypatch.setattr(LlamaCppBackend, "_is_datacenter_gpu", staticmethod(lambda *_a: False))
    # Simulated cards: never let the probe open a context on the host's real ones.
    monkeypatch.setattr(LlamaCppBackend, "_integrated_cuda_gpu_ids", staticmethod(lambda: set()))
    monkeypatch.setattr(
        LlamaCppBackend, "_unmasked_child_gpu_physical_ids", staticmethod(lambda: None)
    )
    backend.probe_server_capabilities = lambda _binary: {"found": True}
    backend._record_server_pid = lambda _pid: None
    backend._clear_server_pid = lambda: None
    monkeypatch.setattr(
        llama_cpp, "_swa_cache_path", lambda: tmp_path / "studio" / "swa_cache.json"
    )
    backend._llama_server_env_for_binary = lambda _binary, **_k: {
        "PATH": os.environ.get("PATH", ""),
        **(parent_env or {}),
    }
    backend._prepare_cpu_fallback_launch = lambda *_a, **_kw: None

    launches = []

    class _Process:
        pid = 123
        stdout = ()
        terminate = kill = lambda self: None

        def __init__(self, returncode):
            self.returncode = returncode

        def poll(self):
            # A stalled child is still running when its health check gives up.
            return None if self.returncode == "stall" else self.returncode

        def wait(self, timeout = None):
            return self.returncode

    _real_popen = subprocess.Popen

    def _popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _real_popen(cmd, **kwargs)
        launches.append((list(cmd), dict(kwargs["env"])))
        idx = len(launches) - 1
        return _Process(returncodes[idx] if idx < len(returncodes) else 1)

    def _wait_for_health(timeout = 600.0, **_kw):
        idx = len(launches) - 1
        code = returncodes[idx] if 0 <= idx < len(returncodes) else 1
        if code is None or code == "stall":
            backend._stdout_lines = []
        else:
            backend._stdout_lines = [outputs[idx] if outputs and idx < len(outputs) else _OOM]
        return code is None

    backend._wait_for_health = _wait_for_health
    monkeypatch.setattr(subprocess, "Popen", _popen)
    error = None
    try:
        backend.load_model(
            GgufLoadIntent(
                gguf_path = str(gguf),
                model_identifier = "owner/model",
                gpu_ids = gpu_ids,
                extra_args = extra_args,
            )
        )
    except Exception as exc:
        error = exc
    return backend, launches, error


def _mask(launch):
    return launch[1].get("CUDA_VISIBLE_DEVICES")


_NO_KERNELS = "CUDA error: no kernel image is available for execution on the device"
_TWO_PICKED = {"pick": ([1], False), "gpu_ids": [0, 1]}


@pytest.mark.parametrize(
    ("kwargs", "masks"),
    [
        ({}, ["0", "0,1"]),
        (_TWO_PICKED, ["1", "0,1"]),
        ({"extra_args": ["-ot", "exps=CPU"]}, ["0", "0,1"]),
        # A user --fit-target keeps the fitter on for the first launch too (#19269).
        ({"extra_args": ["--fit-target", "4096"]}, ["0", "0,1"]),
        # Iterating {8, 1} yields the parent's own order, where the order helper answers None.
        (
            {
                "pick": ([8], False),
                "rows": [(1, 24 * _GIB, 24 * _GIB), (8, 46 * _GIB, 48 * _GIB)],
                "parent_mask": "8,1",
            },
            ["8", "8,1"],
        ),
    ],
    ids = ["auto", "pick-bounds-it", "cpu-only-override", "fit-target-scalar", "parent-order"],
)
def test_the_fit_on_retry_sees_the_cards_auto_left_out(monkeypatch, tmp_path, kwargs, masks):
    backend, launches, error = _load(monkeypatch, tmp_path, returncodes = [1, None], **kwargs)
    assert error is None
    assert [_mask(launch) for launch in launches] == masks
    first = "on" if "--fit-target" in kwargs.get("extra_args", ()) else "off"
    assert [_fit_mode(launch[0]) for launch in launches] == [first, "on"]
    assert backend._child_gpu_physical_ids == tuple(int(i) for i in masks[1].split(","))


_KEEPS_PLANNED_MASK = {
    # These name cards by position in the child's mask; adding a card would move them.
    "device": {"extra_args": ["--device", "CUDA0"]},
    "main-gpu": {"extra_args": ["-mg", "0"]},
    "tensor-split": {"extra_args": ["--tensor-split", "1"]},
    "split-mode": {"extra_args": ["--split-mode", "layer"]},
    "mmproj-device": {"extra_args": ["--mmproj-device", "CUDA0"]},
    "override-tensor": {"extra_args": ["-ot", "blk\\.1\\.ffn_.*=CUDA0"]},
    "override-tensor-draft": {"extra_args": ["-otd", "blk\\..*=CUDA0"]},
    "env-main-gpu": {"parent_env": {"LLAMA_ARG_MAIN_GPU": "0"}},
    "env-override-tensor": {"parent_env": {"LLAMA_ARG_OVERRIDE_TENSOR": "exps=CUDA0"}},
    "user-fit-off": {"extra_args": ["--fit", "off"]},
    "fit-target-list": {"extra_args": ["--fit-target", "12000,1024"]},
    "env-fit-target-list": {"parent_env": {"LLAMA_ARG_FIT_TARGET": "12000/1024"}},
    "sm-gate": {"sm_gate_error": "no kernels for GPU 1"},
    "single-card-pick": {"gpu_ids": [0]},
    "not-out-of-memory": {"outputs": ["llama_model_load: error loading model: bad tensor"]},
}


@pytest.mark.parametrize("kwargs", _KEEPS_PLANNED_MASK.values(), ids = _KEEPS_PLANNED_MASK.keys())
def test_the_retry_keeps_the_planned_mask(monkeypatch, tmp_path, kwargs):
    _backend, launches, error = _load(monkeypatch, tmp_path, returncodes = [1, None], **kwargs)
    assert error is None
    assert [_mask(launch) for launch in launches] == ["0", "0"]


@pytest.mark.parametrize(
    ("returncodes", "outputs", "masks", "pin"),
    [
        ([1, None], None, ["1", "0,1"], [0, 1]),
        ([1, -11, None], [_OOM, "Segmentation fault"], ["1", "0,1", "1"], [1]),
    ],
    ids = ["widened-retry-comes-up", "widened-retry-fails"],
)
def test_a_pick_publishes_the_cards_its_child_runs_on(
    monkeypatch, tmp_path, returncodes, outputs, masks, pin
):
    backend, launches, error = _load(
        monkeypatch, tmp_path, returncodes = returncodes, outputs = outputs, **_TWO_PICKED
    )
    assert error is None
    assert [_mask(launch) for launch in launches] == masks
    assert backend.gpu_ids == pin
    assert backend.matches_gpu_ids([1]) == (pin == [1])
    assert set(backend._planned_vram_mib) == set(pin)


@pytest.mark.parametrize(
    ("returncode", "crash"),
    [
        (1, _NO_KERNELS),
        (-11, "Segmentation fault"),
        (1, _OOM),
        ("stall", ""),
    ],
    ids = ["added-card-has-no-kernels", "signal", "out-of-memory-again", "health-timeout"],
)
def test_a_failed_widened_retry_falls_back_to_the_planned_card(
    monkeypatch, tmp_path, returncode, crash
):
    backend, launches, error = _load(
        monkeypatch,
        tmp_path,
        returncodes = [1, returncode, None],
        outputs = [_OOM, crash],
    )
    assert error is None
    assert backend._healthy
    assert [_mask(launch) for launch in launches] == ["0", "0,1", "0"]
    assert [_fit_mode(launch[0]) for launch in launches] == ["off", "on", "on"]
    assert backend._child_gpu_physical_ids == (0,)


def test_a_failed_widened_retry_hands_later_rungs_the_planned_mask(monkeypatch, tmp_path):
    # Both retries hard-crash, so the flash-attention rung respawns on the planned mask.
    _backend, launches, error = _load(
        monkeypatch,
        tmp_path,
        returncodes = [1, -11, -11, None],
        outputs = [_OOM, "Segmentation fault", "Segmentation fault"],
    )
    assert error is None
    assert len(launches) == 4
    assert launches[3][0][launches[3][0].index("--flash-attn") + 1] == "off"
    assert [_mask(launch) for launch in launches] == ["0", "0,1", "0", "0"]


def test_the_arch_rung_fit_retry_stays_on_the_cards_it_kept(monkeypatch, tmp_path):
    # GPU 0 has no kernels and GPU 1 then runs out of memory: its retry must not add GPU 0.
    backend, launches, error = _load(
        monkeypatch,
        tmp_path,
        returncodes = [1, 1, None],
        outputs = [_NO_KERNELS, _OOM],
    )
    assert error is None
    assert backend._healthy
    assert [_mask(launch) for launch in launches] == ["0", "1", "1"]
    assert [_fit_mode(launch[0]) for launch in launches] == ["off", "off", "on"]


def test_the_widened_attempt_records_no_scheduler_abort(monkeypatch, tmp_path):
    # Only the planned card's own retry decides what Studio remembers for the next load.
    recorded = []
    monkeypatch.setattr(
        LlamaCppBackend,
        "_record_sched_reserve_abort",
        staticmethod(lambda *a, **k: recorded.append(a)),
    )
    _backend, launches, error = _load(
        monkeypatch,
        tmp_path,
        returncodes = [1, -6, 1],
        outputs = [_OOM, "GGML_ASSERT(*cur_backend_id != -1) failed", _OOM],
    )
    assert error is not None
    assert [_mask(launch) for launch in launches[:3]] == ["0", "0,1", "0"]
    assert recorded == []
