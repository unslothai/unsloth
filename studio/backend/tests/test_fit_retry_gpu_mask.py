# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The --fit on retry after a failed full-offload estimate may use every card Auto could pick."""

from __future__ import annotations

import os
import struct
import subprocess
import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


class _LoggerStub:
    def bind(self, *args, **kwargs):
        return self

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: _LoggerStub()
sys.modules.setdefault("loggers", _loggers_stub)
_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: _LoggerStub()
sys.modules.setdefault("structlog", _structlog_stub)
if not hasattr(sys.modules["structlog"], "get_logger"):
    sys.modules["structlog"].get_logger = _structlog_stub.get_logger

from core.inference import llama_cpp  # noqa: E402
from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend  # noqa: E402

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
        encoded = value.encode()
        return struct.pack("<Q", len(encoded)) + encoded

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

        def __init__(self, returncode):
            self.returncode = returncode

        def poll(self):
            return self.returncode

        def terminate(self):
            return None

        def wait(self, timeout = None):
            return self.returncode

        def kill(self):
            return None

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
        if code is None:
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


def test_the_fit_on_retry_sees_the_card_auto_left_out(monkeypatch, tmp_path):
    backend, launches, error = _load(monkeypatch, tmp_path, returncodes = [1, None])
    assert error is None
    assert len(launches) == 2
    assert (_fit_mode(launches[0][0]), _mask(launches[0])) == ("off", "0")
    assert (_fit_mode(launches[1][0]), _mask(launches[1])) == ("on", "0,1")
    # Footprint attribution reads the child's cards.
    assert backend._child_gpu_physical_ids == (0, 1)


def test_a_launch_that_comes_up_keeps_the_planned_card(monkeypatch, tmp_path):
    _backend, launches, error = _load(monkeypatch, tmp_path, returncodes = [None])
    assert error is None
    assert [_mask(launch) for launch in launches] == ["0"]


def test_a_pick_bounds_the_widened_set(monkeypatch, tmp_path):
    # The user picked both cards and Auto pinned one: the retry may use the pick.
    _backend, launches, error = _load(
        monkeypatch, tmp_path, returncodes = [1, None], pick = ([1], False), gpu_ids = [0, 1]
    )
    assert error is None
    assert [_mask(launch) for launch in launches] == ["1", "0,1"]


def test_a_cpu_override_rule_does_not_block_the_widen(monkeypatch, tmp_path):
    _backend, launches, error = _load(
        monkeypatch, tmp_path, returncodes = [1, None], extra_args = ["-ot", "exps=CPU"]
    )
    assert error is None
    assert _mask(launches[1]) == "0,1"


def test_a_single_card_pick_is_not_widened(monkeypatch, tmp_path):
    _backend, launches, error = _load(monkeypatch, tmp_path, returncodes = [1, None], gpu_ids = [0])
    assert error is None
    assert [_mask(launch) for launch in launches] == ["0", "0"]


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--device", "CUDA0"],
        ["-mg", "0"],
        ["--tensor-split", "1"],
        ["--split-mode", "layer"],
        ["--mmproj-device", "CUDA0"],
        ["-ot", "blk\\.1\\.ffn_.*=CUDA0"],
        ["-otd", "blk\\..*=CUDA0"],
    ],
)
def test_positional_placement_keeps_the_planned_mask(monkeypatch, tmp_path, extra_args):
    # These name cards by position in the child's mask; adding a card would move them.
    _backend, launches, error = _load(
        monkeypatch, tmp_path, returncodes = [1, None], extra_args = extra_args
    )
    assert error is None
    assert len(launches) == 2
    assert _mask(launches[1]) == "0"


@pytest.mark.parametrize(
    "parent_env",
    [{"LLAMA_ARG_MAIN_GPU": "0"}, {"LLAMA_ARG_OVERRIDE_TENSOR": "exps=CUDA0"}],
)
def test_an_inherited_device_twin_keeps_the_planned_mask(monkeypatch, tmp_path, parent_env):
    _backend, launches, error = _load(
        monkeypatch,
        tmp_path,
        returncodes = [1, None],
        parent_env = parent_env,
    )
    assert error is None
    assert _mask(launches[1]) == "0"


def test_a_card_the_build_cannot_run_is_not_added(monkeypatch, tmp_path):
    _backend, launches, error = _load(
        monkeypatch,
        tmp_path,
        returncodes = [1, None],
        sm_gate_error = "no kernels for GPU 1",
    )
    assert error is None
    assert [_mask(launch) for launch in launches] == ["0", "0"]


def test_a_failed_widened_retry_hands_later_rungs_the_planned_mask(monkeypatch, tmp_path):
    # The widened retry hard-crashes, so the flash-attention rung respawns; the shared env
    # it inherits must be the planned one again.
    _backend, launches, error = _load(
        monkeypatch,
        tmp_path,
        returncodes = [1, -11, None],
        outputs = [_OOM, "Segmentation fault"],
    )
    assert error is None
    assert len(launches) == 3
    assert "--flash-attn" in launches[2][0]
    assert launches[2][0][launches[2][0].index("--flash-attn") + 1] == "off"
    assert [_mask(launch) for launch in launches] == ["0", "0,1", "0"]


def test_a_trailing_user_fit_off_keeps_the_planned_mask(monkeypatch, tmp_path):
    # The retry flips Studio's --fit, but the user's later off still wins: no fitter runs.
    _backend, launches, _error = _load(
        monkeypatch, tmp_path, returncodes = [1, None], extra_args = ["--fit", "off"]
    )
    assert all(_mask(launch) == "0" for launch in launches)


def test_a_widened_pick_publishes_the_cards_it_runs_on(monkeypatch, tmp_path):
    backend, launches, error = _load(
        monkeypatch, tmp_path, returncodes = [1, None], pick = ([1], False), gpu_ids = [0, 1]
    )
    assert error is None
    assert _mask(launches[1]) == "0,1"
    assert backend.gpu_ids == [0, 1]
    # A later request for GPU 1 alone must not reuse the two-card server.
    assert not backend.matches_gpu_ids([1])
    assert backend.matches_gpu_ids([0, 1])
    # The plan other loads and eviction read spans both cards.
    assert set(backend._planned_vram_mib) == {0, 1}


def test_a_failed_widened_pick_publishes_the_planned_card_again(monkeypatch, tmp_path):
    backend, launches, _error = _load(
        monkeypatch,
        tmp_path,
        returncodes = [1, -11, None],
        outputs = [_OOM, "Segmentation fault"],
        pick = ([1], False),
        gpu_ids = [0, 1],
    )
    assert [_mask(launch) for launch in launches] == ["1", "0,1", "1"]
    assert backend.gpu_ids == [1]


def test_a_kernel_error_on_an_added_card_falls_back_to_the_planned_card(monkeypatch, tmp_path):
    # GPU 0 runs out of memory, then the widened child hits a kernel-image error on GPU 1.
    # The --fit on retry main runs on GPU 0 alone still gets its turn, and comes up.
    backend, launches, error = _load(
        monkeypatch,
        tmp_path,
        returncodes = [1, 1, None],
        outputs = [
            _OOM,
            "CUDA error: no kernel image is available for execution on the device",
        ],
    )
    assert error is None
    assert backend._healthy
    assert [_mask(launch) for launch in launches] == ["0", "0,1", "0"]
    assert [_fit_mode(launch[0]) for launch in launches] == ["off", "on", "on"]


def test_the_arch_rung_fit_retry_stays_on_the_cards_it_kept(monkeypatch, tmp_path):
    # GPU 0 has no kernels; the arch rung moves to GPU 1, which then runs out of memory
    # under --fit off. Its --fit on retry must not add GPU 0 back.
    backend, launches, error = _load(
        monkeypatch,
        tmp_path,
        returncodes = [1, 1, None],
        outputs = [
            "CUDA error: no kernel image is available for execution on the device",
            _OOM,
        ],
    )
    assert error is None
    assert backend._healthy
    assert [_mask(launch) for launch in launches] == ["0", "1", "1"]
    assert [_fit_mode(launch[0]) for launch in launches] == ["off", "off", "on"]


def test_the_widened_mask_keeps_the_parent_order(monkeypatch, tmp_path):
    # Iterating {8, 1} yields the parent's own order, where the order helper answers None.
    _backend, launches, error = _load(
        monkeypatch,
        tmp_path,
        returncodes = [1, None],
        pick = ([8], False),
        rows = [(1, 24 * _GIB, 24 * _GIB), (8, 46 * _GIB, 48 * _GIB)],
        parent_mask = "8,1",
    )
    assert error is None
    assert _mask(launches[1]) == "8,1"


def test_a_crash_that_is_not_out_of_memory_keeps_the_planned_mask(monkeypatch, tmp_path):
    # Only an allocation failure disproves Auto's estimate.
    _backend, launches, _error = _load(
        monkeypatch,
        tmp_path,
        returncodes = [1, None],
        outputs = ["llama_model_load: error loading model: invalid tensor shape"],
    )
    assert len(launches) == 2
    assert _mask(launches[1]) == "0"
