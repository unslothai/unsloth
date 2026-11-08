# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GPU metadata (integrated or discrete, pool size, name) comes from the CUDA driver API in a child process,
never torch.cuda.get_device_properties: that initialises CUDA in the calling process, after which torch's
device calls open a context of hundreds of MiB on each card they touch. CPU-only tests that dropped the GPU
mask and reached load_model left pytest workers holding contexts on six GPUs this way."""

from __future__ import annotations

import os
import subprocess
import sys
import types

import pytest

from core.inference.llama_cpp import LlamaCppBackend
from utils.hardware import nvidia

GIB = 1 << 30
# The real query, captured at import: the autouse conftest fixture replaces it for every test.
_REAL_GET_CUDA_DEVICE_INFO = getattr(nvidia, "get_cuda_device_info", None)


class _CountingTorch(types.ModuleType):
    """torch with CUDA visible, counting every call that would initialise it."""

    def __init__(self, count):
        super().__init__("torch")
        self.calls = []
        self.version = types.SimpleNamespace(hip = None)

        def _counted(name, value = None):
            def _call(*a, **k):
                self.calls.append(name)
                if name == "get_device_properties":
                    return types.SimpleNamespace(
                        name = "NVIDIA B200", total_memory = 180 * GIB, is_integrated = 0
                    )
                return value

            return _call

        self.cuda = types.SimpleNamespace(
            is_available = lambda: True,
            device_count = lambda: count,
            get_device_properties = _counted("get_device_properties"),
            _lazy_init = _counted("_lazy_init"),
            init = _counted("init"),
        )


@pytest.fixture
def host(monkeypatch):
    """Six discrete cards and one integrated SoC behind an unset mask, as the driver child reports them."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    for cache in ("_INTEGRATED_CUDA_IDS", "_INTEGRATED_CUDA_POOL_MIB", "_CUDA_DEVICE_INFO"):
        monkeypatch.setattr(LlamaCppBackend, cache, {}, raising = False)
    torch = _CountingTorch(7)
    monkeypatch.setitem(sys.modules, "torch", torch)
    queries = []
    devices = [(f"0000:{i + 1:02x}:00.0", False, 180 * GIB, "NVIDIA B200") for i in range(6)]
    devices.append(("0000:0f:00.0", True, 120 * GIB, "NVIDIA GB10"))

    def _info():
        queries.append(1)
        return list(devices)

    monkeypatch.setattr(nvidia, "get_cuda_device_info", _info, raising = False)
    return types.SimpleNamespace(torch = torch, queries = queries)


def test_metadata_lookups_never_initialise_torch_cuda(host):
    answers = (
        LlamaCppBackend._integrated_cuda_gpu_ids(),
        LlamaCppBackend._integrated_cuda_pool_total_mib(),
        LlamaCppBackend._integrated_cuda_unified_memory([0]),
        LlamaCppBackend._integrated_cuda_selection_is_all_shared([6]),
        LlamaCppBackend._all_selected_gpus_match(LlamaCppBackend._DATACENTER_GPU_RE, [0, 1]),
    )
    assert host.torch.calls == []
    assert answers == ({6}, {6: 120 * 1024}, False, True, True)
    # One child per visibility mask: every lookup above reads the same answer.
    assert len(host.queries) == 1


def test_an_ordinal_that_did_not_answer_is_used_but_not_settled(host, monkeypatch):
    monkeypatch.setattr(
        nvidia,
        "get_cuda_device_info",
        lambda: [("0000:01:00.0", True, 120 * GIB, "NVIDIA GB10"), None],
    )
    assert LlamaCppBackend._integrated_cuda_gpu_ids() == {0}
    assert LlamaCppBackend._integrated_cuda_probe_is_free() is False
    assert LlamaCppBackend._CUDA_DEVICE_INFO == {}


def test_an_unknown_answer_is_not_cached(host, monkeypatch):
    monkeypatch.setattr(nvidia, "get_cuda_device_info", lambda: None)
    assert LlamaCppBackend._integrated_cuda_gpu_ids() == set()
    assert LlamaCppBackend._integrated_cuda_probe_is_free() is False


def _child(
    monkeypatch,
    returncode,
    stdout = "",
):
    seen = {}

    def _run(cmd, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, returncode, stdout, "")

    monkeypatch.setattr(nvidia.subprocess, "run", _run)
    return seen


@pytest.mark.parametrize(
    "returncode, stdout, expected",
    [
        (
            0,
            "0000:3b:00.0 0 85899345920 NVIDIA H100 80GB HBM3\n0000:00:00.0 1 128000000000 NVIDIA GB10\n",
            [
                ("0000:3b:00.0", False, 85899345920, "NVIDIA H100 80GB HBM3"),
                ("0000:00:00.0", True, 128000000000, "NVIDIA GB10"),
            ],
        ),
        (
            0,
            "0000:3b:00.0 0 85899345920 NVIDIA H100\nERR\n",
            [("0000:3b:00.0", False, 85899345920, "NVIDIA H100"), None],
        ),
        # a driver that cannot report the PCI bus id still answers the rest
        (0, "- 1 128000000000 NVIDIA GB10\n", [("", True, 128000000000, "NVIDIA GB10")]),
        (0, "", []),  # cuInit: CUDA_ERROR_NO_DEVICE
        (2, "", []),  # no NVIDIA driver
        (3, "", None),  # cuInit failed otherwise: unknown
        (0, "garbage\n", None),
    ],
)
def test_the_driver_child_answer_is_parsed(monkeypatch, returncode, stdout, expected):
    monkeypatch.setattr(nvidia, "get_cuda_device_info", _REAL_GET_CUDA_DEVICE_INFO)
    seen = _child(monkeypatch, returncode, stdout)
    assert nvidia.get_cuda_device_info() == expected
    assert seen["cmd"][1:] == ["-c", nvidia._CUDA_DEVICE_INFO_SCRIPT]


def test_a_child_that_times_out_is_unknown(monkeypatch):
    monkeypatch.setattr(nvidia, "get_cuda_device_info", _REAL_GET_CUDA_DEVICE_INFO)

    def _run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))

    monkeypatch.setattr(nvidia.subprocess, "run", _run)
    assert nvidia.get_cuda_device_info() is None


def test_an_unknown_answer_is_not_asked_again_for_a_while(monkeypatch):
    """A slow or failing cuInit must not start a child on every lookup: unknown and partial answers are
    reused for the retry window, per mask; a complete one is left to the callers' caches."""
    monkeypatch.setattr(nvidia, "get_cuda_device_info", _REAL_GET_CUDA_DEVICE_INFO)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    clock = [1000.0]
    monkeypatch.setattr(nvidia.time, "monotonic", lambda: clock[0])
    answers = iter(
        [
            (3, ""),
            (0, "0000:01:00.0 0 8589934592 NVIDIA T4\nERR\n"),
            (0, "0000:01:00.0 0 8589934592 NVIDIA T4\n"),
        ]
    )
    runs = []

    def _run(cmd, **kw):
        runs.append(1)
        rc, out = next(answers)
        return subprocess.CompletedProcess(cmd, rc, out, "")

    monkeypatch.setattr(nvidia.subprocess, "run", _run)
    assert nvidia.get_cuda_device_info() is None
    assert nvidia.get_cuda_device_info() is None and len(runs) == 1
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")  # another mask is its own question
    assert nvidia.get_cuda_device_info() == [("0000:01:00.0", False, 8589934592, "NVIDIA T4"), None]
    assert len(runs) == 2
    clock[0] += nvidia._CUDA_DEVICE_INFO_RETRY_S + 1
    assert nvidia.get_cuda_device_info() == [("0000:01:00.0", False, 8589934592, "NVIDIA T4")]
    assert len(runs) == 3


def test_the_real_query_answers_no_device_under_an_empty_mask(monkeypatch):
    """End to end on this host, whatever it is (Linux, Windows, macOS, x86_64 or ARM64, a driver or none):
    with every GPU hidden the query is a definite empty answer and never raises."""
    monkeypatch.setattr(nvidia, "get_cuda_device_info", _REAL_GET_CUDA_DEVICE_INFO)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert nvidia.get_cuda_device_info() == []


def test_the_driver_child_runs_and_finds_no_device_under_an_empty_mask():
    """The real script under CUDA_VISIBLE_DEVICES="": cuInit sees no device (or there is no driver at all)."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES = "")
    r = subprocess.run(
        [sys.executable, "-c", nvidia._CUDA_DEVICE_INFO_SCRIPT],
        capture_output = True,
        text = True,
        timeout = 60,
        env = env,
    )
    assert r.returncode in (0, 2), r.stderr
    assert r.stdout.strip() == ""


def test_an_ordinal_that_did_not_answer_leaves_the_pool_unsettled(host, monkeypatch):
    """Device 1 did not answer: it may be integrated, so neither its id set nor the pool totals are cached."""
    monkeypatch.setattr(
        nvidia,
        "get_cuda_device_info",
        lambda: [("0000:01:00.0", True, 120 * GIB, "NVIDIA GB10"), None],
    )
    assert LlamaCppBackend._integrated_cuda_pool_total_mib() == {0: 120 * 1024}
    assert LlamaCppBackend._INTEGRATED_CUDA_POOL_MIB == {}


def test_the_inventory_poll_reads_nvidia_cards_from_the_driver(host, monkeypatch):
    from utils.hardware import hardware as hw

    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    inventory = hw._torch_get_device_inventory([4, 6])
    assert host.torch.calls == []
    assert [
        (d["index"], d["visible_ordinal"], d["name"], d["total_gb"], d["_cuda_integrated"])
        for d in inventory
    ] == [
        (4, 0, "NVIDIA B200", 180.0, False),
        (6, 1, "NVIDIA B200", 180.0, False),
    ]


def test_the_device_list_keeps_the_names_that_answered(host, monkeypatch, capsys):
    from utils.hardware import hardware as hw

    monkeypatch.setattr(hw, "_CUDA_DEVICE_NAMES", {})
    monkeypatch.setattr(
        nvidia,
        "get_cuda_device_info",
        lambda: [None, ("0000:02:00.0", False, 180 * GIB, "NVIDIA B200")],
    )
    host.torch.cuda.device_count = lambda: 2
    hw._print_cuda_device_list(False)
    out = capsys.readouterr().out
    assert "[0] <unavailable>" in out and "[1] NVIDIA B200" in out
    assert host.torch.calls == []
    with pytest.raises(RuntimeError, match = "device 0"):
        hw._cuda_device0_name()
