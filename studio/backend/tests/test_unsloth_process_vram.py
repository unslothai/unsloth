# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-process VRAM for the monitor's Unsloth slice. Every process list is pinned: none is read from the host."""

import subprocess
from types import SimpleNamespace

import pytest

from utils.hardware import amd as amd_mod
from utils.hardware import gpu_query, nvidia

OURS = frozenset({100, 101, 102})
UUID0 = "GPU-aaaa"
UUID1 = "GPU-bbbb"


def _done(stdout: str, returncode: int = 0):
    return subprocess.CompletedProcess([], returncode, stdout = stdout, stderr = "")


@pytest.fixture
def smi(monkeypatch):
    """A fake nvidia-smi answering by query, with the calls it saw."""
    answers = {
        "--query-compute-apps=gpu_uuid,pid,used_memory": _done(
            f"{UUID0}, 100, 9000\n{UUID0}, 555, 4000\n{UUID1}, 101, 6000\n{UUID1}, 102, 400\n"
        ),
        "--query-gpu=index,uuid": _done(f"0, {UUID0}\n1, {UUID1}\n2, GPU-cccc\n"),
    }
    calls = []

    def run(argv, **kwargs):
        query = argv[1]
        calls.append(query)
        if query.startswith("--query-gpu=index,utilization"):
            return _done(
                "0, 10, 50, 14000, 12288, 100, 170\n"
                "1, 20, 55, 7000, 12288, 100, 170\n"
                "2, 0, 40, 500, 12288, 20, 170\n"
            )
        return answers[query]

    monkeypatch.setattr(gpu_query, "run_nvidia_smi", run)
    monkeypatch.setattr(nvidia, "_nvidia_smi_executable", lambda: "nvidia-smi")
    monkeypatch.setattr(nvidia, "_process_query_failed_at", None)
    monkeypatch.setattr("utils.process_tree.own_process_tree_pids", lambda: OURS)
    return SimpleNamespace(answers = answers, calls = calls)


def test_nvidia_sums_only_our_process_tree_per_card(smi):
    assert nvidia.get_process_vram_mib_by_uuid(OURS) == {UUID0: 9000.0, UUID1: 6400.0}


def test_nvidia_wddm_na_size_gives_no_figure(smi):
    # Windows WDDM lists the process but not its memory: no figure, not a zero.
    smi.answers["--query-compute-apps=gpu_uuid,pid,used_memory"] = _done(f"{UUID0}, 100, [N/A]\n")
    assert nvidia.get_process_vram_mib_by_uuid(OURS) is None


def test_nvidia_no_listed_process_of_ours_gives_no_figure(smi):
    # A container's PID namespace hides our PIDs; 0 GiB would be a lie.
    smi.answers["--query-compute-apps=gpu_uuid,pid,used_memory"] = _done(f"{UUID0}, 555, 4000\n")
    assert nvidia.get_process_vram_mib_by_uuid(OURS) is None
    smi.answers["--query-compute-apps=gpu_uuid,pid,used_memory"] = _done("")
    assert nvidia.get_process_vram_mib_by_uuid(OURS) is None


def test_nvidia_failed_query_backs_off(smi):
    smi.answers["--query-compute-apps=gpu_uuid,pid,used_memory"] = _done("", returncode = 2)
    assert nvidia.get_process_vram_mib_by_uuid(OURS) is None
    assert nvidia.get_process_vram_mib_by_uuid(OURS) is None
    assert smi.calls.count("--query-compute-apps=gpu_uuid,pid,used_memory") == 1


def test_nvidia_utilization_carries_the_slice_only_when_asked(smi):
    plain = nvidia.get_visible_gpu_utilization([0, 1, 2])
    assert all("unsloth_used_gb" not in d for d in plain["devices"])
    assert not any(call.startswith("--query-compute-apps") for call in smi.calls)

    with gpu_query.process_vram_reads():
        result = nvidia.get_visible_gpu_utilization([0, 1, 2])
    by_index = {d["index"]: d["unsloth_used_gb"] for d in result["devices"]}
    assert by_index == {0: round(9000 / 1024, 2), 1: round(6400 / 1024, 2), 2: 0.0}


def test_nvidia_uuid_mask_reuses_the_uuids_it_already_read(smi, monkeypatch):
    def run(argv, **kwargs):
        query = argv[1]
        smi.calls.append(query)
        if query.startswith("--query-gpu=index,uuid,utilization"):
            return _done(
                f"0, {UUID0}, 10, 50, 14000, 12288, 100, 170\n"
                f"1, {UUID1}, 20, 55, 7000, 12288, 100, 170\n"
            )
        return smi.answers[query]

    monkeypatch.setattr(gpu_query, "run_nvidia_smi", run)
    with gpu_query.process_vram_reads():
        result = nvidia.get_visible_gpu_utilization(None, f"{UUID1}")
    assert [(d["visible_ordinal"], d["unsloth_used_gb"]) for d in result["devices"]] == [
        (0, round(6400 / 1024, 2))
    ]
    assert "--query-gpu=index,uuid" not in smi.calls


_AMD_WRAPPED = [
    {
        "gpu": 0,
        "process_list": [
            {
                "process_info": {
                    "name": "python3",
                    "pid": 100,
                    "memory_usage": {"vram_mem": {"value": 8 * 1024**3, "unit": "B"}},
                }
            },
            {
                "process_info": {
                    "name": "other",
                    "pid": 555,
                    "memory_usage": {"vram_mem": {"value": 4 * 1024**3, "unit": "B"}},
                }
            },
        ],
    },
    {"gpu": 1, "process_list": [{"process_info": "No running processes detected"}]},
]

_AMD_FLAT = [
    {
        "gpu": 0,
        "process_list": [
            {"name": "llama-server", "pid": 101, "memory_usage": {"vram_mem": "2048 MiB"}},
        ],
    },
    {"gpu": 1, "process_list": []},
]


@pytest.fixture
def amd_smi(monkeypatch):
    answer = {"process": _AMD_WRAPPED, "metric": [{"gpu": 0}, {"gpu": 1}]}
    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        return answer.get(args[0])

    monkeypatch.setattr(amd_mod, "_run_amd_smi", run)
    monkeypatch.setattr(amd_mod, "_display_cache", {})
    monkeypatch.setattr("utils.process_tree.own_process_tree_pids", lambda: OURS)
    return SimpleNamespace(answer = answer, calls = calls)


def test_amd_reads_both_process_list_shapes(amd_smi):
    assert amd_mod.get_process_vram_mib_by_gpu(OURS) == {0: 8192.0, 1: 0.0}
    amd_smi.answer["process"] = _AMD_FLAT
    assert amd_mod.get_process_vram_mib_by_gpu(OURS) == {0: 2048.0, 1: 0.0}


def test_amd_no_figure_without_our_process_or_its_size(amd_smi):
    amd_smi.answer["process"] = [_AMD_WRAPPED[1]]
    assert amd_mod.get_process_vram_mib_by_gpu(OURS) is None
    amd_smi.answer["process"] = [{"gpu": 0, "process_list": [{"pid": 100, "memory_usage": {}}]}]
    assert amd_mod.get_process_vram_mib_by_gpu(OURS) is None
    amd_smi.answer["process"] = None
    assert amd_mod.get_process_vram_mib_by_gpu(OURS) is None


def test_amd_utilization_carries_the_slice_only_when_asked(amd_smi, monkeypatch):
    monkeypatch.setattr(
        amd_mod, "_extract_gpu_metrics", lambda gpu: {"vram_used_gb": 10.0, "vram_total_gb": 16.0}
    )
    plain = amd_mod.get_visible_gpu_utilization([0, 1])
    assert all("unsloth_used_gb" not in d for d in plain["devices"])
    assert ("process",) not in amd_smi.calls

    with gpu_query.process_vram_reads():
        result = amd_mod.get_visible_gpu_utilization([0, 1])
    assert {d["index"]: d["unsloth_used_gb"] for d in result["devices"]} == {0: 8.0, 1: 0.0}


def test_system_gpu_devices_carry_the_slice_from_the_monitor_read(monkeypatch):
    import main
    import utils.hardware as hardware
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(main, "_system_gpu_cache", None)
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda: False))
    monkeypatch.setattr(LlamaCppBackend, "_backend_lacks_gpu_lib", staticmethod(lambda: False))
    monkeypatch.setattr(hardware, "get_vulkan_inference_gpu_info", lambda: None)
    monkeypatch.setattr(hardware, "get_cross_vendor_inference_gpu_info", lambda: None)
    monkeypatch.setattr(hardware, "get_mixed_vulkan_inference_gpu_info", lambda info: None)
    monkeypatch.setattr(
        hardware,
        "get_backend_visible_gpu_info",
        lambda: {
            "available": True,
            "backend": "cuda",
            "devices": [{"index": 0, "name": "NVIDIA GeForce RTX 3060", "memory_total_gb": 12.0}],
        },
    )
    asked = []

    def utilization():
        asked.append(gpu_query.process_vram_wanted())
        return {
            "available": True,
            "backend": "cuda",
            "devices": [{"index": 0, "vram_used_gb": 11.0, "unsloth_used_gb": 9.5}],
        }

    monkeypatch.setattr(hardware, "get_visible_gpu_utilization", utilization)

    gpu, _ = main._get_cached_system_gpu_info(SimpleNamespace(debug = lambda *a: None))

    assert asked == [True]
    assert gpu["devices"][0]["unsloth_used_gb"] == 9.5


def test_own_process_tree_includes_the_backend():
    import os

    from utils.process_tree import own_process_tree_pids
    assert os.getpid() in own_process_tree_pids()
