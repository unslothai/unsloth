# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
import typer
from typer.testing import CliRunner

import unsloth_cli
import unsloth_cli.commands.doctor as doctor_mod


def _gpu(index, name = "Tesla T4", cap = "7.5", pci = None):
    return {"index": index, "name": name, "capability": cap, "memory_gib": 15.0, "pci": pci}


def _info(*gpus):
    return {"torch": "2.x", "cuda": "12.4", "hip": None, "nccl": "2.21.5", "gpus": list(gpus)}


def _smoke(ok, hung = False):
    return {"ok": ok, "hung": hung, "busbw_gbps": 5.0 if ok else None, "log": ""}


def test_doctor_is_registered_on_the_cli():
    result = CliRunner().invoke(unsloth_cli.app, ["doctor", "--help"])
    assert result.exit_code == 0, result.output
    assert "--gpus" in result.output


def test_verdict_no_gpu_and_single_gpu():
    healthy, hints = doctor_mod.verdict(_info(), None, None, None, None)
    assert not healthy and "No GPU" in hints[0]
    healthy, hints = doctor_mod.verdict(_info(_gpu(0)), None, None, None, None)
    assert healthy and "skipped" in hints[0]


def test_verdict_p2p_corruption_recommends_nccl_p2p_disable():
    p2p = {(0, 1): "corrupt", (1, 0): "ok"}
    healthy, hints = doctor_mod.verdict(_info(_gpu(0), _gpu(1)), None, p2p, _smoke(True), None)
    assert not healthy
    assert any("NCCL_P2P_DISABLE=1" in h and "(0, 1)" in h for h in hints)


def test_verdict_smoke_fixed_by_disabling_p2p():
    healthy, hints = doctor_mod.verdict(
        _info(_gpu(0), _gpu(1)), None, None, _smoke(False, hung = True), _smoke(True)
    )
    assert not healthy
    assert any("hung" in h and "passed with NCCL_P2P_DISABLE=1" in h for h in hints)


def test_verdict_smoke_failing_regardless_points_at_debug_env():
    healthy, hints = doctor_mod.verdict(
        _info(_gpu(0), _gpu(1)), None, None, _smoke(False), _smoke(False)
    )
    assert not healthy
    assert any("UNSLOTH_DEBUG_DISTRIBUTED=1" in h for h in hints)


@pytest.mark.parametrize("level", ["PHB", "NODE", "SYS"])
def test_verdict_pcie_host_topology_prefers_ddp_for_lora(level):
    topo = {(0, 1): {"level": level, "nvlink": False, "p2p_read": True}}
    healthy, hints = doctor_mod.verdict(_info(_gpu(0), _gpu(1)), topo, {}, _smoke(True), None)
    assert healthy
    assert any("prefer DDP" in h for h in hints)


def test_verdict_nvlink_and_mixed_gpus():
    topo = {(0, 1): {"level": "SYS", "nvlink": True, "p2p_read": True}}
    healthy, hints = doctor_mod.verdict(
        _info(_gpu(0), _gpu(1, name = "NVIDIA L4", cap = "8.9")), topo, {}, _smoke(True), None
    )
    assert healthy
    assert not any("prefer DDP" in h for h in hints)
    assert any("NVLinked" in h for h in hints)
    assert any("compute capability" in h for h in hints)


class _FakeNvml:
    NVML_P2P_CAPS_INDEX_READ = 0
    NVML_P2P_CAPS_INDEX_NVLINK = 2
    NVML_P2P_STATUS_OK = 0

    def __init__(self):
        self.shutdown = False

    def nvmlInit(self):
        pass

    def nvmlShutdown(self):
        self.shutdown = True

    def nvmlDeviceGetHandleByPciBusId(self, bus):
        return bus.decode()

    def nvmlDeviceGetTopologyCommonAncestor(self, a, b):
        return 50 if {a, b} == {"bus-a", "bus-c"} else 30

    def nvmlDeviceGetP2PStatus(self, a, b, idx):
        if idx == self.NVML_P2P_CAPS_INDEX_NVLINK:
            return 0 if {a, b} == {"bus-a", "bus-b"} else 1
        return 0


def test_nvml_topology_keys_by_torch_index_via_pci_bus():
    nvml = _FakeNvml()
    gpus = [_gpu(0, pci = "bus-c"), _gpu(1, pci = "bus-a"), _gpu(2, pci = "bus-b")]
    topo = doctor_mod.nvml_topology(nvml, gpus)
    assert topo[(0, 1)] == {"level": "SYS", "nvlink": False, "p2p_read": True}
    assert topo[(1, 2)] == {"level": "PHB", "nvlink": True, "p2p_read": True}
    assert nvml.shutdown


def test_nvml_topology_absent_without_pynvml_or_pci():
    assert doctor_mod.nvml_topology(None, [_gpu(0, pci = "a"), _gpu(1, pci = "b")]) is None
    assert doctor_mod.nvml_topology(_FakeNvml(), [_gpu(0), _gpu(1)]) is None


def test_allreduce_smoke_gloo_two_ranks():
    pytest.importorskip("torch.distributed")
    result = doctor_mod.allreduce_smoke(2, "gloo", timeout = 60, numel = 1024)
    assert result["ok"], result["log"]
    assert result["busbw_gbps"] is not None


def test_allreduce_smoke_reports_a_hang(monkeypatch):
    monkeypatch.setattr(doctor_mod, "_SMOKE_WORKER", "import time; time.sleep(60)")
    result = doctor_mod.allreduce_smoke(2, "gloo", timeout = -59, numel = 1024)
    assert result["hung"] and not result["ok"]


def _fake_torch(count):
    props = SimpleNamespace(
        name = "Tesla T4", major = 7, minor = 5, total_memory = 15 * 1024**3, pci_bus_id = None
    )
    cuda = SimpleNamespace(
        is_available = lambda: count > 0,
        device_count = lambda: count,
        get_device_properties = lambda i: props,
        can_device_access_peer = lambda a, b: False,
        nccl = SimpleNamespace(version = lambda: (2, 21, 5)),
    )
    return SimpleNamespace(
        __version__ = "2.x",
        version = SimpleNamespace(cuda = "12.4", hip = None),
        cuda = cuda,
        distributed = SimpleNamespace(is_nccl_available = lambda: True),
    )


def _run_doctor(monkeypatch, torch, smoke_results):
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(doctor_mod, "_load_pynvml", lambda: None)
    monkeypatch.setattr(doctor_mod, "driver_version", lambda pynvml = None: "550.1")
    monkeypatch.setattr(
        doctor_mod, "p2p_copy_probe", lambda t, n: {(0, 1): "ok", (1, 0): "ok"}
    )
    calls = []

    def fake_smoke(world, backend, timeout, extra_env = None):
        calls.append((world, backend, extra_env))
        return smoke_results[len(calls) - 1]

    monkeypatch.setattr(doctor_mod, "allreduce_smoke", fake_smoke)
    monkeypatch.delenv("NCCL_P2P_DISABLE", raising = False)
    cli = typer.Typer()
    cli.command()(doctor_mod.doctor)
    return CliRunner().invoke(cli, ["--gpus"]), calls


def test_doctor_two_gpus_healthy(monkeypatch):
    result, calls = _run_doctor(monkeypatch, _fake_torch(2), [_smoke(True)])
    assert result.exit_code == 0, result.output
    assert calls == [(2, "nccl", None)]
    assert "driver 550.1" in result.output
    assert "Verdict: OK" in result.output


def test_doctor_retries_smoke_without_p2p(monkeypatch):
    result, calls = _run_doctor(monkeypatch, _fake_torch(2), [_smoke(False), _smoke(True)])
    assert result.exit_code == 1, result.output
    assert calls[1] == (2, "nccl", {"NCCL_P2P_DISABLE": "1"})
    assert "PROBLEMS FOUND" in result.output
    assert "passed with NCCL_P2P_DISABLE=1" in result.output


def test_doctor_without_gpus_exits_nonzero(monkeypatch):
    result, calls = _run_doctor(monkeypatch, _fake_torch(0), [])
    assert result.exit_code == 1 and calls == []
    assert "No GPU" in result.output
