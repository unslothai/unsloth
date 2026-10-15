# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sys
import threading
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


def _gpu(
    index,
    name = "Tesla T4",
    cap = "7.5",
    pci = None,
):
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
    topo, err = doctor_mod.nvml_topology(nvml, gpus)
    assert err is None
    assert topo[(0, 1)] == {"level": "SYS", "nvlink": False, "p2p_read": True}
    assert topo[(1, 2)] == {"level": "PHB", "nvlink": True, "p2p_read": True}
    assert nvml.shutdown


def test_nvml_topology_absent_without_pynvml_or_pci():
    topo, err = doctor_mod.nvml_topology(None, [_gpu(0, pci = "a"), _gpu(1, pci = "b")])
    assert topo is None and "nvidia-ml-py" in err
    topo, err = doctor_mod.nvml_topology(_FakeNvml(), [_gpu(0), _gpu(1)])
    assert topo is None and "PCI" in err


def test_nvml_topology_survives_a_refused_query():
    class Refusing(_FakeNvml):
        def nvmlDeviceGetP2PStatus(self, a, b, idx):
            raise RuntimeError("Not Supported")

    topo, err = doctor_mod.nvml_topology(Refusing(), [_gpu(0, pci = "bus-a"), _gpu(1, pci = "bus-c")])
    assert err is None
    assert topo[(0, 1)] == {"level": "SYS", "nvlink": False, "p2p_read": False}


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


def _run_doctor(
    monkeypatch,
    torch,
    smoke_results,
    p2p_disable = None,
):
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(doctor_mod, "_load_pynvml", lambda: None)
    monkeypatch.setattr(doctor_mod, "driver_version", lambda pynvml = None: "550.1")
    monkeypatch.setattr(doctor_mod, "p2p_copy_probe", lambda t, n: {(0, 1): "ok", (1, 0): "ok"})
    calls = []

    def fake_smoke(
        world,
        backend,
        timeout,
        extra_env = None,
    ):
        calls.append((world, backend, extra_env))
        return smoke_results[len(calls) - 1]

    monkeypatch.setattr(doctor_mod, "allreduce_smoke", fake_smoke)
    monkeypatch.delenv("NCCL_P2P_DISABLE", raising = False)
    if p2p_disable is not None:
        monkeypatch.setenv("NCCL_P2P_DISABLE", p2p_disable)
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


def test_doctor_without_nccl_is_unverified_not_a_cpu_gloo_pass(monkeypatch):
    torch = _fake_torch(2)
    torch.distributed = SimpleNamespace(is_nccl_available = lambda: False)
    result, calls = _run_doctor(monkeypatch, torch, [_smoke(True)])
    assert calls == []
    assert result.exit_code == 1, result.output
    assert "no NCCL backend" in result.output and "Verdict: OK" not in result.output


def test_p2p_allocation_failure_is_inconclusive_not_corruption():
    class OomTorch:
        float32 = "f32"

        @staticmethod
        def arange(*a, device, **k):
            if device == "cuda:1":
                raise RuntimeError("CUDA out of memory")
            return SimpleNamespace()

        full = arange
        cuda = SimpleNamespace(empty_cache = lambda: None)

    p2p = doctor_mod.p2p_copy_probe(OomTorch, 2, mib = 1)
    assert all(v.startswith("inconclusive") for v in p2p.values()), p2p
    healthy, hints = doctor_mod.verdict(_info(_gpu(0), _gpu(1)), None, p2p, _smoke(True), None)
    assert healthy
    assert not any("NCCL_P2P_DISABLE" in h for h in hints)
    assert any("not tested" in h for h in hints)


def test_rocm_wheel_without_version_hip_is_rocm():
    torch = _fake_torch(0)
    torch.__version__ = "2.5.1+rocm6.2"
    torch.version = SimpleNamespace(cuda = None, hip = None)
    assert doctor_mod.collect_gpus(torch)["hip"] == "6.2"


def test_allreduce_smoke_survives_a_chatty_nonzero_rank(monkeypatch, tmp_path):
    # Rank 1 writes far more than a pipe buffer (64 KiB) before rank 0 can finish.
    worker = (
        "import os, sys, time\n"
        "if os.environ['RANK'] == '1':\n"
        "    sys.stdout.write('x' * (4 << 20)); sys.stdout.flush(); sys.exit(0)\n"
        "while not os.path.exists(os.environ['DONE']): time.sleep(0.05)\n"
        "print('UNSLOTH_DOCTOR_RESULT ok=1 busbw_gbps=1.00', flush = True)\n"
    )
    done = tmp_path / "rank1_done"
    monkeypatch.setattr(
        doctor_mod, "_SMOKE_WORKER", worker.replace("os.environ['DONE']", repr(str(done)))
    )
    real_popen = doctor_mod.subprocess.Popen
    procs = []

    def popen(*a, **k):
        p = real_popen(*a, **k)
        procs.append(p)
        if len(procs) == 2:
            # Rank 0 only finishes once rank 1 has exited, like a collective would.
            threading.Thread(target = lambda: (p.wait(), done.touch()), daemon = True).start()
        return p

    monkeypatch.setattr(doctor_mod.subprocess, "Popen", popen)
    result = doctor_mod.allreduce_smoke(2, "gloo", timeout = -45, numel = 1024)
    assert result["ok"] and not result["hung"], result["log"][-300:]


def test_driver_version_initialises_nvml():
    class Nvml:
        ready = False

        def nvmlInit(self):
            self.ready = True

        def nvmlShutdown(self):
            self.ready = False

        def nvmlSystemGetDriverVersion(self):
            if not self.ready:
                raise RuntimeError("Uninitialized")
            return b"580.65"

    nvml = Nvml()
    assert doctor_mod.driver_version(nvml) == "580.65"
    assert not nvml.ready


def test_nvml_topology_falls_back_to_uuid_without_pci_bus_id():
    class ByUuid(_FakeNvml):
        def nvmlDeviceGetHandleByPciBusId(self, bus):
            raise AssertionError("no PCI id to look up")

        def nvmlDeviceGetHandleByUUID(self, uuid):
            return {b"GPU-u0": "bus-a", b"GPU-u1": "bus-c"}[uuid]

    torch = _fake_torch(2)
    props = SimpleNamespace(name = "Tesla T4", major = 7, minor = 5, total_memory = 15 * 1024**3)
    torch.cuda.get_device_properties = lambda i: SimpleNamespace(**vars(props), uuid = f"u{i}")
    gpus = doctor_mod.collect_gpus(torch)["gpus"]
    assert [g["pci"] for g in gpus] == [None, None]
    topo, err = doctor_mod.nvml_topology(ByUuid(), gpus)
    assert err is None
    assert topo[(0, 1)]["level"] == "SYS"


def test_allreduce_smoke_kills_started_ranks_when_a_spawn_fails(monkeypatch):
    real_popen = doctor_mod.subprocess.Popen
    started = []

    def flaky_popen(*args, **kwargs):
        if started:
            raise OSError("Resource temporarily unavailable")
        started.append(real_popen(*args, **kwargs))
        return started[-1]

    monkeypatch.setattr(doctor_mod, "_SMOKE_WORKER", "import time; time.sleep(60)")
    monkeypatch.setattr(doctor_mod.subprocess, "Popen", flaky_popen)
    result = doctor_mod.allreduce_smoke(2, "gloo", timeout = 60, numel = 1024)
    assert not result["ok"] and not result["hung"]
    assert "could not start rank 1" in result["log"]
    assert started[0].returncode is not None


def test_doctor_reports_a_torch_that_fails_to_load(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def broken(name, *args, **kwargs):
        if name == "torch":
            raise OSError("libcudnn.so.9: cannot open shared object file")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "torch", raising = False)
    monkeypatch.setattr(builtins, "__import__", broken)
    cli = typer.Typer()
    cli.command()(doctor_mod.doctor)
    result = CliRunner().invoke(cli, ["--gpus"])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "libcudnn" in result.output and "PROBLEMS FOUND" in result.output


@pytest.mark.parametrize("value, retried", [("0", True), ("", True), ("1", False)])
def test_doctor_retry_honours_the_inherited_nccl_p2p_disable_value(monkeypatch, value, retried):
    result, calls = _run_doctor(
        monkeypatch, _fake_torch(2), [_smoke(False), _smoke(True)], p2p_disable = value
    )
    assert result.exit_code == 1, result.output
    assert len(calls) == (2 if retried else 1)


def test_nvml_topology_requires_p2p_read_in_both_directions():
    class OneWay(_FakeNvml):
        def nvmlDeviceGetP2PStatus(self, a, b, idx):
            if idx == self.NVML_P2P_CAPS_INDEX_READ:
                return 0 if (a, b) == ("bus-a", "bus-c") else 1
            return 1

    gpus = [_gpu(0, pci = "bus-a"), _gpu(1, pci = "bus-c")]
    topo, err = doctor_mod.nvml_topology(OneWay(), gpus)
    assert err is None
    assert topo[(0, 1)]["p2p_read"] is False


def test_allreduce_smoke_reaps_ranks_when_the_wait_is_interrupted(monkeypatch):
    real_popen = doctor_mod.subprocess.Popen
    started = []

    def popen(*args, **kwargs):
        started.append(real_popen(*args, **kwargs))
        return started[-1]

    def interrupted_wait(self, timeout = None):
        if timeout is not None:
            raise KeyboardInterrupt
        return real_wait(self)

    real_wait = doctor_mod.subprocess.Popen.wait
    monkeypatch.setattr(doctor_mod, "_SMOKE_WORKER", "import time; time.sleep(60)")
    monkeypatch.setattr(doctor_mod.subprocess, "Popen", popen)
    monkeypatch.setattr(real_popen, "wait", interrupted_wait)
    with pytest.raises(KeyboardInterrupt):
        doctor_mod.allreduce_smoke(2, "gloo", timeout = 60, numel = 1024)
    assert len(started) == 2
    assert all(p.poll() is not None for p in started)
