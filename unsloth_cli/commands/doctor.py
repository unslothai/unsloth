# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth doctor --gpus`: a multi-GPU preflight to run before a distributed job.

Reports the GPUs, versions and interconnect, checks that peer copies keep their
data, runs a short all_reduce across every visible GPU, and ends with a verdict
and the settings to try. torch is imported lazily so `unsloth --help` stays fast.
"""

import os
import shutil
import socket
import subprocess
import sys
import time
from typing import List, Optional, Tuple

import typer

# NVML topology levels (nvmlGpuTopologyLevel_t), closest first.
_TOPOLOGY_LABELS = {0: "SAME", 10: "PIX", 20: "PXB", 30: "PHB", 40: "NODE", 50: "SYS"}
_PCIE_HOST_LEVELS = {"PHB", "NODE", "SYS"}
_SENTINEL = -7.0

_SMOKE_WORKER = r"""
import datetime, os, sys, time
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
backend = os.environ["UNSLOTH_DOCTOR_BACKEND"]
if backend == "nccl":
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
else:
    device = torch.device("cpu")
dist.init_process_group(
    backend,
    timeout = datetime.timedelta(seconds = float(os.environ["UNSLOTH_DOCTOR_TIMEOUT"])),
)
numel = int(os.environ["UNSLOTH_DOCTOR_NUMEL"])
x = torch.full((numel,), float(rank + 1), device = device)
expected = world * (world + 1) / 2
ok = True
dist.all_reduce(x)
ok = ok and bool((x == expected).all())
iters = 5
dist.barrier()
if device.type == "cuda":
    torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(iters):
    x.fill_(1.0)
    dist.all_reduce(x)
if device.type == "cuda":
    torch.cuda.synchronize()
elapsed = (time.perf_counter() - start) / iters
ok = ok and bool((x == world).all())
# nccl-tests busbw convention for all_reduce: algbw * 2 (n - 1) / n.
busbw = numel * x.element_size() / elapsed * 2 * (world - 1) / world / 1e9
if rank == 0:
    print(f"UNSLOTH_DOCTOR_RESULT ok={int(ok)} busbw_gbps={busbw:.2f}", flush = True)
dist.destroy_process_group()
sys.exit(0 if ok else 3)
"""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def collect_gpus(torch) -> dict:
    info = {
        "torch": torch.__version__,
        "cuda": getattr(torch.version, "cuda", None),
        "hip": getattr(torch.version, "hip", None),
        "nccl": None,
        "gpus": [],
    }
    if not torch.cuda.is_available():
        return info
    try:
        nccl = torch.cuda.nccl.version()
        info["nccl"] = ".".join(str(v) for v in nccl) if isinstance(nccl, tuple) else str(nccl)
    except Exception:
        pass
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        pci = None
        if getattr(p, "pci_bus_id", None) is not None:
            pci = (
                f"{getattr(p, 'pci_domain_id', 0):08x}:{p.pci_bus_id:02x}:"
                f"{getattr(p, 'pci_device_id', 0):02x}.0"
            )
        info["gpus"].append(
            {
                "index": i,
                "name": p.name,
                "capability": f"{p.major}.{p.minor}",
                "memory_gib": round(p.total_memory / 1024**3, 1),
                "pci": pci,
            }
        )
    return info


def driver_version(pynvml = None) -> Optional[str]:
    if pynvml is not None:
        try:
            v = pynvml.nvmlSystemGetDriverVersion()
            return v.decode() if isinstance(v, bytes) else str(v)
        except Exception:
            pass
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output = True,
            text = True,
            timeout = 20,
        )
        lines = out.stdout.strip().splitlines()
        return lines[0].strip() if out.returncode == 0 and lines else None
    except Exception:
        return None


def nvml_topology(pynvml, gpus: List[dict]) -> Optional[dict]:
    """{(a, b): {"level": "PHB", "nvlink": bool, "p2p_read": bool}} keyed by torch index.

    Handles are looked up by PCI bus id, since NVML numbers GPUs in PCI order and
    ignores CUDA_VISIBLE_DEVICES / CUDA_DEVICE_ORDER, which renumber them for torch.
    """
    if pynvml is None or len(gpus) < 2 or any(g["pci"] is None for g in gpus):
        return None
    read_idx = getattr(pynvml, "NVML_P2P_CAPS_INDEX_READ", 0)
    nvlink_idx = getattr(pynvml, "NVML_P2P_CAPS_INDEX_NVLINK", 2)
    status_ok = getattr(pynvml, "NVML_P2P_STATUS_OK", 0)
    try:
        pynvml.nvmlInit()
        handles = [pynvml.nvmlDeviceGetHandleByPciBusId(g["pci"].encode()) for g in gpus]
        out = {}
        for a in range(len(gpus)):
            for b in range(a + 1, len(gpus)):
                level = pynvml.nvmlDeviceGetTopologyCommonAncestor(handles[a], handles[b])
                out[(a, b)] = {
                    "level": _TOPOLOGY_LABELS.get(int(level), str(level)),
                    "nvlink": pynvml.nvmlDeviceGetP2PStatus(handles[a], handles[b], nvlink_idx)
                    == status_ok,
                    "p2p_read": pynvml.nvmlDeviceGetP2PStatus(handles[a], handles[b], read_idx)
                    == status_ok,
                }
        return out
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def p2p_copy_probe(torch, count: int, mib: int = 16) -> dict:
    """{(src, dst): "ok" | "corrupt" | "error: ..."} for every ordered GPU pair.

    The destination is sentinel-filled first: on a translating IOMMU a PCIe peer copy
    can be dropped while CUDA reports success (scripts/p2p_integrity_probe.py).
    """
    out = {}
    numel = mib * 1024 * 1024 // 4
    for src in range(count):
        for dst in range(count):
            if src == dst:
                continue
            try:
                s = torch.arange(numel, dtype = torch.float32, device = f"cuda:{src}")
                d = torch.full((numel,), _SENTINEL, dtype = torch.float32, device = f"cuda:{dst}")
                d.copy_(s)
                torch.cuda.synchronize(src)
                torch.cuda.synchronize(dst)
                out[(src, dst)] = "ok" if torch.equal(d.cpu(), s.cpu()) else "corrupt"
                del s, d
            except Exception as e:
                out[(src, dst)] = f"error: {e}"
    torch.cuda.empty_cache()
    return out


def allreduce_smoke(
    world: int,
    backend: str,
    timeout: float,
    extra_env: Optional[dict] = None,
    numel: int = 16 * 1024 * 1024,
) -> dict:
    """Run one all_reduce over `world` local ranks in fresh processes.

    Separate interpreters rather than threads: a hung NCCL communicator can only be
    recovered by killing its process, and the doctor must answer instead of hanging.
    """
    env = dict(os.environ)
    env.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(_free_port()),
        WORLD_SIZE = str(world),
        UNSLOTH_DOCTOR_BACKEND = backend,
        UNSLOTH_DOCTOR_TIMEOUT = str(timeout),
        UNSLOTH_DOCTOR_NUMEL = str(numel),
    )
    env.update(extra_env or {})
    procs = []
    for rank in range(world):
        rank_env = dict(env, RANK = str(rank), LOCAL_RANK = str(rank))
        procs.append(
            subprocess.Popen(
                [sys.executable, "-c", _SMOKE_WORKER],
                env = rank_env,
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
                text = True,
            )
        )
    # Startup (torch import, communicator init) is not covered by the collective timeout.
    deadline = time.monotonic() + timeout + 60
    hung = False
    outputs = []
    for p in procs:
        try:
            out, _ = p.communicate(timeout = max(1.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            hung = True
            for q in procs:
                q.kill()
            out, _ = p.communicate()
        outputs.append(out or "")
    result = {"ok": False, "hung": hung, "busbw_gbps": None, "log": ""}
    for line in outputs[0].splitlines():
        if line.startswith("UNSLOTH_DOCTOR_RESULT"):
            fields = dict(kv.split("=", 1) for kv in line.split()[1:])
            result["busbw_gbps"] = float(fields["busbw_gbps"])
            result["ok"] = fields["ok"] == "1"
    result["ok"] = result["ok"] and not hung and all(p.returncode == 0 for p in procs)
    if not result["ok"]:
        result["log"] = "\n".join(o[-2000:] for o in outputs)
    return result


def verdict(
    info: dict,
    topology: Optional[dict],
    p2p: Optional[dict],
    smoke: Optional[dict],
    smoke_no_p2p: Optional[dict],
) -> Tuple[bool, List[str]]:
    """(healthy, hints). Pure, so every branch is testable without GPUs."""
    hints = []
    gpus = info["gpus"]
    if not gpus:
        return False, [
            "No GPU is visible to torch. Check `nvidia-smi`, CUDA_VISIBLE_DEVICES, and "
            "that torch was installed with CUDA or ROCm support."
        ]
    if len(gpus) == 1:
        return True, ["One GPU visible: multi-GPU checks skipped."]
    healthy = True
    if len({g["capability"] for g in gpus}) > 1 or len({g["name"] for g in gpus}) > 1:
        hints.append(
            "The GPUs differ in model or compute capability. Distributed training runs at "
            "the pace of the slowest card, and bf16 / FlashAttention need every card to "
            "support them."
        )
    bad_pairs = sorted(k for k, v in (p2p or {}).items() if v != "ok")
    if bad_pairs:
        healthy = False
        hints.append(
            f"Peer-to-peer copies lost data or failed on {bad_pairs}. Set "
            "NCCL_P2P_DISABLE=1 for every distributed run on this host, and check the "
            "kernel log for IOMMU faults: dmesg | grep -i -E 'DMAR|AMD-Vi'."
        )
    if smoke is not None and not smoke["ok"]:
        healthy = False
        what = "hung" if smoke["hung"] else "failed"
        if smoke_no_p2p is not None and smoke_no_p2p["ok"]:
            hints.append(
                f"The all_reduce smoke test {what}, and passed with NCCL_P2P_DISABLE=1. "
                "Export NCCL_P2P_DISABLE=1 before launching distributed training here."
            )
        else:
            hints.append(
                f"The all_reduce smoke test {what}"
                + (" even with NCCL_P2P_DISABLE=1" if smoke_no_p2p is not None else "")
                + ". Re-run your job with UNSLOTH_DEBUG_DISTRIBUTED=1 (NCCL_DEBUG=INFO, "
                "flight recorder, desync debug) to see where it stops."
            )
    if topology:
        nvlinked = all(v["nvlink"] for v in topology.values())
        far = sorted(k for k, v in topology.items() if not v["nvlink"] and v["level"] in _PCIE_HOST_LEVELS)
        if far:
            hints.append(
                f"GPU pairs {far} talk through the PCIe host bridge or the CPU "
                "interconnect (PHB / NODE / SYS), with no NVLink. For LoRA prefer DDP: "
                "it only all-reduces the small adapter gradients, while FSDP / ZeRO-3 "
                "and tensor parallel move the full weights every step over this link."
            )
        elif nvlinked:
            hints.append("Every GPU pair is NVLinked: FSDP and tensor parallel are viable.")
        no_p2p = sorted(k for k, v in topology.items() if not v["p2p_read"])
        if no_p2p and not bad_pairs:
            hints.append(
                f"NVML reports no P2P read support for {no_p2p}; NCCL will stage those "
                "transfers through host memory, which is slower but correct."
            )
    if healthy and not hints:
        hints.append("No problems found.")
    return healthy, hints


def _load_pynvml():
    try:
        import pynvml

        return pynvml
    except Exception:
        return None


def doctor(
    gpus: bool = typer.Option(
        True,
        "--gpus/--no-gpus",
        help = "Check GPUs, interconnect and a multi-GPU all_reduce (the default check).",
    ),
    timeout: float = typer.Option(
        60.0,
        "--timeout",
        help = "Seconds before the all_reduce smoke test is declared hung.",
    ),
):
    """Preflight checks for this machine. Exit status 1 means a problem was found."""
    if not gpus:
        typer.echo("Nothing to check.")
        raise typer.Exit()
    try:
        import torch
    except ImportError:
        typer.echo("torch is not installed; nothing to check.")
        raise typer.Exit(code = 1)

    info = collect_gpus(torch)
    pynvml = _load_pynvml()
    typer.echo("=== Versions ===")
    typer.echo(f"torch {info['torch']}")
    if info["hip"]:
        typer.echo(f"ROCm (HIP) {info['hip']}")
    else:
        typer.echo(f"CUDA (torch build) {info['cuda']}")
    typer.echo(f"driver {driver_version(pynvml) or 'unknown'}")
    typer.echo(f"NCCL {info['nccl'] or 'unavailable'}")
    typer.echo(f"\n=== GPUs ({len(info['gpus'])}) ===")
    for g in info["gpus"]:
        typer.echo(
            f"  cuda:{g['index']}  {g['name']}  sm{g['capability'].replace('.', '')}  "
            f"{g['memory_gib']} GiB  {g['pci'] or ''}"
        )

    topology = p2p = smoke = smoke_no_p2p = None
    count = len(info["gpus"])
    if count > 1:
        topology = nvml_topology(pynvml, info["gpus"])
        typer.echo("\n=== Topology (NVML) ===")
        if topology is None:
            typer.echo("  unavailable (install nvidia-ml-py for the topology summary)")
        else:
            for (a, b), v in sorted(topology.items()):
                typer.echo(
                    f"  {a}<->{b}  {v['level']:<4}  nvlink={'yes' if v['nvlink'] else 'no'}  "
                    f"p2p_read={'yes' if v['p2p_read'] else 'no'}"
                )
        typer.echo("\n=== Peer copy integrity ===")
        if info["hip"]:
            typer.echo("  skipped on ROCm")
        else:
            p2p = p2p_copy_probe(torch, count)
            for (a, b), v in sorted(p2p.items()):
                peer = "yes" if torch.cuda.can_device_access_peer(a, b) else "no"
                typer.echo(f"  {a}->{b}  peer_access={peer}  {v}")
        backend = "nccl" if torch.distributed.is_nccl_available() else "gloo"
        typer.echo(f"\n=== all_reduce smoke ({count} ranks, {backend}) ===")
        smoke = allreduce_smoke(count, backend, timeout)
        if smoke["ok"]:
            typer.echo(f"  ok  busbw {smoke['busbw_gbps']:.2f} GB/s (64 MiB fp32)")
        else:
            typer.echo(f"  {'HUNG' if smoke['hung'] else 'FAILED'}\n{smoke['log']}")
            if backend == "nccl" and os.environ.get("NCCL_P2P_DISABLE") is None:
                smoke_no_p2p = allreduce_smoke(
                    count, backend, timeout, extra_env = {"NCCL_P2P_DISABLE": "1"}
                )
                typer.echo(
                    f"  retry with NCCL_P2P_DISABLE=1: {'ok' if smoke_no_p2p['ok'] else 'failed'}"
                )

    healthy, hints = verdict(info, topology, p2p, smoke, smoke_no_p2p)
    typer.echo(f"\n=== Verdict: {'OK' if healthy else 'PROBLEMS FOUND'} ===")
    for h in hints:
        typer.echo(f"- {h}")
    raise typer.Exit(code = 0 if healthy else 1)
