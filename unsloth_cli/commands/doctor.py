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
import tempfile
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
    hip = getattr(torch.version, "hip", None)
    # AMD SDK wheels leave version.hip unset and only say "rocm" in __version__.
    if hip is None and "rocm" in str(torch.__version__).lower():
        hip = str(torch.__version__).lower().split("rocm", 1)[1] or "unknown"
    info = {
        "torch": torch.__version__,
        "cuda": getattr(torch.version, "cuda", None),
        "hip": hip,
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
        # torch < 2.8 has no pci_bus_id; the UUID still identifies the card to NVML.
        uuid = getattr(p, "uuid", None)
        uuid = str(uuid) if uuid is not None and not hip else None
        info["gpus"].append(
            {
                "index": i,
                "name": p.name,
                "capability": f"{p.major}.{p.minor}",
                "memory_gib": round(p.total_memory / 1024**3, 1),
                "pci": pci,
                "uuid": uuid,
            }
        )
    return info


def driver_version(pynvml = None) -> Optional[str]:
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            try:
                v = pynvml.nvmlSystemGetDriverVersion()
                return v.decode() if isinstance(v, bytes) else str(v)
            finally:
                pynvml.nvmlShutdown()
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


def nvml_topology(pynvml, gpus: List[dict]):
    """({(a, b): {"level": "PHB", "nvlink": bool, "p2p_read": bool}} keyed by torch index, error).

    Handles are looked up by PCI bus id (else device UUID), since NVML numbers GPUs in
    PCI order and ignores CUDA_VISIBLE_DEVICES / CUDA_DEVICE_ORDER, which renumber them.
    A query a virtualised GPU refuses degrades that field instead of the whole table.
    """
    if pynvml is None:
        return None, "nvidia-ml-py is not installed"
    if len(gpus) < 2 or any(g["pci"] is None and not g.get("uuid") for g in gpus):
        return None, "torch reports neither PCI bus ids nor device UUIDs"
    read_idx = getattr(pynvml, "NVML_P2P_CAPS_INDEX_READ", 0)
    nvlink_idx = getattr(pynvml, "NVML_P2P_CAPS_INDEX_NVLINK", 2)
    status_ok = getattr(pynvml, "NVML_P2P_STATUS_OK", 0)

    def query(fn, *args):
        try:
            return fn(*args)
        except Exception:
            return None

    try:
        pynvml.nvmlInit()
    except Exception as e:
        return None, f"nvmlInit failed: {e}"
    try:
        handles = []
        for g in gpus:
            if g["pci"] is not None:
                handles.append(pynvml.nvmlDeviceGetHandleByPciBusId(g["pci"].encode()))
            else:
                uuid = g["uuid"]
                if not uuid.startswith(("GPU-", "MIG-")):
                    uuid = "GPU-" + uuid
                handles.append(pynvml.nvmlDeviceGetHandleByUUID(uuid.encode()))
        out = {}
        for a in range(len(gpus)):
            for b in range(a + 1, len(gpus)):
                ha, hb = handles[a], handles[b]
                level = query(pynvml.nvmlDeviceGetTopologyCommonAncestor, ha, hb)
                out[(a, b)] = {
                    "level": "?" if level is None else _TOPOLOGY_LABELS.get(int(level), str(level)),
                    "nvlink": query(pynvml.nvmlDeviceGetP2PStatus, ha, hb, nvlink_idx) == status_ok,
                    # Directional: a virtualised or IOMMU-restricted host can grant one way only.
                    "p2p_read": all(
                        query(pynvml.nvmlDeviceGetP2PStatus, x, y, read_idx) == status_ok
                        for x, y in ((ha, hb), (hb, ha))
                    ),
                }
        return out, None
    except Exception as e:
        return None, f"NVML query failed: {type(e).__name__}: {e}"
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def p2p_copy_probe_with_deadline(torch, count: int, timeout: float):
    """p2p_copy_probe in a daemon thread: a wedged peer copy cannot hold up the verdict.
    Returns (results, hung)."""
    import threading

    out = {}
    active = []
    crashed = []

    def run():
        try:
            p2p_copy_probe(torch, count, out = out, active = active)
        except Exception as e:
            crashed.append(f"{type(e).__name__}: {e}")

    worker = threading.Thread(target = run, daemon = True)
    worker.start()
    worker.join(timeout)
    hung = worker.is_alive()
    # Snapshot: a hung worker may still write. Pairs finished before the deadline keep their result.
    results = dict(out)
    for a in range(count):
        for b in range(count):
            if a == b or (a, b) in results:
                continue
            if hung and active and active[-1] == (a, b):
                results[(a, b)] = f"error: peer copy did not finish within {timeout:.0f}s (hung)"
            elif hung:
                results[(a, b)] = "inconclusive: not reached before the probe deadline"
            elif crashed:
                results[(a, b)] = f"inconclusive: probe stopped: {crashed[0]}"
    return results, hung


def p2p_copy_probe(
    torch,
    count: int,
    mib: int = 16,
    out: Optional[dict] = None,
    active: Optional[list] = None,
) -> dict:
    """{(src, dst): "ok" | "corrupt" | "error: ..." | "inconclusive: ..."} per ordered GPU pair.

    The destination is sentinel-filled first: on a translating IOMMU a PCIe peer copy
    can be dropped while CUDA reports success (scripts/p2p_integrity_probe.py).
    """
    out = {} if out is None else out
    numel = mib * 1024 * 1024 // 4
    for src in range(count):
        for dst in range(count):
            if src == dst:
                continue
            if active is not None:
                active.append((src, dst))
            s = d = None
            try:
                s = torch.arange(numel, dtype = torch.float32, device = f"cuda:{src}")
                d = torch.full((numel,), _SENTINEL, dtype = torch.float32, device = f"cuda:{dst}")
            except Exception as e:
                # No copy was attempted, so this says nothing about peer copies.
                out[(src, dst)] = f"inconclusive: could not allocate: {e}"
                del s, d
                continue
            try:
                d.copy_(s)
                torch.cuda.synchronize(src)
                torch.cuda.synchronize(dst)
                out[(src, dst)] = "ok" if torch.equal(d.cpu(), s.cpu()) else "corrupt"
                del s, d
            except Exception as e:
                out[(src, dst)] = f"error: {e}"
    torch.cuda.empty_cache()
    return out


def _nccl_p2p_disabled() -> bool:
    """True when the inherited NCCL_P2P_DISABLE already turns P2P off.

    NCCL's ncclGetLevel (src/graph/paths.cc) parses it with strtol and disables only on 1.
    """
    try:
        return int(os.environ.get("NCCL_P2P_DISABLE", "").strip() or "0", 0) == 1
    except ValueError:
        return False


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
    logs = []
    try:
        # Files, not pipes: an undrained pipe (e.g. NCCL_DEBUG=INFO on rank 1) blocks that
        # rank inside the collective and the doctor would report a healthy system as hung.
        for rank in range(world):
            logs.append(tempfile.TemporaryFile(mode = "w+"))
            rank_env = dict(env, RANK = str(rank), LOCAL_RANK = str(rank))
            procs.append(
                subprocess.Popen(
                    [sys.executable, "-c", _SMOKE_WORKER],
                    env = rank_env,
                    stdout = logs[rank],
                    stderr = subprocess.STDOUT,
                    text = True,
                )
            )
    except Exception as e:
        # Started ranks would wait in init_process_group for ranks that never exist.
        for q in procs:
            q.kill()
            q.wait()
        for f in logs:
            f.close()
        return {
            "ok": False,
            "hung": False,
            "busbw_gbps": None,
            "log": f"could not start rank {len(procs)} of {world}: {type(e).__name__}: {e}",
        }
    # Startup (torch import, communicator init) is not covered by the collective timeout.
    deadline = time.monotonic() + timeout + 60
    hung = False
    outputs = []
    try:
        for p in procs:
            try:
                p.wait(timeout = max(1.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                hung = True
                for q in procs:
                    q.kill()
                break
        for f in logs:
            f.seek(0)
            outputs.append(f.read())
    finally:
        # Ctrl-C on a hung run must not leave ranks holding GPU contexts behind. Bounded:
        # a rank stuck in an uninterruptible driver call may never be reaped.
        for q in procs:
            if q.poll() is None:
                q.kill()
        for q in procs:
            try:
                q.wait(timeout = 10)
            except subprocess.TimeoutExpired:
                hung = True
        for f in logs:
            f.close()
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
    bad_pairs = sorted(k for k, v in (p2p or {}).items() if v == "corrupt" or v.startswith("error"))
    unchecked = sorted(k for k, v in (p2p or {}).items() if v.startswith("inconclusive"))
    if bad_pairs:
        healthy = False
        hints.append(
            f"Peer-to-peer copies lost data or failed on {bad_pairs}. Set "
            "NCCL_P2P_DISABLE=1 for every distributed run on this host, and check the "
            "kernel log for IOMMU faults: dmesg | grep -i -E 'DMAR|AMD-Vi'."
        )
    if unchecked:
        reason = p2p[unchecked[0]][len("inconclusive: ") :]
        hints.append(
            f"Peer copies on {unchecked} were not tested ({reason}). Free GPU memory and "
            "re-run `unsloth doctor --gpus`."
        )
    if smoke is not None and smoke.get("unverified"):
        healthy = False
        hints.append(
            "This PyTorch build has no NCCL backend (for example on Windows), so GPU "
            "collectives could not be tested and multi-GPU training here is unverified."
        )
    elif smoke is not None and not smoke["ok"]:
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
        far = sorted(
            k for k, v in topology.items() if not v["nvlink"] and v["level"] in _PCIE_HOST_LEVELS
        )
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
                f"NVML does not confirm P2P reads for {no_p2p} (common on cloud VMs). "
                "If NCCL is slow there it is staging transfers through host memory."
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
    except Exception as e:
        # A missing or mismatched CUDA / DLL dependency surfaces as OSError, not ImportError.
        typer.echo(f"torch failed to import: {type(e).__name__}: {e}")
        typer.echo("\n=== Verdict: PROBLEMS FOUND ===")
        typer.echo("- Reinstall torch with a build matching this driver and CUDA / ROCm runtime.")
        raise typer.Exit(code = 1)

    try:
        info = collect_gpus(torch)
    except Exception as e:
        # A driver / runtime mismatch or a faulted GPU fails here, after a clean torch import.
        typer.echo(f"GPU enumeration failed: {type(e).__name__}: {e}")
        typer.echo("\n=== Verdict: PROBLEMS FOUND ===")
        typer.echo(
            "- Check that the driver matches this torch build and that every visible GPU is healthy."
        )
        raise typer.Exit(code = 1)
    pynvml = _load_pynvml()
    probe_hung = False
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
        topology, topology_error = nvml_topology(pynvml, info["gpus"])
        typer.echo("\n=== Topology (NVML) ===")
        if topology is None:
            typer.echo(f"  unavailable: {topology_error}")
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
            p2p, probe_hung = p2p_copy_probe_with_deadline(torch, count, timeout)
            for (a, b), v in sorted(p2p.items()):
                if probe_hung:
                    typer.echo(f"  {a}->{b}  {v}")
                    continue
                peer = "yes" if torch.cuda.can_device_access_peer(a, b) else "no"
                typer.echo(f"  {a}->{b}  peer_access={peer}  {v}")
        typer.echo(f"\n=== all_reduce smoke ({count} ranks, nccl) ===")
        if not torch.distributed.is_nccl_available():
            # A gloo run would put the tensors on the CPU and prove nothing about the GPUs.
            smoke = {"ok": False, "hung": False, "unverified": True, "busbw_gbps": None, "log": ""}
            typer.echo("  skipped: this PyTorch build has no NCCL backend")
        else:
            smoke = allreduce_smoke(count, "nccl", timeout)
            if smoke["ok"]:
                typer.echo(f"  ok  busbw {smoke['busbw_gbps']:.2f} GB/s (64 MiB fp32)")
            else:
                typer.echo(f"  {'HUNG' if smoke['hung'] else 'FAILED'}\n{smoke['log']}")
                if not _nccl_p2p_disabled():
                    smoke_no_p2p = allreduce_smoke(
                        count, "nccl", timeout, extra_env = {"NCCL_P2P_DISABLE": "1"}
                    )
                    typer.echo(
                        f"  retry with NCCL_P2P_DISABLE=1: {'ok' if smoke_no_p2p['ok'] else 'failed'}"
                    )

    healthy, hints = verdict(info, topology, p2p, smoke, smoke_no_p2p)
    typer.echo(f"\n=== Verdict: {'OK' if healthy else 'PROBLEMS FOUND'} ===")
    for h in hints:
        typer.echo(f"- {h}")
    if probe_hung:
        # The wedged CUDA call would also hang interpreter shutdown.
        sys.stdout.flush()
        os._exit(1)
    raise typer.Exit(code = 0 if healthy else 1)
