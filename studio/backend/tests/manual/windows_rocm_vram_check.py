# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""TEMPORARY hardware check for Windows + AMD ROCm. Delete before merge.

CI cannot reach these paths. The AMD runner is Linux, so
``rocm_windows_free_is_untrusted()`` is False there and the WDDM cap never
executes; ``_amd_smi_allowed()`` never takes its Windows branch either. Both
are currently argued from code and simulation, not measured.

This answers three questions on a real Windows ROCm box:

  1. Does ``free_gb`` move when ANOTHER process holds VRAM? That is the PR's
     entire claim, and it is the one that could silently be a no-op here:
     hipMemGetInfo is system-wide on Linux but process-local on Windows WDDM.
  2. Does reading the summary ever spawn amd-smi? On Windows without a HIP SDK
     it elevates a child and pops a UAC/DiskPart prompt, so the answer must be
     no spawns.
  3. What do the Windows-only gates actually evaluate to?

Run from ``studio/backend`` with the Unsloth venv's python::

    python tests/manual/windows_rocm_vram_check.py --gib 4

Add ``--json out.json`` to save the raw readings. Holding 4 GiB needs a card
with at least ~6 GiB free; lower it with ``--gib`` on a smaller GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

GIB = 1024**3

# Run from studio/backend, or from anywhere: resolve the backend root either way.
BACKEND = Path(__file__).resolve().parents[2]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


# ========== holder ==========


def run_holder(gib: float, max_seconds: int) -> int:
    """Hold VRAM in this process and announce readiness on stdout."""
    import torch

    if not torch.cuda.is_available():
        print(json.dumps({"status": "ERROR", "error": "no cuda/hip device"}), flush = True)
        return 2

    # TOUCHED, not merely allocated: an uncommitted allocation is not what a
    # resident model is, and on WDDM it may not be committed to VRAM at all.
    chunks, held, step = [], 0, 256 * 1024 * 1024
    want = int(gib * GIB)
    while held < want:
        t = torch.empty(step, dtype = torch.uint8, device = "cuda")
        t.fill_(1)
        chunks.append(t)
        held += step
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "status": "READY",
                "pid": os.getpid(),
                "held_gib": held / GIB,
                "allocated_gib": torch.cuda.memory_allocated() / GIB,
            }
        ),
        flush = True,
    )

    deadline = time.time() + max_seconds
    while time.time() < deadline:
        time.sleep(1)
    return 0


# ========== observer ==========


def _count_amd_smi_spawns(fn):
    """Run fn() with every subprocess spawn recorded, and return the amd-smi ones.

    Patches both entry points: amd.py uses subprocess.run, but a UAC prompt from
    a Popen would be just as visible to a user.
    """
    seen: list[list[str]] = []
    real_run, real_popen = subprocess.run, subprocess.Popen

    def record(args):
        try:
            argv = [str(a) for a in (args if isinstance(args, (list, tuple)) else [args])]
        except Exception:  # noqa: BLE001
            return
        if argv and "amd-smi" in os.path.basename(argv[0]).lower():
            seen.append(argv)

    def fake_run(*a, **kw):
        if a:
            record(a[0])
        return real_run(*a, **kw)

    def fake_popen(*a, **kw):
        if a:
            record(a[0])
        return real_popen(*a, **kw)

    subprocess.run, subprocess.Popen = fake_run, fake_popen
    try:
        result = fn()
    finally:
        subprocess.run, subprocess.Popen = real_run, real_popen
    return result, seen


def _reading(hw, torch) -> dict:
    """One observation. The observer must stay a bystander or this is self-measurement."""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    allocated = torch.cuda.memory_allocated()
    mem = hw.get_gpu_memory_info()
    out = {
        "free_gb": mem.get("free_gb"),
        "total_gb": mem.get("total_gb"),
        "allocated_gb": mem.get("allocated_gb"),
        "reserved_gb": mem.get("reserved_gb"),
        # The pre-PR expression, evaluated here so the base leg is present
        # without needing a second checkout. It is blind to other processes by
        # construction, and that is the point.
        "old_formula_free_gb": (props.total_memory - allocated) / GIB,
        "observer_allocated_gb": allocated / GIB,
    }
    try:
        free_b, _total_b = torch.cuda.mem_get_info()
        out["torch_mem_get_info_free_gb"] = free_b / GIB
    except Exception as e:  # noqa: BLE001
        out["torch_mem_get_info_error"] = f"{type(e).__name__}: {e}"
    return out


def _wait_for_ready(proc, timeout: float) -> dict | None:
    """Poll for the holder's READY line. Never sleep a fixed interval instead."""
    q: queue.Queue = queue.Queue()

    def pump():
        for line in proc.stdout:
            q.put(line)
        q.put(None)

    threading.Thread(target = pump, daemon = True).start()
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            line = q.get(timeout = 1.0)
        except queue.Empty:
            if proc.poll() is not None:
                return None
            continue
        if line is None:
            return None
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if msg.get("status") == "READY":
            return msg
        if msg.get("status") == "ERROR":
            return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__)
    ap.add_argument("--holder", action = "store_true", help = argparse.SUPPRESS)
    ap.add_argument("--gib", type = float, default = 4.0)
    ap.add_argument("--max-seconds", type = int, default = 300)
    ap.add_argument("--json", type = Path, default = None)
    args = ap.parse_args()

    if args.holder:
        return run_holder(args.gib, args.max_seconds)

    import torch

    from utils.hardware import hardware as hw

    # IS_ROCM is False at import and is only set by detect_hardware(). Reading
    # the gates without this reports every ROCm branch as inactive on a ROCm
    # box, and the run then describes a path the machine never took.
    try:
        detected = str(hw.detect_hardware())
    except Exception as e:  # noqa: BLE001
        detected = f"error: {type(e).__name__}: {e}"

    report: dict = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": getattr(torch.version, "cuda", None),
        "hardware_module": hw.__file__,
        "detect_hardware": detected,
    }

    if not torch.cuda.is_available():
        report["fatal"] = "no cuda/hip device visible to torch"
        print(json.dumps(report, indent = 2))
        return 2

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    report["gpu"] = {
        "name": props.name,
        "arch": getattr(props, "gcnArchName", None),
        "props_total_gb": props.total_memory / GIB,
        "is_integrated": getattr(props, "is_integrated", None),
    }

    # --- gates: what the Windows-only branches actually decide here ---
    def _safe(fn, *a):
        try:
            return fn(*a)
        except Exception as e:  # noqa: BLE001
            return f"error: {type(e).__name__}: {e}"

    gates = {
        "IS_ROCM": getattr(hw, "IS_ROCM", None),
        "rocm_windows_free_is_untrusted": _safe(hw.rocm_windows_free_is_untrusted),
        "rocm_props_total_is_carve_out": _safe(hw._rocm_props_total_is_carve_out, props),
        "rocm_device_ordinal_active": _safe(hw._rocm_device_ordinal_active),
    }
    try:
        from utils.hardware import amd as _amd
        gates["amd_smi_allowed"] = _safe(_amd._amd_smi_allowed)
    except Exception as e:  # noqa: BLE001
        gates["amd_smi_allowed"] = f"error: {type(e).__name__}: {e}"
    gates["env"] = {
        k: os.environ.get(k)
        for k in (
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "GPU_DEVICE_ORDINAL",
            "CUDA_VISIBLE_DEVICES",
            "UNSLOTH_ENABLE_AMD_SMI",
        )
    }
    report["gates"] = gates

    # --- question 2: does reading the summary spawn amd-smi at all? ---
    # Asked BEFORE anything below pins a context, since that is the state a real
    # idle backend is in.
    #
    # Two calls, because "0 spawns" means different things on different boxes.
    # With a HIP SDK present amd-smi is ALLOWED and a spawn is correct, so the
    # as-configured run cannot by itself exercise the guard. The forced run
    # drives _amd_smi_allowed() to False explicitly, which every new call site
    # must honour whatever the host looks like.
    import shutil

    _summary, spawns = _count_amd_smi_spawns(hw.get_gpu_summary)
    report["amd_smi_spawns_during_summary"] = spawns
    report["amd_smi_on_path"] = shutil.which("amd-smi") or shutil.which("amd-smi.exe")

    prior = os.environ.get("UNSLOTH_ENABLE_AMD_SMI")
    os.environ["UNSLOTH_ENABLE_AMD_SMI"] = "0"
    try:
        _s2, spawns_refused = _count_amd_smi_spawns(hw.get_gpu_summary)
    finally:
        if prior is None:
            os.environ.pop("UNSLOTH_ENABLE_AMD_SMI", None)
        else:
            os.environ["UNSLOTH_ENABLE_AMD_SMI"] = prior
    report["amd_smi_spawns_when_refused"] = spawns_refused

    # --- questions 1: does free move when someone ELSE holds memory? ---
    # Warm the primary context FIRST. _reading calls mem_get_info, which pins
    # roughly 600 MiB on first use; without this the observer's own context
    # lands between the two readings and is charged to the holder.
    try:
        torch.cuda.mem_get_info()
    except Exception as e:  # noqa: BLE001
        report["context_warmup_error"] = f"{type(e).__name__}: {e}"
    report["before"] = _reading(hw, torch)

    proc = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--holder",
            "--gib",
            str(args.gib),
            "--max-seconds",
            str(args.max_seconds),
        ],
        stdout = subprocess.PIPE,
        stderr = subprocess.DEVNULL,
        text = True,
    )
    ready = _wait_for_ready(proc, timeout = 180)
    report["holder"] = ready
    if ready:
        report["with_holder"] = _reading(hw, torch)
    # Stop by PID. Never pkill by pattern: it matches the shell running it.
    proc.terminate()
    try:
        proc.wait(timeout = 30)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(3)
    report["after"] = _reading(hw, torch)

    print(json.dumps(report, indent = 2))
    if args.json:
        args.json.write_text(json.dumps(report, indent = 2))

    # --- verdict ---
    print("\n" + "=" * 62)
    if not ready:
        print("INCONCLUSIVE: the holder never became ready, so nothing was")
        print("measured against it. A differential against a condition that was")
        print("never established is no result, not a failing one.")
        return 3

    held = float(ready.get("allocated_gib") or 0.0)
    before, during = report["before"], report["with_holder"]
    # The holder costs its tensor PLUS its own primary context, so the driver's
    # free legitimately drops by more than `held`. The question is binary --
    # did free track another process at all, or not move -- so this is a band,
    # not a point. Anything under `floor` means the other process was invisible.
    floor, ceiling = 0.9 * held, held + 2.0

    new_drop = (before["free_gb"] or 0) - (during["free_gb"] or 0)
    old_drop = before["old_formula_free_gb"] - during["old_formula_free_gb"]
    mgi_drop = None
    if "torch_mem_get_info_free_gb" in before and "torch_mem_get_info_free_gb" in during:
        mgi_drop = before["torch_mem_get_info_free_gb"] - during["torch_mem_get_info_free_gb"]

    print(f"holder held                  {held:.2f} GiB (plus its own context)")
    print(f"accepted band                {floor:.2f} to {ceiling:.2f} GiB")
    print(f"free_gb dropped by           {new_drop:+.2f} GiB   (want >= {floor:.2f})")
    print(f"old formula dropped by       {old_drop:+.2f} GiB   (want ~0.00, the defect)")
    if mgi_drop is not None:
        print(f"torch.mem_get_info dropped   {mgi_drop:+.2f} GiB   (attribution only)")
    print(f"observer stayed a bystander  {during['observer_allocated_gb']:.2f} GiB allocated")
    allowed = gates.get("amd_smi_allowed")
    print(f"amd-smi on PATH              {report['amd_smi_on_path'] or 'no'}")
    print(f"amd-smi allowed here         {allowed}")
    print(
        f"amd-smi spawns, as configured {len(spawns)}"
        f"{'  (allowed, so not a fault)' if allowed is True else ''}"
    )
    print(f"amd-smi spawns, when refused  {len(spawns_refused)}   (want 0)")

    problems = []
    # A ROCm build whose ROCm gates read False means detection did not take, and
    # every gate below describes a branch this machine never executed.
    if getattr(torch.version, "hip", None) and not gates.get("IS_ROCM"):
        problems.append(
            "torch is a ROCm build but hw.IS_ROCM is False, so the gate dump "
            "describes the non-ROCm path: do not trust it"
        )
    if during["observer_allocated_gb"] >= 0.5:
        problems.append("observer allocated memory itself, so this measures the wrong thing")
    if abs(old_drop) > 0.5:
        problems.append("the OLD formula moved, so the base does not show the defect here: VOID")
    if new_drop < floor:
        problems.append(
            f"free_gb moved only {new_drop:.2f} GiB against {held:.2f} GiB held: "
            f"the other process is largely invisible here"
        )
    elif new_drop > ceiling:
        problems.append(
            f"free_gb moved {new_drop:.2f} GiB, more than {held:.2f} GiB held plus "
            f"2.00 GiB of process overhead: free is being under-reported"
        )
    # A spawn is only a fault where the guard said no. With a HIP SDK present
    # amd-smi is allowed and calling it is the intended behaviour.
    if allowed is False and spawns:
        problems.append(
            f"amd-smi was spawned {len(spawns)}x despite _amd_smi_allowed() being "
            f"False, which is the UAC/DiskPart prompt path: {spawns}"
        )
    if spawns_refused:
        problems.append(
            f"amd-smi was spawned {len(spawns_refused)}x with UNSLOTH_ENABLE_AMD_SMI=0, "
            f"so a call site bypasses the guard: {spawns_refused}"
        )

    print("=" * 62)
    if problems:
        print("PROBLEMS FOUND")
        for p in problems:
            print("  - " + p)
        if mgi_drop is not None and abs(mgi_drop) < 0.5 and new_drop < floor:
            print("\n  Note: torch.cuda.mem_get_info did not move either, so the")
            print("  driver itself is reporting process-local free here (WDDM),")
            print("  rather than this being a bug in the PR's own logic.")
        return 1
    print(f"PASS: free_gb tracked another process's {held:.2f} GiB, the old formula")
    print("did not, and no call site spawned amd-smi once the guard refused.")
    # State the reach of the amd-smi result rather than letting PASS imply more.
    if platform.system() != "Windows":
        print("\n  Note: not Windows, so _amd_smi_allowed() returned True by")
        print("  platform and the elevation guard was never under test.")
    elif not report["amd_smi_on_path"]:
        print("\n  Note: amd-smi is not on PATH, so zero spawns is trivially true.")
        print("  The refusal path is confirmed, the real no-SDK case is not.")
    elif allowed is True:
        print("\n  Note: a HIP SDK is present, so amd-smi is legitimately allowed")
        print("  here. The forced-refusal leg passed, but the genuine")
        print("  Windows-without-a-HIP-SDK case needs a box without one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
