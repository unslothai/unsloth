#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Temporary hardware check for unslothai/unsloth#17120 (remove before merge).

Runs this PR's GPU metadata code (studio/backend/utils/hardware/nvidia.py: the CUDA driver API, read in a child
process) against torch.cuda on a real NVIDIA machine, under several CUDA_VISIBLE_DEVICES masks and device
orders, and checks that the checking process itself never loads the CUDA driver. Works on Windows, WSL and Linux
with any Python that has a CUDA build of torch (for example Unsloth Desktop's).

    python pr17120_gpu_metadata_check.py [path/to/nvidia.py]

Without a path it uses the nvidia.py of this checkout, else downloads it from the PR branch. Prints one
`RESULT {...}` line and `VERDICT PASS|FAIL`; paste both back on the PR.
"""

import ast
import json
import os
import platform
import subprocess
import sys
import time
import types
import typing
import urllib.request
from pathlib import Path

PR_BRANCH_URL = (
    "https://raw.githubusercontent.com/unslothai/unsloth/fix/studio-integrated-probe-no-cuda-context/"
    "studio/backend/utils/hardware/nvidia.py"
)
KEEP_ASSIGN = {"_CUDA_DEVICE_INFO_SCRIPT", "_CUDA_DEVICE_INFO_RETRY_S", "_CUDA_DEVICE_INFO_CACHE"}
KEEP_DEF = {
    "_normalize_bus_id",
    "get_cuda_device_info",
    "_query_cuda_device_info",
    "get_cuda_ordinal_smi_rows",
}

TORCH_CHILD = r"""
import json, torch
out = []
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    out.append([p.name, int(p.total_memory), bool(getattr(p, "is_integrated", False)),
                "%04x:%02x:%02x.0" % (p.pci_domain_id, p.pci_bus_id, p.pci_device_id)])
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "devices": out}))
"""


def nvidia_py_source(arg: typing.Optional[str]) -> tuple[str, str]:
    if arg:
        return Path(arg).read_text(encoding = "utf-8"), arg
    local = (
        Path(__file__).resolve().parent.parent
        / "studio"
        / "backend"
        / "utils"
        / "hardware"
        / "nvidia.py"
    )
    if local.is_file():
        return local.read_text(encoding = "utf-8"), str(local)
    with urllib.request.urlopen(PR_BRANCH_URL, timeout = 60) as r:
        return r.read().decode("utf-8"), PR_BRANCH_URL


def load(src: str) -> dict:
    """The PR's functions, run with stand-ins for the rest of the Studio backend (nothing else is imported)."""
    body = []
    for node in ast.parse(src).body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if isinstance(target, ast.Name) and target.id in KEEP_ASSIGN:
                body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in KEEP_DEF:
            body.append(node)
    found = {
        getattr(n, "name", None) or (n.targets[0] if isinstance(n, ast.Assign) else n.target).id
        for n in body
    }
    if (KEEP_ASSIGN | KEEP_DEF) - found:
        raise SystemExit(
            f"nvidia.py lacks {sorted((KEEP_ASSIGN | KEEP_DEF) - found)}: not this PR's version"
        )
    child_stdio = types.ModuleType("utils.child_stdio")
    child_stdio.utf8_child_env = lambda env: dict(env, PYTHONIOENCODING = "utf-8")
    sys.modules.setdefault("utils", types.ModuleType("utils"))
    sys.modules["utils.child_stdio"] = child_stdio
    ns = {
        "os": os,
        "subprocess": subprocess,
        "time": time,
        "sys": sys,
        "Optional": typing.Optional,
        "Any": typing.Any,
        "gpu_query": types.SimpleNamespace(
            run_nvidia_smi = lambda argv, **k: subprocess.run(argv, **k), static_generation = lambda: 0
        ),
        "child_env_without_native_path_secret": lambda: dict(os.environ),
        "_windows_hidden_subprocess_kwargs": lambda: {},
        "_nvidia_smi_executable": lambda: "nvidia-smi",
    }
    exec(compile(ast.Module(body = body, type_ignores = []), "nvidia_py_pr17120", "exec"), ns)
    return ns


def smi(*fields):
    r = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
        capture_output = True,
        text = True,
        timeout = 60,
    )
    return [[c.strip() for c in line.split(",")] for line in r.stdout.strip().splitlines()]


def driver_loaded_here() -> bool:
    """Did THIS process load the CUDA driver (what the PR must avoid in the Studio server)?"""
    if sys.platform == "win32":
        import ctypes
        return bool(ctypes.windll.kernel32.GetModuleHandleW("nvcuda.dll"))
    try:
        return "libcuda" in Path("/proc/self/maps").read_text()
    except OSError:
        return False


def is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def main() -> int:
    src, origin = nvidia_py_source(sys.argv[1] if len(sys.argv) > 1 else None)
    ns = load(src)
    gpus = smi("index", "name", "pci.bus_id", "memory.total", "driver_version")
    if not gpus:
        raise SystemExit("nvidia-smi lists no GPU: run this on the NVIDIA machine")
    rows_by_bus = {ns["_normalize_bus_id"](g[2]): int(g[0]) for g in gpus}
    print(f"nvidia.py: {origin}")
    print(
        f"host: {platform.platform()} wsl={is_wsl()} python={sys.version.split()[0]} driver={gpus[0][4]}"
    )
    for g in gpus:
        print(f"  GPU {g[0]}: {g[1]} {g[2]} {g[3]} MiB")
    cases, failures, torch_info = [], [], None
    for order in (None, "PCI_BUS_ID", "FASTEST_FIRST"):
        for mask in (None, "0", "1", "1,0", "", "-1", "7"):
            for k, v in (("CUDA_DEVICE_ORDER", order), ("CUDA_VISIBLE_DEVICES", mask)):
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            ns["_CUDA_DEVICE_INFO_CACHE"].clear()
            t = time.perf_counter()
            info = ns["get_cuda_device_info"]()
            query_ms = (time.perf_counter() - t) * 1e3
            t = time.perf_counter()
            again = ns["get_cuda_device_info"]()
            cached_ms = (time.perf_counter() - t) * 1e3
            rows = ns["get_cuda_ordinal_smi_rows"]()
            tc = subprocess.run(
                [sys.executable, "-c", TORCH_CHILD],
                capture_output = True,
                text = True,
                timeout = 300,
                env = dict(os.environ),
            )
            torch_out = (
                json.loads(tc.stdout.strip().splitlines()[-1])
                if tc.returncode == 0 and tc.stdout.strip()
                else None
            )
            torch_devs = None if torch_out is None else torch_out["devices"]
            torch_info = torch_info or (
                torch_out and {"torch": torch_out["torch"], "cuda": torch_out["cuda"]}
            )
            norm = ns["_normalize_bus_id"]
            want = (
                None
                if torch_devs is None
                else [(norm(bus), integ, total, name) for name, total, integ, bus in torch_devs]
            )
            got = (
                None
                if info is None
                else [None if d is None else (norm(d[0]), d[1], d[2], d[3]) for d in info]
            )
            want_rows = None
            if torch_devs:
                want_rows = (
                    []
                    if any(d[2] for d in torch_devs)
                    else [rows_by_bus.get(norm(d[3])) for d in torch_devs]
                )
            case = {
                "order": order,
                "mask": mask,
                "info_matches_torch": got == want,
                "rows": rows,
                "want_rows": want_rows,
                "rows_match": rows == want_rows,
                "cached": again == info,
                "query_ms": round(query_ms, 1),
                "cached_ms": round(cached_ms, 3),
                "n": None if info is None else len(info),
            }
            cases.append(case)
            ok = case["info_matches_torch"] and case["rows_match"] and case["cached"]
            print(
                f"  order={str(order):13s} mask={str(mask):5s} devices={str(case['n']):4s} "
                f"{'ok' if ok else 'MISMATCH'}  query {case['query_ms']} ms, cached {case['cached_ms']} ms"
            )
            if not ok:
                failures.append(case | {"got": got, "want": want, "torch_err": tc.stderr[-500:]})
    parent_clean = not driver_loaded_here() and "torch" not in sys.modules
    res = {
        "host": platform.platform(),
        "wsl": is_wsl(),
        "python": sys.version.split()[0],
        "torch": torch_info,
        "gpus": gpus,
        "cases": len(cases),
        "failures": failures,
        "parent_never_loaded_the_driver": parent_clean,
    }
    print("RESULT " + json.dumps(res))
    ok = not failures and parent_clean
    print("VERDICT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
