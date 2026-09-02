#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Simulation matrix for the single-root install.

Runs the REAL storage_roots resolver in a subprocess per case: the HF resolver
snapshots explicit env once per process, so an in-process reload would not model
a real launch. The GPU axis is there to PROVE the resolver is hardware
independent, not to assume it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2] / "studio" / "backend"

PROBE = r"""
import json, os, sys
sys.path.insert(0, os.environ["_BACKEND"])
from utils.paths import storage_roots as sr
out = {}
try:
    sr.setup_cache_env()
except Exception as e:
    out["setup_error"] = repr(e)
out["studio_root"] = str(sr.studio_root())
out["cache_root"] = str(sr.cache_root())
out["unsloth_home"] = str(sr.unsloth_home()) if sr.unsloth_home() else None
out["portable"] = sr.portable_mode()
out["projects"] = str(sr.project_workspaces_root())
out["documents"] = str(sr.documents_root())
for k in ("UNSLOTH_COMPILE_LOCATION", "UV_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
          "TRITON_HOME", "TRITON_CACHE_DIR", "MPLCONFIGDIR", "NUMBA_CACHE_DIR",
          "CUDA_CACHE_PATH", "TORCH_EXTENSIONS_DIR", "DATA_DESIGNER_HOME",
          "HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TORCH_HOME",
          "UNSLOTH_STUDIO_PROJECTS_HOME"):
    out[k] = os.environ.get(k)
print("__JSON__" + json.dumps(out))
"""


def run(env_extra: dict, home: Path) -> dict:
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "_BACKEND": str(BACKEND),
    }
    env.update({k: v for k, v in env_extra.items() if v is not None})
    proc = subprocess.run(
        [sys.executable, "-c", PROBE],
        env = env,
        capture_output = True,
        text = True,
        timeout = 180,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("__JSON__"):
            return json.loads(line[len("__JSON__") :])
    raise RuntimeError(
        f"probe failed rc={proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}"
    )


FAILS: list[str] = []


def check(
    label: str,
    cond: bool,
    detail: str = "",
) -> None:
    if cond:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}{(': ' + detail) if detail else ''}")
        FAILS.append(label)


def inside(child: str, parent: str) -> bool:
    try:
        Path(child).relative_to(Path(parent))
        return True
    except ValueError:
        return False


CONTAINED = (
    "UNSLOTH_COMPILE_LOCATION",
    "UV_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_HOME",
    "TRITON_CACHE_DIR",
    "MPLCONFIGDIR",
    "NUMBA_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "TORCH_EXTENSIONS_DIR",
    "DATA_DESIGNER_HOME",
    "HF_HUB_CACHE",
    "HF_DATASETS_CACHE",
    "TORCH_HOME",
    "UNSLOTH_STUDIO_PROJECTS_HOME",
)

ALWAYS = (
    "UNSLOTH_COMPILE_LOCATION",
    "UV_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_HOME",
    "TRITON_CACHE_DIR",
    "MPLCONFIGDIR",
    "NUMBA_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "TORCH_EXTENSIONS_DIR",
    "DATA_DESIGNER_HOME",
)


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    print(f"scratch: {tmp}\n")

    print("[1] platform x GPU matrix (resolver must be hardware independent)")
    baseline = None
    for plat in ("linux", "darwin", "win32", "wsl"):
        for gpu in ("nvidia", "amd", "cpu"):
            home = tmp / f"h_{plat}_{gpu}"
            home.mkdir(parents = True, exist_ok = True)
            root = tmp / f"r_{plat}_{gpu}"
            gpu_env = {
                "nvidia": {"CUDA_VISIBLE_DEVICES": "0", "CUDA_PATH": "/usr/local/cuda"},
                "amd": {
                    "HIP_VISIBLE_DEVICES": "0",
                    "ROCM_PATH": "/opt/rocm",
                    "HSA_OVERRIDE_GFX_VERSION": "11.0.0",
                },
                "cpu": {"CUDA_VISIBLE_DEVICES": ""},
            }[gpu]
            wsl_env = {"WSL_DISTRO_NAME": "Ubuntu", "WSLENV": ""} if plat == "wsl" else {}
            r = run({"UNSLOTH_HOME": str(root), **gpu_env, **wsl_env}, home)
            shape = {
                k: (
                    v.replace(str(root), "<ROOT>").replace(str(home), "<HOME>")
                    if isinstance(v, str)
                    else v
                )
                for k, v in r.items()
            }
            if baseline is None:
                baseline = shape
                print(f"  base  {plat}/{gpu} -> studio_root={shape['studio_root']}")
            else:
                check(
                    f"{plat}/{gpu} resolves identically to baseline",
                    shape == baseline,
                    json.dumps(
                        {k: (baseline.get(k), v) for k, v in shape.items() if baseline.get(k) != v}
                    )[:300],
                )

    print("\n[2] portable mode contains every cache variable")
    home = tmp / "h_contain"
    home.mkdir()
    root = tmp / "r_contain"
    r = run({"UNSLOTH_HOME": str(root)}, home)
    for key in CONTAINED:
        v = r.get(key)
        check(f"{key} inside the root", bool(v) and inside(v, str(root)), f"got {v}")
    check(
        "HF_HOME stays out of the root (owns the token)",
        bool(r["HF_HOME"]) and not inside(r["HF_HOME"], str(root)),
        f"got {r['HF_HOME']}",
    )
    check(
        "documents_root stays the user's own folder",
        not inside(r["documents"], str(root)),
        f"got {r['documents']}",
    )

    print("\n[3] default install is unchanged (backwards compatibility)")
    home = tmp / "h_default"
    home.mkdir()
    r = run({}, home)
    check(
        "studio_root is the legacy ~/.unsloth/studio",
        r["studio_root"] == str(home / ".unsloth" / "studio"),
        r["studio_root"],
    )
    check("portable mode is off", r["portable"] is False)
    check(
        "HF hub cache stays shared with other tools",
        r["HF_HUB_CACHE"] == str(home / ".cache" / "huggingface" / "hub"),
        r["HF_HUB_CACHE"],
    )
    check(
        "projects stay in ~/Documents",
        r["projects"] == str(home / "Documents" / "Unsloth Studio" / "Projects"),
        r["projects"],
    )
    check(
        "TORCH_HOME is not pinned on a default install",
        r["TORCH_HOME"] is None,
        str(r["TORCH_HOME"]),
    )
    for key in ALWAYS:
        v = r.get(key)
        check(
            f"{key} still pinned under the studio root",
            bool(v) and inside(v, str(home / ".unsloth" / "studio")),
            f"got {v}",
        )

    print("\n[4] existing installs keep resolving where they already are")
    home = tmp / "h_legacy"
    (home / ".unsloth" / "studio").mkdir(parents = True)
    (home / ".unsloth" / "studio" / "studio.db").write_text("x")
    r = run({}, home)
    check(
        "legacy tree untouched",
        r["studio_root"] == str(home / ".unsloth" / "studio"),
        r["studio_root"],
    )

    home = tmp / "h_custom"
    home.mkdir()
    custom = tmp / "custom studio"
    custom.mkdir()
    r = run({"UNSLOTH_STUDIO_HOME": str(custom)}, home)
    check(
        "UNSLOTH_STUDIO_HOME install unchanged", r["studio_root"] == str(custom), r["studio_root"]
    )
    check("and is NOT silently made portable", r["portable"] is False)
    check(
        "and keeps the shared HF cache",
        r["HF_HUB_CACHE"] == str(home / ".cache" / "huggingface" / "hub"),
        r["HF_HUB_CACHE"],
    )

    print("\n[5] portable shapes: nested, flat, marker-only")
    home = tmp / "h_nested"
    home.mkdir()
    root = tmp / "r_nested"
    (root / "studio").mkdir(parents = True)
    r = run({"UNSLOTH_HOME": str(root)}, home)
    check(
        "nested: studio is <root>/studio",
        r["studio_root"] == str(root / "studio"),
        r["studio_root"],
    )

    home = tmp / "h_flat"
    home.mkdir()
    flat = tmp / "r_flat"
    (flat / "unsloth_studio").mkdir(parents = True)
    r = run({"UNSLOTH_HOME": str(flat)}, home)
    check(
        "flat: root holding a venv IS the studio root",
        r["studio_root"] == str(flat),
        r["studio_root"],
    )

    home = tmp / "h_marker"
    home.mkdir()
    mroot = tmp / "r_marker"
    (mroot / "studio").mkdir(parents = True)
    (mroot / ".unsloth-portable-root").write_text(str(mroot))
    r = run({"UNSLOTH_STUDIO_HOME": str(mroot / "studio")}, home)
    check("marker alone enables portable mode with no env", r["portable"] is True)
    check(
        "marker alone finds the master root",
        r["unsloth_home"] == str(mroot),
        str(r["unsloth_home"]),
    )

    print("\n[6] precedence: explicit env always wins")
    home = tmp / "h_prec"
    home.mkdir()
    root = tmp / "r_prec"
    chosen = tmp / "chosen hub"
    r = run({"UNSLOTH_HOME": str(root), "HF_HUB_CACHE": str(chosen)}, home)
    check(
        "explicit HF_HUB_CACHE beats portable", r["HF_HUB_CACHE"] == str(chosen), r["HF_HUB_CACHE"]
    )
    r = run({"UNSLOTH_HOME": str(root), "TORCHINDUCTOR_CACHE_DIR": str(chosen)}, home)
    check(
        "explicit TORCHINDUCTOR_CACHE_DIR beats portable",
        r["TORCHINDUCTOR_CACHE_DIR"] == str(chosen),
        r["TORCHINDUCTOR_CACHE_DIR"],
    )
    r = run({"UNSLOTH_HOME": str(root), "UNSLOTH_STUDIO_PROJECTS_HOME": str(chosen)}, home)
    check("explicit projects home beats portable", r["projects"] == str(chosen), r["projects"])
    r = run({"UNSLOTH_HOME": str(root), "MPLCONFIGDIR": "   "}, home)
    check("blank counts as unset", inside(r["MPLCONFIGDIR"], str(root)), str(r["MPLCONFIGDIR"]))

    print("\n[7] hostile install roots")
    cases = {
        "spaces": "port able root",
        "apostrophe": "o'brien root",
        "unicode": "ünïcodé_root",
        "dots": "root.with.dots",
        "dollar": "root$var",
        "hash": "root#hash",
        "plus": "root+plus",
        "deep": "a/b/c/d/e/f/g",
    }
    for label, name in cases.items():
        home = tmp / f"h_{label}"
        home.mkdir()
        root = tmp / "roots" / name
        try:
            r = run({"UNSLOTH_HOME": str(root)}, home)
        except Exception as exc:  # noqa: BLE001 - the point is to report, not raise
            check(f"root with {label}", False, repr(exc)[:200])
            continue
        ok = all(inside(r.get(k) or "", str(root)) for k in CONTAINED)
        check(
            f"root with {label} contains everything",
            ok,
            json.dumps({k: r.get(k) for k in CONTAINED if not inside(r.get(k) or "", str(root))})[
                :200
            ],
        )

    home = tmp / "h_slash"
    home.mkdir()
    root = tmp / "r_slash"
    root.mkdir()
    r = run({"UNSLOTH_HOME": str(root) + "/"}, home)
    check("trailing slash normalises", r["studio_root"] == str(root / "studio"), r["studio_root"])

    home = tmp / "h_tilde"
    (home / "tilde_root").mkdir(parents = True)
    r = run({"UNSLOTH_HOME": "~/tilde_root"}, home)
    check("~ expands", r["studio_root"] == str(home / "tilde_root" / "studio"), r["studio_root"])

    home = tmp / "h_link"
    home.mkdir()
    real = tmp / "real_root"
    real.mkdir()
    link = tmp / "link_root"
    link.symlink_to(real)
    r = run({"UNSLOTH_HOME": str(link)}, home)
    check(
        "symlinked root resolves",
        inside(r["studio_root"], str(real)) or inside(r["studio_root"], str(link)),
        r["studio_root"],
    )

    print("\n[8] degenerate environments")
    home = tmp / "h_blank"
    home.mkdir()
    r = run({"UNSLOTH_HOME": "   "}, home)
    check(
        "blank UNSLOTH_HOME is treated as unset",
        r["studio_root"] == str(home / ".unsloth" / "studio") and r["portable"] is False,
        r["studio_root"],
    )
    r = run({"UNSLOTH_PORTABLE": "0"}, home)
    check("UNSLOTH_PORTABLE=0 is off", r["portable"] is False)
    r = run({"UNSLOTH_PORTABLE": "false"}, home)
    check("UNSLOTH_PORTABLE=false is off", r["portable"] is False)
    r = run({"UNSLOTH_PORTABLE": "1"}, home)
    check("UNSLOTH_PORTABLE=1 is on", r["portable"] is True)

    ro = tmp / "ro_root"
    ro.mkdir()
    os.chmod(ro, 0o500)
    try:
        home = tmp / "h_ro"
        home.mkdir()
        r = run({"UNSLOTH_HOME": str(ro)}, home)
        check(
            "read-only root does not crash the resolver",
            "setup_error" not in r,
            str(r.get("setup_error"))[:200],
        )
    finally:
        os.chmod(ro, 0o700)

    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) failed:")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("ALL SIMULATION CHECKS PASSED")
    return 0


def test_portable_root_matrix():
    """Run the whole matrix; any failed check fails the test with the list."""
    assert main() == 0, "simulation checks failed: " + ", ".join(FAILS)


if __name__ == "__main__":
    raise SystemExit(main())
