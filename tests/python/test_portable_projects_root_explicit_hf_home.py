#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""An explicit HF_HOME moves the Hugging Face caches and nothing else.

_portable_cache_defaults returns early when the user chose their own HF_HOME,
because hf_cache_settings already keeps the hub and xet caches under it and
pinning the derived datasets/assets caches here would split one deliberately
chosen cache across two volumes. That early return must drop ONLY the HF-derived
variables. UNSLOTH_STUDIO_PROJECTS_HOME is not one of them: it is where new
project files are written, and losing it sends them to
`~/Documents/Unsloth Studio/Projects`, outside the portable root, even though
only the Hugging Face cache was ever meant to escape containment.

TORCH_HOME and PIP_CACHE_DIR are checked alongside it as the variables that were
already on the right side of the early return, so a future edit cannot fix one by
breaking the others. PIP_CACHE_DIR is the odd one out in shape: it names
`<master>/cache/pip`, the path install.sh exports, which is a level ABOVE the
Studio root under the nested layout the fixture builds.

Subprocess per case: hf_cache_settings snapshots the explicit cache variables at
import time, and _setup_cache_env seeds os.environ once.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / "studio" / "backend"
MARKER = ".unsloth-portable-root"

PROBE = r"""
import json, os, sys
sys.path.insert(0, os.environ["_BACKEND"])
from utils.paths import storage_roots as sr

defaults = sr._portable_cache_defaults(sr.cache_root())
sr.setup_cache_env()   # what main.py and run.py actually call
print("__JSON__" + json.dumps({
    "portable": sr.portable_mode(),
    "user_set_hf_home": sr._user_set_hf_home(),
    "default_keys": sorted(defaults),
    "projects_root": str(sr.project_workspaces_root()),
    "torch_home": os.environ.get("TORCH_HOME"),
    "pip_cache": os.environ.get("PIP_CACHE_DIR"),
}))
"""


def _run(env_extra: dict, home: Path) -> dict:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "USERPROFILE": str(home),
        "_BACKEND": str(BACKEND),
    }
    env.update({k: v for k, v in env_extra.items() if v is not None})
    proc = subprocess.run(
        [sys.executable, "-c", PROBE], env = env, capture_output = True, text = True, timeout = 300
    )
    for line in proc.stdout.splitlines():
        if line.startswith("__JSON__"):
            return json.loads(line[len("__JSON__") :])
    raise RuntimeError(
        f"probe failed rc={proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}"
    )


FAILS: list[str] = []


def check(label: str, expected, actual) -> None:
    if expected == actual:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label} : expected [{expected}] got [{actual}]")
        FAILS.append(label)


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        home = tmp / "home"
        home.mkdir()
        master = tmp / "portable"
        (master / "studio" / "unsloth_studio").mkdir(parents = True)
        (master / MARKER).write_text(f"{master}\n")

        # root.parent of cache_root(); the spelling _portable_cache_defaults uses.
        projects = master / "studio" / "projects"
        torch_home = master / "studio" / "cache" / "torch"
        # The master root, not cache_root(): install.sh exports
        # PIP_CACHE_DIR="$UNSLOTH_ROOT/cache/pip" and the two must agree.
        pip_cache = master / "cache" / "pip"

        print("\n[1] control: no explicit HF_HOME")
        r = _run({"UNSLOTH_HOME": str(master)}, home)
        check("portable mode on", True, r["portable"])
        check("HF_HOME is ours", False, r["user_set_hf_home"])
        check("projects root inside the root", str(projects), r["projects_root"])
        check("torch home inside the root", str(torch_home), r["torch_home"])
        check("pip cache where install.sh puts it", str(pip_cache), r["pip_cache"])
        check(
            "every default is pinned",
            [
                "HF_ASSETS_CACHE",
                "HF_DATASETS_CACHE",
                "PIP_CACHE_DIR",
                "TORCH_HOME",
                "UNSLOTH_STUDIO_PROJECTS_HOME",
            ],
            r["default_keys"],
        )

        print("\n[2] an explicit HF_HOME drops the HF caches ONLY")
        r = _run({"UNSLOTH_HOME": str(master), "HF_HOME": str(tmp / "my-hf-cache")}, home)
        check("portable mode still on", True, r["portable"])
        check("HF_HOME is the user's", True, r["user_set_hf_home"])
        check("projects root still inside the root", str(projects), r["projects_root"])
        check("torch home still inside the root", str(torch_home), r["torch_home"])
        check("pip cache still where install.sh puts it", str(pip_cache), r["pip_cache"])
        check(
            "only the HF-derived caches are dropped",
            ["PIP_CACHE_DIR", "TORCH_HOME", "UNSLOTH_STUDIO_PROJECTS_HOME"],
            r["default_keys"],
        )

        print("\n[3] a non-portable install pins neither")
        r = _run({"HF_HOME": str(tmp / "my-hf-cache")}, home)
        check("portable mode off", False, r["portable"])
        check("nothing pinned", [], r["default_keys"])
        # ~/.cache/pip is shared with every other tool; a normal install keeps it.
        check("the shared pip cache is left alone", None, r["pip_cache"])
        check(
            "projects root is the user's Documents",
            str(home / "Documents" / "Unsloth Studio" / "Projects"),
            r["projects_root"],
        )

    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
        return 1
    print("All portable projects-root containment checks passed.")
    return 0


def test_explicit_hf_home_keeps_the_projects_root_contained():
    assert main() == 0, "checks failed: " + ", ".join(FAILS)


if __name__ == "__main__":
    raise SystemExit(main())
