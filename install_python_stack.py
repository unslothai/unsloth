#!/usr/bin/env python3
"""Cross-platform Python dependency installer for Unsloth Studio.

Called by both setup.sh (Linux / WSL) and setup.ps1 (Windows) after the
virtual environment is already activated.  Expects `pip` and `python` on
PATH to point at the venv.
"""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REQ_ROOT = SCRIPT_DIR / "studio" / "backend" / "requirements"
SINGLE_ENV = REQ_ROOT / "single-env"
CONSTRAINTS = SINGLE_ENV / "constraints.txt"

# ── Helpers ────────────────────────────────────────────────────────────

def _green(msg: str) -> str:
    return f"\033[92m{msg}\033[0m"

def _cyan(msg: str) -> str:
    return f"\033[96m{msg}\033[0m"

def _red(msg: str) -> str:
    return f"\033[91m{msg}\033[0m"


def run(label: str, cmd: list[str], *, quiet: bool = True) -> None:
    """Run a command; on failure print output and exit."""
    print(_cyan(f"   {label}..."))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE if quiet else None,
        stderr=subprocess.STDOUT if quiet else None,
    )
    if result.returncode != 0:
        print(_red(f"❌ {label} failed (exit code {result.returncode}):"))
        if result.stdout:
            print(result.stdout.decode(errors="replace"))
        sys.exit(result.returncode)


def pip_install(
    label: str,
    *args: str,
    req: Path | None = None,
    constrain: bool = True,
) -> None:
    """Build and run a pip install command."""
    cmd = [sys.executable, "-m", "pip", "install"]
    cmd.extend(args)
    if constrain and CONSTRAINTS.is_file():
        cmd.extend(["-c", str(CONSTRAINTS)])
    if req is not None:
        cmd.extend(["-r", str(req)])
    run(label, cmd)


def download_file(url: str, dest: Path) -> None:
    """Download a file using urllib (no curl dependency)."""
    urllib.request.urlretrieve(url, dest)


def patch_package_file(package_name: str, relative_path: str, url: str) -> None:
    """Download a file from url and overwrite a file inside an installed package."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", package_name],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(_red(f"   ⚠️  Could not find package {package_name}, skipping patch"))
        return

    location = None
    for line in result.stdout.splitlines():
        if line.lower().startswith("location:"):
            location = line.split(":", 1)[1].strip()
            break

    if not location:
        print(_red(f"   ⚠️  Could not determine location of {package_name}"))
        return

    dest = Path(location) / relative_path
    print(_cyan(f"   Patching {dest.name} in {package_name}..."))
    download_file(url, dest)


# ── Main install sequence ─────────────────────────────────────────────

def install_python_stack() -> int:
    print(_cyan("── Installing Python stack ──"))

    # 1. Upgrade pip
    run("Upgrading pip", [sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    # 2. Core packages: unsloth-zoo + unsloth
    pip_install(
        "Installing unsloth-zoo + unsloth",
        "--no-cache-dir",
        req=REQ_ROOT / "base.txt",
    )

    # 3. Extra dependencies
    pip_install(
        "Installing additional unsloth dependencies",
        "--no-cache-dir",
        req=REQ_ROOT / "extras.txt",
    )

    # 4. Overrides (torchao, transformers) — force-reinstall
    pip_install(
        "Installing torchao + transformers overrides",
        "--force-reinstall", "--no-cache-dir",
        req=REQ_ROOT / "overrides.txt",
    )

    # 5. Triton kernels (no-deps, from source)
    pip_install(
        "Installing triton kernels",
        "--no-deps", "--no-cache-dir",
        req=REQ_ROOT / "triton-kernels.txt",
        constrain=False,
    )

    # 6. Patch: override llama_cpp.py with fix from unsloth-zoo main branch
    patch_package_file(
        "unsloth-zoo",
        os.path.join("unsloth_zoo", "llama_cpp.py"),
        "https://raw.githubusercontent.com/unslothai/unsloth-zoo/refs/heads/main/unsloth_zoo/llama_cpp.py",
    )

    # 7. Patch: override vision.py with fix from unsloth PR #4091
    patch_package_file(
        "unsloth",
        os.path.join("unsloth", "models", "vision.py"),
        "https://raw.githubusercontent.com/unslothai/unsloth/80e0108a684c882965a02a8ed851e3473c1145ab/unsloth/models/vision.py",
    )

    # 8. Studio dependencies
    pip_install(
        "Installing studio dependencies",
        "--no-cache-dir",
        req=REQ_ROOT / "studio.txt",
    )

    # 9. Data-designer dependencies
    pip_install(
        "Installing data-designer dependencies",
        "--no-cache-dir",
        req=SINGLE_ENV / "data-designer-deps.txt",
    )

    # 10. Data-designer packages (no-deps to avoid conflicts)
    pip_install(
        "Installing data-designer",
        "--no-cache-dir", "--no-deps",
        req=SINGLE_ENV / "data-designer.txt",
    )

    # 11. Patch metadata for single-env compatibility
    run(
        "Patching single-env metadata",
        [sys.executable, str(SINGLE_ENV / "patch_metadata.py")],
    )

    # 12. Final check
    run("Running pip check", [sys.executable, "-m", "pip", "check"], quiet=False)

    print(_green("✅ Python dependencies installed"))
    return 0


if __name__ == "__main__":
    sys.exit(install_python_stack())
