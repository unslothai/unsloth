#!/usr/bin/env python3

# SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
# Copyright © 2025 Unsloth AI

"""Cross-platform Python dependency installer for Unsloth Studio.

Called by both setup.sh (Linux / WSL) and setup.ps1 (Windows) after the
virtual environment is already activated.  Expects `pip` and `python` on
PATH to point at the venv.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REQ_ROOT = SCRIPT_DIR / "backend" / "requirements"
SINGLE_ENV = REQ_ROOT / "single-env"
CONSTRAINTS = SINGLE_ENV / "constraints.txt"
LOCAL_DD_UNSTRUCTURED_PLUGIN = (
    SCRIPT_DIR / "backend" / "plugins" / "data-designer-unstructured-seed"
)

# ── Color support ──────────────────────────────────────────────────────


def _enable_colors() -> bool:
    """Try to enable ANSI color support. Returns True if available."""
    if not hasattr(sys.stdout, "fileno"):
        return False
    try:
        if not os.isatty(sys.stdout.fileno()):
            return False
    except Exception:
        return False
    if IS_WINDOWS:
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            # Enable ENABLE_VIRTUAL_TERMINAL_PROCESSING (0x0004) on stdout
            handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            mode = ctypes.c_ulong()
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
            return True
        except Exception:
            return False
    return True  # Unix terminals support ANSI by default


_HAS_COLOR = _enable_colors()


def _green(msg: str) -> str:
    return f"\033[92m{msg}\033[0m" if _HAS_COLOR else msg


def _cyan(msg: str) -> str:
    return f"\033[96m{msg}\033[0m" if _HAS_COLOR else msg


def _red(msg: str) -> str:
    return f"\033[91m{msg}\033[0m" if _HAS_COLOR else msg


def run(label: str, cmd: list[str], *, quiet: bool = True) -> None:
    """Run a command; on failure print output and exit."""
    print(_cyan(f"   {label}..."))
    result = subprocess.run(
        cmd,
        stdout = subprocess.PIPE if quiet else None,
        stderr = subprocess.STDOUT if quiet else None,
    )
    if result.returncode != 0:
        print(_red(f"❌ {label} failed (exit code {result.returncode}):"))
        if result.stdout:
            print(result.stdout.decode(errors = "replace"))
        sys.exit(result.returncode)


# Packages to skip on Windows (require special build steps)
WINDOWS_SKIP_PACKAGES = {"open_spiel"}


def _filter_requirements(req: Path, skip: set[str]) -> Path:
    """Return a temp copy of a requirements file with certain packages removed."""
    lines = req.read_text(encoding = "utf-8").splitlines(keepends = True)
    filtered = [
        line
        for line in lines
        if not any(line.strip().lower().startswith(pkg) for pkg in skip)
    ]
    tmp = tempfile.NamedTemporaryFile(
        mode = "w",
        suffix = ".txt",
        delete = False,
        encoding = "utf-8",
    )
    tmp.writelines(filtered)
    tmp.close()
    return Path(tmp.name)


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
    actual_req = req
    if req is not None and IS_WINDOWS and WINDOWS_SKIP_PACKAGES:
        actual_req = _filter_requirements(req, WINDOWS_SKIP_PACKAGES)
    if actual_req is not None:
        cmd.extend(["-r", str(actual_req)])
    try:
        run(label, cmd)
    finally:
        # Clean up temp file if we created one
        if actual_req is not None and actual_req != req:
            actual_req.unlink(missing_ok = True)


def download_file(url: str, dest: Path) -> None:
    """Download a file using urllib (no curl dependency)."""
    urllib.request.urlretrieve(url, dest)


def patch_package_file(package_name: str, relative_path: str, url: str) -> None:
    """Download a file from url and overwrite a file inside an installed package."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", package_name],
        capture_output = True,
        text = True,
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
        req = REQ_ROOT / "base.txt",
    )

    # 3. Extra dependencies
    pip_install(
        "Installing additional unsloth dependencies",
        "--no-cache-dir",
        req = REQ_ROOT / "extras.txt",
    )

    # 3b. Extra dependencies (no-deps) — audio model support etc.
    pip_install(
        "Installing extras (no-deps)",
        "--no-deps",
        "--no-cache-dir",
        req = REQ_ROOT / "extras-no-deps.txt",
    )

    # 4. Overrides (torchao, transformers) — force-reinstall
    pip_install(
        "Installing torchao + transformers overrides",
        "--force-reinstall",
        "--no-cache-dir",
        req = REQ_ROOT / "overrides.txt",
    )

    # 5. Triton kernels (no-deps, from source)
    pip_install(
        "Installing triton kernels",
        "--no-deps",
        "--no-cache-dir",
        req = REQ_ROOT / "triton-kernels.txt",
        constrain = False,
    )

    # 6. Patch: override llama_cpp.py with fix from unsloth-zoo  feature/llama-cpp-windows-support branch
    patch_package_file(
        "unsloth-zoo",
        os.path.join("unsloth_zoo", "llama_cpp.py"),
        "https://raw.githubusercontent.com/unslothai/unsloth-zoo/refs/heads/main/unsloth_zoo/llama_cpp.py",
    )

    # 7a. Patch: override vision.py with fix from unsloth PR #4091
    patch_package_file(
        "unsloth",
        os.path.join("unsloth", "models", "vision.py"),
        "https://raw.githubusercontent.com/unslothai/unsloth/80e0108a684c882965a02a8ed851e3473c1145ab/unsloth/models/vision.py",
    )

    # 7b. Patch : override save.py with fix from feature/llama-cpp-windows-support
    patch_package_file(
        "unsloth",
        os.path.join("unsloth", "save.py"),
        "https://raw.githubusercontent.com/unslothai/unsloth/refs/heads/main/unsloth/save.py",
    )

    # 8. Studio dependencies
    pip_install(
        "Installing studio dependencies",
        "--no-cache-dir",
        req = REQ_ROOT / "studio.txt",
    )

    # 9. Data-designer dependencies
    pip_install(
        "Installing data-designer dependencies",
        "--no-cache-dir",
        req = SINGLE_ENV / "data-designer-deps.txt",
    )

    # 10. Data-designer packages (no-deps to avoid conflicts)
    pip_install(
        "Installing data-designer",
        "--no-cache-dir",
        "--no-deps",
        req = SINGLE_ENV / "data-designer.txt",
    )

    # 11. Local Data Designer seed plugin
    if not LOCAL_DD_UNSTRUCTURED_PLUGIN.is_dir():
        print(
            _red(
                f"❌ Missing local plugin directory: {LOCAL_DD_UNSTRUCTURED_PLUGIN}",
            ),
        )
        return 1
    pip_install(
        "Installing local data-designer unstructured plugin",
        "--no-cache-dir",
        "--no-deps",
        "-e",
        str(LOCAL_DD_UNSTRUCTURED_PLUGIN),
        constrain = False,
    )

    # 12. Patch metadata for single-env compatibility
    run(
        "Patching single-env metadata",
        [sys.executable, str(SINGLE_ENV / "patch_metadata.py")],
    )

    # 13. Final check (silent; third-party conflicts are expected)
    subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL,
    )

    print(_green("✅ Python dependencies installed"))
    return 0


if __name__ == "__main__":
    sys.exit(install_python_stack())
