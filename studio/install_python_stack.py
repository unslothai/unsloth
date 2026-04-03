#!/usr/bin/env python3

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cross-platform Python dependency installer for Unsloth Studio.

Called by both setup.sh (Linux / WSL) and setup.ps1 (Windows) after the
virtual environment is already activated.  Expects `pip` and `python` on
PATH to point at the venv.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
IS_MAC_INTEL = IS_MACOS and platform.machine() == "x86_64"

# ── ROCm / AMD GPU support ─────────────────────────────────────────────────────
# Mapping from detected ROCm (major, minor) to the best PyTorch wheel tag on
# download.pytorch.org.  Entries are checked newest-first (>=).
# ROCm 7.2 only has torch 2.11.0 on download.pytorch.org, which exceeds the
# current torch upper bound (<2.11.0).  Fall back to rocm7.1 (torch 2.10.0).
# TODO: uncomment rocm7.2 when torch upper bound is bumped to >=2.11.0
_ROCM_TORCH_INDEX: dict[tuple[int, int], str] = {
    # (7, 2): "rocm7.2",  # torch 2.11.0 -- requires torch>=2.11
    (7, 1): "rocm7.1",
    (7, 0): "rocm7.0",
    (6, 4): "rocm6.4",
    (6, 3): "rocm6.3",
    (6, 2): "rocm6.2",
    (6, 1): "rocm6.1",
    (6, 0): "rocm6.0",
}
_PYTORCH_WHL_BASE = "https://download.pytorch.org/whl"


def _detect_rocm_version() -> tuple[int, int] | None:
    """Return (major, minor) of the installed ROCm stack, or None."""
    # Check /opt/rocm/.info/version or ROCM_PATH equivalent
    rocm_root = os.environ.get("ROCM_PATH") or "/opt/rocm"
    for path in (
        os.path.join(rocm_root, ".info", "version"),
        os.path.join(rocm_root, "lib", "rocm_version"),
    ):
        try:
            with open(path) as fh:
                parts = fh.read().strip().split("-")[0].split(".")
            return int(parts[0]), int(parts[1])
        except Exception:
            pass

    # Try amd-smi version (outputs "... | ROCm version: X.Y.Z")
    amd_smi = shutil.which("amd-smi")
    if amd_smi:
        try:
            result = subprocess.run(
                [amd_smi, "version"],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text = True,
                timeout = 5,
            )
            if result.returncode == 0:
                import re

                m = re.search(r"ROCm version:\s*(\d+)\.(\d+)", result.stdout)
                if m:
                    return int(m.group(1)), int(m.group(2))
        except Exception:
            pass

    # Try hipconfig --version (outputs bare version like "6.3.21234.2")
    hipconfig = shutil.which("hipconfig")
    if hipconfig:
        try:
            result = subprocess.run(
                [hipconfig, "--version"],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                timeout = 5,
            )
            if result.returncode == 0:
                raw = result.stdout.decode().strip().split("\n")[0]
                parts = raw.split(".")
                if (
                    len(parts) >= 2
                    and parts[0].isdigit()
                    and parts[1].split("-")[0].isdigit()
                ):
                    return int(parts[0]), int(parts[1].split("-")[0])
        except Exception:
            pass

    return None


def _has_rocm_gpu() -> bool:
    """Return True only if an actual AMD GPU is visible (not just ROCm tools installed)."""
    import re

    for cmd, check_fn in (
        # rocminfo: look for "Name: gfxNNNN" indicating an actual GPU agent
        (["rocminfo"], lambda out: "gfx" in out.lower()),
        # amd-smi list: require "GPU: <number>" data rows, not just a header
        (["amd-smi", "list"], lambda out: bool(re.search(r"(?im)^gpu\s*:\s*\d", out))),
    ):
        exe = shutil.which(cmd[0])
        if not exe:
            continue
        try:
            result = subprocess.run(
                [exe, *cmd[1:]],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text = True,
                timeout = 10,
            )
        except Exception:
            continue
        if result.returncode == 0 and result.stdout.strip():
            if check_fn(result.stdout):
                return True
    return False


def _has_usable_nvidia_gpu() -> bool:
    """Return True only when nvidia-smi exists AND reports at least one GPU."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        result = subprocess.run(
            [exe, "-L"],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            timeout = 10,
        )
    except Exception:
        return False
    return result.returncode == 0 and "GPU " in result.stdout


def _ensure_rocm_torch() -> None:
    """Reinstall torch with ROCm wheels when the venv received CPU-only torch.

    Runs only on Linux hosts where ROCm is installed and an AMD GPU is
    present.  No-op when torch already links against HIP (ROCm) or CUDA
    (NVIDIA).  Skips on Windows/macOS and on mixed AMD+NVIDIA hosts
    (NVIDIA takes precedence).
    Uses pip_install() to respect uv, constraints, and --python targeting.
    """
    # NVIDIA takes precedence on mixed hosts -- but only if an actual GPU is usable
    if _has_usable_nvidia_gpu():
        return
    rocm_root = os.environ.get("ROCM_PATH") or "/opt/rocm"
    if not os.path.isdir(rocm_root) and not shutil.which("hipcc"):
        return  # no ROCm toolchain
    if not _has_rocm_gpu():
        return  # ROCm tools present but no AMD GPU

    ver = _detect_rocm_version()
    if ver is None:
        print("   ROCm detected but version unreadable -- skipping torch reinstall")
        return

    # Skip if torch already links against HIP (ROCm is already working).
    # Do NOT skip for CUDA-only builds since they are unusable on AMD-only hosts
    # (the NVIDIA check above already handled mixed AMD+NVIDIA setups).
    try:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch; print(getattr(torch.version,'hip','') or '')",
            ],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            timeout = 30,
        )
    except (OSError, subprocess.TimeoutExpired):
        probe = None
    if probe is not None and probe.returncode == 0 and probe.stdout.decode().strip():
        return  # torch already has HIP/ROCm backend

    # Select best matching wheel tag (newest ROCm version <= installed)
    tag = next(
        (
            t
            for (maj, mn), t in sorted(_ROCM_TORCH_INDEX.items(), reverse = True)
            if ver >= (maj, mn)
        ),
        None,
    )
    if tag is None:
        print(f"   No PyTorch wheel for ROCm {ver[0]}.{ver[1]} -- skipping")
        return

    index_url = f"{_PYTORCH_WHL_BASE}/{tag}"
    print(f"   ROCm {ver[0]}.{ver[1]} -- installing torch from {index_url}")
    pip_install(
        f"ROCm torch ({tag})",
        "--force-reinstall",
        "--no-cache-dir",
        "torch>=2.4,<2.11.0",
        "torchvision<0.26.0",
        "torchaudio<2.11.0",
        "--index-url",
        index_url,
        constrain = False,
    )
    # Also install bitsandbytes for AMD
    pip_install(
        "bitsandbytes (AMD)",
        "--no-cache-dir",
        "bitsandbytes>=0.49.1",
        constrain = False,
    )


def _infer_no_torch() -> bool:
    """Determine whether to run in no-torch (GGUF-only) mode.

    Checks UNSLOTH_NO_TORCH env var first.  When unset, falls back to
    platform detection so that Intel Macs automatically use GGUF-only
    mode even when invoked from ``unsloth studio update`` (which does
    not inject the env var).
    """
    env = os.environ.get("UNSLOTH_NO_TORCH")
    if env is not None:
        return env.strip().lower() in ("1", "true")
    return IS_MAC_INTEL


NO_TORCH = _infer_no_torch()

# -- Verbosity control ----------------------------------------------------------
# By default the installer shows a minimal progress bar (one line, in-place).
# Set UNSLOTH_VERBOSE=1 in the environment to restore full per-step output:
#   CLI:        unsloth studio setup --verbose
#   Linux/Mac:  UNSLOTH_VERBOSE=1 ./studio/setup.sh
#   Windows:    $env:UNSLOTH_VERBOSE="1" ; .\studio\setup.ps1
VERBOSE: bool = os.environ.get("UNSLOTH_VERBOSE", "0") == "1"

# Progress bar state -- updated by _progress() as each install step runs.
# _TOTAL counts: pip-upgrade + 7 shared steps + triton (non-Windows) + local-plugin + finalize
# Update _TOTAL here if you add or remove install steps in install_python_stack().
_STEP: int = 0
_TOTAL: int = 0  # set at runtime in install_python_stack() based on platform

# -- Paths --------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REQ_ROOT = SCRIPT_DIR / "backend" / "requirements"
SINGLE_ENV = REQ_ROOT / "single-env"
CONSTRAINTS = SINGLE_ENV / "constraints.txt"
LOCAL_DD_UNSTRUCTURED_PLUGIN = (
    SCRIPT_DIR / "backend" / "plugins" / "data-designer-unstructured-seed"
)

# -- Unicode-safe printing ---------------------------------------------
# On Windows the default console encoding can be a legacy code page
# (e.g. CP1252) that cannot represent Unicode glyphs such as ✅ or ❌.
# _safe_print() gracefully degrades to ASCII equivalents so the
# installer never crashes just because of a status glyph.

_UNICODE_TO_ASCII: dict[str, str] = {
    "\u2705": "[OK]",  # ✅
    "\u274c": "[FAIL]",  # ❌
    "\u26a0\ufe0f": "[!]",  # ⚠️  (warning + variation selector)
    "\u26a0": "[!]",  # ⚠  (warning without variation selector)
}


def _safe_print(*args: object, **kwargs: object) -> None:
    """Drop-in print() replacement that survives non-UTF-8 consoles."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # Stringify, then swap emoji for ASCII equivalents
        text = " ".join(str(a) for a in args)
        for uni, ascii_alt in _UNICODE_TO_ASCII.items():
            text = text.replace(uni, ascii_alt)
        # Final fallback: replace any remaining unencodable chars
        print(
            text.encode(sys.stdout.encoding or "ascii", errors = "replace").decode(
                sys.stdout.encoding or "ascii", errors = "replace"
            ),
            **kwargs,
        )


# ── Color support ──────────────────────────────────────────────────────
# Same logic as startup_banner: NO_COLOR disables, FORCE_COLOR or TTY enables.


def _stdout_supports_color() -> bool:
    """True if we should emit ANSI colors (matches startup_banner)."""
    if os.environ.get("NO_COLOR", "").strip():
        return False
    if os.environ.get("FORCE_COLOR", "").strip():
        return True
    try:
        if not sys.stdout.isatty():
            return False
    except (AttributeError, OSError, ValueError):
        return False
    if IS_WINDOWS:
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_ulong()
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except (ImportError, AttributeError, OSError):
            return False
    return True


_HAS_COLOR = _stdout_supports_color()


# Column layout — matches setup.sh step() helper:
#   2-space indent, 15-char label (dim), then value.
_LABEL = "deps"
_COL = 15


def _green(msg: str) -> str:
    return f"\033[38;5;108m{msg}\033[0m" if _HAS_COLOR else msg


def _cyan(msg: str) -> str:
    return f"\033[96m{msg}\033[0m" if _HAS_COLOR else msg


def _red(msg: str) -> str:
    return f"\033[91m{msg}\033[0m" if _HAS_COLOR else msg


def _dim(msg: str) -> str:
    return f"\033[38;5;245m{msg}\033[0m" if _HAS_COLOR else msg


def _title(msg: str) -> str:
    return f"\033[38;5;150m{msg}\033[0m" if _HAS_COLOR else msg


_RULE = "\u2500" * 52


def _step(label: str, value: str, color_fn = None) -> None:
    """Print a single step line in the column format."""
    if color_fn is None:
        color_fn = _green
    padded = label[:_COL]
    print(f"  {_dim(padded)}{' ' * (_COL - len(padded))}{color_fn(value)}")


def _progress(label: str) -> None:
    """Print an in-place progress bar aligned to the step column layout."""
    global _STEP
    _STEP += 1
    if VERBOSE:
        return
    width = 20
    filled = int(width * _STEP / _TOTAL)
    bar = "=" * filled + "-" * (width - filled)
    pad = " " * (_COL - len(_LABEL))
    end = "\n" if _STEP >= _TOTAL else ""
    sys.stdout.write(
        f"\r  {_dim(_LABEL)}{pad}[{bar}] {_STEP:2}/{_TOTAL}  {label:<20}{end}"
    )
    sys.stdout.flush()


def run(
    label: str, cmd: list[str], *, quiet: bool = True
) -> subprocess.CompletedProcess[bytes]:
    """Run a command; on failure print output and exit."""
    if VERBOSE:
        _step(_LABEL, f"{label}...", _dim)
    result = subprocess.run(
        cmd,
        stdout = subprocess.PIPE if quiet else None,
        stderr = subprocess.STDOUT if quiet else None,
    )
    if result.returncode != 0:
        _step("error", f"{label} failed (exit code {result.returncode})", _red)
        if result.stdout:
            print(result.stdout.decode(errors = "replace"))
        sys.exit(result.returncode)
    return result


# Packages to skip on Windows (require special build steps)
WINDOWS_SKIP_PACKAGES = {"open_spiel", "triton_kernels"}

# Packages to skip when torch is unavailable (Intel Mac GGUF-only mode).
# These packages either *are* torch extensions or have unconditional
# ``Requires-Dist: torch`` in their published metadata, so installing
# them would pull torch back into the environment.
NO_TORCH_SKIP_PACKAGES = {
    "torch-stoi",
    "timm",
    "torchcodec",
    "torch-c-dlpack-ext",
    "openai-whisper",
    "transformers-cfg",
}

# -- uv bootstrap ------------------------------------------------------

USE_UV = False  # Set by _bootstrap_uv() at the start of install_python_stack()
UV_NEEDS_SYSTEM = False  # Set by _bootstrap_uv() via probe


def _bootstrap_uv() -> bool:
    """Check if uv is available and probe whether --system is needed."""
    global UV_NEEDS_SYSTEM
    if not shutil.which("uv"):
        return False
    # Probe: try a dry-run install targeting the current Python explicitly.
    # Without --python, uv can ignore the activated venv on some platforms.
    probe = subprocess.run(
        ["uv", "pip", "install", "--dry-run", "--python", sys.executable, "pip"],
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
    )
    if probe.returncode != 0:
        # Retry with --system (some envs need it when uv can't find a venv)
        probe_sys = subprocess.run(
            ["uv", "pip", "install", "--dry-run", "--system", "pip"],
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
        )
        if probe_sys.returncode != 0:
            return False  # uv is broken, fall back to pip
        UV_NEEDS_SYSTEM = True
    return True


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


def _translate_pip_args_for_uv(args: tuple[str, ...]) -> list[str]:
    """Translate pip flags to their uv equivalents."""
    translated: list[str] = []
    for arg in args:
        if arg == "--no-cache-dir":
            continue  # uv cache is fast; drop this flag
        elif arg == "--force-reinstall":
            translated.append("--reinstall")
        else:
            translated.append(arg)
    return translated


def _build_pip_cmd(args: tuple[str, ...]) -> list[str]:
    """Build a standard pip install command.

    Strips uv-only flags like --upgrade-package that pip doesn't understand.
    """
    cmd = [sys.executable, "-m", "pip", "install"]
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--upgrade-package":
            skip_next = True  # skip the flag and its value
            continue
        cmd.append(arg)
    return cmd


def _build_uv_cmd(args: tuple[str, ...]) -> list[str]:
    """Build a uv pip install command with translated flags."""
    cmd = ["uv", "pip", "install"]
    if UV_NEEDS_SYSTEM:
        cmd.append("--system")
    # Always pass --python so uv targets the correct environment.
    # Without this, uv can ignore an activated venv and install into
    # the system Python (observed on Colab and similar environments).
    cmd.extend(["--python", sys.executable])
    cmd.extend(_translate_pip_args_for_uv(args))
    # Torch is pre-installed by install.sh/setup.ps1.  Do not add
    # --torch-backend by default -- it can cause solver dead-ends on
    # CPU-only machines.  Callers that need it can set UV_TORCH_BACKEND.
    _tb = os.environ.get("UV_TORCH_BACKEND", "")
    if _tb:
        cmd.append(f"--torch-backend={_tb}")
    return cmd


def pip_install(
    label: str,
    *args: str,
    req: Path | None = None,
    constrain: bool = True,
) -> None:
    """Build and run a pip install command (uses uv when available, falls back to pip)."""
    constraint_args: list[str] = []
    if constrain and CONSTRAINTS.is_file():
        constraint_args = ["-c", str(CONSTRAINTS)]

    actual_req = req
    temp_reqs: list[Path] = []
    if req is not None and IS_WINDOWS and WINDOWS_SKIP_PACKAGES:
        actual_req = _filter_requirements(req, WINDOWS_SKIP_PACKAGES)
        temp_reqs.append(actual_req)
    if actual_req is not None and NO_TORCH and NO_TORCH_SKIP_PACKAGES:
        actual_req = _filter_requirements(actual_req, NO_TORCH_SKIP_PACKAGES)
        temp_reqs.append(actual_req)
    req_args: list[str] = []
    if actual_req is not None:
        req_args = ["-r", str(actual_req)]

    try:
        if USE_UV:
            uv_cmd = _build_uv_cmd(args) + constraint_args + req_args
            if VERBOSE:
                print(f"   {label}...")
            result = subprocess.run(
                uv_cmd,
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
            )
            if result.returncode == 0:
                return
            print(_red(f"   uv failed, falling back to pip..."))
            if result.stdout:
                print(result.stdout.decode(errors = "replace"))

        pip_cmd = _build_pip_cmd(args) + constraint_args + req_args
        run(f"{label} (pip)" if USE_UV else label, pip_cmd)
    finally:
        for temp_req in temp_reqs:
            temp_req.unlink(missing_ok = True)


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
        _step(_LABEL, f"package {package_name} not found, skipping patch", _red)
        return

    location = None
    for line in result.stdout.splitlines():
        if line.lower().startswith("location:"):
            location = line.split(":", 1)[1].strip()
            break

    if not location:
        _step(_LABEL, f"could not locate {package_name}", _red)
        return

    dest = Path(location) / relative_path
    _step(_LABEL, f"patching {dest.name} in {package_name}...", _dim)
    download_file(url, dest)


# -- Main install sequence ---------------------------------------------


def install_python_stack() -> int:
    global USE_UV, _STEP, _TOTAL
    _STEP = 0

    # When called from install.sh (which already installed unsloth into the venv),
    # SKIP_STUDIO_BASE=1 is set to avoid redundant reinstallation of base packages.
    # When called from "unsloth studio update", it is NOT set so base packages
    # (unsloth + unsloth-zoo) are always reinstalled to pick up new versions.
    skip_base = os.environ.get("SKIP_STUDIO_BASE", "0") == "1"
    # When --package is used, install a different package name (e.g. roland-sloth for testing)
    package_name = os.environ.get("STUDIO_PACKAGE_NAME", "unsloth")
    # When --local is used, overlay a local repo checkout after updating deps
    local_repo = os.environ.get("STUDIO_LOCAL_REPO", "")
    base_total = 10 if IS_WINDOWS else 11
    if IS_MACOS:
        base_total -= 1  # triton step is skipped on macOS
    # ROCm torch check step (Linux only, non-macOS, non-no-torch)
    if not IS_WINDOWS and not IS_MACOS and not NO_TORCH:
        base_total += 1
    _TOTAL = (base_total - 1) if skip_base else base_total

    # 1. Try to use uv for faster installs (must happen before pip upgrade
    #    because uv venvs don't include pip by default)
    USE_UV = _bootstrap_uv()

    # 2. Ensure pip is available (uv venvs created by install.sh don't include pip)
    _progress("pip bootstrap")
    if USE_UV:
        run(
            "Bootstrapping pip via uv",
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "pip",
            ],
        )
    else:
        # pip may not exist yet (uv-created venvs omit it). Try ensurepip
        # first, then upgrade. Only fall back to a direct upgrade when pip
        # is already present.
        _has_pip = (
            subprocess.run(
                [sys.executable, "-m", "pip", "--version"],
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
            ).returncode
            == 0
        )

        if not _has_pip:
            run(
                "Bootstrapping pip via ensurepip",
                [sys.executable, "-m", "ensurepip", "--upgrade"],
            )
        else:
            run(
                "Upgrading pip",
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            )

    # 3. Core packages: unsloth-zoo + unsloth (or custom package name)
    if skip_base:
        pass
    elif NO_TORCH:
        # No-torch update path: install unsloth + unsloth-zoo with --no-deps
        # (current PyPI metadata still declares torch as a hard dep), then
        # runtime deps with --no-deps (avoids transitive torch).
        _progress("base packages (no torch)")
        pip_install(
            f"Updating {package_name} + unsloth-zoo (no-torch mode)",
            "--no-cache-dir",
            "--no-deps",
            "--upgrade-package",
            package_name,
            "--upgrade-package",
            "unsloth-zoo",
            package_name,
            "unsloth-zoo",
        )
        pip_install(
            "Installing no-torch runtime deps",
            "--no-cache-dir",
            "--no-deps",
            req = REQ_ROOT / "no-torch-runtime.txt",
        )
        if local_repo:
            pip_install(
                "Overlaying local repo (editable)",
                "--no-cache-dir",
                "--no-deps",
                "-e",
                local_repo,
                constrain = False,
            )
    elif local_repo:
        # Local dev install: update deps from base.txt, then overlay the
        # local checkout as an editable install (--no-deps so torch is
        # never re-resolved).
        _progress("base packages")
        pip_install(
            "Updating base packages",
            "--no-cache-dir",
            "--upgrade-package",
            "unsloth",
            "--upgrade-package",
            "unsloth-zoo",
            req = REQ_ROOT / "base.txt",
        )
        pip_install(
            "Overlaying local repo (editable)",
            "--no-cache-dir",
            "--no-deps",
            "-e",
            local_repo,
            constrain = False,
        )
    elif package_name != "unsloth":
        # Custom package name (e.g. roland-sloth for testing) — install directly
        _progress("base packages")
        pip_install(
            f"Installing {package_name}",
            "--no-cache-dir",
            package_name,
        )
    else:
        # Update path: upgrade only unsloth + unsloth-zoo while preserving
        # existing torch/CUDA installations.  Torch is pre-installed by
        # install.sh / setup.ps1; --upgrade-package targets only base pkgs.
        _progress("base packages")
        pip_install(
            "Updating base packages",
            "--no-cache-dir",
            "--upgrade-package",
            "unsloth",
            "--upgrade-package",
            "unsloth-zoo",
            req = REQ_ROOT / "base.txt",
        )

    # 2b. AMD ROCm: reinstall torch with HIP wheels if the host has ROCm but the
    #     venv received CPU-only torch (common when pip resolves torch from PyPI).
    #     Must come immediately after base packages so torch is present for inspection.
    if not IS_WINDOWS and not IS_MACOS and not NO_TORCH:
        _progress("ROCm torch check")
        _ensure_rocm_torch()

    # Windows + AMD GPU: PyTorch does not publish ROCm wheels for Windows.
    # Detect and warn so users know manual steps are needed for GPU training.
    if IS_WINDOWS and not NO_TORCH and not _has_usable_nvidia_gpu():
        # Validate actual AMD GPU presence (not just tool existence)
        import re as _re_win

        def _win_amd_smi_has_gpu(stdout: str) -> bool:
            return bool(_re_win.search(r"(?im)^gpu\s*:\s*\d", stdout))

        _win_amd_gpu = False
        for _wcmd, _check_fn in (
            (["hipinfo"], lambda out: "gcnarchname" in out.lower()),
            (["amd-smi", "list"], _win_amd_smi_has_gpu),
        ):
            _wexe = shutil.which(_wcmd[0])
            if not _wexe:
                continue
            try:
                _wr = subprocess.run(
                    [_wexe, *_wcmd[1:]],
                    stdout = subprocess.PIPE,
                    stderr = subprocess.DEVNULL,
                    text = True,
                    timeout = 10,
                )
            except Exception:
                continue
            if _wr.returncode == 0 and _check_fn(_wr.stdout):
                _win_amd_gpu = True
                break
        if _win_amd_gpu:
            _safe_print(
                _dim("  Note:"),
                "AMD GPU detected on Windows. ROCm-enabled PyTorch must be",
            )
            _safe_print(
                " " * 8,
                "installed manually. See: https://docs.unsloth.ai/get-started/install-and-update/amd",
            )

    # 3. Extra dependencies
    _progress("unsloth extras")
    pip_install(
        "Installing additional unsloth dependencies",
        "--no-cache-dir",
        req = REQ_ROOT / "extras.txt",
    )

    # 3b. Extra dependencies (no-deps) -- audio model support etc.
    _progress("extra codecs")
    pip_install(
        "Installing extras (no-deps)",
        "--no-deps",
        "--no-cache-dir",
        req = REQ_ROOT / "extras-no-deps.txt",
    )

    # 4. Overrides (torchao, transformers) -- force-reinstall
    #    Skip entirely when torch is unavailable (e.g. Intel Mac GGUF-only mode)
    #    because overrides.txt contains torchao which requires torch.
    if NO_TORCH:
        _progress("dependency overrides (skipped, no torch)")
    else:
        _progress("dependency overrides")
        pip_install(
            "Installing dependency overrides",
            "--force-reinstall",
            "--no-cache-dir",
            req = REQ_ROOT / "overrides.txt",
        )

    # 5. Triton kernels (no-deps, from source)
    #    Skip on Windows (no support) and macOS (no support).
    if not IS_WINDOWS and not IS_MACOS:
        _progress("triton kernels")
        pip_install(
            "Installing triton kernels",
            "--no-deps",
            "--no-cache-dir",
            req = REQ_ROOT / "triton-kernels.txt",
            constrain = False,
        )

    # # 6. Patch: override llama_cpp.py with fix from unsloth-zoo  feature/llama-cpp-windows-support branch
    # patch_package_file(
    #     "unsloth-zoo",
    #     os.path.join("unsloth_zoo", "llama_cpp.py"),
    #     "https://raw.githubusercontent.com/unslothai/unsloth-zoo/refs/heads/main/unsloth_zoo/llama_cpp.py",
    # )

    # # 7a. Patch: override vision.py with fix from unsloth PR #4091
    # patch_package_file(
    #     "unsloth",
    #     os.path.join("unsloth", "models", "vision.py"),
    #     "https://raw.githubusercontent.com/unslothai/unsloth/80e0108a684c882965a02a8ed851e3473c1145ab/unsloth/models/vision.py",
    # )

    # # 7b. Patch : override save.py with fix from feature/llama-cpp-windows-support
    # patch_package_file(
    #     "unsloth",
    #     os.path.join("unsloth", "save.py"),
    #     "https://raw.githubusercontent.com/unslothai/unsloth/refs/heads/main/unsloth/save.py",
    # )

    # 8. Studio dependencies
    _progress("studio deps")
    pip_install(
        "Installing studio dependencies",
        "--no-cache-dir",
        req = REQ_ROOT / "studio.txt",
    )

    # 9. Data-designer dependencies
    _progress("data designer deps")
    pip_install(
        "Installing data-designer base dependencies",
        "--no-cache-dir",
        req = SINGLE_ENV / "data-designer-deps.txt",
    )

    # 10. Data-designer packages (no-deps to avoid conflicts)
    _progress("data designer")
    pip_install(
        "Installing data-designer",
        "--no-cache-dir",
        "--no-deps",
        req = SINGLE_ENV / "data-designer.txt",
    )

    # 11. Local Data Designer seed plugin
    if not LOCAL_DD_UNSTRUCTURED_PLUGIN.is_dir():
        _safe_print(
            _red(
                f"❌ Missing local plugin directory: {LOCAL_DD_UNSTRUCTURED_PLUGIN}",
            ),
        )
        return 1
    _progress("local plugin")
    pip_install(
        "Installing local data-designer unstructured plugin",
        "--no-cache-dir",
        "--no-deps",
        str(LOCAL_DD_UNSTRUCTURED_PLUGIN),
        constrain = False,
    )

    # 12. Patch metadata for single-env compatibility
    _progress("finalizing")
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

    _step(_LABEL, "installed")
    return 0


if __name__ == "__main__":
    sys.exit(install_python_stack())
