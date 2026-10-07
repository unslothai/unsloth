# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Kernel install pieces shared by Studio setup, the training worker, the SSM runtime and
`unsloth install-kernels [names]`.

Imports only stdlib, wheel_utils and child_stdio so setup and torch-less hosts can import it.
The CLI is wheel-only (never a source build) and uses --no-deps so torch is never touched.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable

from utils.child_stdio import utf8_child_env
from utils.wheel_utils import (
    CAUSAL_CONV1D_PACKAGE_VERSION,
    CAUSAL_CONV1D_RELEASE_BASE_URL,
    CAUSAL_CONV1D_RELEASE_TAG,
    MAMBA_SSM_PACKAGE_VERSION,
    MAMBA_SSM_RELEASE_BASE_URL,
    MAMBA_SSM_RELEASE_TAG,
    direct_wheel_url,
    flash_attn_wheel_url,
    install_wheel,
    probe_torch_wheel_env,
    redact_url_credentials,
    url_exists,
    xformers_wheel_url,
)


@dataclass(frozen = True)
class PinnedKernel:
    """A kernel installed from a pinned release: prebuilt wheel first, else `pypi_name==version`."""

    import_name: str
    display_name: str
    pypi_name: str
    package_version: str
    release_tag: str
    release_base_url: str

    def wheel_url(self, env: dict[str, str] | None) -> str | None:
        return direct_wheel_url(
            filename_prefix = self.import_name,
            package_version = self.package_version,
            release_tag = self.release_tag,
            release_base_url = self.release_base_url,
            env = env,
        )


CAUSAL_CONV1D = PinnedKernel(
    import_name = "causal_conv1d",
    display_name = "causal-conv1d",
    pypi_name = "causal-conv1d",
    package_version = CAUSAL_CONV1D_PACKAGE_VERSION,
    release_tag = CAUSAL_CONV1D_RELEASE_TAG,
    release_base_url = CAUSAL_CONV1D_RELEASE_BASE_URL,
)
MAMBA_SSM = PinnedKernel(
    import_name = "mamba_ssm",
    display_name = "mamba-ssm",
    pypi_name = "mamba-ssm",
    package_version = MAMBA_SSM_PACKAGE_VERSION,
    release_tag = MAMBA_SSM_RELEASE_TAG,
    release_base_url = MAMBA_SSM_RELEASE_BASE_URL,
)


def install_prebuilt(
    wheel_url: str,
    *,
    install: Callable[..., list[tuple[str, Any]]],
    verify: Callable[[], bool],
    on_failed: Callable[[str, Any], None],
    **install_kwargs: Any,
) -> str:
    """Install and verify a prebuilt wheel: "installed", "rejected" (installed, `verify()` failed)
    or "failed". `install_kwargs` are forwarded as given so each caller keeps its own installer flags."""
    for installer, result in install(wheel_url, python_executable = sys.executable, **install_kwargs):
        if getattr(result, "returncode", 1) == 0:
            # A wheel can install yet not load (CUDA/ABI or arch mismatch), so the exit code is no proof.
            return "installed" if verify() else "rejected"
        on_failed(installer, result)
    return "failed"


def hipcc_gcc_install_dir() -> str | None:
    """Highest-numbered ``/usr/lib/gcc/x86_64-linux-gnu/<N>`` that has BOTH the gcc runtime dir AND
    ``/usr/include/c++/<N>`` headers, or None. Ubuntu 24.04 ships gcc-14 runtime but not
    ``/usr/include/c++/14``; ROCm clang-20 picks the highest runtime dir, finds no ``<cstdlib>``,
    and the HIP build fails, hence ``--gcc-install-dir``. Mirrors studio/setup.sh (PR #5301)."""
    if not sys.platform.startswith("linux") or platform.machine().lower() != "x86_64":
        return None
    for ver in (14, 13, 12, 11):
        if os.path.isdir(f"/usr/lib/gcc/x86_64-linux-gnu/{ver}/include") and os.path.isdir(
            f"/usr/include/c++/{ver}"
        ):
            return f"/usr/lib/gcc/x86_64-linux-gnu/{ver}"
    return None


def source_build_command(spec: str, *, use_uv: bool, is_hip: bool, reinstall: bool) -> list[str]:
    """`--no-build-isolation --no-deps` install of *spec* against the resident torch; the cache is
    skipped on HIP (uv) or always (pip) so stale partial build artifacts are never reused."""
    if use_uv:
        cmd = [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-build-isolation",
            "--no-deps",
        ]
        if reinstall:
            cmd.append("--reinstall")
        if is_hip:
            cmd.append("--no-cache")
    else:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-deps",
            "--no-cache-dir",
        ]
        if reinstall:
            cmd.append("--force-reinstall")
    cmd.append(spec)
    return cmd


def source_build_run_kwargs(
    *, is_hip: bool, gcc_install_dir: Callable[[], str | None]
) -> tuple[dict[str, Any], str | None]:
    """subprocess.run kwargs for a source build, and the gcc dir added to HIPCC_COMPILE_FLAGS_APPEND
    (None when nothing was added). Non-HIP builds have no timeout so slow builds (causal-conv1d on
    aarch64) are not aborted; ROCm builds take 10-30 min, bounded at 1800 s."""
    kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        # pip and the compilers it drives write UTF-8 down this pipe; the Windows ANSI codepage would mojibake or raise over a fine install.
        "encoding": "utf-8",
        "errors": "replace",
        "env": utf8_child_env(),
    }
    if not is_hip:
        return kwargs, None
    kwargs["timeout"] = 1800
    existing = os.environ.get("HIPCC_COMPILE_FLAGS_APPEND", "")
    if "--gcc-install-dir" in existing:
        return kwargs, None
    gcc_dir = gcc_install_dir()
    if not gcc_dir:
        return kwargs, None
    env = dict(kwargs["env"])
    env["HIPCC_COMPILE_FLAGS_APPEND"] = f"{existing} --gcc-install-dir={gcc_dir}".strip()
    kwargs["env"] = env
    return kwargs, gcc_dir


def uninstall_command(
    distribution: str,
    *,
    use_uv: bool,
    uv_needs_system: bool = False,
) -> list[str]:
    """Uninstall *distribution* from ``sys.executable``. uv gets --python as well as --system:
    --system alone would remove from the system Python and leave the venv copy behind."""
    if use_uv:
        system = ["--system"] if uv_needs_system else []
        return ["uv", "pip", "uninstall", *system, "--python", sys.executable, distribution]
    return [sys.executable, "-m", "pip", "uninstall", "-y", distribution]


# `unsloth install-kernels` from here on.

# name -> (pip distribution, check that loads the compiled extension). The package imports are no
# proof: `import causal_conv1d` never loads it, and `import mamba_ssm` fails on einops under --no-deps.
# Install order: mamba_ssm's fast path imports causal_conv1d.
KERNELS = {
    "xformers": (
        "xformers",
        "from xformers._cpp_lib import _register_extensions; _register_extensions()",
    ),
    "flash_attn": ("flash-attn", "import torch, flash_attn_2_cuda"),
    "causal_conv1d": ("causal-conv1d", "import torch, causal_conv1d_cuda"),
    "mamba_ssm": ("mamba-ssm", "import torch, selective_scan_cuda"),
}

_PINNED = {"causal_conv1d": CAUSAL_CONV1D, "mamba_ssm": MAMBA_SSM}


def resolve_wheel_url(name: str, env: dict[str, str] | None) -> str | None:
    if env is None:
        return None
    if name == "xformers":
        return xformers_wheel_url(env)
    # flash-attn, causal-conv1d and mamba-ssm publish Linux wheels only.
    if not str(env.get("platform_tag") or "").startswith("linux"):
        return None
    if name == "flash_attn":
        return flash_attn_wheel_url(env)
    return _PINNED[name].wheel_url(env)


# FlashAttention 2 needs sm80. mamba_ssm's Triton kernels also run on sm75 with Triton 3.4+
# (torch 2.8+), the same rule unsloth_zoo's patch_mamba_ssm_pre_ampere_fallback applies, which
# turns the fast path off anywhere else, so the wheel would go unused there.
_NEEDS_SM80 = ("flash_attn", "mamba_ssm")
_MAMBA_SM75_MIN_TRITON = (3, 4)
_CAPABILITY: dict = {}


def _triton_version() -> tuple[int, int] | None:
    try:
        return tuple(int(part) for part in importlib.metadata.version("triton").split(".")[:2])
    except Exception:
        return None


def _pre_ampere_skip_reason(name: str, capability: tuple[int, int]) -> str | None:
    """Why a kernel stays off a pre-sm80 GPU, or None when unsloth_zoo uses it there."""
    if name != "mamba_ssm":
        return "needs sm80 or newer"
    forced = os.environ.get("UNSLOTH_MAMBA_PRE_AMPERE_FAST", "").strip()
    if forced in ("0", "1"):
        return None if forced == "1" else "UNSLOTH_MAMBA_PRE_AMPERE_FAST=0 is set"
    if capability < (7, 5):
        return "needs sm80, or sm75 with Triton 3.4+"
    triton = _triton_version()
    if triton is not None and triton >= _MAMBA_SM75_MIN_TRITON:
        return None
    found = "missing" if triton is None else "%d.%d" % triton
    return f"needs sm80, or sm75 with Triton 3.4+ (torch 2.8+), and Triton is {found}"


def _gpu_capability(run: Callable[..., subprocess.CompletedProcess]) -> tuple[int, int] | None:
    """The best compute capability across the visible GPUs, or None. Probed once per runner."""
    if run not in _CAPABILITY:
        _CAPABILITY[run] = _probe_gpu_capability(run)
    return _CAPABILITY[run]


def _probe_gpu_capability(
    run: Callable[..., subprocess.CompletedProcess],
) -> tuple[int, int] | None:
    check = (
        "import torch; n = torch.cuda.device_count() if torch.cuda.is_available() else 0; "
        "print(*max(torch.cuda.get_device_capability(i) for i in range(n))) if n else None"
    )
    try:
        result = run(
            [sys.executable, "-c", check],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 120,
        )
    except subprocess.TimeoutExpired:
        return None
    try:
        major, minor = (int(part) for part in result.stdout.split())
    except ValueError:
        return None
    return major, minor


def _loads(check: str, run: Callable[..., subprocess.CompletedProcess]) -> bool:
    result = run(
        [sys.executable, "-c", check], stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL
    )
    return result.returncode == 0


def _outside_venv() -> bool:
    # Colab and other system interpreters: uv refuses them without --system.
    return sys.prefix == sys.base_prefix


def _uninstall(distribution: str, run: Callable[..., subprocess.CompletedProcess]) -> bool:
    cmd = uninstall_command(
        distribution, use_uv = bool(shutil.which("uv")), uv_needs_system = _outside_venv()
    )
    return run(cmd, stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL).returncode == 0


def install_kernel(
    name: str,
    env: dict[str, str] | None,
    *,
    dry_run: bool = False,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    exists: Callable[[str], bool | None] = url_exists,
) -> int:
    distribution, check = KERNELS[name]
    url = resolve_wheel_url(name, env)
    if name in _NEEDS_SM80 and url is not None:
        capability = _gpu_capability(run)
        if capability is None:
            print(f"Unsloth: skipping {name}, which needs sm80 or newer (no CUDA GPU is visible).")
            return 0
        if capability < (8, 0):
            reason = _pre_ampere_skip_reason(name, capability)
            if reason is not None:
                print(
                    f"Unsloth: skipping {name}, which {reason} "
                    f"(the best GPU is sm{capability[0]}{capability[1]})."
                )
                return 0
    torch_desc = f"torch {env.get('torch_version')}" if env else "this environment"
    # Only a 404 proves nothing is published; an unreachable check falls through to the install.
    if url is None or exists(url) is False:
        print(f"Unsloth: no prebuilt {name} for {torch_desc}; using the torch fallback.")
        return 0
    # UNSLOTH_PYTORCH_MIRROR may carry credentials, and notebook output gets shared.
    shown = redact_url_credentials(url)
    if dry_run:
        print(shown)
        return 0
    if _loads(check, run):
        print(f"Unsloth: {name} already installed and loads.")
        return 0
    failures: list[subprocess.CompletedProcess] = []
    outcome = install_prebuilt(
        url,
        install = install_wheel,
        verify = lambda: _loads(check, run),
        on_failed = lambda _installer, result: failures.append(result),
        use_uv = True,
        uv_needs_system = _outside_venv(),
        reinstall = True,
        run = run,
    )
    if outcome == "failed":
        print(
            f"Unsloth: installing {name} from {shown} failed; using the torch fallback.\n"
            + failures[-1].stdout[-2000:].replace(url, shown)
        )
        return 1
    if outcome == "rejected":
        # A wheel that imports but whose extension cannot load would be picked up and crash later.
        removed = (
            "removed it" if _uninstall(distribution, run) else f"run `pip uninstall {distribution}`"
        )
        print(
            f"Unsloth: {name} from {shown} does not load with {torch_desc}; {removed}, using the torch fallback."
        )
        return 1
    print(f"Unsloth: installed {name} ({shown.rsplit('/', 1)[-1]}).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog = "unsloth install-kernels",
        description = "Install prebuilt kernel wheels matching the installed torch. Never builds from source or changes torch.",
    )
    parser.add_argument(
        "kernels",
        nargs = "*",
        choices = list(KERNELS),
        help = "kernels to install (default: all, each skipped when no wheel matches)",
    )
    parser.add_argument(
        "--dry-run", action = "store_true", help = "print the resolved wheel URL and install nothing"
    )
    args = parser.parse_args(argv)
    env = probe_torch_wheel_env(timeout = 120, include_windows = True)
    status = 0
    for name in [name for name in KERNELS if not args.kernels or name in args.kernels]:
        status = max(status, install_kernel(name, env, dry_run = args.dry_run))
    return status


if __name__ == "__main__":
    sys.exit(main())
