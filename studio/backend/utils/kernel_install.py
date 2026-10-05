# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth install-kernels xformers causal_conv1d`: prebuilt kernel wheels matched to the resident torch.

Wheel-only by design: no wheel for this torch / CUDA / Python means nothing is installed and the
model keeps its torch fallback, never a source build. Installs use --no-deps so torch is never touched.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from typing import Callable

from utils.wheel_utils import (
    CAUSAL_CONV1D_PACKAGE_VERSION,
    CAUSAL_CONV1D_RELEASE_BASE_URL,
    CAUSAL_CONV1D_RELEASE_TAG,
    MAMBA_SSM_PACKAGE_VERSION,
    MAMBA_SSM_RELEASE_BASE_URL,
    MAMBA_SSM_RELEASE_TAG,
    direct_wheel_url,
    install_wheel,
    probe_torch_wheel_env,
    redact_url_credentials,
    url_exists,
    xformers_wheel_url,
)

# name -> (pip distribution, check that loads the compiled extension). The package imports are no
# proof: `import causal_conv1d` never loads it, and `import mamba_ssm` fails on einops under --no-deps.
KERNELS = {
    "xformers": (
        "xformers",
        "from xformers._cpp_lib import _register_extensions; _register_extensions()",
    ),
    "causal_conv1d": ("causal-conv1d", "import torch, causal_conv1d_cuda"),
    "mamba_ssm": ("mamba-ssm", "import torch, selective_scan_cuda"),
}

_RELEASES = {
    "causal_conv1d": (
        CAUSAL_CONV1D_PACKAGE_VERSION,
        CAUSAL_CONV1D_RELEASE_TAG,
        CAUSAL_CONV1D_RELEASE_BASE_URL,
    ),
    "mamba_ssm": (MAMBA_SSM_PACKAGE_VERSION, MAMBA_SSM_RELEASE_TAG, MAMBA_SSM_RELEASE_BASE_URL),
}


def resolve_wheel_url(name: str, env: dict[str, str] | None) -> str | None:
    if env is None:
        return None
    if name == "xformers":
        return xformers_wheel_url(env)
    # causal-conv1d and mamba-ssm publish Linux wheels only.
    if not str(env.get("platform_tag") or "").startswith("linux"):
        return None
    package_version, release_tag, release_base_url = _RELEASES[name]
    return direct_wheel_url(
        filename_prefix = name,
        package_version = package_version,
        release_tag = release_tag,
        release_base_url = release_base_url,
        env = env,
    )


def _loads(check: str, run: Callable[..., subprocess.CompletedProcess]) -> bool:
    result = run(
        [sys.executable, "-c", check], stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL
    )
    return result.returncode == 0


def _outside_venv() -> bool:
    # Colab and other system interpreters: uv refuses them without --system.
    return sys.prefix == sys.base_prefix


def _uninstall(distribution: str, run: Callable[..., subprocess.CompletedProcess]) -> bool:
    if shutil.which("uv"):
        system = ["--system"] if _outside_venv() else []
        cmd = ["uv", "pip", "uninstall", *system, "--python", sys.executable, distribution]
    else:
        cmd = [sys.executable, "-m", "pip", "uninstall", "-y", distribution]
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
    attempts = install_wheel(
        url,
        python_executable = sys.executable,
        use_uv = True,
        uv_needs_system = _outside_venv(),
        reinstall = True,
        run = run,
    )
    result = attempts[-1][1]
    if result.returncode != 0:
        print(
            f"Unsloth: installing {name} from {shown} failed; using the torch fallback.\n"
            + result.stdout[-2000:].replace(url, shown)
        )
        return 1
    if not _loads(check, run):
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
    parser.add_argument("kernels", nargs = "+", choices = sorted(KERNELS))
    parser.add_argument(
        "--dry-run", action = "store_true", help = "print the resolved wheel URL and install nothing"
    )
    args = parser.parse_args(argv)
    env = probe_torch_wheel_env(timeout = 120, include_windows = True)
    status = 0
    for name in dict.fromkeys(args.kernels):
        status = max(status, install_kernel(name, env, dry_run = args.dry_run))
    return status


if __name__ == "__main__":
    sys.exit(main())
