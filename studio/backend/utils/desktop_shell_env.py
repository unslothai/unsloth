# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Give the desktop app the ROCm environment a terminal launch already has.

unsloth#9926: training a small Qwen on a Radeon RX 7600 (gfx1102) SIGSEGVs the
whole backend when Studio is started from the desktop app, and the identical
model, dataset and machine train fine when it is started from a terminal with
``unsloth studio``. The reporter's ``~/.bashrc`` carries

    export HSA_OVERRIDE_GFX_VERSION=11.0.0
    export ROCM_PATH=/opt/rocm
    export USE_CK=0

and those are what differ. A GUI process does not get them: the desktop app
calls ``fix_path_env::fix()``, which is ``fix_vars(&["PATH"])`` -- it spawns the
login shell, reads the whole environment, and then sets exactly one variable
from it. So PATH arrives and every ROCm variable beside it is dropped. Web mode
inherits the terminal it was typed in and keeps all of them. That is the whole
difference between the two modes, and it is why the bug looks like "Desktop is
broken and Web is fine" rather than like a ROCm problem.

The rule here is parity, not policy:

  * a variable is imported only if it is **absent** from this process. Started
    from a terminal, every one of them is already set and this is a no-op, so
    both launch modes end in the same environment rather than in two new ones.
  * only the names in ``ROCM_SHELL_ENV_ALLOWLIST`` are considered, and every one
    of them is AMD/ROCm specific.
  * the login shell is spawned only when this host has an AMD GPU, read from the
    kernel. On an NVIDIA, Intel or Apple host nothing is read, nothing is set,
    and no shell runs, so those paths are untouched by construction rather than
    by an allowlist that happens not to overlap.

Not needed on Windows: the desktop app there inherits the user environment
normally, and there is no login shell to read.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)

# Set this to "1" to keep the desktop app's environment exactly as the desktop
# session handed it over.
DISABLE_ENV_VAR = "UNSLOTH_DISABLE_SHELL_ENV_IMPORT"

# AMD/ROCm runtime knobs only. Deliberately no ``CUDA_*``, no ``ONEAPI_*``, no
# ``PYTORCH_*`` general switches and no ``UNSLOTH_*``: the point of this module
# is the AMD launch gap in #9926, and a wider list would make a GUI launch on an
# NVIDIA box behave differently from the release before it.
ROCM_SHELL_ENV_ALLOWLIST: tuple[str, ...] = (
    # Arch selection and overrides. The reporter's crash is here.
    "HSA_OVERRIDE_GFX_VERSION",
    "PYTORCH_ROCM_ARCH",
    "AMDGPU_TARGETS",
    "GPU_TARGETS",
    # Where ROCm lives, for hosts that did not install to /opt/rocm.
    "ROCM_PATH",
    "ROCM_HOME",
    "HIP_PATH",
    "HIP_PLATFORM",
    # Kernel/library backend selection. USE_CK=0 is the other variable in the
    # report, and the log shows CK being attempted on an unsupported arch.
    "USE_CK",
    "TORCH_BLAS_PREFER_HIPBLASLT",
    "MIOPEN_USER_DB_PATH",
    "MIOPEN_CUSTOM_CACHE_DIR",
    "MIOPEN_FIND_MODE",
    # Which devices ROCm may use, so a GUI launch sees the same cards as a shell.
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    # HSA runtime behaviour. HSA_ENABLE_DXG_DETECTION is the WSL bridge main.py
    # already sets for itself; keeping it here means a host that set it by hand
    # is not second-guessed.
    "HSA_ENABLE_SDMA",
    "HSA_ENABLE_DXG_DETECTION",
    "HSA_XNACK",
    "HSA_FORCE_FINE_GRAIN_PCIE",
    "HSA_TOOLS_LIB",
    "AMD_SERIALIZE_KERNEL",
    "GPU_MAX_HW_QUEUES",
)

_DELIMITER = "__UNSLOTH_SHELL_ENV__"
_SHELL_COMMAND = f'printf %s {_DELIMITER}; env -0; printf %s {_DELIMITER}'


def host_has_amd_gpu() -> bool:
    """Whether the amdgpu kernel driver is presenting a GPU on this host.

    Read from the KFD topology rather than from torch, because this runs before
    torch is imported and importing it here would both cost seconds and create a
    device context on a machine that may not want one. A node whose
    ``gfx_target_version`` is 0 is a CPU node, and every host has those.
    """
    if not sys.platform.startswith("linux"):
        return False
    try:
        if not os.path.exists("/dev/kfd"):
            return False
        nodes = "/sys/class/kfd/kfd/topology/nodes"
        for entry in sorted(os.listdir(nodes)):
            path = os.path.join(nodes, entry, "properties")
            try:
                with open(path, "r", encoding = "utf-8", errors = "replace") as handle:
                    text = handle.read()
            except OSError:
                continue
            for line in text.splitlines():
                key, _, value = line.partition(" ")
                if key != "gfx_target_version":
                    continue
                try:
                    if int(value.strip()) > 0:
                        return True
                except ValueError:
                    pass
                break
    except Exception:
        return False
    return False


def read_login_shell_env(shell: "str | None" = None, timeout: float = 15.0) -> dict:
    """The environment an interactive login shell would have handed us.

    ``-ilc`` is what makes ``~/.bashrc`` and ``~/.zshrc`` run; a non-interactive
    shell skips them and this would read back the environment we already have.
    ``env -0`` rather than ``env``, because a value containing a newline splits a
    line-based parse and silently corrupts the variable after it.

    Returns ``{}`` on any failure. A shell that is slow, missing, or noisy is not
    a reason to fail a launch: the caller's fallback is the status quo.
    """
    shell = shell or os.environ.get("SHELL") or "/bin/sh"
    try:
        completed = subprocess.run(
            [shell, "-ilc", _SHELL_COMMAND],
            capture_output = True,
            timeout = timeout,
            # Oh My Zsh's auto-update prompt can block the shell forever.
            env = {**os.environ, "DISABLE_AUTO_UPDATE": "true"},
            stdin = subprocess.DEVNULL,
        )
    except Exception as error:
        logger.debug("login shell environment unavailable: %s", error)
        return {}
    if completed.returncode != 0:
        logger.debug("login shell exited %s", completed.returncode)
        return {}

    raw = completed.stdout.decode("utf-8", "replace")
    # Login shells print banners, motd and the occasional progress bar. The
    # delimiters bound our own output inside all of it.
    parts = raw.split(_DELIMITER)
    if len(parts) < 3:
        logger.debug("login shell output was not delimited as expected")
        return {}

    out: dict = {}
    for record in parts[1].split("\0"):
        if not record:
            continue
        name, sep, value = record.partition("=")
        if sep and name:
            out[name] = value
    return out


def select_missing_vars(environ, shell_env, allowlist = ROCM_SHELL_ENV_ALLOWLIST) -> dict:
    """The allowlisted names the shell has and this process does not.

    ``in environ`` and not truthiness: a variable deliberately exported empty is
    set, and overwriting it would be this module inventing a policy rather than
    restoring parity.
    """
    out: dict = {}
    for name in allowlist:
        if name in environ:
            continue
        value = shell_env.get(name)
        if isinstance(value, str) and value != "":
            out[name] = value
    return out


def import_rocm_env_from_login_shell(environ = None, shell = None, timeout: float = 15.0) -> dict:
    """Fill in the ROCm variables a desktop launch dropped. Returns what it set.

    Safe to call more than once: the second call finds every name already
    present and imports nothing.
    """
    environ = os.environ if environ is None else environ
    if str(environ.get(DISABLE_ENV_VAR, "")).strip() == "1":
        return {}
    if not sys.platform.startswith(("linux", "darwin")):
        return {}
    # The gate that keeps every other vendor's path byte-identical: no AMD GPU,
    # no shell spawned, nothing read, nothing set.
    if not host_has_amd_gpu():
        return {}
    # Nothing to gain from a shell when every name is already set, and it costs
    # a few hundred milliseconds of startup.
    if all(name in environ for name in ROCM_SHELL_ENV_ALLOWLIST):
        return {}

    imported = select_missing_vars(environ, read_login_shell_env(shell, timeout))
    for name, value in imported.items():
        environ[name] = value
    if imported:
        logger.info(
            "Imported ROCm environment from the login shell (desktop launches do "
            "not inherit it): %s",
            ", ".join(f"{k}={v}" for k, v in sorted(imported.items())),
        )
    return imported
