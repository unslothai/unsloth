# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Give the desktop app the ROCm environment a terminal launch already has.

unsloth#9926: ``fix_path_env::fix()`` is ``fix_vars(&["PATH"])``, so src-tauri
reads the login shell and keeps PATH out of it, dropping every ROCm variable
beside it. Parity, not policy: only a desktop launch, only an AMD host, only
allowlisted names that are absent here. Every other launch reads no shell.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile

logger = logging.getLogger(__name__)

DISABLE_ENV_VAR = "UNSLOTH_DISABLE_SHELL_ENV_IMPORT"

# Set by src-tauri on every CLI child it owns (process.rs, DESKTOP_MANAGED_ENV).
DESKTOP_MANAGED_ENV = "UNSLOTH_DESKTOP_MANAGED"

# From the CLI's #7331 guard, which ran against a GUI environment that never had
# the override: the arbiter travels, its verdict would say nothing here.
ROCM_INSTALLED_ARCH_ENV = "UNSLOTH_ROCM_INSTALLED_ARCH"
HSA_OVERRIDE_ENV = "HSA_OVERRIDE_GFX_VERSION"

# AMD/ROCm only: a wider list would change a GUI launch on someone else's stack.
ROCM_SHELL_ENV_ALLOWLIST: tuple[str, ...] = (
    "HSA_OVERRIDE_GFX_VERSION",
    "PYTORCH_ROCM_ARCH",
    "AMDGPU_TARGETS",
    "GPU_TARGETS",
    "ROCM_PATH",
    "ROCM_HOME",
    "HIP_PATH",
    "HIP_PLATFORM",
    # CK was being attempted on an arch it was not built for (#9926).
    "USE_CK",
    "TORCH_BLAS_PREFER_HIPBLASLT",
    "MIOPEN_USER_DB_PATH",
    "MIOPEN_CUSTOM_CACHE_DIR",
    "MIOPEN_FIND_MODE",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    # DXG_DETECTION: main.py sets it for WSL, a host that set it by hand wins.
    "HSA_ENABLE_SDMA",
    "HSA_ENABLE_DXG_DETECTION",
    "HSA_XNACK",
    "HSA_FORCE_FINE_GRAIN_PCIE",
    "AMD_SERIALIZE_KERNEL",
    "GPU_MAX_HW_QUEUES",
)
# Not HSA_TOOLS_LIB: HSA dlopens it, which loads a library rather than tuning one.

# NVIDIA's open kernel module registers KFD nodes too (4318), hence the check.
_AMD_VENDOR_ID = "4098"


def host_has_amd_gpu() -> bool:
    """Whether the amdgpu driver is presenting a GPU here.

    The KFD topology, not torch: this runs before torch is imported.
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
            if _node_is_an_amd_gpu(text):
                return True
    except Exception:
        return False
    return False


def _node_is_an_amd_gpu(properties: str) -> bool:
    """Whether one KFD node's ``properties`` file describes an AMD GPU."""
    fields = {}
    for line in properties.splitlines():
        key, _, value = line.partition(" ")
        fields[key] = value.strip()
    try:
        if int(fields.get("gfx_target_version", "0")) <= 0:
            return False
    except ValueError:
        return False
    return fields.get("vendor_id") == _AMD_VENDOR_ID


def read_login_shell_env(shell: "str | None" = None, timeout: float = 15.0) -> dict:
    """The environment an interactive login shell would have handed us, or ``{}``.

    ``-i`` runs ``~/.zshrc`` and ``-l`` the profile chain; bash reaches
    ``~/.bashrc`` only because the stock ``~/.profile`` sources it. ``env -0``
    into a FILE: a newline in a value corrupts a line parse, and an rc that
    backgrounds a job leaves that child holding a capture pipe.
    """
    shell = shell or os.environ.get("SHELL") or "/bin/sh"
    with tempfile.TemporaryDirectory(prefix = "unsloth-shell-env-") as work:
        target = os.path.join(work, "env")
        try:
            process = subprocess.Popen(
                [shell, "-ilc", f"env -0 > {shlex.quote(target)}"],
                stdin = subprocess.DEVNULL,
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
                # Oh My Zsh's auto-update prompt can block the shell forever.
                env = {**os.environ, "DISABLE_AUTO_UPDATE": "true"},
                # Its own group, so the cleanup below takes the whole shell.
                start_new_session = True,
            )
        except Exception as error:
            logger.debug("login shell environment unavailable: %s", error)
            return {}
        # pgid == pid, read before the wait reaps it: getpgid then raises.
        group = process.pid
        try:
            returncode = process.wait(timeout = timeout)
        except Exception as error:
            logger.debug("login shell did not finish: %s", error)
            returncode = None
        finally:
            # Every path: a clean exit still leaves an rc's agent running.
            _terminate_group(group, process)
        if returncode != 0:
            logger.debug("login shell exited %s", returncode)
            return {}
        try:
            with open(target, "rb") as handle:
                raw = handle.read()
        except OSError as error:
            logger.debug("login shell wrote no environment: %s", error)
            return {}

    out: dict = {}
    # surrogateescape, as os.environ does: `replace` corrupts a path.
    for record in raw.decode("utf-8", "surrogateescape").split("\0"):
        name, sep, value = record.partition("=")
        if sep and name:
            out[name] = value
    return out


def _terminate_group(group: int, process) -> None:
    """Kill the shell's session and anything left in it. Never raises."""
    try:
        os.killpg(group, signal.SIGKILL)
    except Exception:
        pass
    try:
        process.wait(timeout = 5)
    except Exception:
        pass


def select_missing_vars(
    environ,
    shell_env,
    allowlist = ROCM_SHELL_ENV_ALLOWLIST,
) -> dict:
    """The allowlisted names the shell has and this process does not.

    Membership on both sides, never truthiness: exported empty is a statement, and
    ``ROCR_VISIBLE_DEVICES=`` hides every agent.
    """
    out: dict = {}
    for name in allowlist:
        if name in environ:
            continue
        if name in shell_env and isinstance(shell_env[name], str):
            out[name] = shell_env[name]
    return out


def import_rocm_env_from_login_shell(
    environ = None,
    shell = None,
    timeout: float = 15.0,
) -> dict:
    """Fill in the ROCm variables a desktop launch dropped. Returns what it set."""
    environ = os.environ if environ is None else environ
    if str(environ.get(DISABLE_ENV_VAR, "")).strip() == "1":
        return {}
    # Not a desktop launch: nothing was lost, so nothing is read.
    if str(environ.get(DESKTOP_MANAGED_ENV, "")).strip() != "1":
        return {}
    if not sys.platform.startswith("linux"):
        return {}
    if not host_has_amd_gpu():
        return {}
    if all(name in environ for name in ROCM_SHELL_ENV_ALLOWLIST):
        return {}

    imported = select_missing_vars(environ, read_login_shell_env(shell, timeout))
    if HSA_OVERRIDE_ENV in imported and override_contradicts_install(
        imported[HSA_OVERRIDE_ENV], environ.get(ROCM_INSTALLED_ARCH_ENV)
    ):
        logger.info(
            "Not importing %s=%s from the login shell: this install carries %s kernels "
            "only (#7331).",
            HSA_OVERRIDE_ENV,
            imported[HSA_OVERRIDE_ENV],
            environ.get(ROCM_INSTALLED_ARCH_ENV),
        )
        del imported[HSA_OVERRIDE_ENV]
    for name, value in imported.items():
        environ[name] = value
    if imported:
        logger.info(
            "Imported ROCm environment from the login shell (desktop launches do "
            "not inherit it): %s",
            ", ".join(sorted(imported)),
        )
    return imported


def override_gfx_arch(value):
    """The gfx arch an ``HSA_OVERRIDE_GFX_VERSION`` value names, or None.

    In step with ``_hsa_override_gfx_arch`` in unsloth_cli/commands/studio.py,
    install_python_stack.py and install.sh; the parity is tested.
    """
    if not isinstance(value, str) or not value:
        return None
    # [0-9] rather than str.isdigit()/\d, both of which accept non-ASCII digits.
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", value.strip()):
        return None
    major, minor, step = (int(part) for part in value.strip().split("."))
    # Steppings are a single hex nibble; anything wider is not a real target.
    if not (0 <= step <= 15) or major <= 0 or minor > 9:
        return None
    return f"gfx{major}{minor}{step:x}"


def override_contradicts_install(value, installed_arch) -> bool:
    """Whether this override names an arch the installed ROCm wheels cannot serve.

    False for a value that does not parse, matching the CLI guard, which leaves an
    override it cannot read alone.
    """
    if not installed_arch:
        return False
    named = override_gfx_arch(value)
    return named is not None and named != installed_arch
