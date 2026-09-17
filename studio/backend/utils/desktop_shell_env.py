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

  * it runs only on the launch that loses the environment, the one the desktop
    app marks with ``UNSLOTH_DESKTOP_MANAGED=1``. A terminal launch, a service
    and a container read no shell and set nothing, so they are unchanged by
    construction rather than by an allowlist that happens not to overlap.
  * and only when the host has an AMD GPU, read from the KFD topology and vendor
    checked, so an NVIDIA, Intel or Apple host takes the same early return.
  * a variable is imported only if it is **absent** from this process, so the two
    launch modes end in the same environment rather than in two new ones.
  * only the names in ``ROCM_SHELL_ENV_ALLOWLIST`` are considered, and every one
    of them is AMD/ROCm specific.

Not needed on Windows: the desktop app there inherits the user environment
normally, and there is no login shell to read.
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

# Set this to "1" to keep the desktop app's environment exactly as the desktop
# session handed it over.
DISABLE_ENV_VAR = "UNSLOTH_DISABLE_SHELL_ENV_IMPORT"

# Set by the desktop app on every CLI child it owns (src-tauri/src/process.rs,
# DESKTOP_MANAGED_ENV). This module exists because that launch loses the shell
# environment, so this is the launch it runs on: `unsloth studio` from a terminal
# already has the variables, and a service or container launch gets the same
# startup it got before this module existed, with no shell spawned.
DESKTOP_MANAGED_ENV = "UNSLOTH_DESKTOP_MANAGED"

# The single gfx arch this install carries kernels for, published by the CLI's #7331
# guard (unsloth_cli/commands/studio.py, ROCM_INSTALLED_ARCH_ENV). That guard runs
# against the GUI environment, which on a desktop launch never had the override, so
# it clears nothing and the contradicting value is still in the profile read below.
# The arbiter has to travel, not just its verdict.
ROCM_INSTALLED_ARCH_ENV = "UNSLOTH_ROCM_INSTALLED_ARCH"
HSA_OVERRIDE_ENV = "HSA_OVERRIDE_GFX_VERSION"

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
    "AMD_SERIALIZE_KERNEL",
    "GPU_MAX_HW_QUEUES",
)
# Deliberately NOT here: HSA_TOOLS_LIB. The HSA runtime dlopens whatever it names,
# so importing it would load a library into the backend rather than tune it.

# AMD's PCI vendor id. NVIDIA's open kernel module registers KFD nodes too, with
# 4318, so AMD ownership is confirmed rather than assumed -- the same guard as
# hardware.py::_linux_kfd_reports_an_amd_gpu and install_python_stack.
_AMD_VENDOR_ID = "4098"


def host_has_amd_gpu() -> bool:
    """Whether the amdgpu kernel driver is presenting a GPU on this host.

    Read from the KFD topology rather than from torch, because this runs before
    torch is imported and importing it here would both cost seconds and create a
    device context on a machine that may not want one. A node counts only when it
    is a GPU (``gfx_target_version`` above 0, every host has CPU nodes at 0) AND
    AMD owns it, since an NVIDIA open-driver host presents GPU nodes here too.
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
    """The environment an interactive login shell would have handed us.

    ``-i`` is what makes ``~/.zshrc`` run, and ``-l`` the profile chain; bash
    reads ``~/.bashrc`` only because the stock ``~/.profile`` sources it, so a
    host that has replaced that file gets whatever its own profile exports.

    ``env -0`` rather than ``env``, because a value containing a newline splits a
    line-based parse and silently corrupts the variable after it, and into a FILE
    rather than a pipe: an rc file that backgrounds a job (an agent, a daemon)
    leaves that child holding the capture pipe, so reading stdout would wait for
    the child rather than for the shell, spend the whole timeout and then discard
    an environment the shell had already written correctly.

    Returns ``{}`` on any failure. A shell that is slow, missing, or noisy is not
    a reason to fail a launch: the caller's fallback is the status quo.
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
                # Its own group, so the timeout below can take the shell's
                # children with it instead of orphaning them onto init.
                start_new_session = True,
            )
        except Exception as error:
            logger.debug("login shell environment unavailable: %s", error)
            return {}
        # start_new_session makes the shell its own session leader, so its pgid is
        # its pid. Kept before the wait below reaps it, because getpgid on a reaped
        # pid raises.
        group = process.pid
        try:
            returncode = process.wait(timeout = timeout)
        except Exception as error:
            logger.debug("login shell did not finish: %s", error)
            returncode = None
        finally:
            # On EVERY path, not only the timeout. An rc that backgrounds an agent
            # or a daemon leaves it running here after the shell itself exits
            # cleanly, and that would be one orphan adopted by init per launch.
            # This probe is not the login session those were meant to outlive.
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
    # surrogateescape, not replace: this is how os.environ itself carries a byte
    # that is not valid UTF-8, so a path with one round trips instead of picking
    # up a replacement character.
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
        # Already empty, or a platform without process groups.
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

    Membership on both sides, never truthiness. A variable deliberately exported
    empty is set, so it is not overwritten here; and ``ROCR_VISIBLE_DEVICES=``
    exported empty in the shell hides every agent, so dropping it would leave the
    desktop launch with the cards the terminal launch does not have.
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
    """Fill in the ROCm variables a desktop launch dropped. Returns what it set.

    Never overwrites a name this process already carries, so calling it twice
    imports nothing the second time.
    """
    environ = os.environ if environ is None else environ
    if str(environ.get(DISABLE_ENV_VAR, "")).strip() == "1":
        return {}
    # Every gate below leaves a launch byte-identical to the release before this
    # module existed. Not a desktop launch: nothing was lost, so nothing is read
    # and no shell runs, which is what keeps web mode, a service and a container
    # unchanged rather than merely allowlisted.
    if str(environ.get(DESKTOP_MANAGED_ENV, "")).strip() != "1":
        return {}
    if not sys.platform.startswith("linux"):
        return {}
    if not host_has_amd_gpu():
        return {}
    # Nothing to gain from a shell when every name is already set.
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

    Kept in step with ``_hsa_override_gfx_arch`` in unsloth_cli/commands/studio.py,
    studio/install_python_stack.py and install.sh; the parity is tested.
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

    False when the install is not known to be single-ISA, and false for a value
    that does not parse: the CLI guard leaves an unreadable override alone rather
    than removing it, and the two have to agree or a launch behaves differently
    depending on which one saw the variable first.
    """
    if not installed_arch:
        return False
    named = override_gfx_arch(value)
    return named is not None and named != installed_arch
