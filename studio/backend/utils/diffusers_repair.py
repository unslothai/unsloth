# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Startup self-heal for the pinned Diffusers main build.

Older installers can miss the pinned build during an update. Repair before importing the app:
the build also upgrades huggingface_hub, which would otherwise mix new files with cached modules.
Opt out with UNSLOTH_DISABLE_DIFFUSERS_AUTOREPAIR=1.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

import structlog

logger = structlog.get_logger(__name__)

DISABLE_ENV_VAR = "UNSLOTH_DISABLE_DIFFUSERS_AUTOREPAIR"
_STUDIO_DIR = Path(__file__).resolve().parents[2]
_INSTALLER = _STUDIO_DIR / "install_python_stack.py"
_MAIN_PIN = _STUDIO_DIR / "backend" / "requirements" / "diffusers-main.txt"
# Inside the desktop app's 300 s start deadline, with room left for the app import after it.
_REPAIR_TIMEOUT_S = 120
# The installer's exit codes for --repair-diffusers-main.
_INSTALLED, _NOTHING_TO_DO = 0, 1
# The installer's DIFFUSERS_MAIN_MIN_PYTHON: diffusers main needs 3.10, and 3.9 stays supported.
_MAIN_MIN_PYTHON = (3, 10)

# Unattended, so secrets and index redirects stay out, as in mlx_repair. The Windows names are what
# Python, git and uv need to start at all there; UV_OFFLINE is the operator's no-network switch.
_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "all_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "UV_SYSTEM_CERTS",
        "UV_NATIVE_TLS",
        "UV_OFFLINE",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "UNSLOTH_DIFFUSERS_MAIN",
    }
)


def _opted_out() -> bool:
    if os.environ.get(DISABLE_ENV_VAR) == "1":
        return True
    value = (os.environ.get("UNSLOTH_DIFFUSERS_MAIN") or "").strip().lower()
    return value in ("0", "false", "no", "off")


def _diffusers_is_an_index_install() -> bool:
    """True when diffusers came from an index, the only state the repair acts on.

    Read from metadata, so the common healthy boot spawns nothing: the pinned build is a direct
    reference (git or zip) and records one in direct_url.json.
    """
    try:
        from importlib.metadata import distribution
        dist = distribution("diffusers")
    except Exception:  # noqa: BLE001 - no diffusers at all is the installer's job, not this one
        return False
    try:
        payload = json.loads(dist.read_text("direct_url.json") or "{}")
    except (ValueError, OSError):
        payload = {}
    if not isinstance(payload, dict):
        return True
    # dir_info is a local checkout, `pip install -e` included: the user's, not ours to replace.
    return not ("vcs_info" in payload or "archive_info" in payload or "dir_info" in payload)


def _repair_env() -> dict[str, str]:
    env = {key: os.environ[key] for key in _ENV_ALLOWLIST if key in os.environ}
    env["VIRTUAL_ENV"] = sys.prefix
    try:
        from utils.mlx_repair import _uv_executable
        uv = _uv_executable()
    except Exception:  # noqa: BLE001 - without uv the installer falls back to pip
        uv = None
    # A GUI launch starts with a minimal PATH, and the installer looks uv up there.
    if uv:
        env["PATH"] = os.pathsep.join(filter(None, (str(Path(uv).parent), env.get("PATH"))))
    return env


def _peer_holds_pass() -> bool:
    """Whether another process (a sibling's repair, an update) is inside the dependency pass."""
    try:
        from studio.install_manifest import pass_lock
    except Exception:  # noqa: BLE001 - no lock to read is no peer to wait for
        return False
    with pass_lock() as uncontended:
        return not uncontended


class PeerInstallInProgress(RuntimeError):
    """A peer still holds the install lock at timeout, so importing packages is unsafe."""


PEER_INSTALL_MESSAGE = (
    "Another Unsloth install or update is still changing this environment after "
    f"{_REPAIR_TIMEOUT_S}s. Start Unsloth Studio again once it has finished."
)


def _record_failure() -> None:
    """Suppress startup retries until an explicit update clears the failure key."""
    try:
        from studio.install_manifest import update_manifest
        update_manifest(diffusers_main_repair = "failed")
    except Exception as exc:  # noqa: BLE001 - unrecorded means the next start tries again
        logger.warning("diffusers self-heal could not record its failure: %s", exc)


def _installer_would_skip() -> bool:
    """The installer's no-op gates: a start it would skip must not block or claim to install."""
    if sys.version_info[:2] < _MAIN_MIN_PYTHON:
        return True
    try:
        from studio.install_manifest import read_manifest
        manifest = read_manifest() or {}
    except Exception:  # noqa: BLE001 - an unreadable manifest is no record of a failed try
        return False
    step_results = manifest.get("step_results")
    update_failed = (
        isinstance(step_results, dict) and step_results.get("diffusers-main.txt") == "failed"
    )
    return manifest.get("diffusers_main_repair") == "failed" or update_failed


def _run_repair(echo: Callable[[str], None]) -> bool:
    from utils.child_stdio import utf8_child_env
    from utils.process_lifetime import adopt_pid, child_popen_kwargs, forget_pid, terminate_pid

    kwargs = child_popen_kwargs()
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.Popen(
            [sys.executable, str(_INSTALLER), "--repair-diffusers-main"],
            env = utf8_child_env(_repair_env()),
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            **kwargs,
        )
    except OSError as exc:
        logger.warning("diffusers self-heal could not start the installer: %s", exc)
        return False
    # Tracked, so a backend that exits mid-install takes the installer and its uv/git children down.
    adopt_pid(proc.pid)
    try:
        output, _ = proc.communicate(timeout = _REPAIR_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        # Stop uv/git children too before importing packages they may be replacing.
        terminate_pid(proc.pid, owner_verified = True)
        proc.kill()
        proc.wait()
        # A remaining lock belongs to a peer. Abort without changing its manifest.
        if _peer_holds_pass():
            logger.warning("diffusers self-heal timed out behind a peer's dependency pass")
            raise PeerInstallInProgress(PEER_INSTALL_MESSAGE)
        _record_failure()
        echo(
            f"  - the pinned Diffusers build took over {_REPAIR_TIMEOUT_S}s and was stopped; "
            "run `unsloth studio update` to install it"
        )
        logger.warning("diffusers self-heal timed out after %ss", _REPAIR_TIMEOUT_S)
        return False
    finally:
        if proc.poll() is not None:
            forget_pid(proc.pid)
    if proc.returncode == _INSTALLED:
        echo("  - installed the pinned Diffusers build")
        logger.info("diffusers self-heal installed the pinned Diffusers main build")
        return True
    if proc.returncode != _NOTHING_TO_DO:
        echo(
            "  - could not install the pinned Diffusers build; run `unsloth studio update` to retry"
        )
        logger.warning(
            "diffusers self-heal could not install the pinned build; run `unsloth studio "
            "update` to retry. Installer output:\n%s",
            (output or "")[-4000:],
        )
    return False


def repair_diffusers_before_imports(echo: Callable[[str], None] = lambda _line: None) -> bool:
    """Repair an index install before app imports. Return True if the build was installed.

    Raise PeerInstallInProgress if a peer holds the install lock at timeout.
    """
    if _opted_out() or not _MAIN_PIN.is_file() or not _INSTALLER.is_file():
        return False
    # Check the lock too: an active install may have temporarily removed the metadata.
    if _diffusers_is_an_index_install() and not _installer_would_skip():
        echo("  - installing the pinned Diffusers build (first start after an update)...")
    elif _peer_holds_pass():
        echo("  - waiting for another Unsloth install or update to finish...")
    else:
        return False
    logger.info("installing the pinned Diffusers main build. Set %s=1 to disable.", DISABLE_ENV_VAR)
    return _run_repair(echo)
