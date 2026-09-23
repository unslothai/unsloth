# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Startup self-heal for the pinned Diffusers main build.

An update from a release that predates the installer's Diffusers main step runs that release's
installer, so the build never goes in and Qwen-Image-2.1 refuses to load until a second update. The
backend is the first new code such a host runs, so it runs the installer's own step here, on a
background thread (git, or the zip route without git). Opt out with
UNSLOTH_DISABLE_DIFFUSERS_AUTOREPAIR=1.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

DISABLE_ENV_VAR = "UNSLOTH_DISABLE_DIFFUSERS_AUTOREPAIR"
_STUDIO_DIR = Path(__file__).resolve().parents[2]
_INSTALLER = _STUDIO_DIR / "install_python_stack.py"
_MAIN_PIN = _STUDIO_DIR / "backend" / "requirements" / "diffusers-main.txt"
_REPAIR_TIMEOUT_S = 900
_PEER_POLL_S = 5
# The installer's exit codes for --repair-diffusers-main.
_INSTALLED, _NOTHING_TO_DO = 0, 1

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

_lock = threading.Lock()
_thread: Optional[threading.Thread] = None
_installed = False


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


def _wait_for_peer_pass() -> None:
    """Hold the gate until no process holds the dependency pass. Killing our installer released it,
    but the pass it was waiting behind may still be writing."""
    while _peer_holds_pass():
        time.sleep(_PEER_POLL_S)


def _run_repair() -> None:
    global _installed
    from utils.child_stdio import utf8_child_env
    from utils.process_lifetime import adopt_pid, child_popen_kwargs, forget_pid, terminate_pid

    kwargs = child_popen_kwargs()
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.Popen(
            [sys.executable, str(_INSTALLER), "--repair-diffusers-main"],
            env=utf8_child_env(_repair_env()),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            **kwargs,
        )
    except OSError as exc:
        logger.warning("diffusers self-heal could not start the installer: %s", exc)
        return
    # Tracked, so a backend that exits mid-install takes the installer and its uv/git children down.
    adopt_pid(proc.pid)
    try:
        output, _ = proc.communicate(timeout=_REPAIR_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        # The whole tree, before the gate reopens: uv keeps rewriting diffusers after its parent dies.
        terminate_pid(proc.pid, owner_verified=True)
        proc.kill()
        proc.wait()
        logger.warning("diffusers self-heal timed out after %ss", _REPAIR_TIMEOUT_S)
        _wait_for_peer_pass()
        return
    finally:
        if proc.poll() is not None:
            forget_pid(proc.pid)
    if proc.returncode == _INSTALLED:
        _installed = True
        logger.info("diffusers self-heal installed the pinned Diffusers main build")
    elif proc.returncode != _NOTHING_TO_DO:
        logger.warning(
            "diffusers self-heal could not install the pinned build; run `unsloth studio "
            "update` to retry. Installer output:\n%s",
            (output or "")[-4000:],
        )


def start_diffusers_autorepair_if_needed() -> bool:
    """Start the background install when diffusers is an index release and the pin wants main.
    True iff a repair thread was started; at most once per process."""
    global _thread
    if _opted_out() or not _MAIN_PIN.is_file() or not _INSTALLER.is_file():
        return False
    # A peer mid-install can have removed the metadata, so its pass also starts the installer, which
    # waits the pass out and keeps loads refused meanwhile.
    if not _diffusers_is_an_index_install() and not _peer_holds_pass():
        return False
    with _lock:
        if _thread is not None:
            return False
        _thread = threading.Thread(target=_run_repair, daemon=True, name="diffusers-autorepair")
        _thread.start()
    logger.info(
        "checking for the pinned Diffusers main build in the background. Set %s=1 to disable.",
        DISABLE_ENV_VAR,
    )
    return True


IN_FLIGHT_MESSAGE = (
    "Unsloth is installing the pinned diffusers build in the background, which takes a few "
    "minutes on the first start after an update. Try again shortly."
)


def diffusers_repair_in_flight() -> bool:
    thread = _thread
    return thread is not None and thread.is_alive()


def diffusers_repair_installed() -> bool:
    return _installed
