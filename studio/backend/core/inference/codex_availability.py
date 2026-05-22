# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Codex CLI / SDK availability probe.

This module never imports `codex_app_server` at module top level. The
SDK is optional and not pinned in pyproject.toml -- if it's installed
locally, we use it; if it isn't, the probe simply returns
``installed=False`` and the provider stays hidden in the frontend.

The frontend calls ``GET /api/codex/status`` at startup to decide
whether to surface the "codex" entry in the provider picker. Three
states matter:

* ``installed=False`` -- either the CLI is missing OR the SDK
  (``codex_app_server``) is not importable. The picker hides the
  entry entirely.
* ``installed=True, logged_in=False`` -- everything resolves on the
  Python side but ``codex auth status`` (or equivalent) reports no
  active credentials. The provider config dialog shows a
  ``Sign in to Codex`` button instead of the regular API-key field.
* ``installed=True, logged_in=True`` -- ready to use; the picker
  shows the regular model dropdown.

Detection is best-effort and cheap: we shell out to ``which codex``
plus ``codex --version`` for the CLI and use ``importlib.util.find_spec``
for the SDK. No long-running CLI commands are invoked here so the
status endpoint is safe to poll on every page load.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import shutil
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


# Default catalog of models surfaced in the picker when the CLI is
# present but doesn't advertise a list. The SDK accepts arbitrary model
# ids; this is purely a sensible default.
_DEFAULT_SUPPORTED_MODELS: tuple[str, ...] = (
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.5",
    "o3",
)


def _which_codex() -> Optional[str]:
    """Return absolute path to the ``codex`` CLI, or None if missing.

    Uses :func:`shutil.which` so the lookup honours ``PATH`` exactly
    the way the user's shell would. Returns ``None`` on any failure
    so callers can treat "missing" and "broken probe" the same way.
    """
    try:
        return shutil.which("codex")
    except Exception as exc:
        # shutil.which itself is documented as raising only on
        # genuinely unusual conditions, but a hardened wrapper costs
        # nothing and keeps the status endpoint from 500'ing.
        logger.warning("codex_availability.which_failed", error = str(exc))
        return None


def _sdk_importable() -> bool:
    """True iff ``codex_app_server`` is importable in this interpreter.

    We deliberately use :func:`importlib.util.find_spec` instead of an
    ``import codex_app_server`` so the import never actually runs --
    that keeps the cost negligible and avoids the SDK's own side
    effects (which include reaching out to the CLI subprocess for an
    RPC ping) during a simple availability check.
    """
    try:
        return importlib.util.find_spec("codex_app_server") is not None
    except Exception as exc:
        logger.warning("codex_availability.find_spec_failed", error = str(exc))
        return False


async def _run_cli(args: list[str], *, timeout: float = 4.0) -> tuple[int, str, str]:
    """Run a short ``codex`` CLI command and return (rc, stdout, stderr).

    The probe uses 4s as the wall-clock cap because ``codex --version``
    and ``codex auth status`` both return in well under a second on a
    healthy install. A longer probe would block the
    ``/api/codex/status`` route -- and that route fires on every chat
    page load, so a tight cap matters.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "codex",
            *args,
            stdout = asyncio.subprocess.PIPE,
            stderr = asyncio.subprocess.PIPE,
            env = os.environ.copy(),
        )
    except FileNotFoundError:
        return -1, "", "codex binary not on PATH"
    except Exception as exc:
        logger.warning(
            "codex_availability.spawn_failed",
            args = args,
            error = str(exc),
        )
        return -1, "", str(exc)

    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout = timeout)
    except asyncio.TimeoutError:
        proc.kill()
        try:
            await proc.wait()
        except Exception:
            pass
        return -1, "", f"codex {' '.join(args)} timed out after {timeout:.1f}s"

    return (
        proc.returncode if proc.returncode is not None else -1,
        stdout_b.decode("utf-8", errors = "replace").strip(),
        stderr_b.decode("utf-8", errors = "replace").strip(),
    )


async def _detect_version() -> Optional[str]:
    rc, stdout, stderr = await _run_cli(["--version"])
    if rc != 0:
        return None
    # ``codex --version`` prints something like "codex-cli 0.133.0".
    # Surface the whole line so the UI can show the exact build the
    # user has installed -- it's useful when troubleshooting.
    text = stdout or stderr
    return text.split("\n", 1)[0].strip() if text else None


async def _detect_logged_in() -> bool:
    """Best-effort: assume a non-zero ``codex auth status`` rc means
    "not logged in". The CLI surface has shifted over releases (some
    versions print to stderr, some to stdout, some use "Logged in as
    ..." vs "Not authenticated"). The return code is the most stable
    signal we have, so we lean on it and fall back to substring
    inspection only when the rc itself is ambiguous (rc=0 but no
    output, or rc=-1 from a timeout / crash).
    """
    rc, stdout, stderr = await _run_cli(["auth", "status"])
    combined = f"{stdout}\n{stderr}".lower()
    if rc == 0:
        # Some 0.x releases exit 0 even when the user is logged out.
        # If we got any output, look for the obvious "not logged in"
        # signals; if there's nothing on either pipe at all, treat
        # rc=0 as authenticated (the optimistic default).
        if not combined.strip():
            return True
        if "not authenticated" in combined or "not logged in" in combined:
            return False
        if "logged in" in combined or "authenticated" in combined:
            return True
        return True
    # rc != 0: lean on substring matching one more time before
    # defaulting to False, in case a future CLI uses rc=2 for the
    # logged-out state instead of "no auth command exists".
    if "logged in" in combined or "authenticated as" in combined:
        return True
    return False


async def probe_codex_availability() -> dict[str, Any]:
    """Return the full status payload consumed by ``GET /api/codex/status``.

    Returns a dict with keys:

    * ``installed`` (bool) -- True iff both CLI and SDK are present.
      Note this is the gate the frontend uses to surface the provider
      at all, so if EITHER is missing the picker hides the entry.
    * ``cli_path`` (str | None) -- absolute path to the CLI, or None.
    * ``sdk_importable`` (bool) -- the Python SDK is importable.
    * ``logged_in`` (bool) -- best-effort auth check; meaningless when
      ``installed`` is False.
    * ``version`` (str | None) -- the ``codex --version`` first line.
    * ``supported_models`` (list[str]) -- default model id catalog.
    """
    cli_path = _which_codex()
    sdk_ok = _sdk_importable()

    payload: dict[str, Any] = {
        "installed": bool(cli_path) and sdk_ok,
        "cli_path": cli_path,
        "sdk_importable": sdk_ok,
        "logged_in": False,
        "version": None,
        "supported_models": list(_DEFAULT_SUPPORTED_MODELS),
    }

    if cli_path:
        # version + login probes only matter when the CLI is present;
        # they would otherwise just churn subprocess errors. Run them
        # in parallel because both are independent CLI invocations.
        version, logged_in = await asyncio.gather(
            _detect_version(),
            _detect_logged_in(),
        )
        payload["version"] = version
        payload["logged_in"] = bool(logged_in)

    logger.info(
        "codex_availability.probed",
        installed = payload["installed"],
        sdk_importable = payload["sdk_importable"],
        cli_path = payload["cli_path"],
        version = payload["version"],
        logged_in = payload["logged_in"],
    )
    return payload
