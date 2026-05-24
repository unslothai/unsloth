# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Codex CLI / SDK availability probe.

This module never imports the Codex Python SDK at module top level.
The SDK is optional and not pinned in pyproject.toml -- if it's
installed locally, we use it; if it isn't, the probe simply returns
``installed=False`` and the provider stays hidden in the frontend.

The frontend calls ``GET /api/codex/status`` at startup to decide
whether to surface the "codex" entry in the provider picker. Three
states matter:

* ``installed=False`` -- either the CLI is missing OR the SDK
  (``openai_codex`` canonical, or ``codex_app_server`` legacy alias)
  is not importable. The picker hides the entry entirely.
* ``installed=True, logged_in=False`` -- everything resolves on the
  Python side but ``codex login status`` reports no active credentials.
  The provider config dialog shows a ``Sign in to Codex`` button
  instead of the regular API-key field.
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
# ids; this is purely a sensible default. Mirrored from upstream
# ``codex-rs/models-manager/models.json``.
_DEFAULT_SUPPORTED_MODELS: tuple[str, ...] = (
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.3-codex",
    "gpt-5.2",
)

# Names the upstream Python SDK has shipped under. ``openai_codex`` is the
# canonical package at ``openai/codex/sdk/python``; ``codex_app_server`` is
# kept as a forward-compat alias because the Rust crate uses that name and
# an internal alpha may publish under it.
_SDK_MODULE_NAMES: tuple[str, ...] = ("openai_codex", "codex_app_server")

# Safe-list of environment variables forwarded to the codex subprocess.
# Studio's parent env contains secrets (HF_TOKEN, GH_TOKEN, WANDB_API_KEY,
# OPENAI key for non-codex providers, etc.); a malicious or shimmed codex
# binary earlier on PATH would receive all of them via plain os.environ
# inheritance. We pass only what codex needs to spawn its own helpers
# (PATH), resolve its auth/config dir (HOME / USER / Windows equivalents
# plus CODEX_HOME), and emit log output in the user's locale.
#
# OPENAI_API_KEY is DELIBERATELY excluded. The codex CLI authenticates
# via its own `codex login --device-auth` ChatGPT flow or via stdin
# (`--with-api-key`); Studio's stored OpenAI key belongs to the OpenAI
# provider, not Codex. Forwarding it would let a shimmed `codex` binary
# on PATH exfiltrate the user's OpenAI credential. Users who want to
# wire the same key into Codex should set CODEX_OPENAI_API_KEY or feed
# the key via `codex login --with-api-key` themselves.
_SAFE_CODEX_ENV_KEYS: tuple[str, ...] = (
    "PATH",
    "HOME",
    "USER",
    "USERNAME",
    "SHELL",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "TEMP",
    "TMP",
    "SYSTEMROOT",
    "WINDIR",
    "APPDATA",
    "LOCALAPPDATA",
    "PROGRAMDATA",
    "CODEX_HOME",
    "CODEX_OPENAI_API_KEY",
    # Studio-internal override for the round 6b fail-closed safety
    # pin gate. Kept in the safe-list so the round 6 SDK env-scrub
    # wrapper does not delete it from `os.environ` before
    # `_start_thread_with_system` checks it. The variable is not a
    # secret; the codex subprocess receiving it is harmless.
    "UNSLOTH_CODEX_ALLOW_UNSAFE_DEFAULTS",
)


def _codex_subprocess_env() -> dict[str, str]:
    """Return a scrubbed env mapping for codex subprocess spawning.

    Forwards only keys from `_SAFE_CODEX_ENV_KEYS` that are actually set
    in the parent environment, so secrets from other providers never
    reach the codex CLI.
    """
    env: dict[str, str] = {}
    for key in _SAFE_CODEX_ENV_KEYS:
        value = os.environ.get(key)
        if value is not None:
            env[key] = value
    return env


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
    """True iff the Codex Python SDK is importable in this interpreter.

    We deliberately use :func:`importlib.util.find_spec` instead of an
    actual ``import`` so the import never runs -- that keeps the cost
    negligible and avoids the SDK's own side effects (which include
    reaching out to the CLI subprocess for an RPC ping) during a
    simple availability check.

    Probes both ``openai_codex`` (the canonical upstream package name
    at ``openai/codex/sdk/python``) and ``codex_app_server`` (the Rust
    crate name, kept as a forward-compat alias).
    """
    for name in _SDK_MODULE_NAMES:
        try:
            if importlib.util.find_spec(name) is not None:
                return True
        except Exception as exc:
            logger.warning(
                "codex_availability.find_spec_failed",
                module = name,
                error = str(exc),
            )
    return False


async def _run_cli(args: list[str], *, timeout: float = 4.0) -> tuple[int, str, str]:
    """Run a short ``codex`` CLI command and return (rc, stdout, stderr).

    The probe uses 4s as the wall-clock cap because ``codex --version``
    and ``codex login status`` both return in well under a second on a
    healthy install. A longer probe would block the
    ``/api/codex/status`` route -- and that route fires on every chat
    page load, so a tight cap matters.

    Subprocess lifecycle: detached into its own process group on Unix
    via ``start_new_session=True`` (matching ``stream_codex_device_login``)
    so a hung child cannot survive ``proc.kill()`` on timeout. Without
    this, a shimmed ``codex login status`` that forks a helper then
    blocks would leave the helper running after we killed the parent.
    Windows uses ``CREATE_NEW_PROCESS_GROUP`` for the analogous
    isolation. Round 6 reviewer caught the asymmetry with the
    device-login path that already had this guard.
    """
    import os
    import signal

    spawn_kwargs: dict[str, Any] = {
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
        "env": _codex_subprocess_env(),
    }
    if os.name == "posix":
        spawn_kwargs["start_new_session"] = True
    elif os.name == "nt":
        spawn_kwargs["creationflags"] = 0x00000200  # CREATE_NEW_PROCESS_GROUP

    try:
        proc = await asyncio.create_subprocess_exec("codex", *args, **spawn_kwargs)
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
        # Kill the whole process group, not just the parent, so any
        # child the codex CLI forked also dies.
        if os.name == "posix":
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        else:
            try:
                proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            except Exception:
                pass
        proc.kill()
        try:
            await asyncio.wait_for(proc.wait(), timeout = 1.0)
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
    """Best-effort: parse ``codex login status`` output.

    The upstream subcommand is ``codex login status`` (no ``auth``
    parent). Output shapes seen in the wild:
      * "Logged in using ChatGPT" / "Logged in as user@x.com"  -> True
      * "Not logged in. Run `codex login` ..."                  -> False
      * "Not authenticated"                                      -> False
    Return code is the most stable signal but ``not logged in`` also
    exits 0 on current releases, so we substring-check explicitly.

    Note: a naive ``"logged in" in combined`` check is wrong because
    the substring appears inside "not logged in" too -- we use an
    explicit negative-prefix check first.
    """
    import re

    rc, stdout, stderr = await _run_cli(["login", "status"])
    combined = f"{stdout}\n{stderr}".lower()

    # Negative prefixes win, regardless of rc. We anchor on word
    # boundaries so "not logged in" / "not authenticated" both match
    # without being fooled by the substring "logged in" inside them.
    # Covers the variants seen across CLI releases and locales.
    negative = re.compile(
        r"\b(not\s+(?:logged|signed)\s+in|"
        r"not\s+authenticated|"
        r"please\s+(?:log|sign)\s+in|"
        r"run\s+`?codex\s+login`?)\b"
    )
    if negative.search(combined):
        return False

    positive = re.compile(
        r"\b("
        r"logged in|"
        r"authenticated as|"
        r"authenticated:\s*yes|"
        r"signed in"
        r")\b"
    )
    if positive.search(combined):
        return True

    if rc == 0:
        # rc=0 with nothing useful on either pipe: optimistic default,
        # the user is probably authenticated and the CLI just stayed
        # quiet (e.g. a future release).
        if not combined.strip():
            return True
        return False
    return False


async def probe_codex_availability() -> dict[str, Any]:
    """Return the full status payload consumed by ``GET /api/codex/status``.

    Returns a dict with keys:

    * ``installed`` (bool) -- True iff Studio can actually drive Codex
      end-to-end: BOTH the Python SDK (for chat) AND a `codex`
      executable on PATH (for the device-auth login flow). The
      canonical `openai-codex` package depends on `openai-codex-cli-bin`
      which installs the `codex` shim into the venv's `bin/`, so the
      common SDK-only install in fact gets the CLI on PATH for free
      and this gate triggers correctly. Hosts that import the SDK
      from a wheel without that runtime dep stay hidden because the
      login flow would otherwise fail with "codex CLI not found on
      PATH" after the user clicked Sign in.
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
        # Gate on BOTH because the login flow shells out to `codex`.
        # Round 5 briefly set this to `sdk_ok` alone, but round 6
        # caught that the login route would then fail with
        # `codex CLI not found on PATH` after the user clicked
        # Sign in, leaving them with an unusable provider row.
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
