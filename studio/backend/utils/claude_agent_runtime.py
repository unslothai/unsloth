# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Repair a missing Claude Agent SDK (installs older than its requirements entry) from its sdist."""

import importlib
from importlib.metadata import distributions
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import sysconfig
import tempfile
import time

from filelock import FileLock

from utils.s3_runtime import _INSTALL_ENV_KEYS, _find_uv
from utils.subprocess_compat import windows_hidden_subprocess_kwargs
from utils.utils import hf_env_offline

SDK_PACKAGE = "claude-agent-sdk"
SDK_REQUIREMENT = f"{SDK_PACKAGE}==0.2.178"
# Every wheel bundles a ~150 MB Claude Code CLI that Studio never runs, so only the sdist is allowed.
SDK_SOURCE_ONLY = ("--no-binary", SDK_PACKAGE)
MANUAL_COMMAND = (
    f'uv pip install --python "{sys.executable}" --no-binary {SDK_PACKAGE} "{SDK_REQUIREMENT}"'
)
_INSTALL_SECONDS = 300
_URL_USERINFO = re.compile(r"(\b[a-z][a-z0-9+.-]*://)[^/\s@'\"]+@", re.IGNORECASE)
logger = logging.getLogger(__name__)


class ClaudeAgentSdkUnavailable(RuntimeError):
    pass


def _failure(reason: str) -> ClaudeAgentSdkUnavailable:
    return ClaudeAgentSdkUnavailable(f"{reason} Install it by hand with: {MANUAL_COMMAND}")


def _import_sdk():
    try:
        return importlib.import_module("claude_agent_sdk")
    except ModuleNotFoundError as exc:
        if exc.name == "claude_agent_sdk":
            return None
        raise _failure(f"The Claude Agent SDK is installed but did not import ({exc}).") from exc
    except ImportError as exc:
        raise _failure(f"The Claude Agent SDK is installed but did not import ({exc}).") from exc


def _installer_reason(exc: subprocess.CalledProcessError) -> str:
    output = exc.stderr or exc.stdout or b""
    if isinstance(output, bytes):
        output = output.decode("utf-8", "replace")
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    # Installer output can carry proxy credentials in URLs.
    tail = _URL_USERINFO.sub(r"\1", " ".join(lines[-3:]))[-500:]
    return (
        f"The installer failed: {tail}"
        if tail
        else f"The installer exited with code {exc.returncode}."
    )


def _installed_pins() -> str:
    # Only the environment uv installs into; a stale or shadowing copy elsewhere would pin a package twice.
    paths = list(dict.fromkeys(sysconfig.get_paths()[key] for key in ("purelib", "platlib")))
    versions: dict[str, set[str]] = {}
    names: dict[str, str] = {}
    for dist in distributions(path = paths):
        name = dist.metadata["Name"]
        if name:
            key = re.sub(r"[-_.]+", "-", name).lower()
            names.setdefault(key, name)
            versions.setdefault(key, set()).add(dist.version)
    return "".join(
        f"{names[key]}=={next(iter(found))}\n" for key, found in versions.items() if len(found) == 1
    )


def ensure_claude_agent_sdk():
    sdk = _import_sdk()
    if sdk is not None:
        return sdk
    if hf_env_offline() or any(
        os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
        for name in ("UV_OFFLINE", "UV_NO_INDEX", "PIP_NO_INDEX")
    ):
        raise _failure("Studio is in offline mode, so it cannot download the Claude Agent SDK.")
    deadline = time.monotonic() + _INSTALL_SECONDS
    try:
        with FileLock(
            str(Path(sys.prefix) / ".unsloth-claude-agent-install.lock"), timeout = _INSTALL_SECONDS
        ):
            importlib.invalidate_caches()
            sdk = _import_sdk()
            if sdk is not None:
                return sdk
            logger.info("Installing the Claude Agent SDK")
            with tempfile.TemporaryDirectory(prefix = "unsloth-claude-agent-") as staging:
                # Never upgrade a package Studio already runs on.
                constraints = Path(staging) / "constraints.txt"
                constraints.write_text(_installed_pins(), encoding = "utf-8")
                uv = _find_uv()
                command = (
                    [uv, "pip", "install", "--python", sys.executable, "--no-config"]
                    if uv
                    else [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--disable-pip-version-check",
                        "--no-user",
                    ]
                )
                command += [
                    "--only-binary",
                    ":all:",
                    *SDK_SOURCE_ONLY,
                    "--constraint",
                    str(constraints),
                    "--index-url",
                    "https://pypi.org/simple",
                    SDK_REQUIREMENT,
                ]
                env = {key: value for key, value in os.environ.items() if key in _INSTALL_ENV_KEYS}
                env["PIP_CONFIG_FILE"] = os.devnull
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, _INSTALL_SECONDS)
                subprocess.run(
                    command,
                    check = True,
                    capture_output = True,
                    timeout = remaining,
                    env = env,
                    **windows_hidden_subprocess_kwargs(),
                )
            importlib.invalidate_caches()
            sdk = _import_sdk()
            if sdk is None:
                raise _failure("The installer finished but the Claude Agent SDK is still missing.")
            return sdk
    except ClaudeAgentSdkUnavailable:
        raise
    except subprocess.CalledProcessError as exc:
        raise _failure(_installer_reason(exc)) from exc
    except (subprocess.TimeoutExpired, TimeoutError) as exc:
        raise _failure("The install timed out.") from exc
    except Exception as exc:
        raise _failure("The install could not start.") from exc
