# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Install and remove the computer use add-on. Only the owner's click in Settings calls this."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from filelock import FileLock, Timeout

from loggers import get_logger

from . import ADDON_PACKAGE, ADDON_VERSION, addon_root, detach, installed

logger = get_logger(__name__)

DISTRIBUTION = "unsloth-computer-use"
# The published wheel's sha256. Nothing is downloaded without it: a version number alone would
# trust whoever holds the name on the index.
ADDON_SHA256 = ""
_INSTALL_TIMEOUT_S = 180
_LOCK_TIMEOUT_S = 200
_OFFLINE = ("UV_OFFLINE", "UV_NO_INDEX", "PIP_NO_INDEX")


class AddonInstallError(RuntimeError):
    """Its message is shown to the user, so it never carries installer output."""


def _checkout_source() -> Optional[Path]:
    """The add-on's source beside this backend in a git checkout. An installed wheel has none."""
    source = Path(__file__).resolve().parents[4] / "addons" / "computer_use" / ADDON_PACKAGE
    return source if (source / "tool.py").is_file() else None


def _download_command(staging: Path, requirements: Path) -> list[str]:
    from utils.s3_runtime import _find_uv

    uv = _find_uv()
    command = (
        [uv, "pip", "install", "--python", sys.executable, "--no-config"]
        if uv
        else [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-user"]
    )
    requirements.write_text(
        f"{DISTRIBUTION}=={ADDON_VERSION} --hash=sha256:{ADDON_SHA256}\n", encoding = "utf-8"
    )
    # A wheel only, with no dependencies: nothing from the index runs during the install.
    return command + [
        "--require-hashes",
        "--target",
        str(staging),
        "--no-deps",
        "--only-binary",
        ":all:",
        "--index-url",
        "https://pypi.org/simple",
        "--requirement",
        str(requirements),
    ]


def _download(staging: Path) -> None:
    from utils.s3_runtime import _INSTALL_ENV_KEYS
    from utils.subprocess_compat import windows_hidden_subprocess_kwargs
    from utils.utils import hf_env_offline

    if not ADDON_SHA256:
        raise AddonInstallError("Computer use can't be installed from this build of Unsloth yet.")
    if hf_env_offline() or any(
        os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"} for name in _OFFLINE
    ):
        raise AddonInstallError("Computer use can't be installed while offline mode is on.")
    with tempfile.TemporaryDirectory(prefix = "unsloth-computer-use-") as scratch:
        # TLS and proxy variables only: no credentials, index redirects or package overrides.
        env = {key: value for key, value in os.environ.items() if key in _INSTALL_ENV_KEYS}
        env["PIP_CONFIG_FILE"] = os.devnull
        try:
            subprocess.run(
                _download_command(staging, Path(scratch) / "requirements.txt"),
                check = True,
                capture_output = True,
                timeout = _INSTALL_TIMEOUT_S,
                env = env,
                **windows_hidden_subprocess_kwargs(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            # The output can hold proxy credentials, so only its kind is logged.
            logger.warning("computer use add-on download failed: %s", type(exc).__name__)
            raise AddonInstallError(
                "Computer use could not be downloaded. Check the internet connection and try again."
            ) from None


def install() -> None:
    """Idempotent. Staged beside its final place, then renamed, so a half install is never loaded."""
    if installed():
        return
    root = addon_root()
    try:
        root.mkdir(parents = True, exist_ok = True)
        with FileLock(str(root / ".install.lock"), timeout = _LOCK_TIMEOUT_S):
            if installed():
                return
            staging = Path(tempfile.mkdtemp(prefix = ".staging-", dir = root))
            try:
                source = _checkout_source()
                if source is not None:
                    shutil.copytree(
                        source,
                        staging / ADDON_PACKAGE,
                        ignore = shutil.ignore_patterns("__pycache__"),
                    )
                else:
                    _download(staging)
                if not (staging / ADDON_PACKAGE / "tool.py").is_file():
                    raise AddonInstallError("The computer use download was incomplete. Try again.")
                _remove_versions(root, keep = staging)
                os.replace(staging, root / ADDON_VERSION)
            finally:
                shutil.rmtree(staging, ignore_errors = True)
    except (OSError, Timeout) as exc:
        logger.warning("computer use add-on install failed: %s", exc)
        raise AddonInstallError(
            "Computer use could not be installed in the Unsloth folder."
        ) from None
    if not installed():
        raise AddonInstallError("Computer use was installed but could not be loaded.")
    logger.info("computer use add-on %s installed", ADDON_VERSION)


def _remove_versions(root: Path, keep: Optional[Path] = None) -> None:
    for entry in list(root.iterdir()):
        if not entry.is_dir() or entry.is_symlink() or entry == keep:
            continue
        if entry.name.startswith(".removed-"):
            shutil.rmtree(entry, ignore_errors = True)
            continue
        # Renamed aside first: a delete that stops halfway must not leave a version that loads.
        aside = Path(tempfile.mkdtemp(prefix = ".removed-", dir = root))
        os.replace(entry, aside / entry.name)
        shutil.rmtree(aside, ignore_errors = True)


def uninstall() -> None:
    root = addon_root()
    if not root.is_dir():
        return
    try:
        with FileLock(str(root / ".install.lock"), timeout = _LOCK_TIMEOUT_S):
            detach()
            _remove_versions(root)
    except (OSError, Timeout) as exc:
        logger.warning("computer use add-on removal failed: %s", exc)
        raise AddonInstallError(
            "Computer use could not be removed from the Unsloth folder."
        ) from None
    logger.info("computer use add-on removed")
