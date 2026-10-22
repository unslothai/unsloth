# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Install the optional S3 SDK when an S3 dataset is first requested."""

import importlib
from importlib.metadata import distributions
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from filelock import FileLock

from utils.subprocess_compat import windows_hidden_subprocess_kwargs
from utils.utils import hf_env_offline

INSTALL_COMMAND = 'python -m pip install "unsloth[s3]"'
# Studio venvs are created by uv without pip, so also name the uv spelling.
S3_INSTALL_MESSAGE = (
    "S3 dataset loading requires boto3. Install it in the Unsloth environment with: "
    f'{INSTALL_COMMAND} (or: uv pip install --python "{sys.executable}" "unsloth[s3]")'
)
logger = logging.getLogger(__name__)
_INSTALL_ENV_KEYS = {
    "PATH",
    "HOME",
    "USERPROFILE",
    "SYSTEMROOT",
    "SystemRoot",
    "WINDIR",
    "TEMP",
    "TMP",
    "TMPDIR",
    "APPDATA",
    "LOCALAPPDATA",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "https_proxy",
    "http_proxy",
    "all_proxy",
    "no_proxy",
    "PIP_CERT",
    "PIP_CLIENT_CERT",
    "PIP_PROXY",
    "UV_NATIVE_TLS",
    "UV_SYSTEM_CERTS",
    "UV_CACHE_DIR",
}


class S3DependencyUnavailable(RuntimeError):
    """The SDK could not be installed or imported; includes manual guidance."""


def _find_uv():
    # Desktop GUI launches can start with a minimal PATH, and the pip fallback needs a pip the venv lacks.
    try:
        from utils.mlx_repair import _uv_executable
        return _uv_executable()
    except Exception:  # noqa: BLE001
        return shutil.which("uv")


def _import_sdk():
    try:
        return importlib.import_module("boto3")
    except ModuleNotFoundError as exc:
        if exc.name == "boto3":
            return None
        raise S3DependencyUnavailable(S3_INSTALL_MESSAGE) from exc
    except ImportError as exc:
        raise S3DependencyUnavailable(S3_INSTALL_MESSAGE) from exc


def ensure_boto3():
    """Reuse an installed SDK, or install it without upgrading shared packages.

    Call from a worker thread when used by an async route. The installer runs in
    a subprocess targeting this interpreter; importing the result stays in-process.
    """
    sdk = _import_sdk()
    if sdk is not None:
        return sdk
    if os.environ.get("UNSLOTH_S3_AUTO_INSTALL", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        raise S3DependencyUnavailable(
            f"Automatic S3 installation is disabled. {S3_INSTALL_MESSAGE}"
        )
    if hf_env_offline() or any(
        os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
        for name in ("UV_OFFLINE", "UV_NO_INDEX", "PIP_NO_INDEX")
    ):
        raise S3DependencyUnavailable(
            f"S3 SDK is not installed and offline mode is enabled. {S3_INSTALL_MESSAGE}"
        )
    try:
        # Includes lock contention: the UI's training-start request times out at 60 s.
        deadline = time.monotonic() + 40
        with FileLock(str(Path(sys.prefix) / ".unsloth-s3-install.lock"), timeout = 40):
            importlib.invalidate_caches()
            sdk = _import_sdk()
            if sdk is not None:
                return sdk
            logger.info("Installing optional S3 support on first use")
            with tempfile.TemporaryDirectory(prefix = "unsloth-s3-") as staging:
                constraints = Path(staging) / "constraints.txt"
                constraints.write_text(
                    "".join(
                        f"{dist.metadata['Name']}=={dist.version}\n" for dist in distributions()
                    ),
                    encoding = "utf-8",
                )
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
                    "--constraint",
                    str(constraints),
                    "--index-url",
                    "https://pypi.org/simple",
                    "boto3>=1.34.0",
                ]
                # Installer children need TLS/proxy settings, not AWS/HF credentials
                # or package-source/target overrides from the running Studio process.
                env = {key: value for key, value in os.environ.items() if key in _INSTALL_ENV_KEYS}
                env["PIP_CONFIG_FILE"] = os.devnull
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("S3 installation wait exceeded its time budget")
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
                raise S3DependencyUnavailable(S3_INSTALL_MESSAGE)
            return sdk
    except S3DependencyUnavailable:
        raise
    except Exception as exc:
        # Installer output can contain proxy credentials; expose only our guidance.
        raise S3DependencyUnavailable(
            f"Automatic S3 installation failed. {S3_INSTALL_MESSAGE}"
        ) from exc
