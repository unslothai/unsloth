# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Load PDF parsers on first use, keeping downloaded wheels outside the running environment.

Call from a document worker. Existing imports take precedence; cached wheels are appended to
sys.path without processing .pth files. A published cache contains complete upstream wheels,
including their metadata and licenses, and never replaces an installed native package.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
from importlib.metadata import PackageNotFoundError, distributions, version
import json
import io
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from types import ModuleType

from filelock import FileLock
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from utils.paths.storage_roots import cache_root
from utils.subprocess_compat import windows_hidden_subprocess_kwargs

_REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements" / "pdf.txt"
_MODULES = frozenset({"fitz", "pymupdf", "pymupdf4llm"})
_LOCK = threading.RLock()
_INSTALL_TIMEOUT = 120
_CACHE_VERSION = 1
INSTALL_COMMAND = 'python -m pip install "unsloth[pdf]"'
logger = logging.getLogger(__name__)

# Transport settings, not application credentials or package-source/target overrides.
_INSTALL_ENV_KEYS = frozenset(
    {
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
)


class PDFDependencyUnavailable(ImportError):
    """A PDF dependency could not be loaded; includes the supported manual installation path."""


def _unavailable(reason: str) -> PDFDependencyUnavailable:
    return PDFDependencyUnavailable(
        f"{reason}. Install PDF support in the Unsloth environment with: {INSTALL_COMMAND} "
        f'(or: uv pip install --python "{sys.executable}" "unsloth[pdf]").'
    )


def _requirements() -> list[Requirement]:
    return [
        requirement
        for raw in _REQUIREMENTS.read_text(encoding = "utf-8").splitlines()
        if (line := raw.split("#", 1)[0].strip())
        if (requirement := Requirement(line)).marker is None or requirement.marker.evaluate()
    ]


def _target() -> Path:
    identity = {
        "cache_version": _CACHE_VERSION,
        "requirements": _REQUIREMENTS.read_text(encoding = "utf-8"),
        "prefix": str(Path(sys.prefix).resolve()),
        "executable": str(Path(sys.executable).resolve()),
        "python": list(sys.version_info[:3]),
        "implementation": sys.implementation.name,
        "cache_tag": sys.implementation.cache_tag,
        "platform": sysconfig.get_platform(),
        "machine": platform.machine(),
        "soabi": sysconfig.get_config_var("SOABI"),
        "gil_disabled": sysconfig.get_config_var("Py_GIL_DISABLED"),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys = True).encode()).hexdigest()[:24]
    return cache_root() / "optional-pdf" / digest


def _validate_target(target: Path, packages: list[str]) -> None:
    installed = {
        canonicalize_name(dist.metadata["Name"]): dist for dist in distributions(path = [str(target)])
    }
    for entry in packages:
        requirement = Requirement(entry)
        dist = installed.get(canonicalize_name(requirement.name))
        if dist is None or dist.version not in requirement.specifier:
            raise _unavailable("The PDF wheel cache is incomplete or damaged")
        record = dist.read_text("RECORD")
        if not record:
            raise _unavailable("The PDF wheel cache is incomplete or damaged")
        # Distribution.files can omit missing entries; read RECORD directly to detect damage.
        for row in csv.reader(io.StringIO(record)):
            if not row:
                continue
            path = Path(dist.locate_file(row[0])).resolve()
            if path.is_relative_to(target.resolve()) and not path.is_file():
                raise _unavailable("The PDF wheel cache is incomplete or damaged")


def _activate(target: Path) -> bool:
    marker = target / ".complete"
    if not marker.is_file():
        return False
    try:
        record = json.loads(marker.read_text(encoding = "utf-8"))
        if record["version"] != _CACHE_VERSION or not record["packages"]:
            raise ValueError("Invalid completion marker")
        _validate_target(target, record["packages"])
    except PDFDependencyUnavailable:
        raise
    except (OSError, ValueError, TypeError, KeyError, csv.Error) as exc:
        raise _unavailable("The PDF wheel cache is incomplete or damaged") from exc
    path = str(target)
    if path not in sys.path:
        sys.path.append(path)
    importlib.invalidate_caches()
    return True


def _missing_requirements() -> list[str]:
    missing = []
    for requirement in _requirements():
        try:
            installed = version(requirement.name)
        except PackageNotFoundError:
            missing.append(f"{requirement.name}{requirement.specifier}")
        else:
            # pymupdf4llm declares tabulate without a version floor. The download pin is not
            # a compatibility restriction on a formatter another feature may have loaded.
            if (
                canonicalize_name(requirement.name) != "tabulate"
                and installed not in requirement.specifier
            ):
                raise _unavailable(
                    f"PDF support needs {requirement.name}{requirement.specifier}; "
                    f"this environment has {installed}, which will not be replaced at runtime"
                )
    return missing


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("PDF dependency installation exceeded its time budget")
    return remaining


def _find_uv() -> str | None:
    try:
        from utils.mlx_repair import _uv_executable
        return _uv_executable()
    except Exception:
        return shutil.which("uv")


def _install(target: Path, deadline: float) -> None:
    packages = _missing_requirements()
    if not packages:
        raise _unavailable(
            "The installed PDF parser could not be imported; repair the installation"
        )
    pip = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-user"]
    uv = _find_uv()
    commands = (
        [[uv, "pip", "install", "--python", sys.executable, "--no-config"]] if uv else []
    ) + [pip]
    env = {key: value for key, value in os.environ.items() if key in _INSTALL_ENV_KEYS}
    env["PIP_CONFIG_FILE"] = os.devnull
    logger.info("Installing optional PDF support on first use")
    for index, installer in enumerate(commands):
        with tempfile.TemporaryDirectory(prefix = ".pdf-", dir = target.parent) as staging:
            stage = Path(staging)
            command = installer + [
                "--target",
                str(stage),
                "--no-deps",
                "--only-binary",
                ":all:",
                "--index-url",
                "https://pypi.org/simple",
                *packages,
            ]
            try:
                subprocess.run(
                    command,
                    check = True,
                    capture_output = True,
                    timeout = _remaining(deadline),
                    env = env,
                    **windows_hidden_subprocess_kwargs(),
                )
            except (OSError, subprocess.CalledProcessError):
                if index == len(commands) - 1:
                    raise
                logger.info("PDF wheel installation with uv failed; trying this interpreter's pip")
                continue
            _validate_target(stage, packages)
            (stage / ".complete").write_text(
                json.dumps({"version": _CACHE_VERSION, "packages": packages}), encoding = "utf-8"
            )
            os.replace(stage, target)
            return


def _import(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name in _MODULES | {"tabulate"}:
            return None
        raise _unavailable(
            "An installed PDF parser has a missing dependency; repair the installation"
        ) from exc
    except (ImportError, OSError) as exc:
        raise _unavailable(
            "An installed PDF parser cannot be loaded; repair the installation"
        ) from exc


def load_pdf_module(name: str) -> ModuleType:
    """Reuse installed/cached parsers or install missing wheels once, within a bounded wait."""
    if name not in _MODULES:
        raise ValueError(f"Not an optional PDF module: {name}")
    module = _import(name)
    if module is not None:
        return module
    deadline = time.monotonic() + _INSTALL_TIMEOUT
    acquired = False
    try:
        acquired = _LOCK.acquire(timeout = _remaining(deadline))
        if not acquired:
            raise TimeoutError("Waiting for PDF dependency installation timed out")
        module = _import(name)
        if module is not None:
            return module
        target = _target()
        if not _activate(target):
            if os.environ.get("UNSLOTH_PDF_AUTO_INSTALL", "1").strip().lower() in {
                "0",
                "false",
                "no",
                "off",
            }:
                raise _unavailable(
                    "PDF support is not installed and automatic downloads are disabled"
                )
            if any(
                os.environ.get(key, "").strip().lower() in {"1", "true", "yes", "on"}
                for key in (
                    "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                    "UV_OFFLINE",
                    "UV_NO_INDEX",
                    "PIP_NO_INDEX",
                )
            ):
                raise _unavailable("PDF support is not installed and offline mode is enabled")
            target.parent.mkdir(parents = True, exist_ok = True)
            with FileLock(str(target) + ".lock", timeout = _remaining(deadline)):
                if not _activate(target):
                    if target.exists():
                        raise _unavailable(
                            "The PDF wheel cache is incomplete; repair the installation"
                        )
                    _install(target, deadline)
                    _activate(target)
        module = _import(name)
        if module is None:
            raise _unavailable("Cached PDF support could not be imported; repair the installation")
        return module
    except PDFDependencyUnavailable:
        raise
    except Exception as exc:
        # Package-manager output can contain proxy credentials; only expose our own guidance.
        raise _unavailable("Automatic PDF installation failed or timed out") from exc
    finally:
        if acquired:
            _LOCK.release()
