# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Install the PDF parsers into this environment the first time a PDF or MuPDF e-book is read.

Only missing pinned wheels are installed, with --no-deps, so nothing already installed is replaced.
They land in this interpreter's site-packages, like the S3 SDK, so every interpreter Studio starts
(document workers, the chat PDF export, the Python tool under its OS sandbox) can import them.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import importlib.util
from importlib.metadata import PackageNotFoundError, version
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from types import ModuleType
from typing import Callable, Iterator

from filelock import FileLock
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from utils.subprocess_compat import windows_hidden_subprocess_kwargs

_REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements" / "pdf.txt"
_MODULES = frozenset({"fitz", "pymupdf", "pymupdf4llm"})
_LOCK = threading.RLock()
_INSTALL_TIMEOUT = 120
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


def _missing_requirements() -> list[str]:
    missing = []
    for requirement in _requirements():
        try:
            installed = version(requirement.name)
        except PackageNotFoundError:
            missing.append(f"{requirement.name}{requirement.specifier}")
        else:
            # pymupdf4llm sets no tabulate floor, so any installed tabulate is accepted, not just the pin.
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


def _downloads_blocked() -> str | None:
    if os.environ.get("UNSLOTH_PDF_AUTO_INSTALL", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return "PDF support is not installed and automatic downloads are disabled"
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
        return "PDF support is not installed and offline mode is enabled"
    return None


def _install(deadline: float) -> None:
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
        command = installer + [
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
        importlib.invalidate_caches()
        return


def _lock_path() -> Path:
    return Path(sys.prefix) / ".unsloth-pdf-install.lock"


@contextmanager
def _settled(deadline: float) -> Iterator[None]:
    """Hold the install locks: an import or check made mid-install could cache a half-written package."""
    if not _LOCK.acquire(timeout = _remaining(deadline)):
        raise TimeoutError("Waiting for PDF dependency installation timed out")
    try:
        lock = _lock_path()
        # Another process (a recipe worker, a chat export) may be installing into the same environment.
        # Unwritable means nothing installs here; filelock would wait out its timeout (it reads Windows
        # EACCES as held), and os.access ignores Windows ACLs.
        try:
            with open(lock, "a", encoding = "utf-8"):
                writable = True
        except OSError:
            writable = False
        if not writable:
            yield
            return
        with FileLock(str(lock), timeout = _remaining(deadline)):
            importlib.invalidate_caches()
            yield
    finally:
        _LOCK.release()


def _install_unless(present: Callable[[], bool]) -> None:
    """Install missing PDF wheels once per environment; ``present`` is only checked under the locks."""
    deadline = time.monotonic() + _INSTALL_TIMEOUT
    try:
        with _settled(deadline):
            if present():
                return
            blocked = _downloads_blocked()
            if blocked:
                raise _unavailable(blocked)
            _install(deadline)
    except PDFDependencyUnavailable:
        raise
    except Exception as exc:
        # Package-manager output can contain proxy credentials; only expose our own guidance.
        raise _unavailable("Automatic PDF installation failed or timed out") from exc


def _import(name: str) -> ModuleType | None:
    try:
        # Keep each allowed import literal so the loader audit can verify its target.
        if name == "fitz":
            return importlib.import_module("fitz")
        if name == "pymupdf":
            return importlib.import_module("pymupdf")
        if name == "pymupdf4llm":
            return importlib.import_module("pymupdf4llm")
        raise ValueError(f"Not an optional PDF module: {name}")
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
    """Import a PDF parser, installing the missing wheels once within a bounded wait."""
    if name not in _MODULES:
        raise ValueError(f"Not an optional PDF module: {name}")
    if name not in sys.modules:
        _install_unless(lambda: _import(name) is not None)
    module = _import(name)
    if module is None:
        raise _unavailable("Installed PDF support could not be imported; repair the installation")
    return module


def _all_present() -> bool:
    # tabulate too: pymupdf4llm imports it, and find_spec does not run that import.
    return all(
        importlib.util.find_spec(name) is not None for name in sorted(_MODULES | {"tabulate"})
    )


def ensure_pdf_support() -> None:
    """Install PDF support for a child interpreter without importing it into this process."""
    _install_unless(_all_present)
    if not _all_present():
        raise _unavailable("Installed PDF support could not be found; repair the installation")
