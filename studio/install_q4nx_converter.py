#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Install the pinned FLM Q4NX converter the first time an NPU export asks for it.

The converter is pure Python and runs in Studio's own environment (torch, gguf, einops,
safetensors), so this only fetches and verifies its source archive at a pinned commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path
from typing import Optional

_STUDIO_DIR = os.path.dirname(os.path.abspath(__file__))
if _STUDIO_DIR not in sys.path:
    sys.path.insert(0, _STUDIO_DIR)

import prebuilt_core as core  # noqa: E402
from backend.utils.auth_safe import auth_safe_open  # noqa: E402

PINS_PATH = Path(_STUDIO_DIR) / "q4nx_converter_pins.json"
MARKER_NAME = "UNSLOTH_Q4NX_CONVERTER_INFO.json"
_CHUNK = 1 << 20
_DOWNLOAD_TIMEOUT_S = 60


class Q4nxInstallCancelled(RuntimeError):
    pass


def load_pins(path: Path = PINS_PATH) -> dict:
    with open(path, encoding = "utf-8") as handle:
        pins = json.load(handle)
    if pins.get("schema_version") != 1:
        raise RuntimeError(f"{path} has an unsupported schema_version.")
    return pins


def archive_url(pins: dict) -> str:
    return f"https://codeload.github.com/{pins['repo']}/tar.gz/{pins['commit']}"


def install_dir(root: Path, pins: Optional[dict] = None) -> Path:
    pins = pins or load_pins()
    return Path(root) / pins["commit"][:12]


def installed_converter(root: Path, pins: Optional[dict] = None) -> Optional[Path]:
    """convert.py of a complete install matching the pin, else None."""
    pins = pins or load_pins()
    target = install_dir(root, pins)
    try:
        marker = json.loads((target / MARKER_NAME).read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    if marker.get("commit") != pins["commit"] or marker.get("sha256") != pins["archive_sha256"]:
        return None
    script = target / "convert.py"
    return script if script.is_file() and (target / "q4nx").is_dir() else None


def _download(
    url: str, destination: Path, *, expected_sha256: str, cancel: Optional[threading.Event]
) -> None:
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers = {"User-Agent": "unsloth-studio"})
    with (
        auth_safe_open(request, timeout = _DOWNLOAD_TIMEOUT_S) as response,
        open(destination, "wb") as handle,
    ):
        while True:
            if cancel is not None and cancel.is_set():
                raise Q4nxInstallCancelled("Q4NX converter download cancelled.")
            chunk = response.read(_CHUNK)
            if not chunk:
                break
            handle.write(chunk)
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise core.ReleaseIntegrityError(
            f"{url} has sha256 {actual}, expected {expected_sha256}; refusing to install it."
        )


def install(
    root: Path,
    *,
    cancel: Optional[threading.Event] = None,
    pins_path: Path = PINS_PATH,
    url: Optional[str] = None,
) -> Path:
    """Install the pinned converter under root/<commit>, reusing a matching install."""
    pins = load_pins(pins_path)
    root = Path(root)
    target = install_dir(root, pins)
    with core.install_lock(core.install_lock_path(target)):
        existing = installed_converter(root, pins)
        if existing is not None:
            return existing
        root.mkdir(parents = True, exist_ok = True)
        with tempfile.TemporaryDirectory(prefix = ".q4nx-install-", dir = root) as scratch:
            scratch_dir = Path(scratch)
            archive = scratch_dir / "converter.tar.gz"
            _download(
                url or archive_url(pins),
                archive,
                expected_sha256 = pins["archive_sha256"],
                cancel = cancel,
            )
            extracted = scratch_dir / "extracted"
            extracted.mkdir()
            core.extract_archive(archive, extracted)
            scripts = sorted(extracted.rglob("convert.py"))
            if not scripts:
                raise RuntimeError("The Q4NX converter archive does not contain convert.py.")
            payload = scripts[0].parent
            (payload / MARKER_NAME).write_text(
                json.dumps(
                    {"repo": pins["repo"], "commit": pins["commit"], "sha256": pins["archive_sha256"]},
                    indent = 2,
                )
                + "\n",
                encoding = "utf-8",
            )
            if target.exists():
                shutil.rmtree(target)
            os.replace(payload, target)
    script = installed_converter(root, pins)
    if script is None:
        raise RuntimeError(f"Q4NX converter install at {target} did not verify.")
    return script


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description = __doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type = Path,
        default = Path.home() / ".unsloth" / "studio" / "q4nx_converter",
        help = "Directory that holds one subdirectory per pinned converter commit.",
    )
    args = parser.parse_args(argv)
    print(install(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
