# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Native transport: llama.cpp's own mutual TLS 1.3 RPC."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from .paths import run_dir

TLS_FLAG = "--tls-cert"
DISABLE_ENV = "UNSLOTH_CLUSTER_NATIVE_TLS"  # "0" keeps every link on the relay
TLS_SCHEME = "tls://"

_PROBED: Dict[Tuple[str, float], bool] = {}


def disabled() -> bool:
    return (os.environ.get(DISABLE_ENV) or "").strip() == "0"


def binary_supports_tls(binary: Optional[str]) -> bool:
    """Whether this ggml-rpc-server (and its ggml-rpc library) was built with the TLS transport."""
    if not binary:
        return False
    try:
        key = (str(Path(binary).resolve()), Path(binary).stat().st_mtime)
    except OSError:
        return False
    if key not in _PROBED:
        try:
            out = subprocess.run(
                [binary, "--help"],
                capture_output = True,
                text = True,
                timeout = 5,
                encoding = "utf-8",
                errors = "replace",
            )
            _PROBED[key] = TLS_FLAG in (out.stdout or "") + (out.stderr or "")
        except (OSError, subprocess.SubprocessError):
            _PROBED[key] = False
    return _PROBED[key]


def endpoint(fingerprint: str, host: str, port: int) -> str:
    return f"{TLS_SCHEME}{fingerprint}@{host}:{int(port)}"


def allowlist_path(port: int) -> Path:
    return run_dir() / f"rpc-allow-{int(port)}.txt"


def write_allowlist(path: Path, fingerprints: Iterable[str]) -> None:
    """Atomically: rpc-server re-reads the file on every accept and must never see half of it."""
    tmp = path.with_name(path.name + ".tmp")
    lines = sorted({str(f).strip().lower() for f in fingerprints if str(f).strip()})
    tmp.write_text("".join(f"{line}\n" for line in lines), encoding = "utf-8")
    os.replace(tmp, path)
