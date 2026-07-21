# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Prefixed logger shared by the prebuilt installers.

Each installer calls `configure(prefix)` at import so shared plumbing logs under
that component's prefix (e.g. "[whisper-prebuilt] "). `set_to_stdout` flips the
stream: setup wants progress on stdout, the read-only resolver keeps it on stderr.
"""

from __future__ import annotations

import sys

_prefix = "[prebuilt] "
_to_stdout = False


def configure(prefix: str) -> None:
    global _prefix
    _prefix = prefix


def set_to_stdout(value: bool) -> None:
    global _to_stdout
    _to_stdout = bool(value)


def log(message: str) -> None:
    print(f"{_prefix}{message}", file = sys.stdout if _to_stdout else sys.stderr)
