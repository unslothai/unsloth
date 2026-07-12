# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sandbox-side compatibility shim for ChatGPT code-interpreter paths.

Models trained on code-interpreter transcripts habitually write to /mnt/data
(or /mnt/outputs, /home/sandbox, /workspace), none of which exist in the
Studio sandbox. This module sits on the sandbox subprocess PYTHONPATH (see
``tools._build_safe_env``), so Python's site machinery imports it at
interpreter startup in every sandboxed ``python`` tool run AND in any Python
the ``terminal`` tool launches (the env is inherited).

It remaps those prefixes onto the working directory in ``open()`` and
``os.makedirs()`` only: the two calls model-written file code funnels through
(``pathlib`` I/O also lands in ``open``). A one-line notice is printed to
stderr on the first remap so the model learns the real location. Everything
else (C-level opens, os.listdir, shutil metadata calls) is intentionally NOT
patched; those failures are handled by the model-visible retry hint appended
to the tool result. Failures here must never break the interpreter: the whole
setup is wrapped in try/except.

Identical with and without output streaming because the child env is.
"""

import builtins
import os
import sys

_PREFIXES = ("/mnt/data", "/mnt/outputs", "/home/sandbox", "/workspace")
_notified = False


def _remap(path):
    """Map ``<prefix>/rest`` onto ``./rest`` in the CWD; other paths pass through."""
    global _notified
    try:
        text = os.fspath(path)
    except TypeError:
        return path
    if not isinstance(text, str):
        return path
    for prefix in _PREFIXES:
        if text == prefix or text.startswith(prefix + "/"):
            rel = text[len(prefix) :].lstrip("/")
            mapped = os.path.join(os.getcwd(), rel) if rel else os.getcwd()
            if not _notified:
                _notified = True
                print(
                    f"note: {prefix} does not exist in this sandbox; "
                    f"using the working directory instead ({text} -> {mapped})",
                    file = sys.stderr,
                )
            return mapped
    return path


def _install():
    original_open = builtins.open
    original_makedirs = os.makedirs

    def _open(file, *args, **kwargs):
        return original_open(_remap(file), *args, **kwargs)

    def _makedirs(name, *args, **kwargs):
        return original_makedirs(_remap(name), *args, **kwargs)

    builtins.open = _open
    os.makedirs = _makedirs


try:
    _install()
except Exception:  # noqa: BLE001 - a broken shim must never break user code
    pass
