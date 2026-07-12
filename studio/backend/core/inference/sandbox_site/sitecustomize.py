# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sandbox-side compatibility shim for ChatGPT code-interpreter paths.

Models trained on code-interpreter transcripts habitually write to /mnt/data
(or /mnt/outputs, /home/sandbox, /workspace), none of which exist in the
Studio sandbox. This module sits on the sandbox subprocess PYTHONPATH (see
``tools._build_safe_env``), so Python's site machinery imports it at
interpreter startup in every sandboxed ``python`` tool run AND in any Python
the ``terminal`` tool launches (the env is inherited).

It remaps those prefixes onto the working directory in ``open()`` /
``io.open()``, ``os.makedirs()`` and ``os.mkdir()`` / ``pathlib.Path.mkdir()``:
the calls model-written file code funnels through. ``io.open`` is patched
alongside ``builtins.open`` because ``pathlib.Path.open`` (and therefore
``read_text`` / ``write_text`` / ``read_bytes`` / ``write_bytes``) calls
``io.open`` directly, bypassing the builtins patch. ``os.mkdir`` is patched
alongside ``os.makedirs`` because ``pathlib.Path.mkdir(parents=True)`` -- the
stock ``Path('/mnt/data').mkdir(parents=True, exist_ok=True)`` setup line --
calls ``os.mkdir`` per component, not ``os.makedirs``; ``Path.mkdir`` itself is
patched too so its ``exist_ok`` retry (which probes the intentionally unpatched
``os.stat`` via ``Path.is_dir``) sees the mapped path and stays idempotent. A
one-line notice is printed to
stderr on the first remap so the model learns the real location. Everything
else (C-level opens, os.listdir, shutil metadata calls) is intentionally NOT
patched; those failures are handled by the model-visible retry hint appended
to the tool result. Failures here must never break the interpreter: the whole
setup is wrapped in try/except.

Identical with and without output streaming because the child env is.
"""

import builtins
import io
import os
import sys

# Prefixes that never exist on the sandbox host, so remapping is always safe.
_PREFIXES = ("/mnt/data", "/mnt/outputs", "/home/sandbox", "/workspace")
# /tmp exists on the host and model code can legitimately create /tmp/outputs
# (e.g. Path('/tmp/outputs').mkdir() or a subprocess mkdir that bypasses these
# in-process patches). Remap it only while it is absent -- healing the
# code-interpreter habit into the per-conversation workdir so the files are
# preserved/served and isolated -- and pass a real /tmp/outputs straight
# through so we never shadow a directory the user's own code created.
_CONDITIONAL_PREFIXES = ("/tmp/outputs",)
_notified = False


def _map_onto_cwd(prefix, text):
    """Map ``<prefix>/rest`` onto ``./rest`` in the CWD and note it once."""
    global _notified
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


def _remap(path):
    """Map ``<prefix>/rest`` onto ``./rest`` in the CWD; other paths pass through."""
    try:
        text = os.fspath(path)
    except TypeError:
        return path
    if not isinstance(text, str):
        return path
    for prefix in _PREFIXES:
        if text == prefix or text.startswith(prefix + "/"):
            return _map_onto_cwd(prefix, text)
    for prefix in _CONDITIONAL_PREFIXES:
        # Only heal the habit path while the real directory is absent, so a
        # /tmp/outputs the user code actually created is never shadowed.
        if (text == prefix or text.startswith(prefix + "/")) and not os.path.exists(prefix):
            return _map_onto_cwd(prefix, text)
    return path


def _install():
    import pathlib

    original_open = builtins.open
    original_io_open = io.open
    original_makedirs = os.makedirs
    original_mkdir = os.mkdir
    original_path_mkdir = pathlib.Path.mkdir

    def _open(file, *args, **kwargs):
        return original_open(_remap(file), *args, **kwargs)

    def _io_open(file, *args, **kwargs):
        return original_io_open(_remap(file), *args, **kwargs)

    def _makedirs(name, *args, **kwargs):
        return original_makedirs(_remap(name), *args, **kwargs)

    def _mkdir(path, *args, **kwargs):
        return original_mkdir(_remap(path), *args, **kwargs)

    def _path_mkdir(self, *args, **kwargs):
        # pathlib drives mkdir through os.mkdir and, on FileExistsError, probes
        # Path.is_dir()/os.stat (intentionally unpatched). So a bare os.mkdir
        # remap would still raise on the stock
        # Path('/mnt/data').mkdir(parents=True, exist_ok=True) whenever the
        # mapped target already exists. Remap the receiver to a real Path up
        # front so the whole parents/exist_ok dance runs against the mapped
        # location and stays idempotent.
        mapped = _remap(self)
        target = self if mapped is self else self.__class__(mapped)
        return original_path_mkdir(target, *args, **kwargs)

    builtins.open = _open
    # pathlib.Path.open / write_text / read_text call io.open directly, not
    # the builtins binding, so the remap must be installed on both.
    io.open = _io_open
    os.makedirs = _makedirs
    # pathlib.Path.mkdir(parents=True) calls os.mkdir (not os.makedirs) per
    # missing component, so patch os.mkdir with the same remap; patch
    # Path.mkdir itself too so exist_ok/parents land on the mapped path.
    os.mkdir = _mkdir
    pathlib.Path.mkdir = _path_mkdir


try:
    _install()
except Exception:  # noqa: BLE001 - a broken shim must never break user code
    pass
