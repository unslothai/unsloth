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
the calls model-written file code funnels through. Prefix lists cannot enumerate
every path a model invents, though: models also hallucinate absolute paths from
seeing their own CWD (e.g. ``open('/home/ubuntu/Sandbox/flappy_bird.html',
'w')`` after the sandbox CWD is ``/home/ubuntu/studio_sandbox/<thread>``). So
``open()`` / ``io.open()`` gain a write-mode fallback on top of the prefix
remaps: when a write/create-mode open targets an absolute path that is NOT under
the CWD and whose parent directory does not exist, the file is redirected to the
basename in the CWD. Read modes never hit the fallback (reading a real system
file must fail or succeed truthfully); the prefix remaps still cover reads. The
fallback is deliberately NOT applied to ``os.mkdir`` / ``os.makedirs`` /
``pathlib.Path.mkdir``: making an arbitrary absolute directory can legitimately
succeed (e.g. under writable ``/home/ubuntu``) and must not be second-guessed.
``io.open`` is patched
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


def _note(subject, original, mapped):
    """Print the one-shot stderr notice so the model learns the real location.

    ``subject`` is what "does not exist" (the convention prefix, or the whole
    invented path for the write-mode fallback); ``original`` is the exact path
    the model passed, echoed in the ``(original -> mapped)`` tail.
    """
    global _notified
    if _notified:
        return
    _notified = True
    print(
        f"note: {subject} does not exist in this sandbox; "
        f"using the working directory instead ({original} -> {mapped})",
        file = sys.stderr,
    )


def _map_onto_cwd(prefix, text):
    """Map ``<prefix>/rest`` onto ``./rest`` in the CWD and note it once."""
    rel = text[len(prefix) :].lstrip("/")
    mapped = os.path.join(os.getcwd(), rel) if rel else os.getcwd()
    _note(prefix, text, mapped)
    return mapped


def _is_write_mode(mode):
    """True when an ``open()`` mode string implies writing/creating a file."""
    return isinstance(mode, str) and any(c in mode for c in ("w", "a", "x", "+"))


def _remap_open(file, mode):
    """Remap for ``open()`` / ``io.open()``.

    The prefix remaps run first (they cover reads and writes and preserve
    subpaths). Only if no prefix matched and the call is a write/create does the
    generalized fallback kick in: an absolute target outside the CWD whose
    parent directory is missing is a path the model invented from its CWD, so
    redirect it to the basename in the CWD. Reads are never redirected here.
    """
    mapped = _remap(file)
    if mapped is not file:
        return mapped
    if not _is_write_mode(mode):
        return file
    try:
        text = os.fspath(file)
    except TypeError:
        return file
    # bytes paths are left untouched: os.getcwd() is str and the prefix remaps
    # above already skip non-str, so keep the fallback str-only too.
    if not isinstance(text, str) or not os.path.isabs(text):
        return file
    cwd = os.getcwd()
    # Already inside the CWD: a real target the model meant; leave it alone.
    if text == cwd or text.startswith(cwd + os.sep):
        return file
    parent = os.path.dirname(text)
    # Only redirect when the parent directory is missing. An existing external
    # directory (a real /home/ubuntu path, a user-made dir, a symlink to one --
    # os.path.exists follows symlinks) is a deliberate, working target and must
    # pass through untouched so real writes stay truthful.
    if parent and os.path.exists(parent):
        return file
    remapped = os.path.join(cwd, os.path.basename(text))
    _note(text, text, remapped)
    return remapped


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

    def _open(
        file,
        mode = "r",
        *args,
        **kwargs,
    ):
        return original_open(_remap_open(file, mode), mode, *args, **kwargs)

    def _io_open(
        file,
        mode = "r",
        *args,
        **kwargs,
    ):
        return original_io_open(_remap_open(file, mode), mode, *args, **kwargs)

    # mkdir/makedirs get only the prefix remap, never the write-mode fallback:
    # creating an arbitrary absolute directory can legitimately succeed on the
    # host (e.g. under writable /home/ubuntu), so it must not be second-guessed.
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
