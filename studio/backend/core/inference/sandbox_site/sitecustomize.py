# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sandbox-side compatibility shim for ChatGPT code-interpreter paths.

Models habitually write to /mnt/data (or /mnt/outputs, /home/sandbox,
/workspace), none of which exist in the Studio sandbox. This module sits on the
sandbox subprocess PYTHONPATH (see ``tools._build_safe_env``), so it loads at
interpreter startup in every sandboxed ``python`` run and any Python the
``terminal`` tool launches.

It remaps those prefixes onto the working directory in ``open`` / ``io.open``,
``os.open``, ``os.makedirs`` / ``os.mkdir`` and ``pathlib.Path.mkdir``. A write
(or create) to a convention prefix always heals onto the CWD; a READ of one
heals only when the mapped CWD target already exists (re-reading an artifact an
earlier write produced), so a genuine missing input stays truthful on the path
the model used rather than being silently redirected onto a same-basename
workdir file. Because prefix lists cannot enumerate every path a model invents,
``open`` / ``io.open`` also get a write-mode fallback: a create-mode open of an
absolute path outside the CWD whose parent is missing is redirected to the
basename in the CWD. Reads never hit the fallback, and mkdir never does either
(an arbitrary absolute directory can legitimately succeed). The fallback is
collision-safe: it refuses to redirect onto an existing CWD file (letting the
original open raise). The
patch set (io.open, os.open, os.mkdir, Path.mkdir, and the <3.11
``_NormalAccessor.open``) covers the low-level entry points pathlib routes
through. A one-line stderr notice fires on the first remap. Everything is
wrapped in try/except so a failure never breaks the interpreter.

Identical with and without output streaming because the child env is.
"""

import builtins
import io
import json
import os
import sys

# Code-interpreter convention prefixes. Remapping is gated on the prefix ROOT
# being ABSENT (see _remap) so a genuine host mount / user directory is never
# shadowed and reads/writes under it stay truthful.
_PREFIXES = ("/mnt/data", "/mnt/outputs", "/home/sandbox", "/workspace")
# /tmp exists on the host; kept separate only to document that. The absence gate
# below applies to every prefix alike.
_CONDITIONAL_PREFIXES = ("/tmp/outputs",)
_notified = False
# Invented absolute write path -> healed CWD target, so overwriting the same
# generated artifact re-serves that target instead of tripping the anti-clobber
# guard on the file it created earlier.
_remapped_writes: dict = {}
# Each tool call is a fresh subprocess (in-process map empty on entry), so this
# on-disk sidecar carries the same map across runs. It only records sources the
# fallback itself healed, so an unrelated same-basename file is never adopted.
_REMAP_SIDECAR = ".unsloth_sandbox_remap.json"


def _note(subject, original, mapped):
    """Print the one-shot stderr notice so the model learns the real location.

    ``subject`` is what "does not exist" (the convention prefix, or the whole
    invented path); ``original`` is echoed in the ``(original -> mapped)`` tail.
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


def _contained_join(cwd, rel):
    """Join ``rel`` onto ``cwd`` so the result can never escape ``cwd``.

    A habit path can carry ``..`` segments; joining verbatim would let the mapped
    target climb above the per-conversation sandbox. Parent-traversal components
    are dropped and empty / ``.`` components ignored, keeping the result under
    ``cwd`` while preserving as much of the subpath as possible.
    """
    parts = []
    for part in rel.split("/"):
        if part == "" or part == ".":
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return os.path.join(cwd, *parts) if parts else cwd


def _map_onto_cwd(prefix, text, notify = True):
    """Map ``<prefix>/rest`` onto ``./rest`` in the CWD, noting it once.

    The suffix is contained under the CWD (see ``_contained_join``) so a path
    like ``/mnt/data/../other_session/file`` cannot escape the sandbox workdir.
    ``notify`` is False when the caller has not yet decided to use the mapping
    (a read that may keep the original path), so the one-shot notice is not
    spent on a remap that never happens.
    """
    rel = text[len(prefix) :].lstrip("/")
    mapped = _contained_join(os.getcwd(), rel)
    if notify:
        _note(prefix, text, mapped)
    return mapped


def _sidecar_path(cwd):
    return os.path.join(cwd, _REMAP_SIDECAR)


def _load_sidecar(cwd):
    """Return the persisted ``source -> healed target`` map, or {} on any error
    (missing/corrupt/foreign sidecar degrades to in-process-only behaviour)."""
    try:
        with open(_sidecar_path(cwd)) as fh:
            data = json.load(fh)
    except Exception:  # noqa: BLE001 - a bad sidecar must never break user code
        return {}
    return data if isinstance(data, dict) else {}


def _record_sidecar(cwd, source, target):
    """Persist ``source -> target`` in the sidecar so the next run re-serves it.

    Written atomically (temp + ``os.replace``) and wrapped so a read-only/full
    filesystem never breaks the interpreter. The path is inside the CWD, so the
    patched ``open`` leaves it untouched (no remap, no recursion).
    """
    try:
        data = _load_sidecar(cwd)
        if data.get(source) == target:
            return
        data[source] = target
        tmp = _sidecar_path(cwd) + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(data, fh)
        os.replace(tmp, _sidecar_path(cwd))
    except Exception:  # noqa: BLE001 - persistence is best effort only
        pass


def _is_creating_mode(mode):
    """True only when an ``open()`` mode string can CREATE a missing file.

    Only ``w`` / ``a`` / ``x`` create a target. ``r+`` / ``rb+`` require the path
    to exist, so they must not trip the write fallback (which would corrupt an
    unrelated same-basename file); ``w+`` / ``a+`` / ``x+`` still match.
    """
    return isinstance(mode, str) and any(c in mode for c in ("w", "a", "x"))


def _remap_open(file, mode):
    """Remap for ``open()`` / ``io.open()``.

    A prefix remap runs first. A write/create heals onto the CWD unconditionally;
    a READ heals only when the mapped CWD target already exists (re-reading an
    artifact an earlier write produced). Otherwise the original absolute path is
    kept, so a genuine missing input fails truthfully on the path the caller used
    instead of being silently redirected onto a same-basename workdir file or
    masking the error. Only if no prefix matched and the call is a create does
    the fallback kick in: an absolute target outside the CWD whose parent is
    missing is redirected to the basename in the CWD.

    Collision safety: if ``CWD/<basename>`` already exists (an unrelated
    conversation file), refuse the redirect and return the original path so the
    real ``open()`` raises ``FileNotFoundError`` and the existing file is kept.
    """
    creating = _is_creating_mode(mode)
    # notify=False: only emit the one-shot notice once we commit to the mapping
    # below, so a read that keeps its original path does not spend the notice.
    mapped = _remap(file, notify = False)
    if mapped is not file:
        # A write always heals onto the CWD. A read heals only when the mapped
        # target exists; otherwise keep the original absolute path so a missing
        # input stays truthful instead of silently reading a workdir file.
        if creating or os.path.exists(mapped):
            # Emit the one-shot notice now that we commit to the redirect (the
            # notify=False peek above kept a read that keeps its original path
            # from spending the notice on a remap that never happened).
            _remap(file, notify = True)
            return mapped
        return file
    if not creating:
        return file
    try:
        text = os.fspath(file)
    except TypeError:
        return file
    # bytes paths left untouched (str-only, matching the prefix remaps).
    if not isinstance(text, str) or not os.path.isabs(text):
        return file
    cwd = os.getcwd()
    # Already inside the CWD: a real target the model meant; leave it alone.
    if text == cwd or text.startswith(cwd + os.sep):
        return file
    parent = os.path.dirname(text)
    # Only redirect when the parent is missing; an existing external directory is
    # a deliberate target and must stay truthful (os.path.exists follows symlinks).
    if parent and os.path.exists(parent):
        return file
    base = os.path.basename(text)
    # A trailing separator or '.'/'..' component yields a basename that would
    # redirect onto the CWD itself or its parent; refuse and let open raise.
    if base in ("", ".", ".."):
        return file
    remapped = os.path.join(cwd, base)
    # Never clobber an unrelated workspace file sharing this basename (lexists
    # catches dangling symlinks): refuse and let open raise. But a target THIS
    # fallback already healed for the same invented path (per the in-process map
    # or the cross-run sidecar) is the artifact the model is re-writing, so
    # re-serve it rather than FileNotFoundError on every overwrite after the first.
    if os.path.lexists(remapped) and remapped not in (
        _remapped_writes.get(text),
        _load_sidecar(cwd).get(text),
    ):
        return file
    _remapped_writes[text] = remapped
    _record_sidecar(cwd, text, remapped)
    _note(text, text, remapped)
    return remapped


def _remap(path, notify = True):
    """Map ``<prefix>/rest`` onto ``./rest`` in the CWD; other paths pass through.

    ``notify`` is forwarded to ``_map_onto_cwd``; ``_remap_open`` passes False so
    a read that ends up keeping its original path does not emit a false notice.
    """
    try:
        text = os.fspath(path)
    except TypeError:
        return path
    if not isinstance(text, str):
        return path
    for prefix in _PREFIXES + _CONDITIONAL_PREFIXES:
        # Heal only while the real prefix directory is absent, so a genuine host
        # mount / user directory at that prefix is never shadowed.
        if (text == prefix or text.startswith(prefix + "/")) and not os.path.exists(prefix):
            return _map_onto_cwd(prefix, text, notify = notify)
    return path


def _install():
    import pathlib

    original_open = builtins.open
    original_io_open = io.open
    original_os_open = os.open
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
    # an arbitrary absolute directory can legitimately succeed on the host.
    def _makedirs(name, *args, **kwargs):
        return original_makedirs(_remap(name), *args, **kwargs)

    def _mkdir(path, *args, **kwargs):
        return original_mkdir(_remap(path), *args, **kwargs)

    def _os_open(
        path,
        flags,
        mode = 0o777,
        *,
        dir_fd = None,
    ):
        # Path.touch() and other low-level callers go through os.open, not
        # builtins.open. Only O_CREAT can create a missing target, so only it
        # maps to the "creating" mode; O_TRUNC / O_APPEND without O_CREAT still
        # require the file to exist, so they behave as a read (heal only when the
        # mapped target already exists, matching _remap_open).
        logical_mode = "w" if (flags & os.O_CREAT) else "r"
        mapped = _remap_open(path, logical_mode)
        if dir_fd is None:
            return original_os_open(mapped, flags, mode)
        return original_os_open(mapped, flags, mode, dir_fd = dir_fd)

    def _path_mkdir(self, *args, **kwargs):
        # pathlib probes Path.is_dir()/os.stat (unpatched) on FileExistsError, so
        # a bare os.mkdir remap would still raise when the mapped target exists.
        # Remap the receiver up front so parents/exist_ok stays idempotent.
        mapped = _remap(self)
        target = self if mapped is self else self.__class__(mapped)
        return original_path_mkdir(target, *args, **kwargs)

    builtins.open = _open
    # pathlib.Path.open / write_text / read_text call io.open directly, so patch both.
    io.open = _io_open
    # Python < 3.11 only: pathlib's accessor singleton captured the ORIGINAL
    # io.open at import (``_NormalAccessor.open = io.open``), so the io.open patch
    # never reaches it. Repoint it at the same wrapper (staticmethod so it stays
    # unbound). 3.11+ dropped the accessor, so this is an idempotent no-op there.
    accessor = getattr(pathlib, "_NormalAccessor", None)
    if accessor is not None and hasattr(accessor, "open"):
        accessor.open = staticmethod(_io_open)
    # Path.touch() and other low-level opens call os.open directly, so patch it too.
    os.open = _os_open
    os.makedirs = _makedirs
    # Path.mkdir(parents=True) calls os.mkdir per component, so patch os.mkdir;
    # patch Path.mkdir itself too so exist_ok/parents land on the mapped path.
    os.mkdir = _mkdir
    pathlib.Path.mkdir = _path_mkdir


try:
    _install()
except Exception:  # noqa: BLE001 - a broken shim must never break user code
    pass
