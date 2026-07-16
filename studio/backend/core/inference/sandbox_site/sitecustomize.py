# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sandbox-side compatibility shim for ChatGPT code-interpreter paths.

Models habitually write to /mnt/data (or /mnt/outputs, /home/sandbox,
/workspace), none of which exist in the Studio sandbox. This module sits on the
sandbox subprocess PYTHONPATH (see ``tools._build_safe_env``), so it loads at
interpreter startup in every sandboxed ``python`` run and any Python the
``terminal`` tool launches.

It remaps those prefixes onto the CWD in ``open`` / ``io.open``, ``os.open``,
``os.makedirs`` / ``os.mkdir`` and ``pathlib.Path.mkdir``. A write/create to a
convention prefix always heals onto the CWD; a READ heals only when the mapped
target already exists (re-reading an earlier write), so a genuinely missing
input stays truthful on the path the model used instead of silently reading a
same-basename workdir file. Since prefix lists cannot cover every invented path,
``open`` / ``io.open`` also get a create-mode fallback: an absolute path outside
the CWD whose parent is missing is redirected to the basename in the CWD. Reads
and mkdir never use the fallback (an arbitrary absolute directory can legitimately
succeed). It is collision-safe: it refuses to redirect onto an existing CWD file
(letting open raise). The patch set (io.open, os.open, os.mkdir, Path.mkdir, and
the <3.11 ``_NormalAccessor.open``) covers the low-level entry points pathlib
routes through. A one-line stderr notice fires on the first remap, and everything
is wrapped in try/except so a failure never breaks the interpreter.

Identical with and without output streaming because the child env is.
"""

import builtins
import importlib
import importlib.util
import io
import json
import os
import sys

# Code-interpreter convention prefixes. Remapping is gated on the prefix being
# ABSENT (see _remap) so a genuine host mount / user dir is never shadowed.
_PREFIXES = ("/mnt/data", "/mnt/outputs", "/home/sandbox", "/workspace")
# /tmp exists on the host; separate only to note that. The absence gate applies alike.
_CONDITIONAL_PREFIXES = ("/tmp/outputs",)
_notified = False
# Invented absolute write path -> healed CWD target, so re-writing the same
# artifact re-serves it instead of tripping the anti-clobber guard.
_remapped_writes: dict = {}
# Each tool call is a fresh subprocess (in-process map starts empty), so this
# on-disk sidecar carries the map across runs. It records only sources the
# fallback healed, so an unrelated same-basename file is never adopted.
_REMAP_SIDECAR = ".unsloth_sandbox_remap.json"
_BLOCKED_NETWORK_MODULES = frozenset({"boto3", "botocore"})
# httpx imports httpcore internally, so block only sandbox-user requests.
_DIRECT_BLOCKED_NETWORK_MODULES = frozenset({"httpcore"})
_import_guard_installed = False
_original_import = builtins.__import__
_original_import_module = importlib.import_module


def _path_is_in_sandbox(filename):
    if not isinstance(filename, str) or filename.startswith("<"):
        return False
    try:
        cwd = os.path.realpath(os.getcwd())
        path = os.path.realpath(filename)
        return os.path.commonpath((cwd, path)) == cwd
    except (OSError, ValueError):
        return False


def _sandbox_code_requested_import():
    try:
        frame = sys._getframe(1)
    except ValueError:
        return True
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        if (
            module == __name__
            or module == "importlib"
            or module.startswith("importlib.")
            or module.startswith("_frozen_importlib")
        ):
            frame = frame.f_back
            continue
        if module == "__main__" or _path_is_in_sandbox(frame.f_globals.get("__file__")):
            return True
        return False
    return True


def _blocked_network_module(fullname):
    if not isinstance(fullname, str):
        return None
    root = fullname.split(".", 1)[0]
    if root in _BLOCKED_NETWORK_MODULES:
        return root
    if root in _DIRECT_BLOCKED_NETWORK_MODULES and _sandbox_code_requested_import():
        return root
    return None


def _raise_blocked_network_module(root):
    raise ModuleNotFoundError(
        f"Blocked: low-level network module {root!r} is unavailable in sandboxed code"
    )


def _absolute_import_name(
    name,
    package = None,
    level = 0,
):
    if not isinstance(name, str):
        return name
    if level:
        if not isinstance(package, str) or not package:
            return name
        relative = "." * level + name
    elif name.startswith(".") and isinstance(package, str) and package:
        relative = name
    else:
        return name
    try:
        return importlib.util.resolve_name(relative, package)
    except (ImportError, ValueError):
        return name


def _guarded_import(
    name,
    globals = None,
    locals = None,
    fromlist = (),
    level = 0,
):
    package = globals.get("__package__") if isinstance(globals, dict) else None
    root = _blocked_network_module(_absolute_import_name(name, package, level))
    if root is not None:
        _raise_blocked_network_module(root)
    return _original_import(name, globals, locals, fromlist, level)


def _guarded_import_module(name, package = None):
    root = _blocked_network_module(_absolute_import_name(name, package))
    if root is not None:
        _raise_blocked_network_module(root)
    return _original_import_module(name, package)


def _network_import_audit(event, args):
    if event != "import" or not args:
        return
    root = _blocked_network_module(args[0])
    if root is not None:
        _raise_blocked_network_module(root)


class _BlockedNetworkModuleFinder:
    _unsloth_blocked_network_guard = True

    def find_spec(
        self,
        fullname,
        path = None,
        target = None,
    ):
        root = _blocked_network_module(fullname)
        if root is not None:
            _raise_blocked_network_module(root)
        return None


def _install_import_guard():
    global _import_guard_installed
    if os.environ.get("UNSLOTH_STUDIO_SANDBOXED") != "1":
        return
    if not _import_guard_installed:
        sys.addaudithook(_network_import_audit)
        builtins.__import__ = _guarded_import
        importlib.import_module = _guarded_import_module
        _import_guard_installed = True
    if any(getattr(finder, "_unsloth_blocked_network_guard", False) for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, _BlockedNetworkModuleFinder())


def _note(subject, original, mapped):
    """Print the one-shot stderr notice so the model learns the real location.

    ``subject`` is what "does not exist" (the prefix, or the whole invented
    path); ``original`` is echoed in the ``(original -> mapped)`` tail.
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

    A habit path can carry ``..`` segments; joining verbatim would let the target
    climb above the sandbox. ``..`` components are dropped and empty / ``.`` ones
    ignored, keeping the result under ``cwd``.
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


def _map_onto_cwd(
    prefix,
    text,
    notify = True,
):
    """Map ``<prefix>/rest`` onto ``./rest`` in the CWD, noting it once.

    The suffix is contained under the CWD (see ``_contained_join``) so a path
    like ``/mnt/data/../other_session/file`` cannot escape the workdir.
    ``notify`` is False when the caller may keep the original path (a read), so
    the one-shot notice is not spent on a remap that never happens.
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
    """Persist ``source -> target`` so the next run re-serves it.

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

    Only ``w`` / ``a`` / ``x`` create. ``r+`` / ``rb+`` require the path to exist,
    so they must not trip the write fallback (which would corrupt an unrelated
    same-basename file); ``w+`` / ``a+`` / ``x+`` still match.
    """
    return isinstance(mode, str) and any(c in mode for c in ("w", "a", "x"))


def _remap_open(file, mode):
    """Remap for ``open()`` / ``io.open()``.

    A prefix remap runs first: a write/create heals onto the CWD; a READ heals
    only when the mapped target already exists (re-reading an earlier write),
    else the original path is kept so a genuine missing input fails truthfully
    instead of silently reading a same-basename workdir file. Only if no prefix
    matched and the call creates does the fallback kick in: an absolute target
    outside the CWD whose parent is missing is redirected to the basename in the
    CWD, unless ``CWD/<basename>`` already exists (an unrelated file), in which
    case the original path is kept so open raises.
    """
    creating = _is_creating_mode(mode)
    # notify=False: emit the notice only once we commit to the mapping below.
    mapped = _remap(file, notify = False)
    if mapped is not file:
        # Write always heals; a read only when the mapped target exists (else keep
        # the original path so a missing input stays truthful).
        if creating or os.path.exists(mapped):
            # Commit: emit the notice now (the notify=False peek above deferred it).
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
    # Redirect only when the parent is missing; an existing external directory is
    # a deliberate target and stays truthful (os.path.exists follows symlinks).
    if parent and os.path.exists(parent):
        return file
    base = os.path.basename(text)
    # A trailing sep or '.'/'..' basename would redirect onto the CWD or its
    # parent; refuse and let open raise.
    if base in ("", ".", ".."):
        return file
    remapped = os.path.join(cwd, base)
    # Never clobber an unrelated file sharing this basename (lexists catches
    # dangling symlinks). But a target this fallback already healed for the same
    # invented path (in-process map or cross-run sidecar) is the artifact being
    # re-written, so re-serve it instead of raising on every overwrite.
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
    a read that keeps its original path emits no false notice.
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
        # Path.touch() etc. go through os.open, not builtins.open. Only O_CREAT
        # can create, so only it maps to "creating" mode; O_TRUNC / O_APPEND
        # without O_CREAT still require the file to exist, so behave as a read.
        logical_mode = "w" if (flags & os.O_CREAT) else "r"
        mapped = _remap_open(path, logical_mode)
        if dir_fd is None:
            return original_os_open(mapped, flags, mode)
        return original_os_open(mapped, flags, mode, dir_fd = dir_fd)

    def _path_mkdir(self, *args, **kwargs):
        # pathlib probes Path.is_dir()/os.stat (unpatched) on FileExistsError, so
        # a bare os.mkdir remap would still raise when the target exists. Remap
        # the receiver up front so parents/exist_ok stays idempotent.
        mapped = _remap(self)
        target = self if mapped is self else self.__class__(mapped)
        return original_path_mkdir(target, *args, **kwargs)

    builtins.open = _open
    # pathlib.Path.open / write_text / read_text call io.open directly, so patch both.
    io.open = _io_open
    # Python < 3.11 only: pathlib's accessor captured the ORIGINAL io.open at
    # import (``_NormalAccessor.open = io.open``), so the io.open patch misses it.
    # Repoint it at the same wrapper (staticmethod to stay unbound); 3.11+ dropped
    # the accessor, so this is a no-op there.
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
    _install_import_guard()
except Exception:  # noqa: BLE001 - a broken guard must not break startup
    pass

try:
    _install()
except Exception:  # noqa: BLE001 - a broken shim must never break user code
    pass
