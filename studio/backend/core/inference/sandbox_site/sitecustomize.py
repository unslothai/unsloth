# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sandbox-side compatibility shim for ChatGPT code-interpreter paths.

Models habitually write to /mnt/data (or /mnt/outputs, /home/sandbox,
/workspace), none of which exist in the Unsloth sandbox. This module sits on the
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
import io
import json
import os
import sys

# Code-interpreter convention prefixes. Remapping is gated on the prefix being ABSENT (see _remap) so a genuine host
# mount / user dir is never shadowed.
_PREFIXES = ("/mnt/data", "/mnt/outputs", "/home/sandbox", "/workspace")
# /tmp exists on the host; separate only to note that. The absence gate applies alike.
_CONDITIONAL_PREFIXES = ("/tmp/outputs",)
_notified = False
# Invented absolute write path -> healed CWD target, so re-writing the same artifact re-serves it instead of tripping
# the anti-clobber guard.
_remapped_writes: dict = {}
# Each tool call is a fresh subprocess (in-process map starts empty), so this on-disk sidecar carries the map across
# runs. It records only sources the fallback healed, so an unrelated same-basename file is never adopted.
_REMAP_SIDECAR = ".unsloth_sandbox_remap.json"


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
        with open(_sidecar_path(cwd), encoding = "utf-8") as fh:
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
        with open(tmp, "w", encoding = "utf-8") as fh:
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
        # Write always heals; a read only when the mapped target exists (else keep the original path so a missing input
        # stays truthful).
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
    # Redirect only when the parent is missing; an existing external directory is a deliberate target and stays
    # truthful (os.path.exists follows symlinks).
    if parent and os.path.exists(parent):
        return file
    base = os.path.basename(text)
    # A trailing sep or '.'/'..' basename would redirect onto the CWD or its parent; refuse and let open raise.
    if base in ("", ".", ".."):
        return file
    remapped = os.path.join(cwd, base)
    # Never clobber an unrelated file sharing this basename (lexists catches dangling symlinks). But a target this
    # fallback already healed for the same invented path (in-process map or cross-run sidecar) is the artifact being
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
        # Heal only while the real prefix directory is absent, so a genuine host mount / user directory at that prefix
        # is never shadowed.
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

    # mkdir/makedirs get only the prefix remap, never the write-mode fallback: an arbitrary absolute directory can
    # legitimately succeed on the host.
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
        # Path.touch() etc. go through os.open, not builtins.open. Only O_CREAT can create, so only it maps to
        # "creating" mode; O_TRUNC / O_APPEND without O_CREAT still require the file to exist, so behave as a read.
        logical_mode = "w" if (flags & os.O_CREAT) else "r"
        mapped = _remap_open(path, logical_mode)
        if dir_fd is None:
            return original_os_open(mapped, flags, mode)
        return original_os_open(mapped, flags, mode, dir_fd = dir_fd)

    def _path_mkdir(self, *args, **kwargs):
        # pathlib probes Path.is_dir()/os.stat (unpatched) on FileExistsError, so a bare os.mkdir remap would still
        # raise when the target exists. Remap the receiver up front so parents/exist_ok stays idempotent.
        mapped = _remap(self)
        target = self if mapped is self else self.__class__(mapped)
        return original_path_mkdir(target, *args, **kwargs)

    builtins.open = _open
    # pathlib.Path.open / write_text / read_text call io.open directly, so patch both.
    io.open = _io_open
    # Python < 3.11 only: pathlib's accessor captured the ORIGINAL io.open at import (``_NormalAccessor.open =
    # io.open``), so the io.open patch misses it. Repoint it at the same wrapper (staticmethod to stay unbound); 3.11+
    # dropped the accessor, so this is a no-op there.
    accessor = getattr(pathlib, "_NormalAccessor", None)
    if accessor is not None and hasattr(accessor, "open"):
        accessor.open = staticmethod(_io_open)
    # Path.touch() and other low-level opens call os.open directly, so patch it too.
    os.open = _os_open
    os.makedirs = _makedirs
    # Path.mkdir(parents=True) calls os.mkdir per component, so patch os.mkdir; patch Path.mkdir itself too so
    # exist_ok/parents land on the mapped path.
    os.mkdir = _mkdir
    pathlib.Path.mkdir = _path_mkdir


try:
    _install()
except Exception:  # noqa: BLE001 - a broken shim must never break user code
    pass


# Approved host calls keep chat-installed packages importable, including Python started by Terminal.
# Served by a last-resort finder, not sys.path: a folder on sys.path also exposes its *.dist-info to
# importlib.metadata, so entry-point plugin autoload (pytest11 and the like) would import chat-planted code
# the approved command never named. The finder runs after every other one, so it only supplies top-level
# modules the host lacks (no stdlib or installed-package shadowing, no .pth execution); submodules then
# resolve through the package's own __path__. Metadata is answered only for an exact distribution name
# (importlib.metadata.version("pkg")) with no explicit path, never for the name-less scan that plugin
# discovery uses.
# A PEP 420 namespace package (google, azure, ...) recomputes its __path__ from sys.path alone, so a
# submodule the host's portion lacks is looked up in the folder's portion of that namespace, again last.
# The namespace's __path__ (PEP 420, pkgutil- or pkg_resources-style) then gets that portion appended.
class _HostPackagesFinder:
    _skip = frozenset(("sitecustomize", "usercustomize"))

    def __init__(self, folder):
        self._folder = folder

    def find_spec(
        self,
        name,
        path = None,
        target = None,
    ):
        import importlib.machinery

        if "." not in name:
            if path is not None or name in self._skip:
                return None
            return importlib.machinery.PathFinder.find_spec(name, [self._folder])
        parent, _, _ = name.rpartition(".")
        module = sys.modules.get(parent)
        spec = getattr(module, "__spec__", None)
        if spec is None or spec.submodule_search_locations is None:
            return None
        # Only namespace packages: a regular package's submodules come from its own __path__ alone.
        pkgutil_style = spec.origin not in (None, "namespace")
        if pkgutil_style and not self._extends_path(spec.origin):
            return None
        portion = os.path.join(self._folder, *parent.split("."))
        if not os.path.isdir(portion):
            return None
        found = importlib.machinery.PathFinder.find_spec(name, [portion])
        if found is not None:
            # What extend_path / declare_namespace (or PEP 420) would have collected had the folder been
            # on sys.path: this one portion directory, last, so later lookups, pkgutil.iter_modules and
            # importlib.resources see it too. A PEP 420 _NamespacePath recomputes from sys.path when
            # sys.path changes, which drops it again until the next submodule is served from it.
            search = getattr(module, "__path__", None)
            if search is not None and hasattr(search, "append") and portion not in list(search):
                search.append(portion)
        return found

    _extends = {}

    @classmethod
    def _extends_path(cls, origin):
        # A pkgutil-style (extend_path) or pkg_resources-style (declare_namespace) namespace package
        # collects its portions by scanning sys.path while its __init__ runs, so it never sees this folder.
        # Only the statements that declare one count: `__path__ = <...>extend_path(__path__, __name__)`
        # or `<...>declare_namespace(__name__)`, at module level or in a module-level try body / else /
        # finally (not an except handler, which may not have run). A comment,
        # a string, an unused import or a call under if/def/class leaves the package regular.
        if origin not in cls._extends:
            try:
                import ast
                with open(origin, "rb") as f:
                    tree = ast.parse(f.read())
                cls._extends[origin] = any(map(cls._declares_namespace, cls._top_level(tree.body)))
            except Exception:  # noqa: BLE001 - an unreadable __init__ is treated as a regular package
                cls._extends[origin] = False
        return cls._extends[origin]

    @classmethod
    def _top_level(cls, body):
        import ast
        for node in body:
            yield node
            if isinstance(node, ast.Try):
                # Only blocks that run when the try body completes: a declaration that sits only in
                # an except handler may never have run, so it does not open the package.
                for block in (node.body, node.orelse, node.finalbody):
                    yield from cls._top_level(block)

    @staticmethod
    def _declares_namespace(node):
        import ast

        def call_of(value, name, args):
            if not isinstance(value, ast.Call) or value.keywords:
                return False
            func = value.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            return called == name and [getattr(a, "id", None) for a in value.args] == args

        if isinstance(node, ast.Assign):
            return any(getattr(t, "id", None) == "__path__" for t in node.targets) and call_of(
                node.value, "extend_path", ["__path__", "__name__"]
            )
        return isinstance(node, ast.Expr) and call_of(node.value, "declare_namespace", ["__name__"])

    def find_distributions(self, context = None):
        name = getattr(context, "name", None)
        # An explicit path is the caller's search scope: the stock path finder already honours it
        # (folder included if listed), so only the default sys.path scope gets the folder added.
        if not name or "path" in getattr(context, "__dict__", {}):
            return iter(())
        import importlib.metadata

        return importlib.metadata.MetadataPathFinder.find_distributions(
            importlib.metadata.DistributionFinder.Context(name = name, path = [self._folder])
        )


_host_packages = os.environ.get("_UNSLOTH_HOST_SESSION_PACKAGES")
if _host_packages and not any(isinstance(f, _HostPackagesFinder) for f in sys.meta_path):
    sys.meta_path.append(_HostPackagesFinder(_host_packages))


# Helper modules the approved script imports by name come from the chat folder, after everything
# else, so they never shadow anything. Served by a last-resort finder for exactly those names rather
# than sys.path: site imports `usercustomize` after this file, and a library probing an optional
# import must not find a module an earlier sandboxed call planted there.
class _HostWorkdirFinder:
    _skip = frozenset(("sitecustomize", "usercustomize"))

    def __init__(self, workdir, names):
        self._workdir = workdir
        self._names = frozenset(names) - self._skip

    def find_spec(
        self,
        name,
        path = None,
        target = None,
    ):
        if path is not None or name not in self._names:
            return None
        import importlib.machinery

        spec = importlib.machinery.PathFinder.find_spec(name, [self._workdir])
        if spec is not None and spec.origin and spec.origin.endswith(".py"):
            # A requested helper's own absolute imports come from the chat folder too, as before.
            try:
                import ast

                with open(spec.origin, "rb") as f:
                    tree = ast.parse(f.read())
                found = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        found.update(alias.name.split(".")[0] for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                        found.add(node.module.split(".")[0])
                self._names = self._names | (frozenset(found) - self._skip)
            except Exception:  # noqa: BLE001 - an unreadable helper just adds nothing
                pass
        return spec


_host_workdir = os.environ.get("_UNSLOTH_HOST_WORKDIR")
_host_modules = [n for n in os.environ.get("_UNSLOTH_HOST_WORKDIR_MODULES", "").split(",") if n]
if (
    _host_workdir
    and _host_modules
    and not any(isinstance(f, _HostWorkdirFinder) for f in sys.meta_path)
):
    sys.meta_path.append(_HostWorkdirFinder(_host_workdir, _host_modules))
