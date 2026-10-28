# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Remaps invented paths like /mnt/data onto CWD; reads heal only if the target exists; all wrapped
in try/except so a failure never breaks the interpreter; no redirect onto existing CWD file"""

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
# Invented path -> healed target, so rewrites re-serve it instead of tripping anti-clobber.
_remapped_writes: dict = {}
# Each tool call is a fresh process, so this sidecar carries healed mappings across runs; only fallback-healed sources, never unrelated same-basename files.
_REMAP_SIDECAR = ".unsloth_sandbox_remap.json"


def _note(subject, original, mapped):
    """Prints once per process so the model learns the real location, not on every remap."""
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
    """Drops ``..`` components so a path cannot climb out of the sandbox cwd."""
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
    """Maps prefix/rest onto ./rest under the CWD, contained against .. escapes; notify False for reads so the notice is not spent on a remap that never happens."""
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
    """Writes atomically and never raises; the path sits in the CWD so the patched open leaves it alone."""
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
    """r+ must not trip the write fallback, which would corrupt an unrelated same-basename file."""
    return isinstance(mode, str) and any(c in mode for c in ("w", "a", "x"))


def _remap_open(file, mode):
    """A read heals only if the target exists, so a missing input fails instead of reading a stray file."""
    creating = _is_creating_mode(mode)
    # notify=False: emit the notice only once we commit to the mapping below.
    mapped = _remap(file, notify = False)
    if mapped is not file:
        # Reads heal only when the target exists, so a missing input stays truthful.
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
    # Only when the parent is missing; an existing external dir is a deliberate target.
    if parent and os.path.exists(parent):
        return file
    base = os.path.basename(text)
    # A trailing sep or '.'/'..' basename would redirect onto the CWD or its parent; refuse and let open raise.
    if base in ("", ".", ".."):
        return file
    remapped = os.path.join(cwd, base)
    # Never clobber an unrelated same-basename file (lexists also catches dangling symlinks), but re-serve one this fallback healed.
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
    """``_remap_open`` passes notify False, so a read that keeps its original path emits no false notice."""
    try:
        text = os.fspath(path)
    except TypeError:
        return path
    if not isinstance(text, str):
        return path
    for prefix in _PREFIXES + _CONDITIONAL_PREFIXES:
        # Heal only while the prefix directory is absent, so a real host mount is never shadowed.
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

    # mkdir gets only the prefix remap: arbitrary absolute dirs may legitimately succeed.
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
        # Only O_CREAT creates; O_TRUNC/O_APPEND alone behave as a read.
        logical_mode = "w" if (flags & os.O_CREAT) else "r"
        mapped = _remap_open(path, logical_mode)
        if dir_fd is None:
            return original_os_open(mapped, flags, mode)
        return original_os_open(mapped, flags, mode, dir_fd = dir_fd)

    def _path_mkdir(self, *args, **kwargs):
        # pathlib probes unpatched os.stat on FileExistsError; remap up front for exist_ok.
        mapped = _remap(self)
        target = self if mapped is self else self.__class__(mapped)
        return original_path_mkdir(target, *args, **kwargs)

    builtins.open = _open
    # pathlib.Path.open / write_text / read_text call io.open directly, so patch both.
    io.open = _io_open
    # Python < 3.11: pathlib's _NormalAccessor captured the original io.open; repoint it.
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


# Approved host calls import chat-installed packages through a last-resort finder, NOT sys.path: a folder on
# sys.path exposes its *.dist-info to importlib.metadata, so entry-point autoload (pytest11, ...) would run
# chat-planted code the command never named. Metadata answers only an exact name with no explicit path.
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
            # The portion extend_path / PEP 420 would have collected; a _NamespacePath drops it on sys.path changes.
            search = getattr(module, "__path__", None)
            if search is not None and hasattr(search, "append") and portion not in list(search):
                search.append(portion)
        return found

    _extends = {}

    @classmethod
    def _extends_path(cls, origin):
        # Open only if __init__ declares extend_path / declare_namespace at module level (try body / else /
        # finally count, an except handler may not have run); text, imports or calls under if/def do not.
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
        # An explicit path is the caller's scope, already served by the stock path finder.
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
