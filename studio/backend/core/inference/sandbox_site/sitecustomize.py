# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sandbox-side compatibility shim for code-interpreter path conventions.

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
import importlib
import importlib.machinery
import importlib.util
import io
import json
import os
import sys
import types

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


def _initial_trusted_library_roots():
    """Capture interpreter-managed package roots before sandbox code can edit sys.path."""
    roots = []
    for entry in sys.path:
        if not isinstance(entry, str) or not entry:
            continue
        try:
            path = os.path.realpath(entry)
        except OSError:
            continue
        if os.path.basename(path).lower() not in {"site-packages", "dist-packages"}:
            continue
        if path not in roots:
            roots.append(path)
    return tuple(roots)


_TRUSTED_LIBRARY_ROOTS = _initial_trusted_library_roots()


def _path_is_in_roots(filename, roots):
    if not isinstance(filename, str) or filename.startswith("<"):
        return False
    try:
        path = os.path.realpath(filename)
        return any(os.path.commonpath((root, path)) == root for root in roots)
    except (OSError, ValueError):
        return False


def _blocked_network_module_origin(filename):
    if not isinstance(filename, str) or filename.startswith("<"):
        return None
    try:
        path = os.path.realpath(filename)
        for root in _TRUSTED_LIBRARY_ROOTS:
            if os.path.commonpath((root, path)) != root:
                continue
            relative = os.path.relpath(path, root).replace("\\", "/")
            package = relative.split("/", 1)[0].removesuffix(".py")
            if package in _BLOCKED_NETWORK_MODULES or package in _DIRECT_BLOCKED_NETWORK_MODULES:
                return package
    except (OSError, ValueError):
        return None
    return None


def _frame_uses_trusted_package(frame, package):
    module_name = frame.f_globals.get("__name__", "")
    if not isinstance(module_name, str):
        return False
    if module_name != package and not module_name.startswith(f"{package}."):
        return False
    module = sys.modules.get(module_name)
    if module is None:
        return False
    try:
        module_dict = types.ModuleType.__getattribute__(module, "__dict__")
    except TypeError:
        module_dict = getattr(module, "__dict__", None)
    if module_dict is not frame.f_globals:
        return False
    spec = getattr(module, "__spec__", None)
    origin = getattr(spec, "origin", None) or getattr(module, "__file__", None)
    if not _path_is_in_roots(origin, _TRUSTED_LIBRARY_ROOTS):
        return False
    return _path_is_in_roots(frame.f_code.co_filename, _TRUSTED_LIBRARY_ROOTS)


def _trusted_http_client_frame(frame):
    return _frame_uses_trusted_package(frame, "httpx") or _frame_uses_trusted_package(
        frame, "httpcore"
    )


def _trusted_httpx_in_call_stack(skip = 1):
    try:
        frame = sys._getframe(skip)
    except ValueError:
        return False
    while frame is not None:
        if _frame_uses_trusted_package(frame, "httpx"):
            return True
        frame = frame.f_back
    return False


def _frame_is_importlib(frame):
    module_name = frame.f_globals.get("__name__", "")
    if not isinstance(module_name, str) or (
        module_name != "importlib" and not module_name.startswith("importlib.")
    ):
        return False
    module = sys.modules.get(module_name)
    return module is not None and getattr(module, "__dict__", None) is frame.f_globals


def _sandbox_code_requested_import(skip = 1):
    try:
        frame = sys._getframe(skip)
    except ValueError:
        return True
    while frame is not None:
        if (
            frame.f_code.co_filename == __file__
            or _frame_is_importlib(frame)
            or str(frame.f_code.co_filename).startswith("<frozen importlib")
        ):
            frame = frame.f_back
            continue
        return not _trusted_http_client_frame(frame)
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


def _blocked_network_loader_origin(filename):
    root = _blocked_network_module_origin(filename)
    if root in _BLOCKED_NETWORK_MODULES:
        return root
    if root in _DIRECT_BLOCKED_NETWORK_MODULES and _sandbox_code_requested_import():
        return root
    return None


def _raise_blocked_network_module(root):
    raise ModuleNotFoundError(
        f"Blocked: low-level network module {root!r} is unavailable in sandboxed code"
    )


_HTTP_CORE_METADATA = frozenset(
    {
        "__cached__",
        "__doc__",
        "__file__",
        "__loader__",
        "__name__",
        "__package__",
        "__path__",
        "__spec__",
    }
)


class _GuardedHttpcoreModule(types.ModuleType):
    """Keep httpx working while denying cached low-level APIs to sandbox code."""

    def __getattribute__(self, name):
        if name not in _HTTP_CORE_METADATA and _sandbox_code_requested_import(2):
            _raise_blocked_network_module("httpcore")
        return types.ModuleType.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if _sandbox_code_requested_import(2):
            _raise_blocked_network_module("httpcore")
        return types.ModuleType.__setattr__(self, name, value)

    def __delattr__(self, name):
        if _sandbox_code_requested_import(2):
            _raise_blocked_network_module("httpcore")
        return types.ModuleType.__delattr__(self, name)


def _make_httpcore_backend_dispatchers():
    originals = {}

    def register(original):
        key = object()
        originals[key] = original
        return key

    def dispatch(key, *args, **kwargs):
        return originals[key](*args, **kwargs)

    async def dispatch_async(key, *args, **kwargs):
        return await originals[key](*args, **kwargs)

    return register, dispatch, dispatch_async


(
    _register_httpcore_backend,
    _dispatch_httpcore_backend,
    _dispatch_httpcore_backend_async,
) = _make_httpcore_backend_dispatchers()


def _guard_httpcore_backend_method(cls, method_name):
    original = cls.__dict__.get(method_name)
    if not callable(original) or getattr(original, "_unsloth_httpcore_backend_guard", False):
        return

    key = _register_httpcore_backend(original)
    trusted_request = _trusted_httpx_in_call_stack
    blocked = _raise_blocked_network_module
    code = getattr(original, "__code__", None)
    if code is not None and code.co_flags & 0x80:  # CO_COROUTINE
        dispatch = _dispatch_httpcore_backend_async

        async def guarded(*args, **kwargs):
            if not trusted_request(2):
                blocked("httpcore")
            return await dispatch(key, *args, **kwargs)

    else:
        dispatch = _dispatch_httpcore_backend

        def guarded(*args, **kwargs):
            if not trusted_request(2):
                blocked("httpcore")
            return dispatch(key, *args, **kwargs)

    guarded._unsloth_httpcore_backend_guard = True
    guarded.__name__ = getattr(original, "__name__", method_name)
    guarded.__qualname__ = getattr(original, "__qualname__", guarded.__name__)
    guarded.__doc__ = getattr(original, "__doc__", None)
    setattr(cls, method_name, guarded)


def _guard_httpcore_network_backends(module):
    """Guard httpcore's connection boundary even if module attribute lookup is bypassed."""
    seen = set()
    for value in vars(module).values():
        if not isinstance(value, type) or id(value) in seen:
            continue
        seen.add(id(value))
        _guard_httpcore_backend_method(value, "connect_tcp")
        _guard_httpcore_backend_method(value, "connect_unix_socket")


def _guard_loaded_httpcore_modules():
    """Harden httpcore modules loaded transitively by an approved high-level client."""
    for name, module in tuple(sys.modules.items()):
        if name != "httpcore" and not name.startswith("httpcore."):
            continue
        if not isinstance(module, types.ModuleType) or isinstance(module, _GuardedHttpcoreModule):
            continue
        spec = getattr(module, "__spec__", None)
        if getattr(spec, "_initializing", False):
            continue
        _guard_httpcore_network_backends(module)
        module.__class__ = _GuardedHttpcoreModule


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
    absolute_name = _absolute_import_name(name, package, level)
    root = _blocked_network_module(absolute_name)
    if root is not None:
        _raise_blocked_network_module(root)
    module = _original_import(name, globals, locals, fromlist, level)
    if isinstance(absolute_name, str) and absolute_name.split(".", 1)[0] == "httpcore":
        _guard_loaded_httpcore_modules()
    return module


def _guarded_import_module(name, package = None):
    absolute_name = _absolute_import_name(name, package)
    root = _blocked_network_module(absolute_name)
    if root is not None:
        _raise_blocked_network_module(root)
    module = _original_import_module(name, package)
    if isinstance(absolute_name, str) and absolute_name.split(".", 1)[0] == "httpcore":
        _guard_loaded_httpcore_modules()
    return module


def _guard_legacy_source_loader():
    cls = importlib.machinery.SourceFileLoader
    original_load_module = getattr(cls, "load_module", None)
    original_exec_module = getattr(cls, "exec_module", None)

    def blocked_loader_root(self, fullname = None):
        root = _blocked_network_module(fullname)
        if root is None:
            root = _blocked_network_loader_origin(getattr(self, "path", None))
        return root

    if callable(original_load_module) and not getattr(
        original_load_module, "_unsloth_network_guard", False
    ):

        def guarded_load_module(self, *args, **kwargs):
            fullname = args[0] if args else kwargs.get("fullname", getattr(self, "name", None))
            root = blocked_loader_root(self, fullname)
            if root is not None:
                _raise_blocked_network_module(root)
            return original_load_module(self, *args, **kwargs)

        guarded_load_module._unsloth_network_guard = True
        guarded_load_module.__name__ = getattr(original_load_module, "__name__", "load_module")
        guarded_load_module.__qualname__ = getattr(
            original_load_module, "__qualname__", guarded_load_module.__name__
        )
        guarded_load_module.__doc__ = getattr(original_load_module, "__doc__", None)
        cls.load_module = guarded_load_module

    if callable(original_exec_module) and not getattr(
        original_exec_module, "_unsloth_network_guard", False
    ):

        def guarded_exec_module(self, module):
            fullname = getattr(module, "__name__", getattr(self, "name", None))
            root = blocked_loader_root(self, fullname)
            if root is not None:
                _raise_blocked_network_module(root)
            return original_exec_module(self, module)

        guarded_exec_module._unsloth_network_guard = True
        guarded_exec_module.__name__ = getattr(original_exec_module, "__name__", "exec_module")
        guarded_exec_module.__qualname__ = getattr(
            original_exec_module, "__qualname__", guarded_exec_module.__name__
        )
        guarded_exec_module.__doc__ = getattr(original_exec_module, "__doc__", None)
        cls.exec_module = guarded_exec_module


def _make_network_guard_audit():
    """Create a guard whose decisions do not depend on mutable module globals."""
    blocked = _BLOCKED_NETWORK_MODULES
    direct_blocked = _DIRECT_BLOCKED_NETWORK_MODULES
    trusted_roots = _TRUSTED_LIBRARY_ROOTS
    modules = sys.modules
    module_getattribute = types.ModuleType.__getattribute__
    getframe = sys._getframe
    commonpath = os.path.commonpath
    realpath = os.path.realpath
    relpath = os.path.relpath
    shim_path = realpath(__file__)

    def blocked_error(root):
        raise ModuleNotFoundError(
            f"Blocked: low-level network module {root!r} is unavailable in sandboxed code"
        )

    def blocked_origin(filename):
        if not isinstance(filename, str) or filename.startswith("<"):
            return None
        try:
            path = realpath(filename)
            for root in trusted_roots:
                if commonpath((root, path)) != root:
                    continue
                relative = relpath(path, root).replace("\\", "/")
                package = relative.split("/", 1)[0].removesuffix(".py")
                if package in blocked or package in direct_blocked:
                    return package
        except (OSError, ValueError):
            return None
        return None

    def blocked_origin_in_stack(skip):
        try:
            frame = getframe(skip)
        except ValueError:
            return None
        while frame is not None:
            root = blocked_origin(frame.f_code.co_filename)
            if root is not None:
                return root
            frame = frame.f_back
        return None

    def frame_uses_package(frame, package):
        module_name = frame.f_globals.get("__name__", "")
        if not isinstance(module_name, str):
            return False
        if module_name != package and not module_name.startswith(f"{package}."):
            return False
        module = modules.get(module_name)
        if module is None:
            return False
        try:
            module_dict = module_getattribute(module, "__dict__")
        except TypeError:
            module_dict = getattr(module, "__dict__", None)
        if module_dict is not frame.f_globals:
            return False
        spec = getattr(module, "__spec__", None)
        origin = getattr(spec, "origin", None) or getattr(module, "__file__", None)
        filename = frame.f_code.co_filename
        if not isinstance(origin, str) or not isinstance(filename, str):
            return False
        try:
            origin_path = realpath(origin)
            code_path = realpath(filename)
            for root in trusted_roots:
                if commonpath((root, origin_path)) != root:
                    continue
                if commonpath((root, code_path)) != root:
                    continue
                origin_relative = relpath(origin_path, root).replace("\\", "/")
                code_relative = relpath(code_path, root).replace("\\", "/")
                return (
                    origin_relative == package or origin_relative.startswith(f"{package}/")
                ) and (code_relative == package or code_relative.startswith(f"{package}/"))
        except (OSError, ValueError):
            return False
        return False

    def package_in_stack(package, skip):
        try:
            frame = getframe(skip)
        except ValueError:
            return False
        while frame is not None:
            if frame_uses_package(frame, package):
                return True
            frame = frame.f_back
        return False

    def sandbox_requested_import(skip):
        try:
            frame = getframe(skip)
        except ValueError:
            return True
        while frame is not None:
            filename = frame.f_code.co_filename
            if filename == shim_path or (
                isinstance(filename, str) and filename.startswith("<frozen importlib")
            ):
                frame = frame.f_back
                continue
            return not (frame_uses_package(frame, "httpx") or frame_uses_package(frame, "httpcore"))
        return True

    def audit(event, args):
        if event == "import" and args:
            fullname = args[0]
            if not isinstance(fullname, str):
                return
            root = fullname.split(".", 1)[0]
            if root in blocked or (root in direct_blocked and sandbox_requested_import(2)):
                blocked_error(root)
            return
        if event not in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo"}:
            return
        root = blocked_origin_in_stack(2)
        if root in blocked:
            blocked_error(root)
        if root in direct_blocked:
            if package_in_stack("httpx", 2):
                return
            blocked_error(root)
        if (
            package_in_stack("httpcore", 2)
            or package_in_stack("anyio", 2)
            or package_in_stack("trio", 2)
        ):
            if package_in_stack("httpx", 2):
                return
            blocked_error("httpcore")

    return audit


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


def _loaded_from_sandbox_site():
    """True when this shim is imported from the sandbox site dir on PYTHONPATH.

    The parent adds this directory to a sandbox child's PYTHONPATH, so its
    presence confirms the child is still running under the sandbox launcher even
    if ``UNSLOTH_STUDIO_SANDBOXED`` has been altered in ``os.environ``.
    """
    try:
        module_dir = os.path.realpath(os.path.dirname(__file__))
    except (OSError, NameError, TypeError):
        return False
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if not entry:
            continue
        try:
            if os.path.realpath(entry) == module_dir:
                return True
        except OSError:
            continue
    return False


def _sandbox_guard_should_activate():
    """Decide whether to install the runtime network guard.

    Normal sandbox children set ``UNSLOTH_STUDIO_SANDBOXED=1``. Bypass (full
    access) removes the variable entirely, so an absent flag means "do not
    guard". A flag that is PRESENT but not ``"1"`` (e.g. sandbox code running
    ``os.environ['UNSLOTH_STUDIO_SANDBOXED']='0'`` before spawning a child to
    escape the guard) is tampering: keep the guard on as long as this shim was
    still loaded from the sandbox site dir the launcher put on PYTHONPATH.
    """
    flag = os.environ.get("UNSLOTH_STUDIO_SANDBOXED")
    if flag == "1":
        return True
    if flag is None:
        return False
    return _loaded_from_sandbox_site()


def _install_import_guard():
    global _import_guard_installed
    if __name__ != "sitecustomize" or not _sandbox_guard_should_activate():
        return
    if not _import_guard_installed:
        sys.addaudithook(_make_network_guard_audit())
        builtins.__import__ = _guarded_import
        importlib.import_module = _guarded_import_module
        _guard_legacy_source_loader()
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
