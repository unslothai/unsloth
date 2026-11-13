# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Remaps invented paths like /mnt/data onto CWD; reads heal only if the target exists; all wrapped
in try/except so a failure never breaks the interpreter; no redirect onto existing CWD file"""

import builtins
import io
import json
import os
import sys

# Prefixes remapped only when ABSENT (see _remap), so a real host mount is never shadowed.
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
    # Windows renders Path("/mnt/data/x") as "\\mnt\\data\\x": compare both sides with forward slashes. A drive or UNC
    # path still cannot match a rooted prefix like "/mnt/data".
    windows = os.sep == "\\"
    if windows:
        text = text.replace("\\", "/")
    for prefix in _PREFIXES + _CONDITIONAL_PREFIXES:
        if windows:
            prefix = prefix.replace("\\", "/")
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
    # Path.mkdir calls os.mkdir per part; patch both, since exist_ok/parents need Path.mkdir.
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

    def __init__(self, folders):
        self._folders = [folder for folder in folders.split(os.pathsep) if folder]

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
            return importlib.machinery.PathFinder.find_spec(name, self._folders)
        parent, _, _ = name.rpartition(".")
        module = sys.modules.get(parent)
        spec = getattr(module, "__spec__", None)
        if spec is None or spec.submodule_search_locations is None:
            return None
        # Only namespace packages: a regular package's submodules come from its own __path__ alone.
        pkgutil_style = spec.origin not in (None, "namespace")
        if pkgutil_style and not self._extends_path(spec.origin):
            return None
        portions = [os.path.join(folder, *parent.split(".")) for folder in self._folders]
        portions = [portion for portion in portions if os.path.isdir(portion)]
        if not portions:
            return None
        found = importlib.machinery.PathFinder.find_spec(name, portions)
        if found is not None:
            # The portions extend_path / PEP 420 would have collected; a _NamespacePath drops them on sys.path changes.
            search = getattr(module, "__path__", None)
            if search is not None and hasattr(search, "append"):
                for portion in portions:
                    if portion not in list(search):
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
            importlib.metadata.DistributionFinder.Context(name = name, path = self._folders)
        )


# --python hands the install to another interpreter, whose own pip decides. optparse also takes unambiguous
# abbreviations: the shortest is listed (--pre is its own option; --us, --ro, --pyth are ambiguous in pip).
_PIP_LOCATION_FLAGS = (
    ("--target", 4),
    ("--prefix", 6),
    ("--root", 6),
    ("--home", 6),
    ("--user", 6),
    ("--python", 8),
)


def _chooses_location(arg):
    name = arg.split("=", 1)[0]
    if name.startswith("--"):
        return any(full.startswith(name) and len(name) >= n for full, n in _PIP_LOCATION_FLAGS)
    # -t DIR, -tDIR, or clustered after no-operand flags (-qt DIR, -Ut).
    if name.startswith("-") and len(name) > 1:
        for flag in name[1:]:
            if flag == "t":
                return True
            if flag not in "qvUI":
                return False
    return False


def _is_pip_script(path):
    # pip, pip3, pip3.12, pip.exe, pip3-script.py: the console scripts, not any script whose name starts with pip.
    name = os.path.basename(path).lower()
    for suffix in (".exe", "-script.py"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    version = name[3:]
    return (
        name[:3] == "pip"
        and not version.startswith(".")
        and all(c.isdigit() or c == "." for c in version)
    )


def _is_pip_launch():
    argv = list(getattr(sys, "orig_argv", None) or ())
    module, skip = "", False
    for i, arg in enumerate(argv[1:], 1):
        if skip:
            skip = False
            continue
        if arg == "--check-hash-based-pycs":
            skip = True
            continue
        if not arg.startswith("-") or arg.startswith("--") or arg == "-":
            break
        # Short options cluster (-um pip, -Bm pip); -m, -c, -W and -X take the rest or the next word.
        for j, flag in enumerate(arg[1:], 1):
            rest = arg[j + 1 :]
            if flag == "m":
                module = rest or (argv[i + 1] if i + 1 < len(argv) else "")
                break
            if flag == "c":
                return False
            if flag in "WX":
                skip = not rest
                break
        if module:
            break
    # Python < 3.10 has no orig_argv: any -m launch counts.
    return module in ("pip", "pip.__main__") or (
        not module and ((not argv and sys.argv[0] == "-m") or _is_pip_script(sys.argv[0]))
    )


def _within(path, root):
    # root ends in a separator; the prefix folder itself (where --target installs land) counts too.
    path = os.path.join(os.path.normcase(os.path.abspath(path)), "")
    return path.startswith(root)


def _stdlib_paths():
    # The standard library alone: no session dir, .pth source, cwd or script dir can answer the shim's imports.
    base = os.path.join(os.path.normcase(os.path.realpath(sys.base_prefix)), "")
    return [
        e
        for e in sys.path
        if e
        and os.path.normcase(os.path.realpath(e)).startswith(base)
        and "site-packages" not in os.path.normcase(e)
        and "dist-packages" not in os.path.normcase(e)
    ]


def _config_chooses_location():
    # pip's config files (PIP_CONFIG_FILE, the chat HOME's user files, the interpreter's site file) can set
    # install.target / prefix / root / user too; PIP_PREFIX would then conflict with them.
    home = os.path.expanduser("~")
    files = [
        os.environ.get("PIP_CONFIG_FILE", ""),
        os.path.join(
            os.environ.get("XDG_CONFIG_HOME") or os.path.join(home, ".config"), "pip", "pip.conf"
        ),
        os.path.join(home, ".pip", "pip.conf"),
        os.path.join(sys.prefix, "pip.conf"),
    ]
    files = [f for f in files if f and f != os.devnull and os.path.isfile(f)]
    if not files:
        return False
    saved = sys.path[:]
    sys.path[:] = _stdlib_paths()
    try:
        import configparser
    finally:
        sys.path[:] = saved
    parser = configparser.RawConfigParser()
    try:
        parser.read(files, encoding = "utf-8")
    except (configparser.Error, UnicodeDecodeError):
        return False
    for section in ("global", "install"):
        for key in ("target", "prefix", "root", "home"):
            if parser.get(section, key, fallback = "").strip():
                return True
        if parser.get(section, "user", fallback = "").strip().lower() in (
            "y",
            "yes",
            "t",
            "true",
            "on",
            "1",
        ):
            return True
    return False


_chat_prefix = None


def _route_pip():
    try:
        prefix = os.environ.get("_UNSLOTH_PIP_PREFIX")
        owner = os.environ.get("_UNSLOTH_PIP_PREFIX_FOR")
        if not prefix or not owner:
            return

        def key(path):
            return os.path.normcase(os.path.realpath(path))

        if key(sys.prefix) == key(owner):
            global _chat_prefix
            _chat_prefix = prefix
            # Only a pip process gets PIP_PREFIX: anything else it spawns (venv --upgrade-deps) must not inherit it.
            is_pip = _is_pip_launch()
            # pip --target/--prefix/--user/... already chose a location; PIP_PREFIX would conflict with it.
            placed = any(_chooses_location(a) for a in sys.argv[1:])
            if "--isolated" not in sys.argv:
                # --isolated makes pip ignore these, so then they choose nothing.
                truthy = ("y", "yes", "t", "true", "on", "1")
                placed = placed or any(
                    os.environ.get(k) for k in ("PIP_TARGET", "PIP_ROOT", "PIP_HOME")
                )
                placed = placed or os.environ.get("PIP_USER", "").strip().lower() in truthy
                placed = placed or os.environ.get("PIP_PREFIX", prefix) not in ("", prefix)
                placed = placed or (is_pip and _config_chooses_location())
            if is_pip and not placed:
                os.environ["PIP_PREFIX"] = prefix
                os.environ["PIP_IGNORE_INSTALLED"] = "1"
                if "--isolated" in sys.argv and "install" in sys.argv:
                    # --isolated ignores PIP_* variables: pass the same options on the command line.
                    at = sys.argv.index("install") + 1
                    sys.argv[at:at] = ["--prefix", prefix, "--ignore-installed"]
            elif os.environ.get("PIP_PREFIX") == prefix:
                os.environ.pop("PIP_PREFIX", None)
                os.environ.pop("PIP_IGNORE_INSTALLED", None)
        else:
            # Another venv neither installs into nor sees the chat folder: its pip would count those packages as installed.
            if os.environ.get("PIP_PREFIX") == prefix:
                os.environ.pop("PIP_PREFIX", None)
                os.environ.pop("PIP_IGNORE_INSTALLED", None)
            root = os.path.join(key(prefix), "")

            def chat(entry):
                return bool(entry) and os.path.join(key(entry), "").startswith(root)

            sys.path[:] = [e for e in sys.path if not chat(e)]
            if "PYTHONPATH" in os.environ:
                os.environ["PYTHONPATH"] = os.pathsep.join(
                    e for e in os.environ["PYTHONPATH"].split(os.pathsep) if not chat(e)
                )
    except Exception:  # noqa: BLE001
        pass


_route_pip()


def _session_entries():
    prefix = _chat_prefix or os.environ.get("PIP_PREFIX")
    if not prefix:
        return []
    root = os.path.join(os.path.normcase(os.path.abspath(prefix)), "")
    return [e for e in sys.path if e and _within(e, root)]


def _literal(path, names):
    import ast

    # Decoded here as UTF-8: parsing bytes would honour a coding cookie, i.e. import a codec module.
    with open(path, encoding = "utf-8") as f:
        tree = ast.parse(f.read())
    found = {}
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
        for t in targets:
            if isinstance(t, ast.Name) and t.id in names and node.value is not None:
                found[t.id] = ast.literal_eval(node.value)
    return found


_finder_template = []


def _editable_finder_data(entry, module):
    data = _literal(os.path.join(entry, module + ".py"), {"MAPPING", "NAMESPACES"})
    mapping, namespaces = data.get("MAPPING", {}), data.get("NAMESPACES", {})
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
        return None
    if not all(isinstance(k, str) and isinstance(v, list) for k, v in namespaces.items()):
        return None
    return mapping, namespaces


def _install_editable_finder(module, mapping, namespaces):
    # setuptools' editable finder, built from the template of the setuptools in Studio's own site-packages (never
    # one the chat folder could shadow) with the chat file's literal MAPPING / NAMESPACES: the chat-written module
    # never runs. Read as text: importing setuptools is slow. Without setuptools there (torch ships it from 3.12)
    # such installs import only from their project folder, as under OS isolation.
    if not _finder_template:
        import sysconfig
        source = os.path.join(
            sysconfig.get_paths()["purelib"], "setuptools", "command", "editable_wheel.py"
        )
        _finder_template.append(_literal(source, {"_FINDER_TEMPLATE"})["_FINDER_TEMPLATE"])
    code = _finder_template[0].format(
        name = module, mapping = dict(sorted(mapping.items())), namespaces = namespaces
    )
    import types

    finder_module = types.ModuleType(module)
    exec(compile(code, "<" + module + ">", "exec"), finder_module.__dict__)
    sys.modules[module] = finder_module
    finder_module.install()
    return finder_module._EditableFinder


# PYTHONPATH skips .pth files: take their path lines and setuptools' editable finders, never run their imports.
# Paths go right after the session dirs, so a chat's editable fork wins over Studio's copy like its other installs.
def _read_pth_files(entries, on_sys_path = True):
    added, finders = [], []
    for entry in entries:
        try:
            names = sorted(n for n in os.listdir(entry) if n.endswith(".pth"))
        except OSError:
            continue
        for name in names:
            try:
                with open(os.path.join(entry, name), encoding = "utf-8") as f:
                    lines = f.read().splitlines()
                for line in lines:
                    line = line.strip()
                    if line.startswith(("import ", "import\t")):
                        module = line[len("import ") :].split(";")[0].strip()
                        if module.startswith("__editable___") and module.endswith("_finder"):
                            finders.append((entry, module))
                        continue
                    path = os.path.abspath(os.path.join(entry, line))
                    if (
                        line
                        and not line.startswith("#")
                        and path not in sys.path + added
                        and os.path.exists(path)
                    ):
                        added.append(path)
            except Exception:  # noqa: BLE001
                continue
    installed = []
    if finders and on_sys_path:
        # The helpers they import (ast, sysconfig, pathlib, ...) come from the stdlib, never a chat-planted copy.
        saved = sys.path[:]
        sys.path[:] = _stdlib_paths()
        try:
            # Every finder file is read before any finder is registered, so none can answer the others' imports.
            parsed = []
            for entry, module in finders:
                try:
                    data = _editable_finder_data(entry, module)
                except Exception:  # noqa: BLE001
                    data = None
                if data is not None:
                    parsed.append((module, data))
            for module, (mapping, namespaces) in parsed:
                try:
                    installed.append(_install_editable_finder(module, mapping, namespaces))
                except Exception:  # noqa: BLE001
                    pass
        finally:
            sys.path[:] = saved + [e for e in sys.path if e not in saved]
    if added and on_sys_path:
        at = max(i for i, e in enumerate(sys.path) if e in entries) + 1
        sys.path[at:at] = added
    return added, [f for f in installed if f is not None]


# First on meta_path: a session dir pip creates mid-process is cached as missing, a namespace imported before an
# install lacks the new portion, and an editable fork's concrete module wins over Studio's copy (namespaces are
# left to the normal aggregation, which the editable finders still join from their usual place).
class _SessionPackagesFinder:
    from importlib.machinery import PathFinder as _path_finder
    def __init__(
        self,
        entries,
        editable,
        chat_paths = (),
    ):
        self._entries = entries
        self._editable = editable
        self._chat_paths = set(entries) | set(chat_paths)

    def find_spec(
        self,
        name,
        path = None,
        target = None,
    ):
        cache = sys.path_importer_cache
        for entry in self._entries:
            if entry in cache and cache[entry] is None and os.path.isdir(entry):
                del cache[entry]
        stdlib = getattr(sys, "stdlib_module_names", ())
        for finder in () if name.partition(".")[0] in stdlib else self._editable:
            spec = finder.find_spec(name, path, target)
            if spec is not None and spec.loader is not None:
                # The cwd / script dir and anything else ahead of the session dirs, and for a submodule the
                # chosen parent's own __path__, still come first.
                first = min(i for i, e in enumerate(sys.path) if e in self._entries)
                ahead = sys.path[:first]
                if path is not None:
                    parent = name.rpartition(".")[0]
                    origin = getattr(
                        getattr(sys.modules.get(parent), "__spec__", None), "origin", None
                    )
                    if origin in (None, "namespace"):
                        # An aggregated namespace also lists Studio's portion: only those ahead count.
                        own = {os.path.abspath(os.path.join(e, *parent.split("."))) for e in ahead}
                        ahead = [p for p in path if os.path.abspath(p) in own]
                    else:
                        ahead = list(path)
                if ahead and self._path_finder.find_spec(name, ahead, target):
                    return None
                return spec
        if path is None:
            return None
        parent = getattr(sys.modules.get(name.rpartition(".")[0]), "__spec__", None)
        if parent is None or not hasattr(path, "append"):
            return None
        # Only a package the chat installed into: parsing every parent __init__ would tax all imports.
        portions = [
            p
            for p in (os.path.join(e, *parent.name.split(".")) for e in self._entries)
            if p not in list(path) and os.path.isdir(p)
        ]
        if not portions:
            return None
        if parent.origin not in (None, "namespace"):
            if "ast" not in sys.modules:
                # _extends_path imports ast: the stdlib one, never a chat-planted copy.
                saved = sys.path[:]
                sys.path[:] = _stdlib_paths()
                try:
                    import ast  # noqa: F401
                finally:
                    sys.path[:] = saved
            if not _HostPackagesFinder._extends_path(parent.origin):
                return None
        for portion in portions:
            path.append(portion)
        return None


_entries = _session_entries()
if _entries and not any(isinstance(f, _SessionPackagesFinder) for f in sys.meta_path):
    try:
        _added, _editable = _read_pth_files(_entries)
        _finder = _SessionPackagesFinder(_entries, _editable, _added)
        # Behind the built-in and frozen importers, so an editable mapping can never replace those modules.
        _at = next(
            (i for i, f in enumerate(sys.meta_path) if getattr(f, "__name__", "") == "PathFinder"),
            len(sys.meta_path),
        )
        sys.meta_path.insert(_at, _finder)
    except Exception:  # noqa: BLE001
        pass


# --ignore-installed never uninstalls, so a reinstall leaves stale dist-info importlib.metadata may pick;
# RECORD is written last, so the newest RECORD per project is the finished install.
_record_cache = {}
_real_dirs = {}


def _recorded_files(
    entry,
    dist_info,
    root,
    metadata = False,
):
    record = os.path.join(entry, dist_info, "RECORD")
    try:
        key = (record, root, metadata, os.stat(record).st_mtime_ns)
    except OSError:
        return set()
    if key in _record_cache:
        return _record_cache[key]
    files = set()
    # Resolved too: a recorded folder that is a symlink must not lead the cleanup outside the prefix.
    real_root = os.path.join(os.path.normcase(os.path.realpath(root)), "")
    try:
        with open(record, encoding = "utf-8") as f:
            for line in f:
                path = line.rstrip("\n").rsplit(",", 2)[0].strip('"')
                in_meta = path.startswith(dist_info + "/")
                if not path or (in_meta and (not metadata or path.endswith("/RECORD"))):
                    continue
                full = os.path.normcase(os.path.abspath(os.path.join(entry, path)))
                folder = os.path.dirname(full)
                if folder not in _real_dirs:
                    _real_dirs[folder] = os.path.join(
                        os.path.normcase(os.path.realpath(folder)), ""
                    )
                if full.startswith(root) and _real_dirs[folder].startswith(real_root):
                    files.add(full)
    except OSError:
        pass
    _record_cache[key] = files
    return files


def _claimed(root, skip):
    # Every site dir of the prefix (purelib and platlib can differ): a companion's RECORD may sit in the other.
    claimed = set()
    for entry in sys.path:
        if not entry or not _within(entry, root):
            continue
        try:
            names = [n for n in os.listdir(entry) if n.endswith(".dist-info")]
        except OSError:
            continue
        for name in names:
            if (entry, name) not in skip:
                claimed |= _recorded_files(entry, name, root, True)
    return claimed


def _remove_files(paths, entry):
    # As pip's uninstall: the files, then any folder they leave empty, never above the site dir.
    stop = os.path.normcase(os.path.abspath(entry))
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            continue
        parent = os.path.dirname(path)
        while parent.startswith(stop) and parent != stop:
            try:
                os.rmdir(parent)
            except OSError:
                break
            parent = os.path.dirname(parent)


def _prune_stale_dist_info():
    try:
        prefix = os.environ.get("PIP_PREFIX")
        if not prefix:
            return
        root = os.path.join(os.path.normcase(os.path.abspath(prefix)), "")
        # Grouped across every site dir of the prefix: purelib and platlib can hold two versions apart.
        projects = {}
        for entry in sys.path:
            if not entry or not _within(entry, root):
                continue
            try:
                names = os.listdir(entry)
            except OSError:
                continue
            for name in names:
                if name.endswith(".dist-info"):
                    try:
                        stamp = os.stat(os.path.join(entry, name, "RECORD")).st_mtime_ns
                    except OSError:
                        continue
                    project = (
                        name[:-10].rpartition("-")[0].lower().replace("-", "_").replace(".", "_")
                    )
                    projects.setdefault(project, {})[(entry, name)] = stamp
        stale_groups = [found for found in projects.values() if len(found) > 1]
        if not stale_groups:
            return
        saved = sys.path[:]
        sys.path[:] = _stdlib_paths()
        try:
            import shutil
        finally:
            sys.path[:] = saved
        for found in stale_groups:
            ordered = sorted(found, key = found.get)
            newest, stale = ordered[-1], ordered[:-1]
            # A copy: _recorded_files hands out its cached sets.
            keep = set(_recorded_files(newest[0], newest[1], root))
            # Files another surviving distribution now ships stay too.
            keep |= _claimed(root, set(stale))
            for entry, name in stale:
                # Also the files only the replaced version had, as pip's uninstall would remove.
                _remove_files(_recorded_files(entry, name, root) - keep, entry)
                shutil.rmtree(os.path.join(entry, name), ignore_errors = True)
    except Exception:  # noqa: BLE001 - a broken shim must never break user code
        pass


def _record_snapshot():
    # What each chat-local distribution's RECORD lists before this pip run: a same-version reinstall (an
    # editable over a wheel) rewrites RECORD in place, so the old file list is gone by the time pip exits.
    snapshot = {}
    try:
        root = os.path.join(os.path.normcase(os.path.abspath(os.environ["PIP_PREFIX"])), "")
        for entry in sys.path:
            if not entry or not _within(entry, root):
                continue
            try:
                names = [n for n in os.listdir(entry) if n.endswith(".dist-info")]
            except OSError:
                continue
            for name in names:
                try:
                    stamp = os.stat(os.path.join(entry, name, "RECORD")).st_mtime_ns
                except OSError:
                    continue
                snapshot[(entry, name)] = (stamp, _recorded_files(entry, name, root, True), root)
    except Exception:  # noqa: BLE001
        pass
    return snapshot


def _retire_rewritten(snapshot):
    for (entry, name), (stamp, files, root) in snapshot.items():
        try:
            if os.stat(os.path.join(entry, name, "RECORD")).st_mtime_ns == stamp:
                continue
        except OSError:
            continue
        _remove_files(files - _claimed(root, ()), entry)


if os.environ.get("PIP_PREFIX"):
    import atexit

    _prune_stale_dist_info()
    atexit.register(_prune_stale_dist_info)
    if _is_pip_launch():
        atexit.register(_retire_rewritten, _record_snapshot())


_host_packages = os.environ.get("_UNSLOTH_HOST_SESSION_PACKAGES")
if _host_packages and not any(isinstance(f, _HostPackagesFinder) for f in sys.meta_path):
    _folders = [f for f in _host_packages.split(os.pathsep) if f]
    try:
        # Path-line editables there too, behind the host finder's own boundaries; setuptools' finders would
        # register outside them, so those stay unsupported in host-access calls.
        _folders += _read_pth_files(_folders, on_sys_path = False)[0]
    except Exception:  # noqa: BLE001
        pass
    sys.meta_path.append(_HostPackagesFinder(os.pathsep.join(_folders)))


# Helpers come from the chat folder via a last-resort finder, not sys.path, since site imports
# usercustomize after this file and an earlier sandboxed call must not plant a module there.
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


# The static host check cannot see DNS aliases or redirects; every connect does see the final address.
_METADATA_NAMES = frozenset(
    (
        "metadata",
        "metadata.google.internal",
        "metadata.goog",
        "metadata.tencentyun.com",
        "instance-data.ec2.internal",
    )
)
_METADATA_ADDRS = frozenset(("fd00:ec2::254", "fd20:ce::254", "100.100.100.200", "100.100.100.110"))


def _metadata_address(host):
    import ipaddress

    if not isinstance(host, str):
        return False
    host = host.split("%", 1)[0].rstrip(".").lower()
    if host in _METADATA_NAMES:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if ip.version == 6:
        ip = ip.ipv4_mapped or ip.sixtofour or ip
    if ip.version == 4 and ip in ipaddress.ip_network("169.254.0.0/16"):
        return True
    return ip.compressed in _METADATA_ADDRS


def _resolves_to_metadata(host):
    import ipaddress
    import socket

    try:
        ipaddress.ip_address(host.split("%", 1)[0])
        return False
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None)
    except (OSError, UnicodeError):
        return False
    return any(_metadata_address(info[4][0]) for info in infos)


def _metadata_connect_hook(event, args):
    if event != "socket.connect":
        return
    address = args[1] if len(args) > 1 else None
    if not (isinstance(address, tuple) and address and isinstance(address[0], str)):
        return
    # The event carries the caller's address, so a raw connect to a name is resolved here.
    if _metadata_address(address[0]) or _resolves_to_metadata(address[0]):
        raise PermissionError(f"Blocked: cloud-metadata host ({address[0]})")


if os.environ.get("_UNSLOTH_SANDBOX_BLOCK_METADATA") == "1" and not getattr(
    sys, "_unsloth_metadata_hook", False
):
    sys.addaudithook(_metadata_connect_hook)
    sys._unsloth_metadata_hook = True
