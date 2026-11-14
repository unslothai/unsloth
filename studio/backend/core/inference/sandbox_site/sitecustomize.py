# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Remaps invented paths like /mnt/data onto CWD; reads heal only if the target exists; all wrapped
in try/except so a failure never breaks the interpreter; no redirect onto existing CWD file"""

import builtins
import io
import json
import os
import sys

# First, ahead of everything below: no later failure in this file can leave the hook unregistered.
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
        if "." not in name:
            # Appended last, it would otherwise supply a stdlib name this platform lacks (msvcrt on Linux).
            if path is not None or name in self._skip or name in _stdlib_names():
                return None
            return _PathFinder.find_spec(name, self._folders)
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
        found = _PathFinder.find_spec(name, portions)
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
    ("--python", 8),
)


# pip options that take a value (pip 26 install + general options, plus older pip's): their operand is no option.
_PIP_VALUE_OPTIONS = (
    "--abi --all-releases --build --build-constraint --build-option --cache-dir --cert --client-cert "
    "--config-settings --constraint --default-timeout --download --editable --exists-action "
    "--extra-index-url --find-links --global-option --group --implementation --index-url "
    "--install-option --keyring-provider --local-log --log --log-file --no-binary --only-binary "
    "--only-final --platform --prefix --progress-bar --proxy --pypi-url --python --python-version "
    "--refresh-package --report --requirement --requirements-from-script --resume-retries --retries "
    "--root --root-user-action --source --source-dir --source-directory --src --target --timeout "
    "--trusted-host --upgrade-strategy --uploaded-prior-to --use-deprecated --use-feature"
).split()
_PIP_VALUE_SHORTS = "Ccefirt"


# Exact flags that are also prefixes of a value option: optparse takes the exact match.
_PIP_PREFIX_FLAGS = ("--pre", "--upgrade")


def _pip_words():
    # (argv index, word, is option) over pip's command line; operands of value options (-C --prefix=x) skipped.
    args, i, command = sys.argv, 1, False
    while i < len(args):
        arg, at = args[i], i
        i += 1
        if arg == "--":
            if command or i >= len(args):
                return
            # pip --isolated -- install x: the word after a leading -- is still the subcommand.
            command = True
            yield at, arg, True
            yield i, args[i], False
            i += 1
            continue
        if arg.startswith("--"):
            yield at, arg, True
            name = arg.split("=", 1)[0]
            if (
                "=" not in arg
                and name not in _PIP_PREFIX_FLAGS
                and any(o.startswith(name) for o in _PIP_VALUE_OPTIONS)
            ):
                i += 1
        elif arg.startswith("-") and len(arg) > 1:
            yield at, arg, True
            for j, flag in enumerate(arg[1:], 1):
                if flag in _PIP_VALUE_SHORTS:
                    i += j == len(arg) - 1  # -C VALUE; -CVALUE carries it
                    break
        else:
            command = True
            yield at, arg, False


def _pip_options():
    return [arg for _, arg, option in _pip_words() if option]


def _cli_locations():
    # {target|prefix|root|python: value} as pip ends up with it: the last occurrence wins, an empty one included
    # (pip tests it for truthiness, so --target= after --target vendor chooses nothing).
    found, argv, command = {}, sys.argv + [""], False
    for at, arg, option in _pip_words():
        if not option:
            command = True
            continue
        name, eq, value = arg.partition("=")
        if name.startswith("--"):
            for full, n in _PIP_LOCATION_FLAGS:
                # --python is the main parser's: only before the subcommand does it re-run pip elsewhere.
                if (
                    full.startswith(name)
                    and len(name) >= n
                    and not (command and full == "--python")
                ):
                    found[full[2:]] = value if eq else argv[at + 1]
            continue
        # -t DIR, -tDIR, or clustered after no-operand flags (-qt DIR, -Ut).
        for j, flag in enumerate(arg[1:], 1):
            if flag == "t":
                found["target"] = arg[j + 1 :] or argv[at + 1]
            if flag not in "qvUI":
                break
    return found


def _cli_user():
    # --user / --no-user share one option, the last one given wins; None when neither is on the command line.
    chosen = None
    for arg in _pip_options():
        name = arg.split("=", 1)[0]
        if len(name) >= 6 and "--user".startswith(name):
            chosen = True
        elif len(name) >= 6 and "--no-user".startswith(name):
            chosen = False  # pip accepts unambiguous abbreviations (--no-u)
    return chosen


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
    # Python < 3.10 has no orig_argv, so a -m launch cannot be told apart (venv --upgrade-deps): only pip scripts.
    return module in ("pip", "pip.__main__") or (not module and _is_pip_script(sys.argv[0]))


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


_saved_path = sys.path[:]
sys.path[:] = _stdlib_paths()
try:
    # The stdlib copy, never an importlib/ the chat put on PYTHONPATH (this runs before any import of it).
    from importlib.machinery import PathFinder as _PathFinder
finally:
    sys.path[:] = _saved_path


def _config_chooses_location(
    isolated = False,
    use_user = True,
    skip = (),
):
    # pip's config files can set install.target / prefix / root / user too; PIP_PREFIX would then conflict.
    # Same files as pip's Configuration.iter_config_files: --isolated or an existing PIP_CONFIG_FILE drops the user ones.
    env_file = os.environ.get("PIP_CONFIG_FILE")
    if env_file == os.devnull:
        return False  # pip then loads no configuration file at all
    name = "pip.ini" if sys.platform == "win32" else "pip.conf"
    home = os.path.expanduser("~")
    if sys.platform == "win32":
        system = [os.path.join(os.environ.get("ProgramData", r"C:\ProgramData"), "pip")]
        user_dirs = [
            os.path.join(home, "pip"),
            os.path.join(os.environ.get("APPDATA", home), "pip"),
        ]
    else:
        if sys.platform == "darwin":
            system = ["/Library/Application Support/pip"]
            mac = os.path.join(home, "Library", "Application Support", "pip")
            config = mac if os.path.isdir(mac) else os.path.join(home, ".config", "pip")
        else:
            dirs = os.environ.get("XDG_CONFIG_DIRS") or "/etc/xdg"
            system = [os.path.join(d, "pip") for d in dirs.split(os.pathsep) if d] + ["/etc"]
            config = os.path.join(
                os.environ.get("XDG_CONFIG_HOME") or os.path.join(home, ".config"), "pip"
            )
        user_dirs = [os.path.join(home, ".pip"), config]

    def config_files(isolated):
        files = [os.path.join(d, name) for d in system]
        if not isolated and not (env_file and os.path.exists(env_file)):
            files += [os.path.join(d, name) for d in user_dirs]
        files += [os.path.join(sys.prefix, name), env_file or ""]
        return [f for f in files if f and f != os.devnull and os.path.isfile(f)]

    if not config_files(False):
        return False
    saved = sys.path[:]
    sys.path[:] = _stdlib_paths()
    try:
        import configparser
        import locale
    finally:
        sys.path[:] = saved
    # pip's Configuration._construct_parser: UTF-8 only in UTF-8 mode, else the locale's encoding.
    encoding = "utf-8" if sys.flags.utf8_mode else locale.getpreferredencoding(False)

    def load(files):
        # One parser per file as pip's Configuration._load_file: a [DEFAULT] applies inside its own file only;
        # keys normalised as its _normalize_name (lower case, _ -> -, a leading -- dropped).
        merged = {}
        for path in files:
            parser = configparser.RawConfigParser()
            parser.read(path, encoding = encoding)
            for section in parser.sections():
                items = {
                    k.lower().replace("_", "-").removeprefix("--"): v
                    for k, v in parser.items(section)
                }
                merged.setdefault(section, {}).update(items)
        return merged

    try:
        merged = load(config_files(isolated))
        # The main parser picks --python up before the install command is isolated: from the full set.
        full = load(config_files(False)) if isolated else merged
    except (configparser.Error, UnicodeDecodeError):
        return False

    # pip's precedence per option: [global], then [install], then a PIP_* variable (ignored under --isolated);
    # empty values are skipped, as pip's _get_ordered_configuration_items does.
    def value(key):
        env = "" if isolated else os.environ.get("PIP_" + key.upper(), "")
        if env.strip() and env != os.environ.get("_UNSLOTH_PIP_PREFIX"):
            return env
        for section in ("install", "global"):
            found = merged.get(section, {}).get(key, "")
            if found.strip():
                return found
        return ""

    if any(value(k).strip() for k in ("target", "prefix", "root") if k not in skip):
        return True
    # python: pip re-runs itself in that interpreter, as --python does, which installs there.
    # [global] only: that is the main parser's section ([install] python is never read for delegation).
    if "python" not in skip and full.get("global", {}).get("python", "").strip():
        return True
    items = []
    for section in ("global", "install"):
        items += list(merged.get(section, {}).items())
    return use_user and _user_site(items, isolated)


def _user_site(items = (), isolated = False):
    # user and no-user set one option (pip assigns either key's boolean as given); the last non-empty one in
    # pip's order ([global], [install], then PIP_USER / PIP_NO_USER) wins.
    if not isolated:
        items = list(items) + [
            (k[4:].lower().replace("_", "-"), v)
            for k, v in os.environ.items()
            if k.startswith("PIP_")
        ]
    chosen = False
    for key, val in items:
        if key in ("user", "no-user") and val.strip():
            chosen = val.strip().lower() in ("y", "yes", "t", "true", "on", "1")
    return chosen


_chat_prefix = None


class _LatePipFinder:
    # Python < 3.10 without orig_argv: a -m launch is pip only once runpy asks for pip.__main__.
    def find_spec(
        self,
        name,
        path = None,
        target = None,
    ):
        if name == "pip.__main__":
            sys.meta_path[:] = [f for f in sys.meta_path if f is not self]
            try:
                if os.environ.get("PIP_PREFIX"):
                    _track_pip()  # a prefix set before start (OS-isolated launches): cleanup already began
                else:
                    _route_pip(True)
                    if os.environ.get("PIP_PREFIX"):
                        _start_cleanup(True)
            except Exception:  # noqa: BLE001
                pass
        return None


def _late_pip_launch():
    # Python < 3.10 has no orig_argv: a -m launch is told apart only when runpy asks for pip.__main__.
    if hasattr(sys, "orig_argv") or sys.argv[:1] != ["-m"]:
        return False
    if not any(isinstance(f, _LatePipFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _LatePipFinder())
    return True


def _route_pip(is_pip = None):
    try:
        prefix = os.environ.get("_UNSLOTH_PIP_PREFIX")
        owner = os.environ.get("_UNSLOTH_PIP_PREFIX_FOR")
        if not prefix or not owner:
            return

        def key(path):
            return os.path.normcase(os.path.realpath(path))

        version = "%d.%d" % sys.version_info[:2]
        # Another minor version sharing the prefix (/usr/local) would install into a lib/pythonX.Y never exposed.
        if (
            key(sys.prefix) == key(owner)
            and os.environ.get("_UNSLOTH_PIP_PREFIX_PY", version) == version
        ):
            global _chat_prefix
            _chat_prefix = prefix
            # Only a pip process gets PIP_PREFIX: anything else it spawns (venv --upgrade-deps) must not inherit it.
            if is_pip is None:
                is_pip = _is_pip_launch()
                if not is_pip:
                    _late_pip_launch()
            # pip --target/--prefix/--user/... already chose a location; PIP_PREFIX would conflict with it.
            cli_user = _cli_user()
            options = _pip_options()
            # A command-line value overrides the same option's variable and config entry.
            cli = _cli_locations()
            placed = cli_user is True or any(cli.values())
            # pip's main parser picks PIP_PYTHON up even under --isolated and re-runs pip in that interpreter.
            placed = placed or ("python" not in cli and bool(os.environ.get("PIP_PYTHON")))
            # pip's own trigger is the raw word anywhere after the subcommand, even as an operand (--log --isolated).
            isolated = "--isolated" in options or "--isolated" in sys.argv[1:]
            if not isolated:
                # --isolated makes pip ignore these, so then they choose nothing.
                placed = placed or any(
                    os.environ.get("PIP_" + k.upper()) for k in ("target", "root") if k not in cli
                )
                placed = placed or (cli_user is None and _user_site())
                own = os.environ.get("PIP_PREFIX", prefix) not in ("", prefix)
                placed = placed or ("prefix" not in cli and own)
            placed = placed or (
                is_pip and _config_chooses_location(isolated, cli_user is None, set(cli))
            )
            if is_pip and not placed:
                os.environ["PIP_PREFIX"] = prefix
                os.environ["PIP_IGNORE_INSTALLED"] = "1"
                words = list(_pip_words())
                command = next((at for at, arg, option in words if not option), None)
                if isolated and command and sys.argv[command] == "install":
                    # Only a terminator -- (not one a value option took as its operand, --log --).
                    if (command - 1, "--", True) in words:
                        # Else pip hands install the -- too and takes what follows for requirements.
                        del sys.argv[command - 1]
                        command -= 1
                    # --isolated ignores PIP_* variables: pass the same options on the command line.
                    sys.argv[command + 1 : command + 1] = ["--prefix", prefix, "--ignore-installed"]
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


def _install_editable_finder(
    module,
    mapping,
    namespaces,
    finder_file = None,
):
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
    # The finder registers on its own behind PathFinder: a stdlib name (msvcrt on Linux) is never mapped.
    stdlib = _stdlib_names()
    mapping = {k: v for k, v in sorted(mapping.items()) if k.partition(".")[0] not in stdlib}
    namespaces = {k: v for k, v in namespaces.items() if k.partition(".")[0] not in stdlib}
    code = _finder_template[0].format(name = module, mapping = mapping, namespaces = namespaces)
    import types

    finder_module = types.ModuleType(module)
    exec(compile(code, "<" + module + ">", "exec"), finder_module.__dict__)
    finder = finder_module._EditableFinder
    if finder_file:
        # Retired once its finder file is gone (the editable was replaced or uninstalled mid-process).
        lookup = finder.find_spec

        def find_spec(
            fullname,
            path = None,
            target = None,
        ):
            return lookup(fullname, path, target) if os.path.exists(finder_file) else None

        finder.find_spec = staticmethod(find_spec)
    sys.modules[module] = finder_module
    finder_module.install()
    return finder


# PYTHONPATH skips .pth files: take their path lines and setuptools' editable finders, never run their imports.
# Paths go right after the session dirs, so a chat's editable fork wins over Studio's copy like its other installs.
def _read_pth(path):
    # As site.addpackage: UTF-8 first, else the locale's encoding (setuptools wrote it in that one before 3.13).
    with open(path, "rb") as f:
        data = f.read()
    try:
        return data.decode("utf-8-sig").splitlines()
    except UnicodeDecodeError:
        saved = sys.path[:]
        sys.path[:] = _stdlib_paths()
        try:
            import locale
        finally:
            sys.path[:] = saved
        return data.decode(locale.getpreferredencoding(False)).splitlines()


def _read_pth_files(entries, on_sys_path = True):
    added, finders = [], []
    for entry in entries:
        try:
            names = sorted(n for n in os.listdir(entry) if n.endswith(".pth"))
        except OSError:
            continue
        for name in names:
            try:
                lines = _read_pth(os.path.join(entry, name))
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
                    parsed.append((module, data, os.path.join(entry, module + ".py")))
            for module, (mapping, namespaces), source in parsed:
                try:
                    installed.append(_install_editable_finder(module, mapping, namespaces, source))
                except Exception:  # noqa: BLE001
                    pass
        finally:
            sys.path[:] = saved + [e for e in sys.path if e not in saved]
    if added and on_sys_path:
        # After the stdlib too (the session dirs ride PYTHONPATH ahead of it), still ahead of site-packages.
        stdlib = set(_stdlib_paths())
        at = max(i for i, e in enumerate(sys.path) if e in entries or e in stdlib) + 1
        sys.path[at:at] = added
    return added, [f for f in installed if f is not None]


_stdlib = None


def _stdlib_names():
    # sys.stdlib_module_names is 3.10+; before that, what the interpreter's own stdlib dirs hold.
    global _stdlib
    if _stdlib is None:
        _stdlib = set(getattr(sys, "stdlib_module_names", ())) | set(sys.builtin_module_names)
        if not hasattr(sys, "stdlib_module_names"):
            # Modules only some platforms build (no file here to list): what 3.10's inventory names.
            _stdlib.update(
                "msvcrt winreg winsound _winapi _overlapped nt _msi msilib posix pwd grp fcntl termios "
                "resource _posixsubprocess _posixshmem spwd crypt _crypt nis syslog ossaudiodev readline "
                "curses _curses _curses_panel _scproxy _dbm _gdbm _tkinter _uuid _ctypes _lzma _bz2 "
                "_sqlite3 _ssl _hashlib _decimal _multiprocessing audioop".split()
            )
            for folder in _stdlib_paths():
                try:
                    names = os.listdir(folder)
                except OSError:
                    continue
                _stdlib.update(n.partition(".")[0] for n in names if n.isidentifier() or "." in n)
    return _stdlib


def _pkgutil_namespace(origin):
    if not origin or not origin.endswith("__init__.py"):
        return False
    if "ast" not in sys.modules:
        # _extends_path imports ast: the stdlib one, never a chat-planted copy.
        saved = sys.path[:]
        sys.path[:] = _stdlib_paths()
        try:
            import ast  # noqa: F401
        finally:
            sys.path[:] = saved
    return _HostPackagesFinder._extends_path(origin)


# First on meta_path: a session dir pip creates mid-process is cached as missing, a namespace imported before an
# install lacks the new portion, and an editable fork's concrete module wins over Studio's copy (namespaces are
# left to the normal aggregation, which the editable finders still join from their usual place).
class _SessionPackagesFinder:
    _path_finder = _PathFinder

    def __init__(
        self,
        entries,
        editable,
        chat_paths = (),
    ):
        self._entries = entries
        self._editable = editable
        self._added = set(chat_paths)
        self._chat_paths = set(entries) | self._added

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
        if path is None and self._added and name in _stdlib_names():
            # A path-line editable root must not supply a stdlib name either (msvcrt.py on Linux).
            rest = [e for e in sys.path if e not in self._added]
            spec = self._path_finder.find_spec(name, rest, target)
            if spec is None:
                raise ModuleNotFoundError(f"No module named {name!r}", name = name)
            return spec
        for finder in () if name.partition(".")[0] in _stdlib_names() else self._editable:
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
                    if origin in (None, "namespace") or _pkgutil_namespace(origin):
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
        parts = parent.name.split(".")
        roots = [*self._entries, *self._added]  # path-line editable roots hold portions too
        chat = [p for p in (os.path.join(e, *parts) for e in roots) if os.path.isdir(p)]
        if not chat:
            return None
        listed = {os.path.abspath(p) for p in path}
        portions = [p for p in chat if os.path.abspath(p) not in listed]
        # A regular package keeps its own __path__ order (it may prepend a preferred dir): pkgutil ones only.
        if parent.origin not in (None, "namespace") and not _pkgutil_namespace(parent.origin):
            return None
        for portion in portions:
            path.append(portion)
        # A late portion sits last in __path__: search it in sys.path order, as a fresh start would, for every child.
        rank = {}
        for i, e in enumerate(sys.path):
            rank.setdefault(os.path.abspath(os.path.join(e, *parts)), i)
        ordered = sorted(path, key = lambda p: rank.get(os.path.abspath(p), len(rank)))
        if ordered != list(path):
            try:
                path[:] = ordered  # the stored __path__ too: importlib.resources reads it
            except Exception:  # noqa: BLE001
                pass
        return self._path_finder.find_spec(name, ordered, target)


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
    if "csv" not in sys.modules:
        saved = sys.path[:]
        sys.path[:] = _stdlib_paths()
        try:
            import csv  # noqa: F401
        finally:
            sys.path[:] = saved
    try:
        with open(record, encoding = "utf-8", newline = "") as f:
            # RECORD is CSV: a quoted name ("a""b") must name that file, not a neighbour.
            for row in sys.modules["csv"].reader(f):
                path = row[0] if row else ""
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
    except Exception:  # noqa: BLE001 - unreadable or malformed: nothing listed, nothing removed
        pass
    _record_cache[key] = files
    return files


_INCOMPLETE = ".unsloth-incomplete"
_incomplete = []


def _failed_installs():
    # (site dir, dist-info, RECORD mtime) a failed pip run left behind, kept beside the prefix's site dirs.
    if not _incomplete:
        found = set()
        try:
            with open(os.path.join(os.environ["PIP_PREFIX"], _INCOMPLETE), encoding = "utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) == 3:
                        found.add(tuple(parts))
        except (KeyError, OSError, ValueError):
            pass
        found |= _killed_installs()
        _incomplete.append(found)
    return _incomplete[0]


_PENDING = ".unsloth-installing-"


# Per run, not per pid: bwrap's --unshare-pid makes every tool call PID 2. A run holds an flock on each of its
# notes until it exits, which any later process sees whatever its pid namespace.
_RUN_ID = "%d-%s" % (os.getpid(), os.urandom(6).hex())
_held = []


def _fcntl():
    if sys.platform == "win32":
        return None
    saved = sys.path[:]
    sys.path[:] = _stdlib_paths()
    try:
        import fcntl
        return fcntl
    except ImportError:
        return None
    finally:
        sys.path[:] = saved


def _write_held(path, lines):
    # Written and locked under a temporary name, then renamed: no other process ever sees it unlocked.
    fcntl = _fcntl()
    if fcntl is None:
        # Windows: an open CRT handle blocks the rename, and there is no flock to hold anyway.
        with open(path, "w", encoding = "utf-8") as f:
            f.writelines(lines)
        return None
    temp = f"{path}.tmp"
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        if fcntl:
            fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, "".join(lines).encode("utf-8"))
        os.replace(temp, path)
    except BaseException:
        os.close(fd)
        raise
    _held.append(fd)
    return fd


def _release(fd, path):
    try:
        os.remove(path)
    except OSError:
        pass
    if fd is not None and fd in _held:
        _held.remove(fd)
        os.close(fd)


def _alive(path):
    # Windows has no flock here; a concurrent pip in one chat folder is not expected there.
    fcntl = _fcntl()
    if fcntl is None:
        return False
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True
    finally:
        os.close(fd)
    return False


def _killed_installs():
    # A pip killed mid-wheel (tool timeout, SIGKILL) runs no handler: its pending note, written before the
    # wheel, names the RECORD mtimes from before; whatever differs now is that wheel's half-written state.
    prefix = os.environ.get("PIP_PREFIX", "")
    found = set()
    try:
        names = [n for n in os.listdir(prefix) if n.startswith(_PENDING)]
    except OSError:
        return found
    for name in names:
        run = name[len(_PENDING) :]
        if not run or run == _RUN_ID or run.endswith(".tmp"):
            continue
        path = os.path.join(prefix, name)
        # A live one is another pip mid-wheel: what it changed is unfinished for now, its note stays.
        live = _alive(path)
        try:
            with open(path, encoding = "utf-8") as f:
                rows = [line.rstrip("\n").split("\t") for line in f]
            before = {(r[0], r[1]): r[2] for r in rows if len(r) == 3}
            now = {(os.path.abspath(e), n): str(v) for (e, n), v in _dist_stamps().items()}
            lines = [f"{e}\t{n}\t{v}\n" for (e, n), v in now.items() if before.get((e, n)) != v]
            found.update(tuple(line.rstrip("\n").split("\t")) for line in lines)
            if live:
                continue
            if lines:
                with open(os.path.join(prefix, _INCOMPLETE), "a", encoding = "utf-8") as f:
                    f.writelines(lines)
            os.remove(path)
        except (OSError, ValueError):
            pass
    # A dead run's pre-install file lists, whether it died inside a wheel or between two: kept for the next
    # retirement (lists of untouched distributions only repeat what their RECORD still says).
    for name in os.listdir(prefix):
        run = name[len(_SAVED) + 1 :] if name.startswith(_SAVED + "-") else ""
        if (
            run
            and run != _RUN_ID
            and not run.endswith(".tmp")
            and not _alive(os.path.join(prefix, name))
        ):
            try:
                with open(os.path.join(prefix, name), encoding = "utf-8") as f:
                    rows = f.readlines()
                with open(os.path.join(prefix, _SAVED), "a", encoding = "utf-8") as f:
                    f.writelines(rows)
                os.remove(os.path.join(prefix, name))
            except (OSError, ValueError):
                pass
    return found


def _finished(entry, dist_info):
    # pip and uv record INSTALLER in the RECORD they write last; a wheel's own RECORD (copied in
    # mid-install) never lists it, so an install that died halfway does not count as a replacement.
    record = os.path.join(entry, dist_info, "RECORD")
    try:
        stamp = str(os.stat(record).st_mtime_ns)
    except OSError:
        return False
    if (os.path.abspath(entry), dist_info, stamp) in _failed_installs():
        return False  # a wheel that ships INSTALLER itself, from a pip run that failed
    try:
        with open(record, encoding = "utf-8") as f:
            return any(line.startswith(f"{dist_info}/INSTALLER,") for line in f)
    except OSError:
        return False


def _site_dirs(root):
    # The prefix's site dirs on sys.path, once per real folder (lib64 -> lib) and only those really inside it.
    real = os.path.join(os.path.normcase(os.path.realpath(root)), "")
    found, seen = [], set()
    for entry in sys.path:
        if not entry or not _within(entry, root):
            continue
        key = os.path.join(os.path.normcase(os.path.realpath(entry)), "")
        if key.startswith(real) and key not in seen:
            seen.add(key)
            found.append(entry)
    return found


def _claimed(root, skip):
    # Every site dir of the prefix (purelib and platlib can differ): a companion's RECORD may sit in the other.
    claimed = set()
    for entry in _site_dirs(root):
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
    real = os.path.join(
        os.path.normcase(os.path.realpath(os.environ.get("PIP_PREFIX") or entry)), ""
    )
    for path in paths:
        # Re-resolved now: a folder replaced by a symlink since the list was taken leads nowhere outside.
        folder = os.path.join(os.path.normcase(os.path.realpath(os.path.dirname(path))), "")
        if not folder.startswith(real):
            continue
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
    _incomplete.clear()  # live pending notes are re-read on every pass
    try:
        prefix = os.environ.get("PIP_PREFIX")
        if not prefix:
            return
        root = os.path.join(os.path.normcase(os.path.abspath(prefix)), "")
        real = os.path.join(os.path.normcase(os.path.realpath(prefix)), "")
        # Grouped across every site dir of the prefix: purelib and platlib can hold two versions apart.
        projects = {}
        for entry in _site_dirs(root):
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
        saved = _saved_inventory()
        count = sum(map(len, saved.values()))
        for found in stale_groups:
            ordered = sorted(found, key = found.get)
            done = [d for d in ordered if _finished(*d)]
            if not done:
                continue
            newest = done[-1]
            stale = ordered[: ordered.index(newest)]
            if not stale:
                continue
            # A copy: _recorded_files hands out its cached sets.
            keep = set(_recorded_files(newest[0], newest[1], root))
            # Files another surviving distribution now ships stay too.
            keep |= _claimed(root, set(stale))
            for entry, name in stale:
                # Also the files only the replaced version had, as pip's uninstall would remove; a failed
                # same-version rewrite's saved list too, its RECORD no longer names them.
                old = _recorded_files(entry, name, root) | saved.pop(
                    (os.path.abspath(entry), name), set()
                )
                _remove_files(old - keep, entry)
                if _within(os.path.realpath(os.path.join(entry, name)), real):
                    shutil.rmtree(os.path.join(entry, name), ignore_errors = True)
        if sum(map(len, saved.values())) != count:
            _write_saved(saved)
    except Exception:  # noqa: BLE001 - a broken shim must never break user code
        pass


_SAVED = ".unsloth-old-files"


def _saved_inventory():
    # Old file lists of same-version reinstalls that failed after rewriting RECORD: (site dir, dist-info) -> files.
    saved = {}
    try:
        prefix = os.environ["PIP_PREFIX"]
        root = os.path.join(os.path.normcase(os.path.abspath(prefix)), "")
        real = os.path.join(os.path.normcase(os.path.realpath(prefix)), "")
        with open(os.path.join(prefix, _SAVED), encoding = "utf-8") as f:
            for line in f:
                row = line.rstrip("\n").split("\t")
                # Re-checked now, as _recorded_files does: a folder since replaced by a symlink leads nowhere outside.
                folder = os.path.join(
                    os.path.normcase(os.path.realpath(os.path.dirname(row[-1]))), ""
                )
                if len(row) == 3 and row[2].startswith(root) and folder.startswith(real):
                    saved.setdefault((row[0], row[1]), set()).add(row[2])
    except (KeyError, OSError, ValueError):
        pass
    return saved


def _write_saved(saved):
    path = os.path.join(os.environ["PIP_PREFIX"], _SAVED)
    try:
        if not saved:
            if os.path.exists(path):
                os.remove(path)
            return
        with open(path, "w", encoding = "utf-8") as f:
            f.writelines(
                f"{e}\t{n}\t{p}\n" for (e, n), files in saved.items() for p in sorted(files)
            )
    except OSError:
        pass


def _record_snapshot():
    # What each chat-local distribution's RECORD lists before this pip run: a same-version reinstall (an
    # editable over a wheel) rewrites RECORD in place, so the old file list is gone by the time pip exits.
    snapshot = {}
    try:
        root = os.path.join(os.path.normcase(os.path.abspath(os.environ["PIP_PREFIX"])), "")
        saved = _saved_inventory()
        for entry in _site_dirs(root):
            try:
                names = [n for n in os.listdir(entry) if n.endswith(".dist-info")]
            except OSError:
                continue
            for name in names:
                try:
                    stamp = os.stat(os.path.join(entry, name, "RECORD")).st_mtime_ns
                except OSError:
                    continue
                files = _recorded_files(entry, name, root, True)
                files = files | saved.get((os.path.abspath(entry), name), set())
                snapshot[(entry, name)] = (stamp, files, root)
    except Exception:  # noqa: BLE001
        pass
    return snapshot


def _retire_rewritten(snapshot, skip = ()):
    saved = _saved_inventory()
    for (entry, name), (stamp, files, root) in snapshot.items():
        if (entry, name) in skip:
            continue
        try:
            if os.stat(os.path.join(entry, name, "RECORD")).st_mtime_ns == stamp:
                continue
            if not _finished(entry, name):
                continue
        except OSError:
            continue
        _remove_files(files - _claimed(root, ()), entry)
        saved.pop((os.path.abspath(entry), name), None)
    # A failed rewrite keeps the pre-run list, so a successful retry still retires what only the old one had.
    for key in skip:
        if key in snapshot:
            saved.setdefault((os.path.abspath(key[0]), key[1]), set()).update(snapshot[key][1])
    _write_saved(saved)


_pip_status = []


_failed_wheels = []


def _dist_stamps():
    # RECORD mtime of every dist-info in the prefix's site dirs.
    root = os.path.join(os.path.normcase(os.path.abspath(os.environ["PIP_PREFIX"])), "")
    stamps = {}
    for entry in _site_dirs(root):
        try:
            names = os.listdir(entry)
        except OSError:
            continue
        for name in names:
            if name.endswith(".dist-info"):
                try:
                    stamps[(entry, name)] = os.stat(os.path.join(entry, name, "RECORD")).st_mtime_ns
                except OSError:
                    pass
    return stamps


def _track_status(entry):
    def wrapped(*args, **kwargs):
        try:
            status = entry(*args, **kwargs)
        except BaseException:
            _pip_status.append(1)
            raise
        _pip_status.append(status or 0)
        return status

    return wrapped


def _track_wheel(entry):
    # The dist-info one wheel left half-written when its install raised: other wheels of the run are finished.
    def wrapped(*args, **kwargs):
        before = _dist_stamps()
        prefix = os.environ["PIP_PREFIX"]
        pending, held, written = os.path.join(prefix, f"{_PENDING}{_RUN_ID}"), None, False
        try:
            # Only into an existing prefix: nothing to protect before the first install, and the path remap
            # would move a write under a missing folder into the CWD.
            if before and os.path.isdir(prefix):
                rows = [f"{os.path.abspath(e)}\t{n}\t{v}\n" for (e, n), v in before.items()]
                held, written = _write_held(pending, rows), True
        except OSError:
            pass
        try:
            return entry(*args, **kwargs)
        except BaseException:
            _failed_wheels.append({k: v for k, v in _dist_stamps().items() if before.get(k) != v})
            raise
        finally:
            if written:
                _release(held, pending)

    return wrapped


class _PipStatusFinder:
    # Records what pip's main returns and which wheel failed, so a failed install retires nothing: its RECORD
    # alone cannot tell (a wheel may ship INSTALLER itself). Wraps attributes after pip's modules load.
    _path_finder = _PathFinder

    # cli: console scripts and python -m pip (pip >= 25.1); main: older pip. install_wheel: one wheel.
    _patches = {
        "pip._internal.cli.main": (("cli", "main"), _track_status),
        "pip._internal.operations.install.wheel": (("install_wheel",), _track_wheel),
    }

    def find_spec(
        self,
        name,
        path = None,
        target = None,
    ):
        if name not in self._patches:
            return None
        spec = self._path_finder.find_spec(name, path, target)
        loader = getattr(spec, "loader", None)
        if loader is None or not hasattr(loader, "exec_module"):
            return spec
        run, (attrs, wrap) = loader.exec_module, self._patches[name]

        def exec_module(module):
            run(module)
            for attr in attrs:
                if callable(getattr(module, attr, None)):
                    setattr(module, attr, wrap(getattr(module, attr)))

        loader.exec_module = exec_module
        return spec


def _after_pip(snapshot):
    try:
        _after_pip_run(snapshot)
    except Exception:  # noqa: BLE001
        pass


def _after_pip_run(snapshot):
    # Only once pip's main() returned 0: an install that failed halfway leaves the old version's files alone,
    # and what it wrote is remembered so a later process does not take it for a finished replacement.
    if not _pip_status:
        return
    if all(s == 0 for s in _pip_status):
        _retire_rewritten(snapshot)
        _prune_stale_dist_info()
        return
    try:
        if _failed_wheels:
            failed = {k: v for found in _failed_wheels for k, v in found.items()}
        else:
            # A pip without install_wheel to wrap: everything this run rewrote.
            failed = {k: v for k, v in _dist_stamps().items() if snapshot.get(k, (None,))[0] != v}
        lines = [f"{os.path.abspath(e)}\t{n}\t{stamp}\n" for (e, n), stamp in failed.items()]
        if lines:
            with open(
                os.path.join(os.environ["PIP_PREFIX"], _INCOMPLETE), "a", encoding = "utf-8"
            ) as f:
                f.writelines(lines)
        _failed_installs().update((os.path.abspath(e), n, str(v)) for (e, n), v in failed.items())
        if _failed_wheels:
            # Wheels that finished before the failing one are retired / pruned as on success.
            _retire_rewritten(snapshot, set(failed))
            _prune_stale_dist_info()
    except Exception:  # noqa: BLE001
        pass


def _start_cleanup(is_pip):
    # Chat-writable bookkeeping: whatever it holds, it must not stop this file before the audit hook below.
    try:
        import atexit

        _failed_installs()  # a killed pip's note is read before anything else changes the prefix
        _prune_stale_dist_info()
        if is_pip:
            _track_pip()
        elif _late_pip_launch():
            atexit.register(lambda: _pip_tracked or _prune_stale_dist_info())
        else:
            atexit.register(_prune_stale_dist_info)
    except Exception:  # noqa: BLE001
        pass


_pip_tracked = []


def _track_pip():
    import atexit

    _pip_tracked.append(True)
    sys.meta_path.insert(0, _PipStatusFinder())
    snapshot = _record_snapshot()
    # The file lists on disk too: a SIGKILL mid-reinstall leaves only what is written down.
    prefix = os.environ["PIP_PREFIX"]
    inventory = os.path.join(prefix, f"{_SAVED}-{_RUN_ID}")
    if snapshot and os.path.isdir(prefix):
        try:
            rows = [
                f"{os.path.abspath(entry)}\t{name}\t{p}\n"
                for (entry, name), (_, files, _) in snapshot.items()
                for p in sorted(files)
            ]
            atexit.register(_release, _write_held(inventory, rows), inventory)
        except OSError:
            pass
    atexit.register(_after_pip, snapshot)


if os.environ.get("PIP_PREFIX"):
    _start_cleanup(_is_pip_launch())


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
        spec = _PathFinder.find_spec(name, [self._workdir])
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
