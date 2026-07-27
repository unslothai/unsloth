# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Guard: shipping code must name an encoding on every text read and write.

`Path.read_text()`, `Path.write_text()`, `Path.open()` and builtin `open()` fall
back to `locale.getencoding()` when no encoding is given: UTF-8 on the Linux and
macOS runners, cp1252 on a stock Windows install. Every file this repo reads at
runtime is UTF-8 -- HF `config.json` / `tokenizer_config.json` / `adapter_config.json`
(RFC 8259 mandates UTF-8 for JSON), Ollama manifests, GGUF export metadata -- so on
Windows those reads either crash or, worse, succeed with mojibake:

    A DeepSeek or Qwen tokenizer_config.json carries U+FF5C and U+2581 in its
    chat template. Under cp1252 that read raises UnicodeDecodeError, and at
    utils/models/model_config.py the call sits inside a broad `except Exception:
    logger.debug(...)`, so the token-pattern check silently returned the wrong
    answer with no visible error.

Unlike the import-time rule in test_source_read_encoding.py this is scope
agnostic, which makes it both simpler and stricter: runtime reads live inside
functions, and shipping code has no legitimate reason to want the operator's
locale to decide how a file is decoded. That leaves no reachability analysis to
get wrong, so there is no allowlist and no false positives.

Binary handles are skipped: they have no encoding to name, and passing one is a
ValueError. A non-constant mode is treated as unknown rather than assumed text,
for the same reason -- demanding `encoding =` on a call that may resolve to "rb"
would leave no compliant way to write it.

Known limitation, deliberately not closed: `configparser.ConfigParser.read()`
also takes an `encoding` and also defaults to the locale one, but it cannot be
matched by name without resolving the receiver. `f.read(n)` on a binary handle,
`resp.read(limit)` on an HTTP response and `handle.read(chunk)` are all spelled
identically, and flagging those would be a false positive with no compliant fix
-- the exact failure mode this guard is built to avoid. The one live
`ConfigParser.read` in the tree (/etc/wsl.conf, hub/utils/paths.py) names its
encoding; a future one has to be caught in review.
"""

# `str | None` below is evaluated at import on Python 3.9 without this, and
# pyproject declares requires-python = ">=3.9,<3.15".
from __future__ import annotations

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
# Everything that ships to users. `studio/` covers the installer scripts as
# well as the backend: install_python_stack.py reads /sys/class/kfd to decide
# whether to pull ROCm wheels, which is the same detection path as
# utils/hardware/hardware.py and deserves the same rule. Test trees are covered
# by test_source_read_encoding.py under a narrower import-time rule.
ROOTS = (REPO / "unsloth", REPO / "studio", REPO / "unsloth_cli")
# The frontend tree is TypeScript; node_modules is vendored third-party code.
SKIP_DIRS = {"frontend", "node_modules", "src-tauri", ".venv", "site-packages"}
GUARDED_METHODS = {"read_text", "write_text"}
# Path classes, so an unbound `Path.open(p)` shifts every argument one right.
PATH_CLASSES = {"Path", "PosixPath", "PurePath", "WindowsPath"}
# Modules whose `open` IS the builtin: same signature, same platform default.
BUILTIN_OPEN_MODULES = {"builtins", "io"}
# These wrap the stream in a TextIOWrapper for a "t" mode, so they take an
# encoding, but default to "rb". Value is where that encoding sits positionally.
COMPRESSED_OPENERS = {"bz2": 3, "gzip": 3, "lzma": None}
# Distinct from None so that "no mode argument at all" still means text.
UNKNOWN_MODE = object()


def _mode(call: ast.Call, positional_index: int):
    """The call's mode, or UNKNOWN_MODE when it is not a literal."""
    # open(*args) / open(path, **kw) hide the mode, so it is unknown rather
    # than absent. Falling through to the "r" default would flag a call that
    # may resolve to binary, leaving no compliant way to write it.
    if any(isinstance(a, ast.Starred) for a in call.args):
        return UNKNOWN_MODE
    if any(kw.arg is None for kw in call.keywords):
        return UNKNOWN_MODE
    if len(call.args) > positional_index:
        node = call.args[positional_index]
        return node.value if isinstance(node, ast.Constant) else UNKNOWN_MODE
    for kw in call.keywords:
        if kw.arg == "mode":
            return kw.value.value if isinstance(kw.value, ast.Constant) else UNKNOWN_MODE
    return "r"


def _names_encoding(call: ast.Call) -> bool:
    """True only for an encoding that actually pins one.

    `encoding = None` and `encoding = "locale"` both re-select the platform
    default, so the keyword being present is not enough. A `**kwargs` splat may
    carry an encoding we cannot see, so it counts as named rather than risking
    a demand the contributor has no way to satisfy.
    """
    for kw in call.keywords:
        if kw.arg is None:
            return True
        if kw.arg != "encoding":
            continue
        if isinstance(kw.value, ast.Constant) and kw.value.value in (None, "locale"):
            return False
        return True
    return False


def _is_text(call: ast.Call, positional_index: int) -> bool:
    mode = _mode(call, positional_index)
    return mode is not UNKNOWN_MODE and "b" not in str(mode)


def _imported_names(tree: ast.Module) -> dict:
    """Names this module's imports bind, mapped to where they came from.

    The name alone settles nothing in either direction: `import tarfile as tf`
    hides an opener that takes no encoding behind an unfamiliar name, and
    `from PIL.Image import open` puts one behind the most familiar name there
    is. Resolving the origin covers both without a list of module names to keep
    up to date.
    """
    bound = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                bound[(a.asname or a.name).split(".")[0]] = a.name
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                bound[a.asname or a.name] = f"{node.module}.{a.name}" if node.module else a.name
    return bound


def _origin_root(name, modules) -> str:
    """The top-level module a bound name came from, or the name itself."""
    return modules.get(name, name).split(".")[0]


def _compressed_key(name, modules):
    """The COMPRESSED_OPENERS entry this receiver resolves to, if any."""
    for candidate in (name, _origin_root(name, modules)):
        if candidate in COMPRESSED_OPENERS:
            return candidate
    return None


def _open_alias(name, modules):
    """What a bare callable resolves to: "builtin", a COMPRESSED_OPENERS key, or None."""
    origin = modules.get(name)
    if origin is None:
        return "builtin" if name == "open" else None
    parts = origin.split(".")
    if parts[-1] != "open":
        return None
    if parts[0] in BUILTIN_OPEN_MODULES or origin == "open":
        return "builtin"
    return parts[0] if parts[0] in COMPRESSED_OPENERS else None


def _foreign_receiver(node, modules) -> bool:
    """True when the thing before `.open` is an object another library built.

    `zipfile.ZipFile(p).open(name)` returns a binary member stream and takes no
    encoding, so it needs the same exemption as the bare `zipfile.open` spelling.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        root, name = func.value.id, func.attr
    elif isinstance(func, ast.Name):
        root = name = func.id
    else:
        return False
    return root in modules and name not in PATH_CLASSES


def _offender(call: ast.Call, modules = None) -> str | None:
    """The call's name if it does text I/O without pinning an encoding."""
    modules = {} if modules is None else modules
    func = call.func
    if isinstance(func, ast.Attribute):
        receiver = func.value.id if isinstance(func.value, ast.Name) else None
        if func.attr in GUARDED_METHODS:
            # importlib.metadata's Distribution.read_text takes a positional
            # filename and has no encoding parameter at all, so a positional
            # argument means the receiver is not a Path.
            if func.attr == "read_text" and call.args:
                return None
            return None if _names_encoding(call) else f"{func.attr}()"
        if func.attr == "open":
            if receiver is not None and _origin_root(receiver, modules) in BUILTIN_OPEN_MODULES:
                return (
                    None if not _is_text(call, 1) or _names_encoding(call)
                    else f"{receiver}.open()"
                )
            compressed = _compressed_key(receiver, modules) if receiver else None
            if compressed is not None:
                # "rb" by default, so only an explicit text mode is in scope.
                mode = _mode(call, 1)
                if mode is UNKNOWN_MODE or "t" not in str(mode):
                    return None
                return None if _names_encoding(call) else f"{compressed}.open()"
            # Any other imported receiver is somebody else's opener: tarfile
            # takes a compression mode, Image takes a binary file. Neither has
            # an encoding to name, so demanding one leaves no correct edit.
            if receiver is not None and receiver in modules and receiver not in PATH_CLASSES:
                return None
            if _foreign_receiver(func.value, modules):
                return None
            if not _is_text(call, 0):
                return None
            return None if _names_encoding(call) else "Path.open()"
        return None
    if isinstance(func, ast.Name):
        alias = _open_alias(func.id, modules)
        if alias == "builtin":
            if not _is_text(call, 1):
                return None
            return None if _names_encoding(call) else "open()"
        if alias is not None:
            mode = _mode(call, 1)
            if mode is UNKNOWN_MODE or "t" not in str(mode):
                return None
            return None if _names_encoding(call) else f"{alias}.open()"
    return None


def _is_test_path(path: Path) -> bool:
    parts = path.relative_to(REPO).parts
    if SKIP_DIRS.intersection(parts):
        return True
    if "tests" in parts or "test" in parts:
        return True
    return path.name.startswith("test_") or path.name.endswith("_test.py")


def _offenders_in(src: str, label: str = "<snippet>"):
    tree = ast.parse(src, filename = label)
    modules = _imported_names(tree)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _offender(node, modules)
            if name is not None:
                found.append((node.lineno, name))
    return found


def test_shipping_code_names_an_encoding():
    offenders = []
    for root in ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if _is_test_path(path):
                continue
            src = path.read_text(encoding = "utf-8")
            try:
                tree = ast.parse(src, filename = str(path))
            except SyntaxError:
                continue
            rel = path.relative_to(REPO).as_posix()
            modules = _imported_names(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = _offender(node, modules)
                if name is not None:
                    offenders.append(f"{rel}:{node.lineno}: {name}")
    assert offenders == [], (
        f"{len(offenders)} text read/write call sites in shipping code let the "
        "operator's locale decide the encoding, so they crash or silently "
        'produce mojibake on Windows. Pass encoding = "utf-8": ' + repr(offenders)
    )


# The repo-wide assertion above passes vacuously once the trees are clean, so it
# cannot tell a working detector from one that always returns None. These pin the
# detector itself.


def test_detects_the_plain_cases():
    assert _offenders_in("from pathlib import Path\np = Path('x')\ns = p.read_text()\n")
    assert _offenders_in("p.write_text('hi')\n")
    assert _offenders_in("f = open('x')\n")
    assert _offenders_in("f = open('x', 'w')\n")
    assert _offenders_in("f = p.open()\n")
    # Inside a function body too: shipping reads are not import-time.
    assert _offenders_in("def load(p):\n    return p.read_text()\n")


def test_rejects_encoding_that_reselects_the_platform_default():
    assert _offenders_in("s = p.read_text(encoding = None)\n")
    assert _offenders_in("s = p.read_text(encoding = 'locale')\n")


def test_accepts_a_pinned_encoding():
    assert not _offenders_in("s = p.read_text(encoding = 'utf-8')\n")
    assert not _offenders_in("f = open('x', 'w', encoding = 'utf-8')\n")
    assert not _offenders_in("f = p.open(encoding = 'utf-8')\n")
    assert not _offenders_in("s = p.read_text(encoding = 'utf-8', errors = 'replace')\n")


def test_skips_binary_handles():
    # Binary has no encoding to name; passing one is a ValueError.
    assert not _offenders_in("f = open('x', 'rb')\n")
    assert not _offenders_in("f = open('x', mode = 'wb')\n")
    assert not _offenders_in("f = p.open('rb')\n")


def test_skips_unknown_modes():
    # Demanding encoding = on a call that may resolve to "rb" would leave no
    # compliant way to write it, so an unresolvable mode is not an offence.
    assert not _offenders_in("mode = 'rb' if binary else 'r'\nf = open(path, mode)\n")
    assert not _offenders_in("f = open(path, mode = chosen)\n")


def test_skips_foreign_openers_and_readers():
    assert not _offenders_in("import fitz\nd = fitz.open(stream = b, filetype = 'pdf')\n")
    assert not _offenders_in("import tarfile\nt = tarfile.open(p, 'r:gz')\n")
    # importlib.metadata Distribution.read_text takes a positional filename.
    assert not _offenders_in("s = dist.read_text('direct_url.json')\n")


def test_test_trees_are_out_of_scope():
    assert _is_test_path(REPO / "tests" / "test_x.py")
    assert _is_test_path(REPO / "studio" / "backend" / "tests" / "helpers.py")
    assert not _is_test_path(REPO / "studio" / "backend" / "routes" / "inference.py")
