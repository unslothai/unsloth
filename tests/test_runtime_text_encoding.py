# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Guard: shipping code must name an encoding on every text read and write.

`Path.read_text()`, `Path.write_text()`, `Path.open()` and builtin `open()` fall back
to `locale.getencoding()`: UTF-8 on the Linux and macOS runners, cp1252 on a stock
Windows install. Every file this repo reads at runtime is UTF-8 (HF `config.json` /
`tokenizer_config.json` / `adapter_config.json`, Ollama manifests, GGUF export
metadata), so on Windows those reads crash or, worse, succeed with mojibake: a
DeepSeek or Qwen tokenizer_config.json carries U+FF5C and U+2581 in its chat
template, and at utils/models/model_config.py that read sits inside a broad
`except Exception: logger.debug(...)`, so the token-pattern check silently
returned the wrong answer.

Unlike the import-time rule in test_source_read_encoding.py this is scope agnostic:
runtime reads live inside functions, and shipping code has no legitimate reason to
let the operator's locale decide. No reachability analysis to get wrong, so no
allowlist and no false positives.

Binary handles are skipped (no encoding to name, and passing one is a ValueError),
and a non-constant mode counts as unknown rather than text: demanding `encoding =`
on a call that may resolve to "rb" would leave no compliant way to write it.

Known limitation, deliberately not closed: `configparser.ConfigParser.read()` also
defaults to the locale encoding, but cannot be matched by name without resolving the
receiver, since `f.read(n)`, `resp.read(limit)` and `handle.read(chunk)` are spelled
identically. Flagging it would be a false positive with no compliant fix, the exact
failure mode this guard avoids. The one live `ConfigParser.read` (/etc/wsl.conf,
hub/utils/paths.py) is pinned by hand; a future one has to be caught in review.
"""

# `str | None` below is evaluated at import on Python 3.9 (requires-python >= 3.9).
from __future__ import annotations

import ast
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
# Everything that ships.
# `studio/` covers the installers too: install_python_stack.py reads /sys/class/kfd, the same detection path as
# utils/hardware/hardware.py.
# Test trees fall under the narrower import-time rule in test_source_read_encoding.py.
ROOTS = (REPO / "unsloth", REPO / "studio", REPO / "unsloth_cli")
# The frontend tree is TypeScript;
# node_modules is vendored third-party code.
SKIP_DIRS = {"build", "dist", "frontend", "node_modules", "src-tauri", ".venv", "site-packages"}
GUARDED_METHODS = {"read_text", "write_text"}
# Path classes, so an unbound `Path.open(p)` shifts every argument one right.
PATH_CLASSES = {"Path", "PosixPath", "PurePath", "WindowsPath"}
# Values that re-select the platform default when passed as the encoding.
PLATFORM_DEFAULT_ENCODINGS = (None, "locale")
# Calls that return the platform default, so naming one pins nothing.
PLATFORM_DEFAULT_CALLS = {"getdefaultencoding", "getencoding", "getpreferredencoding"}
# Modules whose `open` IS the builtin: same signature, same platform default.
BUILTIN_OPEN_MODULES = {"builtins", "io"}
# Take an encoding in "t" mode but default to "rb".
COMPRESSED_OPENERS = {"bz2": 3, "gzip": 3, "lzma": None}
# Distinct from None so that "no mode argument at all" still means text.
UNKNOWN_MODE = object()


def _mode(call: ast.Call, positional_index: int):
    """The call's mode, or UNKNOWN_MODE when it is not a literal."""
    # A splat hides the mode, so it is unknown rather than absent: falling through to "r" would flag a call that
    # may resolve to binary, with no compliant way to fix it.
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

    `encoding = None` and `encoding = "locale"` re-select the platform default, so the
    keyword being present is not enough. A `**kwargs` splat may carry an encoding we
    cannot see, so it counts as named rather than as an unsatisfiable demand.
    """
    for kw in call.keywords:
        if kw.arg is None:
            return True
        if kw.arg != "encoding":
            continue
        if isinstance(kw.value, ast.Constant) and kw.value.value in PLATFORM_DEFAULT_ENCODINGS:
            return False
        if isinstance(kw.value, ast.Call) and _callee_name(kw.value.func) in PLATFORM_DEFAULT_CALLS:
            return False  # locale.getencoding() is the default, spelled out
        return True
    return False


def _is_text(call: ast.Call, positional_index: int) -> bool:
    mode = _mode(call, positional_index)
    return mode is not UNKNOWN_MODE and "b" not in str(mode)


def _imports_at_each_call(tree: ast.Module) -> dict:
    """The imports visible at every call, keyed by node id.

    A function's own imports stay in that function: hoisting them would let one local
    `from PIL.Image import open` turn off the builtin check for the whole file.
    """
    visible_at = {}

    def walk(node, visible):
        if isinstance(node, ast.Call):
            visible_at[id(node)] = visible
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                walk(child, {**visible, **_imported_names(child)})
            else:
                walk(child, visible)

    walk(tree, _imported_names(tree))
    return visible_at


def _foreign_names(tree: ast.Module) -> set:
    """Names bound to an object another library built.

    `z = zipfile.ZipFile(p)` then `z.open(name)` is a binary member stream taking no
    encoding, so demanding one leaves no correct edit.
    """
    modules = _imported_names(tree)
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if _foreign_receiver(node.value, modules):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _imported_names(tree) -> dict:
    """Names this module's imports bind, mapped to where they came from.

    The name alone settles nothing: `import tarfile as tf` hides an opener that takes
    no encoding, and `from PIL.Image import open` puts another behind the most familiar
    name there is. Resolving the origin covers both, with no module list to maintain.
    """
    bound = {}
    stack = list(ast.iter_child_nodes(tree))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue  # that function's business, not this scope's
        if isinstance(node, ast.Import):
            for a in node.names:
                bound[(a.asname or a.name).split(".")[0]] = a.name
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                bound[a.asname or a.name] = f"{node.module}.{a.name}" if node.module else a.name
        else:
            stack.extend(ast.iter_child_nodes(node))
    return bound


def _callee_name(func):
    """The bare name a callee ends in, whether or not it is qualified."""
    return func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)


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


def _is_path_class(name, modules) -> bool:
    """True for a pathlib class, including under an alias."""
    if name is None:
        return False
    return (modules.get(name) or name).split(".")[-1] in PATH_CLASSES


def _is_path_attr(node) -> bool:
    """True for a qualified path class, as in `pathlib.Path`."""
    return isinstance(node, ast.Attribute) and node.attr in PATH_CLASSES


def _foreign_receiver(node, modules) -> bool:
    """True when the thing before `.open` is an object another library built.

    `zipfile.ZipFile(p).open(name)` returns a binary member stream taking no encoding,
    so it needs the same exemption as the bare `zipfile.open` spelling.
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
    return root in modules and not _is_path_class(name, modules)


def _offender(
    call: ast.Call,
    modules = None,
    foreign = (),
) -> str | None:
    """The call's name if it does text I/O without pinning an encoding."""
    modules = {} if modules is None else modules
    func = call.func
    if isinstance(func, ast.Attribute):
        receiver = func.value.id if isinstance(func.value, ast.Name) else None
        # `Path.read_text(p)` is `p.read_text()` unbound: the instance takes slot 0, so every argument shifts
        # one place right.
        shift = 1 if _is_path_class(receiver, modules) or _is_path_attr(func.value) else 0
        if func.attr in GUARDED_METHODS:
            if func.attr == "read_text" and not shift and call.args:
                first = call.args[0]
                # Bound read_text takes encoding first, so None or "locale" there is a platform-default read.
                if isinstance(first, ast.Constant) and first.value in PLATFORM_DEFAULT_ENCODINGS:
                    return "read_text()"
                return None
            return None if _names_encoding(call) else f"{func.attr}()"
        if func.attr == "open":
            if receiver is not None and _origin_root(receiver, modules) in BUILTIN_OPEN_MODULES:
                return (
                    None if not _is_text(call, 1) or _names_encoding(call) else f"{receiver}.open()"
                )
            compressed = _compressed_key(receiver, modules) if receiver else None
            if compressed is not None:
                # "rb" by default, so only an explicit text mode is in scope.
                mode = _mode(call, 1)
                if mode is UNKNOWN_MODE or "t" not in str(mode):
                    return None
                return None if _names_encoding(call) else f"{compressed}.open()"
            # Any other imported receiver is somebody else's opener: tarfile takes a compression mode, Image a
            # binary file. Neither has an encoding to name.
            if receiver is not None and receiver in modules and receiver not in PATH_CLASSES:
                return None
            if _foreign_receiver(func.value, modules) or receiver in foreign:
                return None
            if not _is_text(call, shift):
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
    visible_at = _imports_at_each_call(tree)
    foreign = _foreign_names(tree)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _offender(node, visible_at.get(id(node), {}), foreign)
            if name is not None:
                found.append((node.lineno, name))
    return found


def _tracked_sources():
    """Shipping *.py that git is actually tracking.

    A walk also picks up whatever is lying in the checkout (a built `build/lib` copy,
    a nested worktree, a vendored dep). None of those are ours to police, and a stale
    artifact would fail this for everybody who has one.
    """
    listed = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-z", "--", "*.py"],
        capture_output = True,
        timeout = 60,
    )
    if listed.returncode != 0:
        return None
    names = listed.stdout.decode("utf-8", errors = "replace").split("\0")
    return [REPO / n for n in names if n]


def _walked_sources():
    return [p for root in ROOTS if root.is_dir() for p in sorted(root.rglob("*.py"))]


def test_shipping_code_names_an_encoding():
    offenders = []
    sources = _tracked_sources()
    if sources is None:
        sources = _walked_sources()
    roots = {r.resolve() for r in ROOTS}
    for path in sorted(sources):
        if not roots.intersection(path.resolve().parents) or _is_test_path(path):
            continue
        try:
            tree = ast.parse(path.read_text(encoding = "utf-8"), filename = str(path))
        except SyntaxError:
            continue
        rel = path.relative_to(REPO).as_posix()
        visible_at = _imports_at_each_call(tree)
        foreign = _foreign_names(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _offender(node, visible_at.get(id(node), {}), foreign)
                if name is not None:
                    offenders.append(f"{rel}:{node.lineno}: {name}")
    assert offenders == [], (
        f"{len(offenders)} text read/write call sites in shipping code let the "
        "operator's locale decide the encoding, so they crash or silently "
        'produce mojibake on Windows. Pass encoding = "utf-8": ' + repr(offenders)
    )


# The assertion above passes vacuously once the trees are clean, so it cannot tell a working detector from one that
# always returns None. These pin the detector itself.
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
    # A call that may resolve to "rb" has no compliant way to name an encoding.
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
