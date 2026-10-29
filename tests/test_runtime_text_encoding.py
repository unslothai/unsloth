# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runtime text reads and writes must name an encoding; the locale default is cp1252 on Windows."""

# `str | None` below is evaluated at import on Python 3.9.
from __future__ import annotations

import ast
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
# `studio/` covers the installers too; test trees use test_source_read_encoding.py.
ROOTS = (REPO / "unsloth", REPO / "studio", REPO / "unsloth_cli")
SKIP_DIRS = {"build", "dist", "frontend", "node_modules", "src-tauri", ".venv", "site-packages"}
# Vendored laya sites pinned by sha256; laya_runtime rebinds their `open` to UTF-8.
# Pinned by exact site so any vendor change reds this scan until re-reviewed.
VENDORED_LOADER = REPO / "studio/backend/core/systemone/laya_runtime.py"
VENDORED_LOADER_REBIND = ".open = _utf8_open"
REVIEWED_VENDORED_OFFENDERS = {
    "studio/backend/vendor/laya/agent.py:31: open()",
    "studio/backend/vendor/laya/agent.py:47: open()",
    "studio/backend/vendor/laya/agent.py:156: open()",
    # Cloudflare's Clef loader (sha256 in unsloth/_vendor/clef/clef_manifest.json) reading
    # joint_head_config.json, which json.dumps writes as ASCII, so every locale decodes it.
    "unsloth/_vendor/clef/joint_schema_model.py:515: read_text()",
}
# keyed on path + expression, not line, so unrelated edits above it do not break the scan.
REVIEWED_NON_FILE_OPEN = (
    "studio/backend/core/inference/audio_inputs.py",
    "stream.codec_context.open(strict=True)",
)
GUARDED_METHODS = {"read_text", "write_text"}
# path classes, so an unbound `Path.open(p)` shifts every argument one right.
PATH_CLASSES = {"Path", "PosixPath", "PurePath", "WindowsPath"}
# values that re-select the platform default when passed as the encoding.
PLATFORM_DEFAULT_ENCODINGS = (None, "locale")
# calls that return the platform default, so naming one pins nothing.
PLATFORM_DEFAULT_CALLS = {"getdefaultencoding", "getencoding", "getpreferredencoding"}
# modules whose `open` is the builtin: same signature, same platform default.
BUILTIN_OPEN_MODULES = {"builtins", "io"}
# take an encoding in "t" mode but default to "rb".
COMPRESSED_OPENERS = {"bz2": 3, "gzip": 3, "lzma": None}
# distinct from None so that "no mode argument at all" still means text.
UNKNOWN_MODE = object()


def _mode(call: ast.Call, positional_index: int):
    """The call's mode, or UNKNOWN_MODE when it is not a literal."""
    # A splat hides the mode: treat as unknown, since it may resolve to binary.
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
    """Only a real encoding pins one: encoding = None or "locale" re-selects the platform default."""
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
    """Imports stay with their function so a local alias cannot switch off the builtin-open check."""
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
    """Names bound to an object another library built, such as a ZipFile whose open() takes no encoding."""
    modules = _imported_names(tree)
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if _foreign_receiver(node.value, modules):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _imported_names(tree) -> dict:
    """Maps names bound by imports to their origin, since an alias can hide the real opener."""
    bound = {}
    stack = list(ast.iter_child_nodes(tree))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
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
    """True when .open is called on a foreign object like zipfile.ZipFile(p), which yields binary
    streams."""
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
        shift = 1 if _is_path_class(receiver, modules) or _is_path_attr(func.value) else 0
        if func.attr in GUARDED_METHODS:
            if func.attr == "read_text" and not shift and call.args:
                first = call.args[0]
                # Bound read_text takes encoding first, so None or "locale" there is a platform default.
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
                mode = _mode(call, 1)
                if mode is UNKNOWN_MODE or "t" not in str(mode):
                    return None
                return None if _names_encoding(call) else f"{compressed}.open()"
            # Other imported receivers (tarfile, Image) are foreign openers with no encoding to name.
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


def _is_reviewed_non_file_open(label: str, call: ast.Call) -> bool:
    return (label, ast.unparse(call)) == REVIEWED_NON_FILE_OPEN


def _offenders(tree: ast.Module, label: str):
    visible_at = _imports_at_each_call(tree)
    foreign = _foreign_names(tree)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _offender(node, visible_at.get(id(node), {}), foreign)
            if name is not None and not _is_reviewed_non_file_open(label, node):
                found.append((node.lineno, name))
    return found


def _offenders_in(src: str, label: str = "<snippet>"):
    return _offenders(ast.parse(src, filename = label), label)


def _tracked_sources():
    """scan tracked *.py files so untracked builds, worktrees, and dependencies cannot fail it."""
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


def _loader_rebinds_open() -> bool:
    return VENDORED_LOADER.is_file() and VENDORED_LOADER_REBIND in VENDORED_LOADER.read_text(
        encoding = "utf-8"
    )


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
        offenders.extend(f"{rel}:{line}: {name}" for line, name in _offenders(tree, rel))
    if _loader_rebinds_open():
        stale = sorted(REVIEWED_VENDORED_OFFENDERS.difference(offenders))
        assert stale == [], (
            f"Reviewed vendored call sites no longer found, so the vendored code changed: {stale}. "
            "Re-review the new sites and update REVIEWED_VENDORED_OFFENDERS."
        )
        offenders = [o for o in offenders if o not in REVIEWED_VENDORED_OFFENDERS]
    assert offenders == [], (
        f"{len(offenders)} text read/write call sites in shipping code let the "
        "operator's locale decide the encoding, so they crash or silently "
        'produce mojibake on Windows. Pass encoding = "utf-8": ' + repr(offenders)
    )


# The assertion above passes vacuously on clean trees; these pin the detector itself.
def test_detects_the_plain_cases():
    assert _offenders_in("from pathlib import Path\np = Path('x')\ns = p.read_text()\n")
    assert _offenders_in("p.write_text('hi')\n")
    assert _offenders_in("f = open('x')\n")
    assert _offenders_in("f = open('x', 'w')\n")
    assert _offenders_in("f = p.open()\n")
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
    assert not _offenders_in("f = open('x', 'rb')\n")
    assert not _offenders_in("f = open('x', mode = 'wb')\n")
    assert not _offenders_in("f = p.open('rb')\n")


def test_skips_unknown_modes():
    assert not _offenders_in("mode = 'rb' if binary else 'r'\nf = open(path, mode)\n")
    assert not _offenders_in("f = open(path, mode = chosen)\n")


def test_skips_foreign_openers_and_readers():
    assert not _offenders_in("import fitz\nd = fitz.open(stream = b, filetype = 'pdf')\n")
    assert not _offenders_in("import tarfile\nt = tarfile.open(p, 'r:gz')\n")
    # importlib.metadata Distribution.read_text takes a positional filename.
    assert not _offenders_in("s = dist.read_text('direct_url.json')\n")


def test_skips_only_the_reviewed_pyav_codec_open():
    pyav = "\n" * 263 + "stream.codec_context.open(strict = True)\n"
    path = "studio/backend/core/inference/audio_inputs.py"
    assert not _offenders_in(pyav, path)
    assert not _offenders_in("\n" + pyav, path)
    assert _offenders_in(pyav, "studio/backend/core/inference/other.py")
    assert _offenders_in("\n" * 263 + "config.codec_context.open(strict = True)\n", path)


def test_test_trees_are_out_of_scope():
    assert _is_test_path(REPO / "tests" / "test_x.py")
    assert _is_test_path(REPO / "studio" / "backend" / "tests" / "helpers.py")
    assert not _is_test_path(REPO / "studio" / "backend" / "routes" / "inference.py")
