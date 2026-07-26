# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Text I/O must name its encoding, or Windows silently uses the ANSI codepage.

``open()``, ``Path.read_text()`` and ``subprocess(text = True)`` fall back to
``locale.getencoding()`` when no ``encoding`` is passed. On Windows that is
cp1252 (or cp932, cp1251, ... by system locale), not UTF-8, so a chat template,
model config or path containing ``ä ö ü → 世`` mojibakes or raises
``UnicodeDecodeError`` mid-load. Studio's files are UTF-8, so say so.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


BACKEND_ROOT = Path(__file__).resolve().parent.parent

# Not Studio's own runtime source. Shipped plugins under plugins/*/src are, so
# only their build artifacts are skipped.
_SKIPPED_DIRS = ("node_modules", "build", "tests", "__pycache__", "sandbox_site")

# Path.open() takes a mode first and only these keywords. fitz.open(stream=...)
# and av.open(..., metadata_errors=...) are other libraries' open(), so matching
# the signature is what separates them.
_FILE_MODE_CHARS = set("rwxabt+")
_PATH_OPEN_ARGS = ("mode", "buffering", "encoding", "errors", "newline")
_PATH_OPEN_KWARGS = set(_PATH_OPEN_ARGS)
_PATH_OPEN_ENCODING_ARG = _PATH_OPEN_ARGS.index("encoding")

_SUBPROCESS_CALLS = {"run", "Popen", "check_output", "check_call", "call"}


def _studio_sources() -> list[Path]:
    return [
        path
        for path in sorted(BACKEND_ROOT.rglob("*.py"))
        if not any(part in _SKIPPED_DIRS for part in path.relative_to(BACKEND_ROOT).parts)
    ]


def _has_keyword(node: ast.Call, name: str) -> bool:
    return any(keyword.arg == name for keyword in node.keywords)


def _mode_is_binary(node: ast.Call) -> bool:
    mode: str | None = None
    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
        value = node.args[1].value
        mode = value if isinstance(value, str) else None
    for keyword in node.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            value = keyword.value.value
            if isinstance(value, str):
                mode = value
    return bool(mode and "b" in mode)


def _path_open_mode(node: ast.Call) -> str | None:
    if node.args and isinstance(node.args[0], ast.Constant):
        value = node.args[0].value
        if isinstance(value, str):
            return value
    for keyword in node.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            value = keyword.value.value
            if isinstance(value, str):
                return value
    return None


def _is_path_open(node: ast.Call) -> bool:
    """True only for calls matching ``Path.open``'s signature."""
    if len(node.args) > len(_PATH_OPEN_ARGS):
        return False
    if any(k.arg not in _PATH_OPEN_KWARGS for k in node.keywords):
        return False
    mode = _path_open_mode(node)
    if mode is not None:
        return bool(mode) and set(mode) <= _FILE_MODE_CHARS
    return not node.args


def _path_open_has_encoding(node: ast.Call) -> bool:
    """Path.open() also takes encoding positionally: open("w", 1, "utf-8")."""
    return _has_keyword(node, "encoding") or len(node.args) > _PATH_OPEN_ENCODING_ARG


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _subprocess_names(tree: ast.AST) -> set[str]:
    """Names subprocess is reachable under here, e.g. `import subprocess as _sp`."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "subprocess":
                    names.add(alias.asname or alias.name)
    return names


def _is_subprocess_call(node: ast.Call, names: set[str]) -> bool:
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr not in _SUBPROCESS_CALLS:
        return False
    value = func.value
    return isinstance(value, ast.Name) and value.id in names


def _text_mode_subprocess(node: ast.Call) -> bool:
    for keyword in node.keywords:
        if keyword.arg not in ("text", "universal_newlines"):
            continue
        if isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
            return True
    return False


def _offenders(path: Path) -> list[str]:
    source = path.read_text(encoding = "utf-8")
    tree = ast.parse(source, filename = str(path))
    subprocess_names = _subprocess_names(tree)
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)

        if _is_subprocess_call(node, subprocess_names):
            if _text_mode_subprocess(node) and not _has_keyword(node, "encoding"):
                found.append(f"{path.name}:{node.lineno}: subprocess(text = True) without encoding")
            continue

        if name == "open" and isinstance(node.func, ast.Name):
            if _mode_is_binary(node) or _has_keyword(node, "encoding"):
                continue
            found.append(f"{path.name}:{node.lineno}: open() without encoding")
            continue

        if name == "open" and isinstance(node.func, ast.Attribute):
            if not _is_path_open(node) or _path_open_has_encoding(node):
                continue
            if _path_open_mode(node) and "b" in _path_open_mode(node):
                continue
            found.append(f"{path.name}:{node.lineno}: Path.open() without encoding")
            continue

        if name in ("read_text", "write_text") and isinstance(node.func, ast.Attribute):
            if _has_keyword(node, "encoding"):
                continue
            # importlib.metadata Distribution.read_text() takes no encoding kwarg.
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "dist":
                continue
            found.append(f"{path.name}:{node.lineno}: {name}() without encoding")
    return found


@pytest.mark.parametrize("path", _studio_sources(), ids = lambda p: str(p.name))
def test_text_io_names_its_encoding(path: Path) -> None:
    offenders = _offenders(path)
    assert not offenders, (
        "Text I/O without an explicit encoding falls back to the Windows ANSI "
        'codepage and corrupts non-ASCII (ä ö ü → 世). Pass encoding = "utf-8":\n  '
        + "\n  ".join(offenders)
    )


_STATE_STORE = (
    BACKEND_ROOT
    / "plugins/data-designer-github-repo-seed/src"
    / "data_designer_github_repo_seed/scraper_impl/state_store.py"
)


def _load_state_store(codepage: str):
    """Load state_store with the writing machine's codepage pinned."""
    spec = importlib.util.spec_from_file_location(f"state_store_{codepage}", _STATE_STORE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.locale = SimpleNamespace(
        getencoding = lambda: codepage,
        getpreferredencoding = lambda _ = True: codepage,
    )
    return module


@pytest.mark.parametrize(
    ("codepage", "name"), [("cp1252", "Jürgen"), ("cp1251", "Юрий"), ("cp932", "田中")]
)
def test_resuming_a_legacy_jsonl_migrates_it(tmp_path: Path, codepage: str, name: str) -> None:
    """A scrape written before UTF-8 was explicit must resume, not duplicate."""
    path = tmp_path / "out.jsonl"
    records = [{"id": 1, "author": name}, {"id": 2, "author": name}]
    body = "".join(json.dumps(r, ensure_ascii = False) + "\n" for r in records)
    path.write_bytes(body.encode(codepage))

    writer = _load_state_store(codepage).JsonlWriter(path)
    try:
        # Seen keys survive the resume, so a repeat is refused, not appended.
        assert writer.has("id:1") and writer.has("id:2")
        assert writer.write(records[0]) is False
        assert writer.write({"id": 3, "author": name}) is True
    finally:
        writer.close()

    # One consistent encoding, and the legacy text came through intact.
    lines = [json.loads(x) for x in path.read_text(encoding = "utf-8").splitlines() if x.strip()]
    assert len(lines) == 3
    assert [line["author"] for line in lines] == [name] * 3


def test_a_coincidentally_utf8_legacy_line_follows_the_file(tmp_path: Path) -> None:
    """cp1251 `Р°` is D0 B0, which is also UTF-8 `а`, so the shard has to decide."""
    path = tmp_path / "out.jsonl"
    ambiguous = "Р°"
    assert ambiguous.encode("cp1251").decode("utf-8") == "а"  # the trap
    # Ordinary Cyrillic on the other lines is what settles the verdict.
    authors = ["Привет", "Здравствуйте", "Москва", ambiguous]
    path.write_bytes(
        b"".join(
            json.dumps({"id": i, "author": a}, ensure_ascii = False).encode("cp1251") + b"\n"
            for i, a in enumerate(authors)
        )
    )

    _load_state_store("cp1251").JsonlWriter(path).close()

    rows = [json.loads(x) for x in path.read_text(encoding = "utf-8").splitlines() if x.strip()]
    assert [row["author"] for row in rows] == authors


def test_one_damaged_byte_does_not_relabel_a_utf8_shard(tmp_path: Path) -> None:
    """A complete JSON line with a stray 0x96 parses as cp1252, but is only one vote."""
    path = tmp_path / "out.jsonl"
    healthy = ["Jürgen", "Grüße", "Björn"]
    path.write_bytes(
        json.dumps({"id": 0, "author": healthy[0]}, ensure_ascii = False).encode()
        + b"\n"
        + b'{"id": 99, "author": "bad \x96 byte"}\n'
        + b"".join(
            json.dumps({"id": i, "author": a}, ensure_ascii = False).encode() + b"\n"
            for i, a in enumerate(healthy[1:], start = 1)
        )
    )
    before = path.read_bytes()

    _load_state_store("cp1252").JsonlWriter(path).close()

    # Untouched, so the healthy records were never re-read as cp1252.
    assert path.read_bytes() == before
    rows = []
    for line in path.read_bytes().splitlines():
        try:
            rows.append(json.loads(line.decode()))
        except (UnicodeDecodeError, ValueError):
            continue
    assert [row["author"] for row in rows] == healthy


def test_a_torn_line_does_not_relabel_a_utf8_shard(tmp_path: Path) -> None:
    """One interrupted append must not get the whole shard read as cp1252."""
    path = tmp_path / "out.jsonl"
    good = [{"id": 1, "author": "Jürgen"}, {"id": 3, "author": "Grüße"}]
    torn = '{"id": 2, "author": "Jürgen"}'.encode()[:-6]  # cut mid-character
    path.write_bytes(
        json.dumps(good[0], ensure_ascii = False).encode()
        + b"\n"
        + torn
        + b"\n"
        + json.dumps(good[1], ensure_ascii = False).encode()
        + b"\n"
    )
    before = path.read_bytes()

    writer = _load_state_store("cp1252").JsonlWriter(path)
    try:
        assert writer.has("id:1") and writer.has("id:3")
        assert not writer.has("id:2")  # torn line yields no key
    finally:
        writer.close()

    # Untouched: no rewrite, so no record was re-encoded into mojibake.
    after = path.read_bytes()
    assert after.startswith(before)
    assert "Jürgen".encode() in after
    assert "Jürgen".encode("utf-8").decode("cp1252").encode() not in after
