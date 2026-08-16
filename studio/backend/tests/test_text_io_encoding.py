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
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


BACKEND_ROOT = Path(__file__).resolve().parent.parent

# Not runtime source. Shipped plugins under plugins/*/src are, so only builds are skipped.
_SKIPPED_DIRS = ("node_modules", "build", "tests", "__pycache__")

# Path.open()'s signature is what tells it apart from other libraries' open(),
# e.g. fitz.open(stream=...) and av.open(..., metadata_errors=...).
_FILE_MODE_CHARS = set("rwxabt+")
_PATH_OPEN_ARGS = ("mode", "buffering", "encoding", "errors", "newline")
_PATH_OPEN_KWARGS = set(_PATH_OPEN_ARGS)
_PATH_OPEN_ENCODING_ARG = _PATH_OPEN_ARGS.index("encoding")

_SUBPROCESS_CALLS = {"run", "Popen", "check_output", "check_call", "call"}

# open(file, mode, buffering, encoding, ...), and os.fdopen forwards the same
# signature with a descriptor in place of the path.
_OPEN_ENCODING_ARG = 3


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


def _open_has_encoding(node: ast.Call) -> bool:
    """open()/os.fdopen() also take encoding positionally: open(p, "w", 1, "utf-8")."""
    return _has_keyword(node, "encoding") or len(node.args) > _OPEN_ENCODING_ARG


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


def _subprocess_aliases(tree: ast.AST, names: set[str]) -> set[str]:
    """Plain names bound to a subprocess callable, called without the module.

    ``install_wheel(run = subprocess.run)`` calls its injected ``run`` as a bare
    name, so matching only the attribute form leaves those installer calls
    unguarded. Imports, assignments and parameter defaults all bind one.
    """

    def _is_bound(value: ast.expr | None) -> bool:
        return (
            isinstance(value, ast.Attribute)
            and value.attr in _SUBPROCESS_CALLS
            and isinstance(value.value, ast.Name)
            and value.value.id in names
        )

    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "subprocess":
            aliases.update(a.asname or a.name for a in node.names if a.name in _SUBPROCESS_CALLS)
        elif isinstance(node, ast.Assign) and _is_bound(node.value):
            aliases.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and _is_bound(node.value):
            if isinstance(node.target, ast.Name):
                aliases.add(node.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            positional = args.posonlyargs + args.args
            # Defaults cover the tail of the positional parameters; kw_defaults
            # is aligned with kwonlyargs already, holding None where absent.
            padded = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
            pairs = list(zip(positional, padded)) + list(zip(args.kwonlyargs, args.kw_defaults))
            aliases.update(arg.arg for arg, default in pairs if _is_bound(default))
    return aliases


def _is_subprocess_call(node: ast.Call, names: set[str], aliases: set[str]) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in aliases
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


def _text_mode_dict(node: ast.Dict) -> bool:
    """A ``{"text": True, ...}`` literal with no "encoding" key."""
    keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
    if "encoding" in keys:
        return False
    for key, value in zip(node.keys, node.values):
        if not isinstance(key, ast.Constant) or key.value not in (
            "text",
            "universal_newlines",
        ):
            continue
        if isinstance(value, ast.Constant) and value.value is True:
            return True
    return False


def _splatted_names(tree: ast.AST) -> set[str]:
    """Names handed to a call as ``**name``."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg is None and isinstance(keyword.value, ast.Name):
                    names.add(keyword.value.id)
    return names


def _encoding_assigned_later(tree: ast.AST, name: str) -> bool:
    """``name["encoding"] = ...`` somewhere, so the literal need not carry it."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript) or not isinstance(node.ctx, ast.Store):
            continue
        target, key = node.value, node.slice
        if isinstance(target, ast.Name) and target.id == name:
            if isinstance(key, ast.Constant) and key.value == "encoding":
                return True
    return False


def _splatted_kwargs_offenders(tree: ast.AST) -> list[ast.Dict]:
    """Text-mode kwargs built in a dict and splatted into a call.

    Kwargs are collected in a dict and splatted (``run(cmd, **run_kwargs)``)
    where a branch has to add a timeout or an env, and the call is often through
    a helper, so neither the callee nor the keywords are visible at the call
    site. Only dicts that reach a call this way are judged: an unrelated payload
    that happens to carry ``"text": True`` is not subprocess configuration.
    """
    found = []
    # ``run(cmd, **{...})``: the literal is at the call already.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg is None and isinstance(keyword.value, ast.Dict):
                if _text_mode_dict(keyword.value):
                    found.append(keyword.value)
    splatted = _splatted_names(tree)
    if not splatted:
        return found
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target]
        if not targets or not isinstance(node.value, ast.Dict):
            continue
        if not _text_mode_dict(node.value):
            continue
        for target in targets:
            if target.id in splatted and not _encoding_assigned_later(tree, target.id):
                found.append(node.value)
                break
    return found


def _offenders(path: Path) -> list[str]:
    source = path.read_text(encoding = "utf-8")
    tree = ast.parse(source, filename = str(path))
    subprocess_names = _subprocess_names(tree)
    subprocess_aliases = _subprocess_aliases(tree, subprocess_names)
    found: list[str] = []
    for node in _splatted_kwargs_offenders(tree):
        found.append(
            f"{path.name}:{node.lineno}: subprocess kwargs with text = True and no encoding"
        )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)

        if _is_subprocess_call(node, subprocess_names, subprocess_aliases):
            if _text_mode_subprocess(node) and not _has_keyword(node, "encoding"):
                found.append(f"{path.name}:{node.lineno}: subprocess(text = True) without encoding")
            continue

        if name == "open" and isinstance(node.func, ast.Name):
            if _mode_is_binary(node) or _open_has_encoding(node):
                continue
            found.append(f"{path.name}:{node.lineno}: open() without encoding")
            continue

        # os.fdopen(fd, "w") is open() on a descriptor, so text mode takes the
        # same locale default. Its mode defaults to "r", i.e. text, like open's.
        if name == "fdopen":
            if _mode_is_binary(node) or _open_has_encoding(node):
                continue
            found.append(f"{path.name}:{node.lineno}: os.fdopen() without encoding")
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
def test_resuming_a_legacy_jsonl_keeps_one_encoding(
    tmp_path: Path, codepage: str, name: str
) -> None:
    """A scrape written before UTF-8 was explicit must resume, not duplicate."""
    path = tmp_path / "out.jsonl"
    records = [{"id": 1, "author": name}, {"id": 2, "author": name}]
    body = "".join(json.dumps(r, ensure_ascii = False) + "\n" for r in records)
    path.write_bytes(body.encode(codepage))
    before = path.read_bytes()

    writer = _load_state_store(codepage).JsonlWriter(path)
    try:
        # Seen keys survive the resume, so a repeat is refused, not appended.
        assert writer.has("id:1") and writer.has("id:2")
        assert writer.write(records[0]) is False
        assert writer.write({"id": 3, "author": name}) is True
    finally:
        writer.close()

    # Never converted, so it still reads in its own codepage; the append is ASCII.
    blob = path.read_bytes()
    assert blob.startswith(before)
    assert blob[len(before) :].isascii()
    lines = [json.loads(x) for x in blob.decode(codepage).splitlines() if x.strip()]
    assert len(lines) == 3
    assert [line["author"] for line in lines] == [name] * 3


def test_a_coincidentally_utf8_legacy_line_is_left_alone(tmp_path: Path) -> None:
    """cp1251 `Р°` is D0 B0, which is also UTF-8 `а`, and nothing can tell them apart."""
    path = tmp_path / "out.jsonl"
    ambiguous = "Р°"
    assert ambiguous.encode("cp1251").decode("utf-8") == "а"  # the trap
    authors = ["Привет", "Здравствуйте", "Москва", ambiguous]
    path.write_bytes(
        b"".join(
            json.dumps({"id": i, "author": a}, ensure_ascii = False).encode("cp1251") + b"\n"
            for i, a in enumerate(authors)
        )
    )
    before = path.read_bytes()

    _load_state_store("cp1251").JsonlWriter(path).close()

    # Untouched, so the ambiguity never had to be resolved.
    assert path.read_bytes() == before
    rows = [json.loads(x) for x in path.read_text(encoding = "cp1251").splitlines() if x.strip()]
    assert [row["author"] for row in rows] == authors


@pytest.mark.parametrize(
    ("codepage", "word"), [("cp1251", "Привет"), ("cp932", "こんにちは"), ("cp1252", "Jürgen")]
)
def test_a_moved_shard_is_not_rewritten_by_guesswork(
    tmp_path: Path, codepage: str, word: str
) -> None:
    """Off the writing machine there is no codepage to attribute the file to."""
    path = tmp_path / "out.jsonl"
    # Two records: a lone non-UTF-8 line would count as damage, not legacy.
    path.write_bytes(
        b"".join(
            json.dumps({"id": i, "author": word}, ensure_ascii = False).encode(codepage) + b"\n"
            for i in (1, 4)
        )
    )
    before = path.read_bytes()

    # A UTF-8 host: latin-1 would read cp1251 `Привет` back as `Ïðèâåò`.
    writer = _load_state_store("utf-8").JsonlWriter(path)
    try:
        assert writer.has("id:1")  # ASCII keys still recover
        assert writer.write({"id": 2, "author": "Grüße"}) is True
    finally:
        writer.close()

    blob = path.read_bytes()
    assert blob.startswith(before)  # never rewritten
    assert blob[len(before) :].isascii()  # appended as \uXXXX, so no second encoding
    rows = [json.loads(x) for x in blob.decode(codepage).splitlines() if x.strip()]
    assert [row["author"] for row in rows] == [word, word, "Grüße"]


def test_an_all_ambiguous_shard_still_gets_ascii_appends(tmp_path: Path) -> None:
    """Every line valid under both readings still means the append must not pick one."""
    path = tmp_path / "out.jsonl"
    ambiguous = "Р°"  # cp1251 D0 B0, also valid UTF-8 for "а"
    path.write_bytes(
        b"".join(
            json.dumps({"id": i, "a": ambiguous}, ensure_ascii = False).encode("cp1251") + b"\n"
            for i in range(3)
        )
    )
    before = path.read_bytes()

    writer = _load_state_store("cp1251").JsonlWriter(path)
    try:
        assert writer.write({"id": 9, "a": "世界"}) is True
    finally:
        writer.close()

    blob = path.read_bytes()
    assert blob.startswith(before)
    # ASCII, so the appended record survives whichever reading is chosen.
    assert blob[len(before) :].isascii()
    for codec in ("cp1251", "utf-8"):
        rows = [json.loads(x) for x in blob.decode(codec).splitlines() if x.strip()]
        assert rows[-1]["a"] == "世界"


def test_a_damaged_line_in_an_ascii_shard_does_not_block_its_retry(tmp_path: Path) -> None:
    """With no non-ASCII records to outvote it, one damaged line is still damage."""
    path = tmp_path / "out.jsonl"
    path.write_bytes(
        b'{"id": 1, "author": "alice"}\n'
        + b'{"id": 99, "author": "bad \x96 byte"}\n'
        + b'{"id": 2, "author": "bob"}\n'
    )

    writer = _load_state_store("cp1252").JsonlWriter(path)
    try:
        assert writer.has("id:1") and writer.has("id:2")
        assert not writer.has("id:99")
        assert writer.write({"id": 99, "author": "good byte"}) is True
    finally:
        writer.close()


def test_a_damaged_line_does_not_block_its_own_retry(tmp_path: Path) -> None:
    """Its key comes from the codepage reading, which a UTF-8 shard did not pick."""
    path = tmp_path / "out.jsonl"
    path.write_bytes(
        json.dumps({"id": 1, "author": "Jürgen"}, ensure_ascii = False).encode()
        + b"\n"
        + b'{"id": 99, "author": "bad \x96 byte"}\n'
    )

    writer = _load_state_store("cp1252").JsonlWriter(path)
    try:
        assert writer.has("id:1")
        assert not writer.has("id:99")
        assert writer.write({"id": 99, "author": "good byte"}) is True
    finally:
        writer.close()


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


def test_an_undecodable_transport_marker_reads_as_unknown(tmp_path: Path) -> None:
    """Pinning the decode turns an undecodable marker into UnicodeDecodeError,
    which is a ValueError and so is not an OSError. Before the pin those bytes
    simply read as an unknown value and the caller safely purged and restarted
    the partial download; letting the error escape aborts the transfer instead.
    """
    import sys

    backend = str(Path(__file__).resolve().parent.parent)
    if backend not in sys.path:
        sys.path.insert(0, backend)
    from hub.utils import download_registry as registry

    marker = tmp_path / ".transport"
    marker.write_bytes(b"\x80\xffnative\n")
    assert registry._read_marker_value(marker) is None
    # A readable but unknown value takes the same path (the behaviour restored).
    marker.write_text("something-else\n", encoding = "utf-8")
    assert registry._read_marker_value(marker) is None


def test_a_torn_cache_ref_reads_as_not_cached(tmp_path: Path, monkeypatch) -> None:
    """hf_cache_snapshot_dir answers "is this model already on disk", and the
    offline embedding checks turn a raise into a 500. A refs/main holding a byte
    the codepage used to decode into a nonsense commit simply missed the snapshot
    dir before the pin; it has to keep missing it."""
    import sys

    backend = str(Path(__file__).resolve().parent.parent)
    if backend not in sys.path:
        sys.path.insert(0, backend)
    from utils import utils as backend_utils

    good_root = tmp_path / "good"
    torn_root = tmp_path / "torn"
    for root, ref_bytes in ((torn_root, b"\x80\xff\n"), (good_root, b"abc123\n")):
        repo = root / "models--Org--Model"
        (repo / "refs").mkdir(parents = True)
        (repo / "refs" / "main").write_bytes(ref_bytes)
    (good_root / "models--Org--Model" / "snapshots" / "abc123").mkdir(parents = True)

    monkeypatch.setattr(backend_utils, "_hf_cache_roots", lambda: [torn_root])
    assert backend_utils.hf_cache_snapshot_dir("Org/Model") is None
    # The torn root is skipped, not fatal: a healthy second root still answers.
    monkeypatch.setattr(backend_utils, "_hf_cache_roots", lambda: [torn_root, good_root])
    found = backend_utils.hf_cache_snapshot_dir("Org/Model")
    assert found is not None and found.name == "abc123"


def test_a_corrupt_pid_file_does_not_abort_shutdown(tmp_path: Path, monkeypatch) -> None:
    """_remove_pid_file runs first in _graceful_shutdown, so a raise there leaves
    the inference, export, training and tunnel children alive."""
    import sys

    backend = str(Path(__file__).resolve().parent.parent)
    if backend not in sys.path:
        sys.path.insert(0, backend)
    import run as studio_run

    pid_file = tmp_path / "studio.pid"
    pid_file.write_bytes(b"\x80\xff")
    monkeypatch.setattr(studio_run, "_PID_FILE", pid_file)
    # _legacy_heir scans the real Studio root, so without this the last assertion asks whether
    # a server happens to be running on the machine: with one, the record is handed over rather
    # than unlinked. The handoff has its own test below.
    monkeypatch.setattr(studio_run, "_legacy_heir", lambda: None)
    studio_run._remove_pid_file()
    # Not this process's PID, so the file stays; the point is that it returned.
    assert pid_file.exists()

    pid_file.write_text(str(os.getpid()), encoding = "utf-8")
    studio_run._remove_pid_file()
    assert not pid_file.exists()


def test_the_legacy_pid_record_is_handed_to_a_live_sibling(tmp_path: Path, monkeypatch) -> None:
    """Only one server owns studio.pid, so deleting it while a sibling still serves would leave
    that sibling unstoppable from an older CLI that reads no other file."""
    import sys

    backend = str(Path(__file__).resolve().parent.parent)
    if backend not in sys.path:
        sys.path.insert(0, backend)
    import run as studio_run

    pid_file = tmp_path / "studio.pid"
    pid_file.write_text(str(os.getpid()), encoding = "utf-8")
    monkeypatch.setattr(studio_run, "_PID_FILE", pid_file)
    monkeypatch.setattr(studio_run, "_legacy_heir", lambda: 4242)

    studio_run._remove_pid_file()

    assert pid_file.read_text(encoding = "utf-8") == "4242"


def test_the_kwargs_guard_only_judges_dicts_that_reach_a_call(tmp_path: Path) -> None:
    """Only a dict splatted into a call is subprocess configuration. An unrelated
    payload that happens to carry "text": True is not, and neither is one whose
    encoding is filled in on a later line."""
    cases = {
        "offender.py": 'kw = {"text": True}\nrun(cmd, **kw)\n',
        "annotated.py": 'kw: dict = {"universal_newlines": True}\nrun(cmd, **kw)\n',
        "payload.py": 'payload = {"text": True}\nrequests.post(url, json = payload)\n',
        "inline.py": 'run(cmd, **{"text": True})\n',
        "later.py": 'kw = {"text": True}\nkw["encoding"] = "utf-8"\nrun(cmd, **kw)\n',
        "carried.py": 'kw = {"text": True, "encoding": "utf-8"}\nrun(cmd, **kw)\n',
    }
    flagged = set()
    for name, source in cases.items():
        path = tmp_path / name
        path.write_text(source, encoding = "utf-8")
        if any("subprocess kwargs" in line for line in _offenders(path)):
            flagged.add(name)
    assert flagged == {"offender.py", "annotated.py", "inline.py"}, flagged


def test_the_guard_follows_subprocess_through_an_alias(tmp_path: Path) -> None:
    """install_wheel() takes ``run = subprocess.run`` and calls it as a bare
    name, so an attribute-only match let both of its installer calls drop their
    encoding unnoticed. A name bound to something else is still not subprocess."""
    cases = {
        "param_default.py": (
            "import subprocess\n"
            "def install(*, run = subprocess.run):\n"
            "    run(cmd, text = True)\n"
        ),
        "assigned.py": "import subprocess\n_run = subprocess.run\n_run(cmd, text = True)\n",
        "imported.py": "from subprocess import check_output\ncheck_output(cmd, text = True)\n",
        "renamed.py": "from subprocess import run as _r\n_r(cmd, universal_newlines = True)\n",
        "encoded.py": (
            "import subprocess\n"
            "def install(*, run = subprocess.run):\n"
            '    run(cmd, text = True, encoding = "utf-8")\n'
        ),
        "unrelated.py": "def run(cmd, text = False):\n    pass\nrun(cmd, text = True)\n",
    }
    flagged = set()
    for name, source in cases.items():
        path = tmp_path / name
        path.write_text(source, encoding = "utf-8")
        if any("subprocess(text = True)" in line for line in _offenders(path)):
            flagged.add(name)
    assert flagged == {"param_default.py", "assigned.py", "imported.py", "renamed.py"}, flagged


def test_the_guard_sees_os_fdopen(tmp_path: Path) -> None:
    """os.fdopen(fd, mode) is open() on a descriptor and takes the same locale
    default in text mode, so leaving it out let the swap lock file keep the
    codepage on the write side while its reader was pinned to UTF-8."""
    cases = {
        "text.py": 'import os\nos.fdopen(fd, "w")\n',
        "default_mode.py": "import os\nos.fdopen(fd)\n",  # defaults to "r", still text
        "binary.py": 'import os\nos.fdopen(fd, "wb")\n',
        "keyword.py": 'import os\nos.fdopen(fd, "w", encoding = "utf-8")\n',
        "positional.py": 'import os\nos.fdopen(fd, "w", 1, "utf-8")\n',
    }
    flagged = set()
    for name, source in cases.items():
        path = tmp_path / name
        path.write_text(source, encoding = "utf-8")
        if any("fdopen" in line for line in _offenders(path)):
            flagged.add(name)
    assert flagged == {"text.py", "default_mode.py"}, flagged


def test_an_undecodable_bootstrap_password_does_not_stop_startup(
    tmp_path: Path, monkeypatch
) -> None:
    """ensure_default_admin calls _load_bootstrap_password for every existing
    admin and the lifespan calls that with no handler, so a raise here takes the
    whole backend down instead of ignoring an unusable file."""
    import sys

    backend = str(Path(__file__).resolve().parent.parent)
    if backend not in sys.path:
        sys.path.insert(0, backend)
    from auth import storage

    pw_file = tmp_path / ".bootstrap_password"
    pw_file.write_bytes(b"\x80\xffnot-utf8\n")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", pw_file)
    assert storage._load_bootstrap_password() is None

    # A readable one still loads, so this is a narrowing of failure, not of function.
    pw_file.write_text("correct horse battery staple\n", encoding = "utf-8")
    assert storage._load_bootstrap_password() == "correct horse battery staple"


def test_a_damaged_checkpoint_resets_instead_of_resuming_on_a_broken_cursor(tmp_path: Path) -> None:
    """A checkpoint holds only base64 cursors and booleans, so a codepage reading
    can only ever add non-ASCII, never recover any. Resuming on a mojibaked cursor
    sends GitHub one it answers with INVALID_CURSOR_ARGUMENTS, and the empty page
    that comes back marks the stream done and skips the rest of it for good.
    Dropping the checkpoint only replays pages the writers already dedup."""
    module = _load_state_store("cp1252")
    cursor = "Y3Vyc29yOnYyOpK0MjAxMi0wMi0xNlQwNjo1Mzo0MVrOADGL_A=="
    healthy = json.dumps({"issues_cursor": cursor, "issues_done": False}, indent = 2)
    path = tmp_path / "octocat__Hello-World.json"

    path.write_text(healthy, encoding = "utf-8")
    assert module.StateStore(path).get("issues_cursor") == cursor

    # Written by a pre-UTF-8 release in the operator's codepage. Nothing is lost
    # by reading UTF-8 only, because an all-ASCII document is the same bytes.
    path.write_bytes(healthy.encode("cp1252"))
    assert module.StateStore(path).get("issues_cursor") == cursor

    # One damaged byte inside the cursor: still a whole JSON document under a
    # single-byte codepage, so only refusing that reading resets the checkpoint.
    raw = healthy.encode()
    at = raw.index(b"MjAxMi0wMi0xNlQ") + 3
    path.write_bytes(raw[:at] + b"\x96" + raw[at + 1 :])
    assert json.loads(path.read_bytes().decode("latin-1"))["issues_cursor"] != cursor
    store = module.StateStore(path)
    assert store.all() == {}
    assert store.get("issues_cursor") is None


def test_a_utf8_record_is_not_parsed_a_second_time(tmp_path: Path) -> None:
    """These shards reach gigabytes and every resume reads all of one, so a
    record that already read as UTF-8 must not be decoded and parsed again under
    the codepage. The legacy reading exists only to recover keys UTF-8 could not."""
    module = _load_state_store("cp1252")
    calls: list[str] = []
    real_parse = module._parse

    def counting_parse(raw, encoding):
        calls.append(encoding)
        return real_parse(raw, encoding)

    module._parse = counting_parse
    try:
        healthy = json.dumps({"id": 1, "author": "Jürgen"}).encode("utf-8")
        reading = module._read_line(healthy, "cp1252")
        assert reading.as_utf8 == {"id": 1, "author": "Jürgen"}
        assert calls == ["utf-8"], calls

        # A line UTF-8 cannot read still falls through to the codepage, the whole point.
        calls.clear()
        legacy = json.dumps({"id": 2, "author": "Jürgen"}, ensure_ascii = False).encode("cp1252")
        reading = module._read_line(legacy, "cp1252")
        assert reading.as_utf8 is None
        assert reading.as_legacy == {"id": 2, "author": "Jürgen"}
        assert calls == ["utf-8", "cp1252"], calls
    finally:
        module._parse = real_parse


def _too_deeply_nested_json() -> str:
    """A JSON document nested past what this interpreter will descend into.

    Probed rather than hardcoded: the depth json.loads gives up at is bounded by
    sys.getrecursionlimit() up to 3.11 and by the C recursion limit from 3.12,
    which sys.setrecursionlimit no longer moves and which varies by micro
    version. That is ~995 on 3.9 and ~9999 on 3.13.
    """
    depth = 1
    while depth <= 1 << 17:
        document = "[" * depth + "]" * depth
        try:
            json.loads(document)
        except RecursionError:
            return document
        depth *= 2
    pytest.skip("this interpreter parses arbitrarily nested JSON")


def test_an_unparseably_nested_document_is_discarded_not_raised(tmp_path: Path) -> None:
    """json.loads answers nesting it cannot descend with RecursionError, which is
    a RuntimeError and so is neither a ValueError nor a UnicodeDecodeError.
    _parse is called outside any other handler in both StateStore.__init__ and
    JsonlWriter._scan_existing, so letting it escape aborts the scraper at
    startup on a file the catch-all it replaced simply discarded."""
    module = _load_state_store("cp1252")
    nested = _too_deeply_nested_json()

    checkpoint = tmp_path / "octocat__Hello-World.json"
    checkpoint.write_text(nested, encoding = "utf-8")
    assert module.StateStore(checkpoint).all() == {}  # reset, not raised

    shard = tmp_path / "out.jsonl"
    shard.write_text(
        nested + "\n" + json.dumps({"id": 1}) + "\n" + json.dumps({"id": 2}) + "\n",
        encoding = "utf-8",
    )
    writer = module.JsonlWriter(shard)
    try:
        # Skipped like any other unreadable line, so its neighbours still yield the dedup
        # keys that keep the resume from re-fetching them.
        assert writer.has("id:1") and writer.has("id:2")
    finally:
        writer.close()
