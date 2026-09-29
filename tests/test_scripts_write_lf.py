# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Scripts that rewrite tracked files in place must write LF, on every platform.

`.gitattributes` opens with `*.py text eol=lf`, so every tracked Python file in this repo is
LF on disk and is meant to stay that way. Three scripts rewrite tracked files through the same
temp-file-then-os.replace shape, and all three opened the temp file in text mode with the
DEFAULT newline, which translates each "\\n" to `os.linesep`:

    scripts/enforce_kwargs_spacing.py   every tracked .py the formatting hook touches
    scripts/stamp_studio_release.py     studio/backend/utils/_studio_release_build.py
    scripts/scan_packages.py            studio/backend/requirements/*.txt, under --fix

On Linux `os.linesep` is "\\n" and the round trip is a no-op, which is why this survived. On
Windows it rewrites the file to CRLF. For enforce_kwargs_spacing that is every file a
contributor edits, because the read side is `tokenize.open`, which normalises CRLF to LF in
memory, so the write is what decides the ending. The symptom is a whole-file diff on files the
contributor did not change, from running the project's own pre-commit hook.

Found by running tests/test_formatter_fixed_point.py on a real windows-latest runner (see the
staging evidence on the PR): the guard reported ~1300 of ~2650 tracked files as drifted, none
of which had drifted.

Two halves, deliberately:

  * the BEHAVIOURAL tests below prove the end-to-end intent -- a CRLF file in, LF bytes out --
    and on Windows they exercise the translation for real. On Linux they cannot fail if the fix
    is reverted, because `os.linesep` is already "\\n" there. They are the demonstration.
  * `test_every_in_place_rewriter_names_its_newline` is the DURABLE guard. It reads the call
    sites and fails on any platform the moment a `newline =` argument goes missing, which is
    the only way to catch a revert in the Linux job that actually runs.

Same split, and the same reason, as test_formatter_fixed_point.py's Windows command-line check:
assert the property where the platform cannot demonstrate it.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"

#: The in-place rewriters this file covers, as (path, the function holding the write).
#: Named on the real files so a rename fails here rather than silently dropping coverage. A pair,
#: not a {script: func} mapping, because one script can hold more than one rewriter.
#:
#: The list is what it is because the rule is "rewrites a TRACKED file in place", not "uses
#: os.fdopen": the last two write through plain open()/Path.write_text and have exactly the same
#: default-newline defect. sync_allow_scripts_pins is the one that matters most -- it runs as a
#: pre-commit hook with --fix over studio/frontend/package.json, which .gitattributes pins to
#: eol=lf.
_REWRITERS = (
    ("enforce_kwargs_spacing.py", "_atomic_write_text"),
    ("stamp_studio_release.py", "_atomic_write_text"),
    ("scan_packages.py", "update_req_file"),
    ("scan_packages.py", "_write_baseline"),
    ("sync_allow_scripts_pins.py", "main"),
)


def _load(name: str):
    """Import a scripts/ module by filename, without putting scripts/ on sys.path for good."""
    path = _SCRIPTS / name
    assert path.is_file(), f"{path} is gone; _REWRITERS is stale"
    spec = importlib.util.spec_from_file_location(f"_scripts_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ── the durable half: the call sites name a newline ───────────────────────────────────────────


def _text_write_calls(tree: ast.AST, func_name: str) -> list[ast.Call]:
    """Every `os.fdopen(...)`/`open(...)`/`Path.write_text(...)` text-mode WRITE in the function.

    Binary mode is excluded: `newline` is meaningless there and passing it raises. A call with
    no mode argument at all defaults to "r", so it is not a write and is skipped too.

    `write_text` is here because it carries the identical default: `Path.write_text(data,
    encoding = ...)` leaves `newline` at None and so translates to os.linesep exactly like
    `open()` does. Leaving it out would let a rewriter swap one for the other and drop out of
    this guard silently, which is how sync_allow_scripts_pins.py was writing CRLF in the first
    place.
    """
    target = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name
        ),
        None,
    )
    assert target is not None, f"{func_name} is gone; _REWRITERS is stale"

    calls = []
    for node in ast.walk(target):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        named = (
            func.attr
            if isinstance(func, ast.Attribute)
            else func.id
            if isinstance(func, ast.Name)
            else ""
        )
        if named == "write_text":
            # No mode argument to read: write_text is always a text-mode write.
            calls.append(node)
            continue
        if named not in ("fdopen", "open"):
            continue
        # Mode is the second positional for both open() and os.fdopen().
        mode = next(
            (
                arg.value
                for arg in node.args[1:2]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            ),
            None,
        ) or next(
            (
                kw.value.value
                for kw in node.keywords
                if kw.arg == "mode"
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
            ),
            None,
        )
        if mode is None or "b" in mode or not any(ch in mode for ch in "wax+"):
            continue
        calls.append(node)
    return calls


@pytest.mark.parametrize(("script", "func"), _REWRITERS)
def test_every_in_place_rewriter_names_its_newline(script, func):
    """A text-mode write in one of these must say what line ending it wants.

    This is the half that can fail on Linux. `os.linesep` is "\\n" here, so no behavioural test
    in this file can notice the default coming back -- only reading the call site can.
    """
    tree = ast.parse((_SCRIPTS / script).read_text(encoding = "utf-8"))
    calls = _text_write_calls(tree, func)
    assert calls, f"no text-mode write found in {script}:{func}; this guard has gone vacuous"
    for call in calls:
        kwargs = {kw.arg for kw in call.keywords}
        assert "newline" in kwargs, (
            f"{script}:{call.lineno} in {func}() opens a text file for writing without a "
            "`newline =` argument, so Python translates every '\\n' to os.linesep and this "
            "rewrites the whole tracked file to CRLF on Windows. .gitattributes pins these to "
            'LF (`*.py text eol=lf`). Pass newline = "\\n".'
        )
        assert "encoding" in kwargs, (
            f"{script}:{call.lineno} in {func}() opens a text file for writing without an "
            "`encoding =` argument, so it takes the locale codec and round-trips differently "
            'between runners. Pass encoding = "utf-8".'
        )


def test_the_repo_agrees_these_files_are_lf():
    """The premise. If .gitattributes stops pinning LF, everything above is arguing for nothing."""
    attributes = (_ROOT / ".gitattributes").read_text(encoding = "utf-8")
    assert "*.py text eol=lf" in attributes, (
        ".gitattributes no longer pins tracked Python files to LF, so the rewriters this file "
        "guards have no ending to preserve and this whole file needs re-reading"
    )


# ── the demonstration half: a CRLF file in, LF bytes out ──────────────────────────────────────


def test_the_spacing_pass_writes_lf_for_a_crlf_source(tmp_path):
    """The headline path: what the pre-commit hook does to a contributor's file."""
    module = _load("enforce_kwargs_spacing.py")
    target = tmp_path / "sample.py"
    # CRLF on disk, and something the pass will actually rewrite, so the write really happens.
    target.write_bytes(b"def f(a=1, b=2):\r\n    return a + b\r\n")

    module.process_file(target)

    written = target.read_bytes()
    assert (
        b"\r\n" not in written
    ), f"the spacing pass wrote CRLF into a file .gitattributes pins to LF: {written!r}"
    assert written.endswith(b"\n"), written
    # Non-vacuous: it must actually have done its job, or "no CRLF" is just "no write".
    assert b"a = 1" in written, written


def test_the_release_stamp_writes_lf(tmp_path):
    """A tracked .py generated from a Python string literal, so LF is the only correct ending."""
    module = _load("stamp_studio_release.py")
    target = tmp_path / "_studio_release_build.py"

    module._atomic_write_text(target, 'VERSION = "1.2.3"\nBUILD = 7\n', encoding = "utf-8")

    written = target.read_bytes()
    assert b"\r\n" not in written, f"the release stamp wrote CRLF: {written!r}"
    assert written == b'VERSION = "1.2.3"\nBUILD = 7\n', written


def test_the_requirements_fixer_writes_lf_and_utf8(tmp_path):
    """--fix rewrites tracked requirements files, and used to take the locale codec too."""
    module = _load("scan_packages.py")
    target = tmp_path / "reqs.txt"
    # CRLF, plus a non-ASCII comment: the encoding half of the same call.
    target.write_bytes("requests==2.0.0  # naïve pin\r\nurllib3==1.0.0\r\n".encode("utf-8"))

    module.update_req_file(str(target), {2: "urllib3==2.0.0"})

    written = target.read_bytes()
    assert b"\r\n" not in written, f"the requirements fixer wrote CRLF: {written!r}"
    assert "naïve".encode("utf-8") in written, "the non-ASCII comment did not survive as UTF-8"
    assert b"urllib3==2.0.0" in written, written


def test_the_allow_scripts_pin_sync_writes_lf(tmp_path):
    """The other pre-commit hook that rewrites a tracked file, on a file pinned `eol=lf`.

    `.gitattributes` carries `studio/frontend/** text=auto eol=lf`, and .pre-commit-config.yaml
    runs this one with `--fix` on every package.json touch, so it is the same contributor-facing
    surface as the spacing hook above.
    """
    module = _load("sync_allow_scripts_pins.py")
    (tmp_path / "package.json").write_bytes(
        json.dumps({"name": "x", "allowScripts": {"esbuild@0.1.0": True}}).encode("utf-8")
    )
    (tmp_path / "package-lock.json").write_bytes(
        json.dumps(
            {"packages": {"node_modules/esbuild": {"version": "0.2.0", "hasInstallScript": True}}}
        ).encode("utf-8")
    )

    assert module.main(["--fix", "--dir", str(tmp_path)]) == 0

    written = (tmp_path / "package.json").read_bytes()
    assert b"\r\n" not in written, f"the allowScripts pin sync wrote CRLF: {written!r}"
    # Non-vacuous: the re-pin must actually have happened, or "no CRLF" is just "no write".
    assert b'"esbuild@0.2.0"' in written, written
