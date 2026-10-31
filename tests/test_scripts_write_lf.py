# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""In-place rewriters must write LF everywhere; the default newline translates to CRLF on Windows."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"

#: (path, function holding the write); named on real files so a rename fails here.
#: Rule is "rewrites a tracked file in place"; sync_allow_scripts_pins rewrites eol=lf files.
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


def _text_write_calls(tree: ast.AST, func_name: str) -> list[ast.Call]:
    """Text-mode write calls in a function, including Path.write_text, which defaults newline like
    open()."""
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
            calls.append(node)
            continue
        if named not in ("fdopen", "open"):
            continue
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
    """Each text-mode write must name its newline; Linux tests cannot see the default come back."""
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


def test_the_spacing_pass_writes_lf_for_a_crlf_source(tmp_path):
    """The headline path: what the pre-commit hook does to a contributor's file."""
    module = _load("enforce_kwargs_spacing.py")
    target = tmp_path / "sample.py"
    target.write_bytes(b"def f(a=1, b=2):\r\n    return a + b\r\n")

    module.process_file(target)

    written = target.read_bytes()
    assert (
        b"\r\n" not in written
    ), f"the spacing pass wrote CRLF into a file .gitattributes pins to LF: {written!r}"
    assert written.endswith(b"\n"), written
    # Non-vacuous: without a real write, "no CRLF" proves nothing.
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
    target.write_bytes("requests==2.0.0  # naïve pin\r\nurllib3==1.0.0\r\n".encode("utf-8"))

    module.update_req_file(str(target), {2: "urllib3==2.0.0"})

    written = target.read_bytes()
    assert b"\r\n" not in written, f"the requirements fixer wrote CRLF: {written!r}"
    assert "naïve".encode("utf-8") in written, "the non-ASCII comment did not survive as UTF-8"
    assert b"urllib3==2.0.0" in written, written


def test_the_allow_scripts_pin_sync_writes_lf(tmp_path):
    """The allow-scripts pin sync must write LF, as studio/frontend/package.json is pinned eol=lf."""
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
    assert b'"esbuild@0.2.0"' in written, written
