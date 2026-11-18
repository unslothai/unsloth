# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The backend must not put the venv's unsigned hipInfo.exe on PATH (Smart App Control blocks it)."""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_FILES = [_BACKEND / "main.py", _BACKEND / "core" / "training" / "worker.py"]


def _path_writes(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Subscript) and ast.unparse(t) == "os.environ['PATH']":
                    yield node


def _cuda_specs_filters(tree: ast.AST):
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "addFilter"
            and "bitsandbytes.cuda_specs" in ast.unparse(node.func.value)
        ):
            yield eval(compile(ast.Expression(node.args[0]), "<filter>", "eval"))


@pytest.mark.parametrize("path", _FILES, ids = lambda p: p.name)
def test_no_hipinfo_path_injection(path):
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    assert not list(_path_writes(tree))


@pytest.mark.parametrize("path", _FILES, ids = lambda p: p.name)
def test_arch_probe_log_dropped_others_kept(path):
    filters = list(_cuda_specs_filters(ast.parse(path.read_text(encoding = "utf-8"))))
    assert len(filters) == 1

    def record(func):
        r = logging.LogRecord("bitsandbytes.cuda_specs", logging.ERROR, "x", 1, "m", None, None)
        r.funcName = func
        return r

    assert not filters[0](record("get_rocm_gpu_arch"))
    assert filters[0](record("get_cuda_specs"))


def _load_iter_hipinfo_paths():
    src = (_BACKEND.parents[1] / "unsloth" / "import_fixes.py").read_text(encoding = "utf-8")
    fn = next(
        n
        for n in ast.parse(src).body
        if isinstance(n, ast.FunctionDef) and n.name == "_iter_hipinfo_paths"
    )
    import os, sys

    ns = {"os": os, "sys": sys}
    exec(compile(ast.Module([fn], []), "import_fixes.py", "exec"), ns)
    return ns["_iter_hipinfo_paths"]


def test_unsloth_fallback_skips_venv_hipinfo(tmp_path, monkeypatch):
    import shutil, sys, sysconfig

    scripts = tmp_path / "venv" / "Scripts"
    sdk = tmp_path / "sdk"
    (sdk / "bin").mkdir(parents = True)
    scripts.mkdir(parents = True)
    for d in (scripts, sdk / "bin"):
        (d / "hipInfo.exe").write_bytes(b"")
    monkeypatch.setattr(sys, "executable", str(scripts / "python.exe"))
    monkeypatch.setattr(sysconfig, "get_path", lambda name: str(scripts))
    monkeypatch.setenv("PATH", str(scripts))
    monkeypatch.setenv("HIP_PATH", str(sdk))
    monkeypatch.delenv("ROCM_PATH", raising = False)
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "nope"))

    assert list(_load_iter_hipinfo_paths()()) == [str(sdk / "bin" / "hipInfo.exe")]


def test_unsloth_fallback_keeps_searching_path_past_the_venv(tmp_path, monkeypatch):
    import os, sys, sysconfig

    scripts = tmp_path / "venv" / "Scripts"
    custom = tmp_path / "custom-hip" / "bin"
    for d in (scripts, custom):
        d.mkdir(parents = True)
        exe = d / "hipinfo.exe"
        exe.write_bytes(b"")
        exe.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(scripts / "python.exe"))
    monkeypatch.setattr(sysconfig, "get_path", lambda name: str(scripts))
    monkeypatch.setenv("PATH", os.pathsep.join([str(scripts), str(custom)]))
    monkeypatch.delenv("HIP_PATH", raising = False)
    monkeypatch.delenv("ROCM_PATH", raising = False)
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "nope"))

    assert list(_load_iter_hipinfo_paths()()) == [str(custom / "hipinfo.exe")]
