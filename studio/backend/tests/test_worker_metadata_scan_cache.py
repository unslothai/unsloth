# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The training worker shares one packages_distributions() scan between transformers and TRL."""

import ast
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_WORKER = _BACKEND / "core" / "training" / "worker.py"
_HELPER = _BACKEND.parents[1] / "unsloth" / "_import_metadata_cache.py"

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10), reason = "importlib.metadata.packages_distributions needs Python 3.10"
)


def _function_source(name):
    tree = ast.parse(_WORKER.read_text(encoding = "utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in worker.py")


_CHILD = textwrap.dedent(
    """
    import ast, importlib.metadata as md, importlib.util, json, os, sys, types
    worker = os.environ["R12_WORKER"]
    tree = ast.parse(open(worker, encoding = "utf-8").read())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ("_install_metadata_scan_cache", "_uninstall_metadata_scan_cache")]
    module = ast.Module(body = funcs, type_ignores = [])
    ns = {"importlib": importlib, "os": os, "Any": object}
    exec(compile(module, worker, "exec"), ns)
    real = md.packages_distributions
    calls = []
    def scan():
        calls.append(1)
        return real()
    md.packages_distributions = scan
    installed = ns["_install_metadata_scan_cache"]()
    wrapped = md.packages_distributions is not scan
    a = md.packages_distributions(); b = md.packages_distributions()
    ns["_uninstall_metadata_scan_cache"](installed)
    print("R12 " + json.dumps({
        "installed": installed is not None, "wrapped": wrapped, "scans": len(calls), "equal": a == b,
        "restored": md.packages_distributions is scan,
        "unsloth_imported": any(m == "unsloth" or m.startswith("unsloth.") for m in sys.modules),
    }))
    """
)


def _run_child(package_root):
    env = dict(os.environ, R12_WORKER = str(_WORKER))
    env["PYTHONPATH"] = str(package_root)
    result = subprocess.run(
        [sys.executable, "-c", _CHILD],
        env = env,
        cwd = str(package_root),
        capture_output = True,
        text = True,
        timeout = 300,
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("R12 ")]
    assert lines, (result.returncode, result.stderr[-3000:])
    return json.loads(lines[-1][4:])


def test_loads_the_helper_by_path_without_importing_unsloth(tmp_path):
    package = tmp_path / "unsloth"
    package.mkdir()
    # An __init__ that would fail loudly if the worker imported the package.
    (package / "__init__.py").write_text(
        "raise RuntimeError('unsloth/__init__.py must not run here')\n"
    )
    shutil.copy(_HELPER, package / "_import_metadata_cache.py")
    report = _run_child(tmp_path)
    assert report == {
        "installed": True,
        "wrapped": True,
        "scans": 1,
        "equal": True,
        "restored": True,
        "unsloth_imported": False,
    }


def test_older_unsloth_without_the_helper_is_left_alone(tmp_path):
    package = tmp_path / "unsloth"
    package.mkdir()
    (package / "__init__.py").write_text("raise RuntimeError('must not run')\n")
    report = _run_child(tmp_path)
    assert report == {
        "installed": False,
        "wrapped": False,
        "scans": 2,
        "equal": True,
        "restored": True,
        "unsloth_imported": False,
    }


def test_installed_before_any_ml_import_and_removed_after_unsloth():
    func = _function_source("run_training_process")
    lines = _WORKER.read_text(encoding = "utf-8").splitlines()
    body = lines[func.lineno - 1 : func.end_lineno]
    install = next(i for i, l in enumerate(body) if "_install_metadata_scan_cache()" in l)
    for line in body[:install]:
        if line.strip().startswith("#"):
            continue
        assert (
            "transformers" not in line and "import torch" not in line and "unsloth" not in line
        ), line
    # Nothing before the install calls a helper that imports transformers.
    for helper in ("_install_fast_path_hooks(", "_model_load_security_error("):
        hits = [i for i, l in enumerate(body) if helper in l]
        assert all(i > install for i in hits), helper

    # Uninstalled in the finally of the try that imports unsloth through core.training.trainer.
    tries = [n for n in ast.walk(func) if isinstance(n, ast.Try)]
    owner = [
        t
        for t in tries
        if any(
            isinstance(n, ast.ImportFrom) and n.module == "core.training.trainer"
            for s in t.body
            for n in ast.walk(s)
        )
    ]
    assert owner, "no try imports core.training.trainer"
    finals = [ast.unparse(s) for s in owner[0].finalbody]
    assert any("_uninstall_metadata_scan_cache(metadata_scan_cache)" in f for f in finals), finals


def test_a_failing_lookup_returns_none(monkeypatch):
    import importlib.util

    func = _function_source("_install_metadata_scan_cache")
    ns = {"importlib": importlib, "os": os, "Any": object}
    exec(compile(ast.Module(body = [func], type_ignores = []), str(_WORKER), "exec"), ns)

    def boom(name):
        raise ValueError(name)

    monkeypatch.setattr(importlib.util, "find_spec", boom)
    import importlib.metadata as md

    before = md.packages_distributions
    assert ns["_install_metadata_scan_cache"]() is None
    assert md.packages_distributions is before
