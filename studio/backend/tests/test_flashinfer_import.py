# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""import_flashinfer vs flashinfer 0.7.3's ptxas override, CPU only with fake ptxas / Triton / flashinfer."""

import os
import stat
import sys
import textwrap
import types

import pytest

from core.inference import flashinfer_import as fi

ENV = "TRITON_PTXAS_BLACKWELL_PATH"

# flashinfer/triton/__init__.py at v0.7.3 (and v0.6.6), the function as shipped.
_FLASHINFER_PATCH = '''
import os
import shutil
import subprocess
import re


def _patch_triton_ptxas_blackwell():
    """Point Triton at the system ptxas for Blackwell (CUDA >= 13)."""
    if os.environ.get("TRITON_PTXAS_BLACKWELL_PATH"):
        return
    ptxas = shutil.which("ptxas")
    if not ptxas:
        return
    try:
        out = subprocess.check_output(
            [ptxas, "--version"], stderr=subprocess.STDOUT
        ).decode()
    except subprocess.CalledProcessError:
        return
    m = re.search(r"release (\\d+)\\.", out)
    if m and int(m.group(1)) >= 13:
        os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = ptxas


_patch_triton_ptxas_blackwell()
'''


def _fake_ptxas(directory, release: str):
    directory.mkdir(parents = True, exist_ok = True)
    path = directory / "ptxas"
    path.write_text(
        "#!/bin/sh\n"
        "echo 'ptxas: NVIDIA (R) Ptx optimizing assembler'\n"
        f"echo 'Cuda compilation tools, release {release}, V{release}.88'\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(path)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV, raising = False)
    for name in [m for m in sys.modules if m == "flashinfer" or m.startswith("flashinfer.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(fi, "_LOGGED", False)

    def build(
        system = "13.0",
        bundled = "13.1",
        fail_after_patch = False,
    ):
        ptxas = _fake_ptxas(tmp_path / "cuda" / "bin", system) if system else None
        monkeypatch.setenv(
            "PATH", str(tmp_path / "cuda" / "bin") + os.pathsep + os.environ.get("PATH", "")
        )
        if bundled is None:
            monkeypatch.setitem(sys.modules, "triton", None)
        else:
            bundled_path = _fake_ptxas(tmp_path / "triton_bin", bundled)
            compiler = types.ModuleType("triton.backends.nvidia.compiler")

            def get_ptxas(arch):
                assert arch >= 100
                path = os.environ.get(ENV) or bundled_path
                release = fi.ptxas_release(path)
                return types.SimpleNamespace(path = path, version = "%d.%d" % release)

            compiler.get_ptxas = get_ptxas
            for name in ("triton", "triton.backends", "triton.backends.nvidia"):
                monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
            monkeypatch.setitem(sys.modules, "triton.backends.nvidia.compiler", compiler)
        pkg = tmp_path / "site" / "flashinfer"
        (pkg / "triton").mkdir(parents = True, exist_ok = True)
        (pkg / "triton" / "__init__.py").write_text(_FLASHINFER_PATCH)
        init = "__version__ = '0.7.3'\nfrom . import triton as triton\n"
        if fail_after_patch:
            init += "raise ImportError('cutlass missing')\n"
        (pkg / "__init__.py").write_text(init)
        monkeypatch.syspath_prepend(str(tmp_path / "site"))
        return ptxas

    return build


def test_parse_ptxas_release():
    text = "ptxas: NVIDIA (R) Ptx optimizing assembler\nCuda compilation tools, release 13.0, V13.0.88\n"
    assert fi.parse_ptxas_release(text) == (13, 0)
    assert fi.parse_ptxas_release("Cuda compilation tools, release 12.9, V12.9.86") == (12, 9)
    assert fi.parse_ptxas_release("garbage") is None
    assert fi.parse_ptxas_release("") is None


def test_older_system_ptxas_is_undone(env):
    env(system = "13.0", bundled = "13.1")
    module = fi.import_flashinfer()
    assert module.__version__ == "0.7.3"
    assert "flashinfer.triton" in sys.modules
    assert ENV not in os.environ


def test_equal_system_ptxas_is_undone(env):
    # flashinfer #10079 only takes a strictly newer one.
    env(system = "13.3", bundled = "13.3")
    fi.import_flashinfer()
    assert ENV not in os.environ


def test_newer_system_ptxas_is_kept(env):
    ptxas = env(system = "13.3", bundled = "13.1")
    fi.import_flashinfer()
    assert os.environ[ENV] == ptxas


def test_cuda12_system_ptxas_flashinfer_does_nothing(env):
    env(system = "12.9", bundled = "13.1")
    fi.import_flashinfer()
    assert ENV not in os.environ


def test_unknown_bundled_ptxas_leaves_flashinfer_choice(env):
    ptxas = env(system = "13.0", bundled = None)
    fi.import_flashinfer()
    assert os.environ[ENV] == ptxas


def test_user_set_value_is_never_touched(env, tmp_path, monkeypatch):
    env(system = "13.0", bundled = "13.1")
    user = _fake_ptxas(tmp_path / "user_bin", "12.8")
    monkeypatch.setenv(ENV, user)
    fi.import_flashinfer()
    assert os.environ[ENV] == user
    os.environ[ENV] = _fake_ptxas(tmp_path / "other_bin", "13.0")
    assert fi.restore_triton_ptxas(user) is False
    assert os.environ[ENV] == str(tmp_path / "other_bin" / "ptxas")


def test_failed_import_still_restores_and_raises(env):
    env(system = "13.0", bundled = "13.1", fail_after_patch = True)
    with pytest.raises(ImportError, match = "cutlass missing"):
        fi.import_flashinfer()
    assert ENV not in os.environ


def test_already_imported_is_returned_untouched(env, monkeypatch, tmp_path):
    env(system = "13.0", bundled = "13.1")
    sentinel = types.ModuleType("flashinfer")
    monkeypatch.setitem(sys.modules, "flashinfer", sentinel)
    monkeypatch.setenv(ENV, "/some/where/ptxas")
    assert fi.import_flashinfer() is sentinel
    assert os.environ[ENV] == "/some/where/ptxas"


def test_logs_once(env, caplog):
    ptxas = env(system = "13.0", bundled = "13.1")
    with caplog.at_level("INFO", logger = fi.__name__):
        fi.import_flashinfer()
        os.environ[ENV] = ptxas
        assert fi.restore_triton_ptxas(None) is True
    assert ENV not in os.environ
    assert sum("keeping Triton's own" in r.getMessage() for r in caplog.records) == 1


def test_studio_import_sites_go_through_the_guard():
    """Every function in Studio that imports flashinfer calls import_flashinfer first."""
    import ast
    import pathlib

    root = pathlib.Path(fi.__file__).parent
    missing = []
    for path in sorted(root.glob("*.py")):
        if path.name == "flashinfer_import.py":
            continue
        tree = ast.parse(path.read_text())
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            first_fi = first_guard = None
            for node in ast.walk(fn):
                line = getattr(node, "lineno", None)
                if isinstance(node, ast.Import) and any(
                    a.name.split(".")[0] == "flashinfer" for a in node.names
                ):
                    first_fi = min(first_fi or line, line)
                elif (
                    isinstance(node, ast.ImportFrom)
                    and (node.module or "").split(".")[0] == "flashinfer"
                    and not node.level
                ):
                    first_fi = min(first_fi or line, line)
                elif (
                    isinstance(node, ast.Call)
                    and getattr(node.func, "id", None) == "import_flashinfer"
                ):
                    first_guard = min(first_guard or line, line)
            if first_fi is not None and (first_guard is None or first_guard > first_fi):
                missing.append(f"{path.name}:{fn.name}:{first_fi}")
    assert not missing, missing
