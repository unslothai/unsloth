# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The GRPO trainer's `triton.enable_persistent_tma_matmul` follows unsloth_zoo's torch < 2.12
gate (unslothai/unsloth-zoo#2043), so torch 2.10 / 2.11 emit exactly the previous options."""

import ast
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from packaging.version import Version

RL_PY = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl.py"
SOURCE = RL_PY.read_text(encoding = "utf-8")
KEY = '"triton.enable_persistent_tma_matmul"'
SM90 = "<all_sm90>"

VERSIONS = [
    ("2.10.0+cu128", True),
    ("2.11.0", True),
    ("2.11.1+cu130", True),
    ("2.12.0", False),
    ("2.12.0a0+git1234", False),
    ("2.14.1+cu130", False),
    ("3.0.0", False),
]


def _load_gate(torch_version):
    tree = ast.parse(SOURCE)
    (node,) = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_persistent_tma_matmul_ok"
    ]
    ns = {"Version": Version, "torch_version": torch_version}
    exec(compile(ast.Module(body = [node], type_ignores = []), str(RL_PY), "exec"), ns)
    return ns["_persistent_tma_matmul_ok"]


def _parse(raw):
    return Version(raw.split("+")[0].split("a")[0].split("b")[0])


def _fake_zoo(monkeypatch, version, with_helper):
    def gate():
        major, minor = (int(x) for x in version.split(".")[:2])
        return (major, minor) < (2, 12)

    common = types.ModuleType("unsloth_zoo.temporary_patches.common")
    if with_helper:
        common._persistent_tma_matmul_ok = gate
    patches = types.ModuleType("unsloth_zoo.temporary_patches")
    patches.common = common
    zoo = types.ModuleType("unsloth_zoo")
    zoo.temporary_patches = patches
    monkeypatch.setitem(sys.modules, "unsloth_zoo", zoo)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.temporary_patches", patches)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.temporary_patches.common", common)


def _emit(gate_value):
    # The CUDA options text the GRPO patch builds, run from the live rl.py source.
    start = SOURCE.index("            tma_matmul = ")
    end = SOURCE.index("            new_options = base_options + cuda_options", start)
    block = "if True:\n" + SOURCE[start:end]
    ns = {
        "all_sm90": SM90,
        "_persistent_tma_matmul_ok": lambda: gate_value,
        "torch_version": Version("2.11.0"),
        "Version": Version,
    }
    exec(block, ns)
    return ns["cuda_options"]


@pytest.mark.parametrize("version,ok", VERSIONS)
def test_gate_follows_zoo(monkeypatch, version, ok):
    _fake_zoo(monkeypatch, version, with_helper = True)
    assert _load_gate(_parse(version))() is ok


@pytest.mark.parametrize("version,ok", VERSIONS)
def test_gate_without_zoo_helper_uses_same_cutoff(monkeypatch, version, ok):
    _fake_zoo(monkeypatch, version, with_helper = False)
    assert _load_gate(_parse(version))() is ok


@pytest.mark.parametrize("version,ok", VERSIONS)
def test_gate_matches_real_zoo(monkeypatch, version, ok):
    common = pytest.importorskip("unsloth_zoo.temporary_patches.common")
    if not hasattr(common, "_persistent_tma_matmul_ok"):
        pytest.skip("installed unsloth_zoo predates unslothai/unsloth-zoo#2043")
    monkeypatch.setattr(torch, "__version__", version)
    assert common._persistent_tma_matmul_ok() is ok
    assert _load_gate(_parse(version))() is ok


def test_options_keep_sm90_expression_below_2_12():
    # torch 2.10 / 2.11: the emitted option is unchanged, still the every-card sm90+ check.
    assert f"{KEY}: {SM90}," in _emit(True)


def test_options_disable_tma_from_2_12():
    out = _emit(False)
    assert f"{KEY}: False," in out
    assert SM90 not in out.split(KEY, 1)[1].split("\n", 1)[0]
