# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""MLX is_bfloat16_supported() decides from the Metal GPU family (applegpu_g13 = M1, g14 = M2 lack
native bf16) and falls back to the chip name only when no family is reported, so a VM or renamed
device on an M1/M2 host no longer claims bf16. Same answer as unsloth_zoo's _apple_bf16_supported."""

import ast
import sys
import types
from pathlib import Path

import pytest

INIT = Path(__file__).resolve().parents[1] / "unsloth" / "__init__.py"

# (device_info dict, expected)
CASES = [
    pytest.param({"device_name": "Apple M1 Max", "architecture": "applegpu_g13s"}, False, id = "m1"),
    pytest.param({"device_name": "Apple M2", "architecture": "applegpu_g14g"}, False, id = "m2"),
    pytest.param({"device_name": "Apple M3 Pro", "architecture": "applegpu_g15s"}, True, id = "m3"),
    pytest.param({"device_name": "Apple M4", "architecture": "applegpu_g16g"}, True, id = "m4"),
    pytest.param({"device_name": "Apple M5", "architecture": "applegpu_g17g"}, True, id = "m5"),
    pytest.param(
        {"device_name": "Apple Paravirtual device", "architecture": "applegpu_g14s"},
        False,
        id = "vm-on-m2",
    ),
    pytest.param(
        {"device_name": "Apple Paravirtual device", "architecture": "applegpu_g16s"},
        True,
        id = "vm-on-m4",
    ),
    # No family reported: the name check decides, as before.
    pytest.param({"device_name": "Apple M1"}, False, id = "name-only-m1"),
    pytest.param({"device_name": "Apple M2 Ultra", "architecture": ""}, False, id = "name-only-m2"),
    pytest.param({"device_name": "Apple M3"}, True, id = "name-only-m3"),
    pytest.param({"device_name": "Apple Paravirtual device"}, True, id = "name-only-vm"),
    pytest.param({}, True, id = "empty"),
    # The family decides when present, as in unsloth_zoo.device_type._apple_bf16_supported.
    pytest.param(
        {"device_name": "Apple M2", "architecture": "applegpu_g16g"}, True, id = "family-wins"
    ),
    pytest.param(
        {"device_name": "Apple M1", "architecture": "applegpu_g13"}, False, id = "no-class-letter"
    ),
]


def _is_bfloat16_supported():
    tree = ast.parse(INIT.read_text(encoding = "utf-8"))
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "is_bfloat16_supported"
    )
    namespace = {}
    exec(compile(ast.Module(body = [node], type_ignores = []), str(INIT), "exec"), namespace)
    return namespace["is_bfloat16_supported"]


def _install_mlx(
    monkeypatch,
    info,
    top_level = True,
):
    core = types.ModuleType("mlx.core")
    core.metal = types.SimpleNamespace(device_info = lambda: dict(info))
    if top_level:
        core.device_info = lambda: dict(info)
    mlx = types.ModuleType("mlx")
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)


@pytest.mark.parametrize("info,expected", CASES)
def test_bf16_from_gpu_family_then_name(monkeypatch, info, expected):
    _install_mlx(monkeypatch, info)
    assert _is_bfloat16_supported()() is expected


def test_old_mlx_without_top_level_device_info(monkeypatch):
    _install_mlx(monkeypatch, {"device_name": "Apple M1", "architecture": "applegpu_g13g"}, False)
    assert _is_bfloat16_supported()() is False


def test_probe_failure_keeps_true(monkeypatch):
    core = types.ModuleType("mlx.core")

    def boom():
        raise RuntimeError("no Metal")

    core.device_info = boom
    monkeypatch.setitem(sys.modules, "mlx", types.ModuleType("mlx"))
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    assert _is_bfloat16_supported()() is True


@pytest.mark.parametrize("info,_expected", CASES)
def test_matches_zoo_answer(monkeypatch, info, _expected):
    zoo_dt = pytest.importorskip("unsloth_zoo.device_type")
    if not hasattr(zoo_dt, "_apple_bf16_supported"):
        pytest.skip("installed unsloth_zoo predates _apple_bf16_supported")
    _install_mlx(monkeypatch, info)
    assert _is_bfloat16_supported()() is zoo_dt._apple_bf16_supported()
