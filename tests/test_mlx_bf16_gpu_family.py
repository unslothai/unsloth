# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""MLX is_bfloat16_supported(): Metal GPU family first (g13/g14 lack bf16), chip name only as fallback."""

import ast
import sys
import types
from pathlib import Path

import pytest

INIT = Path(__file__).resolve().parents[1] / "unsloth" / "__init__.py"

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
    pytest.param({"device_name": "Apple M1"}, False, id = "name-only-m1"),
    pytest.param({"device_name": "Apple M2 Ultra", "architecture": ""}, False, id = "name-only-m2"),
    pytest.param({"device_name": "Apple M3"}, True, id = "name-only-m3"),
    pytest.param({"device_name": "Apple Paravirtual device"}, True, id = "name-only-vm"),
    pytest.param({}, True, id = "empty"),
    pytest.param(
        {"device_name": "Apple M2", "architecture": "applegpu_g16g"}, True, id = "family-wins"
    ),
    # MLX's device.cpp always reads a class letter after the two family digits; without one, fall back.
    pytest.param(
        {"device_name": "Apple M3", "architecture": "applegpu_g13"},
        True,
        id = "no-class-letter-uses-name",
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


def _mlx_from_pretrained(bf16):
    tree = ast.parse(INIT.read_text(encoding = "utf-8"))
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_mlx_from_pretrained"
    )
    calls = []
    namespace = {
        "is_bfloat16_supported": lambda: bf16,
        "FastMLXModel": types.SimpleNamespace(
            from_pretrained = lambda *a, **k: calls.append((a, k))
        ),
    }
    exec(compile(ast.Module(body = [node], type_ignores = []), str(INIT), "exec"), namespace)
    return namespace["_mlx_from_pretrained"], calls


@pytest.mark.parametrize(
    "args,kwargs,bf16,expected",
    [
        pytest.param(("m",), {}, True, (("m",), {}), id = "bf16-untouched"),
        pytest.param(("m",), {}, False, (("m",), {"dtype": "float16"}), id = "default-kw"),
        pytest.param(("m",), {"dtype": None}, False, (("m",), {"dtype": "float16"}), id = "none-kw"),
        pytest.param(
            ("m", 2048, None, True), {}, False, (("m", 2048, "float16", True), {}), id = "none-pos"
        ),
        pytest.param(
            ("m",), {"dtype": "bfloat16"}, False, (("m",), {"dtype": "bfloat16"}), id = "explicit-wins"
        ),
        pytest.param(
            ("m", 2048, "float32"), {}, False, (("m", 2048, "float32"), {}), id = "explicit-pos-wins"
        ),
    ],
)
def test_loader_default_dtype_follows_gpu_family(args, kwargs, bf16, expected):
    """zoo's loader defaults by name only, so a g13/g14 VM would load bf16 without this."""
    load, calls = _mlx_from_pretrained(bf16)
    load(*args, **kwargs)
    assert calls == [expected]
