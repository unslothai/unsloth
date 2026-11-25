# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import ast
import pathlib

_VISION = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "vision.py"


def _from_pretrained():
    tree = ast.parse(_VISION.read_text(encoding = "utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FastBaseModel")
    return next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "from_pretrained"
    )


def _first_line(fn, predicate):
    return min(node.lineno for node in ast.walk(fn) if predicate(node))


def _calls(name):
    return lambda node: (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    )


def test_inkling2_planner_config_keys_on_the_released_modelopt_marker():
    fn = _from_pretrained()
    marker_read = _first_line(fn, _calls("inkling2_owns_modelopt_quantization"))
    marker_dropped = _first_line(fn, _calls("check_and_disable_bitsandbytes_loading"))
    assert marker_read < marker_dropped
    guards = [
        node.test
        for node in ast.walk(fn)
        if isinstance(node, ast.If)
        and any(
            isinstance(s, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "_planner_config" for t in s.targets)
            and isinstance(s.value, ast.Name)
            and s.value.id == "auto_config"
            for s in node.body
        )
    ]
    names = {n.id for g in guards for n in ast.walk(g) if isinstance(n, ast.Name)}
    assert "_inkling2_modelopt" in names
    assert "_is_inkling2_config" not in names
