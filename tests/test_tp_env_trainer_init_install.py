# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_PP_SIZE alone must install the Trainer wrapper (it refuses PP) so a non-Fast* encoder is not trained as plain DDP."""

import ast
import importlib.util
import os
import sys
import types

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV = ("UNSLOTH_TP_SIZE", "UNSLOTH_PP_SIZE", "UNSLOTH_FSDP2", "WORLD_SIZE")


def _load_tp_env(monkeypatch, calls):
    pkg = types.ModuleType("unsloth")
    pkg.__path__ = [os.path.join(REPO, "unsloth")]
    models = types.ModuleType("unsloth.models")
    models.__path__ = [os.path.join(REPO, "unsloth", "models")]
    utils = types.ModuleType("unsloth.models._utils")
    utils.patch_gradient_accumulation_fix = calls.append
    for name, module in (
        ("unsloth", pkg),
        ("unsloth.models", models),
        ("unsloth.models._utils", utils),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "unsloth.models.tp_env", os.path.join(REPO, "unsloth", "models", "tp_env.py")
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "unsloth.models.tp_env", module)
    spec.loader.exec_module(module)
    return module


def _set_env(monkeypatch, env):
    for key in _ENV:
        monkeypatch.delenv(key, raising = False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)


@pytest.mark.parametrize(
    "env,applies",
    [
        ({"UNSLOTH_TP_SIZE": "2", "WORLD_SIZE": "2"}, True),
        ({"UNSLOTH_PP_SIZE": "2", "WORLD_SIZE": "2"}, True),
        ({"UNSLOTH_PP_SIZE": "2", "UNSLOTH_TP_SIZE": "2", "WORLD_SIZE": "4"}, True),
        ({"UNSLOTH_PP_SIZE": "1", "WORLD_SIZE": "2"}, False),
        ({"UNSLOTH_TP_SIZE": "1", "UNSLOTH_PP_SIZE": " ", "WORLD_SIZE": "2"}, False),
        ({"WORLD_SIZE": "2"}, False),
    ],
)
def test_installer_wraps_trainer_under_tp_or_pp(monkeypatch, env, applies):
    from transformers import Trainer

    _set_env(monkeypatch, env)
    calls = []
    tp_env = _load_tp_env(monkeypatch, calls)
    assert tp_env.install_tensor_parallel_env_trainer_init() is applies
    assert calls == ([Trainer] if applies else [])


def test_gpu_import_path_gates_the_installer_on_both_variables():
    tree = ast.parse(open(os.path.join(REPO, "unsloth", "__init__.py"), encoding = "utf-8").read())
    gates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(n, ast.ImportFrom)
            and n.module == "models.tp_env"
            and any(a.name == "install_tensor_parallel_env_trainer_init" for a in n.names)
            for stmt in node.body
            for n in ast.walk(stmt)
        )
    ]
    assert gates, "unsloth/__init__.py no longer installs the UNSLOTH_TP_SIZE Trainer wrapper"
    names = {n.value for g in gates for n in ast.walk(g.test) if isinstance(n, ast.Constant)}
    assert {"UNSLOTH_TP_SIZE", "UNSLOTH_PP_SIZE"} <= names, names


def test_wrapper_refuses_pp_size_alone(monkeypatch):
    torch = pytest.importorskip("torch")
    _set_env(monkeypatch, {"UNSLOTH_PP_SIZE": "2", "WORLD_SIZE": "2"})
    _load_tp_env(monkeypatch, [])
    tree = ast.parse(
        open(os.path.join(REPO, "unsloth", "models", "_utils.py"), encoding = "utf-8").read()
    )
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_tensor_parallel_env_trainer_pre"
    )
    ns = {"__name__": "unsloth.models._utils", "__package__": "unsloth.models", "os": os}
    exec(compile(ast.Module(body = [fn], type_ignores = []), "_utils.py", "exec"), ns)
    with pytest.raises(ValueError, match = "UNSLOTH_PP_SIZE=2"):
        ns["_tensor_parallel_env_trainer_pre"](torch.nn.Linear(2, 2))
