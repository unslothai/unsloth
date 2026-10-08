# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""sync_load_when_quantizing, extracted with ast so nothing imports CUDA. No GPU needed."""

import ast
import contextlib
import os
import types

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(HERE, "unsloth", "models")
ENV = "HF_DEACTIVATE_ASYNC_LOAD"


def _source(name):
    with open(os.path.join(MODELS, name), encoding = "utf-8") as f:
        return f.read()


def _load_helper():
    tree = ast.parse(_source("loader_utils.py"))
    keep = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "sync_load_when_quantizing")
        or (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "_ASYNC_LOAD_ENV" for t in node.targets)
        )
    ]
    assert len(keep) == 2
    namespace = {"contextlib": contextlib, "os": os}
    exec(compile(ast.Module(body = keep, type_ignores = []), "loader_utils", "exec"), namespace)
    return namespace["sync_load_when_quantizing"]


sync_load_when_quantizing = _load_helper()
BNB = object()
UNQUANTIZED = types.SimpleNamespace()
PREQUANTIZED = types.SimpleNamespace(quantization_config = {"quant_method": "bitsandbytes"})


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(ENV, raising = False)


def test_on_the_fly_quantization_loads_synchronously_then_restores():
    with sync_load_when_quantizing(BNB, UNQUANTIZED):
        assert os.environ.get(ENV) == "1"
    assert ENV not in os.environ


def test_restores_when_the_load_raises():
    with pytest.raises(RuntimeError):
        with sync_load_when_quantizing(BNB, UNQUANTIZED):
            raise RuntimeError("out of memory")
    assert ENV not in os.environ


@pytest.mark.parametrize(
    "quantization_config, config",
    [(BNB, PREQUANTIZED), (None, UNQUANTIZED), (None, PREQUANTIZED)],
    ids = ["prequantized", "16bit", "16bit-prequantized-config"],
)
def test_leaves_async_loading_alone_when_nothing_is_quantized_on_the_fly(
    quantization_config, config
):
    with sync_load_when_quantizing(quantization_config, config):
        assert ENV not in os.environ
    assert ENV not in os.environ


def test_a_caller_setting_wins(monkeypatch):
    monkeypatch.setenv(ENV, "0")
    with sync_load_when_quantizing(BNB, UNQUANTIZED):
        assert os.environ[ENV] == "0"
    assert os.environ[ENV] == "0"


def _calls_inside_sync_load(src, callee):
    """Each `<callee>.from_pretrained(...)` call, paired with whether a `with` around it uses the helper."""
    tree = ast.parse(src)
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    found = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_pretrained"
            and getattr(node.func.value, "id", None) == callee
        ):
            continue
        wrapped, cur = False, node
        while cur in parents:
            cur = parents[cur]
            if isinstance(cur, ast.With) and any(
                isinstance(item.context_expr, ast.Call)
                and getattr(item.context_expr.func, "id", None) == "sync_load_when_quantizing"
                for item in cur.items
            ):
                wrapped = True
                break
        found.append((node.lineno, wrapped))
    return found


@pytest.mark.parametrize(
    "module, callee",
    [
        ("vision.py", "auto_model"),
        ("llama.py", "AutoModelForCausalLM"),
        ("llama.py", "AutoModelForSequenceClassification"),
        ("loader_utils.py", "auto_model"),
        ("../save.py", "auto_model"),
    ],
)
def test_every_on_the_fly_quantizing_load_runs_inside_the_helper(module, callee):
    calls = _calls_inside_sync_load(_source(module), callee)
    assert calls, f"no {callee}.from_pretrained in {module}"
    assert all(wrapped for _, wrapped in calls), calls


def test_transformers_reads_the_variable_where_it_picks_the_loader():
    # Pins the contract the helper leans on: the variable is what turns the thread pool off.
    core = pytest.importorskip("transformers.core_model_loading")
    import inspect

    src = inspect.getsource(core.convert_and_load_state_dict_in_model)
    assert 'is_env_variable_true("HF_DEACTIVATE_ASYNC_LOAD")' in src
    assert "thread_pool = None" in src
