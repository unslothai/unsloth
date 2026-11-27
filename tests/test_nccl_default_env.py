# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import ast
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

_UNSLOTH = Path(__file__).resolve().parents[1] / "unsloth"
_NAME = "NCCL_COMM_SPLIT_SHARE_RESOURCES"


def _module():
    # By path: importing the unsloth package needs a GPU.
    spec = importlib.util.spec_from_file_location("_unsloth_nccl_env", _UNSLOTH / "nccl_env.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sets_share_resources_by_default():
    env = {}
    assert _module().apply_nccl_default_env(env) == [_NAME]
    assert env == {_NAME: "1"}


@pytest.mark.parametrize("value", ["0", "1", ""])
def test_user_value_wins(value):
    env = {_NAME: value}
    assert _module().apply_nccl_default_env(env) == []
    assert env == {_NAME: value}


def test_untouched_once_the_process_group_exists(monkeypatch):
    fake = types.ModuleType("torch.distributed")
    fake.is_available = lambda: True
    fake.is_initialized = lambda: True
    monkeypatch.setitem(sys.modules, "torch.distributed", fake)
    env = {}
    assert _module().apply_nccl_default_env(env) == []
    assert env == {}


def _top_level_calls(tree):
    """(line, dotted name) of every call made at module level, in source order."""
    calls = []
    for node in tree.body:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                f = sub.func
                name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", "")
                calls.append((sub.lineno, name))
    return sorted(calls)


def test_init_applies_it_before_anything_can_create_a_process_group():
    """unsloth/__init__.py must call apply_nccl_default_env before it imports torch, transformers,
    accelerate or any Unsloth module that builds a mesh (parallel_env, models)."""
    tree = ast.parse((_UNSLOTH / "__init__.py").read_text(encoding = "utf-8"))
    calls = _top_level_calls(tree)
    applied = [line for line, name in calls if name == "_apply_nccl_default_env"]
    assert applied, "unsloth/__init__.py no longer applies the NCCL defaults"
    first = applied[0]
    later_imports = {"torch", "transformers", "accelerate", "trl", "peft"}
    for node in tree.body:
        if node.lineno >= first:
            break
        for sub in ast.walk(node):
            if isinstance(sub, ast.Import):
                assert not ({a.name.split(".")[0] for a in sub.names} & later_imports), ast.dump(
                    sub
                )
            if isinstance(sub, ast.ImportFrom):
                mod = sub.module or ""
                assert mod.split(".")[0] not in later_imports, mod
                assert sub.level == 0 or mod in (
                    "_import_gc",
                    "_import_metadata_cache",
                    "distributed_debug",
                    "nccl_env",
                ), mod


def test_process_group_sees_the_default_and_the_override():
    """Flip test: the value NCCL reads at init_process_group (gloo here, so no GPU is needed) is 1 when the
    user set nothing and the user's own value otherwise, with the module applied first, as unsloth/__init__ does."""
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")
    code = (
        "import importlib.util, os, sys\n"
        "spec = importlib.util.spec_from_file_location('ne', sys.argv[1])\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "m.apply_nccl_default_env()\n"
        "import torch.distributed as dist\n"
        "dist.init_process_group('gloo', init_method='tcp://127.0.0.1:' + sys.argv[2], rank=0, world_size=1)\n"
        "late = m.apply_nccl_default_env({})\n"
        "print('AT_INIT=' + os.environ.get('" + _NAME + "', '<unset>') + ' LATE=' + str(late))\n"
        "dist.destroy_process_group()\n"
    )

    def run(extra, port):
        env = {k: v for k, v in os.environ.items() if k != _NAME}
        env.update(extra)
        out = subprocess.run(
            [sys.executable, "-c", code, str(_UNSLOTH / "nccl_env.py"), str(port)],
            env = env,
            capture_output = True,
            text = True,
            timeout = 300,
        )
        assert out.returncode == 0, out.stderr[-2000:]
        return [line for line in out.stdout.splitlines() if line.startswith("AT_INIT=")][0]

    assert run({}, 29611) == "AT_INIT=1 LATE=[]"
    assert run({_NAME: "0"}, 29612) == "AT_INIT=0 LATE=[]"
