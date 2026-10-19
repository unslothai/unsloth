# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import ast
import importlib.util
from pathlib import Path

import pytest

_UNSLOTH = Path(__file__).resolve().parents[1] / "unsloth"


def _module():
    # By path: importing the unsloth package needs a GPU.
    spec = importlib.util.spec_from_file_location(
        "_unsloth_distributed_debug", _UNSLOTH / "distributed_debug.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_off_by_default():
    env = {}
    assert _module().apply_debug_distributed_env(env) == []
    assert env == {}


@pytest.mark.parametrize("flag", ["1", "true", "ON"])
def test_sets_nccl_and_both_flight_recorder_names(flag):
    env = {"UNSLOTH_DEBUG_DISTRIBUTED": flag}
    _module().apply_debug_distributed_env(env)
    assert env["TORCH_NCCL_DESYNC_DEBUG"] == "1"
    assert env["TORCH_NCCL_DUMP_ON_TIMEOUT"] == "1"
    assert env["NCCL_DEBUG"] == "INFO"
    # ProcessGroupNCCL reads only the old name for dump on timeout, the recorder the new one.
    assert env["TORCH_FR_BUFFER_SIZE"] == "2000"
    assert env["TORCH_NCCL_TRACE_BUFFER_SIZE"] == "2000"


def test_user_values_win_and_old_buffer_name_is_respected():
    env = {
        "UNSLOTH_DEBUG_DISTRIBUTED": "1",
        "NCCL_DEBUG": "WARN",
        "TORCH_NCCL_TRACE_BUFFER_SIZE": "0",
    }
    changed = _module().apply_debug_distributed_env(env)
    assert env["NCCL_DEBUG"] == "WARN"
    assert env["TORCH_NCCL_TRACE_BUFFER_SIZE"] == "0"
    assert env["TORCH_FR_BUFFER_SIZE"] == "0"
    assert set(changed) == {
        "TORCH_NCCL_DESYNC_DEBUG",
        "TORCH_NCCL_DUMP_ON_TIMEOUT",
        "TORCH_FR_BUFFER_SIZE",
    }


def test_new_buffer_name_is_mirrored_to_the_one_process_group_nccl_reads():
    env = {"UNSLOTH_DEBUG_DISTRIBUTED": "1", "TORCH_FR_BUFFER_SIZE": "5000"}
    _module().apply_debug_distributed_env(env)
    assert env["TORCH_NCCL_TRACE_BUFFER_SIZE"] == "5000"


def test_hook_runs_before_torch_is_imported():
    tree = ast.parse((_UNSLOTH / "__init__.py").read_text(encoding = "utf-8"))
    hook = first_torch = None
    for node in tree.body:
        src = ast.unparse(node)
        if hook is None and "apply_debug_distributed_env" in src:
            hook = node.lineno
        if first_torch is None and ("import torch" in src or "from torch" in src):
            first_torch = node.lineno
    assert hook is not None and (first_torch is None or hook < first_torch)


def test_existing_process_group_leaves_the_environment_untouched(monkeypatch, capsys):
    import sys
    import types

    dist = types.SimpleNamespace(is_available = lambda: True, is_initialized = lambda: True)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    env = {"UNSLOTH_DEBUG_DISTRIBUTED": "1"}
    assert _module().apply_debug_distributed_env(env) == []
    assert env == {"UNSLOTH_DEBUG_DISTRIBUTED": "1"}
    assert "already exists" in capsys.readouterr().err


@pytest.mark.parametrize("user_value", [None, "TRACE"])
def test_nccl_debug_survives_zoo_logging_switches(user_value):
    # unsloth_zoo sets NCCL_DEBUG=WARN on import under UNSLOTH_ENABLE_LOGGING / UNSLOTH_COMPILE_DEBUG.
    mod = _module()
    env = {"UNSLOTH_DEBUG_DISTRIBUTED": "1", "UNSLOTH_ENABLE_LOGGING": "1"}
    if user_value is not None:
        env["NCCL_DEBUG"] = user_value
    mod.apply_debug_distributed_env(env)
    restore = mod.keep_debug_nccl_env(env)
    env["NCCL_DEBUG"] = "WARN"
    restore()
    assert env["NCCL_DEBUG"] == (user_value or "INFO")


def test_keep_nccl_debug_is_a_no_op_when_off():
    mod = _module()
    env = {"UNSLOTH_ENABLE_LOGGING": "1"}
    restore = mod.keep_debug_nccl_env(env)
    env["NCCL_DEBUG"] = "WARN"
    restore()
    assert env["NCCL_DEBUG"] == "WARN"


def test_gpu_init_restores_nccl_debug_around_zoo_import():
    source = (_UNSLOTH / "_gpu_init.py").read_text(encoding = "utf-8")
    zoo = source.index("\n    import unsloth_zoo\n")
    keep = source.rfind("keep_debug_nccl_env()", 0, zoo)
    restore = source.find("_restore_nccl_debug()", zoo)
    assert keep != -1 and restore != -1
    assert source.count("\n    import unsloth_zoo\n") == 1
    assert restore < source.index("except PackageNotFoundError", zoo)


@pytest.mark.parametrize("switch", ["UNSLOTH_ENABLE_LOGGING", "UNSLOTH_COMPILE_DEBUG"])
def test_real_zoo_import_keeps_nccl_info(switch):
    import os
    import subprocess
    import sys

    if importlib.util.find_spec("unsloth_zoo") is None:
        pytest.skip("unsloth_zoo not installed")
    code = (
        "import importlib.util, os, sys\n"
        "spec = importlib.util.spec_from_file_location('dd', sys.argv[1])\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "m.apply_debug_distributed_env()\n"
        "restore = m.keep_debug_nccl_env()\n"
        "try:\n"
        "    import unsloth_zoo\n"
        "except BaseException:\n"
        "    pass\n"  # CPU hosts raise after the env block ran; only NCCL_DEBUG matters here.
        "restore()\n"
        "print('NCCL_DEBUG=' + os.environ.get('NCCL_DEBUG', ''))\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "NCCL_DEBUG"}
    env.update({"UNSLOTH_DEBUG_DISTRIBUTED": "1", switch: "1"})
    out = subprocess.run(
        [sys.executable, "-c", code, str(_UNSLOTH / "distributed_debug.py")],
        env = env,
        capture_output = True,
        text = True,
        timeout = 600,
    ).stdout
    assert "NCCL_DEBUG=INFO" in out.splitlines()
