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
def test_sets_nccl_and_flight_recorder_env(flag):
    env = {"UNSLOTH_DEBUG_DISTRIBUTED": flag}
    _module().apply_debug_distributed_env(env)
    assert env["TORCH_NCCL_DESYNC_DEBUG"] == "1"
    assert env["TORCH_NCCL_DUMP_ON_TIMEOUT"] == "1"
    assert env["NCCL_DEBUG"] == "INFO"
    assert env["TORCH_FR_BUFFER_SIZE"] == env["TORCH_NCCL_TRACE_BUFFER_SIZE"] == "2000"


def test_user_values_win_and_old_buffer_name_is_respected():
    env = {
        "UNSLOTH_DEBUG_DISTRIBUTED": "1",
        "NCCL_DEBUG": "WARN",
        "TORCH_NCCL_TRACE_BUFFER_SIZE": "0",
    }
    changed = _module().apply_debug_distributed_env(env)
    assert env["NCCL_DEBUG"] == "WARN"
    assert env["TORCH_NCCL_TRACE_BUFFER_SIZE"] == "0"
    assert "TORCH_FR_BUFFER_SIZE" not in env
    assert set(changed) == {"TORCH_NCCL_DESYNC_DEBUG", "TORCH_NCCL_DUMP_ON_TIMEOUT"}


def test_hook_runs_before_torch_is_imported():
    tree = ast.parse((_UNSLOTH / "__init__.py").read_text())
    hook = first_torch = None
    for node in tree.body:
        src = ast.unparse(node)
        if hook is None and "apply_debug_distributed_env" in src:
            hook = node.lineno
        if first_torch is None and ("import torch" in src or "from torch" in src):
            first_torch = node.lineno
    assert hook is not None and (first_torch is None or hook < first_torch)
