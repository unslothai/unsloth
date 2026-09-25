# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A gpt-oss MXFP4 LoRA load without load_in_16bit must take unsloth_zoo's packed-experts path.

load_in_4bit = False (no load_in_16bit) used to keep the native Mxfp4GptOssExperts, whose
triton_kernels matmul_ogs forward has no backward: Unsloth's copy raises "Backwards pass
using MXFP4 is still under construction", and the transformers copy returns an output
detached from the graph, so LoRA below the MoE layers trains on the residual gradient only.
When unsloth_zoo keeps the experts packed (still MXFP4 in memory, decoded one layer at a
time in a differentiable forward), the loader now asks for Mxfp4Config(dequantize = True)
so that path is taken. Without the zoo path, the native load is unchanged.

Source-level, because reaching the branch needs a real checkpoint download. No GPU needed.
"""

import ast
import inspect
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VISION = ROOT / "unsloth" / "models" / "vision.py"
SRC = VISION.read_text(encoding = "utf-8")
TREE = ast.parse(SRC)
ZOO_MXFP4 = "unsloth_zoo.temporary_patches.mxfp4"


def _helper():
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_mxfp4_lora_keeps_experts_packed":
            ns = {}
            exec(ast.get_source_segment(SRC, node), ns)
            return ns["_mxfp4_lora_keeps_experts_packed"]
    raise AssertionError("_mxfp4_lora_keeps_experts_packed not found in vision.py")


def _dequantize_branch():
    """The `if` in FastBaseModel.from_pretrained that sets quantizer_kwargs["dequantize"]."""
    found = []
    for node in ast.walk(TREE):
        if not isinstance(node, ast.If):
            continue
        body = "\n".join(ast.get_source_segment(SRC, stmt) or "" for stmt in node.body)
        if 'quantizer_kwargs["dequantize"] = True' in body and len(node.body) == 1:
            found.append(node)
    assert len(found) == 1, "the dequantize branch moved; update this test"
    return found[0]


class _Mxfp4Config:
    def __init__(self, modules_to_not_convert = None, dequantize = False, **kwargs):
        pass


class _OtherConfig:
    def __init__(self, bits = 4, **kwargs):
        pass


@pytest.fixture
def zoo(monkeypatch):
    """Install a stub unsloth_zoo.temporary_patches.mxfp4 whose gate returns `state["keep"]`."""
    state = {"keep": True, "raise": False}
    module = types.ModuleType(ZOO_MXFP4)

    def keep_mxfp4_experts_packed():
        if state["raise"]:
            raise RuntimeError("probe failed")
        return state["keep"]

    module.keep_mxfp4_experts_packed = keep_mxfp4_experts_packed
    monkeypatch.setitem(sys.modules, ZOO_MXFP4, module)
    return state


def _run_branch(load_in_16bit, quant_method, full_finetuning, quantizer = _Mxfp4Config):
    ns = {
        "inspect": inspect,
        "load_in_16bit": load_in_16bit,
        "quant_method": quant_method,
        "full_finetuning": full_finetuning,
        "quantizer": quantizer,
        "quantizer_kwargs": {},
        "_mxfp4_lora_keeps_experts_packed": _helper(),
    }
    node = _dequantize_branch()
    code = ast.Module(body = [node], type_ignores = [])
    ast.fix_missing_locations(code)
    exec(compile(code, str(VISION), "exec"), ns)
    return ns["quantizer_kwargs"].get("dequantize", False)


def test_helper_on_for_mxfp4_lora_when_zoo_keeps_packed(zoo):
    assert _helper()("mxfp4") is True
    assert _helper()("MXFP4", False) is True


def test_helper_off_for_full_finetuning_and_other_methods(zoo):
    assert _helper()("mxfp4", True) is False
    for method in ("fp8", "bitsandbytes", "compressed-tensors", None):
        assert _helper()(method) is False


def test_helper_off_when_zoo_declines_or_fails(zoo):
    zoo["keep"] = False
    assert _helper()("mxfp4") is False
    zoo["keep"] = True
    zoo["raise"] = True
    assert _helper()("mxfp4") is False


def test_helper_off_without_the_zoo_packed_path(monkeypatch):
    # An unsloth_zoo release without keep_mxfp4_experts_packed.
    monkeypatch.setitem(sys.modules, ZOO_MXFP4, types.ModuleType(ZOO_MXFP4))
    assert _helper()("mxfp4") is False


def test_native_mxfp4_lora_load_dequantizes_through_packed_path(zoo):
    # load_in_4bit = False without load_in_16bit: previously native Mxfp4GptOssExperts.
    assert _run_branch(False, "mxfp4", False) is True


def test_native_load_unchanged_without_packed_path(zoo):
    zoo["keep"] = False
    assert _run_branch(False, "mxfp4", False) is False
    zoo["keep"] = True
    assert _run_branch(False, "mxfp4", True) is False


def test_load_in_16bit_still_dequantizes(zoo):
    zoo["keep"] = False
    assert _run_branch(True, "mxfp4", False) is True
    assert _run_branch(True, "fp8", False) is True


def test_quantizer_without_dequantize_argument_untouched(zoo):
    assert _run_branch(False, "mxfp4", False, quantizer = _OtherConfig) is False
    assert _run_branch(True, "mxfp4", False, quantizer = _OtherConfig) is False
