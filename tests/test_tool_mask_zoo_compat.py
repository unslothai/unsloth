"""Compatibility checks for env/tool mask support with older unsloth_zoo."""

from __future__ import annotations

import ast
import os
import textwrap

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RL_SOURCE_PATH = os.path.join(REPO_ROOT, "unsloth", "models", "rl.py")
RL_REPLACEMENTS_SOURCE_PATH = os.path.join(REPO_ROOT, "unsloth", "models", "rl_replacements.py")


def _read(path: str) -> str:
    with open(path, "r") as fh:
        return fh.read()


def _load_local_align_completion_tool_mask():
    src = _read(RL_SOURCE_PATH)
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.If):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "align_completion_tool_mask":
                    function_src = ast.get_source_segment(src, item)
                    break
            else:
                continue
            break
    else:
        raise AssertionError("local align_completion_tool_mask fallback is missing")

    calls = []

    def align_logprobs_with_mask(
        logprob_tensor,
        completion_mask,
        pad_value = None,
    ):
        calls.append((logprob_tensor, completion_mask, pad_value))
        return torch.tensor(
            [[1, 0, 1], [0, 1, 1]],
            device = completion_mask.device,
            dtype = logprob_tensor.dtype,
        )

    namespace = {
        "torch": torch,
        "align_logprobs_with_mask": align_logprobs_with_mask,
    }
    exec(textwrap.dedent(function_src), namespace)
    return namespace["align_completion_tool_mask"], calls


def test_rl_uses_optional_zoo_tool_mask_helper():
    src = _read(RL_SOURCE_PATH)
    assert 'RL_REPLACEMENTS.get("align_completion_tool_mask")' in src
    assert 'RL_REPLACEMENTS["align_completion_tool_mask"]' not in src


def test_local_tool_mask_fallback_is_only_old_zoo_compat_shim():
    align_completion_tool_mask, calls = _load_local_align_completion_tool_mask()
    completion_mask = torch.tensor(
        [[1, 1, 0], [1, 1, 1]],
        dtype = torch.float32,
    )

    assert align_completion_tool_mask(None, completion_mask) is completion_mask
    assert calls == []

    same_shape_tool_mask = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype = torch.bool)
    with pytest.raises(RuntimeError, match = "Please upgrade unsloth_zoo"):
        align_completion_tool_mask(same_shape_tool_mask, completion_mask)


def test_grpo_accumulated_loss_omits_none_tool_mask_for_old_zoo():
    src = _read(RL_REPLACEMENTS_SOURCE_PATH)
    assert "_grpo_accumulated_loss_kwargs = {}" in src
    assert (
        'if tool_mask is not None:\n                _grpo_accumulated_loss_kwargs["tool_mask"] = tool_mask'
        in src
    )
    assert src.count("**_grpo_accumulated_loss_kwargs") == 2

    accelerated_loss_start = src.find('if hasattr(self.args, "loss_type"):')
    assert accelerated_loss_start != -1
    accelerated_loss_body = src[
        accelerated_loss_start : src.find('if "train" in self._metrics:', accelerated_loss_start)
    ]
    assert "tool_mask = tool_mask" not in accelerated_loss_body


def test_rollout_output_patch_requires_real_tool_mask_symbol():
    src = _read(RL_REPLACEMENTS_SOURCE_PATH)
    assert 're.search(r"\\btool_mask\\b", function)' in src
    assert 'output["tool_mask"]' in src
