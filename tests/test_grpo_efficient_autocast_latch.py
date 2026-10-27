# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""With no scoring forward (vLLM, beta = 0) the efficient call must still latch this trainer's precision."""

import ast
import os
import textwrap
import types
from pathlib import Path

import torch

REPL_SRC = (
    Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"
).read_text(encoding = "utf-8")


def _function(name, parent = None):
    tree = ast.parse(REPL_SRC)
    scope = tree.body
    if parent is not None:
        scope = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == parent
        ).body
    node = next(n for n in scope if isinstance(n, ast.FunctionDef) and n.name == name)
    return textwrap.dedent(ast.get_source_segment(REPL_SRC, node))


def _namespace():
    ns = {"torch": torch, "os": os}
    exec(_function("_unsloth_grpo_autocast_forced_float32"), ns)
    exec(_function("_unsloth_grpo_autocast"), ns)
    exec(_function("_unsloth_grpo_autocast_unscored"), ns)
    exec(_function("_unsloth_grpo_support_mismatch"), ns)
    exec(_function("_unsloth_grpo_refuse_unscored_support"), ns)
    exec(
        _function(
            "_get_per_token_logps_and_entropies", "grpo_trainer__get_per_token_logps_and_entropies"
        ),
        ns,
    )
    return ns


def test_efficient_call_latches_this_trainers_precision(monkeypatch):
    ns = _namespace()
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "fp16")
    trainer = types.SimpleNamespace(
        args = types.SimpleNamespace(mixed_precision = "bf16", bf16 = True, fp16 = False),
        model = None,
        use_vllm = True,
    )
    out = ns["_get_per_token_logps_and_entropies"](
        trainer, None, None, None, 4, compute_efficient = True
    )
    assert out == (None, None)
    assert trainer._autocast_enabled is True
    assert trainer._autocast_dtype is torch.bfloat16


def test_efficient_call_keeps_an_existing_latch(monkeypatch):
    ns = _namespace()
    trainer = types.SimpleNamespace(
        args = None,
        model = None,
        use_vllm = True,
        _autocast_enabled = False,
        _autocast_dtype = torch.float16,
    )
    ns["_get_per_token_logps_and_entropies"](trainer, None, None, None, 4, compute_efficient = True)
    assert trainer._autocast_enabled is False
    assert trainer._autocast_dtype is torch.float16


def test_legacy_logprob_replacement_latches_too(monkeypatch):
    ns = {"torch": torch, "os": os}
    exec(_function("_unsloth_grpo_autocast_forced_float32"), ns)
    exec(_function("_unsloth_grpo_autocast"), ns)
    exec(_function("_unsloth_grpo_autocast_unscored"), ns)
    exec(_function("_get_per_token_logps", "grpo_trainer__get_per_token_logps"), ns)
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "fp16")
    trainer = types.SimpleNamespace(
        args = types.SimpleNamespace(mixed_precision = "bf16", bf16 = True, fp16 = False),
        model = None,
        use_vllm = True,
    )
    assert ns["_get_per_token_logps"](trainer, None, None, None, 4) is None
    assert trainer._autocast_dtype is torch.bfloat16


def _forced_float32_trainer(use_vllm):
    # What rl.py builds for a forced float32 model (Gemma3 on a T4, or dtype = torch.float16).
    return types.SimpleNamespace(
        args = types.SimpleNamespace(mixed_precision = "no", bf16 = False, fp16 = False),
        model = types.SimpleNamespace(_unsloth_forced_float32 = True),
        use_vllm = use_vllm,
    )


def test_efficient_call_without_vllm_records_no_autocast_for_forced_float32(monkeypatch):
    # Without vLLM and with nothing scored, nothing latched before the efficient call did, and zoo then
    # trained a forced float32 model with no loss autocast. A latch here turns on float16 autocast.
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "no")
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    ns = _namespace()
    trainer = _forced_float32_trainer(use_vllm = False)
    assert ns["_get_per_token_logps_and_entropies"](
        trainer, None, None, None, 4, compute_efficient = True
    ) == (None, None)
    assert trainer._autocast_enabled is False

    # Recorded on the trainer: a later model or trainer resetting the process-wide flags changes nothing.
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "fp16")
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "0")
    ns["_get_per_token_logps_and_entropies"](trainer, None, None, None, 4, compute_efficient = True)
    assert trainer._autocast_enabled is False

    legacy = {"torch": torch, "os": os}
    exec(_function("_unsloth_grpo_autocast_forced_float32"), legacy)
    exec(_function("_unsloth_grpo_autocast"), legacy)
    exec(_function("_unsloth_grpo_autocast_unscored"), legacy)
    exec(_function("_get_per_token_logps", "grpo_trainer__get_per_token_logps"), legacy)
    trainer = _forced_float32_trainer(use_vllm = False)
    assert legacy["_get_per_token_logps"](trainer, None, None, None, 4) is None
    assert trainer._autocast_enabled is False


def test_efficient_call_with_vllm_latches_forced_float32_as_the_old_forward_did(monkeypatch):
    # Under vLLM the old-policy forward always ran before the TRL gate came back and latched float16 autocast.
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    ns = _namespace()
    trainer = _forced_float32_trainer(use_vllm = True)
    ns["_get_per_token_logps_and_entropies"](trainer, None, None, None, 4, compute_efficient = True)
    assert trainer._autocast_enabled is True
    assert trainer._autocast_dtype is torch.float16
    assert trainer._autocast_force_float32 is True


def test_efficient_call_without_vllm_still_latches_an_ordinary_trainer(monkeypatch):
    # Only forced float32 models stay unlatched: an ordinary trainer keeps its own precision even when the
    # process-wide env (another trainer, or an explicit setting) says otherwise.
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "fp16")
    monkeypatch.delenv("UNSLOTH_FORCE_FLOAT32", raising = False)
    ns = _namespace()
    trainer = types.SimpleNamespace(
        args = types.SimpleNamespace(mixed_precision = "bf16", bf16 = True, fp16 = False),
        model = types.SimpleNamespace(_unsloth_forced_float32 = False),
        use_vllm = False,
    )
    ns["_get_per_token_logps_and_entropies"](trainer, None, None, None, 4, compute_efficient = True)
    assert trainer._autocast_enabled is True
    assert trainer._autocast_dtype is torch.bfloat16

    legacy = {"torch": torch, "os": os}
    exec(_function("_unsloth_grpo_autocast_forced_float32"), legacy)
    exec(_function("_unsloth_grpo_autocast"), legacy)
    exec(_function("_unsloth_grpo_autocast_unscored"), legacy)
    exec(_function("_get_per_token_logps", "grpo_trainer__get_per_token_logps"), legacy)
    trainer = types.SimpleNamespace(
        args = types.SimpleNamespace(mixed_precision = "bf16", bf16 = True, fp16 = False),
        model = None,
        use_vllm = False,
    )
    assert legacy["_get_per_token_logps"](trainer, None, None, None, 4) is None
    assert trainer._autocast_dtype is torch.bfloat16


def test_efficient_call_without_vllm_keeps_bf16_for_a_forced_family_full_finetune(monkeypatch):
    # rl.py lets forced-family models full finetune in bf16 (args.mixed_precision = "bf16"): keep that latch.
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    ns = _namespace()
    trainer = types.SimpleNamespace(
        args = types.SimpleNamespace(mixed_precision = "bf16", bf16 = True, fp16 = False),
        model = types.SimpleNamespace(_unsloth_forced_float32 = True),
        use_vllm = False,
    )
    ns["_get_per_token_logps_and_entropies"](trainer, None, None, None, 4, compute_efficient = True)
    assert trainer._autocast_enabled is True
    assert trainer._autocast_dtype is torch.bfloat16
