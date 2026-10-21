# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""GKD on TRL < 1.7 masks generated prompt labels (#12007)."""

from __future__ import annotations

import os
import textwrap

import pytest

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
import unsloth.models.rl_replacements as rl

LEGACY = textwrap.dedent(
    """
    def generate_on_policy_outputs(model, inputs, generation_config, pad_token_id=None):
        generated_tokens = model.generate(input_ids=inputs["prompts"]).sequences
        new_attention_mask = torch.ones_like(generated_tokens)
        new_labels = generated_tokens.clone()
        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
        return generated_tokens, new_attention_mask, new_labels
    """
)
MASK = 'new_labels[:, : inputs["prompts"].shape[1]] = -100'


def test_legacy_prompt_labels_are_masked_once():
    out = rl.gkd_trainer_mask_prompt("generate_on_policy_outputs", LEGACY)
    compile(out, "gkd", "exec")
    assert out.count(MASK) == 1
    assert out.index("new_labels = generated_tokens.clone()") < out.index(MASK)
    assert rl.gkd_trainer_mask_prompt("generate_on_policy_outputs", out) == out


def test_upstream_mask_and_other_functions_untouched():
    upstream = LEGACY.replace(
        "    return generated_tokens",
        "    new_labels[:, :prompt_length] = -100\n    return generated_tokens",
    )
    assert rl.gkd_trainer_mask_prompt("generate_on_policy_outputs", upstream) == upstream
    assert rl.gkd_trainer_mask_prompt("compute_loss", LEGACY) == LEGACY


def test_mask_matches_generated_layout():
    torch = pytest.importorskip("torch")
    ns = {"torch": torch}
    exec(rl.gkd_trainer_mask_prompt("generate_on_policy_outputs", LEGACY), ns)

    class Model:
        def generate(self, input_ids):
            completion = torch.full((input_ids.shape[0], 3), 7)
            return type("O", (), {"sequences": torch.cat([input_ids, completion], 1)})()

    prompts = torch.tensor([[0, 0, 5], [3, 4, 5]])
    _, _, labels = ns["generate_on_policy_outputs"](
        Model(), {"prompts": prompts}, None, pad_token_id = 0
    )
    assert (labels[:, :3] == -100).all()
    assert (labels[:, 3:] == 7).all()
