# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""The injected GRPO prompt-trim block must only replace a block that can host it.

`grpo_trainer__generate_and_score_completions` in `unsloth/models/rl_replacements.py`
substitutes a TRL 0.20.0+ block -- it calls `truncate_with_protected_tokens` and reads
`self.pad_token`, `self.image_token` and three vision token ids -- into TRL's
`_generate_and_score_completions`. Driving the real substitution against each release's
actual source, the selecting regex matches 0.18.2 through 0.23.1 and stops matching at
0.24.0, while all four symbols exist only from 0.20.0, so on 0.18.2 and 0.19.1 (both inside
the declared window `trl>=0.18.2,!=0.19.0,<=0.24.0`) it replaced working native slicing with
names that do not resolve. TRL's own call to the helper inside the block being replaced is
the discriminator, so the substitution is gated on it; `getattr(..., None)` for the three ids
covers a fork or subclass that does not set one.

Reads the substitution's OUTPUT and executes the statements it emits, so it needs neither the
affected TRL nor a GPU.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

import pytest


os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# daily-fresh-fetch collects this directory with only pytest installed.
if importlib.util.find_spec("torch") is None:
    pytest.skip("torch not installed", allow_module_level = True)

_SPOOF_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SPOOF_DIR))
import _zoo_aggressive_cuda_spoof as _spoof  # noqa: E402

_spoof.apply()


# Fixtures, not the installed TRL, so both groups are covered on a runner that has neither.
# Both satisfy the selecting regex and the two-statement rule; they differ only in the helper.
_TRL_0_18_BLOCK = """
    def _generate_and_score_completions(self, inputs):
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
        if self.use_vllm:
            pass
        return inputs
"""

_TRL_0_22_BLOCK = """
    def _generate_and_score_completions(self, inputs):
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        if self.max_prompt_length is not None:
            protected = [self.image_token_id, self.vision_start_token_id, self.vision_end_token_id]
            protected = [token for token in protected if token is not None]
            prompt_ids, prompt_mask = truncate_with_protected_tokens(
                prompt_ids, prompt_mask, self.max_prompt_length, protected
            )
            prompts_text = [re.sub(rf"^({re.escape(self.pad_token)})+", "", text) for text in prompts_text]
        if self.use_vllm:
            pass
        return inputs
"""


def _injected(source: str) -> str:
    from unsloth.models.rl_replacements import grpo_trainer__generate_and_score_completions
    return grpo_trainer__generate_and_score_completions("_generate_and_score_completions", source)


class _TrainerWithoutVisionIds:
    """A stand-in for GRPOTrainer on a TRL that never sets the three ids."""

    max_prompt_length = 8


def _protected_lines(emitted: str) -> str:
    """The emitted `protected = [...]` statements, dedented so they can be executed."""
    match = re.search(
        r"^(\s*)protected = \[.*?^\1protected = \[token for token in protected if token is not None\]",
        emitted,
        re.DOTALL | re.MULTILINE,
    )
    assert match, (
        "the substitution no longer emits the two `protected` statements this test drives. "
        "If the block was rewritten, retarget the test; do not delete it."
    )
    indent = match.group(1)
    return "\n".join(line[len(indent) :] for line in match.group(0).split("\n"))


def test_the_pre_0_20_block_is_left_alone() -> None:
    """The regression: 0.18.2 and 0.19.1 have no truncate_with_protected_tokens, no
    self.pad_token and no self.image_token, so the injected block cannot run there. TRL's
    own slicing has to survive untouched."""
    source = _TRL_0_18_BLOCK
    emitted = _injected(source)
    assert "truncate_with_protected_tokens" not in emitted, (
        "the substitution fired on a block with no truncate_with_protected_tokens in it, so "
        "the emitted call has no definition to reach and training raises NameError"
    )
    assert (
        "prompt_ids = prompt_ids[:, -self.max_prompt_length :]" in emitted
    ), "TRL's own prompt slicing was removed on a release that cannot host the replacement"
    for absent in ("self.pad_token", "self.image_token", "protected"):
        assert absent not in emitted, f"{absent!r} was injected into a release that lacks it"


def test_the_substitution_still_fires_on_the_0_20_shape() -> None:
    """Vacuity guard: everything below is about the emitted block, so a guard that stopped
    matching the supported shape would make all of it pass while testing nothing."""
    emitted = _injected(_TRL_0_22_BLOCK)
    assert 'getattr(self, "image_token_id", None)' in emitted, (
        "the prompt-trim substitution did not fire on the trl 0.20.0+ block shape, so the "
        "rest of this file proves nothing about it"
    )


def test_the_emitted_block_runs_without_the_vision_token_ids() -> None:
    """Belt and braces: a trainer that sets none of the three must not raise."""
    lines = _protected_lines(_injected(_TRL_0_22_BLOCK))
    scope = {"self": _TrainerWithoutVisionIds()}
    exec(lines, scope)  # noqa: S102 - executing our own generated source is the point
    assert scope["protected"] == [], (
        f"with none of the three ids set, nothing should be protected, got "
        f"{scope['protected']!r}"
    )


def test_the_ids_are_still_protected_when_trl_does_set_them() -> None:
    """The guard must not quietly stop protecting vision tokens where they exist."""

    class _TrainerWithVisionIds:
        max_prompt_length = 8
        image_token_id = 151655
        vision_start_token_id = 151652
        vision_end_token_id = 151653

    lines = _protected_lines(_injected(_TRL_0_22_BLOCK))
    scope = {"self": _TrainerWithVisionIds()}
    exec(lines, scope)  # noqa: S102
    assert scope["protected"] == [151655, 151652, 151653], (
        f"the vision token ids must still reach truncate_with_protected_tokens, got "
        f"{scope['protected']!r}"
    )


def test_one_missing_id_does_not_discard_the_others() -> None:
    """Partial support is the realistic case, and the None filter has to survive it."""

    class _TrainerWithOneId:
        max_prompt_length = 8
        image_token_id = 151655

    lines = _protected_lines(_injected(_TRL_0_22_BLOCK))
    scope = {"self": _TrainerWithOneId()}
    exec(lines, scope)  # noqa: S102
    assert scope["protected"] == [151655], f"got {scope['protected']!r}"


def test_bare_attribute_access_would_still_have_failed() -> None:
    """Negative control: the three tests above pass trivially if the emitted lines stopped
    reading the ids at all, so prove the pre-fix spelling really does raise here."""
    pre_fix = (
        "protected = [self.image_token_id, self.vision_start_token_id, self.vision_end_token_id]\n"
        "protected = [token for token in protected if token is not None]\n"
    )
    with pytest.raises(AttributeError, match = "image_token_id"):
        exec(pre_fix, {"self": _TrainerWithoutVisionIds()})  # noqa: S102
