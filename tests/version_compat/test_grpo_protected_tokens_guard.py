# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""The injected GRPO prompt-trim block must not require TRL's vision token ids.

`grpo_trainer__generate_and_score_completions` in `unsloth/models/rl_replacements.py`
substitutes a `if self.max_prompt_length is not None:` block into TRL's
`_generate_and_score_completions`. That block builds a `protected` list of vision token
ids. `GRPOTrainer` only sets `image_token_id`, `vision_start_token_id` and
`vision_end_token_id` on part of the TRL range, but the substitution fires on any TRL whose
source the regex matches, which is wider. Measured across the declared window:

    trl 0.18.2   injected, attributes ABSENT  -> AttributeError before this fix
    trl 0.19.1   injected, attributes ABSENT  -> AttributeError before this fix
    trl 0.20.0   injected, attributes present
    trl 0.21.0   injected, attributes present
    trl 0.22.0   injected, attributes present
    trl 0.22.2   injected, attributes present
    trl 0.23.0   injected, attributes present
    trl 0.23.1   injected, attributes present
    trl 0.24.0   NOT injected (the regex no longer matches), attributes absent
    trl 1.13.0   NOT injected, attributes absent

So the break is the bottom of the declared window, not the ceiling. `getattr(..., None)`
makes the emitted block work on all ten, because the next emitted line already drops None.

This test reads the substitution's OUTPUT and executes the `protected` lines it emits
against an object with none of the three attributes, so it fails on any platform and any
transformers, and it does not need the affected TRL installed to prove the point.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest


os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

_SPOOF_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SPOOF_DIR))
import _zoo_aggressive_cuda_spoof as _spoof  # noqa: E402

_spoof.apply()


# The shape trl 0.18.2 and 0.19.1 ship: the block the regex looks for, with two top-level
# statements and eight spaces of indent, and no vision token ids anywhere. Held here as a
# fixture rather than read from the installed TRL, so the test covers the affected versions
# on a runner that has a different one.
_TRL_0_18_BLOCK = '''
    def _generate_and_score_completions(self, inputs):
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
        if self.use_vllm:
            pass
        return inputs
'''


def _injected(source: str) -> str:
    from unsloth.models.rl_replacements import grpo_trainer__generate_and_score_completions

    return grpo_trainer__generate_and_score_completions(
        "_generate_and_score_completions", source
    )


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
    return "\n".join(line[len(indent):] for line in match.group(0).split("\n"))


def test_the_substitution_still_fires_on_the_affected_shape() -> None:
    """Vacuity guard: everything below is about the emitted block, so a regex that stopped
    matching would make all of it pass while testing nothing."""
    emitted = _injected(_TRL_0_18_BLOCK)
    assert "protected" in emitted, (
        "the prompt-trim substitution did not fire on the trl 0.18.2 block shape, so the "
        "rest of this file proves nothing about it"
    )


def test_the_emitted_block_runs_without_the_vision_token_ids() -> None:
    """The regression: attribute access here raised AttributeError mid-training."""
    lines = _protected_lines(_injected(_TRL_0_18_BLOCK))
    scope = {"self": _TrainerWithoutVisionIds()}
    exec(lines, scope)  # noqa: S102 - executing our own generated source is the point
    assert scope["protected"] == [], (
        f"with none of the three ids set, nothing should be protected, got "
        f"{scope['protected']!r}"
    )


def test_the_ids_are_still_protected_when_trl_does_set_them() -> None:
    """The fix must not quietly stop protecting vision tokens where they exist."""

    class _TrainerWithVisionIds:
        max_prompt_length = 8
        image_token_id = 151655
        vision_start_token_id = 151652
        vision_end_token_id = 151653

    lines = _protected_lines(_injected(_TRL_0_18_BLOCK))
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

    lines = _protected_lines(_injected(_TRL_0_18_BLOCK))
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
