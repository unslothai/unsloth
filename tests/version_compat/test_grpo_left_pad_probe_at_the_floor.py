# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The left-pad probe must survive every TRL in the declared window, including the floor.

`grpo_trainer__generate_and_score_completions` injects a "is this batch text only" probe ahead
of the old/reference logprob pass. The name carrying that answer moved twice inside the declared
`trl>=0.18.2,!=0.19.0` window, and at the bottom of it neither name exists:

    0.18.2, 0.19.1   neither `has_images` nor `images`: these have no vision path at all
    0.20.0 - 0.23.1  `has_images`, bound unconditionally before the anchor
    0.24.0 - 1.13.0  `images`, and no `has_images`

The shipped probe tried `has_images` and fell back to `images` with no third branch, so at the
floor the fallback raised NameError out of the except handler and took GRPO training down:

    unsloth_compiled_cache/UnslothGRPOTrainer.py:3696: in _generate_and_score_completions
        if not has_images:
    E   NameError: name 'has_images' is not defined
    During handling of the above exception, another exception occurred:
    unsloth_compiled_cache/UnslothGRPOTrainer.py:3702: in _generate_and_score_completions
        if images is None:
    E   NameError: name 'images' is not defined

The bindings above are read off the real sources; `test_the_window_still_binds_what_this_probe_expects`
re-derives them from upstream so a future TRL that moves the name again fails here rather than in a
user's training loop.
"""

from __future__ import annotations

import re

import pytest

from unsloth.models.rl_replacements import grpo_trainer__generate_and_score_completions


ANCHOR = 'batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size'

# The three binding shapes, each trimmed to the lines the rewrite anchors on. `_kind` is echoed
# back by the compiled function so a test cannot pass by silently running the wrong arm.
_FLOOR = f"""
    def _generate_and_score_completions(self, inputs):
        mode = "train"
        prompt_ids = inputs["prompt_ids"]
        completion_ids = inputs["completion_ids"]
        prompt_completion_ids = inputs["prompt_completion_ids"]
        logits_to_keep = 1
        {ANCHOR}
        return max_left_pad
"""

_HAS_IMAGES = f"""
    def _generate_and_score_completions(self, inputs):
        mode = "train"
        prompt_ids = inputs["prompt_ids"]
        completion_ids = inputs["completion_ids"]
        prompt_completion_ids = inputs["prompt_completion_ids"]
        logits_to_keep = 1
        has_images = inputs["has_images"]
        if has_images:
            images = inputs["images"]
        {ANCHOR}
        return max_left_pad
"""

_IMAGES = f"""
    def _generate_and_score_completions(self, inputs):
        mode = "train"
        prompt_ids = inputs["prompt_ids"]
        completion_ids = inputs["completion_ids"]
        prompt_completion_ids = inputs["prompt_completion_ids"]
        logits_to_keep = 1
        images = inputs["images"]
        {ANCHOR}
        return max_left_pad
"""


class _Args:
    per_device_train_batch_size = 1
    per_device_eval_batch_size = 1


class _Trainer:
    args = _Args()

    class processing_class:
        pad_token_id = 0

    class model:
        @staticmethod
        def for_training(use_gradient_checkpointing = True):
            return None


def _run(source: str, inputs: dict):
    """Rewrite `source`, exec it, and call it. Returns whatever `max_left_pad` ended up as."""
    rewritten = grpo_trainer__generate_and_score_completions(
        "_generate_and_score_completions",
        source,
    )
    # The rewriter also injects calls to helpers rl.py supplies at module scope. Only the
    # left-pad probe is under test here, so the rest are no-ops that pass their input through.
    namespace = {
        "torch": pytest.importorskip("torch"),
        "calculate_pad_tokens_in_prompt": lambda ids, keep, pad: _sentinel_tensor(),
        "_unsloth_reject_grpo_image_list": lambda *a, **k: None,
        "_unsloth_grpo_image_cell": lambda cell: cell if isinstance(cell, list) else [cell],
        "_unsloth_grpo_vision_inputs": lambda inputs: {},
        "sanitize_logprob": lambda x: x,
    }
    exec(
        compile(re.sub(r"^    ", "", rewritten, flags = re.MULTILINE), "<rewritten>", "exec"),
        namespace,
    )
    return namespace["_generate_and_score_completions"](_Trainer(), inputs)


def _sentinel_tensor():
    import torch
    return torch.tensor([7])


_BATCH = {
    "prompt_ids": None,
    "completion_ids": None,
    "prompt_completion_ids": None,
}


def test_the_floor_binds_neither_name_and_still_computes_the_pad() -> None:
    """0.18.2 and 0.19.1: no vision path at all, so the batch is text only and the probe runs."""
    assert _run(_FLOOR, dict(_BATCH)) == 7


def test_has_images_false_still_computes_the_pad() -> None:
    assert _run(_HAS_IMAGES, dict(_BATCH, has_images = False)) == 7


def test_has_images_true_skips_the_pad() -> None:
    assert _run(_HAS_IMAGES, dict(_BATCH, has_images = True, images = ["an image"])) is None


def test_images_none_still_computes_the_pad() -> None:
    assert _run(_IMAGES, dict(_BATCH, images = None)) == 7


def test_images_present_skips_the_pad() -> None:
    assert _run(_IMAGES, dict(_BATCH, images = ["an image"])) is None


def test_the_window_still_binds_what_this_probe_expects() -> None:
    """Re-derive the table in the docstring from upstream, so a fourth spelling fails here."""
    from tests.version_compat._fetch import fetch_text

    expected = {
        "v0.18.2": (False, False),
        "v0.20.0": (True, True),
        "v0.23.1": (True, True),
        "v0.24.0": (False, True),
    }
    for tag, (wants_has_images, wants_images) in expected.items():
        source = fetch_text("huggingface/trl", tag, "trl/trainer/grpo_trainer.py")
        assert source is not None, f"{tag}: grpo_trainer.py not found"
        method = re.search(
            r"\n    def _generate_and_score_completions\(.*?(?=\n    def |\Z)",
            source,
            re.S,
        )
        assert method, f"{tag}: no _generate_and_score_completions"
        before_anchor = method.group(0).split(ANCHOR)[0]
        assert bool(re.search(r"^\s+has_images\s*=", before_anchor, re.M)) == wants_has_images, tag
        assert bool(re.search(r"^\s+images\s*=", before_anchor, re.M)) == wants_images, tag
