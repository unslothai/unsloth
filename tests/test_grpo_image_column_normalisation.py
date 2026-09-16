# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A singular `image` column holding a list of images takes the plural path.

Regression cover for unslothai/unsloth#3605. TRL wraps whatever `example["image"]` holds
in a list of its own, so a cell carrying two images reaches the processor as
`[[[img, img]]]`, and both `make_nested_list_of_images` and the processor reject that with
"Invalid input type. Must be a single image, a list of images, or a list of batches of
images.".
"""

import textwrap

import pytest

from unsloth.models.rl_replacements import (
    _unsloth_grpo_image_cell,
    _unsloth_reject_grpo_image_list,
    grpo_trainer__generate_and_score_completions,
)


class _Img:
    """Stands in for a PIL image; only identity matters here."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<{self.name}>"


def test_cell_helper_keeps_a_list_and_wraps_a_single_image():
    a, b = _Img("a"), _Img("b")
    assert _unsloth_grpo_image_cell(None) is None
    assert _unsloth_grpo_image_cell(a) == [a]
    assert _unsloth_grpo_image_cell([a, b]) == [a, b]
    assert _unsloth_grpo_image_cell((a, b)) == [a, b]
    # A list cell must not gain a level of nesting.
    assert _unsloth_grpo_image_cell([a, b]) != [[a, b]]


def test_the_reporter_shape_is_flattened_not_nested():
    """The batch TRL builds must be a list of per example image lists."""
    a, b = _Img("a"), _Img("b")
    inputs = [{"image": [a, b]}, {"image": [a, b]}]
    # What TRL >= 1.0.0 now does, after the rewrite.
    images = [_unsloth_grpo_image_cell(example.get("image")) for example in inputs]
    assert images == [[a, b], [a, b]]
    # What it did before: one list too many, which is the ValueError in the issue.
    before = [[example.get("image")] for example in inputs]
    assert before == [[[a, b]], [[a, b]]]


def test_single_image_cells_are_unchanged():
    a = _Img("a")
    inputs = [{"image": a}, {"image": a}]
    assert [_unsloth_grpo_image_cell(x.get("image")) for x in inputs] == [[a], [a]]


def test_patcher_rewrites_the_installed_trl_extraction():
    """Read TRL's file, not the class: importing unsloth already replaced the method."""
    trl_grpo = pytest.importorskip("trl.trainer.grpo_trainer")
    with open(trl_grpo.__file__, "r", encoding = "utf-8") as fh:
        module_source = fh.read()
    start = module_source.find("    def _generate_and_score_completions(")
    assert start != -1, "TRL renamed _generate_and_score_completions"
    source = module_source[start:]
    patched = grpo_trainer__generate_and_score_completions(
        "_generate_and_score_completions", source
    )
    assert "_unsloth_grpo_image_cell" in patched or "_unsloth_reject_grpo_image_list" in patched
    if "_unsloth_grpo_image_cell" in patched:
        # The unpatched spellings must be gone, or TRL would still add the extra level.
        assert '[[example.get("image")] if example.get("image") is not None' not in patched
        assert 'kwargs = {"images": [[img] for img in images]}' not in patched


def test_trl_0_22_placeholder_count_follows_the_cell():
    """0.22.x inserted exactly one image placeholder, whatever the cell held."""
    source = (
        '        has_images = "image" in inputs[0]\n'
        "        if has_images:\n"
        '            images = [example.get("image") for example in inputs]\n'
        '            kwargs = {"images": [[img] for img in images]}\n'
        "            for prompt in prompts:\n"
        "                if isinstance(prompt, list):  # i.e., when using conversational data\n"
        "                    prepare_multimodal_messages(prompt, num_images=1)\n"
    )
    patched = grpo_trainer__generate_and_score_completions(
        "_generate_and_score_completions", source
    )
    assert "_unsloth_grpo_image_cell(img)" in patched
    assert "num_images=1)" not in patched
    assert "len(_unsloth_cell)" in patched

    # Run the rewritten block and check both the nesting and the placeholder count.
    a, b = _Img("a"), _Img("b")
    calls = []

    def prepare_multimodal_messages(prompt, num_images):
        calls.append(num_images)

    namespace = {
        "inputs": [{"image": [a, b]}],
        "prompts": [[{"role": "user", "content": "x"}]],
        "prepare_multimodal_messages": prepare_multimodal_messages,
        "_unsloth_grpo_image_cell": _unsloth_grpo_image_cell,
    }
    exec(textwrap.dedent(patched), namespace)
    assert namespace["kwargs"]["images"] == [[a, b]]
    assert calls == [2]


def test_guard_names_the_column_and_the_issue():
    a, b = _Img("a"), _Img("b")
    # One image per row, or a single image object, is fine.
    _unsloth_reject_grpo_image_list([{"image": a}])
    _unsloth_reject_grpo_image_list([{"image": [a]}])
    _unsloth_reject_grpo_image_list([{"prompt": "x"}])
    _unsloth_reject_grpo_image_list([])
    with pytest.raises(ValueError) as excinfo:
        _unsloth_reject_grpo_image_list([{"image": [a, b]}])
    message = str(excinfo.value)
    assert "`images`" in message
    assert "3605" in message


def test_guard_is_injected_when_no_anchor_matches():
    source = "    def _generate_and_score_completions(self, inputs):\n        return inputs\n"
    patched = grpo_trainer__generate_and_score_completions(
        "_generate_and_score_completions", source
    )
    assert "_unsloth_reject_grpo_image_list(inputs)" in patched
    lines = patched.splitlines()
    assert lines[1].strip() == "_unsloth_reject_grpo_image_list(inputs)"
