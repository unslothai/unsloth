# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A singular `image` column holding a list takes the plural path: TRL wraps the cell
again, so two images reach the processor as [[[img, img]]]. unslothai/unsloth#3605."""

import textwrap

import pytest

from unsloth.models.rl_replacements import (
    _unsloth_grpo_image_cell,
    _unsloth_reject_grpo_image_list,
    grpo_trainer__generate_and_score_completions,
)


class _Img:
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
    assert _unsloth_grpo_image_cell([a, b]) != [[a, b]]


def test_the_reporter_shape_is_flattened_not_nested():
    a, b = _Img("a"), _Img("b")
    inputs = [{"image": [a, b]}, {"image": [a, b]}]
    images = [_unsloth_grpo_image_cell(example.get("image")) for example in inputs]
    assert images == [[a, b], [a, b]]
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
        assert '[[example.get("image")] if example.get("image") is not None' not in patched
        assert 'kwargs = {"images": [[img] for img in images]}' not in patched


def test_trl_0_22_placeholder_count_follows_the_cell():
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


def test_guard_is_injected_when_the_legacy_reference_calls_drift():
    """Rewriting the cell is half the job. A legacy TRL also has to carry the image counts
    into the reference logprob calls; without them the images are sliced by sample index and
    all but the first are dropped, which the model reports as a token/feature mismatch."""
    source = (
        "    def _generate_and_score_completions(self, inputs):\n"
        '        has_images = "image" in inputs[0]\n'
        "        if has_images:\n"
        '            images = [example.get("image") for example in inputs]\n'
        '            kwargs = {"images": [[img] for img in images]}\n'
        "        ref = self._get_per_token_logps_and_entropies(\n"
        "            self.model,\n"
        '            pixel_values=prompt_inputs.get("pixel_values"), image_grid_thw=None,\n'
        "        )\n"
        "        return inputs\n"
    )
    patched = grpo_trainer__generate_and_score_completions(
        "_generate_and_score_completions", source
    )
    assert "_unsloth_grpo_image_cell(img)" in patched
    assert "_unsloth_reject_grpo_image_list(inputs)" in patched
    lines = patched.splitlines()
    assert lines[1].strip() == "_unsloth_reject_grpo_image_list(inputs)"


def test_guard_stays_off_a_trl_whose_reference_calls_do_take_the_counts():
    """The installed TRL is fully plumbed, so a multi image row works and must not be refused."""
    trl_grpo = pytest.importorskip("trl.trainer.grpo_trainer")
    with open(trl_grpo.__file__, "r", encoding = "utf-8") as fh:
        module_source = fh.read()
    start = module_source.find("    def _generate_and_score_completions(")
    assert start != -1, "TRL renamed _generate_and_score_completions"
    source = module_source[start:]
    patched = grpo_trainer__generate_and_score_completions(
        "_generate_and_score_completions", source
    )
    assert "_unsloth_reject_grpo_image_list(inputs)" not in patched
    if 'pixel_values=prompt_inputs.get("pixel_values")' in source:
        # A legacy TRL: every reference call site must have taken the counts.
        assert 'pixel_values=prompt_inputs.get("pixel_values")' not in patched
        assert "**_unsloth_legacy_vision," in patched
