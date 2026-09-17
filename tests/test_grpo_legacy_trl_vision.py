# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TRL 0.22.x-0.23.x carries the whole multimodal batch through the GRPO step.

Those versions have neither `forward_kwargs` nor `num_images`: `_generate_and_score_completions`
hands the no-grad old/reference logprob pass four hand named processor tensors and saves the same
four into the training batch. Once a singular `image` cell may hold several images, the counts
decide how every vision tensor is sliced, and a processor that emits `spatial_shapes` or
`num_tiles` loses them entirely. unslothai/unsloth#6960.

The sources below are verbatim from trl 0.22.2 / 0.23.1 `grpo_trainer.py`, trimmed to the lines
the rewrite anchors on, so the patcher is exercised against the text it will actually meet.
"""

import textwrap

import pytest


# trl 0.22.2 grpo_trainer.py lines 1057-1084, 1321-1371 and 1471-1485, with generation replaced by
# a fixed completion. 0.23.0 and 0.23.1 are identical over this range.
_LEGACY_SOURCE = """
    def _generate_and_score_completions(self, inputs):
        mode = "train"
        prompts = [x["prompt"] for x in inputs]
        kwargs = {}
        has_images = "image" in inputs[0]
        if has_images:
            images = [example.get("image") for example in inputs]
            kwargs = {"images": [[img] for img in images]}
            for prompt in prompts:
                if isinstance(prompt, list):  # i.e., when using conversational data
                    prepare_multimodal_messages(prompt, num_images=1)

        prompt_inputs = self.processing_class(text=prompts, **kwargs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        completion_ids = self._completion_ids
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, torch.ones_like(completion_ids)], dim=1)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                self.model,
                prompt_completion_ids,
                attention_mask,
                logits_to_keep,
                batch_size,
                pixel_values=prompt_inputs.get("pixel_values"),
                image_grid_thw=prompt_inputs.get("image_grid_thw"),
                pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                image_sizes=prompt_inputs.get("image_sizes"),
            )
            if self.beta != 0.0:
                ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size=batch_size,
                    pixel_values=prompt_inputs.get("pixel_values"),
                    image_grid_thw=prompt_inputs.get("image_grid_thw"),
                    pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                    image_sizes=prompt_inputs.get("image_sizes"),
                )

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
        }
        if "pixel_values" in prompt_inputs:
            output["pixel_values"] = prompt_inputs["pixel_values"]
        if "image_grid_thw" in prompt_inputs:
            output["image_grid_thw"] = prompt_inputs["image_grid_thw"]
        if "pixel_attention_mask" in prompt_inputs:
            output["pixel_attention_mask"] = prompt_inputs["pixel_attention_mask"]
        if "image_sizes" in prompt_inputs:
            output["image_sizes"] = prompt_inputs["image_sizes"]
        return output
"""


class _Model:
    def for_training(self, use_gradient_checkpointing = True):
        pass


class _Processor:
    """Callable like a processor, and carrying the chat_template the rewrite reads."""

    chat_template = ""
    pad_token_id = 0

    def __init__(self, processor_output):
        self._processor_output = processor_output
        self.seen_image_kwarg = None

    def __call__(
        self,
        text = None,
        **kwargs,
    ):
        self.seen_image_kwarg = kwargs.get("images", None)
        return self._processor_output


class _Trainer:
    """Only what the patched legacy body touches."""

    beta = 0.1
    use_vllm = False

    def __init__(self, processor_output, completion_ids):
        import types

        self.model = _Model()
        self.ref_model = _Model()
        self.args = types.SimpleNamespace(
            per_device_train_batch_size = 1, per_device_eval_batch_size = 1
        )
        self.processing_class = _Processor(processor_output)
        self._completion_ids = completion_ids
        self.calls = []

    @property
    def seen_image_kwarg(self):
        return self.processing_class.seen_image_kwarg

    def _get_per_token_logps_and_entropies(self, model, *args, **kwargs):
        self.calls.append(kwargs)
        return None, None


def _run_legacy(processor_output, inputs, completion_ids):
    """Patch the verbatim legacy source, run it, and return (trainer, output)."""
    import torch

    from unsloth.models.rl_replacements import (
        _unsloth_grpo_image_cell,
        _unsloth_grpo_vision_inputs,
        grpo_trainer__generate_and_score_completions,
    )

    patched = grpo_trainer__generate_and_score_completions(
        "_generate_and_score_completions", _LEGACY_SOURCE
    )
    namespace = {
        "torch": torch,
        "_unsloth_grpo_vision_inputs": _unsloth_grpo_vision_inputs,
        "_unsloth_grpo_image_cell": _unsloth_grpo_image_cell,
        "prepare_multimodal_messages": lambda prompt, num_images = 1: prompt.append(num_images),
        # Injected by the same rewrite for the text-only branch; TRL never sees it.
        "calculate_pad_tokens_in_prompt": lambda ids, keep, pad: torch.zeros(
            ids.shape[0], dtype = torch.long
        ),
    }
    exec(compile(textwrap.dedent(patched), "<legacy-trl>", "exec"), namespace)
    trainer = _Trainer(processor_output, completion_ids)
    output = namespace["_generate_and_score_completions"](trainer, inputs)
    return trainer, output


def _grid_batch():
    """Two samples, the first holding two images. Qwen2-VL style, four patch rows per image."""
    import torch

    processor_output = {
        "input_ids": torch.zeros(2, 5, dtype = torch.long),
        "attention_mask": torch.ones(2, 5, dtype = torch.long),
        "pixel_values": torch.arange(12).reshape(12, 1).float(),
        "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2], [1, 2, 2]]),
    }
    inputs = [
        {"prompt": "a", "image": ["one", "two"]},
        {"prompt": "b", "image": "three"},
    ]
    return processor_output, inputs, torch.zeros(2, 3, dtype = torch.long)


def test_the_legacy_no_grad_calls_are_told_how_many_images_each_sample_has():
    """Without the counts the shared chunker slices an image indexed grid by sample index, so
    the old and reference logprobs of a two image row come from another sample's pixels."""
    trainer, _ = _run_legacy(*_grid_batch())
    assert len(trainer.calls) == 2, "the old-policy and the reference call"
    for call in trainer.calls:
        assert call.get("num_images") == [2, 1], call.get("num_images")
        assert call["image_grid_thw"].shape[0] == 3
        assert call["pixel_values"].shape[0] == 12


def test_the_legacy_training_batch_carries_the_counts_too():
    _, output = _run_legacy(*_grid_batch())
    assert output["num_images"] == [2, 1]


def test_a_legacy_row_with_no_image_of_its_own_counts_zero():
    import torch

    processor_output, inputs, completion_ids = _grid_batch()
    inputs[1]["image"] = None
    _, output = _run_legacy(processor_output, inputs, completion_ids)
    assert output["num_images"] == [2, 0]


def test_a_singular_cell_is_counted_the_way_the_processor_was_given_it():
    processor_output, inputs, completion_ids = _grid_batch()
    inputs[0]["image"] = "one"
    trainer, output = _run_legacy(processor_output, inputs, completion_ids)
    assert output["num_images"] == [1, 1]
    assert trainer.seen_image_kwarg == [["one"], ["three"]]


def _lfm2vl_batch():
    """LFM2-VL style: tile indexed pixels plus spatial_shapes, and no image_grid_thw."""
    import torch

    processor_output = {
        "input_ids": torch.zeros(2, 5, dtype = torch.long),
        "attention_mask": torch.ones(2, 5, dtype = torch.long),
        "pixel_values": torch.arange(5).reshape(5, 1).float(),
        "spatial_shapes": torch.arange(5).reshape(5, 1),
        "pixel_attention_mask": torch.ones(5, 1),
        "num_tiles": [2, 3],
    }
    inputs = [{"prompt": "a", "image": "one"}, {"prompt": "b", "image": "two"}]
    return processor_output, inputs, torch.zeros(2, 3, dtype = torch.long)


def test_a_legacy_processors_extra_keys_reach_the_no_grad_calls():
    """0.22.x-0.23.x names four tensors by hand, so spatial_shapes and num_tiles never left
    the processor and the reference forward ran without arguments the model requires."""
    trainer, _ = _run_legacy(*_lfm2vl_batch())
    for call in trainer.calls:
        assert call["spatial_shapes"].shape[0] == 5, call.keys()
        assert call["num_tiles"] == [2, 3], call.keys()
        assert call["num_images"] == [1, 1]


def test_a_legacy_processors_extra_keys_reach_the_training_batch():
    """The gradient pass reads them off the batch the step returns, not off the processor."""
    _, output = _run_legacy(*_lfm2vl_batch())
    assert output["spatial_shapes"].shape[0] == 5, sorted(output)
    assert output["num_tiles"] == [2, 3], sorted(output)
    assert output["num_images"] == [1, 1], sorted(output)


def test_a_key_the_legacy_processor_did_not_produce_is_not_invented():
    _, output = _run_legacy(*_lfm2vl_batch())
    assert "image_grid_thw" not in output, sorted(output)
    assert "image_position_ids" not in output, sorted(output)


def test_a_text_only_legacy_batch_forwards_nothing_and_saves_nothing():
    import torch

    processor_output = {
        "input_ids": torch.zeros(2, 5, dtype = torch.long),
        "attention_mask": torch.ones(2, 5, dtype = torch.long),
    }
    inputs = [{"prompt": "a"}, {"prompt": "b"}]
    trainer, output = _run_legacy(processor_output, inputs, torch.zeros(2, 3, dtype = torch.long))
    for call in trainer.calls:
        assert set(call) <= {"batch_size"}, call
    assert "num_images" not in output, sorted(output)


def test_the_legacy_token_type_ids_are_widened_over_the_completion():
    """TRL 0.24.0 does this itself right before the same call ("If token_type_ids are used,
    extend them with zeros for the completion part"); the legacy versions never forwarded them,
    so the processor's prompt length copy would not fit a prompt+completion forward."""
    import torch

    processor_output, inputs, completion_ids = _grid_batch()
    processor_output["token_type_ids"] = torch.ones(2, 5, dtype = torch.long)
    trainer, output = _run_legacy(processor_output, inputs, completion_ids)
    for call in trainer.calls:
        assert call["token_type_ids"].shape == (2, 8), call["token_type_ids"].shape
        assert call["token_type_ids"][:, 5:].sum().item() == 0
    assert output["token_type_ids"].shape == (2, 8)


def test_a_legacy_token_type_ids_of_the_wrong_width_is_dropped_not_forwarded():
    """max_prompt_length truncates prompt_ids after the processor ran, so the processor's own
    copy can be wider than the prompt. Sending it would be a shape error in the forward."""
    import torch

    processor_output, inputs, completion_ids = _grid_batch()
    processor_output["token_type_ids"] = torch.ones(2, 9, dtype = torch.long)
    trainer, output = _run_legacy(processor_output, inputs, completion_ids)
    for call in trainer.calls:
        assert "token_type_ids" not in call, call.keys()
    assert "token_type_ids" not in output, sorted(output)


def test_a_modern_trl_is_untouched_by_the_legacy_rewrite():
    """0.24.0 and up keep their own forward_kwargs path; the legacy anchors must not fire."""
    from unsloth.models.rl_replacements import grpo_trainer__generate_and_score_completions

    source = (
        "    def _generate_and_score_completions(self, inputs):\n"
        '        if "image_sizes" in forward_kwargs:\n'
        '            output["image_sizes"] = forward_kwargs["image_sizes"]\n'
        "        if images is not None:\n"
        '            output["num_images"] = num_images\n'
        "        return output\n"
    )
    patched = grpo_trainer__generate_and_score_completions(
        "_generate_and_score_completions", source
    )
    assert "**_unsloth_legacy_vision," not in patched
    assert patched.count('output["num_images"]') == 1, patched


# ---------------------------------------------------------------------------------------------
# The shuffle and the slice, on the layout 0.22.x-0.23.x leaves behind.
# ---------------------------------------------------------------------------------------------


def _legacy_split_pixel_values_by_grid(batch):
    """Verbatim from trl 0.22.2 / 0.23.1 trainer/utils.py: one list element per GRID ROW, which
    is one per image, while everything else in the batch stays indexed by sample."""
    import torch

    if "image_grid_thw" not in batch or "pixel_values" not in batch:
        return batch
    lengths = batch["image_grid_thw"].prod(dim = 1).tolist()
    pixel_values = batch["pixel_values"]
    if sum(lengths) != pixel_values.size(0):
        raise ValueError("Mismatch")
    return {**batch, "pixel_values": list(torch.split(batch["pixel_values"], lengths, dim = 0))}


def _legacy_unsplit_pixel_values_by_grid(batch):
    """Verbatim from trl 0.22.2 / 0.23.1: merges pixel_values and nothing else."""
    import torch

    pixel_values = batch.get("pixel_values")
    if isinstance(pixel_values, list):
        return {**batch, "pixel_values": torch.cat(pixel_values, dim = 0)}
    return batch


def test_a_multi_image_grid_batch_keeps_its_images_with_their_samples():
    """0.22.x-0.23.x splits pixel_values by grid row, so with two images on one sample the list
    is indexed by image while prompt_ids is indexed by sample. shuffle_sequence_dict then takes
    its length from prompt_ids and drops the tail of the list."""
    import torch

    trl_utils = pytest.importorskip("trl.trainer.utils")
    from unsloth.models.rl_replacements import (
        _unsloth_grpo_split_vision_by_sample,
        _unsloth_grpo_unsplit_vision,
    )

    num_images = [2, 1, 1]
    owners = torch.tensor([0, 0, 1, 2])  # one grid row per image, one patch row per image
    batch = {
        "prompt_ids": torch.arange(3).unsqueeze(-1),
        "advantages": torch.zeros(3),
        "num_images": num_images,
        "image_grid_thw": torch.tensor([[1, 1, 1]] * 4),
        "pixel_values": owners.reshape(-1, 1).float(),
        "image_sizes": owners.reshape(-1, 1).clone(),
    }
    split = _unsloth_grpo_split_vision_by_sample(_legacy_split_pixel_values_by_grid(batch))
    assert [t.shape[0] for t in split["pixel_values"]] == num_images, "still indexed by image"
    assert [t.shape[0] for t in split["image_grid_thw"]] == num_images

    torch.manual_seed(0)
    shuffled = trl_utils.shuffle_sequence_dict(split)
    chunk = trl_utils.split_tensor_dict(shuffled, 1)[0]
    restored = _unsloth_grpo_unsplit_vision(_legacy_unsplit_pixel_values_by_grid(chunk))

    assert restored["pixel_values"].shape[0] == len(owners), "an image was dropped"
    assert restored["image_grid_thw"].shape[0] == len(owners)
    at = 0
    seen = []
    for prompt, count in zip(restored["prompt_ids"].tolist(), restored["num_images"]):
        sample = prompt[0]
        seen.append(sample)
        assert count == num_images[sample], (sample, count)
        for key in ("pixel_values", "image_sizes"):
            rows = restored[key][at : at + count].flatten().tolist()
            assert rows == [sample] * count, (key, sample, rows)
        at += count
    assert sorted(seen) == [0, 1, 2], seen


def test_one_image_per_sample_on_a_legacy_trl_is_left_exactly_as_trl_left_it():
    """The grid row axis and the sample axis coincide there, and re-splitting would be churn."""
    import torch

    from unsloth.models.rl_replacements import _unsloth_grpo_split_vision_by_sample

    batch = {
        "num_images": [1, 1],
        "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
        "pixel_values": torch.zeros(8, 1),
    }
    already = _legacy_split_pixel_values_by_grid(batch)
    assert _unsloth_grpo_split_vision_by_sample(already) is already


def test_a_grid_batch_trl_1_1_0_already_split_by_sample_is_left_alone():
    """From 1.1.0 split_pixel_values_by_grid splits both by sample and returns the grid as a
    list, so a second regroup would nest the lists."""
    import torch

    from unsloth.models.rl_replacements import _unsloth_grpo_split_vision_by_sample

    batch = {
        "num_images": [2, 1],
        "image_grid_thw": [torch.tensor([[1, 1, 1]] * 2), torch.tensor([[1, 1, 1]])],
        "pixel_values": [torch.zeros(2, 1), torch.zeros(1, 1)],
    }
    assert _unsloth_grpo_split_vision_by_sample(batch) is batch


def test_a_legacy_grid_row_with_no_image_still_gets_its_neighbours_images_apart():
    """num_images = [0, 2] over two samples makes the image count and the sample count agree
    while the axes still differ, so a bare length comparison would leave the list alone and the
    shuffle would hand both of the second sample's images to the first."""
    import torch

    from unsloth.models.rl_replacements import _unsloth_grpo_split_vision_by_sample

    batch = {
        "prompt_ids": torch.arange(2).unsqueeze(-1),
        "num_images": [0, 2],
        "image_grid_thw": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        "pixel_values": torch.tensor([[7.0], [8.0]]),
    }
    split = _unsloth_grpo_split_vision_by_sample(_legacy_split_pixel_values_by_grid(batch))
    assert [t.shape[0] for t in split["pixel_values"]] == [0, 2]
    assert split["pixel_values"][1].flatten().tolist() == [7.0, 8.0]
    assert [t.shape[0] for t in split["image_grid_thw"]] == [0, 2]


def test_a_legacy_grid_batch_whose_counts_do_not_span_the_samples_is_left_alone():
    """A count list that is not one entry per row cannot index the batch, and regrouping on it
    would silently shorten the list."""
    import torch

    from unsloth.models.rl_replacements import _unsloth_grpo_split_vision_by_sample

    batch = {
        "prompt_ids": torch.arange(3).unsqueeze(-1),
        "num_images": [2, 1],
        "image_grid_thw": torch.tensor([[1, 1, 1]] * 3),
        "pixel_values": torch.arange(3).reshape(3, 1).float(),
    }
    already = _legacy_split_pixel_values_by_grid(batch)
    assert _unsloth_grpo_split_vision_by_sample(already) is already
