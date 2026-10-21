# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TRL builds forward kwargs from image / images only, so video prompts were scored as text."""

import inspect
import types

import pytest
import torch


def _video_prompt(
    path,
    text = "What happens?",
    arrow = False,
):
    video = {"type": "video", "video": path}
    words = {"type": "text", "text": text}
    if arrow:
        # Arrow gives every part the union of the keys.
        video["text"] = None
        words["video"] = None
    return [{"role": "user", "content": [video, words]}]


def test_arrow_none_keys_are_stripped_from_video_prompts_only():
    from unsloth.models.rl_replacements import _unsloth_grpo_clean_video_prompts

    text_only = [{"role": "user", "content": [{"type": "text", "text": "hi", "image": None}]}]
    video = _video_prompt("a.mp4", arrow = True)
    cleaned = _unsloth_grpo_clean_video_prompts([text_only, video])
    assert cleaned[0] is text_only
    assert cleaned[1][0]["content"] == [
        {"type": "video", "video": "a.mp4"},
        {"type": "text", "text": "What happens?"},
    ]
    assert video[0]["content"][1]["video"] is None, "the dataset row was mutated"
    # a text row of a column that also holds videos carries the key too
    text_row = [{"role": "user", "content": [{"type": "text", "text": "hi", "video": None}]}]
    assert _unsloth_grpo_clean_video_prompts([text_row])[0][0]["content"] == [
        {"type": "text", "text": "hi"}
    ]
    batch = [text_only]
    assert _unsloth_grpo_clean_video_prompts(batch) is batch


class _Processor:
    """apply_chat_template stand in: video v of a prompt gets grid (1, 2, 2) * (v + 1) rows
    filled with the prompt's id, so the expansion order can be read back."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, conversation, **kwargs):
        self.calls.append(conversation)
        self.kwargs = kwargs
        grids, rows, seconds = [], [], []
        for prompt in conversation:
            pid = int(prompt[0]["content"][-1]["text"])
            n = sum(1 for p in prompt[0]["content"] if p.get("type") == "video")
            for v in range(n):
                grid = torch.tensor([1, 2, 2 * (v + 1)])
                grids.append(grid)
                rows.append(torch.full((int(grid.prod()), 3), float(pid)))
                seconds.append(float(pid) + v / 10)
        return {
            "input_ids": torch.zeros(len(conversation), 4, dtype = torch.long),
            "pixel_values_videos": torch.cat(rows),
            "video_grid_thw": torch.stack(grids),
            "second_per_grid_ts": torch.tensor(seconds),
        }


def _trainer(processor, use_vllm = False):
    return types.SimpleNamespace(
        processing_class = processor,
        use_vllm = use_vllm,
        chat_template_kwargs = {},
        accelerator = types.SimpleNamespace(device = torch.device("cpu")),
    )


@pytest.fixture
def zoo_with_video_keys(monkeypatch):
    """The helper refuses a zoo whose key tuple lacks video; present one that has it."""
    zoo = pytest.importorskip("unsloth_zoo.rl_replacements")
    keys = tuple(getattr(zoo, "GRPO_VISION_KEYS", ()))
    if "pixel_values_videos" not in keys:
        keys = keys + ("pixel_values_videos", "video_grid_thw", "second_per_grid_ts", "num_videos")
    monkeypatch.setattr(zoo, "GRPO_VISION_KEYS", keys, raising = False)


def _two_video_prompt(pid):
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "a.mp4"},
                {"type": "video", "video": "b.mp4"},
                {"type": "text", "text": str(pid)},
            ],
        }
    ]


def test_each_distinct_prompt_is_decoded_once_and_repeated_per_generation(zoo_with_video_keys):
    from unsloth.models.rl_replacements import _unsloth_grpo_video_inputs

    processor = _Processor()
    one = _video_prompt("a.mp4", text = "1")
    two = _two_video_prompt(2)
    prompts = [one, one, two, two]  # num_generations = 2
    out = _unsloth_grpo_video_inputs(_trainer(processor), prompts)

    assert len(processor.calls) == 1 and len(processor.calls[0]) == 2
    assert out["num_videos"] == [1, 1, 2, 2]
    assert out["video_grid_thw"].tolist() == [
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 4],
        [1, 2, 2],
        [1, 2, 4],
    ]
    owners = out["pixel_values_videos"][:, 0].tolist()
    assert owners == [1.0] * 8 + [2.0] * 12 + [2.0] * 12
    assert out["second_per_grid_ts"].tolist() == pytest.approx([1.0, 1.0, 2.0, 2.1, 2.0, 2.1])


def test_in_memory_videos_are_never_shared_between_prompts(zoo_with_video_keys):
    import numpy as np

    from unsloth.models.rl_replacements import _unsloth_grpo_video_inputs

    processor = _Processor()
    a = np.zeros((64, 64, 3))
    b = np.zeros((64, 64, 3))
    b[32, 32, 0] = 1.0  # same repr() as a
    assert repr(a) == repr(b)
    prompts = [_video_prompt(a, text = "1"), _video_prompt(b, text = "1")]
    _unsloth_grpo_video_inputs(_trainer(processor), prompts)
    assert len(processor.calls[0]) == 2


def test_the_trainers_chat_template_is_used(zoo_with_video_keys):
    from unsloth.models.rl_replacements import _unsloth_grpo_video_inputs

    processor = _Processor()
    trainer = _trainer(processor)
    trainer.chat_template = "{{ messages }}"
    _unsloth_grpo_video_inputs(trainer, [_video_prompt("a.mp4", text = "1")])
    assert processor.kwargs["chat_template"] == "{{ messages }}"


def test_a_batch_without_video_costs_nothing():
    from unsloth.models.rl_replacements import _unsloth_grpo_video_inputs

    processor = _Processor()
    text = [{"role": "user", "content": [{"type": "text", "text": "1"}]}]
    assert _unsloth_grpo_video_inputs(_trainer(processor), [text, text]) is None
    assert _unsloth_grpo_video_inputs(_trainer(processor), ["plain text prompt"]) is None
    assert processor.calls == []


def test_vllm_and_mixed_image_video_batches_are_refused(zoo_with_video_keys):
    from unsloth.models.rl_replacements import _unsloth_grpo_video_inputs

    prompts = [_video_prompt("a.mp4", text = "1")]
    with pytest.raises(NotImplementedError, match = "fast_inference"):
        _unsloth_grpo_video_inputs(_trainer(_Processor(), use_vllm = True), prompts)
    with pytest.raises(NotImplementedError, match = "image and video"):
        _unsloth_grpo_video_inputs(_trainer(_Processor()), prompts, images = [["img"]])
    inline = _video_prompt("a.mp4", text = "1")
    inline[0]["content"].insert(0, {"type": "image", "image": "x.png"})
    with pytest.raises(NotImplementedError, match = "image and video"):
        _unsloth_grpo_video_inputs(_trainer(_Processor()), [inline])


def test_vllm_is_refused_before_generation():
    from unsloth.models.rl_replacements import _unsloth_grpo_clean_video_prompts

    trainer = types.SimpleNamespace(use_vllm = True)
    with pytest.raises(NotImplementedError, match = "fast_inference"):
        _unsloth_grpo_clean_video_prompts([_video_prompt("a.mp4")], trainer)
    text = [[{"role": "user", "content": [{"type": "text", "text": "hi"}]}]]
    assert _unsloth_grpo_clean_video_prompts(text, trainer) is text


def test_an_old_zoo_is_refused_rather_than_training_on_text(monkeypatch):
    zoo = pytest.importorskip("unsloth_zoo.rl_replacements")
    from unsloth.models.rl_replacements import _unsloth_grpo_video_inputs

    monkeypatch.setattr(zoo, "GRPO_VISION_KEYS", ("pixel_values", "image_grid_thw"), raising = False)
    with pytest.raises(RuntimeError, match = "upgrade"):
        _unsloth_grpo_video_inputs(_trainer(_Processor()), [_video_prompt("a.mp4", text = "1")])


def test_video_rows_survive_the_shuffle_and_the_slice():
    trl_utils = pytest.importorskip("trl.trainer.utils")
    from unsloth.models.rl_replacements import (
        _unsloth_grpo_split_vision_by_sample,
        _unsloth_grpo_unsplit_vision,
    )

    # sample 0: one 4 row video, sample 1: 4 + 8 rows, sample 2: one 4 row video
    grid = torch.tensor([[1, 2, 2], [1, 2, 2], [1, 2, 4], [1, 2, 2]])
    owners = torch.tensor([0] * 4 + [1] * 12 + [2] * 4)
    batch = {
        "prompt_ids": torch.arange(3).unsqueeze(-1),
        "advantages": torch.zeros(3),
        "num_videos": [1, 2, 1],
        "pixel_values_videos": owners.reshape(-1, 1).float(),
        "video_grid_thw": grid,
        "second_per_grid_ts": torch.tensor([0.0, 1.0, 1.5, 2.0]),
    }
    torch.manual_seed(0)
    split = _unsloth_grpo_split_vision_by_sample(batch)
    assert isinstance(split["pixel_values_videos"], list)
    shuffled = trl_utils.shuffle_sequence_dict(split)
    for half in trl_utils.split_tensor_dict(shuffled, 3):
        restored = _unsloth_grpo_unsplit_vision(trl_utils.unsplit_pixel_values_by_grid(half))
        sample = restored["prompt_ids"].item()
        n = restored["num_videos"][0]
        assert n == [1, 2, 1][sample]
        assert restored["video_grid_thw"].shape[0] == n
        rows = int(restored["video_grid_thw"].prod(-1).sum())
        assert restored["pixel_values_videos"].flatten().tolist() == [sample] * rows
        expected = {0: [0.0], 1: [1.0, 1.5], 2: [2.0]}[sample]
        assert restored["second_per_grid_ts"].tolist() == expected


def test_the_trl_trainer_gets_the_videos_merged_into_its_forward_kwargs():
    import ast
    import importlib.util

    spec = importlib.util.find_spec("trl.trainer.grpo_trainer")
    if spec is None or spec.origin is None:
        pytest.skip("trl is not installed")
    from unsloth.models.rl_replacements import grpo_trainer__generate_and_score_completions

    # From the file: once unsloth is imported, trl.GRPOTrainer is the patched class.
    with open(spec.origin, "r", encoding = "utf-8") as fh:
        module_source = fh.read()
    node = next(
        n
        for n in ast.walk(ast.parse(module_source))
        if isinstance(n, ast.FunctionDef) and n.name == "_generate_and_score_completions"
    )
    source = "    " + ast.get_source_segment(module_source, node)
    if "forward_kwargs = {}" not in source:
        pytest.skip("this TRL has no forward_kwargs (< 0.24.0)")
    patched = grpo_trainer__generate_and_score_completions(
        "_generate_and_score_completions", source
    )
    assert "prompts = _unsloth_grpo_clean_video_prompts(prompts, self)" in patched
    assert "_unsloth_video_kwargs = _unsloth_grpo_video_inputs(" in patched
    merge = patched.index("forward_kwargs = {**forward_kwargs, **_unsloth_video_kwargs}")
    old_logps = patched.index("_get_per_token_logps_and_entropies(")
    assert merge < old_logps, "the videos arrive after the old and reference logprobs"


def test_both_logprob_passes_treat_video_rows_as_vision_rows():
    # Left packing and sequence packing move tokens, which breaks M-RoPE video positions.
    from unsloth.models import rl_replacements

    source = inspect.getsource(rl_replacements.grpo_trainer__get_per_token_logps_and_entropies)
    sentinel = source.index('pixel_values = vision_inputs.get("pixel_values_videos", None)')
    assert sentinel < source.index(
        "if pixel_values is None:\n                left_pad_tokens_per_prompt"
    )
    for fn in (
        rl_replacements.grpo_trainer__get_per_token_logps_and_entropies,
        rl_replacements.grpo_trainer_compute_loss,
    ):
        assert "video_grid_thw is not None" in inspect.getsource(fn), fn.__name__
