# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO video prompts repeat num_generations times; the bank layout keeps one copy of each distinct
prompt's videos. Every tensor a forward sees must be bit-identical to the flat layout."""

import types

import pytest
import torch

zoo = pytest.importorskip("unsloth_zoo.rl_replacements")
if not hasattr(zoo, "grpo_video_bank_supported"):
    pytest.skip("unsloth_zoo without the video bank", allow_module_level = True)


class _Processor:
    """Video v of prompt pid: grid (1, 2, 2 * (v + 1)), rows filled with distinct values."""

    def apply_chat_template(self, conversation, **kwargs):
        grids, rows, seconds = [], [], []
        for prompt in conversation:
            pid = int(prompt[0]["content"][-1]["text"])
            n = sum(1 for p in prompt[0]["content"] if p.get("type") == "video")
            for v in range(n):
                grid = torch.tensor([1, 2, 2 * (v + 1)])
                grids.append(grid)
                k = int(grid.prod())
                rows.append(
                    pid * 1000 + v * 100 + torch.arange(k * 3, dtype = torch.float32).reshape(k, 3)
                )
                seconds.append(float(pid) + v / 10)
        out = {"input_ids": torch.zeros(len(conversation), 4, dtype = torch.long)}
        if grids:
            out["pixel_values_videos"] = torch.cat(rows)
            out["video_grid_thw"] = torch.stack(grids)
            out["second_per_grid_ts"] = torch.tensor(seconds)
        return out


def _prompt(pid, n_videos):
    content = [{"type": "video", "video": f"{pid}_{v}.mp4"} for v in range(n_videos)]
    content.append({"type": "text", "text": str(pid)})
    return [{"role": "user", "content": content}]


def _trainer():
    return types.SimpleNamespace(
        processing_class = _Processor(),
        use_vllm = False,
        chat_template_kwargs = {},
        accelerator = types.SimpleNamespace(device = torch.device("cpu")),
    )


def _both(prompts, monkeypatch):
    from unsloth.models.rl_replacements import _unsloth_grpo_video_inputs

    monkeypatch.setenv("UNSLOTH_GRPO_VIDEO_BANK", "0")
    flat = _unsloth_grpo_video_inputs(_trainer(), prompts)
    monkeypatch.setenv("UNSLOTH_GRPO_VIDEO_BANK", "1")
    bank = _unsloth_grpo_video_inputs(_trainer(), prompts)
    return flat, bank


CASES = {
    # num_generations = 4, grouped like TRL's RepeatSampler
    "grouped_g4": [(1, 1)] * 4 + [(2, 2)] * 4 + [(3, 1)] * 4,
    # a prompt with two videos and one without any, interleaved order (non-contiguous bank rows)
    "interleaved_mixed": [(1, 2), (2, 0), (3, 1), (1, 2), (2, 0), (3, 1)],
    "g8_single_prompt": [(5, 1)] * 8,
}


def _assert_same(a, b):
    assert a.keys() == b.keys(), (a.keys(), b.keys())
    for key in a:
        if torch.is_tensor(a[key]):
            assert a[key].dtype == b[key].dtype and torch.equal(a[key], b[key]), key
        else:
            assert a[key] == b[key], key


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("batch_size", [1, 2, 3, 5, 100])
def test_chunks_bit_identical(case, batch_size, monkeypatch):
    prompts = [_prompt(pid, n) for pid, n in CASES[case]]
    flat, bank = _both(prompts, monkeypatch)
    assert "video_bank_rows" not in flat and "video_bank_rows" in bank
    assert bank["pixel_values_videos"].numel() < flat["pixel_values_videos"].numel()
    total = len(prompts)
    for vision in (zoo.grpo_get_vision_inputs(flat), zoo.grpo_get_vision_inputs(bank)):
        assert vision["pixel_values_videos"] is not None
    c_flat = zoo.grpo_vision_chunks(zoo.grpo_get_vision_inputs(flat), total, batch_size)
    c_bank = zoo.grpo_vision_chunks(zoo.grpo_get_vision_inputs(bank), total, batch_size)
    assert len(c_flat) == len(c_bank)
    for a, b in zip(c_flat, c_bank):
        _assert_same(a, b)


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_buffer_split_shuffle_unsplit_bit_identical(case, seed, monkeypatch):
    trl_utils = pytest.importorskip("trl.trainer.utils")
    from unsloth.models.rl_replacements import (
        _unsloth_grpo_split_vision_by_sample,
        _unsloth_grpo_unsplit_vision,
    )

    prompts = [_prompt(pid, n) for pid, n in CASES[case]]
    flat, bank = _both(prompts, monkeypatch)
    steps = {}
    for name, vision in (("flat", flat), ("bank", bank)):
        batch = {
            "prompt_ids": torch.arange(len(prompts)).unsqueeze(-1),
            "advantages": torch.zeros(len(prompts)),
        }
        batch.update(zoo.grpo_get_vision_inputs(vision))
        batch = {k: v for k, v in batch.items() if v is not None}
        split = _unsloth_grpo_split_vision_by_sample(trl_utils.split_pixel_values_by_grid(batch))
        assert isinstance(split.get("video_bank_rows", []), list)
        if name == "bank":
            # Views of the one bank copy, not per-sample copies.
            base = bank["pixel_values_videos"].untyped_storage().data_ptr()
            assert all(
                t.numel() == 0 or t.untyped_storage().data_ptr() == base
                for t in split["pixel_values_videos"]
                if t.numel()
            )
        torch.manual_seed(seed)
        shuffled = trl_utils.shuffle_sequence_dict(split)
        steps[name] = [
            _unsloth_grpo_unsplit_vision(trl_utils.unsplit_pixel_values_by_grid(part))
            for part in trl_utils.split_tensor_dict(shuffled, 3)
        ]
    bank_ptr = bank["pixel_values_videos"].untyped_storage().data_ptr()
    for a, b in zip(steps["flat"], steps["bank"]):
        assert "video_bank_rows" not in a
        # every step references the one bank, no per-step copy
        assert b["pixel_values_videos"].untyped_storage().data_ptr() == bank_ptr
        assert b["video_bank_rows"].shape[0] == b["video_grid_thw"].shape[0]
        b_cmp = {k: v for k, v in b.items() if k not in ("pixel_values_videos", "video_bank_rows")}
        a_cmp = {k: v for k, v in a.items() if k != "pixel_values_videos"}
        _assert_same(a_cmp, b_cmp)
        # what grpo_accumulated_loss forwards, per chunk of every size
        n = a["prompt_ids"].shape[0]
        for size in (1, 2, n):
            for x, y in zip(zoo.grpo_vision_chunks(a, n, size), zoo.grpo_vision_chunks(b, n, size)):
                _assert_same(x, y)
        if a["pixel_values_videos"].shape[0]:
            full_b = zoo.grpo_vision_chunks(b, n, n)[0]["pixel_values_videos"]
            assert torch.equal(full_b, a["pixel_values_videos"])


def test_unsplit_falls_back_to_a_copy_when_not_views(monkeypatch):
    from unsloth.models.rl_replacements import _unsloth_grpo_unsplit_vision

    a = torch.arange(12.0).reshape(4, 3)
    step = {
        "pixel_values_videos": [a[0:2].clone(), a[2:4].clone()],
        "video_bank_rows": [torch.tensor([[0, 2]]), torch.tensor([[2, 4]])],
        "video_grid_thw": [torch.tensor([[1, 1, 2]]), torch.tensor([[1, 1, 2]])],
    }
    out = _unsloth_grpo_unsplit_vision(step)
    assert "video_bank_rows" not in out
    assert torch.equal(out["pixel_values_videos"], a)


def test_no_repeats_keeps_flat_layout(monkeypatch):
    prompts = [_prompt(1, 1), _prompt(2, 2)]
    flat, bank = _both(prompts, monkeypatch)
    _assert_same(flat, bank)


def test_old_zoo_gets_flat_layout(monkeypatch):
    from unsloth.models.rl_replacements import _unsloth_grpo_video_inputs

    prompts = [_prompt(1, 1)] * 4
    monkeypatch.delattr(zoo, "grpo_video_bank_supported")
    out = _unsloth_grpo_video_inputs(_trainer(), prompts)
    assert "video_bank_rows" not in out
    assert out["pixel_values_videos"].shape[0] == 4 * 4


def test_restricted_companion_keeps_the_index(monkeypatch):
    prompts = [_prompt(1, 1)] * 4
    _, bank = _both(prompts, monkeypatch)
    monkeypatch.setattr(zoo, "grpo_companion_vision_keys", lambda: zoo.GRPO_VISION_KEYS[:-1])
    shared = zoo.grpo_shared_vision_inputs(bank)
    assert "video_bank_rows" in shared


def test_mismatched_index_raises(monkeypatch):
    prompts = [_prompt(1, 1)] * 4
    _, bank = _both(prompts, monkeypatch)
    vision = zoo.grpo_get_vision_inputs(bank)
    vision["video_bank_rows"] = vision["video_bank_rows"].clone()
    vision["video_bank_rows"][0, 1] += 1
    with pytest.raises(ValueError, match = "video_bank_rows"):
        zoo.grpo_vision_chunks(vision, 4, 1)


def test_vision_keys_tuple_unchanged():
    # The index is not a model kwarg: the companion key-count checks must not see it.
    assert "video_bank_rows" not in zoo.GRPO_VISION_KEYS
    assert "video_bank_rows" in zoo.GRPO_VISION_INDEX_KEYS
