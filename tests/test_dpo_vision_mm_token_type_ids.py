# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""VLM DPO on TRL < 0.29: mm_token_type_ids must match the width of the ids the forward sees.

Transformers 5 processors (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Gemma3) return mm_token_type_ids by default. The legacy
row format keeps it at prompt width, the collator pads it to [B, P] and concatenated_inputs stacks it to [2B, P],
while the model gets prompt + completion ids. Qwen's rope index then masks a [P] row with a [P + C] attention mask
and raises IndexError. The forward rewrite relabels the final ids from the prompt's (token id, type) pairs. CPU only.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_dpo_vision_grid_rows as grid  # noqa: E402

IMAGE, VIDEO = 151655, 151656


def _labels(ids):
    out = torch.zeros_like(ids)
    out[ids == IMAGE] = 1
    out[ids == VIDEO] = 2
    return out


def test_relabels_flushed_truncated_and_packed_ids():
    align = grid._lift()["_unsloth_dpo_align_mm_token_type_ids"]
    prompt = torch.tensor(
        [[0, 0, 7, IMAGE, IMAGE, 8], [5, VIDEO, VIDEO, IMAGE, 9, 4], [0, 0, 0, 3, 4, 5]]
    )
    prompt_mm = _labels(prompt)
    # Prompt + completion after flush_left, a keep_end cut and padding-free packing.
    for ids in (
        torch.tensor(
            [
                [7, IMAGE, IMAGE, 8, 1, 2, 3, 0],
                [5, VIDEO, VIDEO, IMAGE, 9, 4, 6, 6],
                [3, 4, 5, 1, 0, 0, 0, 0],
            ]
        ),
        torch.tensor([[IMAGE, 8, 1, 2], [IMAGE, 9, 4, 6], [5, 1, 0, 0]]),
        torch.tensor([[7, IMAGE, IMAGE, 8, 1, 5, VIDEO, VIDEO, IMAGE, 9, 4, 6, 3, 4, 5, 1]]),
    ):
        got = align(prompt, prompt_mm, ids)
        assert got.shape == ids.shape and got.dtype == prompt_mm.dtype
        assert torch.equal(got, _labels(ids))
    # Already the right width, or a text-only batch: unchanged / all text.
    same = torch.randint(0, 10, (3, 6))
    assert align(prompt, prompt_mm, same) is prompt_mm
    text = torch.tensor([[1, 2, 3], [4, 5, 6]])
    assert torch.equal(
        align(text[:, :2], torch.zeros(2, 2, dtype = torch.long), text), torch.zeros_like(text)
    )


def test_rewrite_lands_before_every_forward_of_the_installed_trl():
    mod = grid._legacy_module()
    if not hasattr(mod.DPOTrainer, "concatenated_forward"):
        pytest.skip(reason = "TRL >= 0.29 has no legacy concatenated_forward")
    import inspect

    ns = grid._lift()
    for name, rewrite in (
        ("concatenated_forward", "dpo_trainer_concatenated_forward"),
        ("_compute_loss_liger", "dpo_trainer_compute_loss_liger"),
    ):
        if rewrite not in ns or not hasattr(mod.DPOTrainer, name):
            continue
        source = ns[rewrite](name, inspect.getsource(getattr(mod.DPOTrainer, name)))
        assert source.count("_unsloth_dpo_align_mm_token_type_ids(") == 1, name


@pytest.mark.parametrize("max_length", [None, 35])
def test_forward_gets_full_width_mm_token_type_ids(max_length, tmp_path):
    mod = grid._legacy_module()
    if not hasattr(mod.DPOTrainer, "concatenated_forward"):
        pytest.skip(reason = "TRL >= 0.29 has no legacy concatenated_forward")
    torch.manual_seed(0)
    model, proc = grid._load(grid.MODELS["qwen2_5_vl"])
    cls = grid._patched_trainer_class(mod)
    cfg = mod.DPOConfig(
        output_dir = str(tmp_path),
        per_device_train_batch_size = 2,
        report_to = "none",
        save_strategy = "no",
        use_cpu = True,
        max_length = max_length,
        truncation_mode = "keep_end",
        max_prompt_length = None,
        remove_unused_columns = False,
        dataset_num_proc = None,
    )
    cfg.dataset_num_proc = None
    trainer = cls(
        model = model,
        ref_model = copy.deepcopy(model),
        args = cfg,
        train_dataset = grid._dataset(),
        processing_class = proc,
    )
    rows = [trainer.train_dataset[i] for i in range(2)]
    batch = trainer.data_collator(rows)
    image_id = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if "mm_token_type_ids" not in batch:
        # Transformers 4.x processors do not return it: build the prompt-width tensor the 5.x row carries.
        batch["mm_token_type_ids"] = (batch["prompt_input_ids"] == image_id).long()
    assert batch["mm_token_type_ids"].shape == batch["prompt_input_ids"].shape

    seen = []
    forward = model.forward

    def checked_forward(input_ids = None, **kwargs):
        mm = kwargs.get("mm_token_type_ids")
        seen.append(
            None
            if mm is None
            else (tuple(mm.shape), torch.equal(mm, (input_ids == image_id).to(mm.dtype)))
        )
        assert mm is not None and mm.shape == input_ids.shape, (
            None if mm is None else tuple(mm.shape),
            tuple(input_ids.shape),
        )
        return forward(input_ids = input_ids, **kwargs)

    model.forward = checked_forward
    model.eval()
    with torch.no_grad():
        out = trainer.concatenated_forward(model, batch)
    assert seen and all(ok for _, ok in seen), seen
    assert torch.isfinite(out["chosen_logps"]).all() and torch.isfinite(out["rejected_logps"]).all()
