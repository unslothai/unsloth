# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Context parallelism plumbing that runs without a process group (the ring itself needs >= 2 GPUs)."""

import contextlib

import pytest
import torch

import unsloth  # noqa: F401
from unsloth import context_parallel as cp
from unsloth.utils import attention_dispatch as ad


def _manager(size = 2):
    manager = object.__new__(cp.ContextParallelManager)
    manager.size = size
    manager._hooked = False
    return manager


def _fake_context_parallel(sharded):
    @contextlib.contextmanager
    def fake(mesh, buffers, buffer_seq_dims, no_restore_buffers):
        sharded.extend(buffers)
        yield

    return fake


def test_labels_shift_before_sharding_and_pad_to_load_balancer_chunks():
    manager = _manager(size = 2)
    labels = torch.tensor([[-100, 5, 6, 7, 8, 9]])
    inputs = {"input_ids": torch.tensor([[1, 5, 6, 7, 8, 9]]), "labels": labels.clone()}
    manager._prepare_inputs(inputs)
    # 6 tokens -> 8 (2 * size chunks); the target of token t is label t + 1, even across a shard seam.
    assert inputs["shift_labels"].tolist() == [[5, 6, 7, 8, 9, -100, -100, -100]]
    assert inputs["labels"].tolist() == [[-100, 5, 6, 7, 8, 9, -100, -100]]
    assert inputs["position_ids"].tolist() == [[0, 1, 2, 3, 4, 5, 6, 7]]
    assert inputs["input_ids"].shape == (1, 8)


def test_active_manager_resets_when_the_step_raises(monkeypatch):
    monkeypatch.setattr(cp, "context_parallel", _fake_context_parallel([]))
    manager = _manager()
    manager.mesh = None
    inputs = {"input_ids": torch.ones(1, 4, dtype = torch.long)}
    with pytest.raises(RuntimeError, match = "boom"):
        with manager.apply(inputs):
            assert cp.get_cp_manager() is manager
            raise RuntimeError("boom")
    assert cp.get_cp_manager() is None


def test_attention_hook_only_drops_the_mask_inside_a_cp_step(monkeypatch):
    monkeypatch.setattr(cp, "context_parallel", _fake_context_parallel([]))
    mask = torch.ones(1, 4)
    _, kwargs = cp._self_attn_pre_forward_hook(None, (), {"attention_mask": mask})
    assert kwargs["attention_mask"] is mask
    manager = _manager()
    manager.mesh = None
    with manager.apply({"input_ids": torch.ones(1, 4, dtype = torch.long)}):
        _, kwargs = cp._self_attn_pre_forward_hook(None, (), {"attention_mask": mask})
    assert kwargs["attention_mask"] is None


def test_backend_is_sdpa_under_cp_and_packing_is_refused(monkeypatch):
    monkeypatch.setattr(cp, "context_parallel", _fake_context_parallel([]))
    manager = _manager()
    manager.mesh = None
    with manager.apply({"input_ids": torch.ones(1, 4, dtype = torch.long)}):
        assert ad.select_attention_backend(False) == ad.SDPA
        with pytest.raises(ValueError, match = "packing"):
            ad.select_attention_backend(True)


def test_sdpa_is_looked_up_per_call(monkeypatch):
    # context_parallel swaps torch.nn.functional.scaled_dot_product_attention for ring attention;
    # a name bound at import would silently keep attending to the local shard only.
    calls = []
    monkeypatch.setattr(
        torch.nn.functional, "scaled_dot_product_attention", lambda *a, **k: calls.append(1)
    )
    ad.scaled_dot_product_attention(None, None, None)
    assert calls == [1]


def test_compute_loss_divides_the_pre_shard_token_count_by_cp_size(monkeypatch):
    seen = {}

    class Trainer:
        def __init__(self):
            pass

        def compute_loss(
            self,
            model,
            inputs,
            return_outputs = False,
            num_items_in_batch = None,
        ):
            seen["n"] = num_items_in_batch
            return torch.tensor(0.0)

        prediction_step = training_step = compute_loss

    import trl

    monkeypatch.setattr(trl, "SFTTrainer", Trainer)
    cp.patch_sft_trainer()
    trainer = Trainer()
    trainer._context_parallel_manager = None
    trainer.compute_loss(None, {}, num_items_in_batch = torch.tensor(12.0))
    assert seen["n"] == 12
    trainer._context_parallel_manager = _manager(size = 4)
    trainer.compute_loss(None, {}, num_items_in_batch = torch.tensor(12.0))
    assert seen["n"] == 3


def test_shift_labels_is_not_an_eval_label_name():
    # find_labels treats every forward parameter containing "label" as required: a named
    # shift_labels made plain eval batches look label-less (KeyError 'prompt' in prediction_step).
    from transformers.utils import find_labels
    from unsloth.models.llama import CausalLM_fast_forward, LlamaModel_fast_forward_inference

    class LM(torch.nn.Module):
        forward = CausalLM_fast_forward(LlamaModel_fast_forward_inference)

    assert find_labels(LM) == ["labels"]
