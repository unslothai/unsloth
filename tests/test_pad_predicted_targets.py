# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A causal target predicted from a pad position is never trained.

With left or interior padding the last pad before a real token predicts it. That pad has no real
key to attend to, so eager, SDPA and FlashAttention-2 each give it a different hidden state, and
the loss term (plus every gradient behind it, up to embed_tokens) depends on the attention backend.
The trainer masks those targets before counting num_items_in_batch."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from unsloth.models import _utils  # noqa: E402

LEFT_MASK = [[1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1]]
RIGHT_MASK = [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]]
INTERIOR_MASK = [[1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 1, 1], [1, 0, 0, 0, 0, 1]]


def _batch(mask):
    mask = torch.tensor(mask)
    input_ids = torch.arange(10, 10 + mask.numel()).view(mask.shape).masked_fill(mask == 0, 0)
    labels = input_ids.masked_fill(mask == 0, -100)
    return {"input_ids": input_ids, "attention_mask": mask, "labels": labels}


def _pad_predicted(mask):
    mask = torch.tensor(mask)
    out = torch.zeros_like(mask, dtype = torch.bool)
    out[:, 1:] = (mask[:, :-1] == 0) & (mask[:, 1:] == 1)
    return out


@pytest.mark.parametrize("mask", [LEFT_MASK, INTERIOR_MASK], ids = ["left", "interior"])
def test_targets_after_a_pad_are_ignored(mask):
    batch = _batch(mask)
    before = batch["labels"].clone()
    out = _utils._mask_pad_predicted_targets(batch)["labels"]
    hit = _pad_predicted(mask)
    assert hit.any()
    assert (out[hit] == -100).all()
    # Nothing else moves.
    assert torch.equal(out[~hit], before[~hit])


def test_right_padding_and_unpadded_rows_are_unchanged():
    batch = _batch(RIGHT_MASK)
    before = batch["labels"].clone()
    assert torch.equal(_utils._mask_pad_predicted_targets(batch)["labels"], before)


def test_labels_sharing_storage_with_input_ids_do_not_corrupt_inputs():
    mask = torch.tensor(LEFT_MASK)
    input_ids = torch.arange(10, 10 + mask.numel()).view(mask.shape)
    batch = {"input_ids": input_ids, "attention_mask": mask, "labels": input_ids}
    _utils._mask_pad_predicted_targets(batch)
    assert (input_ids >= 10).all()


@pytest.mark.parametrize(
    "batch",
    [
        [{"input_ids": torch.ones(2, 3)}],  # GRPO hands a list of dicts
        {"labels": torch.ones(2, 3, dtype = torch.long)},  # no attention mask
        {
            "labels": torch.ones(2, dtype = torch.long),
            "attention_mask": torch.ones(2, dtype = torch.long),
        },
        {
            "labels": torch.ones(2, 3, dtype = torch.long),
            "attention_mask": torch.zeros(2, 3),
        },  # float mask
        {
            "labels": torch.ones(2, 3, dtype = torch.long),
            "attention_mask": torch.zeros(2, 4, dtype = torch.long),
        },
    ],
)
def test_other_batches_pass_through(batch):
    labels = batch["labels"] if isinstance(batch, dict) and "labels" in batch else None
    out = _utils._mask_pad_predicted_targets(batch)
    assert out is batch
    if labels is not None:
        assert out["labels"] is labels


def _tiny(config_name, model_name):
    config_cls = getattr(transformers, config_name, None)
    model_cls = getattr(transformers, model_name, None)
    if config_cls is None or model_cls is None:
        pytest.skip(f"this transformers has no {model_name}")
    config = config_cls(
        vocab_size = 128,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        max_position_embeddings = 64,
    )
    return model_cls(config)


def _trainer(model):
    return SimpleNamespace(
        model = model,
        args = SimpleNamespace(average_tokens_across_devices = False, n_gpu = 1, world_size = 1),
        compute_loss_func = None,
        model_accepts_loss_kwargs = True,
    )


def test_trainer_masks_before_counting_num_items_in_batch():
    stock = lambda self, *a, **k: pytest.fail("a causal LM must take the Unsloth path")
    dispatch = _utils._make_seq2seq_aware_get_batch_samples(stock)
    trainer = _trainer(_tiny("LlamaConfig", "LlamaForCausalLM"))
    batches = [_batch(LEFT_MASK), _batch(INTERIOR_MASK)]
    expected = []
    for b in batches:
        labels = b["labels"].clone()
        labels[_pad_predicted(b["attention_mask"].tolist())] = -100
        expected.append(labels)
    samples, num_items = dispatch(trainer, iter(batches), 2)
    for got, want in zip(samples, expected):
        assert torch.equal(got["labels"], want)
    assert int(num_items) == sum(int((w[:, 1:] != -100).sum()) for w in expected)


def test_keyword_epoch_iterator_is_masked_too():
    dispatch = _utils._make_seq2seq_aware_get_batch_samples(lambda self, *a, **k: None)
    trainer = _trainer(_tiny("LlamaConfig", "LlamaForCausalLM"))
    samples, _ = dispatch(trainer, epoch_iterator = iter([_batch(LEFT_MASK)]), num_batches = 1)
    assert (samples[0]["labels"][_pad_predicted(LEFT_MASK)] == -100).all()


def test_token_classification_labels_are_not_shifted_targets():
    # Unshifted labels: the token after a pad is supervised from its own position.
    trainer = _trainer(_tiny("LlamaConfig", "LlamaForTokenClassification"))
    assert not _utils._trains_shifted_causal_labels(trainer)
    batch = _batch(LEFT_MASK)
    before = batch["labels"].clone()
    args, kwargs = _utils._with_pad_target_masking(trainer, (iter([batch]), 1), {})
    assert torch.equal(next(args[0])["labels"], before)


def _flash_attn_usable():
    from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py

    if not has_real_cuda() or torch.cuda.get_device_capability()[0] < 8:
        return False
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.gpu
@pytest.mark.parametrize("padding", ["left", "interior"])
def test_flash_attention_padded_rows_train_like_eager(padding):
    if not _flash_attn_usable():
        pytest.skip("needs flash-attn on an sm80+ GPU")
    config = transformers.Qwen3Config(
        vocab_size = 512,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        max_position_embeddings = 128,
    )
    torch.manual_seed(0)
    state = transformers.Qwen3ForCausalLM(config).state_dict()
    L, lengths = 32, (32, 24, 14, 6)
    mask = torch.zeros(len(lengths), L, dtype = torch.long)
    for i, n in enumerate(lengths):
        if padding == "left":
            mask[i, L - n :] = 1
        else:
            mask[i, : n // 2] = 1
            mask[i, L - (n - n // 2) :] = 1
    input_ids = torch.randint(3, 512, mask.shape, generator = torch.Generator().manual_seed(1))
    input_ids = input_ids.masked_fill(mask == 0, 0)
    batch = {
        "input_ids": input_ids,
        "attention_mask": mask,
        "labels": input_ids.masked_fill(mask == 0, -100),
    }

    model = transformers.Qwen3ForCausalLM(config)
    model.load_state_dict(state)
    trainer = _trainer(model)
    dispatch = _utils._make_seq2seq_aware_get_batch_samples(lambda self, *a, **k: None)
    (batch,), _ = dispatch(trainer, iter([batch]), 1)

    def grads(attn_implementation, dtype):
        config._attn_implementation = attn_implementation
        m = transformers.Qwen3ForCausalLM(config).to(dtype)
        m.load_state_dict(state)
        m = m.cuda().train()
        embeds = m.model.embed_tokens(batch["input_ids"].cuda()).detach().requires_grad_(True)
        loss = m(
            inputs_embeds = embeds,
            attention_mask = batch["attention_mask"].cuda(),
            labels = batch["labels"].cuda(),
        ).loss
        loss.backward()
        flat = torch.cat([p.grad.float().flatten() for p in m.parameters() if p.grad is not None])
        return flat, embeds.grad.float()

    ref, ref_embeds = grads("eager", torch.float32)
    flash, flash_embeds = grads("flash_attention_2", torch.bfloat16)
    pads = batch["attention_mask"].cuda() == 0
    # No loss term reads a pad position, so nothing flows back into one.
    assert flash_embeds[pads].abs().max() == 0
    assert ref_embeds[pads].abs().max() == 0
    assert float((flash - ref).norm() / ref.norm()) < 0.05
