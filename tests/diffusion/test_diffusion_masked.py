# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Masked-diffusion recipes reproduce each family's published loss code on the same batch and seed."""

import os
import re
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

VOCAB = 97
MASK_ID = 90
EOS_ID = 3
PAD_ID = 5


class _Bidirectional(torch.nn.Module):
    """Every position sees the whole row, so a wrong mask or fill changes every logit."""

    def __init__(self, mask_token_id = MASK_ID):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(VOCAB, 16)
        self.mix = torch.nn.Linear(16, 16)
        self.head = torch.nn.Linear(16, VOCAB)
        self.config = SimpleNamespace(mask_token_id = mask_token_id, eos_token_id = EOS_ID)
        self.seen = []

    def forward(
        self,
        input_ids,
        attention_mask = None,
    ):
        self.seen.append((input_ids.clone(), attention_mask))
        h = self.embed(input_ids)
        h = h + self.mix(h.mean(dim = 1, keepdim = True))
        return SimpleNamespace(logits = self.head(torch.tanh(h)))


def _batch():
    # (prompt, response) lengths; right padded with a pad id distinct from EOS.
    shapes = [(4, 7), (6, 3), (3, 5)]
    length = max(p + r for p, r in shapes) + 2
    gen = torch.Generator().manual_seed(1)
    input_ids = torch.full((len(shapes), length), PAD_ID)
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    prompt_lengths = []
    for i, (p, r) in enumerate(shapes):
        row = torch.randint(10, 80, (p + r,), generator = gen)
        row[-1] = EOS_ID
        input_ids[i, : p + r] = row
        attention_mask[i, : p + r] = 1
        labels[i, p : p + r] = row[p:]
        prompt_lengths.append(p)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }, torch.tensor(prompt_lengths)


def _eos_filled(inputs):
    return inputs["input_ids"].masked_fill(inputs["attention_mask"] == 0, EOS_ID)


def _profile(name):
    from unsloth.models.diffusion_profiles import diffusion_profiles
    return {p.name: p for p in diffusion_profiles()}[name]


# LLaDA GUIDELINES.md, "SFT" section, verbatim apart from names.
def _llada_reference(
    model,
    input_ids,
    prompt_length,
    mask_id,
    eps = 1e-3,
):
    b, l = input_ids.shape
    t = torch.rand(b, device = input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)
    masked_indices = torch.rand((b, l), device = input_ids.device) < p_mask
    noisy_batch = torch.where(masked_indices, mask_id, input_ids)
    token_positions = torch.arange(noisy_batch.shape[1]).expand(
        noisy_batch.size(0), noisy_batch.size(1)
    )
    prompt_mask = token_positions < prompt_length.unsqueeze(1)
    noisy_batch[prompt_mask] = input_ids[prompt_mask]
    prompt_mask = prompt_mask.to(torch.int64)
    answer_lengths = torch.sum((1 - prompt_mask), dim = -1, keepdim = True)
    answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])
    masked_indices = noisy_batch == mask_id
    logits = model(input_ids = noisy_batch).logits
    token_loss = (
        F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction = "none")
        / p_mask[masked_indices]
    )
    return torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]


# Dream src/trainer/fsdp_sft_trainer.py _compute_loss_and_backward + src/diffllm/gen_utils.q_sample,
# with the default config (time_reweighting "original", no token reweighting, treat_eos_as_one off).
def _dream_reference(model, input_ids, loss_mask, mask_id):
    t = torch.rand((input_ids.shape[0],), dtype = torch.float)
    u = torch.rand_like(input_ids, dtype = torch.float)
    t_mask = (u < t[:, None]) & loss_mask
    masked_input_ids = input_ids.masked_fill(t_mask, mask_id)
    logits = model(input_ids = masked_input_ids).logits
    shift_logits = torch.cat([logits[:, 0:1], logits[:, :-1]], dim = 1)
    loss = F.cross_entropy(shift_logits.reshape(-1, VOCAB), input_ids.reshape(-1), reduction = "none")
    loss = loss.masked_fill(~t_mask.reshape(-1), 0)
    loss = loss * (1 / t[:, None].float().expand(input_ids.size())).reshape(-1)
    return torch.sum(loss) / torch.sum(t_mask)


# Nemotron-Labs-Diffusion modeling_nemotron_labs_diffusion.py forward_process + bidirectional loss branch;
# the (sum, count) pair it returns is divided by the trainer, as global_loss_avg does on one rank.
def _nemotron_reference(
    model,
    input_ids,
    loss_mask,
    mask_id,
    eps = 1e-3,
):
    b, l = input_ids.shape
    t = torch.rand(b)
    p_mask = ((1 - eps) * t + eps)[:, None].expand(-1, l)
    masked_indices = torch.rand((b, l)) < p_mask
    masked_indices[loss_mask == 0] = 0
    noisy = torch.where(masked_indices, mask_id, input_ids)
    logits = model(input_ids = noisy).logits
    token_loss = (
        F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction = "none")
        / p_mask[masked_indices]
    )
    return token_loss.sum() / masked_indices.sum()


def _ours(name, model, inputs, seed):
    torch.manual_seed(seed)
    loss, _, metrics = _profile(name).compute_loss(model, inputs, None)
    return loss, metrics


@pytest.mark.parametrize("name", ["llada", "llada_moe"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_llada_matches_guidelines_sft_loss(name, seed):
    inputs, prompt_lengths = _batch()
    model = _Bidirectional()
    torch.manual_seed(seed)
    reference = _llada_reference(model, _eos_filled(inputs), prompt_lengths, MASK_ID)
    loss, _ = _ours(name, model, inputs, seed)
    assert torch.allclose(loss, reference, atol = 0, rtol = 1e-6), (
        loss.item(),
        reference.item(),
        (loss - reference).item(),
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dream_matches_official_sft_loss(seed):
    inputs, _ = _batch()
    model = _Bidirectional()
    loss_mask = (inputs["labels"] != -100) | (inputs["attention_mask"] == 0)
    torch.manual_seed(seed)
    reference = _dream_reference(model, _eos_filled(inputs), loss_mask, MASK_ID)
    loss, _ = _ours("dream", model, inputs, seed)
    assert torch.allclose(loss, reference, atol = 0, rtol = 1e-6), (
        loss.item(),
        reference.item(),
        (loss - reference).item(),
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nemotron_matches_remote_forward_loss(seed):
    inputs, _ = _batch()
    model = _Bidirectional()
    loss_mask = (inputs["labels"] != -100) | (inputs["attention_mask"] == 0)
    torch.manual_seed(seed)
    reference = _nemotron_reference(model, _eos_filled(inputs), loss_mask, MASK_ID)
    loss, _ = _ours("nemotron_labs_diffusion", model, inputs, seed)
    assert torch.allclose(loss, reference, atol = 0, rtol = 1e-6), (
        loss.item(),
        reference.item(),
        (loss - reference).item(),
    )


@pytest.mark.parametrize("name", ["llada", "llada_moe", "dream", "nemotron_labs_diffusion"])
def test_prompt_stays_clean_and_padding_becomes_attended_eos(name):
    inputs, prompt_lengths = _batch()
    model = _Bidirectional()
    _ours(name, model, inputs, 0)
    noisy, attention_mask = model.seen[-1]
    assert attention_mask is None
    for i, p in enumerate(prompt_lengths.tolist()):
        assert torch.equal(noisy[i, :p], inputs["input_ids"][i, :p])
    padding = inputs["attention_mask"] == 0
    assert set(noisy[padding].tolist()) <= {EOS_ID, MASK_ID}
    assert PAD_ID not in noisy.tolist()


def test_unsupervised_batch_masks_every_attended_token_and_no_padding():
    inputs, _ = _batch()
    inputs.pop("labels")
    model = _Bidirectional()
    torch.manual_seed(0)
    _profile("llada").compute_loss(model, inputs, None)
    noisy, _ = model.seen[-1]
    assert not (noisy[inputs["attention_mask"] == 0] == MASK_ID).any()


class _DistributedLike(torch.nn.Module):
    """DDP / FSDP expose neither config nor Unsloth attributes; only forward reaches the module."""

    def __init__(self, inner):
        super().__init__()
        self.module = inner

    def forward(self, **kwargs):
        return self.module(**kwargs)


@pytest.mark.parametrize("name", ["llada", "dream", "nemotron_labs_diffusion"])
def test_loss_through_distributed_wrapper(name):
    inputs, _ = _batch()
    model = _Bidirectional()
    plain, _ = _ours(name, model, inputs, 0)
    wrapped, _ = _ours(name, _DistributedLike(model), inputs, 0)
    torch.testing.assert_close(wrapped, plain)


def test_mask_token_falls_back_to_family_default():
    inputs, _ = _batch()
    model = _Bidirectional(mask_token_id = None)
    _profile("llada_moe").prepare_model(model, None)
    assert model._unsloth_mask_token_id == 156895
    model = _Bidirectional(mask_token_id = None)
    _profile("llada").prepare_model(model, None)
    assert model._unsloth_mask_token_id == 126336


def test_all_masked_rows_absent_gives_zero_loss_with_grad():
    inputs, _ = _batch()
    inputs["labels"].fill_(-100)
    inputs["attention_mask"].fill_(1)
    model = _Bidirectional()
    loss, _ = _ours("dream", model, inputs, 0)
    assert loss.item() == 0.0 and loss.requires_grad


@pytest.mark.parametrize(
    "config, expected",
    [
        (dict(model_type = "llada", architectures = ["LLaDAModelLM"]), "llada"),
        (dict(model_type = "llada", architectures = ["LLaDAMoEModel"]), "llada_moe"),
        (dict(model_type = "Dream", architectures = ["DreamModel"]), "dream"),
        (
            dict(
                model_type = "nemotron_labs_diffusion", architectures = ["NemotronLabsDiffusionModel"]
            ),
            "nemotron_labs_diffusion",
        ),
        (dict(model_type = "nemotron", architectures = ["NemotronForCausalLM"]), None),
        (dict(model_type = "qwen3", architectures = ["Qwen3ForCausalLM"]), None),
    ],
)
def test_profile_routing(config, expected):
    from unsloth.models.diffusion_profiles import resolve_diffusion_profile
    profile = resolve_diffusion_profile(SimpleNamespace(**config))
    assert (profile.name if profile else None) == expected


def test_lora_targets_skip_output_head_router_and_experts():
    llada = _profile("llada").lora_target_modules
    assert re.fullmatch(llada, "model.transformer.blocks.3.ff_out")
    assert re.fullmatch(llada, "model.transformer.blocks.3.attn_out")
    assert not re.fullmatch(llada, "model.transformer.ff_out")
    moe = _profile("llada_moe").lora_target_modules
    assert re.fullmatch(moe, "model.layers.0.self_attn.o_proj")
    assert not re.fullmatch(moe, "model.layers.0.mlp.gate")
    assert not re.fullmatch(moe, "model.layers.0.mlp.experts.7.down_proj")


def test_rotary_buffers_rebuilt_from_rope_init_fn():
    from unsloth.models.diffusion_profiles import restore_rotary_buffers as _rebuild_rotary_buffers

    def init(config, device, **kwargs):
        return 1.0 / (config.rope_theta ** (torch.arange(0, 8, 2, dtype = torch.float32) / 8)), 1.0

    class Rotary(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(rope_theta = 10000.0)
            self.rope_init_fn = init
            self.rope_kwargs = {}
            self.register_buffer("inv_freq", torch.zeros(4), persistent = False)
            self.original_inv_freq = torch.empty(4, device = "meta")

    model = torch.nn.Module()
    model.config = SimpleNamespace()
    model.rotary = Rotary()
    expected = init(model.rotary.config, "cpu")[0]
    assert _rebuild_rotary_buffers(model) == 1
    assert torch.equal(model.rotary.inv_freq, expected)
    assert torch.equal(model.rotary.original_inv_freq, expected)
    assert _rebuild_rotary_buffers(model) == 0 and torch.equal(model.rotary.inv_freq, expected)


def test_dream_generate_defaults_to_filled_model_generation_config():
    from unsloth.models.diffusion_masked import _default_diffusion_generate_config

    seen = {}

    class Dream:
        config = SimpleNamespace(
            mask_token_id = 151666, eos_token_id = 151643, pad_token_id = 151643, bos_token_id = 151643
        )
        generation_config = SimpleNamespace(
            mask_token_id = None, eos_token_id = 151643, pad_token_id = None, bos_token_id = None
        )

        def diffusion_generate(
            self,
            inputs = None,
            generation_config = None,
            **kwargs,
        ):
            seen["config"] = generation_config
            return inputs

    model = Dream()
    _default_diffusion_generate_config(model)
    _default_diffusion_generate_config(model)
    model.diffusion_generate("x", steps = 4)
    assert seen["config"].mask_token_id == 151666 and seen["config"].pad_token_id == 151643
    assert model.generation_config.mask_token_id is None
    explicit = SimpleNamespace(mask_token_id = 7)
    model.diffusion_generate("x", generation_config = explicit)
    assert seen["config"] is explicit


def test_mask_builder_signature_exposed_from_wrapper_closure():
    import inspect

    from unsloth.models import diffusion_masked

    def original(
        config,
        inputs_embeds,
        attention_mask,
        past_key_values,
        position_ids = None,
    ):
        return None

    def make(signature):
        def return_attention_mask(*args, **kwargs):
            return signature

        return return_attention_mask

    wrapper = make(inspect.signature(original))
    fake = SimpleNamespace(create_causal_mask = wrapper, create_sliding_window_causal_mask = None)
    import sys

    real = sys.modules.get("transformers.masking_utils")
    sys.modules["transformers.masking_utils"] = fake
    try:
        import transformers

        saved = getattr(transformers, "masking_utils", None)
        transformers.masking_utils = fake
        diffusion_masked._expose_mask_signatures()
    finally:
        sys.modules["transformers.masking_utils"] = real
        transformers.masking_utils = saved
    assert list(inspect.signature(wrapper).parameters)[:2] == ["config", "inputs_embeds"]


_TINY_TRAIN = r"""
import math, os, sys, torch
import unsloth
from datasets import Dataset
from unsloth import DiffusionConfig, DiffusionTrainer, FastModel

path, out = sys.argv[1], sys.argv[2]
device = os.environ.get("TINY_DEVICE", "cpu")
model, processor = FastModel.from_pretrained(path, trust_remote_code = True, dtype = torch.float32, device_map = {"": device})
assert getattr(model, "_unsloth_slow_diffusion", False), type(model)
tokenizer = getattr(processor, "tokenizer", processor)
for module in model.modules():
    if hasattr(module, "rope_init_fn") and isinstance(getattr(module, "inv_freq", None), torch.Tensor):
        assert module.inv_freq.abs().sum() > 0, "RoPE inv_freq left uninitialised"
model = FastModel.get_peft_model(model, r = 16, lora_alpha = 32, use_gradient_checkpointing = True)
before = {k: v.detach().clone() for k, v in model.named_parameters() if "lora_B" in k}
rows = [{"prompt": f"Question {i}: what is {i} plus {i}?", "completion": f" The answer is {2 * i}."} for i in range(4)] * 16
args = DiffusionConfig(
    output_dir = out, max_steps = 20, per_device_train_batch_size = 4, learning_rate = 2e-2, logging_steps = 1,
    report_to = [], save_strategy = "no", seed = 3407, use_cpu = device == "cpu", max_length = 64,
    bf16 = False, fp16 = False, dataset_num_proc = 1, gradient_checkpointing = True,
)
trainer = DiffusionTrainer(model = model, args = args, train_dataset = Dataset.from_list(rows), processing_class = tokenizer)
trainer.train()
losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
assert len(losses) == 20 and all(math.isfinite(x) for x in losses), losses
# The 1 / t weight makes the objective itself too noisy over 20 steps; unweighted masked CE is the trend.
ce = [h["masked_ce"] for h in trainer.state.log_history if "masked_ce" in h]
first, last = sum(ce[:5]) / 5, sum(ce[-5:]) / 5
assert last < first, ce
after = {k: v for k, v in model.named_parameters() if k in before}
assert any(not torch.equal(before[k], after[k].detach()) for k in before)
model.save_pretrained(os.path.join(out, "adapter"))
tokenizer.save_pretrained(os.path.join(out, "adapter"))
reloaded, _ = FastModel.from_pretrained(os.path.join(out, "adapter"), trust_remote_code = True, dtype = torch.float32, device_map = {"": device})
saved = {k.replace(".default", ""): v for k, v in model.named_parameters() if "lora_B" in k}
loaded = {k.replace(".default", ""): v for k, v in reloaded.named_parameters() if "lora_B" in k}
assert saved.keys() == loaded.keys() and saved, (list(saved)[:3], list(loaded)[:3])
assert all(torch.equal(saved[k].detach().cpu(), loaded[k].detach().cpu()) for k in saved)
print("TINY_OK", trainer.diffusion_profile.name, round(first, 4), round(last, 4))
"""


# Real remote code, tiny random weights: UNSLOTH_DIFFUSION_TINY_DIR/<name> built from the published config and code.
_TINY = os.environ.get("UNSLOTH_DIFFUSION_TINY_DIR")


@pytest.mark.skipif(not _TINY, reason = "set UNSLOTH_DIFFUSION_TINY_DIR to tiny remote-code copies")
@pytest.mark.parametrize(
    "name",
    [
        "LLaDA-8B-Instruct",
        "LLaDA-MoE-7B-A1B-Instruct",
        "Dream-v0-Instruct-7B",
        "Nemotron-Labs-Diffusion-3B",
    ],
)
def test_tiny_remote_checkpoint_trains_and_reloads(name, tmp_path):
    import subprocess
    import sys

    # One process per family: LLaDA's remote config registers "llada" globally, shadowing LLaDA-MoE's.
    out = subprocess.run(
        [sys.executable, "-c", _TINY_TRAIN, os.path.join(_TINY, name), str(tmp_path)],
        capture_output = True,
        text = True,
        timeout = 900,
    )
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    assert "TINY_OK" in out.stdout
