# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""LoRA with trained bfloat16 weights (modules_to_save embed_tokens / lm_head): the trainer hook rounds them
stochastically; a LoRA run with only float32 adapters is left untouched. CPU, tiny random Llama."""

import importlib.util
from pathlib import Path

import pytest
import torch

from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py

pytest.importorskip("peft")
pytest.importorskip("transformers")

STEPS = 12
LR = 5e-6  # embedding-sized: far below bfloat16's spacing at these weights


@pytest.fixture
def sr(monkeypatch):
    monkeypatch.delenv("UNSLOTH_BF16_STOCHASTIC_ROUNDING", raising = False)
    path = Path(__file__).resolve().parents[1] / "unsloth/optimizers/bf16_stochastic_rounding.py"
    spec = importlib.util.spec_from_file_location("_test_bf16_sr_lora", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trainer_class(sr):
    class Trainer:
        def create_scheduler(
            self,
            num_training_steps,
            optimizer = None,
        ):
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)
            return self.lr_scheduler

    sr.patch_trainer_bf16_stochastic_rounding(Trainer)
    return Trainer


def _lora(
    modules_to_save,
    master = torch.bfloat16,
    device = "cpu",
):
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size = 512,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        tie_word_embeddings = False,
    )
    model = LlamaForCausalLM(config).to(device = device, dtype = torch.bfloat16)
    model._unsloth_full_finetuning = False  # what a FastLanguageModel / FastModel LoRA load stamps
    model = get_peft_model(
        model,
        LoraConfig(
            r = 4,
            lora_alpha = 8,
            target_modules = ["q_proj", "v_proj"],
            modules_to_save = modules_to_save,
        ),
    )
    if master == torch.float32:
        # float32 master copies computing in bfloat16: the expected value of the bfloat16 update.
        for name, module in model.named_modules():
            if name.endswith("modules_to_save.default"):
                module.weight.data = module.weight.data.float()
                if isinstance(module, torch.nn.Embedding):
                    module.forward = (
                        lambda m: lambda ids: torch.nn.functional.embedding(
                            ids, m.weight.to(torch.bfloat16)
                        )
                    )(module)
                else:
                    module.forward = (
                        lambda m: lambda x: torch.nn.functional.linear(x, m.weight.to(x.dtype))
                    )(module)
    return model


def _train(
    sr,
    model,
    steps = STEPS,
    optimizer = torch.optim.AdamW,
):
    trainer = _trainer_class(sr)()
    trainer.model, trainer.lr_scheduler = model, None
    trainer.args = type("Args", (), {"seed": 3407})()
    params = [p for p in model.parameters() if p.requires_grad]
    trainer.optimizer = optimizer(params, lr = LR, weight_decay = 0.0)
    trainer.create_scheduler(steps)
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 512, (4, 32), generator = g).to(next(model.parameters()).device)
    for _ in range(steps):
        model(input_ids = ids, labels = ids).loss.backward()
        trainer.optimizer.step()
        trainer.lr_scheduler.step()
        trainer.optimizer.zero_grad(set_to_none = True)
    return trainer.optimizer


def _saved(model):
    return {
        n: p.detach().float().clone()
        for n, p in model.named_parameters()
        if p.requires_grad and "modules_to_save" in n
    }


def _kept(
    sr,
    device = "cpu",
    optimizer = torch.optim.AdamW,
):
    """Fraction of the float32-master update each trained bfloat16 copy keeps, and the optimizer."""
    ref_model = _lora(["embed_tokens", "lm_head"], master = torch.float32, device = device)
    ref_start = _saved(ref_model)
    _train(sr, ref_model, optimizer = optimizer)
    ref = {n: (w - ref_start[n]).flatten().double() for n, w in _saved(ref_model).items()}

    model = _lora(["embed_tokens", "lm_head"], device = device)
    start = _saved(model)
    assert {p.dtype for p in model.parameters() if p.requires_grad} == {
        torch.bfloat16,
        torch.float32,
    }
    opt = _train(sr, model, optimizer = optimizer)
    kept = {}
    for name, weight in _saved(model).items():
        delta = (weight - start[name]).flatten().double()
        kept[name] = (delta @ ref[name] / (ref[name] @ ref[name])).item()
    return kept, opt


def test_lora_modules_to_save_keep_the_float32_update(sr):
    kept, optimizer = _kept(sr)
    # Round to nearest keeps under 40% of these updates (embed_tokens: almost none).
    assert all(abs(k - 1) < 0.15 for k in kept.values()), kept
    # Only the two trained bfloat16 copies are covered, never the float32 adapters.
    assert getattr(optimizer, "_unsloth_bf16_stochastic_rounding", 0) == 2


@pytest.mark.gpu
@pytest.mark.skipif(not has_real_cuda(), reason = "requires CUDA")
@pytest.mark.parametrize("name", ["torch", "AdamW8bit"])
def test_lora_modules_to_save_keep_the_float32_update_cuda(sr, name):
    """CUDA: torch AdamW takes the one-pass kernel, bitsandbytes 8-bit AdamW its fused kernel."""
    if name == "torch":
        optimizer, counter = torch.optim.AdamW, "_unsloth_sr_fused_steps"
    else:
        bnb = pytest.importorskip("bitsandbytes")
        optimizer, counter = bnb.optim.AdamW8bit, "_unsloth_sr_bnb_fused_steps"
    kept, opt = _kept(sr, device = "cuda", optimizer = optimizer)
    assert all(abs(k - 1) < 0.15 for k in kept.values()), kept
    assert getattr(opt, "_unsloth_bf16_stochastic_rounding", 0) == 2
    if sr._triton_ok(torch.device("cuda")):
        assert getattr(opt, counter, 0) == 2 * STEPS, vars(opt).get(counter)


def test_lora_float32_adapters_only_is_untouched(sr):
    model = _lora(None)
    assert {p.dtype for p in model.parameters() if p.requires_grad} == {torch.float32}
    optimizer = _train(sr, model, steps = 1)
    assert getattr(optimizer, "_unsloth_bf16_stochastic_rounding", None) is None
    assert "_unsloth_sr_index" not in vars(optimizer)  # no step wrapper installed


def test_lora_kill_switch(sr, monkeypatch):
    monkeypatch.setenv("UNSLOTH_BF16_STOCHASTIC_ROUNDING", "0")
    optimizer = _train(sr, _lora(["embed_tokens", "lm_head"]), steps = 1)
    assert getattr(optimizer, "_unsloth_bf16_stochastic_rounding", None) is None
    assert "_unsloth_sr_index" not in vars(optimizer)  # no step wrapper installed


def test_lora_adapters_step_bitwise_as_without_rounding(sr, monkeypatch):
    """The float32 adapters beside the covered weights take torch's own step, unchanged."""

    def adapters(env):
        monkeypatch.setenv("UNSLOTH_BF16_STOCHASTIC_ROUNDING", env)
        model = _lora(["embed_tokens", "lm_head"])
        _train(sr, model, steps = 1)
        return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}

    on, off = adapters("1"), adapters("0")
    assert on.keys() == off.keys() and on
    assert all(torch.equal(on[n], off[n]) for n in on)
