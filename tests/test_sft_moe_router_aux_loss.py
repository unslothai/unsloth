# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The Unsloth SFT compute_loss keeps the MoE router aux loss that stock TRL adds; ast-loaded, CPU-only.

TRL 1.7 to 1.17 wrote `output_router_logits = True` into the model config, so the model forward added
`router_aux_loss_coef * aux_loss` itself. TRL 1.18 stopped writing the config and requests the router logits per
forward inside its own compute_loss, which the Unsloth nll path does not call.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import tempfile
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

SOURCE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"
COEF = 0.9  # gpt-oss's router_aux_loss_coef


def _generated_compute_loss_source():
    tree = ast.parse(SOURCE_PATH.read_text(encoding = "utf-8"))
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "sft_trainer_compute_loss"
    ]
    namespace = {"inspect": inspect, "RL_FUNCTIONS": None}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(SOURCE_PATH), "exec"), namespace)
    return namespace["sft_trainer_compute_loss"](
        "compute_loss", "def compute_loss(self):\n    pass\n"
    )


def _generated_class(base):
    # Compiled into a class body, as in the generated trainer, so the zero-argument super() resolves.
    namespace = {"torch": torch, "Base": base}
    source = _generated_compute_loss_source()
    exec("class Generated(Base):\n" + textwrap.indent(textwrap.dedent(source), "    "), namespace)
    return namespace["Generated"]


class _RecordingBase:
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs = False,
        num_items_in_batch = None,
    ):
        self.seen = dict(inputs)
        return torch.tensor(0.0)


def _recording_trainer(**attrs):
    trainer = _generated_class(_RecordingBase)()
    trainer.args = SimpleNamespace(loss_type = "nll", average_tokens_across_devices = False)
    trainer.compute_loss_func = None
    for k, v in attrs.items():
        setattr(trainer, k, v)
    return trainer


def test_router_logits_requested_only_when_trl_enables_the_aux_loss():
    # TRL >= 1.7 with an MoE model and a non-zero coefficient.
    trainer = _recording_trainer(aux_loss_enabled = True, router_aux_loss_coef = COEF)
    inputs = {"input_ids": torch.ones(1, 2, dtype = torch.long)}
    trainer.compute_loss(None, inputs)
    assert trainer.seen["output_router_logits"] is True
    assert "output_router_logits" not in inputs, "the caller's batch must not be mutated"
    # Dense models, router_aux_loss_coef = 0, and TRL without the attribute (0.22): batch passes through unchanged.
    for attrs in ({"aux_loss_enabled": False}, {}):
        trainer = _recording_trainer(**attrs)
        trainer.compute_loss(None, dict(inputs))
        assert "output_router_logits" not in trainer.seen


def _tiny_moe(kind):
    import transformers

    common = dict(
        vocab_size = 64,
        hidden_size = 16,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 8,
        num_experts_per_tok = 2,
        router_aux_loss_coef = COEF,
        output_router_logits = False,
        pad_token_id = 1,
        eos_token_id = 2,
        bos_token_id = None,
    )
    if kind == "qwen3_moe":
        config = transformers.Qwen3MoeConfig(
            intermediate_size = 32, moe_intermediate_size = 16, num_experts = 4, **common
        )
        model = transformers.Qwen3MoeForCausalLM(config)
    else:
        if not hasattr(transformers, "GptOssConfig"):
            pytest.skip("transformers without gpt-oss")
        config = transformers.GptOssConfig(
            intermediate_size = 16,
            num_local_experts = 4,
            layer_types = ["full_attention", "full_attention"],
            **common,
        )
        config._attn_implementation = "eager"
        model = transformers.GptOssForCausalLM(config)
    # fp32: the grouped-mm experts of transformers 5.x before 5.23 have no fp64 kernel. The oracle reduces in fp64.
    return model.to(torch.float32)


def _tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    vocab = {f"t{i}": i for i in range(64)}
    return PreTrainedTokenizerFast(
        tokenizer_object = Tokenizer(WordLevel(vocab, unk_token = "t0")),
        unk_token = "t0",
        pad_token = "t1",
        eos_token = "t2",
    )


def _oracle_loss(model, batch, num_experts, top_k):
    """fp64 CE over the shifted labels / N + coef * Switch load-balancing loss over every layer's router logits."""
    out = model(
        input_ids = batch["input_ids"],
        attention_mask = batch["attention_mask"],
        output_router_logits = True,
        use_cache = False,
    )
    logits = out.logits[:, :-1].double()
    labels = batch["labels"][:, 1:]
    mask = labels != -100
    logp = logits.log_softmax(-1).gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    ce = -(logp * mask).sum() / mask.sum()
    gate = torch.cat([r.double() for r in out.router_logits], dim = 0)
    probs = gate.softmax(-1)
    selected = probs.topk(top_k, dim = -1).indices
    tokens_per_expert = F.one_hot(selected, num_experts).double().mean(0)
    aux = num_experts * (tokens_per_expert * probs.mean(0).unsqueeze(0)).sum()
    return ce + COEF * aux, ce, aux


def _grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("kind", ["qwen3_moe", "gpt_oss"])
def test_nll_loss_and_grads_match_stock_trl_and_fp64_oracle(kind):
    if importlib.util.find_spec("trl") is None:
        pytest.skip("trl not installed")
    from trl import SFTConfig, SFTTrainer

    if "router_aux_loss_coef" not in SFTConfig.__dataclass_fields__:
        pytest.skip("TRL < 1.7 has no SFT router aux loss")
    from datasets import Dataset

    torch.manual_seed(0)
    model = _tiny_moe(kind)
    model.train()
    text_config = model.config.get_text_config()
    num_experts = getattr(text_config, "num_experts", None) or text_config.num_local_experts
    gen = torch.Generator().manual_seed(1)
    input_ids = torch.randint(3, 64, (2, 9), generator = gen)
    labels = input_ids.clone()
    labels[:, :3] = -100
    batch = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": labels}
    n_items = (labels[:, 1:] != -100).sum()

    with tempfile.TemporaryDirectory() as out:
        args = SFTConfig(
            output_dir = out,
            report_to = "none",
            use_cpu = True,
            per_device_train_batch_size = 2,
            max_steps = 1,
            bf16 = False,
            fp16 = False,
        )
        trainer = SFTTrainer(
            model = model,
            args = args,
            train_dataset = Dataset.from_dict({"input_ids": input_ids.tolist()}),
            processing_class = _tokenizer(),
        )
    assert trainer.aux_loss_enabled and trainer.router_aux_loss_coef == COEF
    trainer.current_gradient_accumulation_steps = 1
    # TRL >= 1.18 keeps the loaded router config (it is saved with the model); Unsloth must not write it either.
    config_flag = text_config.output_router_logits

    def run(loss_fn):
        model.zero_grad(set_to_none = True)
        loss = loss_fn()
        loss.backward()
        return loss.detach(), _grads(model)

    try:
        trl_loss, trl_grads = run(
            lambda: SFTTrainer.compute_loss(trainer, model, dict(batch), num_items_in_batch = n_items)
        )
    except RuntimeError as e:
        # TRL 1.15's fused LM head launches a Triton kernel even on CPU tensors when Triton is installed.
        if "active drivers" not in str(e) or torch.cuda.is_available():
            raise
        trl_loss = trl_grads = None
    # The generated UnslothSFTTrainer subclasses TRL's base class, so super().compute_loss is Trainer.compute_loss.
    trainer.__class__ = _generated_class(type(trainer).__mro__[1])
    unsloth_loss, unsloth_grads = run(
        lambda: trainer.compute_loss(model, dict(batch), num_items_in_batch = n_items)
    )
    assert text_config.output_router_logits == config_flag
    top_k = text_config.num_experts_per_tok
    with torch.no_grad():
        _, ce, aux = _oracle_loss(model, batch, num_experts, top_k)
    oracle_loss, oracle_grads = run(lambda: _oracle_loss(model, batch, num_experts, top_k)[0])

    assert COEF * aux.item() > 0.1, "the aux term is too small for the test to tell the arms apart"
    if trl_loss is not None:
        assert trl_loss.item() == pytest.approx(oracle_loss.item(), rel = 1e-5)
    assert unsloth_loss.item() == pytest.approx(
        oracle_loss.item(), rel = 1e-5
    ), f"Unsloth {unsloth_loss.item():.6f} vs CE {ce.item():.6f} + {COEF} * aux {aux.item():.6f}"
    assert set(unsloth_grads) == set(oracle_grads)
    for name in oracle_grads:
        if trl_grads is not None:
            torch.testing.assert_close(trl_grads[name], oracle_grads[name], rtol = 1e-4, atol = 1e-6)
        torch.testing.assert_close(unsloth_grads[name], oracle_grads[name], rtol = 1e-4, atol = 1e-6)
