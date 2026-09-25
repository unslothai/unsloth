# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Standalone evaluate() under fp16/bf16_full_eval must not change trainable dtypes, buffers or placement."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest
import torch

from real_accelerator import has_real_cuda


SOURCE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl.py"
HELPERS = (
    "_accelerator_indices",
    "_keep_in_fp32_names",
    "_model_spans_devices",
    "_full_eval_autocasts",
    "_cast_frozen_for_full_eval",
    "_wrap_full_eval_keeps_trainable_dtype",
)


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


def _load_helpers():
    tree = ast.parse(SOURCE_PATH.read_text(encoding = "utf-8"), filename = str(SOURCE_PATH))
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in HELPERS]
    missing = set(HELPERS) - {n.name for n in found}
    assert not missing, f"missing module-level defs in {SOURCE_PATH}: {sorted(missing)}"
    namespace = {"torch": torch, "logger": _Logger()}
    exec(compile(ast.Module(body = found, type_ignores = []), str(SOURCE_PATH), "exec"), namespace)
    return namespace


def _cuda_kernels_run(count):
    """Torch can launch a kernel on the card (not just see it)."""
    if not has_real_cuda() or torch.cuda.device_count() < count:
        return False
    try:
        for index in range(count):
            (torch.ones(2, device = f"cuda:{index}") * 2).sum().item()
        return True
    except Exception:
        return False


def test_wrapper_is_applied_to_every_generated_trainer():
    """Wrap sits at function level so every trainer gets it."""
    tree = ast.parse(SOURCE_PATH.read_text(encoding = "utf-8"))
    impl = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_patch_trl_rl_trainers_impl"
    )

    def calls_wrapper(node):
        return any(
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Name)
            and c.func.id == "_wrap_full_eval_keeps_trainable_dtype"
            for c in ast.walk(node)
        )

    top_level = [s for s in impl.body if calls_wrapper(s)]
    assert top_level, "_wrap_full_eval_keeps_trainable_dtype is never called"
    assert all(
        not isinstance(s, ast.If) for s in top_level
    ), "the full-eval wrap is gated on a trainer_file branch"


transformers = pytest.importorskip("transformers")


def _tiny_model(frozen_dtype = torch.float32):
    """Attention projections trainable in fp32; the rest frozen in ``frozen_dtype``."""
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size = 64,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 2,
        max_position_embeddings = 64,
        tie_word_embeddings = False,
    )
    model = transformers.LlamaForCausalLM(config).float()
    for name, param in model.named_parameters():
        trainable = "q_proj" in name or "v_proj" in name
        param.requires_grad_(trainable)
        if not trainable:
            param.data = param.data.to(frozen_dtype)
    return model


class _Rows(torch.utils.data.Dataset):
    def __init__(
        self,
        n = 4,
        length = 8,
    ):
        g = torch.Generator().manual_seed(1)
        self.ids = torch.randint(0, 64, (n, length), generator = g)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "labels": self.ids[i].clone()}


def _args(
    tmp_path,
    precision,
    autocast = True,
    **extra,
):
    kwargs = dict(
        output_dir = str(tmp_path),
        use_cpu = True,
        per_device_train_batch_size = 2,
        per_device_eval_batch_size = 2,
        max_steps = 2,
        learning_rate = 1e-3,
        report_to = [],
        save_strategy = "no",
        logging_steps = 1,
    )
    kwargs[f"{precision}_full_eval"] = True
    kwargs[precision] = autocast
    kwargs.update(extra)
    return transformers.TrainingArguments(**kwargs)


def _trainer(
    tmp_path,
    precision,
    wrap = True,
    model = None,
    helpers = None,
    **extra,
):
    cls = type("_T", (transformers.Trainer,), {})
    if wrap:
        (helpers or _load_helpers())["_wrap_full_eval_keeps_trainable_dtype"](cls)
    return cls(
        model = model if model is not None else _tiny_model(),
        args = _args(tmp_path, precision, **extra),
        train_dataset = _Rows(),
        eval_dataset = _Rows(),
    )


def _half(precision):
    return torch.float16 if precision == "fp16" else torch.bfloat16


def _state(model):
    return {
        name: (t.data_ptr(), t.dtype, t.detach().clone())
        for name, t in list(model.named_parameters()) + list(model.named_buffers())
        if t.is_floating_point() and (not isinstance(t, torch.nn.Parameter) or t.requires_grad)
    }


def _assert_untouched(model, before):
    after = _state(model)
    assert after.keys() == before.keys()
    for name, (ptr, dtype, value) in before.items():
        assert after[name][0] == ptr, f"{name} was reallocated"
        assert after[name][1] is dtype, name
        assert torch.equal(after[name][2], value), name


def test_premise_transformers_leaves_the_whole_model_cast(tmp_path):
    """Upstream still casts; if it casts back, the wrapper is redundant, not wrong."""
    trainer = _trainer(tmp_path, "bf16", wrap = False)
    trainer.evaluate()
    dtypes = {p.dtype for p in trainer.model.parameters() if p.requires_grad}
    if dtypes == {torch.float32}:
        pytest.skip("this Transformers no longer leaves trainable parameters cast after full eval")
    assert dtypes == {torch.bfloat16}


@pytest.mark.parametrize("precision", ["bf16", "fp16"])
@pytest.mark.parametrize("entry", ["evaluate", "predict"])
def test_standalone_eval_casts_only_frozen_weights(tmp_path, precision, entry):
    trainer = _trainer(tmp_path, precision)
    model = trainer.model
    before = _state(model)
    assert any(".inv_freq" in n or n.endswith("inv_freq") for n in before), "no rotary buffer"

    if entry == "evaluate":
        trainer.evaluate()
    else:
        trainer.predict(_Rows())

    _assert_untouched(model, before)
    # No fp16 autocast on CPU, so nothing may be cast there.
    frozen = _half(precision) if trainer.accelerator.native_amp else torch.float32
    for name, param in model.named_parameters():
        if not param.requires_grad:
            assert param.dtype is frozen, name
    assert getattr(trainer.args, f"{precision}_full_eval") is True


@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_standalone_eval_matches_an_eval_inside_train(tmp_path, precision):
    half = _half(precision)
    first = _trainer(tmp_path / "a", precision, model = _tiny_model(half))
    if not first.accelerator.native_amp:
        pytest.skip(f"no {precision} autocast on this device, so a half base cannot run here")
    standalone = first.evaluate()
    in_train = _trainer(tmp_path / "b", precision, model = _tiny_model(half))
    in_train.is_in_train = True
    try:
        reference = in_train.evaluate()
    finally:
        in_train.is_in_train = False
    assert standalone["eval_loss"] == reference["eval_loss"]


def test_without_autocast_nothing_is_cast(tmp_path):
    """bf16 full finetuning: bf16_full_eval without autocast, so nothing is cast."""
    trainer = _trainer(tmp_path, "bf16", autocast = False)
    dtypes = {n: p.dtype for n, p in trainer.model.named_parameters()}
    metrics = trainer.evaluate()
    assert "eval_loss" in metrics
    assert {n: p.dtype for n, p in trainer.model.named_parameters()} == dtypes


def test_evaluate_then_train_matches_train_alone(tmp_path):
    first = _trainer(tmp_path / "a", "bf16")
    first.evaluate()
    first.train()
    second = _trainer(tmp_path / "b", "bf16")
    for p in second.model.parameters():
        if not p.requires_grad:
            p.data = p.data.to(torch.bfloat16)
    second.train()
    for (name, p1), (_, p2) in zip(first.model.named_parameters(), second.model.named_parameters()):
        if p1.requires_grad:
            assert p1.dtype is torch.float32, name
            assert torch.equal(p1, p2), name


def test_evaluate_after_train_then_resume(tmp_path):
    trainer = _trainer(tmp_path, "bf16")
    trainer.train()
    trainer.evaluate()
    assert {p.dtype for p in trainer.model.parameters() if p.requires_grad} == {torch.float32}
    trainer.args.max_steps = 4
    trainer.train()
    assert {p.dtype for p in trainer.model.parameters() if p.requires_grad} == {torch.float32}


def test_eval_inside_train_is_passed_through(tmp_path):
    trainer = _trainer(tmp_path, "bf16")
    before = {n: p.dtype for n, p in trainer.model.named_parameters()}
    trainer.is_in_train = True
    try:
        trainer.evaluate()
    finally:
        trainer.is_in_train = False
    assert {n: p.dtype for n, p in trainer.model.named_parameters()} == before


def test_legacy_prediction_loop(tmp_path):
    if not hasattr(transformers.Trainer, "prediction_loop"):
        pytest.skip("this Transformers has no legacy prediction_loop")
    trainer = _trainer(tmp_path, "bf16", use_legacy_prediction_loop = True)
    before = _state(trainer.model)
    trainer.evaluate()
    _assert_untouched(trainer.model, before)


def test_no_full_eval_is_untouched(tmp_path):
    trainer = _trainer(tmp_path, "bf16", bf16_full_eval = False)
    trainer.evaluate()
    assert {p.dtype for p in trainer.model.parameters()} == {torch.float32}


def test_keep_in_fp32_modules_are_not_cast(tmp_path):
    model = _tiny_model()
    model._keep_in_fp32_modules_strict = ["lm_head"]
    trainer = _trainer(tmp_path, "bf16", model = model)
    trainer.evaluate()
    assert model.lm_head.weight.dtype is torch.float32
    assert model.model.embed_tokens.weight.dtype is torch.bfloat16


def test_split_model_is_not_moved():
    """Split model must not be collapsed; cpu/meta simulates a split."""
    helpers = _load_helpers()

    class _Trainer:
        args = transformers.TrainingArguments(output_dir = "unused", use_cpu = True, bf16 = True)

    model = _tiny_model()
    model.model.layers[1].to("meta")
    model.hf_device_map = {"model.layers.0": "cpu", "model.layers.1": "meta"}
    before = {n: p.device for n, p in model.named_parameters()}
    helpers["_cast_frozen_for_full_eval"](_Trainer(), model, torch.bfloat16, torch.device("cpu"))
    assert {n: p.device for n, p in model.named_parameters()} == before

    spans = helpers["_model_spans_devices"]
    split = _tiny_model()
    split.hf_device_map = {"model.layers.0": 0, "model.layers.1": 1}
    assert spans(split)
    offloaded = _tiny_model()
    offloaded.hf_device_map = {"model.layers.0": 0, "model.layers.1": "disk"}
    assert spans(offloaded)
    same = _tiny_model()
    same.hf_device_map = {"model.layers.0": 0, "model.layers.1": "cuda:0"}
    assert not spans(same)
    assert not spans(_tiny_model())


def test_a_failing_cast_still_evaluates(tmp_path):
    helpers = _load_helpers()

    def cast_then_fail(trainer, model, target_dtype, device):
        raise RuntimeError("boom")

    helpers["_cast_frozen_for_full_eval"] = cast_then_fail
    trainer = _trainer(tmp_path, "bf16", helpers = helpers)
    before = _state(trainer.model)
    assert "eval_loss" in trainer.evaluate()
    assert trainer.args.bf16_full_eval is True
    _assert_untouched(trainer.model, before)
    assert any("boom" in m for m in helpers["logger"].messages)


@pytest.mark.parametrize("where", ["cast", "eval"])
def test_keyboard_interrupt_leaves_flags_and_weights(tmp_path, where):
    """BaseException propagates with flags restored."""
    helpers = _load_helpers()
    if where == "cast":
        real_cast = helpers["_cast_frozen_for_full_eval"]

        def cast_then_interrupt(trainer, model, target_dtype, device):
            real_cast(trainer, model, target_dtype, device)
            raise KeyboardInterrupt

        helpers["_cast_frozen_for_full_eval"] = cast_then_interrupt
    trainer = _trainer(tmp_path, "fp16", helpers = helpers)
    if where == "eval":

        def interrupt(*args, **kwargs):
            assert trainer.args.fp16_full_eval is False
            raise KeyboardInterrupt

        trainer.compute_loss = interrupt
    before = _state(trainer.model)
    with pytest.raises(KeyboardInterrupt):
        trainer.evaluate()
    assert trainer.args.fp16_full_eval is True and trainer.args.bf16_full_eval is False
    _assert_untouched(trainer.model, before)


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_kernels_run(2), reason = "needs two CUDA devices torch can run on")
def test_split_model_evaluate_train_evaluate_on_two_gpus(tmp_path):
    accelerate = pytest.importorskip("accelerate")
    peft = pytest.importorskip("peft")

    model = _tiny_model().to(torch.bfloat16)
    for param in model.parameters():
        param.requires_grad_(False)
    device_map = {
        "model.embed_tokens": 1,
        "model.layers": 0,
        "model.norm": 0,
        "model.rotary_emb": 1,
        "lm_head": 1,
    }
    model = accelerate.dispatch_model(model, device_map = device_map)
    model.hf_device_map = device_map
    model = peft.get_peft_model(
        model, peft.LoraConfig(r = 4, lora_alpha = 4, target_modules = ["q_proj", "v_proj"])
    )
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    cls = type("_T", (transformers.Trainer,), {})
    _load_helpers()["_wrap_full_eval_keeps_trainable_dtype"](cls)
    args = transformers.TrainingArguments(
        output_dir = str(tmp_path),
        per_device_train_batch_size = 2,
        per_device_eval_batch_size = 2,
        max_steps = 2,
        report_to = [],
        bf16 = True,
        save_strategy = "no",
    )
    trainer = cls(model = model, args = args, train_dataset = _Rows(), eval_dataset = _Rows())
    trainer.args.bf16_full_eval = True
    devices = {p.device for p in model.parameters()}
    assert len(devices) == 2

    def trainable_dtypes():
        return {p.dtype for p in model.parameters() if p.requires_grad}

    trainer.evaluate()
    assert {p.device for p in model.parameters()} == devices
    assert trainable_dtypes() == {torch.float32}
    trainer.train()
    trainer.evaluate()
    assert {p.device for p in model.parameters()} == devices
    assert trainable_dtypes() == {torch.float32}


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_kernels_run(1), reason = "runs the generated SFTTrainer on a GPU")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_generated_sft_trainer_evaluate_then_train(tmp_path, dtype):
    """On main fp16 raised "Attempting to unscale FP16 gradients"."""
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    dtype = getattr(torch, dtype)
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("no bf16 on this card")
    model_name = os.environ.get(
        "UNSLOTH_TEST_TINY_QWEN3", "trl-internal-testing/tiny-Qwen3ForCausalLM"
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name, max_seq_length = 64, dtype = dtype, load_in_4bit = False
    )
    model = FastLanguageModel.get_peft_model(
        model, r = 8, lora_alpha = 8, target_modules = ["q_proj", "v_proj"], random_state = 0
    )
    texts = [f"Question {i}? Answer {i}." for i in range(8)]
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = Dataset.from_dict({"text": texts}),
        eval_dataset = Dataset.from_dict({"text": texts[:4]}),
        args = SFTConfig(
            output_dir = str(tmp_path),
            dataset_text_field = "text",
            max_steps = 2,
            per_device_train_batch_size = 2,
            per_device_eval_batch_size = 2,
            report_to = "none",
            save_strategy = "no",
            max_length = 64,
        ),
    )
    assert trainer.args.fp16_full_eval or trainer.args.bf16_full_eval
    before = {n: p.dtype for n, p in trainer.model.named_parameters() if p.requires_grad}
    assert set(before.values()) == {torch.float32}
    trainer.evaluate()
    after = {n: p.dtype for n, p in trainer.model.named_parameters() if p.requires_grad}
    assert after == before
    trainer.train()
    after = {n: p.dtype for n, p in trainer.model.named_parameters() if p.requires_grad}
    assert after == before
