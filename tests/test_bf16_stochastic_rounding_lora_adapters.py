# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""bfloat16 LoRA adapters (PEFT autocast_adapter_dtype = False, or a cast after get_peft_model): the trainer hook
rounds lora_A / lora_B, DoRA magnitude vectors and expert-stack adapters (target_parameters) stochastically, so a
small learning rate keeps the float32 adapters' update. CPU, tiny random models; CUDA variants for the fused
kernels."""

import importlib.util
from pathlib import Path

import pytest
import torch

from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py

peft = pytest.importorskip("peft")
pytest.importorskip("transformers")

STEPS = 32
LR = 1e-5  # far below bfloat16's spacing at lora_A's kaiming-uniform init (about 4e-4)


@pytest.fixture
def sr(monkeypatch):
    monkeypatch.delenv("UNSLOTH_BF16_STOCHASTIC_ROUNDING", raising = False)
    path = Path(__file__).resolve().parents[1] / "unsloth/optimizers/bf16_stochastic_rounding.py"
    spec = importlib.util.spec_from_file_location("_test_bf16_sr_lora_adapters", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Experts(torch.nn.Module):
    """A 3-D expert stack, as MoE models hold gate_up_proj / down_proj."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 64, 64) * 0.05)

    def forward(self, x):
        return torch.einsum("bsi,eoi->bso", x, self.weight) / 4


class _MoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(512, 64)
        self.experts = _Experts()
        self.head = torch.nn.Linear(64, 512, bias = False)

    def forward(self, input_ids, labels):
        logits = self.head(self.experts(self.embed(input_ids))).float()
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten())
        return type("Output", (), {"loss": loss})()


def _model(
    kind,
    adapters,
    device = "cpu",
):
    """kind: lora | dora | experts. adapters: the trained adapters' dtype (the base stays bfloat16)."""
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    if kind == "experts":
        if "target_parameters" not in LoraConfig.__dataclass_fields__:
            pytest.skip("PEFT without target_parameters")
        model = _MoE()
        config = LoraConfig(
            r = 4, lora_alpha = 8, target_modules = [], target_parameters = ["experts.weight"]
        )
    else:
        from transformers import LlamaConfig, LlamaForCausalLM
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size = 512,
                hidden_size = 64,
                intermediate_size = 128,
                num_hidden_layers = 2,
                num_attention_heads = 4,
                num_key_value_heads = 2,
                tie_word_embeddings = False,
            )
        )
        config = LoraConfig(
            r = 4,
            lora_alpha = 8,
            target_modules = ["q_proj", "v_proj"],
            use_dora = kind == "dora",
        )
    model = model.to(device = device, dtype = torch.bfloat16)
    model._unsloth_full_finetuning = False  # what a FastLanguageModel / FastModel LoRA load stamps
    model = get_peft_model(model, config, autocast_adapter_dtype = adapters == torch.float32)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.to(adapters)  # same init values in both dtypes
    assert {p.dtype for p in model.parameters() if p.requires_grad} == {adapters}
    return model


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


def _optimizer(model, optimizer = torch.optim.AdamW):
    return optimizer([p for p in model.parameters() if p.requires_grad], lr = LR, weight_decay = 0.0)


def _train(
    sr,
    model,
    steps = None,
    optimizer = torch.optim.AdamW,
    state = None,
    start = 0,
):
    """The Trainer's order: build the optimizer, create_scheduler (the hook), load a checkpoint, train."""
    steps = STEPS if steps is None else steps
    trainer = _trainer_class(sr)()
    trainer.model, trainer.lr_scheduler = model, None
    trainer.args = type("Args", (), {"seed": 3407})()
    trainer.optimizer = _optimizer(model, optimizer)
    trainer.create_scheduler(steps)
    if state is not None:
        trainer.optimizer.load_state_dict(state)
    device = next(model.parameters()).device
    for step in range(start, start + steps):
        g = torch.Generator().manual_seed(100 + step)
        ids = torch.randint(0, 512, (4, 32), generator = g).to(device)
        model(input_ids = ids, labels = ids).loss.backward()
        trainer.optimizer.step()
        trainer.lr_scheduler.step()
        trainer.optimizer.zero_grad(set_to_none = True)
    return trainer.optimizer


def _trained(model):
    return {n: p.detach().double().clone() for n, p in model.named_parameters() if p.requires_grad}


def _part(name):
    for part in ("lora_magnitude_vector", "lora_A", "lora_B"):
        if part in name:
            return part
    raise AssertionError(name)


_REFERENCE = {}


def _kept(
    sr,
    kind,
    device = "cpu",
    optimizer = torch.optim.AdamW,
):
    """Per adapter kind (lora_A / lora_B / lora_magnitude_vector): the projection of the bfloat16 adapters'
    update onto the float32 adapters' update (1 = the expected update kept), and the optimizer."""
    key = (kind, device, optimizer, STEPS, LR)
    if (
        key not in _REFERENCE
    ):  # float32 adapters cover nothing: the same with the rounding on or off
        ref = _model(kind, torch.float32, device)
        ref_start = _trained(ref)
        _train(sr, ref, optimizer = optimizer)
        _REFERENCE[key] = {n: w - ref_start[n] for n, w in _trained(ref).items()}
    ref_delta = _REFERENCE[key]

    model = _model(kind, torch.bfloat16, device)
    start = _trained(model)
    opt = _train(sr, model, optimizer = optimizer)
    delta = {n: w - start[n] for n, w in _trained(model).items()}
    kept = {}
    for part in sorted({_part(n) for n in delta}):
        names = [n for n in delta if _part(n) == part]
        x = torch.cat([delta[n].flatten() for n in names])
        r = torch.cat([ref_delta[n].flatten() for n in names])
        kept[part] = (x @ r / (r @ r)).item()
    return kept, opt, len(start)


def _close(kept, tol):
    """lora_A / lora_B within tol of the float32 update; DoRA's few magnitude elements are noisier."""
    return all(
        abs(k - 1) < (2 * tol if part == "lora_magnitude_vector" else tol)
        for part, k in kept.items()
    )


@pytest.mark.parametrize("kind", ["lora", "dora", "experts"])
def test_bf16_adapters_keep_the_float32_update(sr, kind):
    kept, optimizer, trained = _kept(sr, kind)
    assert _close(kept, 0.15), kept
    # Every trained adapter tensor is covered: A, B (and DoRA's magnitude vectors).
    assert getattr(optimizer, "_unsloth_bf16_stochastic_rounding", 0) == trained
    if kind == "dora":
        assert "lora_magnitude_vector" in kept


@pytest.mark.parametrize("kind", ["lora", "dora", "experts"])
def test_bf16_adapters_lose_the_update_without_rounding(sr, monkeypatch, kind):
    """The same harness with the rounding off: round to nearest drops most of lora_A's (and the magnitude
    vectors') update at this learning rate, so the test above can fail."""
    monkeypatch.setenv("UNSLOTH_BF16_STOCHASTIC_ROUNDING", "0")
    kept, optimizer, _ = _kept(sr, kind)
    assert getattr(optimizer, "_unsloth_bf16_stochastic_rounding", None) is None
    assert kept["lora_A"] < 0.3, kept
    if kind == "dora":
        assert kept["lora_magnitude_vector"] < 0.3, kept


def test_bf16_adapters_kill_switch_is_the_stock_step(sr, monkeypatch):
    def run(env, hook):
        monkeypatch.setenv("UNSLOTH_BF16_STOCHASTIC_ROUNDING", env)
        model = _model("lora", torch.bfloat16)
        if hook:
            opt = _train(sr, model, steps = 3)
        else:  # no trainer hook at all
            opt = _optimizer(model)
            for step in range(3):
                g = torch.Generator().manual_seed(100 + step)
                ids = torch.randint(0, 512, (4, 32), generator = g)
                model(input_ids = ids, labels = ids).loss.backward()
                opt.step()
                opt.zero_grad(set_to_none = True)
        return opt, _trained(model)

    killed, killed_weights = run("0", hook = True)
    assert getattr(killed, "_unsloth_bf16_stochastic_rounding", None) is None
    assert "_unsloth_sr_index" not in vars(killed)
    _, stock = run("0", hook = False)
    assert killed_weights.keys() == stock.keys()
    assert all(torch.equal(killed_weights[n], stock[n]) for n in stock)


def test_float32_adapters_are_untouched(sr):
    model = _model("lora", torch.float32)
    optimizer = _train(sr, model, steps = 1)
    assert getattr(optimizer, "_unsloth_bf16_stochastic_rounding", None) is None
    assert "_unsloth_sr_index" not in vars(optimizer)


def _resume_is_bitwise(
    sr,
    kind,
    device = "cpu",
    optimizer = torch.optim.AdamW,
    cut = 2,
    steps = 5,
):
    whole = _model(kind, torch.bfloat16, device)
    _train(sr, whole, steps = steps, optimizer = optimizer)

    half = _model(kind, torch.bfloat16, device)
    opt = _train(sr, half, steps = cut, optimizer = optimizer)
    weights, state = half.state_dict(), opt.state_dict()
    resumed = _model(kind, torch.bfloat16, device)
    resumed.load_state_dict(weights)
    opt = _train(sr, resumed, steps = steps - cut, optimizer = optimizer, state = state, start = cut)
    assert getattr(opt, "_unsloth_bf16_stochastic_rounding", 0) > 0
    a, b = _trained(whole), _trained(resumed)
    bad = [n for n in a if not torch.equal(a[n], b[n])]
    assert not bad, bad[:4]


@pytest.mark.parametrize("kind", ["lora", "dora"])
def test_bf16_adapters_resume_is_bitwise(sr, kind):
    # cut = 1: lora_A's first gradient is zero (lora_B starts at zero), the resumed run must still agree.
    for cut in (1, 2):
        _resume_is_bitwise(sr, kind, cut = cut)


def _cuda_optimizer(name):
    if name == "torch":
        return torch.optim.AdamW, "_unsloth_sr_fused_steps"
    bnb = pytest.importorskip("bitsandbytes")
    return bnb.optim.AdamW8bit, "_unsloth_sr_bnb_fused_steps"


@pytest.mark.gpu
@pytest.mark.skipif(not has_real_cuda(), reason = "requires CUDA")
@pytest.mark.parametrize("name", ["torch", "AdamW8bit"])
@pytest.mark.parametrize("kind", ["lora", "dora", "experts"])
def test_bf16_adapters_keep_the_float32_update_cuda(sr, name, kind):
    """CUDA: torch AdamW takes the one-pass kernel, bitsandbytes 8-bit AdamW its fused kernel."""
    optimizer, counter = _cuda_optimizer(name)
    kept, opt, trained = _kept(sr, kind, device = "cuda", optimizer = optimizer)
    assert getattr(opt, "_unsloth_bf16_stochastic_rounding", 0) == trained
    # Other draws than on CPU, and bitsandbytes quantizes the state of tensors from 4096 elements on.
    assert _close(kept, 0.25), kept
    if sr._triton_ok(torch.device("cuda")):
        assert getattr(opt, counter, 0) > 0, vars(opt).get(counter)


@pytest.mark.gpu
@pytest.mark.skipif(not has_real_cuda(), reason = "requires CUDA")
@pytest.mark.parametrize("name", ["torch", "AdamW8bit"])
def test_bf16_adapters_resume_is_bitwise_cuda(sr, name):
    optimizer, _ = _cuda_optimizer(name)
    _resume_is_bitwise(sr, "lora", device = "cuda", optimizer = optimizer)


def test_resolve_lora_dtype(sr):
    assert sr.resolve_lora_dtype(None) is None
    assert sr.resolve_lora_dtype(torch.float32) is None and sr.resolve_lora_dtype("float32") is None
    for value in (torch.bfloat16, "bfloat16", "bf16", "torch.bfloat16"):
        assert sr.resolve_lora_dtype(value) is torch.bfloat16
    for value in (torch.float16, "float16", "int8"):
        with pytest.raises(ValueError):
            sr.resolve_lora_dtype(value)


@pytest.mark.parametrize("kind", ["lora", "dora", "experts"])
def test_cast_lora_adapters_keeps_the_float32_update(sr, kind):
    """lora_dtype = torch.bfloat16: Unsloth's float32 adapters cast back in place, then covered and rounded."""
    model = _model(kind, torch.float32)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert sr.cast_lora_adapters_(model) == len(params)
    assert all(
        p is dict(model.named_parameters())[n] for n, p in params.items()
    )  # Parameter identity kept
    assert {p.dtype for p in params.values()} == {torch.bfloat16}
    assert sr.cast_lora_adapters_(model) == 0  # idempotent
    optimizer = _train(sr, model, steps = 1)
    assert getattr(optimizer, "_unsloth_bf16_stochastic_rounding", 0) == len(params)


def test_cast_lora_adapters_leaves_other_trained_weights(sr):
    """modules_to_save copies and frozen weights keep their dtype; only the adapters are cast."""
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size = 128,
            hidden_size = 32,
            intermediate_size = 64,
            num_hidden_layers = 1,
            num_attention_heads = 2,
            num_key_value_heads = 2,
        )
    )
    model = get_peft_model(
        model, LoraConfig(r = 2, target_modules = ["q_proj"], modules_to_save = ["lm_head"])
    )
    before = {n: p.dtype for n, p in model.named_parameters()}
    cast = sr.cast_lora_adapters_(model)
    after = {n: p.dtype for n, p in model.named_parameters()}
    changed = {n for n in before if before[n] != after[n]}
    assert cast == len(changed) == 2 and all(".lora_" in n for n in changed), changed


@pytest.mark.gpu
@pytest.mark.skipif(not has_real_cuda(), reason = "requires CUDA")
def test_fast_language_model_lora_dtype_bfloat16(monkeypatch, tmp_path):
    """FastLanguageModel.get_peft_model(lora_dtype = torch.bfloat16) on a tiny Qwen3: bfloat16 adapters, every one
    stepped by the fused kernel under the HF Trainer; the default keeps float32 adapters and an untouched step."""
    monkeypatch.delenv("UNSLOTH_BF16_STOCHASTIC_ROUNDING", raising = False)
    unsloth = pytest.importorskip("unsloth")
    from transformers import Trainer, TrainingArguments

    def run(**kw):
        model, _ = unsloth.FastLanguageModel.from_pretrained(
            "trl-internal-testing/tiny-Qwen3ForCausalLM",
            max_seq_length = 64,
            dtype = torch.bfloat16,
            load_in_4bit = False,
        )
        model = unsloth.FastLanguageModel.get_peft_model(
            model, r = 8, target_modules = ["q_proj", "v_proj", "down_proj"], **kw
        )
        adapters = [p for p in model.parameters() if p.requires_grad]
        ids = torch.randint(0, 100, (8, 32), generator = torch.Generator().manual_seed(0)).tolist()
        rows = [{"input_ids": x, "labels": x} for x in ids]
        trainer = Trainer(
            model = model,
            args = TrainingArguments(
                output_dir = str(tmp_path),
                per_device_train_batch_size = 4,
                max_steps = 3,
                learning_rate = 1e-4,
                bf16 = True,
                report_to = [],
                save_strategy = "no",
                logging_steps = 1,
                remove_unused_columns = False,
            ),
            train_dataset = rows,
            data_collator = lambda b: {k: torch.tensor([r[k] for r in b]) for k in b[0]},
        )
        trainer.train()
        opt = trainer.optimizer
        while hasattr(opt, "optimizer"):
            opt = opt.optimizer
        return {p.dtype for p in adapters}, len(adapters), opt

    dtypes, n, opt = run(lora_dtype = torch.bfloat16)
    assert dtypes == {torch.bfloat16}
    assert getattr(opt, "_unsloth_bf16_stochastic_rounding", 0) == n
    assert getattr(opt, "_unsloth_sr_fused_steps", 0) == 3 * n

    dtypes, _, opt = run()
    assert dtypes == {torch.float32}
    assert getattr(opt, "_unsloth_bf16_stochastic_rounding", None) is None

    with pytest.raises(ValueError, match = "lora_dtype"):
        run(lora_dtype = torch.float16)
