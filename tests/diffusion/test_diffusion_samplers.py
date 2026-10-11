# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FastLanguageModel routes text-diffusion models to the diffusion PEFT / mode helpers, and generate_text decodes
with each family's sampler."""

import importlib.util
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

# The LLaDA sampler is re-implemented; check it against the official one when its generate.py is available.
_LLADA_GENERATE = os.environ.get("UNSLOTH_LLADA_GENERATE_PY")
_TINY = os.environ.get("UNSLOTH_DIFFUSION_TINY_DIR")

VOCAB, MASK_ID, EOS_ID = 61, 60, 2


class _TinyMaskPredictor(torch.nn.Module):
    """Bidirectional toy predictor: every logit depends on the whole row."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(VOCAB, 16)
        self.mix = torch.nn.Linear(16, 16)
        self.head = torch.nn.Linear(16, VOCAB)
        self.config = SimpleNamespace(mask_token_id = MASK_ID, eos_token_id = EOS_ID)
        self.calls = 0

    @property
    def device(self):
        return self.head.weight.device

    def forward(
        self,
        input_ids,
        attention_mask = None,
    ):
        self.calls += 1
        h = self.embed(input_ids)
        h = h + self.mix(h.mean(dim = 1, keepdim = True))
        logits = self.head(torch.tanh(h))
        # Never predict [MASK] itself, as a trained model would not.
        logits[..., MASK_ID] = -1e4
        return SimpleNamespace(logits = logits)


def _load_official_generate():
    spec = importlib.util.spec_from_file_location("llada_generate", _LLADA_GENERATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate


@pytest.mark.skipif(
    not _LLADA_GENERATE, reason = "set UNSLOTH_LLADA_GENERATE_PY to LLaDA's generate.py"
)
@pytest.mark.parametrize("temperature", [0.0, 0.7])
@pytest.mark.parametrize("remasking", ["low_confidence", "random"])
def test_llada_sampler_matches_official(temperature, remasking):
    from unsloth.models.diffusion_samplers import _llada_sample

    official = _load_official_generate()
    model = _TinyMaskPredictor()
    prompt = torch.tensor([[1, 7, 9, 11, 13]])
    torch.manual_seed(1)
    expected = official(
        model,
        prompt,
        steps = 8,
        gen_length = 16,
        block_length = 8,
        temperature = temperature,
        remasking = remasking,
        mask_id = MASK_ID,
    )[:, prompt.shape[1] :]
    torch.manual_seed(1)
    got = _llada_sample(
        model,
        None,
        prompt,
        None,
        max_new_tokens = 16,
        steps = 8,
        block_length = 8,
        temperature = temperature,
        mask_id = MASK_ID,
        eos_ids = [],
        remasking = remasking,
    )
    assert torch.equal(got, expected), (got, expected)


def test_llada_sampler_stops_after_eos_block():
    from unsloth.models.diffusion_samplers import _llada_sample

    model = _TinyMaskPredictor()
    prompt = torch.tensor([[1, 7, 9]])
    full = _llada_sample(
        model,
        None,
        prompt,
        None,
        max_new_tokens = 32,
        steps = 32,
        block_length = 8,
        temperature = 0.0,
        mask_id = MASK_ID,
        eos_ids = [],
        eos_early_stop = False,
    )
    tokens = full[0].tolist()
    eos = next(t for t in tokens[:8] if t != MASK_ID)
    calls = model.calls
    early = _llada_sample(
        model,
        None,
        prompt,
        None,
        max_new_tokens = 32,
        steps = 32,
        block_length = 8,
        temperature = 0.0,
        mask_id = MASK_ID,
        eos_ids = [eos],
    )
    # The first block already holds the "EOS", so only its 8 steps ran and its tokens are unchanged.
    assert model.calls - calls == 8
    assert torch.equal(early, full[:, :8])


def test_fast_language_model_routes_diffusion_helpers(monkeypatch):
    from unsloth import FastLanguageModel
    from unsloth.models import diffusion as diffusion_module
    from unsloth.models.llama import FastLlamaModel

    seen = {}
    monkeypatch.setattr(
        diffusion_module.FastDiffusionModel,
        "get_peft_model",
        staticmethod(lambda model, **kwargs: seen.setdefault("peft", kwargs) and model),
    )
    monkeypatch.setattr(
        diffusion_module.FastDiffusionModel,
        "for_training",
        staticmethod(
            lambda model, use_gradient_checkpointing = True: seen.setdefault(
                "train", use_gradient_checkpointing
            )
            or model
        ),
    )
    monkeypatch.setattr(
        diffusion_module.FastDiffusionModel,
        "for_inference",
        staticmethod(lambda model: seen.setdefault("infer", True) and model),
    )
    model = torch.nn.Linear(2, 2)
    model._unsloth_slow_diffusion = True
    model._unsloth_gradient_checkpointing = False
    # Positional r and target_modules keep FastLanguageModel's meaning; omitted ones stay omitted.
    FastLanguageModel.get_peft_model(
        model, 8, ["q_proj"], lora_alpha = 16, use_rslora = True, offload_layers = None
    )
    assert seen["peft"] == {
        "r": 8,
        "target_modules": ["q_proj"],
        "lora_alpha": 16,
        "use_rslora": True,
        "offload_layers": None,
    }
    FastLanguageModel.for_training(model)
    assert seen["train"] is False
    FastLanguageModel.for_inference(model)
    assert seen["infer"] is True
    # Signatures still read as FastLlamaModel's (introspecting callers, docs).
    import inspect

    assert inspect.signature(FastLanguageModel.get_peft_model) == inspect.signature(
        FastLlamaModel.get_peft_model
    )


def test_fast_language_model_autoregressive_path_unchanged(monkeypatch):
    from unsloth import FastLanguageModel
    from unsloth.models.llama import FastLlamaModel

    calls = []
    monkeypatch.setattr(
        FastLlamaModel,
        "get_peft_model",
        staticmethod(lambda *a, **k: calls.append(("peft", a, k)) or "peft"),
    )
    monkeypatch.setattr(
        FastLlamaModel,
        "for_training",
        staticmethod(lambda *a, **k: calls.append(("train", a, k)) or "train"),
    )
    monkeypatch.setattr(
        FastLlamaModel,
        "for_inference",
        staticmethod(lambda *a, **k: calls.append(("infer", a, k)) or "infer"),
    )
    model = torch.nn.Linear(2, 2)
    assert FastLanguageModel.get_peft_model(model, 4, target_modules = ["x"]) == "peft"
    assert FastLanguageModel.for_training(model, True) == "train"
    assert FastLanguageModel.for_inference(model) == "infer"
    assert calls == [
        ("peft", (model, 4), {"target_modules": ["x"]}),
        ("train", (model, True), {}),
        ("infer", (model,), {}),
    ]


@pytest.mark.parametrize("bad", [{"use_dora": True}, {"loftq_config": {"loftq_bits": 4}}])
def test_unsupported_lora_options_refused(bad):
    from unsloth.models.diffusion import FastDiffusionModel
    model = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match = "not supported for text-diffusion"):
        FastDiffusionModel.get_peft_model(model, r = 4, **bad)


def test_diffusion_profile_name_and_flag():
    from unsloth.models.diffusion import is_diffusion_model
    from unsloth.models.diffusion_samplers import diffusion_profile_name

    model = torch.nn.Linear(2, 2)
    assert not is_diffusion_model(model) and not is_diffusion_model(None)
    # Autoregressive Nemotron shares no profile; Nemotron-Labs-Diffusion matches by architecture.
    model.config = SimpleNamespace(model_type = "nemotron", architectures = ["NemotronForCausalLM"])
    assert not is_diffusion_model(model)
    model.config = SimpleNamespace(
        model_type = "nemotron", architectures = ["NemotronLabsDiffusionModel"]
    )
    assert is_diffusion_model(model)
    assert diffusion_profile_name(model) == "nemotron_labs_diffusion"
    del model.config
    model._unsloth_slow_diffusion = True
    model._unsloth_diffusion_profile = "llada"
    assert is_diffusion_model(model)
    assert diffusion_profile_name(model) == "llada"


def test_generate_text_cuts_at_eos_and_budget(monkeypatch):
    from unsloth.models import diffusion_samplers

    class _Tok:
        eos_token_id = EOS_ID
        mask_token_id = None

        def __call__(
            self,
            text,
            return_tensors = None,
            add_special_tokens = True,
        ):
            return {
                "input_ids": torch.tensor([[1, 7]]),
                "attention_mask": torch.ones(1, 2, dtype = torch.long),
            }

        def decode(
            self,
            tokens,
            skip_special_tokens = True,
        ):
            return " ".join(map(str, tokens))

    model = _TinyMaskPredictor()
    model._unsloth_diffusion_profile = "fake"
    produced = torch.tensor([[11, MASK_ID, 12, 13, EOS_ID, 14, 15]])
    seen = {}

    def fake(base, tokenizer, input_ids, attention_mask, **kwargs):
        seen.update(kwargs)
        return produced

    monkeypatch.setitem(diffusion_samplers._SAMPLERS, "fake", fake)
    text = diffusion_samplers.generate_text(
        model, _Tok(), prompt = "hi", max_new_tokens = 32, steps = 4, alg = "x"
    )
    # Stops at the first EOS, drops leftover [MASK] ids.
    assert text == "11 12 13"
    assert (
        seen["mask_id"] == MASK_ID
        and seen["eos_ids"] == [EOS_ID]
        and seen["alg"] == "x"
        and seen["steps"] == 4
    )
    # max_new_tokens caps what is decoded even when a sampler returns more.
    assert diffusion_samplers.generate_text(model, _Tok(), prompt = "hi", max_new_tokens = 2) == "11"
    with pytest.raises(ValueError, match = "no diffusion sampler"):
        model._unsloth_diffusion_profile = "missing"
        diffusion_samplers.generate_text(model, _Tok(), prompt = "hi")


_PAD_SCRIPT = r"""
import os, sys
import torch
from unsloth import FastModel

path = sys.argv[1]
model, processor = FastModel.from_pretrained(path, trust_remote_code = True, dtype = torch.float32, device_map = {"": os.environ.get("TINY_DEVICE", "cpu")}, load_in_4bit = False)
tok = getattr(processor, "tokenizer", processor)
assert tok.pad_token is not None and tok.pad_token == tok.eos_token, (tok.pad_token, tok.eos_token)
assert tok.pad_token_id != getattr(model, "_unsloth_mask_token_id", None)
assert model.config.pad_token_id == tok.pad_token_id
assert getattr(model.generation_config, "pad_token_id", tok.pad_token_id) == tok.pad_token_id
batch = tok.pad([{"input_ids": [1, 2, 3]}, {"input_ids": [4]}], return_tensors = "pt")
assert batch["input_ids"].shape == (2, 3)
print("PAD_OK", tok.pad_token)
"""


def test_tiny_nemotron_gets_eos_pad_token():
    path = os.path.join(_TINY or "", "Nemotron-Labs-Diffusion-3B")
    if not _TINY or not os.path.isdir(path):
        pytest.skip("set UNSLOTH_DIFFUSION_TINY_DIR to tiny remote-code copies")
    out = subprocess.run(
        [sys.executable, "-c", _PAD_SCRIPT, path], capture_output = True, text = True, timeout = 900
    )
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-4000:]
    assert "PAD_OK" in out.stdout


_TINY_SCRIPT = r"""
import os, sys
import torch
from unsloth import FastLanguageModel
from unsloth.models.diffusion import DiffusionGenerationCancelled, FastDiffusionModel, is_diffusion_model

path = sys.argv[1]
device = os.environ.get("TINY_DEVICE", "cpu")
model, processor = FastLanguageModel.from_pretrained(path, trust_remote_code = True, dtype = torch.float32, device_map = {"": device}, load_in_4bit = False)
assert is_diffusion_model(model), type(model)
# Studio's text path: FastLanguageModel.get_peft_model with its usual knobs.
model = FastLanguageModel.get_peft_model(model, r = 8, lora_alpha = 16, lora_dropout = 0.0, bias = "none", use_gradient_checkpointing = True, random_state = 3407, use_rslora = True, loftq_config = None, use_dora = False)
config = model.peft_config["default"]
assert config.use_rslora is True
assert sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora_" in n) > 0


def checkpointing_on(m):
    return any(getattr(x, "gradient_checkpointing", False) is True or getattr(x, "activation_checkpointing_strategy", None) is not None for x in m.modules())


assert checkpointing_on(model), "get_peft_model(use_gradient_checkpointing = True) did not enable it"
FastLanguageModel.for_training(model, use_gradient_checkpointing = False)
assert model.training and not checkpointing_on(model)
FastLanguageModel.for_training(model, use_gradient_checkpointing = True)
assert checkpointing_on(model)
FastLanguageModel.for_inference(model)
assert not model.training and not checkpointing_on(model)

messages = [{"role": "user", "content": "What is 2 + 2?"}]
first = FastDiffusionModel.generate_text(model, processor, messages = messages, max_new_tokens = 16, seed = 0)
second = FastDiffusionModel.generate_text(model, processor, messages = messages, max_new_tokens = 16, seed = 0)
assert isinstance(first, str) and first == second, (first, second)
plain = FastDiffusionModel.generate_text(model, processor, prompt = "Hello", max_new_tokens = 8, seed = 0)
assert isinstance(plain, str)

calls = {"n": 0}


def stop():
    calls["n"] += 1
    return calls["n"] > 1


try:
    FastDiffusionModel.generate_text(model, processor, messages = messages, max_new_tokens = 64, stop_callback = stop, seed = 0)
    raise AssertionError("stop_callback did not cancel")
except DiffusionGenerationCancelled:
    pass
assert calls["n"] == 2, calls
assert not model.training
print("SAMPLER_OK", repr(first[:40]))
"""

_FAMILIES = [
    "LLaDA-8B-Instruct",
    "LLaDA-MoE-7B-A1B-Instruct",
    "Dream-v0-Instruct-7B",
    "Nemotron-Labs-Diffusion-3B",
    "LLaDA2-mini",
    "SDAR-1.7B-Chat",
    "trl-internal-testing/tiny-DiffusionGemmaForBlockDiffusion",
]


@pytest.mark.parametrize("name", _FAMILIES)
def test_tiny_family_routes_and_generates(name):
    if "/" in name:
        path = name
    elif _TINY and os.path.isdir(os.path.join(_TINY, name)):
        path = os.path.join(_TINY, name)
    else:
        pytest.skip("set UNSLOTH_DIFFUSION_TINY_DIR to tiny remote-code copies")
    # One process per family: LLaDA's remote config registers "llada" globally, shadowing LLaDA-MoE's.
    out = subprocess.run(
        [sys.executable, "-c", _TINY_SCRIPT, path], capture_output = True, text = True, timeout = 900
    )
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-4000:]
    assert "SAMPLER_OK" in out.stdout
