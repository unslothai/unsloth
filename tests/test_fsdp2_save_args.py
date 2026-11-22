# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Which FSDP2 saves gather (unsloth/save.py `_fsdp2_collective_save`). No GPU work."""

import types

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")

from unsloth import save as unsloth_save  # noqa: E402
from unsloth.models import fsdp2_save  # noqa: E402


def _merged(
    self,
    save_directory,
    tokenizer = None,
    save_method = "merged_16bit",
    push_to_hub = False,
    token = None,
    is_main_process = True,
    state_dict = None,
):
    return ("wrote", state_dict)


class _Model:
    def original_model_save_pretrained(
        self,
        save_directory,
        is_main_process = True,
        state_dict = None,
        **kwargs,
    ):
        return ("wrote", state_dict)


def _forwarding(self, *args, **kwargs):
    return self.original_model_save_pretrained(*args, **kwargs)


@pytest.fixture
def no_gather(monkeypatch):
    monkeypatch.setattr(fsdp2_save, "fsdp2_units", lambda model: [object()])
    monkeypatch.setattr(fsdp2_save, "fsdp2_gather_active", lambda: False)

    def _refuse(*args, **kwargs):
        raise AssertionError("a caller's own state_dict must not start a collective gather")

    monkeypatch.setattr(fsdp2_save, "fsdp2_gathered", _refuse)


def test_a_positional_state_dict_is_passed_through(no_gather):
    sd = {"w": torch.zeros(1)}
    merged = unsloth_save._fsdp2_collective_save(lambda *a: False)(_merged)
    # (save_directory, tokenizer, save_method, push_to_hub, token, is_main_process, state_dict)
    assert merged(_Model(), "out", None, "merged_16bit", False, None, True, sd) == ("wrote", sd)
    assert merged(_Model(), "out", state_dict = sd) == ("wrote", sd)
    forwarding = unsloth_save._fsdp2_collective_save(lambda *a: False)(_forwarding)
    # PreTrainedModel.save_pretrained(save_directory, is_main_process, state_dict, ...)
    assert forwarding(_Model(), "out", True, sd) == ("wrote", sd)


def test_peft_saves_gather_the_frozen_tensors_peft_writes():
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        vocab_size = 64,
    )
    model = get_peft_model(LlamaForCausalLM(cfg), LoraConfig(r = 2, target_modules = ["q_proj"]))
    model.add_adapter("other", LoraConfig(r = 2, target_modules = ["q_proj"]))
    include = unsloth_save._fsdp2_peft_saved_frozen(model, {})
    names = {n for n, p in model.named_parameters() if not p.requires_grad and include(n, p)}
    assert any(".other." in n for n in names), names
    assert any("embed_tokens" in n for n in names) and any("lm_head" in n for n in names), names
    assert not any("q_proj.base_layer" in n for n in names), names
    # A biased output head: PEFT saves its bias with the weight.
    head = model.get_output_embeddings()
    head.bias = torch.nn.Parameter(torch.zeros(head.out_features), requires_grad = False)
    include = unsloth_save._fsdp2_peft_saved_frozen(model, {})
    assert any(n.endswith("lm_head.bias") for n, p in model.named_parameters() if include(n, p))
    include = unsloth_save._fsdp2_peft_saved_frozen(model, {"save_embedding_layers": False})
    names = {n for n, p in model.named_parameters() if include(n, p)}
    assert not any("embed_tokens" in n for n in names), names
    model.original_model_save_pretrained = model.save_pretrained
    arguments = unsloth_save._fsdp2_save_arguments(
        _forwarding, model, ("out", True, None, False), {}
    )
    assert arguments["save_embedding_layers"] is False


def test_peft_embedding_membership_survives_a_reshard():
    """The gather reshards before it walks the parameters, and a parameter an eval forward left
    unsharded is a different object from the DTensor the gather then sees: the embeddings must still be
    selected (by module, not by parameter identity), or they stay sharded in the saved adapter."""
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        vocab_size = 64,
    )
    model = get_peft_model(LlamaForCausalLM(cfg), LoraConfig(r = 2, target_modules = ["q_proj"]))
    include = unsloth_save._fsdp2_peft_saved_frozen(model, {})
    for getter in ("get_input_embeddings", "get_output_embeddings"):
        module = getattr(model, getter)()
        # What FSDPModule.reshard does: the module's parameter becomes a new object.
        module.weight = torch.nn.Parameter(module.weight.detach().clone(), requires_grad = False)
    names = {n for n, p in model.named_parameters() if include(n, p)}
    assert any("embed_tokens" in n for n in names), names
    assert any("lm_head" in n for n in names), names
    assert not any("q_proj.base_layer" in n for n in names), names


class _QLoraModel(torch.nn.Module):
    """One LoRA layer over a packed 4-bit base, one over a plain Linear."""

    def __init__(self):
        super().__init__()
        for name, quantized in (("q4", True), ("plain", False)):
            layer = torch.nn.Module()
            layer.base_layer = torch.nn.Linear(4, 4)
            if quantized:
                layer.base_layer.weight.quant_state = object()
            layer.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(4, 2, bias = False)})
            layer.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 4, bias = False)})
            setattr(self, name, layer)


@pytest.mark.parametrize(
    "entry", ["unsloth_generic_save_pretrained_merged", "unsloth_generic_push_to_hub_merged"]
)
@pytest.mark.parametrize(
    "method,offload",
    [
        ("merged_16bit", True),
        ("lora", True),
        ("merged_4bit", False),
        ("merged_4bit_forced", False),
        ("forced_merged_4bit", False),
    ],
)
def test_merged_4bit_keeps_gathered_weights_on_their_device(monkeypatch, entry, method, offload):
    # PEFT's 4-bit merge runs on the packed weight's GPU: CPU-offloaded LoRA fails there (save and Hub).
    import contextlib

    seen = {}

    @contextlib.contextmanager
    def _gathered(model, **kwargs):
        seen.update(kwargs)
        yield False

    monkeypatch.setattr(fsdp2_save, "fsdp2_units", lambda model: [object()])
    monkeypatch.setattr(fsdp2_save, "fsdp2_gather_active", lambda: False)
    monkeypatch.setattr(fsdp2_save, "fsdp2_gathered", _gathered)
    save = getattr(unsloth_save, entry)
    model = _QLoraModel()
    for call in (dict(save_method = method), {}):
        seen.clear()
        positional = () if call else (method,)
        assert save(model, "out", None, *positional, **call) is None
        got = seen.get("offload_to_cpu", True)
        if offload:
            assert got is True
            continue
        assert callable(got)
        params = dict(model.named_parameters())
        decide = lambda name: got(name, params[name])
        assert not decide("q4.lora_A.default.weight")
        assert not decide("q4.lora_B.default.weight")
        assert decide("plain.lora_A.default.weight")
        assert decide("q4.base_layer.bias")


def test_merged_4bit_keeps_only_merge_operands_on_the_gpu():
    # Embeddings and LoRA on unquantized layers still go to the CPU; only the adapters of layers
    # with a packed 4-bit base stay with it.
    class Quant(torch.nn.Parameter):
        pass

    def lora_layer(quantized):
        layer = torch.nn.Module()
        layer.base_layer = torch.nn.Linear(4, 4)
        if quantized:
            layer.base_layer.weight.quant_state = object()
        layer.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(4, 2, bias = False)})
        layer.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 4, bias = False)})
        return layer

    model = torch.nn.Module()
    model.embed = torch.nn.Embedding(8, 4)
    model.q4 = lora_layer(True)
    model.plain = lora_layer(False)
    offload = fsdp2_save.offload_all_but_4bit_merge_operands(model)
    assert not offload("q4.lora_A.default.weight", model.q4.lora_A["default"].weight)
    assert not offload("q4.lora_B.default.weight", model.q4.lora_B["default"].weight)
    assert offload("embed.weight", model.embed.weight)
    assert offload("plain.lora_A.default.weight", model.plain.lora_A["default"].weight)
    assert offload("plain.base_layer.weight", model.plain.base_layer.weight)
