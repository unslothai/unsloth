# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Which exports run under FSDP2 (unsloth/save.py). save_pretrained and save_pretrained_merged
already gather the shards to rank 0; push_to_hub_merged and push_to_hub did not, so every rank
merged and uploaded its own shards. They now gather like the saves. GGUF and torchao exports
rebuild or quantize the whole model in-process and are refused under FSDP2 with a way to get the
same file. No GPU work: FSDP2 is faked by fsdp2_units, the gather by a recording context manager."""

import contextlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from unsloth import save as unsloth_save  # noqa: E402
from unsloth.models import fsdp2_save  # noqa: E402


@pytest.fixture
def fake_fsdp2(monkeypatch):
    calls = []

    @contextlib.contextmanager
    def gathered(
        model,
        trainable_only = False,
        **kwargs,
    ):
        calls.append(trainable_only)
        yield False  # a non-writer rank: the export must not run here

    monkeypatch.setattr(fsdp2_save, "fsdp2_units", lambda model: [object()])
    monkeypatch.setattr(fsdp2_save, "fsdp2_gather_active", lambda: False)
    monkeypatch.setattr(fsdp2_save, "fsdp2_gathered", gathered)
    return calls


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        vocab_size = 32,
    )
    return LlamaForCausalLM(cfg)


def test_push_to_hub_merged_gathers_like_save_pretrained_merged(fake_fsdp2):
    model = _tiny_llama()
    # Not rank 0: gathered, nothing merged or uploaded, None returned.
    assert unsloth_save.unsloth_generic_push_to_hub_merged(model, "user/repo", None) is None
    assert (
        unsloth_save.unsloth_generic_push_to_hub_merged(
            model, "user/repo", None, save_method = "lora"
        )
        is None
    )
    # A plain (non-PEFT) model gathers everything, as save_pretrained_merged does.
    assert fake_fsdp2 == [False, False]


def test_push_to_hub_gathers(fake_fsdp2):
    model = unsloth_save.patch_saving_functions(_tiny_llama())
    assert model.push_to_hub("user/repo") is None
    assert fake_fsdp2 == [False]


@pytest.mark.parametrize(
    "method, call",
    [
        ("save_pretrained_gguf", lambda m: unsloth_save.unsloth_save_pretrained_gguf(m, "out")),
        ("push_to_hub_gguf", lambda m: unsloth_save.unsloth_push_to_hub_gguf(m, "user/repo")),
        (
            "save_pretrained_torchao",
            lambda m: unsloth_save.unsloth_save_pretrained_torchao(m, "out"),
        ),
    ],
)
def test_gguf_and_torchao_are_refused_under_fsdp2(fake_fsdp2, method, call):
    with pytest.raises(
        RuntimeError, match = rf"`{method}` does not support a model sharded by FSDP2"
    ):
        call(_tiny_llama())
    assert fake_fsdp2 == []


def test_refusal_passes_through_without_fsdp2(monkeypatch):
    monkeypatch.setattr(fsdp2_save, "fsdp2_units", lambda model: [])
    export = unsloth_save._fsdp2_refuses_save("save_pretrained_gguf")(lambda self, x: ("ran", x))
    assert export(object(), 1) == ("ran", 1)


def test_full_and_merged_saves_stage_4bit_on_the_cpu(monkeypatch):
    """save_pretrained / push_to_hub write the packed 4-bit bytes as they are, merged_16bit reads its
    base from the 16-bit checkpoint, and merged_4bit brings one packed layer at a time back to its
    device to merge, so every one of them gathers the packed bytes onto the CPU."""
    seen = []

    @contextlib.contextmanager
    def gathered(model, **kwargs):
        seen.append(kwargs.get("offload_4bit", False))
        yield False

    monkeypatch.setattr(fsdp2_save, "fsdp2_units", lambda model: [object()])
    monkeypatch.setattr(fsdp2_save, "fsdp2_gather_active", lambda: False)
    monkeypatch.setattr(fsdp2_save, "fsdp2_gathered", gathered)
    model = unsloth_save.patch_saving_functions(_tiny_llama())
    assert model.save_pretrained("out") is None
    assert model.push_to_hub("user/repo") is None
    assert unsloth_save.unsloth_generic_save_pretrained_merged(model, "out", None) is None
    assert (
        unsloth_save.unsloth_generic_save_pretrained_merged(
            model, "out", None, save_method = "merged_4bit"
        )
        is None
    )
    assert unsloth_save.unsloth_generic_push_to_hub_merged(model, "user/repo", None) is None
    assert seen == [True, True, True, True, True], seen


def test_streamed_merge_failure_clears_every_merged_layer(monkeypatch):
    # A later layer's merge raising must not leave earlier layers claiming a merge the gather undoes.
    import torch

    class _Layer(torch.nn.Module):
        def __init__(self, fail):
            super().__init__()
            self.base_layer = torch.nn.Linear(2, 2, bias = False)
            self.merged_adapters = []
            self.fail = fail

        def merge(self):
            if self.fail:
                raise RuntimeError("merge failed")
            self.merged_adapters.append("default")

    model = torch.nn.Sequential(_Layer(False), _Layer(False), _Layer(True))
    staged = {id(layer.base_layer.weight): "cpu" for layer in model}
    monkeypatch.setattr(fsdp2_save, "_STAGED_4BIT", [staged])
    monkeypatch.setattr(fsdp2_save, "_restage_4bit", lambda weight, device: weight)
    with pytest.raises(RuntimeError, match = "merge failed"):
        fsdp2_save.merge_4bit_lora_layer_by_layer(model)
    assert [layer.merged_adapters for layer in model] == [[], [], []]


def test_streamed_merge_restage_failure_clears_merge_flags(monkeypatch):
    # A failing restage after a successful merge must not leave the layer claiming the merge.
    import torch

    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = torch.nn.Linear(2, 2, bias = False)
            self.base_layer.weight.quant_state = object()
            self.merged_adapters = []

        def merge(self):
            self.merged_adapters.append("default")

    model = torch.nn.Sequential(_Layer(), _Layer())
    staged = {id(layer.base_layer.weight): "cuda" for layer in model}
    monkeypatch.setattr(fsdp2_save, "_STAGED_4BIT", [staged])
    calls = []

    def restage(weight, device):
        calls.append(device)
        if device == "cpu" and len(calls) == 4:
            raise RuntimeError("host out of memory")
        return weight

    monkeypatch.setattr(fsdp2_save, "_restage_4bit", restage)
    with pytest.raises(RuntimeError, match = "host out of memory"):
        fsdp2_save.merge_4bit_lora_layer_by_layer(model)
    assert [layer.merged_adapters for layer in model] == [[], []]


def test_streamed_merge_gpu_restage_failure_clears_earlier_merge_flags(monkeypatch):
    # The GPU restage of a later layer failing must also clear the layers merged before it.
    import torch

    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = torch.nn.Linear(2, 2, bias = False)
            self.base_layer.weight.quant_state = object()
            self.merged_adapters = []

        def merge(self):
            self.merged_adapters.append("default")

    model = torch.nn.Sequential(_Layer(), _Layer())
    staged = {id(layer.base_layer.weight): "cuda" for layer in model}
    monkeypatch.setattr(fsdp2_save, "_STAGED_4BIT", [staged])
    calls = []

    def restage(weight, device):
        calls.append(device)
        if device == "cuda" and len(calls) == 3:
            raise RuntimeError("CUDA out of memory")
        return weight

    monkeypatch.setattr(fsdp2_save, "_restage_4bit", restage)
    second = model[1].base_layer.weight
    with pytest.raises(RuntimeError, match = "CUDA out of memory"):
        fsdp2_save.merge_4bit_lora_layer_by_layer(model)
    assert [layer.merged_adapters for layer in model] == [[], []]
    assert model[1].base_layer._parameters["weight"] is second
