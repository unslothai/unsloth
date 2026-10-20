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
