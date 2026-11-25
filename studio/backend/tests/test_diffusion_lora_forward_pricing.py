# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LoRA forward pricing: mismatched adapter dtypes and edit reference tokens. CPU only."""

import types
import weakref

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

peft = pytest.importorskip("peft")
from peft import LoraConfig, inject_adapter_in_model  # noqa: E402

from core.inference import diffusion_lora_memory as lm  # noqa: E402
from core.inference import diffusion_memory as dm  # noqa: E402


class _Live(TorchDispatchMode):
    """Peak bytes of op outputs alive at once (views of inputs excluded)."""

    def __init__(self):
        super().__init__()
        self.live, self.cur, self.peak = {}, 0, 0

    def _free(self, key):
        self.cur -= self.live.pop(key, 0)

    def __torch_dispatch__(
        self,
        func,
        types_,
        args = (),
        kwargs = None,
    ):
        out = func(*args, **(kwargs or {}))
        inputs = {
            a.untyped_storage().data_ptr()
            for a in list(args) + list((kwargs or {}).values())
            if isinstance(a, torch.Tensor)
        }
        for t in out if isinstance(out, (tuple, list)) else (out,):
            if isinstance(t, torch.Tensor):
                key = t.untyped_storage().data_ptr()
                if key and key not in self.live and key not in inputs:
                    self.live[key] = t.untyped_storage().nbytes()
                    self.cur += self.live[key]
                    self.peak = max(self.peak, self.cur)
                    weakref.finalize(t, self._free, key)
        return out


def _peak(module, x):
    with torch.no_grad():
        mode = _Live()
        with mode:
            y = module(x)
        del y
        return mode.peak


@pytest.mark.parametrize("act", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dims", [(512, 512), (2048, 512), (512, 2048)])
def test_a_wider_adapter_dtype_is_priced_at_least_its_forward(act, dims):
    tokens = 256
    torch.manual_seed(0)
    holder = torch.nn.Module()
    holder.proj = torch.nn.Linear(dims[0], dims[1], dtype = act)
    x = torch.randn(tokens, dims[0], dtype = act)
    plain = _peak(holder.proj, x)
    inject_adapter_in_model(
        LoraConfig(r = 16, lora_alpha = 16, target_modules = ["proj"]), holder, adapter_name = "s"
    )
    for n, p in holder.named_parameters():
        if "lora_" in n:
            p.data = p.data.float()
    actual = (_peak(holder.proj, x) - plain) / tokens
    pipe = types.SimpleNamespace(
        transformer = holder, _unsloth_loras = (("s", "/l/s.safetensors", 1.0),)
    )
    pricing = lm.price_attached_loras(pipe, cfg = False, compute_bytes = 2)
    assert pricing is not None
    assert pricing.bytes_per_token >= actual, (pricing.bytes_per_token, actual)


def test_a_same_dtype_adapter_keeps_the_pr_price():
    holder = torch.nn.Module()
    holder.proj = torch.nn.Linear(512, 2048, dtype = torch.float16)
    inject_adapter_in_model(
        LoraConfig(r = 16, lora_alpha = 16, target_modules = ["proj"]), holder, adapter_name = "s"
    )
    for n, p in holder.named_parameters():
        if "lora_" in n:
            p.data = p.data.half()
    pipe = types.SimpleNamespace(
        transformer = holder, _unsloth_loras = (("s", "/l/s.safetensors", 1.0),)
    )
    assert (
        lm.price_attached_loras(pipe, cfg = False, compute_bytes = 2).bytes_per_token
        == (2 * 2048 + 16) * 2
    )


def test_reference_images_raise_a_priced_lora_request(monkeypatch):
    pipe = types.SimpleNamespace(
        _unsloth_loras = (("s", "/l/s.safetensors", 1.0),), _unsloth_measured_reserve = (6144,)
    )
    setattr(
        pipe,
        lm.PRICED_ATTR,
        lm.LoraPricing(
            signature = lm.lora_signature(pipe),
            bytes_per_token = 60_000,
            unet = False,
            branches = 2,
            mib = lm._mib(60_000, False, 2, None, None, 1),
        ),
    )
    monkeypatch.setattr(dm, "_measured_request_extra_mib", lambda *a, **k: 0)
    plain = dm.measured_request_extra_mib(pipe, width = 1024, height = 1024)
    two_refs = 2 * 1024 * 1024
    with_refs = dm.measured_request_extra_mib(
        pipe, width = 1024, height = 1024, condition_pixels = two_refs, reference_pixels = two_refs
    )
    assert plain == 0
    assert with_refs >= 60_000 * 8192 * 2 // (1 << 20) - 64, with_refs


@pytest.mark.parametrize("adapter_dtype", [torch.float16, torch.float32])
def test_an_embedding_adapter_is_priced_at_least_its_forward(adapter_dtype):
    # PEFT keeps an Embedding's factors in lora_embedding_A/B (Anima's llm_adapter.embed); a smaller
    # adapted Linear beside it must not set the price for the whole set
    tokens = 256
    torch.manual_seed(0)
    holder = torch.nn.Module()
    holder.embed = torch.nn.Embedding(128, 2048, dtype = torch.float16)
    holder.proj = torch.nn.Linear(16, 8, dtype = torch.float16)
    x = torch.randint(0, 128, (tokens,))
    plain = _peak(holder.embed, x)
    inject_adapter_in_model(
        LoraConfig(r = 4, lora_alpha = 4, target_modules = ["embed", "proj"]),
        holder,
        adapter_name = "s",
    )
    for n, p in holder.named_parameters():
        if "lora_" in n:
            p.data = p.data.to(adapter_dtype)
    assert "s" in holder.embed.lora_embedding_A
    actual = (_peak(holder.embed, x) - plain) / tokens
    assert actual > 0
    pipe = types.SimpleNamespace(
        transformer = holder, _unsloth_loras = (("s", "/l/s.safetensors", 1.0),)
    )
    pricing = lm.price_attached_loras(pipe, cfg = False, compute_bytes = 2, compute_dtype = torch.float16)
    assert pricing is not None
    assert pricing.bytes_per_token >= actual, (pricing.bytes_per_token, actual)
