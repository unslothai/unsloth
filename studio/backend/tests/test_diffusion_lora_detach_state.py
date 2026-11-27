# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A LoRA attached then removed leaves the denoiser exactly as it was, requires_grad included.

PEFT freezes every base weight it injects into and unloading leaves them frozen. torch.matmul reads requires_grad even
under no_grad: HiDream-I1 splits its joint attention output into image and text tokens, and the non-contiguous halves
fold into one mm while the weight requires grad, else run as bmm. Those kernels differ in their low bits on CUDA, so
every plain HiDream render after a LoRA differed from the one before it."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("safetensors")
diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "HiDreamImagePipeline"):
    pytest.skip("this diffusers has no HiDream-I1 pipeline", allow_module_level = True)

from safetensors.torch import save_file  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from core.inference import diffusion_lora_files as lf  # noqa: E402

BATCH = 2  # the CFG pair: a batch of one stays foldable without a copy, so its kernel never changes


def _pipe():
    torch.manual_seed(0)
    transformer = diffusers.HiDreamImageTransformer2DModel(
        patch_size = 2,
        in_channels = 4,
        out_channels = 4,
        num_layers = 1,
        num_single_layers = 1,
        attention_head_dim = 8,
        num_attention_heads = 4,
        caption_channels = [8, 4],
        text_emb_dim = 8,
        num_routed_experts = 4,
        num_activated_experts = 2,
        axes_dims_rope = (4, 2, 2),
        max_resolution = (32, 32),
        llama_layers = (0, 1),
    ).eval()
    components = dict.fromkeys(
        (
            "scheduler",
            "vae",
            "text_encoder",
            "tokenizer",
            "text_encoder_2",
            "tokenizer_2",
            "text_encoder_3",
            "tokenizer_3",
            "text_encoder_4",
            "tokenizer_4",
        )
    )
    return diffusers.HiDreamImagePipeline(transformer = transformer, **components)


def _lora_file(tmp_path, transformer):
    g = torch.Generator().manual_seed(3)
    sd = {}
    for name, module in transformer.named_modules():
        if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in (
            "to_q",
            "to_k",
            "to_v",
            "to_out",
        ):
            sd[f"transformer.{name}.lora_A.weight"] = torch.randn(
                2, module.in_features, generator = g
            )
            sd[f"transformer.{name}.lora_B.weight"] = torch.randn(
                module.out_features, 2, generator = g
            )
    path = tmp_path / "hidream.safetensors"
    save_file(sd, str(path))
    return str(path)


class _MatmulKernels(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(
        self,
        func,
        types,
        args = (),
        kwargs = None,
    ):
        if func.__name__.split(".")[0] in ("mm", "bmm", "addmm", "baddbmm"):
            self.ops.append(func.__name__)
        return func(*args, **(kwargs or {}))


def _forward(transformer):
    g = torch.Generator().manual_seed(11)
    inputs = dict(
        hidden_states = torch.randn(BATCH, 4, 8, 8, generator = g),
        timesteps = torch.tensor([500] * BATCH),
        encoder_hidden_states_t5 = torch.randn(BATCH, 6, 8, generator = g),
        encoder_hidden_states_llama3 = torch.randn(2, BATCH, 6, 4, generator = g),
        pooled_embeds = torch.randn(BATCH, 8, generator = g),
        return_dict = False,
    )
    with torch.no_grad(), _MatmulKernels() as kernels:
        out = transformer(**inputs)[0]
    return out, kernels.ops


def _requires_grad(transformer):
    return {name: p.requires_grad for name, p in transformer.named_parameters()}


def test_attach_then_detach_restores_requires_grad_and_the_kernels(tmp_path):
    pipe = _pipe()
    path = _lora_file(tmp_path, pipe.transformer)
    flags = _requires_grad(pipe.transformer)
    out, kernels = _forward(pipe.transformer)

    lf.load_adapter(pipe, path, "a", "hidream-i1")
    pipe.set_adapters(["a"], adapter_weights = [1.0])
    with_lora, _ = _forward(pipe.transformer)
    assert not torch.equal(with_lora, out)  # the adapter really ran
    lf.unload_adapters(pipe)

    assert _requires_grad(pipe.transformer) == flags
    after, kernels_after = _forward(pipe.transformer)
    # The matmul each Linear dispatches to: bmm would replace mm on the attention outputs if the weights stayed frozen.
    assert kernels_after == kernels
    assert torch.equal(after, out)


def test_two_adapters_then_detach_restore_the_flags_from_before_the_first(tmp_path):
    pipe = _pipe()
    path = _lora_file(tmp_path, pipe.transformer)
    flags = _requires_grad(pipe.transformer)
    lf.load_adapter(pipe, path, "a", "hidream-i1")
    lf.load_adapter(pipe, path, "b", "hidream-i1")  # the base weights are already frozen here
    lf.unload_adapters(pipe)
    assert _requires_grad(pipe.transformer) == flags
    # A later set starts from the restored flags, not a stale record.
    lf.load_adapter(pipe, path, "c", "hidream-i1")
    lf.unload_adapters(pipe)
    assert _requires_grad(pipe.transformer) == flags


def test_frozen_weights_stay_frozen_after_detach(tmp_path):
    pipe = _pipe()
    path = _lora_file(tmp_path, pipe.transformer)
    pipe.transformer.requires_grad_(False)
    lf.load_adapter(pipe, path, "a", "hidream-i1")
    lf.unload_adapters(pipe)
    assert not any(_requires_grad(pipe.transformer).values())


def test_the_record_does_not_keep_replaced_weights_alive(tmp_path):
    import gc
    import weakref

    pipe = _pipe()
    path = _lora_file(tmp_path, pipe.transformer)
    lf.load_adapter(pipe, path, "a", "hidream-i1")
    # A load-time bake then quantizes: the dense weight is swapped for a new Parameter and must be freed.
    block = pipe.transformer.double_stream_blocks[0].block.attn1.to_q.base_layer
    old = weakref.ref(block.weight)
    block.weight = torch.nn.Parameter(block.weight.detach().clone())
    gc.collect()
    assert old() is None
    lf.unload_adapters(pipe)
