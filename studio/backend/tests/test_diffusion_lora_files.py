# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ComfyUI / kohya LoRA and LyCORIS LoKR files: key normalization and the Kronecker adapter."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_lora_files as lf  # noqa: E402


def test_alpha_folds_into_lora_b_and_down_up_rename():
    a, b = torch.randn(4, 8), torch.randn(6, 4)
    out = lf.normalize_lora(
        {
            "diffusion_model.blocks.0.attn.wq.lora_down.weight": a,
            "diffusion_model.blocks.0.attn.wq.lora_up.weight": b,
            "diffusion_model.blocks.0.attn.wq.alpha": torch.tensor(2.0),
            "diffusion_model.blocks.0.attn.wk.alpha": torch.tensor(
                1.0
            ),  # an orphan alpha is dropped
        },
        "krea-2",
    )
    assert sorted(out) == [
        "diffusion_model.blocks.0.attn.wq.lora_A.weight",
        "diffusion_model.blocks.0.attn.wq.lora_B.weight",
    ]
    assert torch.allclose(out["diffusion_model.blocks.0.attn.wq.lora_B.weight"], b * 0.5)


def test_anima_embedding_pair_becomes_a_peft_embedding_lora():
    a, b = (
        torch.randn(3, 5),
        torch.randn(11, 3),
    )  # ComfyUI: delta = B @ A on the [vocab 11, dim 5] table
    out = lf.normalize_lora(
        {
            "diffusion_model.llm_adapter.embed.lora_A.weight": a,
            "diffusion_model.llm_adapter.embed.lora_B.weight": b,
        },
        "anima",
    )
    ea = out["diffusion_model.llm_adapter.embed.lora_embedding_A"]
    eb = out["diffusion_model.llm_adapter.embed.lora_embedding_B"]
    # PEFT's embedding delta is (B' @ A').T
    assert torch.allclose((eb @ ea).T, b @ a)


def test_moss_fused_qkv_lora_splits_rows_and_shares_a():
    a, b = torch.randn(4, 6), torch.randn(9, 4)
    out = lf.normalize_lora(
        {
            "diffusion_model.layers.0.proj.qkv.lora_A.weight": a,
            "diffusion_model.layers.0.proj.qkv.lora_B.weight": b,
        },
        "moss-image",
    )
    for i, name in enumerate(("q", "k", "v")):
        assert torch.equal(out[f"diffusion_model.layers.0.proj.{name}.lora_A.weight"], a)
        assert torch.equal(
            out[f"diffusion_model.layers.0.proj.{name}.lora_B.weight"], b[i * 3 : (i + 1) * 3]
        )


def test_split_lokr_keeps_alpha_with_its_module():
    w1, w2 = torch.randn(2, 2), torch.randn(3, 3)
    lokr, rest = lf.split_lokr(
        {
            "m.lokr_w1": w1,
            "m.lokr_w2": w2,
            "m.alpha": torch.tensor(4.0),
            "n.lora_A.weight": torch.randn(1, 6),
            "n.alpha": torch.tensor(1.0),
        }
    )
    assert set(lokr["m"]) == {"lokr_w1", "lokr_w2", "alpha"}
    assert set(rest) == {"n.lora_A.weight", "n.alpha"}


def test_kron_factors_scale_like_comfyui():
    w1, w2 = torch.randn(2, 2), torch.randn(3, 3)
    # Full factors: alpha is ignored (scale 1).
    assert lf._kron_factors({"lokr_w1": w1, "lokr_w2": w2, "alpha": torch.tensor(8.0)})[2] == 1.0
    # A low-rank factor: alpha / its rank.
    _, _, scale = lf._kron_factors(
        {
            "lokr_w1": w1,
            "lokr_w2_a": torch.randn(3, 2),
            "lokr_w2_b": torch.randn(2, 3),
            "alpha": torch.tensor(1.0),
        }
    )
    assert scale == 0.5


class _Block(torch.nn.Module):
    def __init__(self, out_features = 36):
        super().__init__()
        self.proj = torch.nn.ModuleDict({"q": torch.nn.Linear(24, out_features, bias = False)})

    def forward(self, x):
        return self.proj["q"](x)


def test_kron_adapter_matches_the_dense_delta_and_restores_the_layer():
    torch.manual_seed(0)
    block = _Block().double()
    base = block.proj["q"]
    w1, w2 = torch.randn(4, 3).double(), torch.randn(9, 8).double()
    first = lf.KronAdapter(block, base, w1, w2, 1.0)
    second = lf.KronAdapter(block, block.proj["q"], w1, w2, 1.0)
    first.set_weight(0.7)
    second.set_weight(0.3)
    x = torch.randn(2, 5, 24, dtype = torch.float64)
    expected = x @ (base.weight + torch.kron(w1, w2)).T
    assert torch.allclose(block(x), expected, atol = 1e-10)
    assert block.proj["q"].weight is base.weight
    first.remove()
    second.remove()
    assert block.proj["q"] is base


def test_kron_adapter_row_slice_for_a_fused_qkv_factor():
    torch.manual_seed(1)
    block = _Block(out_features = 12).double()
    w1, w2 = (
        torch.randn(4, 3).double(),
        torch.randn(9, 8).double(),
    )  # 36 rows, this layer takes rows 12..24
    lf.KronAdapter(block, block.proj["q"], w1, w2, 1.0, rows = (12, 24))
    x = torch.randn(3, 24, dtype = torch.float64)
    base = block.proj["q"].base_layer
    assert torch.allclose(block(x), x @ (base.weight + torch.kron(w1, w2)[12:24]).T, atol = 1e-10)


def test_peft_loads_around_an_existing_lokr_wrapper():
    """A plain LoRA added after a LoKR on the same layer: PEFT injects into the bare Linear, and the LoKR wrapper is put
    back around the PEFT layer."""
    pytest.importorskip("peft")
    from core.inference.diffusion_torchao_patches import install_peft_torchao_dispatch_guard

    install_peft_torchao_dispatch_guard()  # as every diffusion entry point does
    block = _Block().double()

    class _Pipe:
        def __init__(self, transformer):
            self.transformer = transformer

        def load_lora_weights(
            self,
            _state,
            adapter_name = None,
        ):
            from peft import LoraConfig, inject_adapter_in_model
            inject_adapter_in_model(
                LoraConfig(r = 2, target_modules = ["q"]), self.transformer, adapter_name
            )

    pipe = _Pipe(block)
    adapter = lf.KronAdapter(
        block, block.proj["q"], torch.randn(4, 3).double(), torch.randn(9, 8).double(), 1.0
    )
    pipe.__dict__[lf.KRON_ATTR] = {"k": [adapter]}
    wrappers = lf._unwrap_kron(pipe)
    pipe.load_lora_weights({}, adapter_name = "l")
    lf._rewrap_kron(wrappers)
    wrapper = block.proj["q"]
    assert isinstance(wrapper, lf.kron_linear_class())
    assert type(wrapper.base_layer).__name__ == "Linear" and hasattr(wrapper.base_layer, "lora_A")


def test_an_embedding_lokr_is_refused_at_load(tmp_path):
    """ComfyUI's trainer writes LoKR for every weighted module, Anima's ``llm_adapter.embed`` table included. Its 2-D
    weight passes the shape check, but the wrapper's delta takes dense features, so the file must fail at load rather
    than on the first render."""
    conversion = pytest.importorskip("diffusers.loaders.lora_conversion_utils")
    if not hasattr(conversion, "_convert_non_diffusers_anima_lora_to_diffusers"):
        pytest.skip("this diffusers has no Anima LoRA converter")
    from safetensors.torch import save_file

    class _Conditioner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(12, 6)

    class _Pipe:
        def __init__(self):
            self.text_conditioner = _Conditioner()

    pipe = _Pipe()
    embed = pipe.text_conditioner.embed
    path = tmp_path / "lokr.safetensors"
    # ComfyUI's LokrAdapter.create_train on the [12, 6] table: w1 [3, 2], w2 [4, 3].
    save_file(
        {
            "diffusion_model.llm_adapter.embed.lokr_w1": torch.randn(3, 2),
            "diffusion_model.llm_adapter.embed.lokr_w2": torch.randn(4, 3),
        },
        path,
    )
    with pytest.raises(ValueError, match = "embedding"):
        lf.load_adapter(pipe, str(path), "k", "anima")
    assert pipe.text_conditioner.embed is embed
    assert lf.kron_names(pipe) == ()


def test_compiled_block_reads_a_new_lokr_weight():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    block = _Block().cuda().bfloat16()
    adapter = lf.KronAdapter(block, block.proj["q"], torch.randn(4, 3), torch.randn(9, 8), 1.0)
    block.compile(fullgraph = True)
    x = torch.randn(2, 24, device = "cuda", dtype = torch.bfloat16)
    on = block(x)
    adapter.set_weight(0.0)
    off = block(x)
    assert (on - off).abs().max() > 0
    assert torch.allclose(off, x @ block.proj["q"].weight.T)


def test_moss_and_anima_have_a_lora_loader():
    from core.inference.diffusion_lora import diffusers_lora_family_supported
    assert diffusers_lora_family_supported("moss-image")
    assert diffusers_lora_family_supported("anima")


@pytest.mark.parametrize("use_stream", [False, True])
def test_lokr_attached_under_group_offload_moves_with_its_group(tmp_path, use_stream):
    """A LoKR file applied to a transformer that already streams through diffusers group offload (Studio attaches
    adapters at generate time): the offload groups are rebuilt, so the factors onload with their block, and unloading
    rebuilds them again."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from diffusers.hooks import apply_group_offloading
    from safetensors.torch import save_file

    torch.manual_seed(0)

    class _Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([_Block(), _Block(24)])

        def forward(self, x):
            return self.blocks[0](self.blocks[1](x))

    class _Pipe:
        def __init__(self, transformer):
            self.transformer = transformer

        def unload_lora_weights(self):
            pass

    net = _Net().bfloat16()
    apply_group_offloading(
        net,
        onload_device = torch.device("cuda"),
        offload_device = torch.device("cpu"),
        offload_type = "block_level",
        num_blocks_per_group = 1,
        use_stream = use_stream,
    )
    x = torch.randn(2, 24, device = "cuda", dtype = torch.bfloat16)
    bare = net(x)
    w1, w2 = torch.randn(4, 3), torch.randn(9, 8)
    path = tmp_path / "lokr.safetensors"
    save_file(
        {
            "diffusion_model.blocks.0.proj.q.lokr_w1": w1,
            "diffusion_model.blocks.0.proj.q.lokr_w2": w2,
        },
        path,
    )
    pipe = _Pipe(net)
    lf.load_adapter(pipe, str(path), "k", "moss-image")
    out = net(x)
    inner = net.blocks[1](x)
    expected = bare + (inner @ torch.kron(w1, w2).to("cuda", torch.bfloat16).T)
    assert torch.allclose(out.float(), expected.float(), atol = 0.25, rtol = 0.05)
    lf.unload_adapters(pipe)
    assert torch.equal(net(x), bare)
