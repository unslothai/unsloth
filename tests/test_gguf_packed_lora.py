# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
"""`gguf_packed = True`: quantized Linears keep their GGUF blocks, LoRA trains on top, and everything matches the
same GGUF dequantized to dense (transformers' own load) bit for bit."""

import contextlib
import logging
import os

import numpy as np
import pytest
import torch
from torch import nn

from unsloth.kernels import gguf_reference as ref
from unsloth.models import gguf_packed as gp

# Byte offsets of the fp16 scale fields in one block, made small and finite so products stay meaningful.
_F16_FIELDS = {
    "Q4_0": (0,),
    "Q5_0": (0,),
    "Q8_0": (0,),
    "Q4_K": (0, 2),
    "Q5_K": (0, 2),
    "Q6_K": (208,),
    "IQ4_XS": (0,),
    "IQ2_XXS": (0,),
}
_TYPES = ("Q4_K", "Q5_K", "Q6_K", "Q8_0", "Q4_0", "Q5_0", "IQ4_XS", "IQ2_XXS")


def _blocks(
    name,
    rows,
    cols,
    seed,
    scale = 0.01,
):
    t = ref.GGML_TYPE_IDS[name]
    elems, size = ref.GGML_BLOCK[t]
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size = (rows * cols // elems, size), dtype = np.uint8)
    scale = np.array([scale], dtype = np.float16).view(np.uint8)
    for offset in _F16_FIELDS[name]:
        raw[:, offset : offset + 2] = scale
    return torch.from_numpy(raw.reshape(rows, ref.row_bytes(t, cols)).copy())


def _install_fake_zoo_kernels(monkeypatch, fake):
    # `import unsloth_zoo.gguf_kernels` reads the parent package's attribute once the real module was imported.
    import sys

    monkeypatch.setitem(sys.modules, "unsloth_zoo.gguf_kernels", fake)
    parent = sys.modules.get("unsloth_zoo")
    if parent is not None:
        monkeypatch.setattr(parent, "gguf_kernels", fake, raising = False)


def _pair(
    name,
    rows = 64,
    cols = 256,
    seed = 0,
    dtype = torch.float32,
    permutation = None,
):
    t = ref.GGML_TYPE_IDS[name]
    packed = _blocks(name, rows, cols, seed)
    module = gp.UnslothGGUFLinear.from_packed(
        packed, t, cols, rows, input_permutation = permutation, compute_dtype = dtype
    )
    weight = ref.dequantize_gguf(packed, t, (rows, cols), dtype)
    if permutation is not None:
        weight = weight[:, torch.argsort(permutation)]
    dense = nn.Linear(cols, rows, bias = False, dtype = dtype)
    dense.weight = nn.Parameter(weight, requires_grad = False)
    return module, dense


@pytest.mark.parametrize("name", _TYPES)
def test_packed_linear_forward_backward_match_dense(name):
    module, dense = _pair(name)
    assert module.weight_packed.dtype == torch.uint8 and module.shape == (64, 256)
    assert torch.equal(module.weight, dense.weight)
    x = torch.randn(3, 5, 256, requires_grad = True)
    x_dense = x.detach().clone().requires_grad_(True)
    y, y_dense = module(x), dense(x_dense)
    assert torch.equal(y, y_dense)
    grad = torch.randn_like(y)
    y.backward(grad)
    y_dense.backward(grad)
    assert torch.equal(x.grad, x_dense.grad)
    assert not module.weight_packed.requires_grad and module.weight_packed.grad is None


def test_compiled_layers_of_mixed_types_see_constant_types_and_shapes():
    # One compiled decoder-layer function runs layers whose packed Linears differ in ggml type and size (Q4_K_M puts
    # Q6_K on some down_proj). Every graph must see the type and shape as constants: a SymInt there sends the zoo
    # kernels' type lookup out of the graph (a graph break per Linear call) and the reference to generic code.
    torch._dynamo.reset()
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(
            [n.meta.get("example_value") for n in gm.graph.nodes if n.op == "placeholder"]
        )
        return gm.forward

    @torch.compile(backend = backend, fullgraph = True)
    def run(module, x):
        return module(x) * module.ggml_type + module.shape[0] + module.shape[1]

    modules = [
        _pair("Q4_K", rows = 64, cols = 256)[0],
        _pair("Q6_K", rows = 64, cols = 256)[0],
        _pair("Q8_0", rows = 32, cols = 512)[0],
        _pair("Q4_K", rows = 96, cols = 512)[0],
    ]
    for module in modules:
        x = torch.randn(4, module.in_features)
        expected = module(x) * module.ggml_type + module.out_features + module.in_features
        assert torch.equal(run(module, x), expected)
    symbolic = [
        value
        for placeholders in graphs
        for value in placeholders
        if isinstance(value, torch.SymInt)
    ]
    assert graphs and not symbolic
    assert [m.ggml_type for m in modules] == [12, 14, 8, 12]
    assert all(isinstance(m.ggml_type, int) for m in modules)
    assert [m.shape for m in modules] == [(64, 256), (64, 256), (32, 512), (96, 512)]
    torch._dynamo.reset()


@pytest.mark.parametrize("permuted", (False, True))
def test_checkpoint_recompute_keeps_its_decoded_weight_for_dx(monkeypatch, permuted):
    # A gradient-checkpoint recompute is followed at once by its backward: the Linear keeps the weight it decoded for
    # the forward and dX decodes nothing. Outside a recompute nothing 16-bit is saved. Same values either way.
    zoo_gc = pytest.importorskip("unsloth_zoo.gradient_checkpointing")
    if not hasattr(zoo_gc, "_gradient_checkpoint_recompute_marker"):
        pytest.skip("this unsloth_zoo has no recompute marker")
    permutation = (
        torch.randperm(256, generator = torch.Generator().manual_seed(1)) if permuted else None
    )
    module, dense = _pair("Q4_K", permutation = permutation)
    assert gp._mirror_checkpoint_recompute()
    ops = gp.gguf_ops()
    calls = {"decode": 0, "dx": 0}
    real_dequantize, real_dx = ops.dequantize, ops.linear_backward_input

    def dequantize(*args, **kwargs):
        calls["decode"] += 1
        return real_dequantize(*args, **kwargs)

    def dx(*args, **kwargs):
        calls["dx"] += 1
        return real_dx(*args, **kwargs)

    monkeypatch.setattr(ops, "dequantize", dequantize)
    monkeypatch.setattr(ops, "linear_backward_input", dx)
    grads = {}
    for recompute in (False, True):
        x = torch.randn(3, 256, generator = torch.Generator().manual_seed(2)).requires_grad_(True)
        marker = (
            zoo_gc._gradient_checkpoint_recompute_marker(None)
            if recompute
            else contextlib.nullcontext()
        )
        with marker:
            y = module(x)
        assert gp._IN_CHECKPOINT_RECOMPUTE[0] is False
        before = dict(calls)
        y.backward(torch.ones_like(y))
        grads[recompute] = x.grad
        decoded_in_backward = calls["decode"] - before["decode"] + calls["dx"] - before["dx"]
        assert decoded_in_backward == (0 if recompute else 1)
    assert torch.equal(grads[False], grads[True])
    x = torch.randn(3, 256, generator = torch.Generator().manual_seed(2)).requires_grad_(True)
    dense(x).backward(torch.ones(3, 64))
    assert torch.equal(grads[True], x.grad)
    monkeypatch.setattr(gp, "_RECOMPUTE_REUSE", False)
    with zoo_gc._gradient_checkpoint_recompute_marker(None):
        assert gp._reuse_recompute_weight() is False


def test_kernel_shim_default_follows_the_zoo(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("unsloth_zoo.gguf_kernels")
    fake.dequantize_gguf = lambda packed, t, shape, dtype: ref.dequantize_gguf(
        packed, t, shape, dtype
    )
    fake.gguf_linear = lambda x, packed, t, shape, bias = None: gp._reference_linear(
        x, packed, t, shape, bias
    )
    fake.gguf_linear_backward_input = (
        lambda g, packed, t, shape: gp._reference_linear_backward_input(g, packed, t, shape)
    )
    fake.SUPPORTED_TYPES = frozenset({ref.GGML_TYPE_IDS["Q8_0"]})
    _install_fake_zoo_kernels(monkeypatch, fake)
    monkeypatch.delenv("UNSLOTH_GGUF_KERNELS", raising = False)
    # A zoo whose decodes are Python custom ops in compiled graphs (slower than the reference): opt-in only.
    assert gp._resolve_ops().name == "reference"
    monkeypatch.setenv("UNSLOTH_GGUF_KERNELS", "zoo")
    assert gp._resolve_ops().name == "unsloth_zoo.gguf_kernels"
    # A zoo that launches its dequantize kernel from the graph: the default.
    fake.COMPILE_TRACED_DEQUANTIZE = True
    monkeypatch.delenv("UNSLOTH_GGUF_KERNELS", raising = False)
    assert gp._resolve_ops().name == "unsloth_zoo.gguf_kernels"
    monkeypatch.setenv("UNSLOTH_GGUF_KERNELS", "auto")
    assert gp._resolve_ops().name == "unsloth_zoo.gguf_kernels"
    monkeypatch.setenv("UNSLOTH_GGUF_KERNELS", "reference")
    assert gp._resolve_ops().name == "reference"


@pytest.mark.parametrize("kernels", ("reference", "zoo"))
def test_gguf_expert_stack_grouped_mm_routing(monkeypatch, kernels):
    # zoo kernels: the stack's own grouped GEMM (gguf_grouped_mm) runs forward and dX; reference: the zoo's
    # chunked dequantize + grouped GEMM. Both equal the dense stack.
    import sys
    import types

    pe = pytest.importorskip("unsloth_zoo.packed_experts")
    if not getattr(pe, "PACKED_EXPERT_OWN_GROUPED_MM", False):
        pytest.skip("this unsloth_zoo has no own-grouped-GEMM hook")
    from unsloth_zoo.mxfp4_stacked_experts import _grouped_mm

    calls = []

    def fake_grouped(
        x,
        packed,
        t,
        shape,
        counts,
        transpose_b = True,
    ):
        calls.append((tuple(shape), counts.dtype, transpose_b))
        dense = ref.dequantize_gguf(packed, t, shape, x.dtype)
        weight = dense.transpose(-1, -2) if transpose_b else dense
        return _grouped_mm(x, weight, torch.cumsum(counts, 0, dtype = torch.int32))

    fake = types.ModuleType("unsloth_zoo.gguf_kernels")
    fake.dequantize_gguf = lambda packed, t, shape, dtype: ref.dequantize_gguf(
        packed, t, shape, dtype
    )
    fake.gguf_linear = lambda x, packed, t, shape, bias = None: gp._reference_linear(
        x, packed, t, shape, bias
    )
    fake.gguf_linear_backward_input = gp._reference_linear_backward_input
    fake.gguf_grouped_mm = fake_grouped
    _install_fake_zoo_kernels(monkeypatch, fake)
    monkeypatch.setenv("UNSLOTH_GGUF_KERNELS", kernels)
    monkeypatch.setattr(gp, "_OPS", [])
    cls = gp.gguf_expert_param_class()
    assert cls is not None
    E, N, K = 3, 32, 256
    t = ref.GGML_TYPE_IDS["Q8_0"]
    blocks = torch.stack([_blocks("Q8_0", N, K, seed) for seed in range(E)])
    param = cls(blocks, ggml_type = t, packed_cols = K)
    dense = ref.dequantize_gguf(blocks, t, (E, N, K), torch.float32)
    counts = torch.tensor([2, 0, 5])
    x = torch.randn(7, K, requires_grad = True)
    x2 = x.detach().clone().requires_grad_(True)
    out = pe.packed_expert_grouped_linear(x, param, counts, True)
    want = _grouped_mm(x2, dense.transpose(-1, -2), torch.cumsum(counts, 0, dtype = torch.int32))
    assert torch.equal(out, want)
    g = torch.randn_like(out)
    out.backward(g)
    want.backward(g)
    assert torch.equal(x.grad, x2.grad)
    assert calls == (
        [((E, N, K), torch.int32, True), ((E, N, K), torch.int32, False)]
        if kernels == "zoo"
        else []
    )


def test_input_permutation_is_the_dense_column_reorder():
    torch.manual_seed(0)
    permutation = torch.randperm(256)
    module, dense = _pair("Q4_K", permutation = permutation)
    x = torch.randn(4, 256)
    # Stored columns are the model's reordered: y = blocks @ x[perm].
    blocks = ref.dequantize_gguf(
        module.weight_packed, module.ggml_type, module.shape, torch.float32
    )
    # Same sums in another order: equal up to float32 rounding.
    torch.testing.assert_close(module(x), x[:, permutation] @ blocks.t(), rtol = 1e-5, atol = 1e-4)
    assert torch.equal(module(x), dense(x))


@pytest.mark.parametrize("name", ("Q4_K", "Q8_0", "Q6_K"))
def test_peft_lora_on_packed_base_matches_dense(name):
    peft = pytest.importorskip("peft")
    torch.manual_seed(0)
    module, dense = _pair(name)
    trees = []
    for base in (module, dense):
        tree = nn.Sequential()
        tree.add_module("proj", base)
        config = peft.LoraConfig(
            r = 4, lora_alpha = 8, target_modules = ["proj"], init_lora_weights = False
        )
        torch.manual_seed(1)
        trees.append(peft.get_peft_model(tree, config))
    x = torch.randn(6, 256)
    losses, grads = [], []
    for tree in trees:
        loss = tree(x).square().mean()
        loss.backward()
        losses.append(loss)
        grads.append({n: p.grad for n, p in tree.named_parameters() if p.requires_grad})
    assert torch.equal(losses[0], losses[1])
    assert grads[0].keys() == grads[1].keys() and len(grads[0]) == 2
    for key in grads[0]:
        assert torch.equal(grads[0][key], grads[1][key]), key
    # merge_and_unload densifies the packed base, then folds the adapter in.
    merged = [tree.merge_and_unload() for tree in trees]
    assert type(merged[0].proj) is nn.Linear
    assert torch.equal(merged[0].proj.weight, merged[1].proj.weight)


def test_peft_unmerge_restores_the_packed_base():
    peft = pytest.importorskip("peft")
    module, _ = _pair("Q4_K")
    packed = module.weight_packed
    tree = nn.Sequential()
    tree.add_module("proj", module)
    tree = peft.get_peft_model(
        tree, peft.LoraConfig(r = 4, target_modules = ["proj"], init_lora_weights = False)
    )
    x = torch.randn(2, 256)
    before = tree(x)
    tree.base_model.model.proj.merge()
    tree.base_model.model.proj.unmerge()
    base = tree.base_model.model.proj.base_layer
    assert isinstance(base, gp.UnslothGGUFLinear) and base.weight_packed is packed
    assert torch.allclose(tree(x), before, atol = 1e-5)


@pytest.mark.parametrize("fast", (False, True))
def test_pissa_init_refused_on_packed_base(fast):
    # fast: Unsloth's get_peft_model swaps in its own PiSSA init (lora_init.fast_lora_init), which must route a
    # packed base to the guarded PEFT init instead of writing a residual into a weight that cannot hold it.
    import contextlib

    peft = pytest.importorskip("peft")
    module, _ = _pair("Q8_0")
    tree = nn.Sequential()
    tree.add_module("proj", module)
    if fast:
        from unsloth.models.lora_init import fast_lora_init
        scope = fast_lora_init(force = True)
    else:
        scope = contextlib.nullcontext()
    with scope, pytest.raises(NotImplementedError, match = "packed in its GGUF"):
        peft.get_peft_model(
            tree, peft.LoraConfig(r = 4, target_modules = ["proj"], init_lora_weights = "pissa")
        )


@pytest.mark.parametrize("name", ("Q4_K", "Q6_K"))
def test_fused_lora_kernels_read_the_packed_state(name):
    from unsloth.kernels.utils import fast_dequantize, matmul_lora

    module, dense = _pair(name)
    state = module.mxfp4_quant_state()
    assert getattr(type(state), "_unsloth_packed_weight_state", False)
    W = module.weight_packed
    assert torch.equal(fast_dequantize(W, state), dense.weight)
    assert torch.equal(fast_dequantize(W.t(), state), dense.weight.t())
    x = torch.randn(7, 256)
    A, B = torch.randn(4, 256) * 0.1, torch.randn(64, 4) * 0.1
    out = matmul_lora(x, W, state, A, B, 0.5)
    expected = x @ dense.weight.t() + 0.5 * (x @ A.t()) @ B.t()
    assert torch.allclose(out, expected, atol = 1e-4)
    grad = torch.randn(7, 64)
    # Backward orientation: W.t() view, X is the output gradient.
    assert torch.allclose(
        matmul_lora(grad, W.t(), state, None, None, None), grad @ dense.weight, atol = 1e-4
    )


def test_request_validation(monkeypatch):
    assert gp.check_gguf_packed_request({"gguf_file": "a.gguf"}) is False
    with pytest.raises(ValueError, match = "gguf_file"):
        gp.check_gguf_packed_request({"gguf_packed": True})
    with pytest.raises(NotImplementedError, match = "full_finetuning"):
        gp.check_gguf_packed_request(
            {"gguf_packed": True, "gguf_file": "a.gguf"}, full_finetuning = True
        )
    monkeypatch.setattr(
        gp, "gguf_packed_supported", lambda: "transformers 4.57.6 is older than 5.18.0"
    )
    with pytest.raises(RuntimeError, match = "5.18.0"):
        gp.check_gguf_packed_request({"gguf_packed": True, "gguf_file": "a.gguf"})


def test_dense_default_unchanged():
    from unsloth.models._utils import gguf_dequantize_kwargs

    kwargs = gguf_dequantize_kwargs({"gguf_file": "a.gguf"})
    config = kwargs.get("quantization_config")
    assert config is None or config.dequantize is True


# ----------------------------------------------------------------------------------------------------------
# A tiny llama GGUF written here: packed load vs transformers' dense load of the same file.
# ----------------------------------------------------------------------------------------------------------

_HIDDEN, _HEADS, _KV, _HEAD_DIM, _FFN, _LAYERS, _VOCAB = 256, 2, 1, 128, 512, 2, 512
_LLAMA_TYPES = {
    "attn_q": "Q4_K",
    "attn_k": "Q8_0",
    "attn_v": "Q6_K",
    "attn_output": "IQ4_XS",
    "ffn_gate": "Q5_0",
    "ffn_up": "Q4_0",
    "ffn_down": "Q5_K",
}


def _write_llama_gguf(
    path,
    arch = "llama",
    rename = None,
):
    gguf = pytest.importorskip("gguf")
    writer = gguf.GGUFWriter(str(path), arch)
    writer.add_uint32(f"{arch}.context_length", 256)
    writer.add_uint32(f"{arch}.embedding_length", _HIDDEN)
    writer.add_uint32(f"{arch}.block_count", _LAYERS)
    writer.add_uint32(f"{arch}.feed_forward_length", _FFN)
    writer.add_uint32(f"{arch}.rope.dimension_count", _HEAD_DIM)
    writer.add_uint32(f"{arch}.attention.head_count", _HEADS)
    writer.add_uint32(f"{arch}.attention.head_count_kv", _KV)
    writer.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", 1e-5)
    writer.add_float32(f"{arch}.rope.freq_base", 10000.0)
    writer.add_string("tokenizer.ggml.model", "llama")
    writer.add_array("tokenizer.ggml.tokens", [f"t{i}" for i in range(_VOCAB)])
    rng = np.random.default_rng(0)

    def quantized(name, rows, cols, type_name, seed):
        t = getattr(gguf.GGMLQuantizationType, type_name)
        # Small scales keep the logits O(1), so the LoRA gradients compare at a tight tolerance.
        writer.add_tensor(
            name, _blocks(type_name, rows, cols, seed, scale = 1e-4).numpy(), raw_dtype = t
        )

    quantized("token_embd.weight", _VOCAB, _HIDDEN, "Q8_0", 1)
    quantized("output.weight", _VOCAB, _HIDDEN, "Q6_K", 2)
    writer.add_tensor(
        "output_norm.weight", (1 + 0.1 * rng.standard_normal(_HIDDEN)).astype(np.float32)
    )
    shapes = {
        "attn_q": (_HEADS * _HEAD_DIM, _HIDDEN),
        "attn_k": (_KV * _HEAD_DIM, _HIDDEN),
        "attn_v": (_KV * _HEAD_DIM, _HIDDEN),
        "attn_output": (_HIDDEN, _HEADS * _HEAD_DIM),
        "ffn_gate": (_FFN, _HIDDEN),
        "ffn_up": (_FFN, _HIDDEN),
        "ffn_down": (_HIDDEN, _FFN),
    }
    for layer in range(_LAYERS):
        for leaf, (rows, cols) in shapes.items():
            name = (rename or {}).get(leaf, leaf)
            quantized(
                f"blk.{layer}.{name}.weight", rows, cols, _LLAMA_TYPES[leaf], 10 * layer + len(leaf)
            )
        for norm in ("attn_norm", "ffn_norm"):
            writer.add_tensor(
                f"blk.{layer}.{norm}.weight",
                (1 + 0.1 * rng.standard_normal(_HIDDEN)).astype(np.float32),
            )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _new_gguf_path():
    if gp.gguf_packed_supported() is not None:
        pytest.skip(f"packed GGUF loading needs transformers >= {gp.GGUF_PACKED_MIN_TRANSFORMERS}")


def test_packed_llama_gguf_matches_transformers_dense_load(tmp_path):
    _new_gguf_path()
    from transformers import AutoModelForCausalLM

    _write_llama_gguf(tmp_path / "tiny.gguf")
    dense = AutoModelForCausalLM.from_pretrained(
        tmp_path, gguf_file = "tiny.gguf", dtype = torch.float32
    )
    with gp.gguf_packed_load():
        packed = AutoModelForCausalLM.from_pretrained(
            tmp_path,
            gguf_file = "tiny.gguf",
            dtype = torch.float32,
            attn_implementation = "eager",
            quantization_config = gp.gguf_packed_quantization_config(),
        )
    census = packed._unsloth_gguf_packed_census
    assert census["packed_linear"] == 7 * _LAYERS
    assert set(census["dequantized"]) == {"model.embed_tokens", "lm_head"}
    assert getattr(packed, "hf_quantizer", None) is None
    reference = dense.state_dict()
    for name, module in packed.named_modules():
        if isinstance(module, gp.UnslothGGUFLinear):
            assert torch.equal(module.weight, reference[f"{name}.weight"]), name
    for key, value in packed.state_dict().items():
        if value.dtype != torch.uint8:
            assert torch.equal(value, reference[key]), key
    ids = torch.randint(0, _VOCAB, (2, 9))
    # Bitwise-equal weights; the products may differ in the last bits (CPU GEMM blocking follows the buffers).
    with torch.no_grad():
        torch.testing.assert_close(
            packed(input_ids = ids).logits, dense(input_ids = ids).logits, rtol = 1e-5, atol = 1e-3
        )

    # One LoRA step on both: same loss and gradients, the base never trains.
    peft = pytest.importorskip("peft")
    grads, losses = [], []
    for model in (dense, packed):
        torch.manual_seed(0)
        model = peft.get_peft_model(
            model,
            peft.LoraConfig(
                r = 4, lora_alpha = 8, target_modules = "all-linear", init_lora_weights = False
            ),
        )
        out = model(input_ids = ids, labels = ids)
        out.loss.backward()
        losses.append(out.loss.detach())
        grads.append({n: p.grad for n, p in model.named_parameters() if p.requires_grad})
    torch.testing.assert_close(losses[0], losses[1], rtol = 1e-5, atol = 0)
    assert grads[0].keys() == grads[1].keys() and len(grads[0]) == 2 * 7 * _LAYERS
    for key in grads[0]:
        torch.testing.assert_close(grads[0][key], grads[1][key], rtol = 1e-4, atol = 1e-5, msg = key)


def test_dense_base_for_merges_round_trips(tmp_path):
    # merged_16bit / GGUF exports read the base checkpoint: for a packed GGUF it is the dequantized base.
    _new_gguf_path()
    peft = pytest.importorskip("peft")
    from transformers import AutoModelForCausalLM

    _write_llama_gguf(tmp_path / "tiny.gguf")
    dense = AutoModelForCausalLM.from_pretrained(
        tmp_path, gguf_file = "tiny.gguf", dtype = torch.float32
    )
    with gp.gguf_packed_load():
        packed = AutoModelForCausalLM.from_pretrained(
            tmp_path,
            gguf_file = "tiny.gguf",
            dtype = torch.float32,
            quantization_config = gp.gguf_packed_quantization_config(),
        )
    packed = peft.get_peft_model(packed, peft.LoraConfig(r = 4, target_modules = "all-linear"))
    assert gp.is_gguf_packed_model(packed)
    with gp.gguf_dense_base_for_merge(packed, tmp_path / "merged") as resolver:
        base_dir = resolver("anything")
        reloaded = AutoModelForCausalLM.from_pretrained(base_dir, dtype = torch.float32)
        reference = dense.state_dict()
        for key, value in reloaded.state_dict().items():
            assert torch.equal(value, reference[key]), key
    assert not os.path.exists(base_dir)
    with gp.gguf_dense_base_for_merge(dense, tmp_path / "merged") as resolver:
        assert resolver is None
    # Several shards: each is written and released before the next fills.
    import gc
    import weakref

    import safetensors.torch as st

    written, alive, saved_save_file = [], [], st.save_file

    def recording_save_file(
        tensors,
        filename,
        metadata = None,
    ):
        gc.collect()
        alive.append(sum(ref() is not None for refs in written for ref in refs))
        written.append([weakref.ref(t) for t in tensors.values()])
        return saved_save_file(tensors, filename, metadata = metadata)

    st.save_file = recording_save_file
    try:
        out = gp.write_gguf_dense_base(packed, tmp_path / "sharded", max_shard_bytes = 1 << 14)
    finally:
        st.save_file = saved_save_file
    assert len(written) > 2
    assert alive == [0] * len(written), alive
    reloaded = AutoModelForCausalLM.from_pretrained(out, dtype = torch.float32)
    for key, value in reloaded.state_dict().items():
        assert torch.equal(value, reference[key]), key


def test_bf16_packed_load_has_no_float32_advice(tmp_path):
    # transformers advises float32 for its own packed kernels; Unsloth's dequantize into the compute dtype.
    _new_gguf_path()
    from transformers import AutoModelForCausalLM
    from transformers.utils import logging as hf_logging

    _write_llama_gguf(tmp_path / "tiny.gguf")
    messages = []

    class _Collect(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    handler = _Collect()
    hf_logger = hf_logging.get_logger("transformers.quantizers.quantizer_gguf")
    hf_logger.addHandler(handler)
    try:
        with gp.gguf_packed_load():
            packed = AutoModelForCausalLM.from_pretrained(
                tmp_path,
                gguf_file = "tiny.gguf",
                dtype = torch.bfloat16,
                quantization_config = gp.gguf_packed_quantization_config(),
            )
    finally:
        hf_logger.removeHandler(handler)
    assert gp.is_gguf_packed_model(packed)
    assert not [m for m in messages if "fastest in float32" in m], messages
    assert packed.model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16


def test_dense_base_unwraps_modules_to_save(tmp_path):
    # Trained embeddings / lm_head (modules_to_save): the base checkpoint holds their original weights under the
    # canonical names, not the PEFT wrapper's.
    _new_gguf_path()
    peft = pytest.importorskip("peft")
    from transformers import AutoModelForCausalLM

    _write_llama_gguf(tmp_path / "tiny.gguf")
    dense = AutoModelForCausalLM.from_pretrained(
        tmp_path, gguf_file = "tiny.gguf", dtype = torch.float32
    )
    with gp.gguf_packed_load():
        packed = AutoModelForCausalLM.from_pretrained(
            tmp_path,
            gguf_file = "tiny.gguf",
            dtype = torch.float32,
            quantization_config = gp.gguf_packed_quantization_config(),
        )
    config = peft.LoraConfig(
        r = 4, target_modules = ["q_proj"], modules_to_save = ["lm_head", "embed_tokens"]
    )
    lora = peft.get_peft_model(packed, config)
    with torch.no_grad():
        for name, param in lora.named_parameters():
            if ".modules_to_save." in name:
                param.add_(1.0)  # trained copies differ from the base
    out = gp.write_gguf_dense_base(lora, tmp_path / "base")
    reloaded = AutoModelForCausalLM.from_pretrained(out, dtype = torch.float32).state_dict()
    reference = dense.state_dict()
    assert set(reloaded) == set(reference)
    for key, value in reloaded.items():
        assert torch.equal(value, reference[key]), key


def test_dense_base_keeps_float32_parameters(tmp_path):
    # A bf16 model with parameters held in float32 (Qwen3.5's A_log / dt_bias): the base keeps them float32.
    _new_gguf_path()
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    _write_llama_gguf(tmp_path / "tiny.gguf")
    with gp.gguf_packed_load():
        packed = AutoModelForCausalLM.from_pretrained(
            tmp_path,
            gguf_file = "tiny.gguf",
            dtype = torch.bfloat16,
            quantization_config = gp.gguf_packed_quantization_config(),
        )
    norm = packed.model.layers[0].input_layernorm
    norm.weight.data = norm.weight.data.float() + 1e-3  # not representable in bf16
    out = gp.write_gguf_dense_base(packed, tmp_path / "base")
    tensors = {}
    for f in os.listdir(out):
        if f.endswith(".safetensors"):
            tensors.update(load_file(os.path.join(out, f)))
    saved = tensors["model.layers.0.input_layernorm.weight"]
    assert saved.dtype == torch.float32 and torch.equal(saved, norm.weight.data)
    assert tensors["model.layers.0.self_attn.q_proj.weight"].dtype == torch.bfloat16


def test_merged_16bit_save_of_an_unwrapped_packed_model(tmp_path):
    # After merge_and_unload the model is no PeftModel and its non-target linears are still packed: the
    # export writes dequantized weights, never the uint8 blocks.
    _new_gguf_path()
    peft = pytest.importorskip("peft")
    from transformers import AutoModelForCausalLM

    from unsloth.save import unsloth_generic_save

    _write_llama_gguf(tmp_path / "tiny.gguf")
    with gp.gguf_packed_load():
        packed = AutoModelForCausalLM.from_pretrained(
            tmp_path,
            gguf_file = "tiny.gguf",
            dtype = torch.float32,
            quantization_config = gp.gguf_packed_quantization_config(),
        )
    lora = peft.get_peft_model(packed, peft.LoraConfig(r = 4, target_modules = ["q_proj"]))
    torch.manual_seed(0)
    with torch.no_grad():
        for name, param in lora.named_parameters():
            if "lora_B" in name:
                param.normal_(0, 1e-2)
    merged = lora.merge_and_unload()
    assert gp.is_gguf_packed_model(merged)  # k / v / o / MLP stay packed
    import copy

    want = copy.deepcopy(merged)
    gp.densify_gguf_packed_model(want)
    want = want.state_dict()
    with pytest.raises(ValueError, match = "state_dict"):
        unsloth_generic_save(
            merged,
            None,
            str(tmp_path / "bad"),
            save_method = "merged_16bit",
            state_dict = merged.state_dict(),
        )
    with pytest.raises(RuntimeError, match = "merged_16bit"):
        unsloth_generic_save(merged, None, str(tmp_path / "bad"), save_method = "merged_4bit_forced")
    # An older single-file export in the destination must not shadow the new shards.
    os.makedirs(tmp_path / "out")
    from safetensors.torch import save_file

    save_file({"stale": torch.zeros(1)}, str(tmp_path / "out" / "model.safetensors"))
    (tmp_path / "out" / "pytorch_model.bin").write_bytes(b"stale")
    unsloth_generic_save(
        merged, None, str(tmp_path / "out"), save_method = "merged_16bit", max_shard_size = "20KB"
    )
    assert len(list((tmp_path / "out").glob("model-*-of-*.safetensors"))) > 2
    assert not (tmp_path / "out" / "pytorch_model.bin").exists()
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / "out", dtype = torch.float32)
    got = reloaded.state_dict()
    assert set(got) == set(want)
    for key, value in got.items():
        assert torch.equal(value, want[key]), key


def test_gguf_export_of_an_unwrapped_packed_model(tmp_path, monkeypatch):
    # save_pretrained_gguf on a non-PEFT packed model hands the converter dequantized weights, not uint8 blocks.
    _new_gguf_path()
    from transformers import AutoModelForCausalLM

    import unsloth.save as save_mod

    _write_llama_gguf(tmp_path / "tiny.gguf")
    with gp.gguf_packed_load():
        packed = AutoModelForCausalLM.from_pretrained(
            tmp_path,
            gguf_file = "tiny.gguf",
            dtype = torch.float32,
            quantization_config = gp.gguf_packed_quantization_config(),
        )
    import copy

    want = copy.deepcopy(packed)
    gp.densify_gguf_packed_model(want)
    want = want.state_dict()
    seen = {}

    def _save_to_gguf(**kwargs):
        from safetensors.torch import load_file

        folder = kwargs.get("model_directory") or kwargs.get("save_directory")
        tensors = {}
        for name in os.listdir(folder):
            if name.endswith(".safetensors"):
                tensors.update(load_file(os.path.join(folder, name)))
        seen.update(tensors)
        output = tmp_path / "out_gguf" / "model.Q8_0.gguf"
        output.parent.mkdir(exist_ok = True)
        output.write_bytes(b"GGUF")
        return [str(output)], True, False

    monkeypatch.setattr(save_mod, "_is_vlm", lambda _model: False)
    monkeypatch.setattr(save_mod, "_is_gpt_oss", lambda _model: False)
    monkeypatch.setattr(save_mod, "fix_tokenizer_bos_token", lambda _t: (False, None))
    monkeypatch.setattr(save_mod, "_resolve_imatrix_file", lambda *_a, **_kw: None)
    monkeypatch.setattr(save_mod, "_preflight_gguf_disk", lambda **kw: (kw["save_directory"], True))
    monkeypatch.setattr(save_mod, "create_ollama_modelfile", lambda *_a, **_kw: None)
    monkeypatch.setattr(save_mod, "save_to_gguf", _save_to_gguf)

    class _Tokenizer:
        bos_token_id = 1
        chat_template = None

        def __call__(self, text, **kwargs):
            return {"input_ids": [[self.bos_token_id, 7]]}

        def save_pretrained(self, path):
            pass

    with pytest.raises(ValueError, match = "state_dict"):
        save_mod.unsloth_save_pretrained_gguf(
            packed,
            str(tmp_path / "bad"),
            tokenizer = _Tokenizer(),
            quantization_method = "q8_0",
            state_dict = packed.state_dict(),
        )
    save_mod.unsloth_save_pretrained_gguf(
        packed, str(tmp_path / "out"), tokenizer = _Tokenizer(), quantization_method = "q8_0"
    )
    assert seen and not [k for k in seen if "weight_packed" in k]
    for key, value in seen.items():
        assert torch.equal(value, want[key]), key


def test_unknown_architecture_is_named(tmp_path):
    _new_gguf_path()
    from transformers import AutoModelForCausalLM

    _write_llama_gguf(tmp_path / "tiny.gguf", arch = "starcoder2")
    with gp.gguf_packed_load(), pytest.raises(NotImplementedError, match = "starcoder2"):
        AutoModelForCausalLM.from_pretrained(
            tmp_path,
            gguf_file = "tiny.gguf",
            quantization_config = gp.gguf_packed_quantization_config(),
        )


def test_layout_the_map_does_not_know_is_refused(tmp_path):
    # As a converter that fuses tensors under a new name: those parameters would be randomly initialised.
    _new_gguf_path()
    from transformers import AutoModelForCausalLM

    _write_llama_gguf(tmp_path / "tiny.gguf", rename = {"attn_q": "attn_q_fused"})
    with gp.gguf_packed_load(), pytest.raises(NotImplementedError, match = "q_proj"):
        AutoModelForCausalLM.from_pretrained(
            tmp_path,
            gguf_file = "tiny.gguf",
            quantization_config = gp.gguf_packed_quantization_config(),
        )


def test_load_context_restores_transformers(tmp_path):
    _new_gguf_path()
    import transformers.integrations.gguf.dequant as dequant
    import transformers.integrations.gguf.gguf_conversion_mapping as mapping
    import transformers.quantizers.quantizer_gguf as quantizer

    before = (
        dict(dequant.GGML_BLOCK),
        dict(mapping.GGUF_ARCHS),
        quantizer.get_gguf_kernel,
        dict(quantizer.GgufHfQuantizer.__dict__),
    )
    with pytest.raises(RuntimeError, match = "boom"):
        with gp.gguf_packed_load():
            assert "llama" in mapping.GGUF_ARCHS
            raise RuntimeError("boom")
    after = (
        dict(dequant.GGML_BLOCK),
        dict(mapping.GGUF_ARCHS),
        quantizer.get_gguf_kernel,
        dict(quantizer.GgufHfQuantizer.__dict__),
    )
    assert after == before


# A tiny Qwen3.5 GGUF in both layouts: the separate attn_qkv / attn_gate / ssm_beta / ssm_alpha tensors, or the
# `[attn_qkv | attn_gate]` and `[ssm_beta | ssm_alpha]` row stacks current llama.cpp converters (and so
# `save_pretrained_gguf`) write. Same bytes either way.
# ----------------------------------------------------------------------------------------------------------

_Q35 = dict(hidden = 256, ffn = 512, heads = 2, kv = 1, head_dim = 128, k_heads = 1, v_heads = 2, layers = 4)


def _write_qwen35_gguf(
    path,
    fused,
    fused_type = "Q8_0",
    keep = None,
    shard = None,
):
    """keep: write only tensors whose name it accepts (values are drawn either way, so shards of one file agree);
    shard = (index, count) adds llama.cpp's split.* keys."""
    gguf = pytest.importorskip("gguf")
    c = _Q35
    value_dim, key_dim = c["v_heads"] * c["head_dim"], c["k_heads"] * c["head_dim"]
    arch = "qwen35"
    writer = gguf.GGUFWriter(str(path), arch)
    for key, value in {
        "context_length": 256,
        "embedding_length": c["hidden"],
        "feed_forward_length": c["ffn"],
        "block_count": c["layers"],
        "attention.head_count": c["heads"],
        "attention.head_count_kv": c["kv"],
        "attention.key_length": c["head_dim"],
        "attention.value_length": c["head_dim"],
        "rope.dimension_count": 32,
        "full_attention_interval": 4,
        "ssm.conv_kernel": 4,
        "ssm.state_size": c["head_dim"],
        "ssm.group_count": c["k_heads"],
        "ssm.time_step_rank": c["v_heads"],
        "ssm.inner_size": value_dim,
    }.items():
        writer.add_uint32(f"{arch}.{key}", value)
    writer.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6)
    writer.add_float32(f"{arch}.rope.freq_base", 10000.0)
    writer.add_array(f"{arch}.rope.dimension_sections", [6, 5, 5, 0])
    if shard is not None:
        writer.add_uint16("split.no", shard[0])
        writer.add_uint16("split.count", shard[1])
    writer.add_string("tokenizer.ggml.model", "llama")
    writer.add_array("tokenizer.ggml.tokens", [f"t{i}" for i in range(_VOCAB)])
    rng = np.random.default_rng(0)
    seed = iter(range(1000))

    def blocks(rows, cols, type_name):
        return _blocks(type_name, rows, cols, next(seed), scale = 1e-3).numpy()

    def add(name, data, **kwargs):
        if keep is None or keep(name):
            writer.add_tensor(name, data, **kwargs)

    def quantized(name, data, type_name):
        add(name, data, raw_dtype = getattr(gguf.GGMLQuantizationType, type_name))

    def floats(
        name,
        *shape,
        offset = 0.0,
    ):
        add(name, (offset + 0.1 * rng.standard_normal(shape)).astype(np.float32))

    quantized("token_embd.weight", blocks(_VOCAB, c["hidden"], "Q8_0"), "Q8_0")
    floats("output_norm.weight", c["hidden"], offset = 1.0)
    for layer in range(c["layers"]):
        p = f"blk.{layer}."
        floats(p + "attn_norm.weight", c["hidden"], offset = 1.0)
        floats(p + "post_attention_norm.weight", c["hidden"], offset = 1.0)
        quantized(p + "ffn_gate.weight", blocks(c["ffn"], c["hidden"], "Q4_K"), "Q4_K")
        quantized(p + "ffn_up.weight", blocks(c["ffn"], c["hidden"], "Q4_K"), "Q4_K")
        quantized(p + "ffn_down.weight", blocks(c["hidden"], c["ffn"], "Q6_K"), "Q6_K")
        if (layer + 1) % 4 == 0:
            quantized(
                p + "attn_q.weight",
                blocks(2 * c["heads"] * c["head_dim"], c["hidden"], "Q4_K"),
                "Q4_K",
            )
            quantized(
                p + "attn_k.weight", blocks(c["kv"] * c["head_dim"], c["hidden"], "Q4_K"), "Q4_K"
            )
            quantized(
                p + "attn_v.weight", blocks(c["kv"] * c["head_dim"], c["hidden"], "Q6_K"), "Q6_K"
            )
            quantized(
                p + "attn_output.weight",
                blocks(c["hidden"], c["heads"] * c["head_dim"], "Q4_K"),
                "Q4_K",
            )
            floats(p + "attn_q_norm.weight", c["head_dim"], offset = 1.0)
            floats(p + "attn_k_norm.weight", c["head_dim"], offset = 1.0)
            continue
        qkv = blocks(2 * key_dim + value_dim, c["hidden"], fused_type)
        gate = blocks(value_dim, c["hidden"], fused_type)
        beta = blocks(c["v_heads"], c["hidden"], fused_type)
        alpha = blocks(c["v_heads"], c["hidden"], fused_type)
        if fused:
            quantized(p + "attn_qkv_gate.weight", np.concatenate([qkv, gate]), fused_type)
            quantized(p + "ssm_beta_alpha.weight", np.concatenate([beta, alpha]), fused_type)
        else:
            quantized(p + "attn_qkv.weight", qkv, fused_type)
            quantized(p + "attn_gate.weight", gate, fused_type)
            quantized(p + "ssm_beta.weight", beta, fused_type)
            quantized(p + "ssm_alpha.weight", alpha, fused_type)
        quantized(p + "ssm_out.weight", blocks(c["hidden"], value_dim, "Q4_K"), "Q4_K")
        add(p + "ssm_a", -np.exp(0.1 * rng.standard_normal(c["v_heads"])).astype(np.float32))
        floats(p + "ssm_dt.bias", c["v_heads"])
        floats(p + "ssm_conv1d.weight", 2 * key_dim + value_dim, 4)
        floats(p + "ssm_norm.weight", c["head_dim"])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _packed_from(path):
    from transformers import AutoModelForCausalLM
    with gp.gguf_packed_load(str(path)):
        return AutoModelForCausalLM.from_pretrained(
            path.parent,
            gguf_file = path.name,
            dtype = torch.float32,
            attn_implementation = "eager",
            quantization_config = gp.gguf_packed_quantization_config(),
        )


@pytest.mark.parametrize("fused_type", ["Q8_0", "Q4_K"])
def test_fused_qwen35_layout_loads_packed(tmp_path, fused_type):
    # Main refuses the fused file (parameters with no tensor); it must load to exactly the unfused file's model.
    _new_gguf_path()
    pytest.importorskip("transformers.models.qwen3_5")
    from transformers import AutoModelForCausalLM

    _write_qwen35_gguf(tmp_path / "fused.gguf", fused = True, fused_type = fused_type)
    _write_qwen35_gguf(tmp_path / "split.gguf", fused = False, fused_type = fused_type)
    fused, split = _packed_from(tmp_path / "fused.gguf"), _packed_from(tmp_path / "split.gguf")
    packed = {n for n, m in fused.named_modules() if isinstance(m, gp.UnslothGGUFLinear)}
    for leaf in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
        assert f"model.layers.0.linear_attn.{leaf}" in packed, leaf
    a, b = fused.state_dict(), split.state_dict()
    assert a.keys() == b.keys()
    for key in a:
        assert a[key].dtype == b[key].dtype and torch.equal(a[key], b[key]), key

    # And to transformers' own dense load of the unfused file, weight for weight.
    dense = AutoModelForCausalLM.from_pretrained(
        tmp_path, gguf_file = "split.gguf", dtype = torch.float32, attn_implementation = "eager"
    )
    reference = dense.state_dict()
    for name, module in fused.named_modules():
        if isinstance(module, gp.UnslothGGUFLinear):
            assert torch.equal(module.weight, reference[f"{name}.weight"]), name
    ids = torch.randint(0, _VOCAB, (2, 7), generator = torch.Generator().manual_seed(0))
    with torch.no_grad():
        logits = fused(input_ids = ids).logits
        assert torch.equal(logits, split(input_ids = ids).logits)
        torch.testing.assert_close(logits, dense(input_ids = ids).logits, rtol = 1e-5, atol = 1e-4)


def test_fused_qwen35_layout_in_a_split_gguf(tmp_path):
    # Fused tensors in one shard, the ssm_out their split points come from in another, and a metadata-only first
    # shard: the load must equal the single unfused file's.
    _new_gguf_path()
    pytest.importorskip("transformers.models.qwen3_5")
    fused_names = ("attn_qkv_gate.weight", "ssm_beta_alpha.weight")

    def in_second(name):
        return name.endswith(fused_names) or name == "token_embd.weight"

    paths = [tmp_path / f"m-{i:05d}-of-00003.gguf" for i in (1, 2, 3)]
    _write_qwen35_gguf(paths[0], fused = True, keep = lambda name: False, shard = (0, 3))
    _write_qwen35_gguf(paths[1], fused = True, keep = in_second, shard = (1, 3))
    _write_qwen35_gguf(paths[2], fused = True, keep = lambda name: not in_second(name), shard = (2, 3))
    _write_qwen35_gguf(tmp_path / "unfused.gguf", fused = False)
    loaded, want = _packed_from(paths[0]), _packed_from(tmp_path / "unfused.gguf")
    a, b = loaded.state_dict(), want.state_dict()
    assert a.keys() == b.keys()
    for key in a:
        assert a[key].dtype == b[key].dtype and torch.equal(a[key], b[key]), key


def test_fused_split_is_row_slices_of_the_same_bytes():
    _new_gguf_path()
    from transformers.integrations.gguf.reader import GgufHeader, TensorInfo

    q8 = ref.GGML_TYPE_IDS["Q8_0"]
    row = ref.row_bytes(q8, 256)
    tensors = (
        TensorInfo("blk.0.attn_qkv_gate.weight", (768, 256), q8, 4096, 768 * row),
        TensorInfo("blk.0.ssm_beta_alpha.weight", (4, 256), 0, 0, 4 * 256 * 4),
        TensorInfo("blk.0.ssm_out.weight", (256, 256), q8, 0, 256 * row),
        TensorInfo("blk.3.attn_q.weight", (512, 256), q8, 0, 512 * row),
    )
    header = GgufHeader("x.gguf", "qwen35", tensors, 0)
    split = {t.name: t for t in gp._split_fused_gguf_tensors(header).tensors}
    assert split["blk.0.attn_qkv.weight"] == TensorInfo(
        "blk.0.attn_qkv.weight", (512, 256), q8, 4096, 512 * row
    )
    assert split["blk.0.attn_gate.weight"] == TensorInfo(
        "blk.0.attn_gate.weight", (256, 256), q8, 4096 + 512 * row, 256 * row
    )
    assert (
        split["blk.0.ssm_beta.weight"].offset == 0
        and split["blk.0.ssm_alpha.weight"].offset == 2 * 256 * 4
    )
    assert "blk.0.attn_qkv_gate.weight" not in split and "blk.3.attn_q.weight" in split
    # A file without fused tensors passes through as is; one the split cannot place is refused, not guessed.
    plain = GgufHeader("x.gguf", "qwen35", tensors[2:], 0)
    assert gp._split_fused_gguf_tensors(plain) is plain
    orphan = GgufHeader("x.gguf", "qwen35", tensors[:1], 0)
    with pytest.raises(NotImplementedError, match = "attn_qkv_gate"):
        gp._split_fused_gguf_tensors(orphan)
