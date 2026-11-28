# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
"""Packed GGUF bases: exports after merge_adapter(), the checkpoint-recompute flag across threads, and which kernels
the zoo shim hands a call to."""

import threading
import types

import numpy as np
import pytest
import torch
from torch import nn

from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py

from unsloth.kernels import gguf_reference as ref
from unsloth.models import gguf_packed as gp


def _packed_linear(
    rows,
    cols,
    seed,
    permutation = None,
    dtype = torch.float32,
):
    t = ref.GGML_TYPE_IDS["Q8_0"]
    elems, size = ref.GGML_BLOCK[t]
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size = (rows * cols // elems, size), dtype = np.uint8)
    raw[:, 0:2] = np.array([0.01], dtype = np.float16).view(np.uint8)
    packed = torch.from_numpy(raw.reshape(rows, ref.row_bytes(t, cols)).copy())
    return gp.UnslothGGUFLinear.from_packed(
        packed, t, cols, rows, input_permutation = permutation, compute_dtype = dtype
    )


class _Block(nn.Module):
    def __init__(self, permutation = None):
        super().__init__()
        self.q = _packed_linear(32, 64, 0, permutation)
        self.k = _packed_linear(32, 64, 1)

    def forward(self, x):
        return self.q(x) + self.k(x)


@pytest.mark.parametrize("targets", (["q"], ["q", "k"]))
@pytest.mark.parametrize("permuted", (False, True))
def test_merge_adapter_then_export_writes_the_unmerged_base(targets, permuted):
    # merged_16bit / GGUF exports read the base from the dense copy and add the adapter themselves: after
    # merge_adapter() that copy must still hold the base, not base + delta (the delta would be added twice).
    peft = pytest.importorskip("peft")
    permutation = (
        torch.randperm(64, generator = torch.Generator().manual_seed(3)) if permuted else None
    )
    block = _Block(permutation)
    bases = {name: getattr(block, name).dequantize_weight(torch.float32) for name in ("q", "k")}
    model = peft.get_peft_model(block, peft.LoraConfig(r = 4, lora_alpha = 8, target_modules = targets))
    with torch.no_grad():
        for name in targets:
            layer = getattr(model.base_model.model, name)
            layer.lora_A["default"].weight.fill_(0.25)
            layer.lora_B["default"].weight.fill_(0.5)
    model.merge_adapter()
    merged = model.base_model.model.q.base_layer
    assert not torch.equal(merged.weight, bases["q"])  # the merge really wrote the delta in
    assert gp.is_gguf_packed_model(model)
    items = dict(gp._base_state_items(model, torch.float32))
    for name in ("q", "k"):
        assert torch.equal(items[f"{name}.weight"], bases[name]), name
    model.unmerge_adapter()
    assert isinstance(model.base_model.model.q.base_layer, gp.UnslothGGUFLinear)
    items = dict(gp._base_state_items(model, torch.float32))
    assert torch.equal(items["q.weight"], bases["q"])
    # merge_and_unload drops the wrapper: the dense weight is now the model's own and is what gets written.
    unloaded = model.merge_and_unload()
    items = dict(gp._base_state_items(unloaded, torch.float32))
    assert torch.equal(items["q.weight"], unloaded.q.weight)
    assert not torch.equal(items["q.weight"], bases["q"])


def test_overlapping_recomputes_on_two_threads_leave_the_flag_off():
    # Autograd runs each device's backward on its own thread: two recomputes overlap without nesting
    # (A enters, B enters, A leaves, B leaves). Afterwards no forward may keep its decoded weight.
    zoo_gc = pytest.importorskip("unsloth_zoo.gradient_checkpointing")
    if not hasattr(zoo_gc, "_gradient_checkpoint_recompute_marker"):
        pytest.skip("this unsloth_zoo has no recompute marker")
    assert gp._mirror_checkpoint_recompute()
    a_in, b_in, a_out = threading.Event(), threading.Event(), threading.Event()
    seen, errors = {}, []

    def run(first):
        try:
            with zoo_gc._gradient_checkpoint_recompute_marker(None):
                if first:
                    a_in.set()
                    b_in.wait(5)
                else:
                    b_in.set()
                    a_out.wait(5)
                    seen["b_inside_after_a_left"] = gp._IN_CHECKPOINT_RECOMPUTE[0]
            if first:
                a_out.set()
        except Exception as err:  # pragma: no cover - surfaced below
            errors.append(err)

    thread_a = threading.Thread(target = run, args = (True,))
    thread_a.start()
    assert a_in.wait(5)
    thread_b = threading.Thread(target = run, args = (False,))
    thread_b.start()
    thread_a.join(10)
    thread_b.join(10)
    assert not errors
    assert seen["b_inside_after_a_left"] is True
    assert gp._IN_CHECKPOINT_RECOMPUTE[0] is False
    assert gp._reuse_recompute_weight() is False
    module = _packed_linear(32, 64, 4)
    x = torch.randn(3, 64, requires_grad = True)
    y = module(x)
    # Outside a recompute nothing 16-bit is saved for backward (only the packed bytes).
    saved = [t for t in y.grad_fn.saved_tensors if t is not None]
    assert all(t.dtype == torch.uint8 for t in saved), [t.dtype for t in saved]


def _fake_zoo(calls):
    fake = types.ModuleType("unsloth_zoo.gguf_kernels")
    fake.COMPILE_TRACED_DEQUANTIZE = True

    def dequantize_gguf(packed, t, shape, dtype):
        calls.append("dequantize")
        return ref.dequantize_gguf(packed, t, shape, dtype)

    def gguf_linear(
        x,
        packed,
        t,
        shape,
        bias = None,
    ):
        calls.append("linear")
        # The zoo's dequantize + F.linear path: the bias goes in as given.
        return nn.functional.linear(x, ref.dequantize_gguf(packed, t, shape, x.dtype), bias)

    def gguf_linear_backward_input(g, packed, t, shape):
        calls.append("dx")
        return g @ ref.dequantize_gguf(packed, t, shape, g.dtype)

    fake.dequantize_gguf, fake.gguf_linear = dequantize_gguf, gguf_linear
    fake.gguf_linear_backward_input = gguf_linear_backward_input
    return fake


def _install(monkeypatch, fake):
    import sys

    monkeypatch.setitem(sys.modules, "unsloth_zoo.gguf_kernels", fake)
    parent = sys.modules.get("unsloth_zoo")
    if parent is not None:
        monkeypatch.setattr(parent, "gguf_kernels", fake, raising = False)
    monkeypatch.delenv("UNSLOTH_GGUF_KERNELS", raising = False)


def test_zoo_shim_keeps_cpu_blocks_on_the_traceable_reference(monkeypatch):
    # Off CUDA the zoo decodes with the same torch code inside an opaque custom op, which torch.compile cannot fuse:
    # by default those blocks stay on the reference (UNSLOTH_GGUF_KERNELS=zoo still forces the zoo).
    calls = []
    _install(monkeypatch, _fake_zoo(calls))
    ops = gp._resolve_ops()
    assert ops.name == "unsloth_zoo.gguf_kernels"
    module = _packed_linear(32, 64, 5)
    packed, t, shape = module.weight_packed, module.ggml_type, module.shape
    x = torch.randn(4, 64)
    expected = x @ ref.dequantize_gguf(packed, t, shape, torch.float32).t()
    assert torch.equal(ops.linear(x, packed, t, shape), expected)
    ops.dequantize(packed, t, shape, torch.float32)
    ops.linear_backward_input(torch.randn(4, 32), packed, t, shape)
    assert calls == []
    assert ops.supports(t)
    monkeypatch.setenv("UNSLOTH_GGUF_KERNELS", "zoo")
    forced = gp._resolve_ops()
    assert torch.equal(forced.linear(x, packed, t, shape), expected)
    assert calls == ["linear"]


@pytest.mark.skipif(not has_real_cuda(), reason = "the zoo kernels run CUDA blocks only")
def test_zoo_linear_takes_a_bias_of_another_dtype_like_the_reference():
    # PEFT's preparation can upcast a frozen bias to float32 while activations stay bf16: the reference casts it to
    # x's dtype, the zoo's dequantize + F.linear path (many rows) would reject it.
    gk = pytest.importorskip("unsloth_zoo.gguf_kernels")
    if not getattr(gk, "COMPILE_TRACED_DEQUANTIZE", False):
        pytest.skip("this unsloth_zoo is not the default kernel shim")
    gp._OPS.clear()
    try:
        ops = gp.gguf_ops()
        assert ops.name == "unsloth_zoo.gguf_kernels"
        module = _packed_linear(32, 64, 6, dtype = torch.bfloat16).cuda()
        packed, t, shape = module.weight_packed, module.ggml_type, module.shape
        bias = torch.randn(32, device = "cuda", dtype = torch.float32)
        for rows in (1, 4, 300):
            x = torch.randn(rows, 64, device = "cuda", dtype = torch.bfloat16)
            y = ops.linear(x, packed, t, shape, bias)
            expected = gp._reference_linear(x, packed, t, shape, bias)
            assert y.dtype == torch.bfloat16
            torch.testing.assert_close(y, expected, atol = 2e-2, rtol = 2e-2)
    finally:
        gp._OPS.clear()


@pytest.mark.parametrize("pre_merged", (False, True))
def test_float32_merged_export_decodes_the_blocks_in_float32(tmp_path, monkeypatch, pre_merged):
    # save_pretrained_merged(dtype = float32) of a bf16-loaded packed model: the dense base copy the merge reads must
    # be decoded in float32 (as the unwrapped export does), not rounded through bf16 first.
    peft = pytest.importorskip("peft")
    from transformers import LlamaConfig, LlamaForCausalLM

    from unsloth.save import unsloth_generic_save

    if not torch.cuda.is_available():
        # The merge names the GPU to size its shards; a GPU-less runner that still reports a CUDA build has none.
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args, **kwargs: "")
    config = LlamaConfig(
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 1,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        vocab_size = 128,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).to(torch.bfloat16)
    attn = model.model.layers[0].self_attn
    attn.q_proj = _packed_linear(64, 64, 7, dtype = torch.bfloat16)
    attn.k_proj = _packed_linear(64, 64, 8, dtype = torch.bfloat16)
    model.config._name_or_path = str(tmp_path / "missing.gguf")
    blocks = attn.q_proj.weight_packed, attn.q_proj.ggml_type
    lora = peft.get_peft_model(model, peft.LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj"]))
    layer = lora.base_model.model.model.layers[0].self_attn.q_proj
    with torch.no_grad():
        layer.lora_A["default"].weight.normal_(0, 0.1)
        layer.lora_B["default"].weight.normal_(0, 0.1)
    delta = (layer.lora_B["default"].weight.float() @ layer.lora_A["default"].weight.float()) * 2.0
    base = ref.dequantize_gguf(blocks[0], blocks[1], (64, 64), torch.float32)
    if pre_merged:
        lora.merge_adapter()
    out = tmp_path / "out"
    unsloth_generic_save(lora, None, str(out), save_method = "merged_16bit", dtype = torch.float32)
    from safetensors.torch import load_file

    saved = {}
    for file in out.glob("*.safetensors"):
        saved.update(load_file(str(file)))
    q = saved["model.layers.0.self_attn.q_proj.weight"]
    assert q.dtype == torch.float32
    torch.testing.assert_close(q, base + delta, atol = 1e-6, rtol = 1e-6)
    k = saved["model.layers.0.self_attn.k_proj.weight"]
    assert torch.equal(
        k,
        ref.dequantize_gguf(
            attn.k_proj.weight_packed, attn.k_proj.ggml_type, (64, 64), torch.float32
        ),
    )
