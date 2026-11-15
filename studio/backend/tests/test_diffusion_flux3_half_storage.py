# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.3 float16 loads (fp16-only cards) read bf16 as stored instead of converting each component on the host.

The text encoder keeps its bf16 Linears / Embeddings with an exact per-call cast; a transformer that a native quant
rebuilds is read as stored, quantised from the float16 rounding of each weight, then left exactly as a float16 load
would hold it. Every check here compares against the widened float16 load bit for bit."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_flux3 as f3  # noqa: E402
from core.inference import diffusion_manual_cast as mc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_diffusion_flux3_manual_cast import _stored_bf16, _tiny_flux3  # noqa: E402

FP16, BF16, FP32 = torch.float16, torch.bfloat16, torch.float32


def _widened_fp16(model, keep = ()):
    """What from_pretrained(torch_dtype=float16) yields from a bf16 checkpoint."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not any(k in name for k in keep):
                p.data = p.data.to(FP16)
    return model


def _with_subnormals_(model):
    """bf16 weights that do not survive a float16 round trip (below float16's normal range)."""
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                w = m.weight.data
                w.view(-1)[::7] = (
                    torch.rand(w.numel() // 7 + 1)[: w.view(-1)[::7].numel()] * 4e-5
                ).to(w.dtype)
    return model


def _pair(seed = 0):
    """(bf16 stored model, its widened float16 twin)."""
    model, kw = _tiny_flux3()
    keep = tuple(model._keep_in_fp32_modules or ())
    stored = _with_subnormals_(_stored_bf16(model, keep = keep))
    widened = _widened_fp16(copy.deepcopy(stored), keep = keep)
    return stored, widened, keep, kw


def _tensors(module):
    out = dict(module.state_dict())
    for name, sub in module.named_modules():
        for b, buf in sub._buffers.items():
            if buf is not None:
                out[f"{name}.{b}"] = buf
    return out


def _bits(t):
    return t.contiguous().view(torch.uint8) if t.dtype != torch.bool else t


def test_storage_dtype_float16_kill_switch(monkeypatch):
    cuda = SimpleNamespace(device = "cuda")
    monkeypatch.delenv(mc.MANUAL_CAST_ENV, raising = False)
    monkeypatch.delenv(f3.HALF_STORAGE_ENV, raising = False)
    assert f3.flux3_storage_dtype(cuda, FP16) is BF16
    for off in ("0", "off", "false", "no"):
        monkeypatch.setenv(f3.HALF_STORAGE_ENV, off)
        assert f3.flux3_storage_dtype(cuda, FP16) is None
        assert f3.flux3_storage_dtype(cuda, FP32) is BF16
    monkeypatch.setenv(f3.HALF_STORAGE_ENV, "1")
    assert f3.flux3_storage_dtype(SimpleNamespace(device = "cpu"), FP16) is None


def test_transformer_storage_routing():
    pick = f3.transformer_storage_dtype
    # default: only a wider compute dtype reads the transformer as stored (a dense float16 one would cast per step)
    assert pick(BF16, FP32, None) is BF16
    assert pick(BF16, FP16, None) is None
    # a native quant rebuilds the Linears: read as stored at any width
    assert pick(BF16, FP16, True) is BF16
    assert pick(BF16, FP32, True) is BF16
    assert pick(BF16, FP32, False) is None
    assert pick(None, FP16, True) is None
    assert f3.storage_saves_memory(BF16, FP32) and not f3.storage_saves_memory(BF16, FP16)


def test_flux3_float16_manual_cast_is_bit_identical_to_the_float16_load():
    stored, widened, _keep, kw = _pair()
    assert mc.apply_manual_cast_(stored, FP16) > 0
    assert stored.dtype is FP16
    kw = {
        k: (
            v.to(FP16)
            if torch.is_tensor(v) and v.is_floating_point() and k.endswith("states")
            else v
        )
        for k, v in kw.items()
    }
    with torch.no_grad():
        want = widened(**kw)[0]
        got = stored(**kw)[0]
    assert got.dtype is want.dtype
    assert torch.equal(_bits(got), _bits(want))


def _quantise(model):
    from core.inference.diffusion_transformer_quant import _quantize_native
    pipe = SimpleNamespace(transformer = model)
    return _quantize_native(
        pipe,
        "int8",
        family = "flux.3-klein",
        min_features = 32,
        act_int8 = True,
        logger = None,
        compute_dtype = FP16,
    )


def test_native_int8_from_stored_bf16_equals_the_float16_build():
    stored, widened, _keep, _kw = _pair()
    mc.apply_manual_cast_(stored, FP16)
    assert _quantise(widened) == "int8" and _quantise(stored) == "int8"
    a, b = _tensors(widened), _tensors(stored)
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape, k
        assert torch.equal(_bits(a[k]), _bits(b[k])), k
    # nothing casts per call any more, and the marker that skips QKV fusion is gone
    assert not any(type(m).__name__.startswith("ManualCast") for m in stored.modules())
    assert mc.manual_cast_info(stored) is None
    assert sorted(type(m).__name__ for m in stored.modules()) == sorted(
        type(m).__name__ for m in widened.modules()
    )


def test_codes_round_through_the_compute_dtype_first():
    """bf16 -> float16 is not exact below float16's normal range: quantising the bf16 values directly would give other
    codes than the float16 load for a row whose scale is that small."""
    from core.inference.diffusion_native_quant import native_linear_class

    cls = native_linear_class()
    lin = torch.nn.Linear(256, 4, bias = False)
    with torch.no_grad():
        lin.weight.copy_((torch.randn(4, 256) * 1e-5))
    stored = copy.deepcopy(lin).to(BF16)
    widened = copy.deepcopy(stored).to(FP16)
    assert not torch.equal(stored.weight.float(), widened.weight.float())
    # the first float owner converts whole (module.dtype); the Linear under test stays stored
    mc.apply_manual_cast_(torch.nn.Sequential(torch.nn.Linear(1, 1).to(BF16), stored), FP16)
    assert type(stored).__name__ == "ManualCastLinear"
    want = cls(widened, "int8", act_int8 = True, rot_group = 0)
    got = cls(stored, "int8", act_int8 = True, rot_group = 0)
    assert got.compute_dtype is FP16
    assert torch.equal(got.weight_q, want.weight_q)
    assert torch.equal(got.weight_scale, want.weight_scale)
    # the uncast bf16 values themselves quantise differently: the round through float16 is what makes them agree
    raw = torch.nn.Linear(256, 4, bias = False).to(FP32)
    with torch.no_grad():
        raw.weight.copy_(stored.weight.float())
    assert not (
        torch.equal(cls(raw, "int8", act_int8 = True).weight_q, want.weight_q)
        and torch.equal(cls(raw, "int8", act_int8 = True).weight_scale, want.weight_scale)
    )


def test_undo_keeps_a_float32_manual_cast():
    m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)).to(BF16)
    mc.apply_manual_cast_(m, FP32)
    assert mc.undo_same_width_manual_cast_(m) == 0
    assert type(m[1]).__name__ == "ManualCastLinear" and m[1].weight.dtype is BF16
    assert mc.manual_cast_info(m) is not None


def test_undo_float16_restores_plain_modules_with_the_same_forward():
    torch.manual_seed(0)
    m = torch.nn.Sequential(
        torch.nn.Embedding(10, 8), torch.nn.Linear(8, 8), torch.nn.Linear(8, 8, bias = False)
    ).to(BF16)
    ref = copy.deepcopy(m).to(FP16)
    mc.apply_manual_cast_(m, FP16)
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        before = m(ids)
    n = mc.undo_same_width_manual_cast_(m)
    assert n > 0
    assert [type(x) for x in m] == [torch.nn.Embedding, torch.nn.Linear, torch.nn.Linear]
    assert all(p.dtype is FP16 for p in m.parameters())
    assert not any(hasattr(x, "_unsloth_compute_dtype") for x in m)
    assert mc.manual_cast_info(m) is None
    with torch.no_grad():
        after = m(ids)
        want = ref(ids)
    assert torch.equal(_bits(after), _bits(want)) and torch.equal(_bits(before), _bits(want))


def test_qwen3vl_float16_manual_cast_matches_the_float16_load():
    transformers = pytest.importorskip("transformers")
    if getattr(transformers, "Qwen3VLConfig", None) is None:
        pytest.skip("transformers without Qwen3-VL")
    cfg = transformers.Qwen3VLConfig(
        text_config = dict(
            hidden_size = 32,
            intermediate_size = 64,
            num_hidden_layers = 2,
            num_attention_heads = 4,
            num_key_value_heads = 2,
            head_dim = 8,
            vocab_size = 128,
            rope_scaling = {"rope_type": "default", "mrope_section": [1, 1, 2]},
        ),
        vision_config = dict(
            depth = 1,
            hidden_size = 16,
            intermediate_size = 32,
            num_heads = 2,
            out_hidden_size = 32,
            deepstack_visual_indexes = [0],
        ),
    )
    torch.manual_seed(0)
    stored = _with_subnormals_(
        _stored_bf16(transformers.Qwen3VLForConditionalGeneration(cfg).eval())
    )
    widened = _widened_fp16(copy.deepcopy(stored))
    assert mc.apply_manual_cast_(stored, FP16) > 0
    assert stored.dtype is FP16
    ids = torch.randint(0, 128, (1, 7), generator = torch.Generator().manual_seed(2))
    with torch.no_grad():
        want = widened(input_ids = ids, output_hidden_states = True).hidden_states
        got = stored(input_ids = ids, output_hidden_states = True).hidden_states
    for a, b in zip(got, want):
        assert a.dtype is FP16 and torch.equal(_bits(a), _bits(b))


def test_undo_never_reads_the_weight_of_a_quantised_linear():
    """A native quantised Linear dequantises its whole weight on ``.weight``: the undo must not touch it."""

    class Quantised(torch.nn.Module):
        @property
        def weight(self):
            raise AssertionError("dequantised a quantised weight")

    m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)).to(BF16)
    mc.apply_manual_cast_(m, FP16)
    m.append(Quantised())  # what the native quant swaps in after the manual cast
    assert mc.undo_same_width_manual_cast_(m) > 0
    assert type(m[1]) is torch.nn.Linear and m[1].weight.dtype is FP16
