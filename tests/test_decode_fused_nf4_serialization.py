# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import copy
import gc
import types

import pytest
import torch

from .test_decode_fused_nf4 import _driver


def _model():
    bnb = pytest.importorskip("bitsandbytes")
    model = torch.nn.Module()
    with torch.random.fork_rng(devices = []):
        torch.manual_seed(3407)
        for name, n, k in (
            ("q_proj", 256, 256),
            ("k_proj", 128, 256),
            ("v_proj", 128, 256),
            ("o_proj", 256, 256),
            ("gate_proj", 512, 256),
            ("up_proj", 512, 256),
            ("down_proj", 256, 512),
        ):
            module = (
                bnb.nn.Linear4bit(
                    k,
                    n,
                    bias = False,
                    quant_type = "nf4",
                    compress_statistics = True,
                )
                .to(dtype = torch.bfloat16)
                .to("cpu")
            )
            if module.weight.quant_state is None:
                count = n * k // 64
                nested = bnb.functional.QuantState(
                    absmax = torch.ones((count + 255) // 256),
                    shape = (count,),
                    code = torch.linspace(0, 1, 256),
                    blocksize = 256,
                    dtype = torch.float32,
                )
                state = bnb.functional.QuantState(
                    absmax = torch.randint(0, 256, (count,), dtype = torch.uint8),
                    shape = (n, k),
                    code = bnb.functional.get_4bit_type("nf4", device = "cpu"),
                    blocksize = 64,
                    quant_type = "nf4",
                    dtype = torch.bfloat16,
                    offset = torch.tensor(0.25),
                    state2 = nested,
                )
                module.weight = bnb.nn.Params4bit(
                    torch.randint(0, 256, (n * k // 2, 1), dtype = torch.uint8),
                    requires_grad = False,
                    quant_state = state,
                    quant_type = "nf4",
                    bnb_quantized = True,
                    module = module,
                )
                module.quant_state = state
            model.add_module(name, module)
        model.embed_tokens = torch.nn.Embedding(64, 256)
        model.lm_head = torch.nn.Linear(256, 64, bias = False)
        model.norms = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.ones(256)) for _ in range(3)]
        )
    return model


def _decoded(weight):
    state = weight.quant_state
    packed = weight.data.reshape(-1)
    codes = torch.stack((packed >> 4, packed & 15), dim = -1).reshape(-1).long()
    scales = (
        state.state2.code[state.absmax.long()]
        * state.state2.absmax.repeat_interleave(state.state2.blocksize)[: state.absmax.numel()]
    )
    scales = scales + state.offset
    return (
        (state.code[codes] * scales.repeat_interleave(state.blocksize))
        .reshape(state.shape)
        .to(state.dtype)
    )


def _sites(model, monkeypatch):
    driver = _driver()

    def parts(projection, width):
        weight = projection.weight
        assert weight.quant_state.shape[1] == width
        return weight, weight.quant_state, weight.quant_state.shape[0], None

    def capture(kind, packed, absmax, code, bounds, width, blocksize, **kwargs):
        return types.SimpleNamespace(
            packed = packed,
            absmax = absmax,
            absmax2 = kwargs["absmax2"],
            segments = kwargs["segments"],
            bounds = bounds,
        )

    monkeypatch.setattr(driver, "_proj_parts", parts)
    monkeypatch.setattr(driver.F, "FusedSite", capture, raising = False)
    for kind, names, width in (
        ("qkv", ("q_proj", "k_proj", "v_proj"), 256),
        ("o", ("o_proj",), 256),
        ("gate_up", ("gate_proj", "up_proj"), 256),
        ("down", ("down_proj",), 512),
    ):
        projections = [getattr(model, name) for name in names]
        site = driver._site(model, kind, kind, projections, width)
        assert not isinstance(site, str), site
        assert len(site.segments) == len(projections)
        assert len(site.tensor_bindings) == len(projections)
        for (bound_projection, name, bound_weight), projection in zip(
            site.tensor_bindings, projections
        ):
            assert bound_projection is projection and name == "weight"
            assert bound_weight is projection.weight
        for segment, projection in zip(site.segments, projections):
            weight = projection.weight
            assert segment[0] is weight
            assert segment[1] is weight.quant_state.absmax
            assert segment[2] is weight.quant_state.state2.absmax
        first = projections[0].weight
        assert site.packed is first
        assert site.absmax is first.quant_state.absmax
        assert site.absmax2 is first.quant_state.state2.absmax


def _snapshot(model):
    bnb = pytest.importorskip("bitsandbytes")
    result = []
    for module in model.modules():
        if not isinstance(module, bnb.nn.Linear4bit):
            continue
        weight = module.weight
        state = weight.quant_state
        tensors = (
            weight,
            state.absmax,
            state.code,
            state.offset,
            state.state2.absmax,
            state.state2.code,
        )
        result.append(
            (
                module,
                weight,
                state,
                state.state2,
                tuple(
                    (
                        tensor,
                        tensor.data_ptr(),
                        tensor.untyped_storage()._cdata,
                        tensor.storage_offset(),
                        tensor.shape,
                        tensor.stride(),
                    )
                    for tensor in tensors
                ),
            )
        )
    return result


def _unchanged(snapshot):
    for module, weight, state, nested, tensors in snapshot:
        assert module.weight is weight
        assert weight.quant_state is state
        assert state.state2 is nested
        actual = (weight, state.absmax, state.code, state.offset, nested.absmax, nested.code)
        for value, (original, pointer, storage, offset, shape, stride) in zip(actual, tensors):
            assert value is original
            assert (
                value.data_ptr(),
                value.untyped_storage()._cdata,
                value.storage_offset(),
                value.shape,
                value.stride(),
            ) == (pointer, storage, offset, shape, stride)


def _save_load(model, folder, api):
    safe = pytest.importorskip("safetensors.torch")
    folder.mkdir()
    if api == "torch":
        torch.save(model.state_dict(), folder / "model.pt")
        return torch.load(folder / "model.pt", weights_only = True)
    if api == "accelerate":
        pytest.importorskip("accelerate").Accelerator(cpu = True).save_model(model, folder)
    elif api == "hub":
        pytest.importorskip("huggingface_hub").save_torch_model(model, folder)
    else:
        safe.save_model(model, str(folder / "model.safetensors"))
    restored = {}
    for path in folder.glob("*.safetensors"):
        restored.update(safe.load_file(path))
    return restored


@pytest.mark.parametrize("api", ["accelerate", "hub", "safetensors", "torch"])
def test_fused_nf4_checkpoints_keep_every_key_and_reload(tmp_path, monkeypatch, api):
    bnb = pytest.importorskip("bitsandbytes")
    model = _model()
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    baseline = _save_load(model, tmp_path / "base", api)
    original = _snapshot(model)
    with torch.inference_mode():
        _sites(model, monkeypatch)
    _unchanged(original)
    current = model.state_dict()
    assert expected.keys() == current.keys()
    assert all(torch.equal(value, current[key]) for key, value in expected.items())
    restored = _save_load(model, tmp_path / "head", api)
    _unchanged(original)
    assert expected.keys() == baseline.keys() == restored.keys()
    for key in expected:
        assert torch.equal(expected[key], baseline[key]), key
        assert torch.equal(expected[key], restored[key]), key
    for name, module in model.named_modules():
        if not isinstance(module, bnb.nn.Linear4bit):
            continue
        key = name + ".weight"
        stats = {k[len(key) + 1 :]: v for k, v in restored.items() if k.startswith(key + ".")}
        weight = bnb.nn.Params4bit.from_prequantized(restored[key], stats, device = "cpu")
        expected_weight = _decoded(module.weight)
        loaded_weight = _decoded(weight)
        assert torch.equal(expected_weight, loaded_weight), name


def test_fused_nf4_checkpoint_with_real_cpu_dequantization(tmp_path, monkeypatch):
    bnb = pytest.importorskip("bitsandbytes")
    try:
        packed, state = bnb.functional.quantize_4bit(
            torch.ones(128, 256, dtype = torch.bfloat16),
            quant_type = "nf4",
            compress_statistics = True,
        )
        bnb.functional.dequantize_4bit(packed, state)
    except NotImplementedError as error:
        pytest.skip(f"plain bitsandbytes has no CPU quantization/dequantization backend: {error}")
    model = _model()
    expected = {
        name: bnb.functional.dequantize_4bit(module.weight.data, module.weight.quant_state)
        for name, module in model.named_modules()
        if isinstance(module, bnb.nn.Linear4bit)
    }
    original = _snapshot(model)
    _sites(model, monkeypatch)
    _unchanged(original)
    restored = _save_load(model, tmp_path / "checkpoint", "safetensors")
    for name, expected_weight in expected.items():
        key = name + ".weight"
        stats = {k[len(key) + 1 :]: v for k, v in restored.items() if k.startswith(key + ".")}
        weight = bnb.nn.Params4bit.from_prequantized(restored[key], stats, device = "cpu")
        loaded = bnb.functional.dequantize_4bit(weight.data, weight.quant_state)
        assert torch.equal(loaded, expected_weight)


def test_fused_sites_preserve_aliases_training_and_deepcopy(monkeypatch):
    model = _model()
    model.lm_head.weight = model.embed_tokens.weight
    model.k_proj.weight = model.v_proj.weight
    model.k_proj.quant_state = model.v_proj.weight.quant_state
    original = _snapshot(model)
    x = torch.linspace(-1, 1, 512).reshape(2, 256)
    a = torch.nn.Parameter(torch.ones(8, 256) / 256)
    b = torch.nn.Parameter(torch.ones(256, 8) / 8)

    def step():
        a.grad = b.grad = None
        weight = _decoded(model.q_proj.weight).float()
        loss = (x @ weight.t() + x @ a.t() @ b.t()).square().mean()
        loss.backward()
        return loss.detach(), a.grad.clone(), b.grad.clone()

    before = step()
    with torch.inference_mode():
        _sites(model, monkeypatch)
    gc.collect()
    _unchanged(original)
    after = step()
    for expected, actual in zip(before, after):
        assert torch.equal(expected, actual)
    assert before[1].norm() > 0 and before[2].norm() > 0
    assert model.lm_head.weight is model.embed_tokens.weight
    assert model.k_proj.weight is model.v_proj.weight
    restored = copy.deepcopy(model)
    assert restored.lm_head.weight is restored.embed_tokens.weight
    assert restored.k_proj.weight is restored.v_proj.weight
    assert restored.q_proj.weight is not model.q_proj.weight
    assert restored.q_proj.weight.quant_state is not model.q_proj.weight.quant_state
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key]), key
    with torch.no_grad():
        restored.q_proj.weight.data[0] ^= 1
    _unchanged(original)
    for expected, actual in zip(before, step()):
        assert torch.equal(expected, actual)
