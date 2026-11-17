# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.3 on fp16-only cards: bf16 weights kept as stored, float32 compute per op (diffusion_manual_cast)."""

from __future__ import annotations

import ast
import inspect
import json
import struct
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_flux3 as f3  # noqa: E402
from core.inference import diffusion_manual_cast as mc  # noqa: E402


def _bf16_rounded_(model):
    """Round every float tensor through bf16 so the float32 model equals the bf16-stored one exactly."""
    with torch.no_grad():
        for p in model.parameters():
            p.data = p.data.to(torch.bfloat16).to(torch.float32)
    return model


def _stored_bf16(model, keep = ()):
    """What from_pretrained(torch_dtype=bf16) yields: bf16 everywhere except the class's fp32-kept modules."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not any(k in name for k in keep):
                p.data = p.data.to(torch.bfloat16)
    return model


def _tiny_flux3():
    fmod = pytest.importorskip("diffusers.models.transformers.transformer_flux3")
    torch.manual_seed(0)
    model = fmod.Flux3Transformer2DModel(
        in_channels = 16,
        cond_in_channels = 16,
        num_layers = 2,
        num_single_layers = 2,
        attention_head_dim = 32,
        num_attention_heads = 4,
        joint_attention_dim = 32,
        timestep_channels = 32,
        axes_dims_rope = (8, 8, 8, 8),
    ).eval()
    g = torch.Generator().manual_seed(1)
    img, txt, ref = 64, 9, 32
    kw = {
        "hidden_states": torch.randn(2, img, 16, generator = g),
        "encoder_hidden_states": torch.randn(2, txt, 32, generator = g),
        "timestep": torch.tensor([0.7, 0.7]),
        "img_ids": torch.randint(0, 16, (img, 4), generator = g).float(),
        "txt_ids": torch.randint(0, 16, (txt, 4), generator = g).float(),
        "image_cond_hidden_states": torch.randn(2, ref, 16, generator = g),
        "image_cond_ids": torch.randint(0, 16, (ref, 4), generator = g).float(),
        "return_dict": False,
    }
    if "target_timestep" in inspect.signature(model.forward).parameters and model.config.get(
        "flow_map", False
    ):
        kw["target_timestep"] = torch.tensor([0.5, 0.5])
    return model, kw


def test_flux3_manual_cast_is_bit_identical_to_the_float32_load():
    import copy

    model, kw = _tiny_flux3()
    ref_model = _bf16_rounded_(model)
    cast_model = _stored_bf16(copy.deepcopy(ref_model), keep = ref_model._keep_in_fp32_modules or ())
    kept = mc.apply_manual_cast_(cast_model, torch.float32)
    assert kept > 0
    # module.dtype drives the latents' dtype in the FLUX.3 denoise block
    assert cast_model.dtype is torch.float32
    linears = [m for m in cast_model.modules() if isinstance(m, torch.nn.Linear)]
    stored = [m for m in linears if m.weight.dtype is torch.bfloat16]
    assert stored and len(stored) >= len(linears) - 1
    assert all(type(m).__name__ == "ManualCastLinear" for m in stored)
    for m in cast_model.modules():
        if type(m).__name__ == "ManualCastLinear":
            continue
        for p in m.parameters(recurse = False):
            assert p.dtype is torch.float32
    with torch.no_grad():
        want = ref_model(**kw)[0]
        got = cast_model(**kw)[0]
    assert got.dtype is torch.float32
    assert torch.equal(got, want)
    info = mc.manual_cast_info(cast_model)
    assert info["kept_bytes"] == kept and info["compute_dtype"] == "torch.float32"


def test_qwen3vl_encoder_manual_cast_matches_float32():
    import copy

    transformers = pytest.importorskip("transformers")
    Qwen3VLConfig = getattr(transformers, "Qwen3VLConfig", None)
    if Qwen3VLConfig is None:
        pytest.skip("transformers without Qwen3-VL")
    cfg = Qwen3VLConfig(
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
    ref = _bf16_rounded_(transformers.Qwen3VLForConditionalGeneration(cfg).eval())
    cast = _stored_bf16(copy.deepcopy(ref))
    assert mc.apply_manual_cast_(cast, torch.float32) > 0
    assert cast.dtype is torch.float32
    assert cast.get_input_embeddings().weight.dtype is torch.bfloat16
    ids = torch.randint(0, 128, (1, 7), generator = torch.Generator().manual_seed(2))
    with torch.no_grad():
        want = ref(input_ids = ids, output_hidden_states = True).hidden_states
        got = cast(input_ids = ids, output_hidden_states = True).hidden_states
    assert len(got) == len(want)
    for a, b in zip(got, want):
        assert a.dtype is torch.float32
        assert torch.equal(a, b)


def test_manual_cast_linear_traces_without_graph_breaks():
    lin = torch.nn.Linear(8, 4).to(torch.bfloat16)
    holder = torch.nn.Sequential(torch.nn.LayerNorm(8), lin)
    mc.apply_manual_cast_(holder, torch.float32)
    assert type(holder[1]).__name__ == "ManualCastLinear"
    x = torch.randn(3, 8)
    compiled = torch.compile(holder, backend = "eager", fullgraph = True)
    with torch.no_grad():
        assert torch.equal(compiled(x), holder(x))


def test_manual_cast_leaves_subclassed_linears_alone():
    class Quantised(torch.nn.Linear):
        pass

    m = torch.nn.Sequential(torch.nn.Linear(4, 4), Quantised(4, 4), torch.nn.Linear(4, 4)).to(
        torch.bfloat16
    )
    mc.apply_manual_cast_(m, torch.float32)
    assert type(m[0]) is torch.nn.Linear and m[0].weight.dtype is torch.float32
    assert type(m[1]) is Quantised and m[1].weight.dtype is torch.float32
    assert type(m[2]).__name__ == "ManualCastLinear" and m[2].weight.dtype is torch.bfloat16


def test_storage_dtype_only_for_cuda_float32_and_float16_loads(monkeypatch):
    cuda = SimpleNamespace(device = "cuda")
    monkeypatch.delenv(mc.MANUAL_CAST_ENV, raising = False)
    monkeypatch.delenv(f3.HALF_STORAGE_ENV, raising = False)
    assert f3.flux3_storage_dtype(cuda, torch.float32) is torch.bfloat16
    assert f3.flux3_storage_dtype(cuda, torch.float16) is torch.bfloat16
    assert f3.flux3_storage_dtype(cuda, torch.bfloat16) is None
    for device in ("cpu", "mps", "xpu", None):
        assert f3.flux3_storage_dtype(SimpleNamespace(device = device), torch.float32) is None
    assert f3.flux3_storage_dtype(None, torch.float32) is None
    for off in ("0", "off", "false", "no"):
        monkeypatch.setenv(mc.MANUAL_CAST_ENV, off)
        assert f3.flux3_storage_dtype(cuda, torch.float32) is None


def _write_safetensors(path, dtypes):
    header = {}
    offset = 0
    for i, dt in enumerate(dtypes):
        header[f"t{i}.weight"] = {"dtype": dt, "shape": [2], "data_offsets": [offset, offset + 4]}
        offset += 4
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)


@pytest.mark.parametrize(
    "dtypes, stored",
    [
        (["BF16", "BF16"], True),
        (["BF16", "I64"], True),
        (["BF16", "F32"], False),
        (["F16", "F16"], False),
        (["F32"], False),
    ],
)
def test_manual_cast_load_reads_the_stored_dtype(tmp_path, dtypes, stored):
    sub = tmp_path / "transformer"
    sub.mkdir()
    _write_safetensors(sub / "diffusion_pytorch_model.safetensors", dtypes)
    seen = []

    def loader(load_dtype):
        seen.append(load_dtype)
        return torch.nn.Sequential(torch.nn.LayerNorm(4), torch.nn.Linear(4, 4)).to(load_dtype)

    module = f3._manual_cast_load(
        loader,
        name = "transformer",
        repo_id = str(tmp_path),
        dtype = torch.float32,
        storage_dtype = torch.bfloat16,
        hf_token = None,
        local_files_only = True,
    )
    if stored:
        assert seen == [torch.bfloat16]
        assert type(module[1]).__name__ == "ManualCastLinear"
        assert module[1].weight.dtype is torch.bfloat16
    else:
        # not exactly bf16 on disk: widened load (bf16 would round it)
        assert seen == [torch.float32]
        assert mc.manual_cast_info(module) is None


def test_manual_cast_load_without_storage_dtype_is_the_widened_load(tmp_path):
    seen = []
    module = f3._manual_cast_load(
        lambda d: seen.append(d) or torch.nn.Linear(2, 2).to(d),
        name = "text_encoder",
        repo_id = str(tmp_path),
        dtype = torch.float16,
        storage_dtype = None,
        hf_token = None,
        local_files_only = True,
    )
    assert seen == [torch.float16] and type(module) is torch.nn.Linear


def test_missing_component_keeps_the_widened_load(tmp_path):
    assert not f3._stored_as(str(tmp_path), "transformer", torch.bfloat16, None, True)


def test_every_flux3_load_site_passes_the_storage_dtype():
    from core.inference import diffusion as d

    tree = ast.parse(inspect.getsource(d))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "load_flux3_pipeline"
    ]
    assert len(calls) == 3
    for call in calls:
        kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
        assert kw.get("storage_dtype", "").startswith("flux3_storage_dtype(target"), kw
    # a quantised transformer stays widened: its quantiser rebuilds the Linears
    gated = [c for c in calls if any(k.arg == "transformer_storage" for k in c.keywords)]
    assert len(gated) == 1
    text = ast.unparse(next(k.value for k in gated[0].keywords if k.arg == "transformer_storage"))
    for name in (
        "transformer_quant_pinned is None",
        "native_scheme is None",
        "native_offload_scheme is None",
    ):
        assert name in text


def test_load_flux3_pipeline_routes_both_components_through_the_cast():
    src = inspect.getsource(f3.load_flux3_pipeline)
    assert src.count("_manual_cast_load(") == 2
    assert "transformer_load_storage(" in src
    helper = inspect.getsource(f3.transformer_load_storage)
    assert "transformer_storage_dtype(storage_dtype, dtype, transformer_storage)" in helper


def test_fuse_qkv_skips_a_manual_cast_denoiser():
    from core.inference import diffusion_speed as ds

    transformer = torch.nn.Sequential(torch.nn.Linear(4, 4).to(torch.bfloat16))
    mc.apply_manual_cast_(transformer, torch.float32)
    calls = []

    class _Pipe:
        def __init__(self):
            self.transformer = transformer

        def fuse_qkv_projections(self):
            calls.append(1)

    assert ds._fuse_qkv(_Pipe(), None) is False
    assert calls == []
