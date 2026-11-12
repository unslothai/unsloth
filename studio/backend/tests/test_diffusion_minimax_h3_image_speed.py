# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3-Image speed levers on a tiny random H3-Image DiT (CPU)."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
for p in (_BACKEND, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("accelerate")

from core.inference import diffusion_minimax_h3_image as h3i  # noqa: E402
from core.inference import diffusion_minimax_h3_image_speed as S  # noqa: E402
from core.inference import video_minimax_h3_attn as A  # noqa: E402


@pytest.fixture(autouse = True)
def _restore_backend():
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

    active = _AttentionBackendRegistry._active_backend
    yield
    _AttentionBackendRegistry.set_active_backend(active)


def _model(dtype = torch.float32):
    import _minimax_h3_image_tiny as tiny

    cfg = h3i.transformer_config_from_native(tiny.TINY_TRANSFORMER_NATIVE)
    torch.manual_seed(0)
    model = h3i.transformer_class()(**cfg).eval()
    tiny.randomize(model, 0)
    return model.to(dtype)


def _inputs(
    model,
    dtype = torch.float32,
    n_text = 5,
    lh = 8,
    lw = 8,
):
    g = torch.Generator().manual_seed(1)
    patch = tuple(model.config.patch_size)
    tags = torch.full((n_text,), h3i.H3_IMAGE_TEXT_TAG, dtype = torch.long)
    pos, tags, image_idx, text_idx, _n_ref = h3i.build_packed_layout(tags, [], lh, lw, patch)
    seq = int(image_idx.numel() + text_idx.numel())
    unique, index = h3i.row_timestep_plan(torch.tensor([0.75]), seq, image_idx[:0])[0]
    noise = torch.randn((1, 24, 1, lh, lw), generator = g)
    text_dim = int(model.context_embedder.in_features)
    return dict(
        hidden_states = h3i.patchify(noise, patch)[None].to(dtype),
        encoder_hidden_states = torch.randn(1, n_text, text_dim, generator = g).to(dtype),
        timestep = unique,
        timestep_indices = index,
        token_tags = tags,
        position_ids = pos,
        image_indices = image_idx,
        text_indices = text_idx,
        pdd_step_index = 0,
        return_dict = False,
    )


def _run(model, inputs):
    with torch.no_grad():
        return model(**inputs)[0]


def _quantized_padded(model):
    pytest.importorskip("torchao")
    from torchao.quantization import quantize_

    from core.inference.diffusion_transformer_quant import _make_quant_config

    model = model.to(torch.bfloat16)
    quantize_(
        model,
        _make_quant_config("int8"),
        filter_fn = lambda m, fqn: isinstance(m, torch.nn.Linear)
        and fqn.startswith("transformer_blocks.")
        and fqn.endswith(h3i.QUANT_LINEAR_SUFFIXES),
    )
    wrapped = sum(
        h3i.wrap_row_multiple(b, h3i.QUANT_LINEAR_SUFFIXES) for b in model.transformer_blocks
    )
    if not wrapped:
        pytest.skip("this torchao gives no per-row int8 Linears")
    return model


def test_the_vendored_attention_is_recognised_by_structure():
    model = _model()
    mods = A._h3_attention_modules(model)
    assert len(mods) == 4
    base = A.h3_processor_base(mods[0])
    assert base is type(model.transformer_blocks[0].attn.processor)
    cls = A.strided_processor_class(base)
    assert issubclass(cls, base) and cls is A.strided_processor_class(base)
    assert A.h3_processor_base(model.transformer_blocks[0].ff) is None
    assert A.h3_processor_base(torch.nn.Linear(2, 2)) is None


@pytest.mark.parametrize("qk_rope", ["1", "0"])
def test_strided_processor_matches_stock_bit_for_bit(monkeypatch, qk_rope):
    monkeypatch.setenv("UNSLOTH_H3_QK_ROPE", qk_rope)
    model = _model()
    model.set_attention_backend("_native_math")
    stock = copy.deepcopy(model)
    assert A.install_strided_attention(model) == 4
    assert A.strided_attention_count(model) == 4
    proc = model.transformer_blocks[0].attn.processor
    assert A._backend_value(proc._attention_backend) == "_native_math"
    assert proc._unsloth_qk_rope is (qk_rope == "1")
    inputs = _inputs(model)
    assert torch.equal(_run(stock, inputs), _run(model, inputs))


@pytest.mark.parametrize(
    "resident, image_env, video_env, want",
    [
        (False, None, None, False),
        (True, None, None, True),
        (False, "1", None, True),
        (True, "0", None, False),
        (True, None, "0", False),
        (False, "1", "0", False),
    ],
)
def test_fused_qk_rope_default_follows_the_tier(monkeypatch, resident, image_env, video_env, want):
    for env, value in ((S.QK_ROPE_ENV, image_env), ("UNSLOTH_H3_QK_ROPE", video_env)):
        if value is None:
            monkeypatch.delenv(env, raising = False)
        else:
            monkeypatch.setenv(env, value)
    monkeypatch.setattr(
        "core.inference.video_minimax_h3_qknorm.qk_norm_rope_op", lambda: None, raising = False
    )
    model = _model()
    assert A.install_strided_attention(model, qk_rope = S.image_qk_rope_enabled(resident)) == 4
    assert model.transformer_blocks[0].attn.processor._unsloth_qk_rope is want


def test_strided_processor_defers_to_stock_for_an_unpinned_backend():
    model = _model()
    stock = copy.deepcopy(model)
    A.install_strided_attention(model)
    for m in A._h3_attention_modules(model):
        m.processor._attention_backend = None
    inputs = _inputs(model)
    assert torch.equal(_run(stock, inputs), _run(model, inputs))


def test_strided_kill_switch(monkeypatch):
    monkeypatch.setenv(A.STRIDED_ATTN_ENV, "0")
    model = _model()
    assert A.install_strided_attention(model) == 0
    assert A.strided_attention_count(model) == 0


def test_int8_qkv_fuses_through_the_row_pads_exactly():
    model = _quantized_padded(_model())
    stock = copy.deepcopy(model)
    fused = S._fuse_qkv(model, None)
    assert fused == 2
    attn = model.transformer_blocks[0].attn
    assert attn.fused_projections and not hasattr(attn, "to_q")
    assert type(attn.to_qkv).__name__ == "H3ImageRowPad"
    assert type(attn.to_out[0]).__name__ == "H3ImageRowPad"
    assert not model.token_refiner.refiner_blocks[0].attn.fused_projections
    inputs = _inputs(model, torch.bfloat16)
    assert torch.equal(_run(stock, inputs), _run(model, inputs))


def test_fused_qkv_kill_switch_keeps_the_pads(monkeypatch):
    monkeypatch.setenv(A.FUSED_QKV_ENV, "0")
    model = _quantized_padded(_model())
    assert S._fuse_qkv(model, None) == 0
    attn = model.transformer_blocks[0].attn
    assert not attn.fused_projections
    assert all(type(getattr(attn, n)).__name__ == "H3ImageRowPad" for n in ("to_q", "to_k", "to_v"))


def test_shared_qkv_needs_the_int8_swap():
    model = _quantized_padded(_model())
    A.install_strided_attention(model)
    assert A.shared_qkv_count(model) == 0


def test_install_is_off_off_cuda_and_under_the_umbrella_switch(monkeypatch):
    model = _model()
    pipe = type("P", (), {})()
    pipe.transformer = model
    plan = type("Plan", (), {"placement": "dit_resident"})()
    assert S.install_h3_image_speed(pipe, plan, "cpu") == ()
    assert pipe._unsloth_h3_speed_levers == ()
    monkeypatch.setenv(S.SPEED_ENV, "0")
    assert not S.speed_enabled()
    assert S.install_h3_image_speed(pipe, plan, "cuda") == ()
    assert A.strided_attention_count(model) == 0


def test_compile_is_opt_in(monkeypatch):
    monkeypatch.delenv(S.COMPILE_ENV, raising = False)
    assert not S.compile_enabled()
    monkeypatch.setenv(S.COMPILE_ENV, "1")
    assert S.compile_enabled()


def test_the_loader_installs_the_levers_once_and_reports_them():
    import ast
    import inspect

    src = inspect.getsource(h3i.load_minimax_h3_image_pipeline)
    calls = [
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "install_h3_image_speed"
    ]
    assert len(calls) == 1
    assert src.index("install_h3_image_speed(") > src.rindex("early_streamer.remove()")
    assert src.index("install_h3_image_speed(") < src.index("H3ImageLoadReport(")
    assert "speed_levers" in h3i.H3ImageLoadReport.__dataclass_fields__
