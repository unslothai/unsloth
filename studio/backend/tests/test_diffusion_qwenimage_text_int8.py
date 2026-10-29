# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image streamed int8 text stream (``int8_text_stream_scope``). CPU-only."""

from __future__ import annotations

import pytest

import core.inference.diffusion_transformer_quant as tq
from core.inference.diffusion_transformer_quant import TQ_FP8, TQ_INT8, exclude_tokens_for_scheme

_TEXT_STREAM = ("add_q_proj", "add_k_proj", "add_v_proj", "to_add_out", "txt_mlp")


_UNSCOPED = {
    "test_resident_build_is_main",
    "test_streamed_scope_is_scoped_and_nests",
    "test_streamed_scope_resets_on_error",
    "test_streamed_scope_does_not_leak_into_other_threads",
    "test_load_wraps_lookup_build_and_write_in_the_placement_scope",
}


@pytest.fixture(autouse = True)
def _default_switch(monkeypatch, request):
    monkeypatch.delenv(tq.INT8_TEXT_STREAM_ENV, raising = False)
    if request.node.name in _UNSCOPED:
        yield
        return
    with tq.int8_text_stream_scope(True):
        yield


def test_resident_build_is_main():
    assert tq.runtime_quant_tokens_for_scheme(TQ_INT8, "qwen-image") == ()
    assert exclude_tokens_for_scheme(TQ_INT8, "qwen-image") == exclude_tokens_for_scheme(
        TQ_INT8, "qwen-image", baked = True
    )
    assert tq.pad_tokens_for_scheme(TQ_INT8, "qwen-image") == ()
    with tq.int8_text_stream_scope(False):
        assert tq.runtime_quant_tokens_for_scheme(TQ_INT8, "qwen-image") == ()


def test_streamed_scope_is_scoped_and_nests():
    with tq.int8_text_stream_scope(True):
        assert tq.runtime_quant_tokens_for_scheme(TQ_INT8, "qwen-image") == _TEXT_STREAM
        assert tq.pad_tokens_for_scheme(TQ_INT8, "qwen-image") == _TEXT_STREAM
        with tq.int8_text_stream_scope(False):
            assert tq.runtime_quant_tokens_for_scheme(TQ_INT8, "qwen-image") == ()
        assert tq.runtime_quant_tokens_for_scheme(TQ_INT8, "qwen-image") == _TEXT_STREAM
    assert tq.runtime_quant_tokens_for_scheme(TQ_INT8, "qwen-image") == ()


def test_streamed_scope_resets_on_error():
    with pytest.raises(RuntimeError):
        with tq.int8_text_stream_scope(True):
            raise RuntimeError("load failed")
    assert tq.runtime_quant_tokens_for_scheme(TQ_INT8, "qwen-image") == ()


def test_streamed_scope_does_not_leak_into_other_threads():
    import threading

    seen = []
    with tq.int8_text_stream_scope(True):
        t = threading.Thread(
            target = lambda: seen.append(tq.runtime_quant_tokens_for_scheme(TQ_INT8, "qwen-image"))
        )
        t.start()
        t.join()
    assert seen == [()]


def test_load_wraps_lookup_build_and_write_in_the_placement_scope():
    import inspect

    from core.inference import diffusion

    src = inspect.getsource(diffusion.DiffusionBackend)
    assert src.count("with int8_text_stream_scope(not keeps_resident):") == 3
    for call in (
        "self._pipeline_quant_from_cache(",
        "quantize_transformer(",
        "self._save_pipeline_quant_to_cache(",
    ):
        i = src.index(call, src.index("with int8_text_stream_scope(not keeps_resident):"))
        assert "with int8_text_stream_scope(not keeps_resident):" in src[max(0, i - 400) : i], call


def test_runtime_exclusions_drop_the_text_stream():
    tokens = exclude_tokens_for_scheme(TQ_INT8, "qwen-image")
    for token in _TEXT_STREAM:
        assert token not in tokens
    assert "txt_in" in tokens
    assert set(tq._INT8_EXCLUDE_NAME_TOKENS) <= set(tokens)
    assert tq.runtime_quant_tokens_for_scheme(TQ_INT8, "qwen-image") == _TEXT_STREAM
    assert exclude_tokens_for_scheme(TQ_INT8, " Qwen-Image ") == tokens


def test_baked_list_is_unchanged():
    assert (
        exclude_tokens_for_scheme(TQ_INT8, "qwen-image", baked = True)
        == tq._INT8_EXCLUDE_NAME_TOKENS + tq._QWENIMAGE_INT8_EXCLUDES
    )
    for token in ("txt_in",) + _TEXT_STREAM:
        assert token in tq._QWENIMAGE_INT8_EXCLUDES


@pytest.mark.parametrize("value", ["0", "off", "false", "no", " OFF "])
def test_kill_switch_restores_the_dense_text_stream(monkeypatch, value):
    monkeypatch.setenv(tq.INT8_TEXT_STREAM_ENV, value)
    assert tq.runtime_quant_tokens_for_scheme(TQ_INT8, "qwen-image") == ()
    assert exclude_tokens_for_scheme(TQ_INT8, "qwen-image") == exclude_tokens_for_scheme(
        TQ_INT8, "qwen-image", baked = True
    )


@pytest.mark.parametrize(
    "family", ["qwen-image-edit", "qwen-image-layered", "qwen-image-2.1", "z-image", None]
)
def test_other_families_are_untouched(family):
    # only qwen-image t2i was measured; the edit / layered siblings share the DiT but not the evidence
    assert tq.runtime_quant_tokens_for_scheme(TQ_INT8, family) == ()
    assert exclude_tokens_for_scheme(TQ_INT8, family) == exclude_tokens_for_scheme(
        TQ_INT8, family, baked = True
    )


def test_other_schemes_are_untouched():
    assert tq.runtime_quant_tokens_for_scheme(TQ_FP8, "qwen-image") == ()
    assert exclude_tokens_for_scheme(TQ_FP8, "qwen-image") == ()


def test_runtime_tokens_are_a_subset_of_the_baked_tokens():
    # _validate_checkpoint accepts an artifact whose exclusions cover the runtime list: a hosted artifact built under
    # the baked list must keep validating, so a runtime token can only ever REMOVE a baked one
    for family, runtime in tq._INT8_FAMILY_RUNTIME_QUANT_TOKENS.items():
        baked = tq._INT8_FAMILY_EXCLUDE_NAME_TOKENS.get(family, ())
        assert set(runtime) <= set(baked), family
        assert set(exclude_tokens_for_scheme(TQ_INT8, family)) <= set(
            exclude_tokens_for_scheme(TQ_INT8, family, baked = True)
        )


def test_training_keeps_the_baked_list():
    from core.training.diffusion_dit_trainer import _lowp_exclude_tokens
    tokens = _lowp_exclude_tokens("qwen-image")
    for token in _TEXT_STREAM:
        assert token in tokens


def test_compile_switch_follows_the_kill_switch(monkeypatch):
    from core.inference import diffusion_compile_cache as cc

    assert "UNSLOTH_DIFFUSION_INT8_TEXT_STREAM" in cc._GRAPH_SWITCHES
    on = exclude_tokens_for_scheme(TQ_INT8, "qwen-image")
    monkeypatch.setenv(tq.INT8_TEXT_STREAM_ENV, "0")
    assert exclude_tokens_for_scheme(TQ_INT8, "qwen-image") != on
    assert cc.graph_switches() == {"UNSLOTH_DIFFUSION_INT8_TEXT_STREAM": "off"}


def _tiny_qwen_like(torch):
    nn = torch.nn

    class Attn(nn.Module):
        def __init__(self, d):
            super().__init__()
            for name in ("to_q", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"):
                setattr(self, name, nn.Linear(d, d))

    class FF(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.ModuleList([nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)])

    class Block(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.attn = Attn(d)
            self.txt_mlp = FF(d)
            self.img_mlp = FF(d)
            self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))

    class Model(nn.Module):
        def __init__(self, d = 512):
            super().__init__()
            self.txt_in = nn.Linear(d, d)
            self.transformer_blocks = nn.ModuleList([Block(d), Block(d)])
            self.small = nn.Linear(64, 64)

    return Model().to(torch.bfloat16)


def _kinds(model, torch):
    return {
        name: type(module.weight).__name__
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }


def test_leftovers_quantise_exactly_the_text_stream():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchao")
    model = _tiny_qwen_like(torch)
    baked = list(exclude_tokens_for_scheme(TQ_INT8, "qwen-image", baked = True))
    x = torch.randn(3, 512, dtype = torch.bfloat16)
    before = model.transformer_blocks[0].attn.add_q_proj(x)
    n = tq.quantize_runtime_leftovers(model, TQ_INT8, "qwen-image", baked)
    kinds = _kinds(model, torch)
    text = [k for k in kinds if any(t in k for t in _TEXT_STREAM)]
    assert n == len(text) == 2 * (4 + 2)
    for name, kind in kinds.items():
        quantised = kind not in ("Parameter", "Tensor")
        assert quantised == (name in text), name
    assert tq.quantize_runtime_leftovers(model, TQ_INT8, "qwen-image", baked) == 0
    after = model.transformer_blocks[0].attn.add_q_proj(x)
    assert after.shape == before.shape
    assert torch.isfinite(after.float()).all()
    rel = (after.float() - before.float()).norm() / before.float().norm()
    assert rel < 0.05


def test_leftovers_noop_cases(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchao")
    baked = list(exclude_tokens_for_scheme(TQ_INT8, "qwen-image", baked = True))
    assert (
        tq.quantize_runtime_leftovers(
            _tiny_qwen_like(torch), TQ_INT8, "qwen-image", ["norm", "txt_in"]
        )
        == 0
    )
    assert tq.quantize_runtime_leftovers(_tiny_qwen_like(torch), TQ_INT8, "qwen-image", None) == 0
    assert tq.quantize_runtime_leftovers(_tiny_qwen_like(torch), TQ_FP8, "qwen-image", baked) == 0
    assert (
        tq.quantize_runtime_leftovers(_tiny_qwen_like(torch), TQ_INT8, "qwen-image-edit", baked)
        == 0
    )
    monkeypatch.setenv(tq.INT8_TEXT_STREAM_ENV, "0")
    model = _tiny_qwen_like(torch)
    assert tq.quantize_runtime_leftovers(model, TQ_INT8, "qwen-image", baked) == 0
    assert set(_kinds(model, torch).values()) == {"Parameter"}


def test_leftovers_ignore_a_non_cuda_work_device():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchao")
    model = _tiny_qwen_like(torch)
    baked = list(exclude_tokens_for_scheme(TQ_INT8, "qwen-image", baked = True))
    assert tq.quantize_runtime_leftovers(model, TQ_INT8, "qwen-image", baked, device = "cpu") == 12
    assert all(p.device.type == "cpu" for p in model.parameters())


def test_local_prequant_cache_key_follows_the_switch(monkeypatch):
    pytest.importorskip("torch")
    from core.inference import diffusion_prequant_cache as pc

    def key():
        parts = pc.key_parts(
            family = "qwen-image",
            transformer_class = "QwenImageTransformer2DModel",
            scheme = TQ_INT8,
            component = "transformer",
            source_fp = "fp",
            base = "Qwen/Qwen-Image-2512",
        )
        return pc.cache_key(parts), parts["exclude_name_tokens"]

    on, on_tokens = key()
    with tq.int8_text_stream_scope(False):
        resident, resident_tokens = key()
    monkeypatch.setenv(tq.INT8_TEXT_STREAM_ENV, "0")
    off, off_tokens = key()
    assert on != off
    assert "add_q_proj" in off_tokens and "add_q_proj" not in on_tokens
    assert resident != on and resident == off and resident_tokens == off_tokens
