# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Prompt-length dims compiled dynamic from the first forward (``diffusion_dynamic_text.py``)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_dynamic_text as dt  # noqa: E402

pytestmark = pytest.mark.skipif(
    not dt.supported(), reason = "torch lacks a re-read compiler.config.dynamic_sources (2.8+)"
)


def _cfg():
    import torch.compiler.config as cfg
    return cfg


class QwenImage21Transformer2DModel(torch.nn.Module):
    def __init__(self, raise_in_forward = False):
        super().__init__()
        self.seen = None
        self.raise_in_forward = raise_in_forward

    def forward(self, x):
        self.seen = _cfg().dynamic_sources
        if self.raise_in_forward:
            raise RuntimeError("boom")
        return x


class OtherTransformer(torch.nn.Module):
    def forward(self, x):
        return x


def test_allowlist_is_set_only_inside_forward():
    cfg = _cfg()
    before = cfg.dynamic_sources
    m = QwenImage21Transformer2DModel()
    assert dt.install(m) is True
    m(torch.zeros(1))
    for name in (
        "L['hidden_states']",
        "L['layer_cache'].k",
        "L['segments'][0][1]",
        "L['cache_write_slice'].stop",
    ):
        assert name in m.seen.split(",")
    assert cfg.dynamic_sources == before


def test_allowlist_restored_when_forward_raises():
    cfg = _cfg()
    before = cfg.dynamic_sources
    m = QwenImage21Transformer2DModel(raise_in_forward = True)
    dt.install(m)
    with pytest.raises(RuntimeError):
        m(torch.zeros(1))
    assert cfg.dynamic_sources == before


def test_existing_allowlist_is_kept():
    cfg = _cfg()
    before = cfg.dynamic_sources
    try:
        cfg.dynamic_sources = "L['user_thing']"
        m = QwenImage21Transformer2DModel()
        dt.install(m)
        m(torch.zeros(1))
        assert m.seen.split(",")[0] == "L['user_thing']"
        assert cfg.dynamic_sources == "L['user_thing']"
    finally:
        cfg.dynamic_sources = before


def test_other_families_are_untouched():
    m = OtherTransformer()
    assert dt.install(m) is False
    assert not m._forward_pre_hooks and not m._forward_hooks
    assert dt.fingerprint(m, None) is None


def test_install_is_idempotent_and_uninstall_removes_hooks():
    m = QwenImage21Transformer2DModel()
    assert dt.install(m) and dt.install(m)
    assert len(m._forward_pre_hooks) == 1
    dt.uninstall(m)
    assert not m._forward_pre_hooks and not m._forward_hooks


def test_fingerprint_only_for_automatic_dynamic():
    m = QwenImage21Transformer2DModel()
    assert dt.fingerprint(m, None)
    assert dt.fingerprint(m, True) is None
    assert dt.fingerprint(m, False) is None


def test_first_segment_start_stays_static():
    # A symbol for segments[0][0] (always 0) trips torchao's CantSplit.
    builder = pytest.importorskip("torch._dynamo.variables.builder")
    is_dynamic = getattr(builder, "is_dynamic_source", None)
    if is_dynamic is None:
        pytest.skip("regex allowlist needs torch 2.8+")
    cfg = _cfg()
    before = cfg.dynamic_sources
    try:
        cfg.dynamic_sources = ",".join(dt.sources_for(QwenImage21Transformer2DModel()))
        assert not is_dynamic("L['segments'][0][0]")
        for name in (
            "L['segments'][0][1]",
            "L['segments'][2][0]",
            "L['segments'][2][1]",
            "L['layer_cache'].v",
        ):
            assert is_dynamic(name), name
        for name in ("L['modulation']", "L['hidden_states_2']", "L['layer_cache'].kv"):
            assert not is_dynamic(name), name
    finally:
        cfg.dynamic_sources = before


def test_compile_cache_key_changes_only_for_armed_family():
    from core.inference import diffusion_compile_cache as cc

    kwargs = {"fullgraph": True, "dynamic": None, "mode": "max-autotune-no-cudagraphs"}
    other = cc.model_fingerprint(
        family = "x",
        transformer = OtherTransformer(),
        dtype = "bf16",
        quant = None,
        attention_backend = None,
        compile_kwargs = kwargs,
    )
    assert "dynamic_text" not in other
    q21 = cc.model_fingerprint(
        family = "qwen-image-2.1",
        transformer = QwenImage21Transformer2DModel(),
        dtype = "bf16",
        quant = None,
        attention_backend = None,
        compile_kwargs = kwargs,
    )
    assert q21["dynamic_text"]
    q21_default = cc.model_fingerprint(
        family = "qwen-image-2.1",
        transformer = QwenImage21Transformer2DModel(),
        dtype = "bf16",
        quant = None,
        attention_backend = None,
        compile_kwargs = {**kwargs, "dynamic": True},
    )
    assert "dynamic_text" not in q21_default


class QwenImageTransformer2DModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, x):
        self.seen = _cfg().dynamic_sources
        return x


def test_qwen_image_text_stream_is_armed():
    m = QwenImageTransformer2DModel()
    assert dt.install(m) is True
    m(torch.zeros(1))
    seen = m.seen.split(",")
    for name in (
        "L['encoder_hidden_states']",
        "L['encoder_hidden_states_mask']",
        "L['image_rotary_emb'][1]",
    ):
        assert name in seen
    assert "L['hidden_states']" not in seen


def test_qwen_image_hook_paths_match_on_regex_torch():
    builder = pytest.importorskip("torch._dynamo.variables.builder")
    is_dynamic = getattr(builder, "is_dynamic_source", None)
    if is_dynamic is None:
        pytest.skip("regex allowlist needs torch 2.8+")
    cfg = _cfg()
    before = cfg.dynamic_sources
    try:
        cfg.dynamic_sources = ",".join(dt.sources_for(QwenImageTransformer2DModel()))
        for name in (
            "L['kwargs']['encoder_hidden_states']",
            "___stack0[1]['encoder_hidden_states']",
            "L['kwargs']['encoder_hidden_states_mask']",
            "L['kwargs']['image_rotary_emb'][1]",
        ):
            assert is_dynamic(name), name
        for name in (
            "L['hidden_states']",
            "L['kwargs']['hidden_states']",
            "L['kwargs']['image_rotary_emb'][0]",
            "L['temb']",
        ):
            assert not is_dynamic(name), name
    finally:
        cfg.dynamic_sources = before


def test_torch_that_reads_the_allowlist_once_is_not_armed(monkeypatch):
    from torch._dynamo.variables import builder

    monkeypatch.delattr(builder, "is_dynamic_source")
    model = QwenImage21Transformer2DModel()
    assert not dt.supported()
    assert not dt.install(model)
    assert dt.fingerprint(model, None) is None
    model(torch.zeros(1))
    assert model.seen == _cfg().dynamic_sources
