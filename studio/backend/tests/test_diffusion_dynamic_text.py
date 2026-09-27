# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import types

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
    from torch._dynamo.variables.builder import is_dynamic_source as is_dynamic

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
    from torch._dynamo.variables.builder import is_dynamic_source as is_dynamic

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


class MiniMaxH3Transformer3DModel(torch.nn.Module):
    def __init__(self, dynamic = None):
        super().__init__()
        self.seen = None
        self.block = torch.compile(self._block, backend = "eager", dynamic = dynamic)

    @staticmethod
    def _block(hidden_states, temb, adaln_indices, rotary_emb):
        mod = temb.index_select(0, adaln_indices)
        return (hidden_states * rotary_emb[0] + rotary_emb[1]) * (1 + mod)

    def forward(self, seq_len, n_timesteps):
        self.seen = _cfg().dynamic_sources
        self.seen_unbacked = getattr(_cfg(), "unbacked_sources", "")
        hidden_states = torch.randn(1, seq_len, 8)
        temb = torch.randn(n_timesteps, 8)
        adaln_indices = torch.arange(seq_len) % n_timesteps
        rotary_emb = (torch.randn(seq_len, 8), torch.randn(seq_len, 8))
        return self.block(hidden_states, temb, adaln_indices, rotary_emb)


def test_minimax_h3_packed_length_is_armed():
    m = MiniMaxH3Transformer3DModel()
    assert dt.install(m) is True
    m(12, 2)
    seen = m.seen.split(",")
    for name in (
        "L['hidden_states']",
        "L['adaln_indices']",
        "L['rotary_emb'][0]",
        "L['rotary_emb'][1]",
    ):
        assert name in seen
    # temb is unbacked where torch has the knob, else dynamic: never both, never neither
    unbacked = [s for s in (m.seen_unbacked or "").split(",") if s]
    assert ("L['temb']" in seen) != ("L['temb']" in unbacked)
    assert ("L['temb']" in unbacked) == dt.unbacked_supported()
    dt.uninstall(m)
    assert _cfg().dynamic_sources == "" or "L['temb']" not in _cfg().dynamic_sources
    if dt.unbacked_supported():
        assert "L['temb']" not in _cfg().unbacked_sources


def test_minimax_h3_new_caption_and_i2v_reuse_the_first_graphs():
    from torch._dynamo.utils import counters

    def graphs_per_render(install):
        torch._dynamo.reset()
        counters.clear()
        m = MiniMaxH3Transformer3DModel()
        if install:
            assert dt.install(m)
        seen = []
        for render in ((100, 1), (100, 2), (104, 1), (104, 2), (130, 3)):
            before = counters["stats"]["unique_graphs"]
            m(*render)
            seen.append(counters["stats"]["unique_graphs"] - before)
        dt.uninstall(m)
        torch._dynamo.reset()
        return seen

    assert graphs_per_render(False)[2:] != [0, 0, 0]
    assert graphs_per_render(True)[2:] == [0, 0, 0]


@pytest.mark.skipif(
    not dt.unbacked_supported(), reason = "torch lacks compiler.config.unbacked_sources"
)
def test_minimax_h3_first_render_compiles_the_block_once():
    """temb has 1 row on the first step and 2 after it: a backed symbol specialises the 1 and compiles twice."""
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    m = MiniMaxH3Transformer3DModel()
    assert dt.install(m)
    try:
        for render in ((100, 1), (100, 2), (104, 1), (130, 3)):
            m(*render)
        assert counters["stats"]["unique_graphs"] == 1
    finally:
        dt.uninstall(m)
        torch._dynamo.reset()


def test_unbacked_list_is_restored_after_forward():
    if not dt.unbacked_supported():
        pytest.skip("torch lacks compiler.config.unbacked_sources")
    cfg = _cfg()
    before = cfg.unbacked_sources
    try:
        cfg.unbacked_sources = "L['user_thing']"
        m = MiniMaxH3Transformer3DModel()
        dt.install(m)
        m(12, 1)
        assert m.seen_unbacked.split(",") == ["L['user_thing']", "L['temb']"]
        assert cfg.unbacked_sources == "L['user_thing']"
        dt.uninstall(m)
    finally:
        cfg.unbacked_sources = before


def test_fingerprint_keys_on_the_unbacked_list(monkeypatch):
    m = MiniMaxH3Transformer3DModel()
    fp = dt.fingerprint(m, None)
    assert fp
    assert ("unbacked:L['temb']" in fp) == dt.unbacked_supported()
    monkeypatch.setattr(dt, "unbacked_supported", lambda: False)
    fp_old = dt.fingerprint(m, None)
    assert "unbacked" not in fp_old and "L['temb']" in fp_old


def test_dense_h3_dynamic_true_arms_only_the_unbacked_temb():
    """A dense (non-torchao) H3 compiles with dynamic=True: every dim is already dynamic, only temb must go unbacked."""
    m = MiniMaxH3Transformer3DModel(dynamic = True)
    before = _cfg().dynamic_sources
    installed = dt.install(m, dynamic = True)
    assert installed is dt.unbacked_supported()
    try:
        m(12, 2)
        assert m.seen == before  # the prompt-length list is left alone under dynamic=True
        if dt.unbacked_supported():
            assert "L['temb']" in m.seen_unbacked.split(",")
            assert "L['temb']" not in (_cfg().unbacked_sources or "").split(",")
    finally:
        dt.uninstall(m)


def test_static_compile_and_unarmed_families_are_not_hooked():
    m = MiniMaxH3Transformer3DModel()
    assert dt.install(m, dynamic = False) is False
    assert not m._forward_pre_hooks
    other = QwenImage21Transformer2DModel()
    assert dt.install(other, dynamic = True) is False
    assert not other._forward_pre_hooks


def test_fingerprint_for_dynamic_true_keys_only_on_unbacked():
    h3 = MiniMaxH3Transformer3DModel()
    want = "unbacked:L['temb']" if dt.unbacked_supported() else None
    assert dt.fingerprint(h3, True) == want
    assert dt.fingerprint(h3, False) is None
    assert dt.fingerprint(QwenImage21Transformer2DModel(), True) is None
    assert dt.fingerprint(OtherTransformer(), True) is None


@pytest.mark.skipif(
    not dt.unbacked_supported(), reason = "torch lacks compiler.config.unbacked_sources"
)
def test_dense_h3_first_render_compiles_the_block_once():
    """dynamic=True still specialises a backed size of 1, so without the unbacked temb the first render compiles
    twice (temb 1 row, then 2)."""
    from torch._dynamo.utils import counters

    def graphs(install):
        torch._dynamo.reset()
        counters.clear()
        m = MiniMaxH3Transformer3DModel(dynamic = True)
        if install:
            assert dt.install(m, dynamic = True)
        try:
            for render in ((100, 1), (100, 2), (104, 1), (130, 3)):
                m(*render)
            return counters["stats"]["unique_graphs"]
        finally:
            dt.uninstall(m)
            torch._dynamo.reset()

    assert graphs(False) == 2
    assert graphs(True) == 1


def test_regional_compile_of_a_dense_h3_arms_the_unbacked_temb(monkeypatch):
    """Studio's own regional compile: a dense H3 resolves to dynamic=True and must still arm temb."""
    from core.inference import diffusion_speed as ds

    monkeypatch.setattr(ds, "_install_inductor_backports", lambda logger: False)
    # _compile_repeated_blocks sets these process-wide; monkeypatch restores them after the test.
    import torch._dynamo.config as dynamo_cfg
    import torch._inductor.config as inductor_cfg

    for cfg, name in (
        (dynamo_cfg, "recompile_limit"),
        (dynamo_cfg, "cache_size_limit"),
        (inductor_cfg, "emulate_precision_casts"),
    ):
        if hasattr(cfg, name):
            monkeypatch.setattr(cfg, name, getattr(cfg, name))
    seen = {}

    class MiniMaxH3Transformer3DModel(torch.nn.Module):  # noqa: N801 - the family key is the class name
        _repeated_blocks = ["MiniMaxH3TransformerBlock"]

        def compile_repeated_blocks(self, **kwargs):
            seen.update(kwargs)

    m = MiniMaxH3Transformer3DModel()
    try:
        assert ds._compile_repeated_blocks(types.SimpleNamespace(transformer = m), None) is True
        assert seen["dynamic"] is True
        assert (getattr(m, "_unsloth_dynamic_text", None) is not None) is dt.unbacked_supported()
    finally:
        dt.uninstall(m)
