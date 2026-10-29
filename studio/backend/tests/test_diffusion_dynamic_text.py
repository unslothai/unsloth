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


class Flux3Transformer2DModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, x):
        self.seen = _cfg().dynamic_sources
        return x


def test_flux3_block_inputs_are_armed():
    """Both repeated blocks see the prompt length: the per-mode block's text call and the joint block's text stream."""
    m = Flux3Transformer2DModel()
    assert dt.install(m) is True
    m(torch.zeros(1))
    seen = m.seen.split(",")
    for name in (
        "L['hidden_states']",
        "L['temb_mod']",
        "L['encoder_hidden_states']",
        "L['temb_mod_context']",
        "L['image_rotary_emb'][0]",
        "L['image_rotary_emb'][1]",
    ):
        assert name in seen
    # No slice bound among them, so the slice identity patch is not needed for this family.
    assert not dt._arms_slice_bound(dt.sources_for(m))
    # Automatic dynamic only: a blanket dynamic=True compile arms nothing, a static one nothing either.
    assert (
        dt.fingerprint(m, None)
        and dt.fingerprint(m, True) is None
        and dt.fingerprint(m, False) is None
    )


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
        assert m.seen_unbacked.split(",") == ["L['user_thing']", *dt.unbacked_sources_for(m)]
        assert dt.unbacked_sources_for(m)[0] == "L['temb']"
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
        assert m.seen == before
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
    want = (
        "unbacked:" + ",".join(dt._FAMILY_UNBACKED[type(h3).__name__])
        if dt.unbacked_supported()
        else None
    )
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


class _Q21Block(torch.nn.Module):
    """The shape of diffusers' QwenImage21 prefill: ``cache_write_slice is not None`` guards the prefix KV write."""

    def forward(
        self,
        hidden_states,
        kv_cache_mode = None,
        cache_write_slice = None,
    ):
        if kv_cache_mode == "extract" and cache_write_slice is not None:
            return hidden_states[:, cache_write_slice].clone() * 2
        return hidden_states * 3


class _Q21Transformer(torch.nn.Module):
    def __init__(self, dynamic = None):
        super().__init__()
        self.block = torch.compile(_Q21Block(), dynamic = dynamic)

    def forward(self, prefix_len, total):
        x = torch.randn(1, total, 8)
        out = self.block(x, kv_cache_mode = "extract", cache_write_slice = slice(0, prefix_len))
        torch.testing.assert_close(out, x[:, :prefix_len] * 2)
        return out


@pytest.mark.parametrize("dynamic", [None, True])
def test_new_prompt_length_does_not_recompile_the_prefix_kv_write(dynamic):
    """torch 2.13+ guarded ``cache_write_slice.stop`` via ``is not None``: each new prompt length recompiled the block.
    dynamic=True is an unquantized DiT's default tier once the divisibility backport is in."""
    from torch._dynamo.utils import counters

    Transformer = type("QwenImage21Transformer2DModel", (_Q21Transformer,), {})
    torch._dynamo.reset()
    counters.clear()
    m = Transformer(dynamic)
    dt.install(m, dynamic = dynamic)
    try:
        for prefix_len, total in ((26, 90), (31, 95), (40, 104)):
            m(prefix_len, total)
        assert counters["stats"]["unique_graphs"] == 1
    finally:
        dt.uninstall(m)
        dt.uninstall_slice_identity_fix()
        torch._dynamo.reset()


def test_slice_identity_fix_keeps_identity_answers():
    """Patched or not, a slice is never None and constant slices still fold."""
    dt.install_slice_identity_fix()
    try:

        def f(x, s):
            a = 1 if s is None else 2
            b = 1 if s is not None else 2
            return x[:, s] * a + b

        c = torch.compile(f, fullgraph = True)
        x = torch.randn(2, 16)
        for s in (slice(0, 4), slice(2, None), slice(None, 8, 2)):
            torch.testing.assert_close(c(x, s), f(x, s))
    finally:
        dt.uninstall_slice_identity_fix()
        torch._dynamo.reset()


def test_slice_identity_fix_is_probe_gated_and_reversible(monkeypatch):
    from torch._dynamo.variables import SliceVariable

    stock = SliceVariable.__dict__.get("get_real_python_backed_value")
    monkeypatch.setenv(dt.SLICE_IDENTITY_ENV, "0")
    assert dt.install_slice_identity_fix() is False
    assert SliceVariable.__dict__.get("get_real_python_backed_value") is stock
    monkeypatch.delenv(dt.SLICE_IDENTITY_ENV)
    active = dt.install_slice_identity_fix()
    try:
        # Only a torch whose stock identity test reads the slice's bounds (2.13+) is patched.
        assert active == (
            stock is None
            and callable(getattr(SliceVariable, "get_real_python_backed_value", None))
            and dt._SLICE_STATE.get("original") is not None
        )
    finally:
        dt.uninstall_slice_identity_fix()
    assert SliceVariable.__dict__.get("get_real_python_backed_value") is stock


class Float8Tensor(torch.Tensor):
    """Stands in for torchao's float8 weight subclass (its module and class name are read)."""


Float8Tensor.__module__ = "torchao.quantization.quantize_.workflows.float8.float8_tensor"


def _h3_with_float8_weight():
    m = MiniMaxH3Transformer3DModel()
    m.proj = torch.nn.Linear(8, 8)
    m.proj.weight = torch.nn.Parameter(
        torch.Tensor._make_subclass(Float8Tensor, torch.zeros(8, 8)), requires_grad = False
    )
    assert type(m.proj.weight).__name__ == "Float8Tensor"
    return m


def test_minimax_h3_float8_weights_keep_temb_backed_dynamic():
    """torchao float8 activation quantisation guards on the row count, so an unbacked temb failed the compile."""
    if not dt.unbacked_supported():
        pytest.skip("torch lacks compiler.config.unbacked_sources")
    m = _h3_with_float8_weight()
    assert dt.unbacked_sources_for(m) == ()
    assert "L['temb']" in dt.sources_for(m)
    assert dt.install(m) is True
    m(12, 2)
    assert "L['temb']" in m.seen.split(",")
    assert "L['temb']" not in (m.seen_unbacked or "").split(",")
    dt.uninstall(m)
    # a plain (bf16 / int8) H3 keeps the unbacked temb
    assert dt.unbacked_sources_for(MiniMaxH3Transformer3DModel())[0] == "L['temb']"
    assert not set(dt.unbacked_sources_for(MiniMaxH3Transformer3DModel())) & set(
        dt.sources_for(MiniMaxH3Transformer3DModel())
    )


def test_minimax_h3_float8_inside_a_plain_parameter_keeps_temb_backed_dynamic():
    """Some torchao builds keep ``Linear.weight`` a plain Parameter whose ``.data`` is the float8 subclass."""
    if not dt.unbacked_supported():
        pytest.skip("torch lacks compiler.config.unbacked_sources")
    m = MiniMaxH3Transformer3DModel()
    plain = types.SimpleNamespace(data = torch.Tensor._make_subclass(Float8Tensor, torch.zeros(8, 8)))
    m.parameters = lambda recurse = True: iter([plain])
    assert dt.unbacked_sources_for(m) == ()
    assert "L['temb']" in dt.sources_for(m)


class ZImageTransformer2DModel(torch.nn.Module):
    """Z-Image's one block class runs the caption refiner (caption tokens), the noise refiner (image tokens) and the
    main layers (caption + image), positionally as (x, attn_mask, freqs_cis); the caption is padded to 32-token buckets."""

    def __init__(self):
        super().__init__()
        self.block = torch.compile(self._block, backend = "eager", dynamic = None)

    @staticmethod
    def _block(x, attn_mask, freqs_cis):
        return (x * freqs_cis).masked_fill(~attn_mask[..., None], 0.0) + x

    def forward(self, caption_len, image_len):
        def run(n):
            x = torch.randn(1, n, 8)
            return self.block(x, torch.ones(1, n, dtype = torch.bool), torch.randn(1, n, 8))

        run(caption_len)
        run(image_len)
        return run(caption_len + image_len)


def test_zimage_new_caption_bucket_reuses_the_first_graphs():
    """An automatic-dynamic Z-Image recompiled its block for each new length (caption, image, unified) inside the
    first render, then again on the first longer prompt; armed, the first compile covers every length."""
    from torch._dynamo.utils import counters

    def graphs_per_render(install):
        torch._dynamo.reset()
        counters.clear()
        m = ZImageTransformer2DModel()
        if install:
            assert dt.install(m)
        seen = []
        for caption_len, image_len in ((32, 64), (32, 64), (96, 64), (32, 48)):
            before = counters["stats"]["unique_graphs"]
            m(caption_len, image_len)
            seen.append(counters["stats"]["unique_graphs"] - before)
        dt.uninstall(m)
        torch._dynamo.reset()
        return seen

    assert graphs_per_render(False)[0] > 1
    assert graphs_per_render(True) == [1, 0, 0, 0]


class FluxTransformer2DModel(torch.nn.Module):
    """FLUX.1's two compiled blocks: the double block sees the text stream beside the image tokens, the single block
    concatenates them, and the rotary tables span text + image."""

    def __init__(self):
        super().__init__()
        self.seen = None
        self.double = torch.compile(self._double, backend = "eager", dynamic = None)
        self.single = torch.compile(self._single, backend = "eager", dynamic = None)

    @staticmethod
    def _double(hidden_states, encoder_hidden_states, temb, image_rotary_emb):
        joint = (
            torch.cat([encoder_hidden_states, hidden_states], dim = 1) * image_rotary_emb[0]
            + image_rotary_emb[1]
        )
        text = encoder_hidden_states.shape[1]
        return joint[:, text:] + temb[:, None], joint[:, :text]

    @staticmethod
    def _single(hidden_states, encoder_hidden_states, temb, image_rotary_emb):
        joint = torch.cat([encoder_hidden_states, hidden_states], dim = 1)
        out = joint * image_rotary_emb[0] + image_rotary_emb[1] + temb[:, None]
        return out[:, encoder_hidden_states.shape[1] :]

    def forward(self, text_len, image_len):
        self.seen = _cfg().dynamic_sources
        hidden_states = torch.randn(1, image_len, 8)
        encoder_hidden_states = torch.randn(1, text_len, 8)
        temb = torch.randn(1, 8)
        rotary = (torch.randn(text_len + image_len, 8), torch.randn(text_len + image_len, 8))
        hidden_states, encoder_hidden_states = self.double(
            hidden_states, encoder_hidden_states, temb, rotary
        )
        return self.single(hidden_states, encoder_hidden_states, temb, rotary)


def test_flux1_new_prompt_length_and_resolution_reuse_the_first_graphs():
    """T5 pads to 256 or 512 tokens; a torchao FLUX.1 compiles with automatic dynamic, which recompiled both blocks on
    the first 512-token prompt and again on the first new resolution."""
    from torch._dynamo.utils import counters

    def graphs_per_render(install):
        torch._dynamo.reset()
        counters.clear()
        m = FluxTransformer2DModel()
        if install:
            assert dt.install(m)
        seen = []
        for text_len, image_len in ((16, 64), (16, 64), (32, 64), (16, 48), (32, 80)):
            before = counters["stats"]["unique_graphs"]
            m(text_len, image_len)
            seen.append(counters["stats"]["unique_graphs"] - before)
        dt.uninstall(m)
        torch._dynamo.reset()
        return seen

    assert graphs_per_render(False)[2:] != [0, 0, 0]
    assert graphs_per_render(True) == [2, 0, 0, 0, 0]


def test_qwen_image_21_hook_paths_match_and_first_segment_start_stays_static():
    from torch._dynamo.variables.builder import is_dynamic_source as is_dynamic

    cfg = _cfg()
    before = cfg.dynamic_sources
    try:
        cfg.dynamic_sources = ",".join(dt.sources_for(QwenImage21Transformer2DModel()))
        for root in ("L['kwargs']", "___stack0[1]"):
            for leaf in (
                "['hidden_states']",
                "['rotary_emb']",
                "['target_token_mask']",
                "['key_valid']",
                "['attention_mask']",
                "['layer_cache'].k",
                "['layer_cache'].v",
                "['cache_write_slice'].stop",
                "['segments'][0][1]",
                "['segments'][2][0]",
            ):
                assert is_dynamic(root + leaf), root + leaf
            for leaf in (
                "['segments'][0][0]",
                "['modulation']",
                "['cache_write_slice'].start",
                "['hidden_states_2']",
            ):
                assert not is_dynamic(root + leaf), root + leaf
    finally:
        cfg.dynamic_sources = before


class _Q21HookedBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8)

    def forward(
        self,
        hidden_states,
        rotary_emb = None,
        attention_mask = None,
        kv_cache_mode = None,
    ):
        return self.proj(hidden_states) * rotary_emb[..., :1] + attention_mask.float().mean()


class _Q21HookedTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([_Q21HookedBlock() for _ in range(2)])

    def forward(self, x, rope, mask):
        for block in self.transformer_blocks:
            x = block(hidden_states = x, rotary_emb = rope, attention_mask = mask, kv_cache_mode = "cached")
        return x


def test_qwen_image_21_under_group_offload_hooks_keeps_one_graph_per_block():
    """An edit request at 16 GB hooks the resident transformer; a new prompt length then recompiled every block."""
    hooks = pytest.importorskip("diffusers.hooks")
    from torch._dynamo.utils import counters

    from core.inference.diffusion_memory import install_group_offload_hooks_eager

    Transformer = type("QwenImage21Transformer2DModel", (_Q21HookedTransformer,), {})
    torch._dynamo.reset()
    counters.clear()
    m = Transformer()
    for block in m.transformer_blocks:
        block.compile(backend = "eager", dynamic = None)
    assert dt.install(m) is True
    install_group_offload_hooks_eager()
    cpu = torch.device("cpu")
    hooks.apply_group_offloading(
        m, onload_device = cpu, offload_device = cpu, offload_type = "block_level", num_blocks_per_group = 1
    )
    try:
        with torch.no_grad():
            m(torch.randn(1, 90, 8), torch.randn(90, 2), torch.ones(1, 1, 1, 90, dtype = torch.bool))
            first = counters["stats"]["unique_graphs"]
            for n in (95, 93, 89):
                m(torch.randn(1, n, 8), torch.randn(n, 2), torch.ones(1, 1, 1, n, dtype = torch.bool))
        assert first > 0
        assert counters["stats"]["unique_graphs"] == first
    finally:
        dt.uninstall(m)
        torch._dynamo.reset()


class _Lin(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8)


class _Flux3ModeBlock(_Lin):
    def forward(self, hidden_states, temb_mod, image_rotary_emb):
        return self.proj(hidden_states) * image_rotary_emb[0] + image_rotary_emb[1] + temb_mod


class _Flux3SingleBlock(_Lin):
    def forward(
        self, hidden_states, encoder_hidden_states, temb_mod, temb_mod_context, image_rotary_emb
    ):
        joint = (
            torch.cat([encoder_hidden_states, hidden_states], 1) * image_rotary_emb[0]
            + image_rotary_emb[1]
        )
        text = encoder_hidden_states.shape[1]
        return self.proj(joint[:, :text]) + temb_mod_context, joint[:, text:] + temb_mod


class _Flux3Hooked(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([_Flux3ModeBlock() for _ in range(2)])
        self.single_transformer_blocks = torch.nn.ModuleList(
            [_Flux3SingleBlock() for _ in range(2)]
        )

    def forward(
        self,
        text_len,
        image_len = 64,
    ):
        txt, img, mod = (
            torch.randn(1, text_len, 8),
            torch.randn(1, image_len, 8),
            torch.randn(1, 1, 8),
        )
        txt_rope = (torch.randn(text_len, 8), torch.randn(text_len, 8))
        img_rope = (torch.randn(image_len, 8), torch.randn(image_len, 8))
        for block in self.transformer_blocks:
            txt = block(txt, mod, txt_rope)
        for block in self.transformer_blocks:
            img = block(img, mod, img_rope)
        rope = (torch.cat([txt_rope[0], img_rope[0]]), torch.cat([txt_rope[1], img_rope[1]]))
        for block in self.single_transformer_blocks:
            txt, img = block(
                hidden_states = img,
                encoder_hidden_states = txt,
                temb_mod = mod,
                temb_mod_context = mod,
                image_rotary_emb = rope,
            )
        return img


class _ZImageBlock(_Lin):
    def forward(self, x, attn_mask, freqs_cis):
        return self.proj(x * freqs_cis).masked_fill(~attn_mask[..., None], 0.0) + x


class _ZImageHooked(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.context_refiner = torch.nn.ModuleList([_ZImageBlock()])
        self.noise_refiner = torch.nn.ModuleList([_ZImageBlock()])
        self.layers = torch.nn.ModuleList([_ZImageBlock() for _ in range(2)])

    def forward(
        self,
        text_len,
        image_len = 64,
    ):
        def args(n):
            return torch.randn(1, n, 8), torch.ones(1, n, dtype = torch.bool), torch.randn(1, n, 8)

        cap, img = args(text_len), args(image_len)
        for block in self.context_refiner:
            cap = (block(*cap), *cap[1:])
        for block in self.noise_refiner:
            img = (block(*img), *img[1:])
        x = tuple(torch.cat([a, b], 1) for a, b in zip(img, cap))
        for block in self.layers:
            x = (block(*x), *x[1:])
        return x[0]


class _MossBlock(_Lin):
    def forward(self, x, rope, modulation):
        return self.proj(x) * rope[0] + rope[1] + modulation


class _MossHooked(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_MossBlock() for _ in range(2)])

    def forward(
        self,
        text_len,
        image_len = 64,
    ):
        n = text_len + image_len
        x, rope, mod = (
            torch.randn(1, n, 8),
            (torch.randn(n, 8), torch.randn(n, 8)),
            torch.randn(1, 1, 8),
        )
        for layer in self.layers:
            x = layer(x, rope, mod)
        return x


class _H3Block(_Lin):
    def forward(self, hidden_states, temb, adaln_indices, rotary_emb):
        mod = temb.index_select(0, adaln_indices)
        return (self.proj(hidden_states) * rotary_emb[0] + rotary_emb[1]) * (1 + mod)


class _H3Hooked(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([_H3Block() for _ in range(2)])

    def forward(
        self,
        text_len,
        image_len = 64,
    ):
        n = text_len + image_len
        x, rope = torch.randn(1, n, 8), (torch.randn(n, 8), torch.randn(n, 8))
        for n_timesteps in (1, 2):
            temb, idx = torch.randn(n_timesteps, 8), torch.arange(n) % n_timesteps
            for block in self.transformer_blocks:
                x = block(x, temb, idx, rope)
        return x


class _FluxHookedBlock(_Lin):
    def forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb):
        joint = (
            torch.cat([encoder_hidden_states, hidden_states], 1) * image_rotary_emb[0]
            + image_rotary_emb[1]
        )
        text = encoder_hidden_states.shape[1]
        return joint[:, :text], self.proj(joint[:, text:]) + temb[:, None]


class _FluxHooked(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([_FluxHookedBlock() for _ in range(2)])

    def forward(
        self,
        text_len,
        image_len = 64,
    ):
        txt, img, temb = (
            torch.randn(1, text_len, 8),
            torch.randn(1, image_len, 8),
            torch.randn(1, 8),
        )
        n = text_len + image_len
        rope = (torch.randn(n, 8), torch.randn(n, 8))
        for block in self.transformer_blocks:
            txt, img = block(
                hidden_states = img, encoder_hidden_states = txt, temb = temb, image_rotary_emb = rope
            )
        return img


_HOOKED_FAMILIES = {
    "FluxTransformer2DModel": _FluxHooked,
    "Flux3Transformer2DModel": _Flux3Hooked,
    "ZImageTransformer2DModel": _ZImageHooked,
    "MOSSImageTransformer": _MossHooked,
    "MiniMaxH3Transformer3DModel": _H3Hooked,
}


def _hooked_graphs_per_render(name):
    hooks = pytest.importorskip("diffusers.hooks")
    from torch._dynamo.utils import counters

    from core.inference.diffusion_memory import install_group_offload_hooks_eager

    torch._dynamo.reset()
    counters.clear()
    m = type(name, (_HOOKED_FAMILIES[name],), {})()
    for block in m.modules():
        if isinstance(block, _Lin):
            block.compile(backend = "eager", dynamic = None)
    assert dt.install(m) is True
    install_group_offload_hooks_eager()
    cpu = torch.device("cpu")
    hooks.apply_group_offloading(
        m, onload_device = cpu, offload_device = cpu, offload_type = "block_level", num_blocks_per_group = 1
    )
    seen = []
    try:
        with torch.no_grad():
            for text_len in (16, 16, 24, 20, 40):
                before = counters["stats"]["unique_graphs"]
                m(text_len)
                seen.append(counters["stats"]["unique_graphs"] - before)
    finally:
        dt.uninstall(m)
        torch._dynamo.reset()
    return seen


@pytest.mark.parametrize("name", sorted(_HOOKED_FAMILIES))
def test_new_prompt_length_under_group_offload_hooks_reuses_the_first_graphs(name):
    """Group offload hooks wrap block inputs as ``L['args'][i]`` / ``L['kwargs'][...]`` / ``L['___stack0'][...]``."""
    seen = _hooked_graphs_per_render(name)
    assert seen[0] > 0 and seen[1:] == [0, 0, 0, 0], seen
    if name == "ZImageTransformer2DModel" or (
        name == "MiniMaxH3Transformer3DModel" and dt.unbacked_supported()
    ):
        assert seen[0] == 1, seen
