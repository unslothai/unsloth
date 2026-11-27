# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.3 / 2.5 connectors build the video- and audio-scaled projection inputs one at a time: bit-identical outputs,
and only the hidden states plus one prompt-sized input alive when each projection runs."""

from __future__ import annotations

import gc
import types

import pytest

torch = pytest.importorskip("torch")
connectors_mod = pytest.importorskip("diffusers.pipelines.ltx2.connectors")

from core.inference import video_ltx2 as L  # noqa: E402

B, SEQ, CAPTION, FACTOR = 2, 12, 16, 3


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv(L.LTX2_LEAN_CONNECTOR_ENV, raising = False)


def _connectors(per_modality = True, dtype = torch.float32):
    torch.manual_seed(0)
    module = connectors_mod.LTX2TextConnectors(
        caption_channels = CAPTION,
        text_proj_in_factor = FACTOR,
        video_connector_num_attention_heads = 2,
        video_connector_attention_head_dim = 8,
        video_connector_num_layers = 1,
        video_connector_num_learnable_registers = 4,
        audio_connector_num_attention_heads = 2,
        audio_connector_attention_head_dim = 4,
        audio_connector_num_layers = 1,
        audio_connector_num_learnable_registers = 4,
        connector_rope_base_seq_len = 64,
        per_modality_projections = per_modality,
        video_hidden_dim = 16,
        audio_hidden_dim = 8,
    )
    return module.to(dtype).eval()


def _inputs(dtype = torch.float32, ndim = 3):
    g = torch.Generator().manual_seed(1)
    hidden = torch.randn(B, SEQ, CAPTION * FACTOR, generator = g).to(dtype)
    if ndim == 4:
        hidden = hidden.unflatten(2, (CAPTION, FACTOR))
    mask = torch.ones(B, SEQ, dtype = torch.int64)
    mask[0, :5] = 0  # left padding on the first prompt
    return hidden, mask


def _prompt_sized_alive(numel):
    return sum(
        1
        for obj in gc.get_objects()
        if type(obj) is torch.Tensor and not obj._is_view() and obj.numel() == numel
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("ndim", [3, 4])
def test_lean_forward_is_bit_identical_to_the_diffusers_forward(dtype, ndim):
    module = _connectors(dtype = dtype)
    hidden, mask = _inputs(dtype, ndim)
    with torch.inference_mode():
        expected = module(hidden, mask)
        assert L.install_ltx2_lean_connector_features(types.SimpleNamespace(connectors = module))
        got = module(hidden, mask)
    assert module._unsloth_lean_connector_runs == 1
    assert len(got) == len(expected) == 3
    for a, b in zip(got, expected):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(a, b)


def test_only_the_hidden_states_and_one_input_are_alive_at_each_projection():
    module = _connectors()
    hidden, mask = _inputs()
    numel = hidden.numel()
    seen = {}

    def spy(name):
        def hook(mod, args):
            gc.collect()
            seen[name] = _prompt_sized_alive(numel)

        return hook

    module.video_text_proj_in.register_forward_pre_hook(spy("video"))
    module.audio_text_proj_in.register_forward_pre_hook(spy("audio"))
    with torch.inference_mode():
        module(hidden, mask)
        diffusers_seen = dict(seen)
        assert L.install_ltx2_lean_connector_features(types.SimpleNamespace(connectors = module))
        module(hidden, mask)
    # diffusers: hidden states, masked norm, video- and audio-scaled copies all alive at the video projection
    assert diffusers_seen == {"video": 4, "audio": 4}
    assert seen == {"video": 2, "audio": 2}


def test_an_offload_hook_registered_after_install_wraps_the_lean_forward():
    from diffusers.hooks import HookRegistry, ModelHook

    module = _connectors()
    hidden, mask = _inputs()
    assert L.install_ltx2_lean_connector_features(types.SimpleNamespace(connectors = module))
    calls = []

    class Spy(ModelHook):
        def pre_forward(self, mod, *args, **kwargs):
            calls.append(mod)
            return args, kwargs

    HookRegistry.check_if_exists_or_initialize(module).register_hook(Spy(), "spy")
    with torch.inference_mode():
        module(hidden, mask)
    assert calls == [module]
    assert module._unsloth_lean_connector_runs == 1


def test_not_installed_where_it_would_change_or_bypass_anything(monkeypatch):
    # LTX-2.0's single projection has no per-modality copies
    assert not L.install_ltx2_lean_connector_features(
        types.SimpleNamespace(connectors = _connectors(per_modality = False))
    )
    # a forward already wrapped on the instance (an offload hook) would be bypassed
    hooked = _connectors()
    hooked.forward = hooked.forward
    assert not L.install_ltx2_lean_connector_features(types.SimpleNamespace(connectors = hooked))
    assert not hasattr(hooked, "_unsloth_lean_connector_features")
    # a diffusers forward this copy was not checked against
    monkeypatch.setattr(L, "_LTX2_CONNECTORS_FORWARD_SHA256", frozenset({"0" * 64}))
    assert not L.install_ltx2_lean_connector_features(
        types.SimpleNamespace(connectors = _connectors())
    )
    monkeypatch.undo()
    monkeypatch.setenv(L.LTX2_LEAN_CONNECTOR_ENV, "0")
    module = _connectors()
    assert not L.install_ltx2_lean_connector_features(types.SimpleNamespace(connectors = module))
    assert "forward" not in module.__dict__
    assert not L.install_ltx2_lean_connector_features(types.SimpleNamespace(connectors = None))


def test_the_pinned_diffusers_forward_is_the_one_the_lean_copy_follows():
    # fails when a diffusers bump changes LTX2TextConnectors.forward: re-check the copy, then add the new digest
    assert L._ltx2_connectors_forward_matches(connectors_mod.LTX2TextConnectors)


def test_compiling_the_lean_connectors_adds_no_recompile():
    """The engagement count is eager-only: the compiled lean forward captures as often as the compiled diffusers
    forward (a counter write inside the trace guarded on its value and recompiled)."""
    from torch._dynamo.testing import CompileCounter

    def captures(lean):
        torch._dynamo.reset()
        module = _connectors()
        if lean:
            assert L.install_ltx2_lean_connector_features(types.SimpleNamespace(connectors = module))
        counter = CompileCounter()
        compiled = torch.compile(module, backend = counter)
        g = torch.Generator().manual_seed(2)
        with torch.inference_mode():
            for seq in (12, 12, 12, 16, 20):
                hidden = torch.randn(1, seq, CAPTION * FACTOR, generator = g)
                compiled(hidden, torch.ones(1, seq, dtype = torch.int64))
        return counter.frame_count

    base = captures(False)
    assert captures(True) == base
