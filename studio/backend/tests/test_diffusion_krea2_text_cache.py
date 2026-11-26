# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""diffusion_krea2_text_cache on a tiny CPU Krea2Transformer2DModel: bit-identical, one fusion run per prompt."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "Krea2Transformer2DModel"):
    pytest.skip("diffusers without Krea 2", allow_module_level = True)

from core.inference import diffusion_krea2_prefix_kv as pkv  # noqa: E402
from core.inference import diffusion_krea2_text_cache as tc  # noqa: E402
from core.inference import diffusion_krea2_text_trim as trim  # noqa: E402

TEXT, GRID, LAYERS, DIM, BLOCKS = 40, 4, 3, 32, 2


def _model():
    torch.manual_seed(0)
    return diffusers.Krea2Transformer2DModel(
        in_channels = 8,
        num_layers = BLOCKS,
        attention_head_dim = 16,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        intermediate_size = 32,
        timestep_embed_dim = 16,
        text_hidden_dim = DIM,
        num_text_layers = LAYERS,
        text_num_attention_heads = 2,
        text_num_key_value_heads = 2,
        text_intermediate_size = 32,
        num_layerwise_text_blocks = 1,
        num_refiner_text_blocks = 1,
        axes_dims_rope = (4, 6, 6),
    ).eval()


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    for env in (tc.TEXT_CACHE_ENV, trim.TEXT_TRIM_ENV, pkv.PREFIX_ENV, pkv.BACKEND_ENV):
        monkeypatch.delenv(env, raising = False)
    monkeypatch.setattr(trim, "BUCKET", 8)
    for k in tc.STATS:
        tc.STATS[k] = 0


class _Pipe:
    def __init__(self, transformer):
        self.transformer = transformer


def _branch(seed: int, valid: int):
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros(1, TEXT, dtype = torch.bool)
    mask[0, :valid] = True
    mask[0, TEXT - 3 :] = True
    return {
        "encoder_hidden_states": torch.randn(1, TEXT, LAYERS, DIM, generator = g),
        "encoder_attention_mask": mask,
        "position_ids": diffusers.Krea2Pipeline.prepare_position_ids(
            TEXT, GRID, GRID, torch.device("cpu")
        ),
    }


def _render(
    model,
    branches,
    steps = 3,
):
    g = torch.Generator().manual_seed(7)
    latents = torch.randn(1, GRID * GRID, 8, generator = g)
    outs = []
    for i in range(steps):
        t = torch.full((1,), 1.0 - 0.25 * i)
        for b in branches:
            outs.append(model(hidden_states = latents, timestep = t, return_dict = False, **b)[0])
        latents = latents - 0.1 * outs[-1]
    return outs


def _counting(model):
    calls = {n: 0 for n in tc.MODULES}
    for name in tc.MODULES:
        mod = getattr(model, name)
        real = type(mod).forward

        def fwd(
            self,
            *a,
            _n = name,
            _real = real,
            **k,
        ):
            calls[_n] += 1
            return _real(self, *a, **k)

        mod.forward = fwd.__get__(mod)
    return calls


@pytest.mark.parametrize("trimmed", [False, True])
@pytest.mark.parametrize("cfg", [False, True])
@pytest.mark.parametrize("inference", [False, True])
def test_text_modules_run_once_per_prompt_bitwise(trimmed, cfg, inference):
    branches = [_branch(1, 12)] + ([_branch(2, 5)] if cfg else [])
    ctx = torch.inference_mode() if inference else torch.no_grad()
    with ctx:
        model = _model()
        if trimmed:
            pkv.install(model)
            assert trim.install_for_pipe(_Pipe(model))
        want = _render(model, branches)
        calls = _counting(model)
        assert tc.install(model)
        got = _render(model, branches)
    for a, b in zip(want, got):
        assert torch.equal(a, b)
    n = len(branches)
    assert calls == {name: n for name in tc.MODULES}
    assert tc.STATS["misses"] == 3 * n and tc.STATS["hits"] == 3 * n * 2
    trim.uninstall(model)


def test_switch_off_and_new_prompt_recompute(monkeypatch):
    with torch.no_grad():
        model = _model()
        calls = _counting(model)
        assert tc.install(model)
        a, b = _branch(1, 12), _branch(3, 20)
        _render(model, [a], steps = 2)
        _render(model, [b], steps = 2)
        assert calls["text_fusion"] == 2
        monkeypatch.setenv(tc.TEXT_CACHE_ENV, "0")
        _render(model, [b], steps = 2)
        assert calls["text_fusion"] == 4


def test_in_place_edit_of_the_prompt_recomputes():
    with torch.no_grad():
        model = _model()
        calls = _counting(model)
        assert tc.install(model)
        b = _branch(1, 12)
        first = _render(model, [b], steps = 1)
        b["encoder_hidden_states"].mul_(2.0)
        again = _render(model, [b], steps = 1)
    assert calls["text_fusion"] == 2 and not torch.equal(first[0], again[0])


def test_streamed_text_fusion_stays_stock():
    from diffusers.hooks import apply_group_offloading
    with torch.no_grad():
        model = _model()
        apply_group_offloading(
            model.text_fusion,
            onload_device = torch.device("cpu"),
            offload_type = "block_level",
            num_blocks_per_group = 1,
        )
        calls = _counting(model)
        assert tc.install(model)
        _render(model, [_branch(1, 12)])
    assert calls["text_fusion"] == 3 and tc.STATS["hits"] <= 2 * 3  # txt_in / rotary may still hit


def test_grad_calls_run_stock():
    model = _model()
    calls = _counting(model)
    assert tc.install(model)
    _render(model, [_branch(1, 12)], steps = 2)
    assert calls["text_fusion"] == 2 and tc.STATS["hits"] == 0


def test_install_is_per_krea2_instance_and_idempotent():
    model = _model()
    assert tc.install(model) and tc.install(model)
    assert getattr(model.text_fusion.forward, "_unsloth_krea2_text_cache", False)
    assert not tc.install(torch.nn.Linear(2, 2))


def _projection():
    from diffusers.models.transformers.transformer_krea2 import Krea2TextProjection
    torch.manual_seed(0)
    return Krea2TextProjection(8, 8, eps = 1e-6).eval()


def test_compiled_caller_traces_once():
    """State touched inside the trace became a guard failing every call."""
    from torch._dynamo.utils import counters

    proj = _projection()
    wrapped = tc._wrap(proj, proj.forward)
    for config in ({}, {"fullgraph": True}, {"fullgraph": True, "dynamic": True}):
        torch._dynamo.reset()
        counters.clear()
        fn = torch.compile(wrapped, backend = "eager", **config)
        with torch.no_grad():
            for _ in range(5):
                fn(torch.randn(2, 5, 8))
        assert counters["stats"]["unique_graphs"] == 1, config
    torch._dynamo.reset()


def test_autocast_dtype_is_part_of_the_key():
    proj = _projection()
    wrapped = tc._wrap(proj, proj.forward)
    with torch.inference_mode():
        x = torch.randn(2, 5, 8)
        assert wrapped(x).dtype == torch.float32
        with torch.autocast("cpu", dtype = torch.bfloat16):
            want = proj(x)
            got = wrapped(x)
        assert got.dtype == want.dtype == torch.bfloat16 and torch.equal(got, want)
        assert wrapped(x).dtype == torch.float32


def test_entries_end_with_the_render():
    """Inference tensors have no version counter: end_render must drop entries."""
    with torch.inference_mode():
        model = _model()
        assert tc.install(model)
        x = torch.randn(2, 5, 32)
        first = model.txt_in(x)
        assert torch.equal(model.txt_in(x), first)
        tc.end_render(_Pipe(model))
        x.zero_()
        want = type(model.txt_in).forward(model.txt_in, x)
        assert torch.equal(model.txt_in(x), want)


def test_generate_ends_the_text_cache_render():
    from pathlib import Path

    src = (Path(tc.__file__).parent / "diffusion.py").read_text(encoding = "utf-8")
    finally_block = src[src.index('end_cfg_batch_render(getattr(state, "pipe", None))') :][:400]
    assert 'end_krea2_text_render(getattr(state, "pipe", None))' in finally_block


def test_cfg_parallel_keeps_krea2_with_the_cache_installed():
    """A closure over the module was declined by CFG-parallel's replica check; a bound method is accepted."""
    from core.inference import diffusion_multigpu as mg

    model = _model()
    assert tc.install(model)
    assert mg._hooks_present(model) is None

    class Pipe:
        transformer = model

    sel = mg.MultiGpuSelection(primary = 0, secondaries = (1,))
    why = mg.cfg_parallel_decline(Pipe(), offload_policy = "none", selection = sel)
    assert why is None or ("hooked" not in why and "override" not in why), why


def test_replica_rebinds_and_starts_empty():
    import copy

    with torch.no_grad():
        model = _model()
        assert tc.install(model)
        b = _branch(1, 12)
        want = _render(model, [b], steps = 2)
        replica = copy.deepcopy(model)
        assert replica.text_fusion.forward.__self__ is replica.text_fusion
        assert replica.text_fusion not in tc._MEMOS  # noqa: SLF001
        hits = tc.STATS["hits"]
        got = _render(replica, [b], steps = 2)
    for a, c in zip(want, got):
        assert torch.equal(a, c)
    assert tc.STATS["hits"] > hits
