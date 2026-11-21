# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Krea 2 inference runs on its valid text rows: the same output as the 512 padded rows, fewer tokens per step.

A tiny random Krea2Transformer2DModel on CPU, fed the pipeline's layout (``[prompt | PAD | suffix]`` masked in the
middle, text rows at the rotary origin): the trimmed call matches the padded one, keeps a bucketed length, reads the
mask once per prompt and leaves every other call alone."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "Krea2Transformer2DModel"):
    pytest.skip("diffusers without Krea 2", allow_module_level = True)

from core.inference import diffusion_krea2_text_trim as trim  # noqa: E402

TEXT, GRID, LAYERS, DIM = 40, 4, 3, 32


@pytest.fixture
def dit(monkeypatch):
    monkeypatch.delenv(trim.TEXT_TRIM_ENV, raising = False)
    monkeypatch.setattr(trim, "BUCKET", 8)
    torch.manual_seed(0)
    model = diffusers.Krea2Transformer2DModel(
        in_channels = 8,
        num_layers = 2,
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
    yield model
    trim.uninstall(model)


class _Pipe:
    def __init__(self, transformer):
        self.transformer = transformer


def _inputs(
    valid_rows,
    batch = None,
    suffix = 3,
):
    """The pipeline's layout per row: ``valid`` prompt tokens, padding, then ``suffix`` template tokens."""
    batch = batch or len(valid_rows)
    g = torch.Generator().manual_seed(1)
    mask = torch.zeros(batch, TEXT, dtype = torch.bool)
    for b, n in enumerate(valid_rows):
        mask[b, :n] = True
        mask[b, TEXT - suffix :] = True
    states = torch.randn(batch, TEXT, LAYERS, DIM, generator = g)
    hidden = torch.randn(batch, GRID * GRID, 8, generator = g)
    pos = diffusers.Krea2Pipeline.prepare_position_ids(TEXT, GRID, GRID, torch.device("cpu"))
    return {
        "hidden_states": hidden,
        "encoder_hidden_states": states,
        "timestep": torch.full((batch,), 0.5),
        "position_ids": pos,
        "encoder_attention_mask": mask,
        "return_dict": False,
    }


def _seen(model):
    got = {}

    def spy(module, args, kwargs):
        got.update(kwargs)

    # registered after the trim hook: sees what the forward receives
    handle = model.register_forward_pre_hook(spy, with_kwargs = True)
    return got, handle


@pytest.mark.parametrize("valid", [(5,), (12,), (5, 20), (1, 26)])
def test_the_trimmed_call_matches_the_padded_one(dit, valid):
    kw = _inputs(valid)
    with torch.no_grad():
        ref = dit(**kw)[0]
        assert trim.install_for_pipe(_Pipe(dit))
        before = trim.call_count()
        out = dit(**kw)[0]
    assert trim.call_count() == before + 1
    torch.testing.assert_close(out, ref, rtol = 1e-5, atol = 1e-5)


def test_the_kept_rows_are_the_valid_ones_padded_to_a_bucket(dit):
    kw = _inputs(
        (5, 9)
    )  # 12 valid columns across the batch (5 + 3 suffix, 9 + 3 suffix share the suffix)
    trim.install_for_pipe(_Pipe(dit))
    got, handle = _seen(dit)
    try:
        with torch.no_grad():
            dit(**kw)
    finally:
        handle.remove()
    assert got["encoder_hidden_states"].shape[1] == 16  # 12 valid -> the next multiple of 8
    assert got["position_ids"].shape[0] == 16 + GRID * GRID
    mask = got["encoder_attention_mask"]
    assert mask is not None and int(mask[1].sum()) == 12 and int(mask[0].sum()) == 8
    # the template order kept: prompt rows, the masked columns that fill the bucket, then the suffix
    torch.testing.assert_close(
        got["encoder_hidden_states"][1, :9], kw["encoder_hidden_states"][1, :9]
    )
    torch.testing.assert_close(
        got["encoder_hidden_states"][1, 13:], kw["encoder_hidden_states"][1, TEXT - 3 :]
    )
    assert not bool(mask[1, 9:13].any()) and bool(mask[1, 13:].all())


def test_a_prompt_filling_its_bucket_runs_unmasked(dit):
    kw = _inputs((13,))  # 13 + 3 suffix = 16 valid: a whole bucket, nothing masked
    trim.install_for_pipe(_Pipe(dit))
    got, handle = _seen(dit)
    try:
        with torch.no_grad():
            dit(**kw)
    finally:
        handle.remove()
    assert got["encoder_attention_mask"] is None and got["encoder_hidden_states"].shape[1] == 16


def test_the_mask_is_read_once_per_prompt_and_steps_reuse_the_tensors(dit, monkeypatch):
    kw = _inputs((5,))
    trim.install_for_pipe(_Pipe(dit))
    reads = []
    real = trim._columns
    monkeypatch.setattr(trim, "_columns", lambda *a: reads.append(1) or real(*a))
    got, handle = _seen(dit)
    seen = []
    try:
        with torch.no_grad():
            for _ in range(4):
                dit(**kw)
                seen.append((got["encoder_hidden_states"], got["position_ids"]))
    finally:
        handle.remove()
    assert len(reads) == 1
    assert all(a is seen[0][0] and b is seen[0][1] for a, b in seen)


def test_a_cfg_pair_of_one_bucket_shares_its_position_ids(dit):
    cond, neg = _inputs((5,)), _inputs((1,))
    neg["position_ids"] = cond["position_ids"]
    trim.install_for_pipe(_Pipe(dit))
    got, handle = _seen(dit)
    try:
        with torch.no_grad():
            dit(**cond)
            first = got["position_ids"]
            dit(**neg)
    finally:
        handle.remove()
    assert got["position_ids"] is first


def test_text_rows_off_the_rotary_origin_keep_the_padded_call(dit):
    kw = _inputs((5,))
    kw["position_ids"] = kw["position_ids"].clone()
    kw["position_ids"][:TEXT, 0] = torch.arange(TEXT, dtype = kw["position_ids"].dtype)
    trim.install_for_pipe(_Pipe(dit))
    before = trim.call_count()
    with torch.no_grad():
        dit(**kw)
    assert trim.call_count() == before


def test_no_mask_or_a_full_prompt_is_left_alone(dit):
    trim.install_for_pipe(_Pipe(dit))
    before = trim.call_count()
    with torch.no_grad():
        kw = _inputs((5,))
        kw["encoder_attention_mask"] = None
        dit(**kw)
        dit(**_inputs((TEXT,)))
    assert trim.call_count() == before


def test_the_kill_switch_and_an_explicit_off_load_keep_the_padded_rows(dit, monkeypatch):
    kw = _inputs((5,))
    assert trim.install_for_pipe(_Pipe(dit))
    monkeypatch.setenv(trim.TEXT_TRIM_ENV, "0")
    before = trim.call_count()
    with torch.no_grad():
        dit(**kw)
    assert trim.call_count() == before
    monkeypatch.delenv(trim.TEXT_TRIM_ENV)
    assert not trim.install_for_pipe(_Pipe(dit), explicit_off = True)
    assert not trim.installed(dit)
    with torch.no_grad():
        dit(**kw)
    assert trim.call_count() == before


def test_other_denoisers_are_not_hooked():
    assert not trim.install_for_pipe(_Pipe(torch.nn.Linear(2, 2)))


def test_the_load_stage_registers_it_for_krea_only():
    from core.inference.diffusion_autoopt import patches as ap
    spec = next(s for s in ap.specs(ap.STAGE_LOAD) if s.name == "krea2_text_trim")
    assert spec.classes == ("Krea2Transformer2DModel",) and spec.backends == ("image",)


def test_bucket_lengths():
    assert [trim.bucket_length(n, 512) for n in (1, 127, 128, 129, 400, 500)] == [
        128,
        128,
        128,
        256,
        512,
        512,
    ]
    assert trim.bucket_length(30, 40) == 40


@pytest.mark.parametrize("cond_valid, neg_valid", [(30, 1), (1, 30)])
def test_a_cfg_pair_of_two_buckets_keeps_one_length(dit, cond_valid, neg_valid):
    # The pipeline passes one position_ids to both branches: they keep the longer branch's rows, so the CFG router
    # can stack them (one doubled call), and each stays exact.
    cond, neg = _inputs((cond_valid,)), _inputs((neg_valid,))
    neg["position_ids"] = cond["position_ids"]
    trim.install_for_pipe(_Pipe(dit))
    got, handle = _seen(dit)
    shapes = []
    outs = []
    try:
        with torch.no_grad():
            for _ in range(3):
                for kw in (cond, neg):
                    outs.append(dit(**kw)[0])
                    shapes.append(tuple(got["encoder_hidden_states"].shape[:2]))
    finally:
        handle.remove()
    assert shapes[-1] == shapes[-2]  # steady state: one length for the pair
    trim.uninstall(dit)
    with torch.no_grad():
        ref = (dit(**cond)[0], dit(**neg)[0])
    torch.testing.assert_close(outs[-2], ref[0], rtol = 1e-5, atol = 1e-5)
    torch.testing.assert_close(outs[-1], ref[1], rtol = 1e-5, atol = 1e-5)
