# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Krea 2 block attention reads the valid key prefix with no mask: the trim puts each row's pads first and the
blocks attend over keys from the first valid row on. Same output as the padded masked call (fp32), no masked block
call, one compiled graph across prompts."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "Krea2Transformer2DModel"):
    pytest.skip("diffusers without Krea 2", allow_module_level = True)

from diffusers.models.transformers import transformer_krea2  # noqa: E402

from core.inference import diffusion_krea2_prefix_kv as pkv  # noqa: E402
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


@pytest.fixture
def dit(monkeypatch):
    monkeypatch.delenv(trim.TEXT_TRIM_ENV, raising = False)
    monkeypatch.delenv(pkv.PREFIX_ENV, raising = False)
    monkeypatch.delenv(pkv.BACKEND_ENV, raising = False)
    monkeypatch.setattr(trim, "BUCKET", 8)
    model = _model()
    yield model
    trim.uninstall(model)


class _Pipe:
    def __init__(self, transformer):
        self.transformer = transformer


def _inputs(
    valid_rows,
    suffix = 3,
    pos = None,
):
    """The pipeline's layout per row: ``valid`` prompt tokens, padding, then ``suffix`` template tokens."""
    g = torch.Generator().manual_seed(1 + sum(valid_rows))
    batch = len(valid_rows)
    mask = torch.zeros(batch, TEXT, dtype = torch.bool)
    for b, n in enumerate(valid_rows):
        mask[b, :n] = True
        mask[b, TEXT - suffix :] = True
    if pos is None:
        pos = diffusers.Krea2Pipeline.prepare_position_ids(TEXT, GRID, GRID, torch.device("cpu"))
    return {
        "hidden_states": torch.randn(batch, GRID * GRID, 8, generator = g),
        "encoder_hidden_states": torch.randn(batch, TEXT, LAYERS, DIM, generator = g),
        "timestep": torch.full((batch,), 0.5),
        "position_ids": pos,
        "encoder_attention_mask": mask,
        "return_dict": False,
    }


def _install(model):
    assert pkv.install(model) == BLOCKS
    assert trim.install_for_pipe(_Pipe(model))


def _spy_dispatch(monkeypatch):
    """Every main-block dispatch call (queries longer than the image grid): (mask given, key rows)."""
    calls = []
    real = transformer_krea2.dispatch_attention_fn

    def spy(
        query,
        key,
        value,
        attn_mask = None,
        *args,
        **kwargs,
    ):
        if query.shape[1] > GRID * GRID:
            calls.append((attn_mask is not None, int(key.shape[1])))
        return real(query, key, value, attn_mask, *args, **kwargs)

    monkeypatch.setattr(transformer_krea2, "dispatch_attention_fn", spy)
    return calls


@pytest.mark.parametrize("valid", [(5,), (12,), (5, 20), (1, 26)])
def test_prefix_call_matches_the_padded_masked_call(dit, valid):
    kw = _inputs(valid)
    with torch.no_grad():
        ref = dit(**kw)[0]
        _install(dit)
        before = dict(pkv.STATS)
        out = dit(**kw)[0]
    assert pkv.STATS["prefix"] == before["prefix"] + BLOCKS
    assert pkv.STATS["masked"] == before["masked"]
    torch.testing.assert_close(out, ref, rtol = 1e-5, atol = 1e-5)


def test_no_block_attention_carries_a_mask_and_keys_start_at_the_first_valid_row(dit, monkeypatch):
    kw = _inputs((6,))  # 6 + 3 suffix = 9 valid of 40 -> a 16-row bucket, 7 of them pads
    _install(dit)
    calls = _spy_dispatch(monkeypatch)
    with torch.no_grad():
        dit(**kw)
    assert len(calls) == BLOCKS
    assert all(not masked and keys == 9 + GRID * GRID for masked, keys in calls), calls


def test_the_main_tree_masks_every_block_call(dit, monkeypatch):
    """The same call without the prefix processors: every block attention carries the key-padding mask."""
    kw = _inputs((6,))
    assert trim.install_for_pipe(_Pipe(dit))
    calls = _spy_dispatch(monkeypatch)
    with torch.no_grad():
        dit(**kw)
    assert len(calls) == BLOCKS and all(
        masked and keys == 16 + GRID * GRID for masked, keys in calls
    ), calls


def test_a_cfg_pair_of_unequal_lengths_shares_the_rows_and_each_branch_matches(dit):
    pos = diffusers.Krea2Pipeline.prepare_position_ids(TEXT, GRID, GRID, torch.device("cpu"))
    cond, uncond = _inputs((20,), pos = pos), _inputs((2,), pos = pos)
    with torch.no_grad():
        ref_c, ref_u = dit(**cond)[0], dit(**uncond)[0]
        _install(dit)
        before = pkv.STATS["prefix"]
        seen = []
        handle = dit.register_forward_pre_hook(
            lambda m, a, k: seen.append((k["encoder_hidden_states"].shape[1], pkv.plan_key(m))),
            with_kwargs = True,
        )
        try:
            for _ in range(2):  # two steps: the second reuses the memoised tensors
                out_c, out_u = dit(**cond)[0], dit(**uncond)[0]
        finally:
            handle.remove()
    torch.testing.assert_close(out_c, ref_c, rtol = 1e-5, atol = 1e-5)
    torch.testing.assert_close(out_u, ref_u, rtol = 1e-5, atol = 1e-5)
    assert pkv.STATS["prefix"] == before + 4 * BLOCKS
    # one row count for both branches (so a CFG router can stack them), each with its own key start
    assert {rows for rows, _ in seen} == {24}
    assert seen[-2][1] == (1, 24 - 23) and seen[-1][1] == (1, 24 - 5)


def test_a_batch_the_hook_did_not_plan_reads_its_plan_off_the_mask():
    """A CFG router stacking two planned B=1 calls into one B=2 forward: the plan names one row, so the op reads each
    row's key start off the pads-first mask (once per mask) and still runs mask-free; a mask whose masked keys are
    not a leading run keeps the masked call. Every path matches the masked reference."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 12, 2, 8) for _ in range(3))

    def ref(m4):
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask = m4
        ).transpose(1, 2)

    mask = torch.ones(2, 12, dtype = torch.bool)
    mask[0, :3] = False
    mask[1, :5] = False
    m4 = mask[:, None, None, :]
    starts = pkv.new_starts()
    before = dict(pkv.STATS)
    starts[0], starts[1] = 1, 3  # the hook planned one row; the batch has two
    stacked = pkv.attention(q, k, v, m4, starts)
    again = pkv.attention(q, k, v, m4, starts)  # same mask object: no second host read
    assert (
        pkv.STATS["prefix"] == before["prefix"] + 2
        and pkv.STATS["from_mask"] == before["from_mask"] + 2
    )
    assert pkv._MASK_MEMO["last"][1] == [3, 5]
    starts[0], starts[1], starts[2] = 2, 3, 5
    planned = pkv.attention(q, k, v, m4, starts)
    assert pkv.STATS["prefix"] == before["prefix"] + 3
    for got in (stacked, again, planned):
        torch.testing.assert_close(got, ref(m4), rtol = 1e-5, atol = 1e-5)
    mid = torch.ones(2, 12, dtype = torch.bool)
    mid[:, 4:7] = False  # the main order: pads between valid keys
    mid4 = mid[:, None, None, :]
    starts[0] = 1
    masked = pkv.attention(q, k, v, mid4, starts)
    assert pkv.STATS["masked"] == before["masked"] + 1
    torch.testing.assert_close(masked, ref(mid4), rtol = 1e-5, atol = 1e-5)
    starts[0] = 0  # no plan from the hook at all (a padded call): masked, no host read
    assert (
        pkv.attention(q, k, v, m4, starts) is not None
        and pkv.STATS["masked"] == before["masked"] + 2
    )


def test_a_planned_call_is_never_recorded_by_a_cuda_graph(dit):
    """The key start is fixed at capture time and follows the prompt: the graph-key hook raises for a non-zero start
    (the graph layers then run the call eager) and keys a zero start or no plan as before."""
    _install(dit)
    with torch.no_grad():
        dit(**_inputs((6,)))
        with pytest.raises(pkv.PlanNotRecordable):
            dit._unsloth_graph_key_extra()
        dit(**_inputs((13,)))  # 13 + 3 = 16 valid rows: a full bucket, no pads, no mask
        assert dit._unsloth_graph_key_extra() == (1, 0)
        pkv.set_plan(dit, None)
        assert dit._unsloth_graph_key_extra() == (0,)


def test_one_compiled_graph_serves_every_key_start(dit):
    _install(dit)
    torch._dynamo.reset()
    for blk in dit.transformer_blocks:
        blk.compile(backend = "aot_eager", fullgraph = True, dynamic = False)
    with torch.no_grad():
        dit(**_inputs((6,)))
        with torch._dynamo.config.patch(error_on_recompile = True):
            for valid in ((7,), (10,), (12,)):  # one 16-row bucket, key starts 6, 3, 1
                dit(**_inputs(valid))
    torch._dynamo.reset()


def test_the_switch_off_keeps_the_stock_processors(dit, monkeypatch):
    monkeypatch.setenv(pkv.PREFIX_ENV, "0")
    assert pkv.install(dit) == 0
    assert all(
        type(b.attn.processor).__name__ == "Krea2AttnProcessor" for b in dit.transformer_blocks
    )
    assert not pkv.installed(dit)


def test_a_padded_call_after_a_planned_one_clears_the_plan(dit, monkeypatch):
    _install(dit)
    with torch.no_grad():
        dit(**_inputs((6,)))
        assert pkv.plan_key(dit) == (1, 7)
        monkeypatch.setenv(trim.TEXT_TRIM_ENV, "0")
        dit(**_inputs((6,)))
    assert pkv.plan_key(dit) == (0,)


def test_the_int8_route_takes_the_block_attention_and_its_unmasked_calls(dit):
    """The prefix processors are routable by the INT8 dispatch (main: Krea 2's attention names no ``heads`` and every
    block call carries a mask, so nothing routes), and a planned block call reaches the route without a mask."""
    from core.inference import diffusion_int8_attention_dispatch as D

    _install(dit)
    routable = D.routable_modules(dit)
    assert [id(m) for m in routable] == [id(b.attn) for b in dit.transformer_blocks]
    route = D.Int8Route(None, 1, torch.float32, 0, 16)
    seen = []

    def spy(
        query,
        key,
        value,
        attn_mask = None,
        *args,
        backend = None,
        **kwargs,
    ):
        if query.shape[1] > GRID * GRID:
            seen.append(
                D._reason(backend, (query, key, value, attn_mask), {})
                if isinstance(backend, D.Int8Route)
                else "?"
            )
        return torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask = None if attn_mask is None else attn_mask,
        ).transpose(1, 2)

    for m in routable:
        m.processor._attention_backend = route
    real = transformer_krea2.dispatch_attention_fn
    transformer_krea2.dispatch_attention_fn = spy
    try:
        with torch.no_grad():
            dit(**_inputs((6,)))
    finally:
        transformer_krea2.dispatch_attention_fn = real
    # CPU tensors: the route's only objection is the device, never the mask
    assert seen == ["device"] * BLOCKS


def test_the_mask_free_call_takes_flash_only_in_place_of_the_masked_pin(monkeypatch):
    from diffusers.models.attention_dispatch import AttentionBackendName as N

    from core.inference import diffusion_int8_attention_dispatch as D

    cuda = torch.device("cuda", 0)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device = None: (8, 9))
    assert (
        pkv._prefix_backend(N._NATIVE_CUDNN, cuda) == N._NATIVE_FLASH
    )  # L4: the masked-DiT pin -> flash
    assert pkv._prefix_backend(None, cuda) == N._NATIVE_FLASH
    assert pkv._prefix_backend(N.SAGE, cuda) == N.SAGE  # an explicit pick stays
    route = D.Int8Route(N._NATIVE_CUDNN, 1024, torch.bfloat16, 0, 128)
    assert (
        pkv._prefix_backend(route, cuda) is route
    )  # the INT8 route takes the unmasked call itself
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device = None: (9, 0))
    assert (
        pkv._prefix_backend(N._NATIVE_CUDNN, cuda) == N._NATIVE_CUDNN
    )  # unmeasured card: the pin stays
    assert pkv._prefix_backend(N._NATIVE_CUDNN, torch.device("cpu")) == N._NATIVE_CUDNN
    monkeypatch.setenv(pkv.BACKEND_ENV, "incumbent")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device = None: (8, 9))
    assert pkv._prefix_backend(N._NATIVE_CUDNN, cuda) == N._NATIVE_CUDNN
