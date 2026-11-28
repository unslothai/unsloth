# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Krea 2 block attention over the valid key prefix with no mask matches the padded masked call."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "Krea2Transformer2DModel"):
    pytest.skip("diffusers without Krea 2", allow_module_level = True)

from diffusers.models.transformers import transformer_krea2  # noqa: E402

from core.inference import diffusion_krea2_prefix_kv as pkv  # noqa: E402
from core.inference import diffusion_krea2_text_trim as trim  # noqa: E402
from core.inference import diffusion_cuda_graph as cg  # noqa: E402

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
    """A CFG-stacked B=2 call reads starts off the pads-first mask; a non-leading mask stays masked."""
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
    slot = pkv._Slot(starts, None, None, True)  # a transformer whose pre-hook plans its calls
    pkv._register(slot)
    before = dict(pkv.STATS)
    starts[0], starts[1] = 1, 3  # the hook planned one row; the batch has two
    stacked = pkv.attention(q, k, v, m4, starts)
    again = pkv.attention(q, k, v, m4, starts)  # same mask object: no second host read
    assert (
        pkv.STATS["prefix"] == before["prefix"] + 2
        and pkv.STATS["from_mask"] == before["from_mask"] + 2
    )
    assert pkv._MASK_MEMO[id(m4)][1] == [3, 5]
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
    """The graph-key hook raises for a non-zero start and keys a zero start or no plan as before."""
    _install(dit)
    keys = []

    def key(*_):
        try:
            keys.append(dit._unsloth_graph_key_extra())
        except pkv.PlanNotRecordable as exc:
            # the graph layers name this decline in the status (diffusion_cuda_graph.branch_decline_reason)
            assert cg.branch_decline_reason(exc).startswith(
                "declined by Krea 2 prefix K/V attention"
            )
            keys.append("unrecordable")

    # read where the graph layers read it: inside the forward
    handle = dit.transformer_blocks[0].register_forward_pre_hook(key)
    try:
        with torch.no_grad():
            dit(**_inputs((6,)))
            dit(**_inputs((13,)))  # 13 + 3 = 16 valid rows: a full bucket, no pads, no mask
    finally:
        handle.remove()
    assert keys == ["unrecordable", (1, 0)]
    assert dit._unsloth_graph_key_extra() == (0,)  # cleared after the call


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
    seen = []
    handle = dit.transformer_blocks[0].register_forward_pre_hook(
        lambda *_: seen.append(pkv.plan_key(dit))
    )
    try:
        with torch.no_grad():
            dit(**_inputs((6,)))
            monkeypatch.setenv(trim.TEXT_TRIM_ENV, "0")
            dit(**_inputs((6,)))
    finally:
        handle.remove()
    assert seen == [(1, 7), (0,)]


def test_a_call_that_skips_the_hooks_never_reads_the_last_plan(dit):
    """A direct ``forward`` after a planned call runs masked, never the stale key start."""
    _install(dit)
    planned, other = _inputs((6,)), _inputs((2,))
    with torch.no_grad():
        ref = dit.forward(**other)[0]  # no hooks: the padded, masked call
        dit(**planned)
        assert pkv.plan_key(dit) == (0,)
        before = dict(pkv.STATS)
        out = dit.forward(**other)[0]
    assert pkv.STATS["prefix"] == before["prefix"]
    torch.testing.assert_close(out, ref, rtol = 1e-5, atol = 1e-5)


def test_a_cfg_replica_reads_its_own_mask_never_the_copied_plan(dit):
    """A deepcopy CFG replica must not keep the copied key start: it plans off its own mask."""
    from core.inference import diffusion_multigpu as mg

    stock = _model()
    _install(dit)
    first, second = _inputs((6,)), _inputs((10,))  # one 16-row bucket, key starts 7 and 3
    with torch.no_grad():
        _, trimmed = trim._pre_hook(dit, (), dict(first))
        assert pkv.plan_key(dit) == (1, 7)
        replica = mg._copy_module(dit, torch.device("cpu"))
        _, changed = trim._pre_hook(dit, (), dict(second))  # the source plans the next branch
        assert trimmed["encoder_hidden_states"].shape == changed["encoder_hidden_states"].shape
        source_plan = pkv.plan_key(dit)
        assert source_plan != (1, 7)
        mg._mirror_hook_state(list(zip(dit.modules(), replica.modules())))
        oracle = stock(**second)[0]
        before = dict(pkv.STATS)
        got = replica(**{k: v.clone() if torch.is_tensor(v) else v for k, v in changed.items()})[0]
    torch.testing.assert_close(got, oracle, rtol = 1e-5, atol = 1e-5)
    assert pkv.STATS["prefix"] == before["prefix"] + BLOCKS
    assert pkv.STATS["from_mask"] == before["from_mask"] + BLOCKS
    assert pkv.plan_key(dit) == source_plan  # the source's own plan is untouched
    rstarts = replica.transformer_blocks[0].attn.processor._unsloth_starts
    slot = pkv._slot_of(rstarts)
    assert slot is not None and not slot.planned and slot.owner() is replica
    assert slot.proc() is replica.transformer_blocks[0].attn.processor
    assert all(b.attn.processor._unsloth_starts is rstarts for b in replica.transformer_blocks)
    assert pkv._slot_of(dit.transformer_blocks[0].attn.processor._unsloth_starts).planned


def test_a_call_that_needs_gradients_runs_the_stock_masked_attention(dit):
    """Grad calls run the stock masked attention (no autograd formula); gradients match."""
    stock = _model()
    _install(dit)
    kw = _inputs((6,))
    # the same trimmed, pads-first call for both models
    _, trimmed = trim._pre_hook(dit, (), dict(kw))
    trimmed = {k: v.clone() if torch.is_tensor(v) else v for k, v in trimmed.items()}
    stock(**trimmed)[0].square().mean().backward()
    before = dict(pkv.STATS)
    dit(**trimmed)[0].square().mean().backward()
    assert pkv.STATS["prefix"] == before["prefix"]  # no op call
    for (name, a), (_, b) in zip(stock.named_parameters(), dit.named_parameters()):
        if a.grad is None:
            assert b.grad is None, name
            continue
        torch.testing.assert_close(b.grad, a.grad, rtol = 1e-5, atol = 1e-6, msg = name)
    assert dit.transformer_blocks[0].attn.to_q.weight.grad.abs().sum() > 0


def test_the_int8_route_takes_the_block_attention_and_its_unmasked_calls(dit):
    """Prefix processors are INT8-routable and a planned block call reaches the route without a mask."""
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


def _cuda_card(monkeypatch, capability, kernels):
    from core.inference import diffusion_attention as A

    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device = None: capability)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    asked = []

    def probe(target):
        asked.append((target.device, target.dtype))
        return kernels

    monkeypatch.setattr(A, "available_sdpa_kernels", probe)
    monkeypatch.setattr(pkv, "_FLASH_OK", {})
    monkeypatch.delenv(A.NATIVE_FLASH_ATTN_ENV, raising = False)
    monkeypatch.delenv(pkv.BACKEND_ENV, raising = False)
    return asked


def test_the_mask_free_call_takes_flash_only_in_place_of_auto_s_masked_pin(monkeypatch):
    from diffusers.models.attention_dispatch import AttentionBackendName as N

    from core.inference import diffusion_int8_attention_dispatch as D

    cuda, bf16 = torch.device("cuda", 0), torch.bfloat16
    asked = _cuda_card(monkeypatch, (8, 9), ("flash", "mem_efficient", "cudnn", "math"))
    # L4, auto's cuDNN pick, flash in this build -> flash (probed once per card and dtype)
    assert pkv._prefix_backend(N._NATIVE_CUDNN, cuda, bf16, auto_pick = True) == N._NATIVE_FLASH
    assert pkv._prefix_backend(N._NATIVE_CUDNN, cuda, bf16, auto_pick = True) == N._NATIVE_FLASH
    assert asked == [("cuda:0", bf16)]
    # an explicit cudnn request stays cuDNN; native stays native (torch picks flash itself for a mask-free call)
    assert pkv._prefix_backend(N._NATIVE_CUDNN, cuda, bf16, auto_pick = False) == N._NATIVE_CUDNN
    assert pkv._prefix_backend(None, cuda, bf16, auto_pick = True) is None
    assert pkv._prefix_backend(N.NATIVE, cuda, bf16, auto_pick = True) == N.NATIVE
    assert pkv._prefix_backend(N.SAGE, cuda, bf16, auto_pick = True) == N.SAGE
    route = D.Int8Route(N._NATIVE_CUDNN, 1024, torch.bfloat16, 0, 128)
    assert (
        pkv._prefix_backend(route, cuda, bf16, auto_pick = True) is route
    )  # the INT8 route takes it itself
    assert (
        pkv._prefix_backend(N._NATIVE_CUDNN, cuda, torch.float32, auto_pick = True) == N._NATIVE_CUDNN
    )
    assert (
        pkv._prefix_backend(N._NATIVE_CUDNN, torch.device("cpu"), bf16, auto_pick = True)
        == N._NATIVE_CUDNN
    )
    # the native-flash kill switch keeps the pin
    from core.inference import diffusion_attention as A

    monkeypatch.setenv(A.NATIVE_FLASH_ATTN_ENV, "0")
    assert pkv._prefix_backend(N._NATIVE_CUDNN, cuda, bf16, auto_pick = True) == N._NATIVE_CUDNN
    monkeypatch.delenv(A.NATIVE_FLASH_ATTN_ENV)
    monkeypatch.setenv(pkv.BACKEND_ENV, "incumbent")
    assert pkv._prefix_backend(N._NATIVE_CUDNN, cuda, bf16, auto_pick = True) == N._NATIVE_CUDNN
    monkeypatch.setenv(pkv.BACKEND_ENV, "flash")
    assert pkv._prefix_backend(N._NATIVE_CUDNN, cuda, bf16, auto_pick = False) == N._NATIVE_FLASH
    # an unmeasured card keeps the pin
    _cuda_card(monkeypatch, (9, 0), ("flash", "cudnn"))
    assert pkv._prefix_backend(N._NATIVE_CUDNN, cuda, bf16, auto_pick = True) == N._NATIVE_CUDNN


@pytest.mark.parametrize("hip, version", [("6.4", "2.9.0"), (None, "2.9.0+rocm7.1")])
def test_a_rocm_build_keeps_the_pin(monkeypatch, hip, version):
    """ROCm never takes the CUDA flash pick, including AMD SDK / Radeon wheels that leave version.hip unset."""
    from diffusers.models.attention_dispatch import AttentionBackendName as N

    cuda, bf16 = torch.device("cuda", 0), torch.bfloat16
    _cuda_card(monkeypatch, (8, 9), ("flash", "mem_efficient", "cudnn", "math"))
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch, "__version__", version)
    assert pkv._prefix_backend(N._NATIVE_CUDNN, cuda, bf16, auto_pick = True) == N._NATIVE_CUDNN


def test_a_build_without_flash_sdpa_keeps_the_pin(monkeypatch):
    """PyTorch Windows (MSVC) builds 2.10-2.14 ship no CUDA flash SDPA: exclusive flash would have no kernel."""
    from diffusers.models.attention_dispatch import AttentionBackendName as N

    cuda, bf16 = torch.device("cuda", 0), torch.bfloat16
    _cuda_card(monkeypatch, (8, 9), ("mem_efficient", "cudnn", "math"))
    assert pkv._prefix_backend(N._NATIVE_CUDNN, cuda, bf16, auto_pick = True) == N._NATIVE_CUDNN
    _cuda_card(
        monkeypatch, (8, 0), ()
    )  # an unanswered probe never forces flash (and is asked again)
    assert pkv._prefix_backend(N._NATIVE_CUDNN, cuda, bf16, auto_pick = True) == N._NATIVE_CUDNN
    assert pkv._FLASH_OK == {}


def test_apply_tags_auto_s_cudnn_pick_apart_from_an_explicit_request(dit):
    from core.inference import diffusion_attention as A

    pipe = _Pipe(dit)
    assert A._apply_attention_backend(pipe, A.AUTO_CUDNN) == "_native_cudnn"
    assert getattr(dit, A.ATTENTION_AUTO_PICK_ATTR) is True
    assert A._apply_attention_backend(pipe, "_native_cudnn") == "_native_cudnn"
    assert getattr(dit, A.ATTENTION_AUTO_PICK_ATTR) is False
    A._apply_attention_backend(pipe, None)
    assert getattr(dit, A.ATTENTION_AUTO_PICK_ATTR) is False


def test_the_op_forwards_the_auto_pick_of_its_own_transformer(dit, monkeypatch):
    from core.inference import diffusion_attention as A

    _install(dit)
    seen = []
    monkeypatch.setattr(pkv, "_prefix_backend", lambda b, d, t, auto: seen.append(auto) or b)
    with torch.no_grad():
        dit(**_inputs((6,)))
        setattr(dit, A.ATTENTION_AUTO_PICK_ATTR, True)
        dit(**_inputs((6,)))
    assert seen == [False] * BLOCKS + [True] * BLOCKS
