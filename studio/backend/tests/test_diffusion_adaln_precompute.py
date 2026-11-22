# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AdaLN precompute on tiny FLUX.1 / Qwen-Image transformers: outputs stay bit-identical."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from core.inference import diffusion_adaln_precompute as ap  # noqa: E402

DTYPES = [torch.float32, torch.bfloat16]


def _flux(guidance: bool = True):
    torch.manual_seed(0)
    m = diffusers.FluxTransformer2DModel(
        patch_size = 1,
        in_channels = 16,
        num_layers = 2,
        num_single_layers = 2,
        attention_head_dim = 16,
        num_attention_heads = 2,
        joint_attention_dim = 32,
        pooled_projection_dim = 24,
        guidance_embeds = guidance,
        axes_dims_rope = (4, 6, 6),
    )
    return m.eval()


def _qwen(zero_cond_t: bool = False):
    torch.manual_seed(0)
    m = diffusers.QwenImageTransformer2DModel(
        patch_size = 2,
        in_channels = 16,
        out_channels = 4,
        num_layers = 2,
        attention_head_dim = 16,
        num_attention_heads = 2,
        joint_attention_dim = 32,
        axes_dims_rope = (4, 6, 6),
        zero_cond_t = zero_cond_t,
    )
    return m.eval()


class _Pipe:
    def __init__(self, timesteps):
        self.scheduler = types.SimpleNamespace(timesteps = timesteps)


_PIPES: list = []


def _keep(p):
    _PIPES.append(p)
    return p


def _pipe(steps: int):
    p = _Pipe(torch.linspace(1000.0, 30.0, steps))
    _PIPES.append(p)
    return p


def _flux_calls(
    dtype,
    batch = 1,
    guidance = 3.5,
    pooled_seed = 1,
    steps = 4,
):
    g = torch.Generator().manual_seed(pooled_seed)
    hs = torch.randn(batch, 16, 16, generator = g).to(dtype)
    ehs = torch.randn(batch, 8, 32, generator = g).to(dtype)
    pooled = torch.randn(batch, 24, generator = g).to(dtype)
    img_ids = torch.zeros(16, 3)
    img_ids[:, 1] = torch.arange(16) // 4
    img_ids[:, 2] = torch.arange(16) % 4
    txt_ids = torch.zeros(8, 3)
    sched = torch.linspace(1000.0, 30.0, steps)
    calls = []
    for t in sched:
        ts = t.expand(batch).to(dtype)
        kw = dict(
            hidden_states = hs,
            encoder_hidden_states = ehs,
            pooled_projections = pooled,
            timestep = ts / 1000,
            img_ids = img_ids,
            txt_ids = txt_ids,
            return_dict = False,
        )
        if guidance is not None:
            kw["guidance"] = torch.full([batch], guidance).to(dtype)
        calls.append(kw)
    return calls


def _qwen_calls(
    dtype,
    batch = 1,
    steps = 4,
    zero_cond_t = False,
):
    g = torch.Generator().manual_seed(2)
    n_img = 16 if not zero_cond_t else 32
    hs = torch.randn(batch, n_img, 16, generator = g).to(dtype)
    ehs = torch.randn(batch, 6, 32, generator = g).to(dtype)
    mask = torch.ones(batch, 6, dtype = torch.long)
    shapes = [[(1, 4, 4)] if not zero_cond_t else [(1, 4, 4), (1, 4, 4)]] * batch
    sched = torch.linspace(1000.0, 30.0, steps)
    calls = []
    for t in sched:
        ts = t.expand(batch).to(dtype)
        calls.append(
            dict(
                hidden_states = hs,
                encoder_hidden_states = ehs,
                encoder_hidden_states_mask = mask,
                timestep = ts / 1000,
                img_shapes = shapes,
                return_dict = False,
            )
        )
    return calls


def _run(model, calls):
    with torch.no_grad():
        return [model(**kw)[0].clone() for kw in calls]


def _counts():
    return dict(ap.COUNTS)


@pytest.fixture(autouse = True)
def _verify(monkeypatch):
    monkeypatch.setenv(ap.VERIFY_ENV, "1")
    yield


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch", [1, 2])
def test_flux_bit_identical_one_weight_pass(dtype, batch):
    model = _flux().to(dtype)
    calls = _flux_calls(dtype, batch = batch)
    ref = _run(model, calls)
    mod_params = sum(
        p.numel() for n, p in model.named_parameters() if "norm" in n and "linear" in n
    )
    total = sum(p.numel() for p in model.parameters())
    before = _counts()
    assert ap.install(model, _pipe(len(calls)), requested = ap.MODE_ON)
    after_total = sum(p.numel() for p in model.parameters())
    # every modulation Linear left the module tree (the parameters a placement / offloader would move)
    assert total - after_total == mod_params
    assert not any("norm1.linear" in n or "norm.linear" in n for n, _ in model.named_parameters())
    out = _run(model, calls)
    c = _counts()
    for a, b in zip(ref, out):
        assert torch.equal(a, b)
    assert (
        c["weight_passes"] - before["weight_passes"] == 1
    )  # every step predicted from the scheduler
    assert c["served"] - before["served"] == len(calls)
    assert c["verify_checks"] > before["verify_checks"]
    out2 = _run(model, calls)
    assert _counts()["weight_passes"] == c["weight_passes"]
    assert all(torch.equal(a, b) for a, b in zip(ref, out2))


@pytest.mark.parametrize("dtype", DTYPES)
def test_flux_cfg_branches_and_guidance_change(dtype):
    model = _flux().to(dtype)
    cond = _flux_calls(dtype, pooled_seed = 1)
    uncond = _flux_calls(dtype, pooled_seed = 7)
    other_g = _flux_calls(dtype, pooled_seed = 1, guidance = 5.0)
    interleaved = [kw for pair in zip(cond, uncond) for kw in pair] + other_g
    ref = _run(model, interleaved)
    assert ap.install(model, _pipe(4), requested = ap.MODE_ON)
    out = _run(model, interleaved)
    for a, b in zip(ref, out):
        assert torch.equal(a, b)
    assert ap.summary(model)["cached_rows"] == 12


def test_flux_schnell_no_guidance():
    model = _flux(guidance = False)
    calls = _flux_calls(torch.float32, guidance = None)
    ref = _run(model, calls)
    assert ap.install(model, _pipe(len(calls)), requested = ap.MODE_ON)
    assert all(torch.equal(a, b) for a, b in zip(ref, _run(model, calls)))


def test_unpredicted_timestep_computes_lazily():
    model = _flux()
    calls = _flux_calls(torch.float32)
    ref = _run(model, calls)
    # a scheduler grid that never matches: every step is its own lazy pass, still exact
    assert ap.install(model, _keep(_Pipe(torch.linspace(999.0, 31.0, 9))), requested = ap.MODE_ON)
    before = _counts()["weight_passes"]
    assert all(torch.equal(a, b) for a, b in zip(ref, _run(model, calls)))
    assert _counts()["weight_passes"] - before == len(calls)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("zero_cond_t", [False, True])
def test_qwen_image_bit_identical(dtype, zero_cond_t):
    model = _qwen(zero_cond_t = zero_cond_t).to(dtype)
    calls = _qwen_calls(dtype, zero_cond_t = zero_cond_t)
    ref = _run(model, calls)
    assert ap.install(model, _pipe(len(calls)), requested = ap.MODE_ON)
    assert not any("img_mod" in n or "txt_mod" in n for n, _ in model.named_parameters())
    before = _counts()["weight_passes"]
    out = _run(model, calls)
    assert all(torch.equal(a, b) for a, b in zip(ref, out))
    assert _counts()["weight_passes"] - before == 1


def test_uninstall_restores_stock_modules():
    model = _flux()
    calls = _flux_calls(torch.float32)
    ref = _run(model, calls)
    n_params = sum(p.numel() for p in model.parameters())
    assert ap.install(model, _pipe(4), requested = ap.MODE_ON)
    assert ap.uninstall(model, reason = "test")
    assert not ap.installed(model)
    assert sum(p.numel() for p in model.parameters()) == n_params
    assert isinstance(model.transformer_blocks[0].norm1.linear, torch.nn.Linear)
    assert all(torch.equal(a, b) for a, b in zip(ref, _run(model, calls)))


def test_modes_and_declines(monkeypatch):
    model = _flux()
    monkeypatch.setenv(ap.ENV, "0")
    assert ap.decline_reason(model, offloaded = True) is not None
    monkeypatch.setenv(ap.ENV, "auto")
    assert ap.decline_reason(model, offloaded = False) is not None
    assert ap.decline_reason(model, offloaded = True) is None
    monkeypatch.setenv(ap.ENV, "1")
    assert ap.decline_reason(model, offloaded = False) is None
    assert ap.adapter_for(torch.nn.Linear(2, 2)) is None


def test_compiled_blocks_bit_identical():
    model = _flux()
    calls = _flux_calls(torch.float32)
    ref = _run(model, calls)
    assert ap.install(model, _pipe(len(calls)), requested = ap.MODE_ON)
    import os

    os.environ.pop(ap.VERIFY_ENV, None)
    for blk in list(model.transformer_blocks) + list(model.single_transformer_blocks):
        blk.compile(backend = "aot_eager", fullgraph = True)
    out = _run(model, calls)
    assert all(torch.allclose(a, b, atol = 1e-6, rtol = 0) for a, b in zip(ref, out))


def test_layer_split_declines_and_rows_follow_block_device():
    model = _flux()
    assert ap.install(model, _pipe(4), requested = ap.MODE_ON, split = True) is False
    assert not ap.installed(model)
    # a stand-in whose block sits on another device returns its row there (CPU mock: "meta" as the second card)
    stand = ap.AdaLNTable(torch.nn.Linear(4, 6), "transformer_blocks.0.norm1.linear", "silu")
    stand.unsloth_adaln_cur = torch.randn(1, 6)
    out = stand(torch.empty(1, 4, device = "meta"))
    assert out.device.type == "meta" and tuple(out.shape) == (1, 6)
    same = torch.randn(1, 4)
    assert stand(same) is stand.unsloth_adaln_cur


def test_uninstall_restores_beside_owning_block():
    model = _flux()
    assert ap.install(model, _pipe(4), requested = ap.MODE_ON)
    model.to("meta")
    st = model.__dict__[ap.STATE_ATTR]
    # the placement is resolved from the owning block, not the parameter-less parent (norm / Sequential)
    for name, *_rest in st.sites:
        assert ap._owner_device(model, name).type == "meta", name
    assert ap.uninstall(model, reason = "LoRA adapters requested")
    # meta is never a target: real weights are kept rather than dropped (sequential offload's meta params)
    assert model.transformer_blocks[0].norm1.linear.weight.device.type == "cpu"


def test_qwen_uninstall_restores_beside_owning_block():
    model = _qwen()
    assert ap.install(model, _pipe(4), requested = ap.MODE_ON)
    model.to("meta")
    st = model.__dict__[ap.STATE_ATTR]
    for name, *_rest in st.sites:
        assert ap._owner_device(model, name).type == "meta", name
    assert ap.uninstall(model)
    assert model.transformer_blocks[0].img_mod[1].weight.device.type == "cpu"


def test_predicted_rows_over_budget_keep_the_served_row(monkeypatch):
    model = _flux()
    calls = _flux_calls(torch.float32, steps = 6)
    ref = _run(model, calls)
    assert ap.install(model, _pipe(len(calls)), requested = ap.MODE_ON)
    # room for about two rows: the predicted rows of the request overflow it on the first miss
    with torch.no_grad():
        model(**calls[0])
        st = model.__dict__[ap.STATE_ATTR]
        assert len(st.rows) == len(calls)  # every step predicted on the first miss
        row = st.row_bytes / len(st.rows)
        ap.uninstall(model)
        assert ap.install(model, _pipe(len(calls)), requested = ap.MODE_ON)
        monkeypatch.setattr(ap, "_table_budget_bytes", lambda: int(row * 2.5))
        out = [model(**kw)[0].clone() for kw in calls]
    assert all(torch.equal(a, b) for a, b in zip(ref, out))
    assert ap.summary(model)["cached_rows"] <= 2


def test_learned_peak_label_records_the_lever():
    from core.inference.diffusion_memory import learned_peak_label

    model = _flux()
    pipe = _keep(_Pipe(torch.linspace(1000.0, 30.0, 4)))
    pipe.transformer = model
    off = learned_peak_label(pipe, "native", "m")
    assert ap.install(model, pipe, requested = ap.MODE_ON)
    on = learned_peak_label(pipe, "native", "m")
    assert on != off and on.endswith("|adaln:on")
    ap.uninstall(model)
    assert learned_peak_label(pipe, "native", "m") == off


def _mark_streamed(model):
    sub = model.transformer_blocks[0]
    sub._diffusers_hook = types.SimpleNamespace(hooks = {"group_offloading": object()})
    return model


def _flux_cfg_loop(model, steps = 4):
    """FluxPipeline true-CFG loop: per step one latent object, cond then uncond call (different pooled)."""
    calls_c = _flux_calls(torch.float32, pooled_seed = 1, steps = steps)
    calls_u = _flux_calls(torch.float32, pooled_seed = 7, steps = steps)
    outs = []
    with torch.no_grad():
        for c, u in zip(calls_c, calls_u):
            latent = c["hidden_states"].clone()  # the pipeline's one latent object per step
            c = dict(c, hidden_states = latent)
            u = dict(u, hidden_states = latent, img_ids = c["img_ids"], txt_ids = c["txt_ids"])
            cond = model(**c)[0]
            uncond = model(**u)[0]
            outs.append((uncond + 3.0 * (cond - uncond)).clone())
    return outs


def test_cfg_batch_router_merged_call_reads_each_branch_rows(monkeypatch):
    from core.inference import diffusion_cfg_batch as cb

    monkeypatch.setenv(cb.CFG_BATCH_ENV, "force")
    base = _mark_streamed(_flux())
    pipe_b = _keep(_Pipe(torch.linspace(1000.0, 30.0, 4)))
    pipe_b.transformer = base
    assert cb.install_cfg_batch(pipe_b, offload_policy = "group") is None
    ref = _flux_cfg_loop(base)
    assert cb.router_for(pipe_b).stats["batched_steps"] > 0

    head = _flux()
    assert ap.install(head, None, requested = ap.MODE_ON)
    head = _mark_streamed(head)
    pipe_h = _keep(_Pipe(torch.linspace(1000.0, 30.0, 4)))
    pipe_h.transformer = head
    ap.bind_pipe(pipe_h)
    assert cb.install_cfg_batch(pipe_h, offload_policy = "group") is None
    out = _flux_cfg_loop(head)
    assert cb.router_for(pipe_h).stats["batched_steps"] > 0
    for a, b in zip(ref, out):
        assert torch.equal(a, b)


def _lora_file(tmp_path, keys):
    from safetensors.torch import save_file

    path = tmp_path / f"lora_{abs(hash(tuple(keys)))}.safetensors"
    save_file({k: torch.zeros(2, 2) for k in keys}, str(path))
    return str(path)


def _group_offloaded(model, fake_stream = True):
    """Real diffusers block-level group offload on CPU; ``fake_stream`` adds the stream-mode slot ring."""
    from core.inference.diffusion_memory import _offload_groups
    from core.inference.diffusion_offload_prefetch import PREFETCHER_ATTR, GroupPrefetcher

    model.enable_group_offload(
        torch.device("cpu"),
        offload_device = torch.device("cpu"),
        offload_type = "block_level",
        num_blocks_per_group = 1,
    )
    groups = _offload_groups(model)
    assert groups
    if not fake_stream:
        return groups, None
    for g in groups:
        g.stream = object()
        g.low_cpu_mem_usage = True
        g.cpu_param_dict = g._init_cpu_param_dict()
    pf = GroupPrefetcher(model, groups, "cpu", depth = 1)
    model.__dict__[PREFETCHER_ATTR] = pf
    assert pf.enable_slots() >= 0
    return groups, pf


def test_streamed_restore_rebuilds_offload_bookkeeping():
    from core.inference.diffusion_offload_prefetch import (
        _group_nbytes,
        _group_tensors,
        _packed_bytes,
    )

    model = _flux()
    assert ap.install(model, _pipe(4), requested = ap.MODE_ON)  # Studio's order: before the offload
    groups, pf = _group_offloaded(model)
    before = {id(g): pf.nbytes[id(g)] for g in groups}
    assert ap.uninstall(model, reason = "LoRA adapters requested")
    assert isinstance(model.transformer_blocks[0].norm1.linear, torch.nn.Linear)
    grew = 0
    for g in groups:
        for t in _group_tensors(g):
            assert t in g.cpu_param_dict
        assert pf.nbytes[id(g)] == _group_nbytes(g)
        grew += pf.nbytes[id(g)] > before[id(g)]
        if id(g) in pf.slot_of:
            need = _packed_bytes([g.cpu_param_dict[t] for t in _group_tensors(g)])
            assert need <= pf.slot_size
            views = pf._views_for(g, pf.slot_of[id(g)])  # KeyError before the rebuild
            assert len(views) == len(_group_tensors(g))
    assert (
        grew == len(model.transformer_blocks) + len(model.single_transformer_blocks) + 1
    )  # + norm_out's group
    assert pf.window >= (pf.depth + 1) * max(pf.nbytes.values())


def test_group_offloaded_render_after_restore_matches_stock():
    calls = _flux_calls(torch.float32)
    stock = _flux()
    _group_offloaded(stock, fake_stream = False)
    ref = _run(stock, calls)
    model = _flux()
    assert ap.install(model, _pipe(len(calls)), requested = ap.MODE_ON)
    _group_offloaded(model, fake_stream = False)
    assert all(torch.equal(a, b) for a, b in zip(ref, _run(model, calls)))
    assert ap.uninstall(model)
    assert all(torch.equal(a, b) for a, b in zip(ref, _run(model, calls)))


def test_lora_on_streamed_load_restores_modules(tmp_path):
    from core.inference import diffusion as D
    from core.inference.diffusion_offload_prefetch import _group_tensors

    model = _flux()
    pipe = _keep(_Pipe(torch.linspace(1000.0, 30.0, 4)))
    pipe.transformer = model
    assert ap.install(model, pipe, requested = ap.MODE_ON)
    groups, _pf = _group_offloaded(model)
    path = _lora_file(tmp_path, ["transformer.transformer_blocks.0.norm1.linear.lora_A.weight"])
    D._adaln_before_loras(types.SimpleNamespace(cuda_graphs = ()), pipe, [("m", path, 1.0)], None)
    assert not ap.installed(model)
    assert isinstance(model.transformer_blocks[0].norm1.linear, torch.nn.Linear)
    assert all(t in g.cpu_param_dict for g in groups for t in _group_tensors(g))


def test_lora_on_resident_load_restores_modules(tmp_path):
    from core.inference import diffusion as D

    model = _flux()
    pipe = _keep(_Pipe(torch.linspace(1000.0, 30.0, 4)))
    pipe.transformer = model
    assert ap.install(model, pipe, requested = ap.MODE_ON)
    path = _lora_file(tmp_path, ["transformer.transformer_blocks.0.norm1.linear.lora_A.weight"])
    D._adaln_before_loras(types.SimpleNamespace(cuda_graphs = ()), pipe, [("m", path, 1.0)], None)
    assert not ap.installed(model)
    assert isinstance(model.transformer_blocks[0].norm1.linear, torch.nn.Linear)


def test_step_protected_modulation_layers_stay_live():
    model = _flux()
    calls = _flux_calls(torch.float32)
    ref = _run(model, calls)
    protected = model.transformer_blocks[0].norm1.linear
    protected.protect = types.SimpleNamespace(
        armed = False, protected = False
    )  # NVFP4 per-step protection
    assert ap.install(model, _pipe(len(calls)), requested = ap.MODE_ON)
    assert model.transformer_blocks[0].norm1.linear is protected
    assert isinstance(model.transformer_blocks[0].norm1_context.linear, ap.AdaLNTable)
    assert all(torch.equal(a, b) for a, b in zip(ref, _run(model, calls)))


def test_baked_lora_load_declines():
    model = _flux()
    pipe = _keep(_Pipe(torch.linspace(1000.0, 30.0, 4)))
    pipe._unsloth_loras = (("a", "x.safetensors", 1.0),)
    assert ap.install(model, pipe, requested = ap.MODE_ON) is False


def test_sequential_cpu_offload_restore_keeps_weights_and_renders():
    """Sequential CPU offload: a LoRA request restores the modules under accelerate hooks."""
    from accelerate import cpu_offload
    from accelerate.hooks import remove_hook_from_module

    calls = _flux_calls(torch.float32)
    stock = _flux()
    cpu_offload(stock, execution_device = torch.device("cpu"))
    ref = _run(stock, calls)

    model = _flux()
    assert ap.install(
        model, _pipe(len(calls)), requested = ap.MODE_ON
    )  # before placement, as in Studio
    cpu_offload(model, execution_device = torch.device("cpu"))
    assert all(torch.equal(a, b) for a, b in zip(ref, _run(model, calls)))
    assert ap.uninstall(model, reason = "LoRA adapters requested")
    lin = model.transformer_blocks[0].norm1.linear
    hook = getattr(lin, "_hf_hook", None)
    assert isinstance(lin, torch.nn.Linear) and hook is not None
    align = [
        h
        for h in (getattr(hook, "hooks", None) or (hook,))
        if type(h).__name__ == "AlignDevicesHook" and h.offload
    ]
    assert align and align[0].offload
    assert all(align[0].weights_map[k].device.type != "meta" for k in ("weight", "bias"))
    assert all(torch.equal(a, b) for a, b in zip(ref, _run(model, calls)))
    remove_hook_from_module(model, recurse = True)
    assert lin.weight.device.type != "meta"
    cpu_offload(model, execution_device = torch.device("cpu"))
    assert all(torch.equal(a, b) for a, b in zip(ref, _run(model, calls)))


def test_precompute_holds_at_most_the_table_budget(monkeypatch):
    import weakref

    calls = _flux_calls(torch.float32, steps = 10)
    ref = _run(_flux(), calls)
    model = _flux()
    assert ap.install(model, _pipe(len(calls)), requested = ap.MODE_ON)
    st = model.__dict__[ap.STATE_ATTR]
    row = sum(int(stand.out_features) for stand in st.stand_ins) * 4
    budget = int(row * 3.5)
    monkeypatch.setattr(ap, "_table_budget_bytes", lambda: budget)
    live: list = []
    peak = [0]

    def watch(_mod, _inp, out):
        live.append(weakref.ref(out))
        held = sum(r().numel() * r().element_size() for r in live if r() is not None)
        peak[0] = max(peak[0], held + st.row_bytes)

    for stand in st.stand_ins:
        stand._unsloth_orig.register_forward_hook(watch)
    out = _run(model, calls)
    assert all(torch.equal(a, b) for a, b in zip(ref, out))
    assert peak[0] <= budget + row, (peak[0], budget, row)
    assert st.row_bytes <= budget


def test_cfg_batch_doubled_call_failure_falls_back_on_its_own_rows(monkeypatch):
    """The doubled forward raises after arming; the fallback and later calls read their own rows."""
    from core.inference import diffusion_cfg_batch as cb

    monkeypatch.setenv(cb.CFG_BATCH_ENV, "force")
    ref = _flux_cfg_loop(_flux())  # router off, lever off

    head = _flux()
    assert ap.install(head, None, requested = ap.MODE_ON)
    head = _mark_streamed(head)
    pipe = _keep(_Pipe(torch.linspace(1000.0, 30.0, 4)))
    pipe.transformer = head
    ap.bind_pipe(pipe)
    assert cb.install_cfg_batch(pipe, offload_policy = "group") is None
    router = cb.router_for(pipe)
    real = router.orig_forward
    doubled = []

    def forward(*args, **kwargs):
        hs = kwargs.get("hidden_states", args[0] if args else None)
        if hs is not None and hs.shape[0] == 2:
            doubled.append(1)
            raise torch.OutOfMemoryError("simulated: no room for the doubled activations")
        return real(*args, **kwargs)

    router.orig_forward = forward  # the forward the router calls past module.__call__
    served = ap.COUNTS["served"]
    out = _flux_cfg_loop(head)
    assert doubled, "the doubled call never ran"
    assert ap.COUNTS["served"] > served
    for a, b in zip(ref, out):
        assert torch.equal(a, b)


def test_renders_under_inference_mode_like_diffusion_backend(monkeypatch):
    """Inference-mode renders (no version counter) plain, CFG-batched and no_grad all match stock."""
    from core.inference import diffusion_cfg_batch as cb

    with torch.inference_mode():  # the pipeline builds its timesteps inside it: inference tensors
        calls = _flux_calls(torch.bfloat16)
        assert calls[0]["timestep"].is_inference()
        ref = [_flux().to(torch.bfloat16)(**kw)[0].clone() for kw in calls]
        ref_cfg = _flux_cfg_loop(_flux())
    model = _flux().to(torch.bfloat16)
    assert ap.install(model, _pipe(len(calls)), requested = ap.MODE_ON)
    with torch.inference_mode():
        out = [model(**kw)[0].clone() for kw in calls]
    assert all(torch.equal(a, b) for a, b in zip(ref, out))
    with (
        torch.no_grad()
    ):  # a later call outside inference_mode refills the same row buffer in place
        plain_calls = _flux_calls(torch.bfloat16)
        again = [model(**kw)[0].clone() for kw in plain_calls]
    assert all(torch.equal(a, b) for a, b in zip(ref, again))

    monkeypatch.setenv(cb.CFG_BATCH_ENV, "force")
    base = _mark_streamed(_flux())
    pipe_b = _keep(_Pipe(torch.linspace(1000.0, 30.0, 4)))
    pipe_b.transformer = base
    assert cb.install_cfg_batch(pipe_b, offload_policy = "group") is None
    head = _flux()
    assert ap.install(head, None, requested = ap.MODE_ON)
    head = _mark_streamed(head)
    pipe_h = _keep(_Pipe(torch.linspace(1000.0, 30.0, 4)))
    pipe_h.transformer = head
    ap.bind_pipe(pipe_h)
    assert cb.install_cfg_batch(pipe_h, offload_policy = "group") is None
    with torch.inference_mode():
        ref_batched = _flux_cfg_loop(base)
        out_batched = _flux_cfg_loop(head)
    assert cb.router_for(pipe_h).stats["batched_steps"] > 0
    assert all(torch.equal(a, b) for a, b in zip(ref_batched, out_batched))
    assert len(ref_cfg) == len(out_batched)


def test_cfg_replica_copy_takes_the_real_modules():
    from core.inference.diffusion_multigpu import _copy_module

    calls = _flux_calls(torch.float32)
    ref = _run(_flux(), calls)
    model = _flux()
    assert ap.install(model, _pipe(len(calls)), requested = ap.MODE_ON)
    replica = _copy_module(model, torch.device("cpu"))
    assert ap.STATE_ATTR not in replica.__dict__
    for blk in replica.transformer_blocks:
        assert type(blk.norm1.linear) is torch.nn.Linear
        assert type(blk.norm1_context.linear) is torch.nn.Linear
    for blk in replica.single_transformer_blocks:
        assert type(blk.norm.linear) is torch.nn.Linear
    assert type(replica.norm_out.linear) is torch.nn.Linear
    assert all(torch.equal(a, b) for a, b in zip(ref, _run(replica, calls)))
    assert ap.installed(model)
    assert all(torch.equal(a, b) for a, b in zip(ref, _run(model, calls)))


def test_layerwise_cast_modulation_layers_stay_live():
    """A layerwise-cast (hook-bound) Linear stays live: its device clone would use host weights."""
    from core.inference.diffusion_small_host import prepare_streamed_encoder_

    model = _flux().to(torch.bfloat16)
    prepare_streamed_encoder_(model, torch.float32)
    calls = _flux_calls(torch.float32)
    ref = _run(model, calls)
    assert ap.install(model, _pipe(len(calls)), requested = ap.MODE_ON, offloaded = True) is False
    assert not any(isinstance(m, ap.AdaLNTable) for m in model.modules())
    assert all(torch.equal(a, b) for a, b in zip(ref, _run(model, calls)))


def test_hook_bound_clone_would_read_the_host_original():
    """Why: the clone of a hook-bound Linear keeps the original's bound ``forward``, so its own weights are ignored."""
    from diffusers.hooks import apply_layerwise_casting

    lin = torch.nn.Linear(8, 4)
    apply_layerwise_casting(lin, storage_dtype = torch.bfloat16, compute_dtype = torch.float32)
    clone = ap._device_clone(lin, torch.device("cpu"))
    clone._parameters["weight"] = torch.nn.Parameter(torch.zeros(4, 8), requires_grad = False)
    clone._parameters["bias"] = torch.nn.Parameter(torch.zeros(4), requires_grad = False)
    assert clone(torch.ones(1, 8)).abs().sum() > 0
    assert ap._hook_bound(lin) and not ap._hook_bound(torch.nn.Linear(8, 4))


def test_small_host_route_declines():
    from core.inference.diffusion_small_host import SMALL_HOST_ATTR

    model = _flux()
    pipe = _keep(_Pipe(torch.linspace(1000.0, 30.0, 4)))
    setattr(pipe, SMALL_HOST_ATTR, {"reason": "dense load does not fit", "components": {}})
    assert ap.install(model, pipe, requested = ap.MODE_ON, offloaded = True) is False
    assert not any(isinstance(m, ap.AdaLNTable) for m in model.modules())


def test_streamed_restore_without_moduleGroup_to_cpu(tmp_path, monkeypatch):
    """diffusers 0.36 has no ``ModuleGroup._to_cpu``: the restore builds the same host copy."""
    from diffusers.hooks import group_offloading as go

    from core.inference import diffusion as D
    from core.inference.diffusion_offload_prefetch import _group_tensors

    model = _flux()
    pipe = _keep(_Pipe(torch.linspace(1000.0, 30.0, 4)))
    pipe.transformer = model
    assert ap.install(model, pipe, requested = ap.MODE_ON)
    groups, _pf = _group_offloaded(model)
    monkeypatch.delattr(go.ModuleGroup, "_to_cpu", raising = False)
    path = _lora_file(tmp_path, ["transformer.transformer_blocks.0.norm1.linear.lora_A.weight"])
    D._adaln_before_loras(types.SimpleNamespace(cuda_graphs = ()), pipe, [("m", path, 1.0)], None)
    assert not ap.installed(model)
    lin = model.transformer_blocks[0].norm1.linear
    assert isinstance(lin, torch.nn.Linear)
    assert all(t in g.cpu_param_dict for g in groups for t in _group_tensors(g))
    host = next(g for g in groups if lin.weight in g.cpu_param_dict).cpu_param_dict[lin.weight]
    assert torch.equal(host, lin.weight.data)


def test_modulation_weights_pin_through_the_packed_arena(monkeypatch):
    """Every weight moved to host goes through one PinnedArena (per-tensor pins round up)."""
    from core.inference import diffusion_pinned_arena as arena_mod

    seen = []

    class _Arena:
        def __init__(
            self,
            slab_bytes = 1 << 30,
            expected_bytes = None,
        ):
            self.expected_bytes = expected_bytes

        def pin(self, t):
            seen.append(t.numel() * t.element_size())
            return t.clone()

    monkeypatch.setattr(arena_mod, "PinnedArena", _Arena)
    monkeypatch.setenv(arena_mod.PIN_ARENA_ENV, "1")
    model = _flux()
    calls = _flux_calls(torch.float32)
    ref = _run(model, calls)
    mod_bytes = sum(
        p.numel() * p.element_size()
        for n, p in model.named_parameters()
        if "norm" in n and "linear" in n
    )
    assert ap.install(model, _pipe(len(calls)), requested = ap.MODE_ON)
    assert sum(seen) == mod_bytes and ap.summary(model)["host_gib"] == round(mod_bytes / 2**30, 3)
    assert all(torch.equal(a, b) for a, b in zip(ref, _run(model, calls)))


def test_multigpu_batch_split_arms_each_half(monkeypatch):
    """A distilled batch of 2 on a multi-GPU router splits along the batch: the source half runs past
    ``model.__call__``, after the pre-hook armed the full batch, so it must arm its own half's rows (and so must the
    source-side retry of the second half and the whole-batch fallback)."""
    from core.inference import diffusion_multigpu as mg

    monkeypatch.delenv("UNSLOTH_DIFFUSION_MULTIGPU_ALL", raising = False)
    calls = _flux_calls(torch.float32, batch = 2, steps = 4)
    for (
        kw
    ) in calls:  # a new latent each step, as a sampler makes: one call per step, no guidance pair
        kw["hidden_states"] = kw["hidden_states"].clone()
    ref = _run(_flux(), calls)
    model = _flux()
    assert ap.install(model, _pipe(len(calls)), requested = ap.MODE_ON)
    router = mg.CfgParallelRouter(
        model,
        primary = "cpu",
        secondary = "cpu",
        replica_factory = lambda: mg.build_replica(model, torch.device("cpu")),
    )
    model.forward = router
    with torch.no_grad():
        out = [model(**kw)[0].clone() for kw in calls]
    assert router.stats["batch_splits"] == len(calls) - 1, router.stats
    for a, b in zip(ref, out):
        torch.testing.assert_close(a, b, rtol = 1e-5, atol = 1e-6)

    # the replica refuses: the second half retries on the source, with its own rows
    monkeypatch.setattr(router, "run_pair", lambda primary_call, a, k, **kw: (primary_call(), None))
    with torch.no_grad():
        again = [model(**kw)[0].clone() for kw in calls]
    for a, b in zip(ref, again):
        torch.testing.assert_close(a, b, rtol = 1e-5, atol = 1e-6)


def test_host_move_keeps_gguf_parameters_and_their_quant_type():
    """An arena-pinned GGUF weight stays a GGUFParameter with its quant_type."""
    utils = pytest.importorskip("diffusers.quantizers.gguf.utils")

    def arena_like(t):
        dst = (
            torch.empty(t.numel() * t.element_size(), dtype = torch.uint8).view(t.dtype).view(t.shape)
        )
        dst.copy_(t)
        return dst

    lin = torch.nn.Linear(4, 4, bias = True)
    packed = utils.GGUFParameter(torch.arange(16, dtype = torch.uint8).view(4, 4), quant_type = 8)
    lin.weight = packed
    ap._to_host(lin, pin = arena_like)
    assert type(lin.weight) is utils.GGUFParameter
    assert lin.weight.quant_type == 8 and lin.weight.quant_shape == packed.quant_shape
    assert torch.equal(
        lin.weight.detach().view(torch.uint8), torch.arange(16, dtype = torch.uint8).view(4, 4)
    )
    clone = ap._device_clone(lin, torch.device("cpu"))
    assert getattr(clone.weight, "quant_type", None) == 8
    assert type(lin.bias) is torch.nn.Parameter  # plain parameters keep the plain wrapper


def test_cross_device_rows_do_not_recompile_a_fullgraph_block():
    # The counter read is a Dynamo value guard: counted under compile, every call recompiles until the limit.
    stand = ap.AdaLNTable(torch.nn.Linear(4, 6), "transformer_blocks.0.norm1.linear", "silu")
    stand.unsloth_adaln_cur = torch.randn(1, 6)
    compiled = torch.compile(stand, backend = "eager", fullgraph = True)
    x = torch.empty(1, 4, device = "meta")
    for _ in range(12):
        out = compiled(x)
    assert out.device.type == "meta" and tuple(out.shape) == (1, 6)


def _gguf_q8(model, match):
    """Swaps matching Linears for the GGUFLinear diffusers' GGUF loader builds from a Q8_0 tensor."""
    gguf = pytest.importorskip("gguf")
    utils = pytest.importorskip("diffusers.quantizers.gguf.utils")
    dtype = next(model.parameters()).dtype
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if not (isinstance(child, torch.nn.Linear) and match(full)):
                continue
            q = gguf.quantize(child.weight.detach().float().numpy(), gguf.GGMLQuantizationType.Q8_0)
            lin = utils.GGUFLinear(
                child.in_features,
                child.out_features,
                bias = child.bias is not None,
                compute_dtype = dtype,
            )
            lin.weight = utils.GGUFParameter(
                torch.from_numpy(q), quant_type = gguf.GGMLQuantizationType.Q8_0
            )
            if child.bias is not None:
                lin.bias = torch.nn.Parameter(child.bias.detach(), requires_grad = False)
            setattr(mod, cname, lin)
    return model


def _gguf_qwen_edit(dtype = torch.float32):
    """Qwen-Image-Edit-2511's layout: zero_cond_t, GGUF time embedder and per-block modulation."""
    model = _qwen(zero_cond_t = True).to(dtype)
    return _gguf_q8(model, lambda n: "time_text_embed" in n or "_mod." in n).requires_grad_(False)


def test_gguf_time_embedder_gets_a_host_copy():
    """A deep-copied GGUFParameter loses its quant_type, so the copy is built tensor by tensor."""
    model = _gguf_qwen_edit()
    calls = _qwen_calls(torch.float32, zero_cond_t = True)
    ref = _run(model, calls)
    assert ap.install(model, _pipe(len(calls)), requested = ap.MODE_ON)
    host = model.__dict__[ap.STATE_ATTR].tte_host
    assert host is not None
    live = model.time_text_embed.timestep_embedder.linear_1.weight
    copied = host.timestep_embedder.linear_1.weight
    assert copied.quant_type == live.quant_type and copied.data_ptr() != live.data_ptr()
    t = torch.rand(2) / 2
    hs = torch.empty(1)
    with torch.no_grad():
        assert torch.equal(host(t, hs, None), model.time_text_embed(t, hs, None))
    assert all(torch.equal(a, b) for a, b in zip(ref, _run(model, calls)))
    assert ap.installed(model)


def test_fallback_restore_joins_the_top_level_host_map():
    """Studio's pinned / no-copy-back top-level group onloads only its host map: a restored norm_out must join it."""
    from diffusers.hooks import group_offloading as go

    from core.inference.diffusion_memory import TOP_GROUP_HOST_ATTR

    model = _qwen(zero_cond_t = True)
    calls = _qwen_calls(torch.float32, zero_cond_t = True)
    ref = _run(model, calls)
    assert ap.install(model, _pipe(len(calls)), requested = ap.MODE_ON)
    _group_offloaded(model, fake_stream = False)
    top = model._diffusers_hook.get_hook(go._GROUP_OFFLOADING).group
    host = {t: t.data for m in top.modules for t in list(m.parameters()) + list(m.buffers())}
    setattr(top, TOP_GROUP_HOST_ATTR, host)
    # the stand-in held norm_out.linear when the map was built
    assert not any(t is model.norm_out.linear._unsloth_orig.weight for t in host)
    assert ap.uninstall(model, reason = "unsupported call: test")
    lin = model.norm_out.linear
    assert lin.weight in host and lin.bias in host
    assert all(torch.equal(a, b) for a, b in zip(ref, _run(model, calls)))


def test_host_copy_runs_on_its_own_weights():
    """A deep copy: an instance-bound forward (MethodType patch) is rebound to the copy, never the live module."""
    model = _gguf_qwen_edit()
    lin = model.time_text_embed.timestep_embedder.linear_2
    lin.forward = types.MethodType(type(lin).forward, lin)
    host = ap._host_copy(model.time_text_embed)
    copied = host.timestep_embedder.linear_2
    assert copied.forward.__self__ is copied
    assert copied.weight.quant_type == lin.weight.quant_type


@pytest.mark.parametrize("guidance", [True, False])
def test_gguf_flux_time_embedder_gets_a_host_copy(guidance):
    """FLUX.1 GGUF (dev and schnell embedders): same deepcopy failure, same host copy."""
    model = _gguf_q8(
        _flux(guidance = guidance), lambda n: "timestep_embedder" in n or "guidance_embedder" in n
    )
    calls = _flux_calls(torch.float32, guidance = 3.5 if guidance else None)
    ref = _run(model, calls)
    assert ap.install(model, _pipe(len(calls)), requested = ap.MODE_ON)
    assert model.__dict__[ap.STATE_ATTR].tte_host is not None
    assert all(torch.equal(a, b) for a, b in zip(ref, _run(model, calls)))


# CUDA only: the failure is a CPU weight meeting a CUDA activation after an offload onload
@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
@pytest.mark.parametrize("policy", ["streaming", "group"])
@pytest.mark.parametrize("top", ["pinned", "pageable"])
@pytest.mark.parametrize("fallback", [False, True])
def test_gguf_qwen_edit_streamed_render_matches_stock(monkeypatch, policy, top, fallback):
    """Issue 18759: GGUF Qwen-Image-Edit placed by Studio's streaming / group offload, top-level group pinned or
    pageable with the copy-back skipped, rendered every step (precompute on, and after a fallback) bit-identical
    to the stock model."""
    from diffusers.hooks import group_offloading as go

    from core.inference import diffusion_memory as dm

    if top == "pageable":
        monkeypatch.setattr(dm, "_pin_top_level_group", lambda *_a, **_k: False)

    def build(precompute):
        model = _gguf_qwen_edit(torch.bfloat16)
        calls = _qwen_calls(torch.bfloat16, zero_cond_t = True)
        pipe = _keep(_Pipe(torch.linspace(1000.0, 30.0, len(calls))))
        pipe.transformer = model
        pipe.components = {"transformer": model}
        if precompute:
            assert ap.install(model, pipe, offloaded = True)
            if fallback:
                model.__dict__[ap.STATE_ATTR].tte_host = None
        if policy == "streaming":
            dm._apply_streaming_offload(pipe, "cuda", None)
        else:
            assert dm._apply_group_offload(pipe, "cuda", None)
        group = model._diffusers_hook.get_hook(go._GROUP_OFFLOADING).group
        flag = "_unsloth_pinned_top" if top == "pinned" else "_unsloth_top_no_copy_back"
        if not getattr(group, flag, False):
            pytest.skip(f"{top} top-level group not built here")
        cuda = []
        for kw in calls:
            kw = dict(kw)
            for k in (
                "hidden_states",
                "encoder_hidden_states",
                "encoder_hidden_states_mask",
                "timestep",
            ):
                kw[k] = kw[k].to("cuda")
            cuda.append(kw)
        return model, cuda

    stock, calls = build(False)
    ref = [o.cpu() for o in _run(stock, calls)]
    model, calls = build(True)
    out = [o.cpu() for o in _run(model, calls)]
    assert all(torch.equal(a, b) for a, b in zip(ref, out))
    assert all(torch.equal(a, b) for a, b in zip(ref, [o.cpu() for o in _run(model, calls)]))
    assert ap.installed(model) is not fallback
