# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AdaLN precompute against real diffusers FLUX.1 / Qwen-Image transformers (tiny random configs): the denoiser
outputs stay bit-identical while the modulation weights live on the host and are read once per request."""

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


# the transformer holds its pipe weakly, as Studio's pipe outlives every render


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
    # a second request with the same inputs re-reads no weights
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
    # each pooled / guidance set has its own rows: 3 requests' worth
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
    # shared-modulation families are not installed
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


# ---- review fixes (PR #18342)


def test_layer_split_declines_and_rows_follow_block_device():
    model = _flux()
    # a denoiser planned split across cards takes no lever
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
    # the load places the denoiser after install (CPU mock: "meta" as the GPU); restore without a device
    model.to("meta")
    assert ap.uninstall(model, reason = "LoRA adapters requested")
    for blk in model.transformer_blocks:
        assert blk.norm1.linear.weight.device.type == "meta"
        assert blk.norm1_context.linear.weight.device.type == "meta"
    for blk in model.single_transformer_blocks:
        assert blk.norm.linear.weight.device.type == "meta"
    assert model.norm_out.linear.weight.device.type == "meta"


def test_qwen_uninstall_restores_beside_owning_block():
    model = _qwen()
    assert ap.install(model, _pipe(4), requested = ap.MODE_ON)
    model.to("meta")
    assert ap.uninstall(model)
    assert model.transformer_blocks[0].img_mod[1].weight.device.type == "meta"
    assert model.transformer_blocks[1].txt_mod[1].weight.device.type == "meta"


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


# ---- review round 2 (PR #18342)


def _mark_streamed(model):
    # what Studio's offload detection reads: a diffusers group-offloading registry on a submodule
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


def test_streamed_denoiser_never_gets_modules_restored():
    model = _flux()
    assert ap.install(model, _pipe(4), requested = ap.MODE_ON)
    _mark_streamed(model)
    assert ap.offload_reason(model) is not None
    with pytest.raises(RuntimeError, match = "streamed"):
        ap.uninstall(model)
    assert ap.installed(model)


def _lora_file(tmp_path, keys):
    from safetensors.torch import save_file

    path = tmp_path / f"lora_{abs(hash(tuple(keys)))}.safetensors"
    save_file({k: torch.zeros(2, 2) for k in keys}, str(path))
    return str(path)


def test_lora_on_streamed_load_keeps_lever_or_refuses(tmp_path):
    from core.inference import diffusion as D
    from core.inference.diffusion_lora import LoraSelectionError

    model = _flux()
    pipe = _keep(_Pipe(torch.linspace(1000.0, 30.0, 4)))
    pipe.transformer = model
    assert ap.install(model, pipe, requested = ap.MODE_ON)
    _mark_streamed(model)
    state = types.SimpleNamespace(cuda_graphs = ())
    attn_only = _lora_file(tmp_path, ["transformer.transformer_blocks.0.attn.to_q.lora_A.weight"])
    D._adaln_before_loras(state, pipe, [("a", attn_only, 1.0)], None)
    assert ap.installed(model)  # adapters off the modulation path: lever kept, nothing restored
    for key in (
        "transformer.transformer_blocks.0.norm1.linear.lora_A.weight",
        "lora_unet_double_blocks_0_img_mod_lin.lora_down.weight",
        "transformer.time_text_embed.timestep_embedder.linear_1.lora_A.weight",
    ):
        with pytest.raises(LoraSelectionError):
            D._adaln_before_loras(state, pipe, [("m", _lora_file(tmp_path, [key]), 1.0)], None)
        assert ap.installed(model)


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
