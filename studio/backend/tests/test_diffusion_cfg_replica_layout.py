# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The CFG replica layout's pure parts (price, fit, OOM fallback). CPU only."""

import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_multigpu as mg  # noqa: E402


@pytest.mark.parametrize(
    "guided, compute, stream, steps, encoders, wins",
    [
        (True, 2.65, 1.06, 10, 0.0, True),
        (True, 1.0, 0.0, 20, 0.0, True),
        (True, 0.3, 4.7, 10, 0.0, False),
        (True, 1.0, 0.97, 10, 0.0, False),
        (False, 2.65, 0.0, 10, 0.0, False),
        (False, 2.65, 1.06, 10, 0.0, False),
        (True, 1.5, 0.0, 50, 1.0, True),
        (True, 0.05, 0.0, 1, 1.0, False),
    ],
)
def test_replica_layout_price_table(guided, compute, stream, steps, encoders, wins):
    take, replica, alternative = mg.choose_replica_layout(
        guided = guided,
        compute_s = compute,
        primary_stream_s = stream,
        steps = steps,
        encoder_stream_s = encoders,
    )
    assert take is wins, (replica, alternative)


def test_price_matches_the_kaggle_measurement():
    replica, split = mg.price_replica_layout(
        guided = True, compute_s = 2.65, primary_stream_s = 1.06, steps = 10
    )
    assert abs(replica / 10 - 3.71) < 0.01 and abs(split / 10 - 5.30) < 0.01


@pytest.mark.parametrize(
    "free, total, need, fits",
    [
        (14600, 15360, 9582, True),  # Wan 5B fp16 replica on an idle T4
        (10400, 15360, 9582, False),  # the 1 GiB reserve does not fit beside it
        (None, 15360, 100, False),  # unknown free memory never fits
    ],
)
def test_replica_fits_keeps_the_reserve(free, total, need, fits):
    assert mg.replica_fits(free, total, need) is fits


class _Blocks(torch.nn.Module):
    _repeated_blocks = ["Block"]

    def __init__(self, width):
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block(width) for _ in range(3)])


class Block(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.w = torch.nn.Parameter(torch.empty(width, width, dtype = torch.float16, device = "meta"))


def test_replica_need_adds_only_a_working_set_over_the_reserve(monkeypatch):
    small = _Blocks(64)
    assert mg.replica_working_mib(small) == 1024  # the floor, already inside the 1 GiB reserve
    weights = sum(p.numel() * p.element_size() for p in small.parameters()) >> 20
    assert mg.replica_need_mib(small) == weights
    monkeypatch.setattr(mg, "_module_mib", lambda m: 600)
    assert mg.replica_working_mib(small) == 1200
    assert mg.replica_need_mib(small) == weights + 1200 - 1024


# Meta-device stand-ins: real configs, weights as recorded on the Kaggle 2x T4 sweep.
_WAN_5B = dict(
    config = dict(num_attention_heads = 24, attention_head_dim = 128, ffn_dim = 14336, num_layers = 30),
    weights_mib = 9558,
    dtype = torch.float16,
    tokens = 27280,
)
_MOSS = dict(
    config = dict(dim = 3072, num_heads = 24, ffn_dim = 8192, num_layers = 42),
    weights_mib = 4857,
    dtype = torch.float32,
    tokens = 4096,
)


class _Denoiser(torch.nn.Module):
    def __init__(self, config, weights_mib, dtype):
        super().__init__()
        self.config = types.SimpleNamespace(**config) if config is not None else None
        self.w = torch.nn.Parameter(
            torch.empty((weights_mib << 20) // dtype.itemsize, dtype = dtype, device = "meta")
        )


class _Pipe:
    pass


def _family(spec, monkeypatch):
    model = _Denoiser(spec["config"], spec["weights_mib"], spec["dtype"])
    pipe = _Pipe()
    pipe.transformer = model
    monkeypatch.setattr(mg, "_DEFAULT_REQUESTS", mg.weakref.WeakKeyDictionary())
    mg.note_default_request(pipe, guided = True, steps = 20, tokens = spec["tokens"])
    return pipe, model


def test_replica_activation_matches_the_measured_wan_peak():
    wan = _Denoiser(_WAN_5B["config"], 1, torch.float16)
    # Kaggle T4, 1280x704x49 (11440 tokens): replica card peaked 10956 MiB allocated over 9558 of weights
    assert abs(mg.replica_activation_mib(wan, 11440) - (10956 - 9558)) < 100
    assert mg.replica_activation_mib(wan, 2 * 11440) <= 2 * mg.replica_activation_mib(wan, 11440)


def test_replica_need_prices_the_default_request_tokens(monkeypatch):
    pipe, wan = _family(_WAN_5B, monkeypatch)
    activation = mg.replica_activation_mib(wan, 27280)
    assert activation > 3000
    assert mg.replica_need_mib(wan) == 9558 + activation
    monkeypatch.setattr(mg, "_DEFAULT_REQUESTS", mg.weakref.WeakKeyDictionary())
    assert mg.replica_need_mib(wan) == 9558 + mg.replica_activation_mib(wan, 4096)


@pytest.mark.parametrize(
    "config", [None, dict(attention_head_dim = [5, 10, 20]), dict(sample_size = 128)]
)
def test_a_denoiser_without_a_dit_width_keeps_the_weights_only_need(config):
    unet = _Denoiser(config, 2048, torch.float16)
    assert mg.replica_activation_mib(unet) == 0
    assert mg.replica_need_mib(unet) == 2048


def _pair(total, name):
    return [mg.DeviceProbe(o, name, total, total - 400, util_pct = 0.0) for o in (0, 1)]


@pytest.mark.parametrize(
    "spec, total, admitted",
    [
        (_MOSS, 15360, True),  # 2x T4 (the recorded host)
        (_MOSS, 23034, True),  # 2x L4
        (_MOSS, 40960, True),  # 2x A100-40
        (_MOSS, 24564, True),  # 2x RTX 4090
        (_MOSS, 12288, True),  # 2x 12 GB
        (_WAN_5B, 15360, True),
        (_WAN_5B, 23034, True),
        (_WAN_5B, 40960, True),
        (_WAN_5B, 24564, True),
        # 9558 + 1024 fits 11888 free, but the replica peaked 11756 MiB allocated on a T4: no replica here
        (_WAN_5B, 12288, False),
    ],
)
def test_automatic_replica_admission_counts_activations(monkeypatch, spec, total, admitted):
    _pipe, model = _family(spec, monkeypatch)
    primary, secondary = _pair(total, f"GPU {total}")
    pick, why = mg.automatic_partner(primary, [secondary], mg.replica_need_mib(model))
    assert (pick == 1) is admitted, why
    if not admitted:
        assert "short of the" in why


_WAN_T4 = dict(_WAN_5B, config = dict(_WAN_5B["config"], patch_size = (1, 2, 2)))


def _wan_latent(frames):
    # 1280x704 through the Wan 2.2 5B VAE (16x spatial, 4x temporal)
    return torch.empty(1, 48, (frames - 1) // 4 + 1, 704 // 16, 1280 // 16, device = "meta")


@pytest.mark.parametrize("frames, tokens", [(49, 11440), (121, 27280)])
def test_latent_tokens_read_the_actual_request(frames, tokens):
    model = _Denoiser(_WAN_T4["config"], 1, torch.float16)
    assert mg.latent_tokens(model, _wan_latent(frames)) == tokens
    assert mg.latent_tokens(model, torch.empty(1, 4096, 64, device = "meta")) == 4096
    assert mg.latent_tokens(model, None) is None


@pytest.mark.parametrize("force", [False, True])
def test_an_explicit_selection_is_not_declined_on_the_default_request(monkeypatch, force):
    # default (121 frames) needs 12915+1024 MiB; 49 frames needs 10966+1024 of 13000 free
    pipe, model = _family(_WAN_T4, monkeypatch)
    if force:
        monkeypatch.setenv(mg.CFG_PARALLEL_ENV, "force")
    else:
        monkeypatch.delenv(mg.CFG_PARALLEL_ENV, raising = False)
    selection = mg.MultiGpuSelection(0, (1,), automatic = False)
    probes = {0: mg.DeviceProbe(0, "T4", 15360, 14000), 1: mg.DeviceProbe(1, "T4", 15360, 13000)}
    monkeypatch.setattr(mg, "current_selection", lambda *a, **k: selection)
    monkeypatch.setattr(mg, "probe_devices", lambda *a, **k: probes)
    monkeypatch.setattr(mg, "_execution_device", lambda *a, **k: torch.device("cuda:0"))
    assert mg.install_cfg_parallel(pipe, offload_policy = "none") is None
    assert isinstance(model.forward, mg.CfgParallelRouter)


@pytest.mark.parametrize("frames, built", [(49, True), (121, False)])
def test_the_lazy_replica_build_prices_the_first_guided_branch(monkeypatch, frames, built):
    _pipe, model = _family(_WAN_T4, monkeypatch)
    router = mg.CfgParallelRouter(model, primary = "cpu", secondary = "cpu")
    monkeypatch.setattr(router, "_secondary_free_total_mib", lambda: (13000, 15360))

    def past_the_memory_gate(module):
        raise RuntimeError("past the memory gate")

    monkeypatch.setattr(mg, "weights_fingerprint", past_the_memory_gate)
    router._build_replica(_wan_latent(frames))
    if built:
        assert "past the memory gate" in router.disabled_reason
    else:
        assert "replica needs 12915 MiB, 13000 MiB free" in router.disabled_reason


def test_automatic_still_prices_the_default_request(monkeypatch):
    monkeypatch.delenv(mg.CFG_PARALLEL_ENV, raising = False)
    assert mg._prices_default_request(mg.MultiGpuSelection(0, (1,), automatic = True))
    assert not mg._prices_default_request(mg.MultiGpuSelection(0, (1,), automatic = False))
    monkeypatch.setenv(mg.CFG_PARALLEL_ENV, "force")
    assert not mg._prices_default_request(mg.MultiGpuSelection(0, (1,), automatic = True))


def test_video_default_tokens_follow_the_vae_and_patch():
    wan = types.SimpleNamespace(
        vae_scale_factor_spatial = 16,
        vae_scale_factor_temporal = 4,
        transformer = types.SimpleNamespace(config = types.SimpleNamespace(patch_size = (1, 2, 2))),
    )
    assert mg.video_default_tokens(wan, 832, 480, 49) == 5070
    ltx = types.SimpleNamespace(
        vae_scale_factor_spatial = 32,
        vae_scale_factor_temporal = 8,
        transformer = types.SimpleNamespace(
            config = types.SimpleNamespace(patch_size = 1, patch_size_t = 1)
        ),
    )
    assert mg.video_default_tokens(ltx, 768, 512, 121) == 16 * 24 * 16
    assert mg.video_default_tokens(types.SimpleNamespace(), 832, 480, 49) is None


@pytest.mark.parametrize(
    "cls, refused",
    [
        ("WanTransformer3DModel", None),
        ("QwenImageTransformer2DModel", None),
        ("FluxTransformer2DModel", None),
        ("ZImageTransformer2DModel", None),
        ("HunyuanVideo15Transformer3DModel", None),
        ("HiDreamImageTransformer2DModel", "llama states"),
        ("Ideogram4Transformer2DModel", "unconditional transformer"),
        ("MiniMaxH3Transformer3DModel", "one denoiser call"),
        ("SomeNewTransformer", "not a denoiser class"),
    ],
)
def test_replica_layout_family_table(cls, refused):
    module = type(cls, (torch.nn.Module,), {})()
    why = mg.replica_layout_refusal(module)
    assert (why is None) if refused is None else (refused in why), why


def test_kill_switch_reads():
    for raw, mode in (("0", "off"), ("force", "force"), ("", "auto"), ("1", "auto")):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv(mg.CFG_REPLICA_PLACEMENT_ENV, raw)
            assert mg.cfg_replica_placement_mode() == mode


class WanTransformer3DModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x = torch.nn.Linear(8, 8)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states = None,
        timestep = None,
        return_dict = False,
    ):
        h = self.x(hidden_states) * timestep[:, None, None] + encoder_hidden_states.mean(
            1, keepdim = True
        )
        return (h,)


def _guided(model, steps = 4):
    g = torch.Generator().manual_seed(3)
    latents = torch.randn(1, 16, 8, generator = g)
    cond, uncond = torch.randn(1, 5, 8, generator = g), torch.randn(1, 5, 8, generator = g)
    for i in range(steps):
        t = torch.full((1,), 1.0 - i / steps)
        c = model(hidden_states = latents, encoder_hidden_states = cond, timestep = t)[0]
        u = model(hidden_states = latents, encoder_hidden_states = uncond, timestep = t)[0]
        latents = latents - 0.1 * (u + 5.0 * (c - u))
    return latents


def test_replica_out_of_memory_falls_back_to_one_card_and_says_so():
    torch.manual_seed(0)
    ref = _guided(WanTransformer3DModel())
    torch.manual_seed(0)
    model = WanTransformer3DModel()

    class OutOfMemory(torch.nn.Module):
        def forward(self, *a, **k):
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    router = mg.CfgParallelRouter(
        model, primary = "cpu", secondary = "cpu", replica_factory = OutOfMemory
    )
    model.forward = router

    class Pipe:
        transformer = model

    pipe = Pipe()
    mg._CFG_ROUTERS[pipe] = router
    selection = mg.MultiGpuSelection(primary = 0, secondaries = (1,))
    mg._record_decision(
        pipe,
        selection,
        "cfg_replica",
        "one card plus a CFG replica on cuda:1",
        (mg.PlacementCandidate("layer_split", "predicted slower"),),
        fallback_resident = False,
    )
    out = _guided(model)
    torch.testing.assert_close(out, ref, rtol = 0, atol = 0)
    assert router.disabled_reason is not None and "OutOfMemoryError" in router.disabled_reason
    assert router.stats["replica_failures"] == 1
    decision = mg.placement_status(pipe)["decision"]
    assert decision["chosen"] == "offload"
    assert decision["rejected"][0]["kind"] == "cfg_replica"
    assert "fell back to one card" in decision["rejected"][0]["reason"]
    assert "OutOfMemoryError" in decision["rejected"][0]["reason"]


@pytest.mark.parametrize(
    "cls, env, value",
    [
        ("LTX2VideoTransformer3DModel", "UNSLOTH_DIFFUSION_BATCH_SPLIT", "0"),
        ("ZImageTransformer2DModel", "UNSLOTH_DIFFUSION_ZIMAGE_CFG_SPLIT", "0"),
    ],
)
def test_a_family_whose_guidance_split_is_off_refuses_the_replica(monkeypatch, cls, env, value):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MULTIGPU_ALL", "1")
    denoiser = type(cls, (torch.nn.Module,), {})()
    assert mg.replica_layout_refusal(denoiser) is None
    monkeypatch.setenv(env, value)
    assert env in (mg.replica_layout_refusal(denoiser) or "")
