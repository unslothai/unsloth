# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The CFG replica layout's pure parts, CPU only: the mode price, the per-card fit of a replica, the default-request
shape a backend notes, and the clean fallback when a replica runs out of memory mid-render."""

import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_multigpu as mg  # noqa: E402


# (guided, compute s, primary stream s, steps, encoder stream s, replica wins)
@pytest.mark.parametrize(
    "guided, compute, stream, steps, encoders, wins",
    [
        # Kaggle 2x T4, Wan 2.2 TI2V-5B 832x480x49: 2.65 s forward, about 1.06 s streamed on the group-offloaded
        # primary. Measured 3.71 s/step replica vs 5.30 s/step layer split; the price predicts the same pair.
        (True, 2.65, 1.06, 10, 0.0, True),
        # nothing streamed (the denoiser resident on the primary): one forward per step against two
        (True, 1.0, 0.0, 20, 0.0, True),
        # the primary streams more per step than a forward computes: two resident forwards are faster
        (True, 0.3, 4.7, 10, 0.0, False),
        # within the margin: kept on the split
        (True, 1.0, 0.97, 10, 0.0, False),
        # unguided: a copy idles for one image, the split is never slower
        (False, 2.65, 0.0, 10, 0.0, False),
        (False, 2.65, 1.06, 10, 0.0, False),
        # a companion split keeps the text encoder resident; streaming it once per prompt is cheap next to steps
        (True, 1.5, 0.0, 50, 1.0, True),
        # but not on a one-step render
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
    # a 600 MiB block (Qwen-Image class): twice the block, the part over the reserve added
    monkeypatch.setattr(mg, "_module_mib", lambda m: 600)
    assert mg.replica_working_mib(small) == 1200
    assert mg.replica_need_mib(small) == weights + 1200 - 1024


def test_video_default_tokens_follow_the_vae_and_patch():
    wan = types.SimpleNamespace(
        vae_scale_factor_spatial = 16,
        vae_scale_factor_temporal = 4,
        transformer = types.SimpleNamespace(config = types.SimpleNamespace(patch_size = (1, 2, 2))),
    )
    # Wan 2.2 TI2V-5B 832x480x49: the (1, 5070, 24, 128) attention shape of the Kaggle record
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
    # every branch the replica could not run was recomputed on the source: the render is the one-card render
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
    # both branches ride one call: with its splitter off the replica would sit idle
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MULTIGPU_ALL", "1")
    denoiser = type(cls, (torch.nn.Module,), {})()
    assert mg.replica_layout_refusal(denoiser) is None
    monkeypatch.setenv(env, value)
    assert env in (mg.replica_layout_refusal(denoiser) or "")
