# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Auto static step skip: the per-model policy, its kill switch, joint (video, audio) outputs, and the image loader."""

from __future__ import annotations

import contextlib
import types

import pytest
import torch

from core.inference import diffusion_cache as dcache
from core.inference import diffusion_step_skip as ss
from core.inference.diffusion_families import default_generation_params

from .test_diffusion_backend import _load_into, fake_runtime  # noqa: F401 - fixture


# --------------------------------------------------------------------------------------------- schedule / policy
def test_min_steps_raises_the_floor_but_never_lowers_it():
    assert ss.static_schedule(14, min_steps = 20) == ()
    assert ss.static_schedule(20, min_steps = 20) == ss.static_schedule(20)
    # Below STATIC_MIN_STEPS stays off whatever is asked.
    assert ss.static_schedule(11, min_steps = 4) == ()
    assert ss.static_schedule(14, min_steps = 4) == ss.static_schedule(14)


@pytest.fixture
def table(monkeypatch):
    monkeypatch.setattr(
        dcache,
        "AUTO_STATIC_SKIP",
        {
            "black-forest-labs/flux.1-krea-dev": {"default": 2, "max": 3},
            "black-forest-labs/flux.1-dev": {"max": 2},
        },
    )
    monkeypatch.delenv(dcache.ENV_AUTO_STEP_SKIP, raising = False)


KREA = "black-forest-labs/FLUX.1-Krea-dev"
DEV = "black-forest-labs/FLUX.1-dev"


@pytest.mark.parametrize(
    "ids, tier, steps, want",
    [
        ((KREA,), "default", 28, {"every": 2, "min_steps": 20}),
        ((KREA,), "max", 28, {"every": 3, "min_steps": 20}),
        ((KREA,), "eager", 28, None),
        ((KREA,), "off", 28, None),
        ((KREA,), None, 28, None),
        ((KREA,), "default", 12, None),  # a short default schedule is never measured
        ((DEV,), "default", 28, None),  # max-only row
        ((DEV,), "max", 28, {"every": 2, "min_steps": 20}),
        # The unsloth mirror and a local path with the upstream as its base both resolve to the upstream row.
        (("unsloth/FLUX.1-Krea-dev",), "default", 28, {"every": 2, "min_steps": 20}),
        (("/models/my-krea", KREA), "default", 28, {"every": 2, "min_steps": 20}),
        (("black-forest-labs/FLUX.1-schnell",), "max", 28, None),
        ((None,), "max", 28, None),
        (KREA, "default", 28, {"every": 2, "min_steps": 20}),  # a bare string
    ],
)
def test_plan_follows_the_tier_and_the_default_steps(table, ids, tier, steps, want):
    assert dcache.auto_static_skip_plan(ids, tier, steps) == want


@pytest.mark.parametrize("value", ["0", "false", "off", "NO"])
def test_kill_switch_turns_auto_static_off(table, monkeypatch, value):
    monkeypatch.setenv(dcache.ENV_AUTO_STEP_SKIP, value)
    assert dcache.auto_static_skip_plan((KREA,), "default", 28) is None
    # Auto falls back to what it did before: FBCache on max at 20+ steps, nothing on default.
    assert dcache.resolve_auto_step_cache("max", 28, static_plan = None) == dcache.TC_FBCACHE
    assert dcache.resolve_auto_step_cache("default", 28, static_plan = None) is None


def test_static_plan_wins_over_fbcache_on_max(table):
    plan = dcache.auto_static_skip_plan((DEV,), "max", 28)
    assert dcache.resolve_auto_step_cache("max", 28, static_plan = plan) == dcache.TC_STATIC


def test_auto_settings_use_the_plan_unless_the_env_pins_every():
    plan = {"every": 3, "min_steps": 20}
    knobs = ss.auto_static_settings(plan, env = {})
    assert (knobs["every"], knobs["min_steps"], knobs["auto"]) == (3, 20, True)
    assert knobs["mode"] == ss.DEFAULT_MODE
    assert ss.auto_static_settings(plan, env = {ss.ENV_EVERY: "2"})["every"] == 2


# The shipped table: only the checkpoints that were measured, each at its default steps.
MEASURED = {
    "Qwen/Qwen-Image-2.1": ("default", "max"),
    "Qwen/Qwen-Image": ("default", "max"),
    "black-forest-labs/FLUX.1-Krea-dev": ("default", "max"),
    "black-forest-labs/FLUX.2-klein-base-4B": ("default", "max"),
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers": ("default", "max"),
    "black-forest-labs/FLUX.1-dev": ("max",),
    "hunyuanvideo-community/HunyuanImage-2.1-Diffusers": ("max",),
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v": ("max",),
    "MiniMaxAI/MiniMax-H3": ("max",),
}


def test_shipped_table_lists_only_measured_models():
    assert set(dcache.AUTO_STATIC_SKIP) == {k.lower() for k in MEASURED}
    for repo, tiers in MEASURED.items():
        entry = dcache.AUTO_STATIC_SKIP[repo.lower()]
        assert set(entry) == set(tiers)
        assert all(int(v) >= 2 for v in entry.values())
        # The default tier never skips more than the max tier.
        assert entry.get("default", 99) <= entry["max"] or "default" not in entry


@pytest.mark.parametrize("repo", sorted(MEASURED))
def test_every_measured_model_reaches_the_auto_floor_at_its_default_steps(repo):
    from core.inference.video_families import default_video_generation_params

    steps, _ = default_generation_params(repo)
    if repo.startswith(("Wan-AI", "hunyuanvideo-community/HunyuanVideo")):
        steps, _ = default_video_generation_params(repo)
    if repo == "MiniMaxAI/MiniMax-H3":
        steps = 30  # the family default; no generation-defaults row
    assert steps >= dcache.AUTO_STATIC_MIN_STEPS
    assert dcache.auto_static_skip_plan((repo,), "max", steps) is not None


@pytest.mark.parametrize(
    "repo",
    [
        "black-forest-labs/FLUX.1-schnell",
        "black-forest-labs/FLUX.2-klein-4B",
        "black-forest-labs/FLUX.2-klein-base-9B",
        "Tongyi-MAI/Z-Image-Turbo",
        "Tongyi-MAI/Z-Image",
        "Qwen/Qwen-Image-2512",
        "Qwen/Qwen-Image-Edit-2511",
        "unsloth/Krea-2-Turbo",
    ],
)
def test_unmeasured_or_distilled_siblings_never_auto_skip(repo):
    steps, _ = default_generation_params(repo)
    for tier in ("default", "max"):
        assert dcache.auto_static_skip_plan((repo,), tier, max(steps, 50)) is None


# --------------------------------------------------------------------------------------------- joint outputs (H3)
class _JointDiT:
    """MiniMax-H3's shape: one call per step returning (video velocity, audio velocity) of different sizes."""

    def __init__(self):
        self.calls = 0

    def forward(
        self,
        hidden_states = None,
        audio_hidden_states = None,
        timestep = None,
        return_dict = True,
    ):
        self.calls += 1
        t = float(timestep.reshape(-1).max())
        return torch.full((1, 6, 2), t), torch.full((1, 3), 2 * t)


def _run_joint(pipe, steps):
    outs = []
    for i in range(steps):
        t = torch.tensor([0.0, 1.0 - i / steps])  # conditioning rows at 0, noisy rows at t
        outs.append(pipe.transformer.forward(hidden_states = None, timestep = t, return_dict = False))
    return outs


@pytest.mark.parametrize("mode", ["reuse", "taylor1"])
def test_joint_video_audio_outputs_are_skipped_stream_by_stream(mode):
    dit = _JointDiT()
    pipe = types.SimpleNamespace(transformer = dit)
    knobs = {**ss.static_skip_settings({}), "mode": mode}
    assert ss.install_static_step_skip(pipe, settings = knobs) == dcache.TC_STATIC
    ss.reset_static_step_skip(pipe, 30)
    outs = _run_joint(pipe, 30)
    plan = ss.static_schedule(30)
    assert dit.calls == sum(plan)
    assert ss.static_skip_stats(pipe)["stats"]["skipped"] == 30 - sum(plan)
    for i, (video, audio) in enumerate(outs):
        assert type(outs[i]) is tuple and video.shape == (1, 6, 2) and audio.shape == (1, 3)
        t = 1.0 - i / 30
        if plan[i] or mode == "taylor1":
            # Computed, or extrapolated linearly in t, which is exact for these linear-in-t outputs.
            assert torch.allclose(video, torch.full_like(video, t), atol = 1e-5)
            assert torch.allclose(audio, torch.full_like(audio, 2 * t), atol = 1e-5)
        else:
            prev = max(j for j in range(i) if plan[j])
            assert torch.allclose(video, torch.full_like(video, 1.0 - prev / 30))


def test_a_non_tensor_member_still_declines():
    class _KV(_JointDiT):
        def forward(self, **kw):
            self.calls += 1
            return torch.zeros(1, 2), {"kv": 1}

    dit = _KV()
    pipe = types.SimpleNamespace(transformer = dit)
    ss.install_static_step_skip(pipe, settings = ss.static_skip_settings({}))
    ss.reset_static_step_skip(pipe, 25)
    for i in range(25):
        dit.forward(timestep = torch.tensor([1.0 - i / 25]))
    assert dit.calls == 25


# --------------------------------------------------------------------------------------------- image loader
def _probe_plan(entry):
    """The real plan function over a one-row table, whatever ids the loader passes (a tmp-path GGUF here)."""

    def plan(
        ids,
        tier,
        steps,
        env = None,
    ):
        saved = dict(dcache.AUTO_STATIC_SKIP)
        dcache.AUTO_STATIC_SKIP.clear()
        dcache.AUTO_STATIC_SKIP["probe/model"] = entry
        try:
            return dcache.auto_static_skip_plan(("probe/model",), tier, steps, env)
        finally:
            dcache.AUTO_STATIC_SKIP.clear()
            dcache.AUTO_STATIC_SKIP.update(saved)

    return plan


@pytest.mark.parametrize(
    "speed, killed, want_static",
    [
        ("default", False, True),
        ("max", False, True),
        ("eager", False, False),
        ("default", True, False),
    ],
)
def test_auto_load_installs_static_for_a_listed_model(
    fake_runtime, tmp_path, monkeypatch, speed, killed, want_static
):
    from core.inference import diffusion as dmod

    monkeypatch.setattr(dmod, "auto_static_skip_plan", _probe_plan({"default": 3, "max": 3}))
    monkeypatch.setattr(dmod, "default_generation_params", lambda *a, **k: (28, 3.5))
    if killed:
        monkeypatch.setenv(dcache.ENV_AUTO_STEP_SKIP, "0")
    else:
        monkeypatch.delenv(dcache.ENV_AUTO_STEP_SKIP, raising = False)
    installs, fb = [], []
    monkeypatch.setattr(
        dmod,
        "install_static_step_skip",
        lambda pipe, settings = None, logger = None: installs.append(settings) or dcache.TC_STATIC,
    )
    monkeypatch.setattr(
        dmod, "apply_step_cache", lambda pipe, mode = None, **k: fb.append(mode) or None
    )
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = dmod.DiffusionBackend()
    status = _load_into(backend, tmp_path, speed_mode = speed, transformer_cache = None)
    entry = status["resolved"]["transformer_cache"]
    if want_static:
        assert len(installs) == 1 and installs[0]["every"] == 3 and installs[0]["min_steps"] == 20
        assert installs[0]["auto"] is True
        assert status["transformer_cache"] == "static" and entry["value"] == "static"
        assert entry["requested"] is None and "UNSLOTH_DIFFUSION_AUTO_STEP_SKIP" in entry["reason"]
        assert fb == []
        # The generate-time FBCache toggle is never armed over an engaged static skip.
        assert backend._state.cache_auto is False
    else:
        assert installs == []
        assert status["transformer_cache"] in (None, dcache.TC_FBCACHE)
    backend.unload()


def test_explicit_requests_ignore_the_table(fake_runtime, tmp_path, monkeypatch):
    from core.inference import diffusion as dmod

    monkeypatch.setattr(dmod, "auto_static_skip_plan", _probe_plan({"default": 3, "max": 3}))
    monkeypatch.setattr(
        dmod,
        "install_static_step_skip",
        lambda *a, **k: pytest.fail("explicit off installed static"),
    )
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = dmod.DiffusionBackend()
    status = _load_into(backend, tmp_path, speed_mode = "default", transformer_cache = "off")
    assert status["transformer_cache"] is None
    backend.unload()


def test_auto_static_declined_falls_back_to_the_previous_auto(fake_runtime, tmp_path, monkeypatch):
    from core.inference import diffusion as dmod

    monkeypatch.setattr(dmod, "auto_static_skip_plan", _probe_plan({"default": 2, "max": 2}))
    monkeypatch.setattr(dmod, "default_generation_params", lambda *a, **k: (28, 3.5))
    monkeypatch.delenv(dcache.ENV_AUTO_STEP_SKIP, raising = False)
    monkeypatch.setattr(dmod, "install_static_step_skip", lambda *a, **k: None)
    modes = []
    monkeypatch.setattr(
        dmod, "apply_step_cache", lambda pipe, mode = None, **k: modes.append(mode) or None
    )
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = dmod.DiffusionBackend()
    _load_into(backend, tmp_path, speed_mode = "max", transformer_cache = None)
    assert modes == [dcache.TC_FBCACHE]
    backend.unload()


# --------------------------------------------------------------------------------------------- generate scope
@pytest.mark.parametrize("auto", [True, False])
def test_auto_skip_runs_on_txt2img_only_explicit_everywhere(
    fake_runtime, tmp_path, monkeypatch, auto
):
    from core.inference import diffusion as dmod

    from .test_diffusion_backend import _loaded_backend, _tiny_png_b64

    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = _loaded_backend(tmp_path)
    pipe = backend._state.pipe
    pipe.transformer = types.SimpleNamespace(forward = lambda *a, **k: None, __dict__ = {})
    layer = types.SimpleNamespace(auto = auto)
    monkeypatch.setattr(dmod, "static_skip_is_auto", lambda p: layer.auto)
    armed = []
    monkeypatch.setattr(
        dmod, "reset_static_step_skip", lambda p, steps, **k: armed.append(steps) or True
    )
    monkeypatch.setattr(dmod, "effective_request_strength", lambda *a, **k: None)
    object.__setattr__(backend._state, "transformer_cache", dcache.TC_STATIC)
    backend.generate(prompt = "a car", steps = 28, seed = 1)
    assert armed[0] == 28
    armed.clear()
    backend.generate(prompt = "a car", steps = 28, seed = 1, init_image = _tiny_png_b64(), strength = 1.0)
    # img2img: an AUTO layer computes every step; an explicit one keeps its schedule.
    assert armed[0] == (None if auto else 28)
    backend.unload()


def test_deferred_default_tier_installs_the_auto_skip_on_the_third_image(
    fake_runtime, tmp_path, monkeypatch
):
    # Speed unset (the UI default): a dense load stays eager, then the 3rd image engages `default`, and with it the
    # model's default-tier skip. The first two renders stay full-step.
    from core.inference import diffusion as dmod

    from .test_diffusion_backend import DiffusionBackend

    monkeypatch.setattr(dmod, "auto_static_skip_plan", _probe_plan({"default": 2, "max": 3}))
    monkeypatch.setattr(dmod, "default_generation_params", lambda *a, **k: (28, 3.5))
    monkeypatch.delenv(dcache.ENV_AUTO_STEP_SKIP, raising = False)
    monkeypatch.setattr(dmod, "compile_eligible", lambda *a, **k: True)
    monkeypatch.setattr(
        dmod,
        "apply_speed_optims",
        lambda pipe, target, **k: {"compiled": k.get("speed_mode") == "default"},
    )
    monkeypatch.setattr(dmod.compile_cache, "begin", lambda **k: None)
    installs = []
    monkeypatch.setattr(
        dmod,
        "install_static_step_skip",
        lambda pipe, settings = None, logger = None: installs.append(settings) or dcache.TC_STATIC,
    )
    monkeypatch.setattr(dmod, "reset_static_step_skip", lambda *a, **k: True)
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    backend = DiffusionBackend()
    status = _load_into(
        backend, tmp_path, gguf_filename = "model.safetensors", family_override = "qwen-image"
    )
    assert status["speed_mode"] == "off" and status["transformer_cache"] is None
    backend.generate(prompt = "one")
    backend.generate(prompt = "two")
    assert installs == [] and backend.status()["transformer_cache"] is None
    backend.generate(prompt = "three")
    st = backend.status()
    assert st["speed_mode"] == "default"
    assert len(installs) == 1 and installs[0]["every"] == 2 and installs[0]["auto"] is True
    assert st["transformer_cache"] == "static"
    assert st["resolved"]["transformer_cache"]["value"] == "static"
    backend.unload()
