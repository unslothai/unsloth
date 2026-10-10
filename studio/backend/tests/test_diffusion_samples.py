# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sample images during diffusion LoRA training: config validation, RNG isolation (a round must
not move the training streams or the training scheduler), the flow sampler's sign convention,
the run state / record, and the route that serves the images (listed paths only, owner only)."""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.training import diffusion_samples as ds
from core.training.diffusion_train_common import DiffusionLoraConfig, _config_from_dict

_ZIMAGE = "unsloth/Z-Image-Turbo-unsloth-bnb-4bit"


def _cfg(**kw):
    base = dict(base_model = _ZIMAGE, data_dir = "d", output_dir = "o")
    base.update(kw)
    return DiffusionLoraConfig(**base)


# ── config ───────────────────────────────────────────────────────────────────
def test_sampling_is_off_by_default():
    cfg = _cfg().normalized()
    assert cfg.sample_every == 0 and cfg.sample_prompts == ()
    assert ds.plan_samples(cfg, cfg.resolved_family, ["a dog"]) is None


def test_sample_settings_normalised():
    cfg = _config_from_dict(
        {
            "base_model": _ZIMAGE,
            "data_dir": "d",
            "output_dir": "o",
            "sample_every": "50",
            "sample_prompts": [" a dog ", "", "  "],
        }
    ).normalized()
    assert cfg.sample_every == 50
    assert cfg.sample_prompts == ("a dog",)


@pytest.mark.parametrize(
    "kw, msg",
    [
        ({"sample_every": -1}, ">= 0"),
        ({"sample_every": "x"}, "whole number"),
        ({"sample_every": 10, "sample_prompts": ["p"] * 5}, "at most 4"),
        ({"sample_every": 10, "sample_prompts": ["x" * 1001]}, "1000 characters"),
    ],
)
def test_sample_settings_rejected(kw, msg):
    with pytest.raises(ValueError, match = msg):
        _cfg(**kw).normalized()


@pytest.mark.parametrize("family", ["ltx-2", "minimax-h3"])
def test_video_families_refuse_sampling(family):
    with pytest.raises(ValueError, match = "not supported"):
        ds.validate_sample_settings(10, (), family)
    # 0 stays valid, so a replayed config from an image family never breaks a video start.
    assert ds.validate_sample_settings(0, (), family) == (0, ())


def test_request_model_carries_sample_fields():
    from models.training import DiffusionTrainingStartRequest

    body = DiffusionTrainingStartRequest(
        base_model = _ZIMAGE,
        data_dir = "d",
        output_dir = "o",
        sample_every = 25,
        sample_prompts = ["a"],
    ).model_dump()
    assert body["sample_every"] == 25 and body["sample_prompts"] == ["a"]
    with pytest.raises(Exception):
        DiffusionTrainingStartRequest(
            base_model = _ZIMAGE, data_dir = "d", output_dir = "o", sample_prompts = ["a"] * 5
        )


def test_family_info_reports_supports_samples():
    from core.training.diffusion_train_common import SAMPLE_FAMILIES
    assert "sdxl" in SAMPLE_FAMILIES and "z-image" in SAMPLE_FAMILIES
    assert not {"ltx-2", "minimax-h3"} & SAMPLE_FAMILIES


def test_sampling_is_not_part_of_resume_identity():
    from core.training.diffusion_checkpoint import identity_for_config

    a = identity_for_config(_cfg().normalized())
    b = identity_for_config(_cfg(sample_every = 10, sample_prompts = ("x",)).normalized())
    assert a == b


# ── plan ─────────────────────────────────────────────────────────────────────
def test_plan_defaults_to_instance_prompt_then_first_caption(tmp_path):
    cfg = _cfg(output_dir = str(tmp_path), sample_every = 5, instance_prompt = "sks dog").normalized()
    plan = ds.plan_samples(cfg, "z-image", ["cap one", "cap two"])
    assert plan.prompts == ("sks dog",)
    cfg = _cfg(output_dir = str(tmp_path), sample_every = 5).normalized()
    assert ds.plan_samples(cfg, "z-image", ["", "cap one"]).prompts == ("cap one",)


def test_plan_schedule_and_paths(tmp_path):
    cfg = _cfg(output_dir = str(tmp_path), sample_every = 50, resolution = 1024).normalized()
    plan = ds.plan_samples(cfg, "z-image", ["c"])
    assert [s for s in range(0, 131) if plan.due(s, 130)] == [50, 100, 130]
    assert plan.resolution == ds.MAX_SAMPLE_RESOLUTION
    rel = plan.relpath(plan.path_for(100, 0))
    assert ds.SAMPLE_PATH_RE.match(rel), rel
    # Turbo is distilled: no CFG, so no empty-prompt embedding is needed.
    assert plan.steps == 8 and not plan.uses_cfg and plan.encode_texts == ["c"]
    sdxl = ds.plan_samples(
        _cfg(
            base_model = "stabilityai/stable-diffusion-xl-base-1.0",
            output_dir = str(tmp_path),
            sample_every = 1,
        ).normalized(),
        "sdxl",
        ["c"],
    )
    assert sdxl.uses_cfg and sdxl.encode_texts == ["c", ""]


@pytest.mark.parametrize(
    "bad",
    [
        "samples/../x.png",
        "/etc/passwd",
        "samples/20261010-014028-0993fe/step-1-0.jpg",
        "x/step-1-0.png",
    ],
)
def test_sample_path_regex_rejects(bad):
    assert not ds.SAMPLE_PATH_RE.match(bad)


# ── RNG isolation ────────────────────────────────────────────────────────────
def test_isolated_sampling_restores_global_rng(tmp_path):
    plan = ds.plan_samples(
        _cfg(output_dir = str(tmp_path), sample_every = 1).normalized(), "z-image", ["c"]
    )
    torch.manual_seed(1234)
    expected = torch.rand(8)
    torch.manual_seed(1234)
    py_state = random.getstate()
    with ds.isolated_sampling("cpu", compiled = False):
        torch.randn(100)  # anything the sampler (or a stochastic module) draws
        noise = ds.initial_noise(plan, 0, (1, 4, 8, 8), "cpu")
        assert not torch.is_grad_enabled()
    assert torch.equal(torch.rand(8), expected)
    assert random.getstate() == py_state
    # Fixed seed: the same noise every round.
    assert torch.equal(noise, ds.initial_noise(plan, 0, (1, 4, 8, 8), "cpu"))
    assert not torch.equal(noise, ds.initial_noise(plan, 1, (1, 4, 8, 8), "cpu"))


def test_initial_noise_never_touches_global_stream(tmp_path):
    plan = ds.plan_samples(
        _cfg(output_dir = str(tmp_path), sample_every = 1).normalized(), "z-image", ["c"]
    )
    torch.manual_seed(7)
    before = torch.get_rng_state()
    ds.initial_noise(plan, 0, (2, 3), "cpu")
    assert torch.equal(torch.get_rng_state(), before)


def test_flow_sigmas_leave_training_scheduler_untouched():
    from diffusers import FlowMatchEulerDiscreteScheduler

    sched = FlowMatchEulerDiscreteScheduler(shift = 3.0)
    sig, ts = sched.sigmas.clone(), sched.timesteps.clone()
    sigmas, num_train = ds.flow_sigmas(sched.config, "z-image", 8, 1024)
    assert torch.equal(sched.sigmas, sig) and torch.equal(sched.timesteps, ts)
    assert len(sigmas) == 9 and float(sigmas[-1]) == 0.0 and float(sigmas[0]) == pytest.approx(1.0)
    assert num_train == 1000.0
    dyn = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting = True)
    assert len(ds.flow_sigmas(dyn.config, "qwen-image", 4, 4096)[0]) == 5


def test_euler_flow_sample_recovers_target_with_exact_velocity(tmp_path):
    """The training target is v = noise - latents; an oracle velocity must land exactly on x0,
    with and without CFG (identical cond/uncond predictions make CFG a no-op)."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    x0 = torch.randn(1, 4, 8, 8, generator = torch.Generator().manual_seed(0))
    for guidance in (1.0, 4.0):
        plan = ds.plan_samples(
            _cfg(output_dir = str(tmp_path), sample_every = 1).normalized(), "z-image", ["c"]
        )
        plan.guidance = guidance
        noise = ds.initial_noise(plan, 0, x0.shape, "cpu")

        def velocity(x, t, sig, cond):
            return noise - x0

        out = ds.euler_flow_sample(
            plan = plan,
            index = 0,
            latent_shape = x0.shape,
            image_seq_len = 16,
            family = "z-image",
            scheduler_config = FlowMatchEulerDiscreteScheduler(shift = 3.0).config,
            velocity = velocity,
            cond = "c",
            uncond = "u",
            device = "cpu",
            weight_dtype = torch.float32,
        )
        assert torch.allclose(out, x0, atol = 1e-5)


# ── files ────────────────────────────────────────────────────────────────────
def test_round_saves_pngs_and_emits_event(tmp_path):
    plan = ds.plan_samples(
        _cfg(output_dir = str(tmp_path), sample_every = 1, sample_prompts = ("a", "b")).normalized(),
        "z-image",
        ["c"],
    )
    events = []
    ds.run_sample_round(
        plan,
        10,
        lambda i, p: torch.zeros(1, 3, 16, 16),
        None,
        lambda _cb, t, **kw: events.append({"type": t, **kw}),
    )
    (ev,) = events
    assert ev["type"] == "sample" and ev["step"] == 10
    assert [e["prompt"] for e in ev["images"]] == ["a", "b"]
    for e in ev["images"]:
        assert ds.SAMPLE_PATH_RE.match(e["path"]) and (tmp_path / e["path"]).is_file()
    ds.discard_samples(plan)
    assert not (tmp_path / "samples").exists()


def test_parent_discard_removes_the_whole_run_folder_only(tmp_path):
    other = tmp_path / "samples" / "20260101-000000-aaaaaa" / "step-1-0.png"
    mine = tmp_path / "samples" / "20260102-000000-bbbbbb" / "step-1-0.png"
    thinned = mine.parent / "step-2-0.png"  # on disk but no longer listed
    for p in (other, mine, thinned):
        p.parent.mkdir(parents = True, exist_ok = True)
        p.write_bytes(b"x")
    ds.discard_sample_paths(str(tmp_path), [mine.relative_to(tmp_path).as_posix(), "../evil.png"])
    assert other.is_file() and not mine.parent.exists()


def test_thinning_deletes_the_evicted_files(tmp_path):
    from core.training.diffusion_training_service import _SAMPLES_CAP, _append_samples

    state = {"samples": []}
    tag = "samples/20261010-014028-0993fe"
    (tmp_path / tag).mkdir(parents = True)
    for step in range(_SAMPLES_CAP + 1):
        rel = f"{tag}/step-{step}-0.png"
        (tmp_path / rel).write_bytes(b"x")
        _append_samples(state, step, [{"path": rel}], str(tmp_path))
    listed = {e["path"] for e in state["samples"]}
    on_disk = {p.relative_to(tmp_path).as_posix() for p in (tmp_path / tag).iterdir()}
    assert len(listed) < _SAMPLES_CAP + 1 and on_disk == listed


# ── service state + record ───────────────────────────────────────────────────
def test_service_folds_sample_events_and_persists(monkeypatch, tmp_path):
    import core.training.diffusion_training_service as dts

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(dts, "_runs_dir", lambda: runs)
    svc = dts.DiffusionTrainingService()
    svc._state.update(job_id = "a" * 32, status = "running", output_dir = str(tmp_path))
    good = "samples/20261010-014028-0993fe/step-50-0.png"
    svc._apply_event(
        {
            "type": "sample",
            "step": 50,
            "images": [{"path": good, "prompt": "p", "seed": 3}, {"path": "../x.png"}],
        }
    )
    assert svc.status()["samples"] == [{"step": 50, "path": good, "prompt": "p", "seed": 3}]
    svc._apply_event({"type": "complete", "output_dir": str(tmp_path), "lora_path": "x"})
    svc._persist_run_record()
    rec = json.loads((runs / f"{'a' * 32}.json").read_text())
    assert rec["samples"][0]["path"] == good
    assert dts.get_diffusion_run("a" * 32)["samples"][0]["path"] == good
    assert "samples" not in dts.list_diffusion_runs()[0]


def test_sample_list_is_bounded():
    from core.training.diffusion_training_service import _SAMPLES_CAP, _append_samples

    state = {"samples": []}
    for step in range(0, 2000, 1):
        _append_samples(
            state, step, [{"path": f"samples/20261010-014028-0993fe/step-{step}-0.png"}]
        )
    steps = [e["step"] for e in state["samples"]]
    assert len(steps) <= _SAMPLES_CAP and steps[0] == 0 and steps[-1] == 1999


# ── route ────────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_client(monkeypatch, tmp_path):
    import core.training.diffusion_training_service as dts
    import utils.paths as up
    from auth.authentication import get_current_subject
    from routes.training import router

    runs = tmp_path / "runs"
    runs.mkdir()
    out_root = tmp_path / "outputs"
    run_dir = out_root / "my-lora"
    rel = "samples/20261010-014028-0993fe/step-50-0.png"
    (run_dir / rel).parent.mkdir(parents = True)
    (run_dir / rel).write_bytes(b"\x89PNG fake")
    (run_dir / "samples/20261010-014028-0993fe/step-60-0.png").write_bytes(b"unlisted")
    rec = {
        "job_id": "b" * 32,
        "status": "completed",
        "output_dir": str(run_dir),
        "samples": [{"step": 50, "path": rel, "prompt": "p"}],
    }
    (runs / f"{'b' * 32}.json").write_text(json.dumps(rec))
    monkeypatch.setattr(dts, "_runs_dir", lambda: runs)
    monkeypatch.setattr(up, "outputs_root", lambda: out_root)
    svc = dts.DiffusionTrainingService()
    monkeypatch.setattr(dts, "get_diffusion_training_service", lambda: svc)
    app = FastAPI()
    app.include_router(router, prefix = "/api/train")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return SimpleNamespace(
        client = TestClient(app), svc = svc, rel = rel, run_dir = run_dir, out_root = out_root
    )


def _url(job, path):
    return f"/api/train/diffusion/runs/{job}/sample", {"path": path}


def test_route_serves_listed_sample(sample_client):
    url, q = _url("b" * 32, sample_client.rel)
    r = sample_client.client.get(url, params = q)
    assert r.status_code == 200 and r.content == b"\x89PNG fake"
    assert r.headers["content-type"] == "image/png"


@pytest.mark.parametrize(
    "job, path",
    [
        ("b" * 32, "samples/20261010-014028-0993fe/step-60-0.png"),  # on disk, never reported
        ("b" * 32, "samples/../../../etc/passwd"),
        ("c" * 32, "samples/20261010-014028-0993fe/step-50-0.png"),  # no such run
        ("../" * 3, "samples/20261010-014028-0993fe/step-50-0.png"),
    ],
)
def test_route_refuses_unlisted_or_foreign(sample_client, job, path):
    url, q = _url(job, path)
    assert sample_client.client.get(url, params = q).status_code == 404


def test_route_refuses_record_outside_outputs_root(sample_client, tmp_path):
    rec_path = next((tmp_path / "runs").glob("*.json"))
    rec = json.loads(rec_path.read_text())
    outside = tmp_path / "elsewhere"
    (outside / sample_client.rel).parent.mkdir(parents = True)
    (outside / sample_client.rel).write_bytes(b"x")
    rec["output_dir"] = str(outside)
    rec_path.write_text(json.dumps(rec))
    url, q = _url("b" * 32, sample_client.rel)
    assert sample_client.client.get(url, params = q).status_code == 404


def test_route_serves_live_job_and_hides_it_from_other_accounts(sample_client, monkeypatch):
    import core.training.account_jobs as aj

    svc = sample_client.svc
    live = "d" * 32
    svc._state.update(job_id = live, samples = [{"step": 50, "path": sample_client.rel}])
    svc._config = {"output_dir": str(sample_client.run_dir)}
    url, q = _url(live, sample_client.rel)
    assert sample_client.client.get(url, params = q).status_code == 200
    monkeypatch.setattr(aj, "job_is_foreign", lambda service: True)
    assert sample_client.client.get(url, params = q).status_code == 404


# ── per-family CFG + schedules (diffusers 0.41 pipelines) ──────────────────────
def test_cfg_combination_matches_each_pipeline():
    c = torch.randn(1, 16, 1, 8, 8, generator = torch.Generator().manual_seed(1))
    u = torch.randn(1, 16, 1, 8, 8, generator = torch.Generator().manual_seed(2))
    g = 4.0
    # pipeline_stable_diffusion_xl.py / pipeline_flux2_klein.py: u + g(c - u)
    assert torch.allclose(ds.combine_cfg("uncond", g, c, u), u + g * (c - u))
    # pipeline_z_image.py:547, pipeline_krea2.py:666: c + g(c - u)
    assert torch.allclose(ds.combine_cfg("cond", g, c, u), c + g * (c - u))
    # pipeline_qwenimage.py:668-672 on the PACKED sequence, rescaled to the conditional norm per token.
    from diffusers import QwenImagePipeline

    pack = lambda t: QwenImagePipeline._pack_latents(t, 1, 16, 8, 8)  # noqa: E731
    comb = pack(u) + g * (pack(c) - pack(u))
    ref = comb * (
        torch.norm(pack(c), dim = -1, keepdim = True) / torch.norm(comb, dim = -1, keepdim = True)
    )
    assert torch.allclose(pack(ds.combine_cfg("qwen", g, c, u)), ref, atol = 1e-5)


@pytest.mark.parametrize(
    "family, base, steps, guidance, uses_cfg, mode, mu",
    [
        ("z-image", "unsloth/Z-Image-Turbo-unsloth-bnb-4bit", 8, 0.0, False, "cond", None),
        ("z-image", "Tongyi-MAI/Z-Image", 28, 4.0, True, "cond", None),
        ("krea-2", "krea/Krea-2-Turbo", 8, 0.0, False, "cond", 1.15),
        ("krea-2", "krea/Krea-2-Raw", 20, 4.5, True, "cond", None),
        ("qwen-image", "Qwen/Qwen-Image", 20, 4.0, True, "qwen", None),
        ("flux.2-klein", "black-forest-labs/FLUX.2-klein-base-4B", 28, 4.0, True, "uncond", None),
        ("sdxl", "stabilityai/stable-diffusion-xl-base-1.0", 20, 5.0, True, "uncond", None),
        ("sdxl", "stabilityai/sdxl-turbo", 4, 0.0, False, "uncond", None),
    ],
)
def test_family_sample_settings(family, base, steps, guidance, uses_cfg, mode, mu):
    st = ds.sample_inference_settings(family, base)
    assert (st.steps, st.guidance, st.uses_cfg, st.cfg_mode, st.mu) == (
        steps,
        guidance,
        uses_cfg,
        mode,
        mu,
    )


def test_fixed_mu_reaches_the_schedule():
    from diffusers import FlowMatchEulerDiscreteScheduler

    sched = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting = True)
    ref = FlowMatchEulerDiscreteScheduler.from_config(sched.config)
    import numpy as np

    ref.set_timesteps(sigmas = np.linspace(1.0, 1 / 8, 8).tolist(), mu = 1.15)
    got, _ = ds.flow_sigmas(sched.config, "krea-2", 8, 4096, mu = 1.15)
    assert torch.allclose(got, ref.sigmas.float())


# ── stop during a round ──────────────────────────────────────────────────────
def test_stop_between_prompts_ends_the_round_keeping_finished_images(tmp_path):
    plan = ds.plan_samples(
        _cfg(output_dir = str(tmp_path), sample_every = 1, sample_prompts = ("a", "b", "c")).normalized(),
        "z-image",
        ["c"],
    )
    rendered, events = [], []

    def render(i, p):
        rendered.append(p)
        return torch.zeros(1, 3, 8, 8)

    stopped = ds.run_sample_round(
        plan, 5, render, None, lambda _cb, t, **kw: events.append(kw), lambda: len(rendered) >= 1
    )
    assert stopped and rendered == ["a"] and [e["prompt"] for e in events[0]["images"]] == ["a"]


def test_stop_mid_denoise_aborts_the_image(tmp_path):
    from diffusers import FlowMatchEulerDiscreteScheduler

    plan = ds.plan_samples(
        _cfg(output_dir = str(tmp_path), sample_every = 1).normalized(), "z-image", ["c"]
    )
    calls = []

    def velocity(x, t, sig, cond):
        calls.append(1)
        return torch.zeros_like(x)

    with pytest.raises(ds.SampleRoundStopped):
        ds.euler_flow_sample(
            plan = plan,
            index = 0,
            latent_shape = (1, 4, 8, 8),
            image_seq_len = 16,
            family = "z-image",
            scheduler_config = FlowMatchEulerDiscreteScheduler(shift = 3.0).config,
            velocity = velocity,
            cond = "c",
            uncond = None,
            device = "cpu",
            weight_dtype = torch.float32,
            stop_requested = lambda: len(calls) >= 2,
        )
    assert len(calls) == 2
    events = []
    assert ds.run_sample_round(
        plan,
        1,
        lambda i, p: (_ for _ in ()).throw(ds.SampleRoundStopped()),
        None,
        lambda *a, **k: events.append(k),
    )
    assert events == []  # nothing finished, nothing reported


@pytest.mark.parametrize("module", ["diffusion_dit_trainer", "diffusion_lora_trainer"])
def test_trainers_latch_a_stop_seen_mid_round_and_render_a_resume_baseline(module):
    import importlib
    import inspect

    src = inspect.getsource(importlib.import_module(f"core.training.{module}"))
    # The stop poll drains the request; a round that saw it must hand it to the loop.
    assert "stop_now = stop_latched" in src and "if stop_latched:" in src
    # A resumed run renders its baseline at the restored step, after restore_resume_state.
    assert src.index("_sample(resumed)") > src.index("restore_resume_state(")
