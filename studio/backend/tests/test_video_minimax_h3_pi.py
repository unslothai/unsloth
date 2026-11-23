# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 4-step fast mode (pi-H3): grid, sampler, refusals, adapter source and the in-place merge."""

from __future__ import annotations

import hashlib
import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import video_minimax_h3_pi as PI  # noqa: E402
from core.inference import video_minimax_h3_sampling as VS  # noqa: E402


def _lakonlab_sigmas(
    shift,
    steps = 4,
    top = 0.96,
    scale = 0.6,
):
    # FlowEulerODEScheduler.set_timesteps (min_raw_t 0) plus the terminal 0
    import numpy as np
    raw = torch.from_numpy(
        np.linspace(top, top * scale / (steps - 1 + scale), steps, dtype = np.float32)
    )
    return torch.cat([shift * raw / (1 + (shift - 1) * raw), torch.zeros(1)])


def test_pi_grid_is_lakonlabs_4_step_grid():
    sigmas = VS.h3_pi_sigmas(12.0, 4)
    assert torch.allclose(sigmas, _lakonlab_sigmas(12.0), atol = 1e-7, rtol = 0)
    assert [round(v, 4) for v in sigmas.tolist()] == [0.9965, 0.9645, 0.8993, 0.6957, 0.0]
    with pytest.raises(ValueError, match = "exactly 4 steps"):
        VS.h3_pi_sigmas(12.0, 20)


def _schedulers(grid):
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

    video, audio = MiniMaxH3Scheduler(shift = 12.0), MiniMaxH3Scheduler(shift = 3.0)
    assert VS.install_h3_schedulers(
        types.SimpleNamespace(scheduler = video, audio_scheduler = audio), VS.H3_SAMPLER_EULER, grid
    )
    return video, audio


def test_the_pi_grid_drives_both_streams_and_audio_takes_its_own_shift():
    video, audio = _schedulers(VS.H3_GRID_PI)
    video.set_timesteps(5)
    audio.set_timesteps(5)
    assert torch.equal(video.sigmas, VS.h3_pi_sigmas(12.0, 4))
    assert video.num_inference_steps == audio.num_inference_steps == 4
    # LakonLab's audio scheduler: the same raw grid at shift 3
    assert torch.allclose(audio.sigmas, _lakonlab_sigmas(3.0), atol = 2e-6, rtol = 0)
    assert torch.allclose(audio.timesteps, 1 - _lakonlab_sigmas(3.0)[:-1], atol = 2e-6, rtol = 0)


def test_euler_on_the_pi_grid_is_plain_euler():
    video, _ = _schedulers(VS.H3_GRID_PI)
    video.set_timesteps(5)
    sig = video.sigmas
    x = torch.randn(6, 4)
    v = [torch.randn(6, 4) for _ in range(4)]
    want = x.clone()
    for i in range(4):
        x = video.step(v[i], video.timesteps[i], x, return_dict = False)[0]
        want = want + v[i] * (sig[i] - sig[i + 1])
    assert torch.allclose(x, want, atol = 1e-6)


def test_the_default_grid_is_untouched():
    plain, _ = _schedulers(VS.H3_GRID_SIMPLE)
    plain.set_timesteps(21)
    assert torch.equal(plain.sigmas, VS.h3_simple_sigmas(12.0, 20))
    assert "grid" not in plain.__dict__["_studio"]


def test_the_recipe_reports_the_pi_grid_only_when_asked():
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

    pipe = types.SimpleNamespace(
        _blocks = None,
        scheduler = MiniMaxH3Scheduler(shift = 12.0),
        audio_scheduler = MiniMaxH3Scheduler(shift = 3.0),
    )
    off = VS.install_h3_reference_recipe(pipe)
    assert "sigma_grid" not in off and off["sampler"] == VS.h3_sampler(False)
    on = VS.install_h3_reference_recipe(pipe, pi = True)
    assert on["sampler"] == VS.H3_SAMPLER_EULER and on["sigma_grid"] == VS.H3_GRID_PI
    pipe.scheduler.set_timesteps(5)
    assert torch.equal(pipe.scheduler.sigmas, VS.h3_pi_sigmas(12.0, 4))
    # the next plain render puts the reference grid back
    VS.install_h3_reference_recipe(pipe)
    pipe.scheduler.set_timesteps(5)
    assert torch.equal(pipe.scheduler.sigmas, VS.h3_simple_sigmas(12.0, 4))


# ── refusals ──


def _h3():
    from core.inference.video_families import detect_video_family
    return detect_video_family("MiniMaxAI/MiniMax-H3")


def test_static_refusals():
    fam = _h3()
    assert PI.h3_pi_static_refusal(fam, "pipeline") is None
    assert (
        PI.h3_pi_static_refusal(fam, "pipeline", h3_task = "fl2va", transformer_quant = "int8") is None
    )
    assert "MiniMax-H3 option" in PI.h3_pi_static_refusal(
        types.SimpleNamespace(name = "wan2.2"), "pipeline"
    )
    assert "GGUF" in PI.h3_pi_static_refusal(
        fam, "gguf", gguf_filename = "minimax_h3_fl2va-Q4_K_M.gguf"
    )
    assert "reference-to-video" in PI.h3_pi_static_refusal(fam, "pipeline", h3_task = "ref2va")
    assert "PDD" in PI.h3_pi_static_refusal(
        fam,
        "single_file",
        gguf_filename = "minimax_h3_ref2va_pruned_pdd_acc_8step_baked_with_heads_int8_convrot.safetensors",
    )
    assert "LoRA" in PI.h3_pi_static_refusal(fam, "pipeline", has_loras = True)
    assert "dense" in PI.h3_pi_static_refusal(fam, "pipeline", transformer_quant = "none")


def test_begin_load_refuses_before_any_work():
    from core.inference.video import VideoBackend

    backend = VideoBackend()
    with pytest.raises(ValueError, match = "reference-to-video"):
        backend.begin_load("MiniMaxAI/MiniMax-H3", h3_task = "ref2va", h3_fast_4step = True)
    assert backend._loading is None


# ── adapter source ──


def test_the_adapter_only_ever_comes_from_the_unsloth_mirror(monkeypatch, tmp_path):
    monkeypatch.delenv(PI.H3_PI_DIR_ENV, raising = False)
    assert PI.H3_PI_REPO == "unsloth/pi-H3"
    seen = []
    import huggingface_hub

    class RepositoryNotFoundError(Exception):
        pass

    def fake_download(repo_id, filename, **kw):
        seen.append(repo_id)
        raise RepositoryNotFoundError("404")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(PI, "_cached_path", lambda cache_dir: None)
    with pytest.raises(PI.H3PiUnavailable, match = "unsloth/pi-H3 is not available"):
        PI.resolve_h3_pi_adapter(None, cache_dir = str(tmp_path))
    assert seen == ["unsloth/pi-H3"]


def test_a_file_other_than_the_pinned_adapter_is_refused(monkeypatch, tmp_path):
    bad = tmp_path / PI.H3_PI_FILE
    bad.parent.mkdir(parents = True)
    bad.write_bytes(b"not the adapter")
    monkeypatch.setenv(PI.H3_PI_DIR_ENV, str(tmp_path))
    with pytest.raises(PI.H3PiUnavailable, match = "sha256"):
        PI.resolve_h3_pi_adapter(None)
    # the right bytes pass (size and hash pinned)
    data = b"x" * 64
    monkeypatch.setattr(PI, "H3_PI_SIZE", len(data))
    monkeypatch.setattr(PI, "H3_PI_SHA256", hashlib.sha256(data).hexdigest())
    bad.write_bytes(data)
    assert PI.resolve_h3_pi_adapter(None) == str(bad)


def test_availability_reports_the_reason(monkeypatch, tmp_path):
    monkeypatch.delenv(PI.H3_PI_DIR_ENV, raising = False)
    monkeypatch.setattr(PI, "_cached_path", lambda cache_dir: None)
    offline = PI.h3_pi_availability(None, local_files_only = True)
    assert not offline["available"] and "local cache" in offline["reason"]
    import huggingface_hub

    class RepositoryNotFoundError(Exception):
        pass

    def fake_info(self, repo_id, paths, **kw):
        assert repo_id == "unsloth/pi-H3"
        raise RepositoryNotFoundError("404")

    monkeypatch.setattr(huggingface_hub.HfApi, "get_paths_info", fake_info)
    gone = PI.h3_pi_availability(None)
    assert not gone["available"] and "unsloth/pi-H3 is not available" in gone["reason"]
    entry = types.SimpleNamespace(lfs = types.SimpleNamespace(sha256 = PI.H3_PI_SHA256))
    monkeypatch.setattr(huggingface_hub.HfApi, "get_paths_info", lambda self, *a, **k: [entry])
    assert PI.h3_pi_availability(None)["available"]
    other = types.SimpleNamespace(lfs = types.SimpleNamespace(sha256 = "0" * 64))
    monkeypatch.setattr(huggingface_hub.HfApi, "get_paths_info", lambda self, *a, **k: [other])
    assert "different adapter" in PI.h3_pi_availability(None)["reason"]


# ── merge ──


class _Block(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = torch.nn.Module()
        self.attn.to_out = torch.nn.ModuleList([torch.nn.Linear(dim, dim, bias = False)])
        self.adaln_proj = torch.nn.Module()
        self.adaln_proj.linear = torch.nn.Linear(4, 6 * dim, bias = False)


class _Tiny(torch.nn.Module):
    def __init__(
        self,
        dim = 8,
        n = 2,
    ):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([_Block(dim) for _ in range(n)])


def _adapter(
    model,
    rank = 2,
    adaln_shape = None,
):
    g = torch.Generator().manual_seed(0)
    sd = {}
    for i, block in enumerate(model.transformer_blocks):
        lin = block.attn.to_out[0]
        sd[f"transformer_blocks.{i}.attn.to_out.0.lora_A.weight"] = torch.randn(
            rank, lin.in_features, generator = g
        )
        sd[f"transformer_blocks.{i}.attn.to_out.0.lora_B.weight"] = torch.randn(
            lin.out_features, rank, generator = g
        )
        shape = adaln_shape or tuple(block.adaln_proj.linear.weight.shape)
        sd[f"transformer_blocks.{i}.adaln_proj.linear.weight"] = torch.randn(shape, generator = g)
    return sd


@pytest.fixture
def _tiny_rules(monkeypatch):
    monkeypatch.setattr(PI, "_expected_counts", lambda: (2, 2))
    monkeypatch.setattr(PI, "_off_accelerator", lambda t: t.device.type == "meta")


def _save(tmp_path, sd):
    from safetensors.torch import save_file

    path = tmp_path / "adapter.safetensors"
    save_file({k: v.contiguous() for k, v in sd.items()}, str(path))
    return str(path)


def test_the_merge_adds_every_lora_delta_and_copies_every_tensor(tmp_path, _tiny_rules):
    model = _Tiny()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    sd = _adapter(model)
    report = PI.apply_h3_pi_adapter(types.SimpleNamespace(transformer = model), _save(tmp_path, sd))
    assert (report["lora_layers"], report["overwrites"]) == (2, 2)
    assert report["weight_kinds"] == {"dense/g0": 2} and report["restore"] == "reload"
    for i in range(2):
        key = f"transformer_blocks.{i}.attn.to_out.0"
        want = before[f"{key}.weight"] + sd[f"{key}.lora_B.weight"] @ sd[f"{key}.lora_A.weight"]
        assert torch.allclose(model.state_dict()[f"{key}.weight"], want, atol = 1e-5)
        ada = f"transformer_blocks.{i}.adaln_proj.linear.weight"
        assert torch.equal(model.state_dict()[ada], sd[ada])
    assert PI.pipe_h3_pi(types.SimpleNamespace(transformer = model)) is report


def test_a_shape_mismatch_writes_nothing(tmp_path, _tiny_rules):
    model = _Tiny()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # the dense released denoiser: adaLN reads the 2688-wide embedding, not the 8-wide curve
    sd = _adapter(model, adaln_shape = (48, 16))
    with pytest.raises(ValueError, match = "pruned MiniMax-H3"):
        PI.apply_h3_pi_adapter(types.SimpleNamespace(transformer = model), _save(tmp_path, sd))
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), k


def test_missing_layers_and_host_weights_are_refused(tmp_path, monkeypatch, _tiny_rules):
    model = _Tiny()
    sd = _adapter(model)
    sd["transformer_blocks.7.attn.to_out.0.lora_A.weight"] = sd.pop(
        "transformer_blocks.1.attn.to_out.0.lora_A.weight"
    )
    sd["transformer_blocks.7.attn.to_out.0.lora_B.weight"] = sd.pop(
        "transformer_blocks.1.attn.to_out.0.lora_B.weight"
    )
    with pytest.raises(ValueError, match = "not in this denoiser"):
        PI.plan_h3_pi_merge(model, sd)
    monkeypatch.setattr(PI, "_off_accelerator", lambda t: True)
    with pytest.raises(ValueError, match = "resident on the GPU"):
        PI.plan_h3_pi_merge(model, _adapter(model))


def test_the_merge_is_deterministic(tmp_path, _tiny_rules):
    a, b = _Tiny(), _Tiny()
    b.load_state_dict(a.state_dict())
    path = _save(tmp_path, _adapter(a))
    PI.apply_h3_pi_adapter(types.SimpleNamespace(transformer = a), path)
    PI.apply_h3_pi_adapter(types.SimpleNamespace(transformer = b), path)
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k]), k


# ── backend: status and render params ──


def _state(h3_fast):
    from core.inference.video import _VideoLoadState, _detect_load_family
    fam = _detect_load_family("leejet/MiniMax-H3-GGUF", None, "minimax-h3")
    return _VideoLoadState(
        pipe = object(),
        family = fam,
        repo_id = "MiniMaxAI/MiniMax-H3",
        base_repo = fam.base_repo,
        device = "cpu",
        dtype = "bfloat16",
        kind = "gguf",
        engine = "sd_cpp",
        gguf_filename = "minimax_h3_fl2va-Q4_K_M.gguf",
        h3_fast = h3_fast,
    )


_REPORT = {"enabled": True, "steps": 4, "guidance": 1.0, "sampler": "euler", "sigma_grid": "pi_h3"}


def test_status_reports_the_mode_and_its_defaults():
    from core.inference.video import VideoBackend

    backend = VideoBackend()
    backend._state = _state(dict(_REPORT))
    status = backend.status()
    assert status["h3_fast"]["sigma_grid"] == "pi_h3"
    assert (status["defaults"]["steps"], status["defaults"]["guidance"]) == (4, 1.0)
    backend._state = _state(None)
    status = backend.status()
    assert status["h3_fast"] is None and status["defaults"]["steps"] != 4


def test_a_fast_render_runs_4_steps_without_cfg(monkeypatch):
    from core.inference.video import VideoBackend

    backend = VideoBackend()
    seen = {}
    monkeypatch.setattr(
        backend, "_generate_h3_native", lambda **kw: seen.update(kw) or {"mp4_bytes": b""}
    )
    backend._state = _state(dict(_REPORT))
    backend.generate(prompt = "a fox", steps = 20, guidance = 3.0, negative_prompt = "blur")
    assert (seen["steps"], seen["guidance"]) == (4, 1.0)
    assert not seen.get("negative_prompt")
    seen.clear()
    backend._state = _state(None)
    backend.generate(prompt = "a fox", steps = 20, guidance = 3.0)
    # off: the request's steps stand
    assert seen["steps"] == 20


# ── resident identity, automatic reloads, download plan / progress, draft label ──


@pytest.fixture
def _clean_choices(monkeypatch):
    monkeypatch.setattr(PI, "_CHOICES", {})


_FAST_STATUS = {
    "loaded": True,
    "repo_id": "MiniMaxAI/MiniMax-H3",
    "h3_task": "fl2va",
    "h3_fast": dict(_REPORT),
}
_PLAIN_STATUS = {
    "loaded": True,
    "repo_id": "MiniMaxAI/MiniMax-H3",
    "h3_task": "fl2va",
    "h3_fast": None,
}


def test_keepwarm_identity_tells_the_two_builds_apart():
    from core.inference.media_keepwarm import _identity

    assert _identity(_FAST_STATUS) != _identity(_PLAIN_STATUS)
    assert _identity(_FAST_STATUS) == _identity(
        {**_FAST_STATUS, "h3_fast": {"enabled": True, "merge_s": 9}}
    )
    hash(_identity(_FAST_STATUS))


def test_a_resident_answers_only_for_the_mode_its_model_was_loaded_with(_clean_choices):
    from core.inference.media_model_index import MediaModelPick, resident_is_pick, satisfied_by

    pick = MediaModelPick("MiniMaxAI/MiniMax-H3", "MiniMaxAI/MiniMax-H3", model_kind = "pipeline")
    name = "MiniMaxAI/MiniMax-H3"
    # never chosen: a fast resident does not serve a plain request, a plain one does
    assert not satisfied_by(_FAST_STATUS, name, pick)
    assert satisfied_by(_PLAIN_STATUS, name, pick)
    # loaded by the user in the mode: the other way round
    PI.remember_h3_fast(("MiniMaxAI/MiniMax-H3",), "fl2va", True)
    assert satisfied_by(_FAST_STATUS, name, pick) and resident_is_pick(_FAST_STATUS, name, pick)
    assert not satisfied_by(_PLAIN_STATUS, name, pick)
    PI.remember_h3_fast(("MiniMaxAI/MiniMax-H3",), "fl2va", False)
    assert not resident_is_pick(_FAST_STATUS, name, pick)


def test_mcp_load_never_takes_a_resident_in_the_other_mode():
    from studio_mcp.loading import _resident_matches

    model = "MiniMaxAI/MiniMax-H3"
    assert not _resident_matches(_FAST_STATUS, model, {})
    assert _resident_matches(_PLAIN_STATUS, model, {})
    assert _resident_matches(_FAST_STATUS, model, {"h3_fast_4step": True})
    assert not _resident_matches(_PLAIN_STATUS, model, {"h3_fast_4step": True})


def test_an_automatic_reload_keeps_the_mode_or_drops_it_with_a_reason(_clean_choices, monkeypatch):
    names = ("MiniMaxAI/MiniMax-H3",)
    assert PI.auto_reload_h3_fast(names, "fl2va") is False
    PI.remember_h3_fast(names, "fl2va", True)
    seen = []

    def avail(token, **kw):
        seen.append(kw)
        return {"available": True, "reason": None, "repo": PI.H3_PI_REPO, "cached": True}

    monkeypatch.setattr(PI, "h3_pi_availability", avail)
    assert PI.auto_reload_h3_fast(names, "fl2va") is True
    # an automatic load may not reach the Hub
    assert seen[-1]["local_files_only"] is True
    monkeypatch.setattr(
        PI,
        "h3_pi_availability",
        lambda token, **kw: {
            "available": False,
            "reason": "unsloth/pi-H3 is not in the local cache",
            "repo": "",
            "cached": False,
        },
    )
    assert PI.auto_reload_h3_fast(names, "fl2va") is False
    assert not PI.wanted_h3_fast(names, "fl2va")
    assert "dropped on an automatic reload" in PI.h3_fast_drop_reason(names, "fl2va")
    # a later user load clears it
    PI.remember_h3_fast(names, "fl2va", True)
    assert PI.h3_fast_drop_reason(names, "fl2va") is None


def test_the_auto_switch_load_carries_the_mode(_clean_choices, monkeypatch):
    import asyncio

    import routes.video as RV
    from core.inference import media_auto_switch as MAS
    from core.inference.media_model_index import MediaModelPick

    sent = []

    async def fake_load(request, subject, **kw):
        sent.append(request)

    monkeypatch.setattr(RV, "load_video_model_gated", fake_load)
    monkeypatch.setattr(
        PI,
        "h3_pi_availability",
        lambda token, **kw: {"available": True, "reason": None, "repo": "", "cached": True},
    )
    pick = MediaModelPick("MiniMaxAI/MiniMax-H3", "MiniMaxAI/MiniMax-H3", model_kind = "pipeline")
    asyncio.run(MAS._start_load(MAS.VIDEO, pick, "subject"))
    assert sent[-1].h3_fast_4step is False
    PI.remember_h3_fast(("MiniMaxAI/MiniMax-H3",), "fl2va", True)
    asyncio.run(MAS._start_load(MAS.VIDEO, pick, "subject"))
    assert sent[-1].h3_fast_4step is True and sent[-1].h3_task == "fl2va"


def test_status_reports_a_dropped_mode(_clean_choices):
    from core.inference.video import VideoBackend

    PI.remember_h3_fast(("MiniMaxAI/MiniMax-H3",), "fl2va", False, reason = "dropped: no adapter")
    backend = VideoBackend()
    import dataclasses

    backend._state = dataclasses.replace(_state(None), h3_task = "fl2va")
    assert backend.status()["h3_fast_dropped"] == "dropped: no adapter"
    backend._state = dataclasses.replace(_state(dict(_REPORT)), h3_task = "fl2va")
    status = backend.status()
    assert status["h3_fast_dropped"] is None
    assert (
        status["h3_fast"]["quality"] == "draft"
        and "prompt adherence can drop" in status["h3_fast"]["note"]
    )


def test_the_download_plan_carries_the_adapter(monkeypatch, tmp_path):
    monkeypatch.delenv(PI.H3_PI_DIR_ENV, raising = False)
    monkeypatch.setattr(PI, "_cached_path", lambda cache_dir: None)
    base = {
        "entries": [{"repo_id": "unsloth/MiniMax-H3", "files": ["a"], "bytes": 10}],
        "total_bytes": 10,
        "required_bytes": 30,
    }
    plan = PI.add_h3_pi_to_plan(base)
    assert plan["entries"][-1]["repo_id"] == "unsloth/pi-H3"
    assert plan["entries"][-1]["files"] == [PI.H3_PI_FILE]
    assert (
        plan["total_bytes"] == 10 + PI.H3_PI_SIZE and plan["required_bytes"] == 30 + PI.H3_PI_SIZE
    )
    # already on disk: required, but nothing left to stage
    monkeypatch.setattr(PI, "_cached_path", lambda cache_dir: "/x")
    cached = PI.add_h3_pi_to_plan(base)
    assert [e["repo_id"] for e in cached["entries"]] == ["unsloth/MiniMax-H3"]
    assert cached["total_bytes"] == 10 and cached["required_bytes"] == 30 + PI.H3_PI_SIZE


def test_the_load_progress_counts_the_adapter(monkeypatch, tmp_path):
    from core.inference import video as V

    plan = {
        V._FETCH_BASE: V._PlannedFetch(
            "unsloth/MiniMax-H3", None, (("a.safetensors", "k" * 64, 100),)
        ),
        V._FETCH_H3_PI: V._PlannedFetch(PI.H3_PI_REPO, None, PI.h3_pi_fetch_files()),
    }
    assert V._plan_total(plan) == 100 + PI.H3_PI_SIZE
    # a re-plan after the checkpoint fetch keeps the adapter role
    assert V._FETCH_H3_PI in V._keep_fetched_checkpoint({V._FETCH_BASE: plan[V._FETCH_BASE]}, plan)
    monkeypatch.setattr(V.VideoBackend, "_plan_cache_roots", staticmethod(lambda: [tmp_path]))
    assert V.VideoBackend._planned_bytes(plan) == 0
    blobs = tmp_path / "models--unsloth--pi-H3" / "blobs"
    blobs.mkdir(parents = True)
    (blobs / f"{PI.H3_PI_SHA256}.incomplete").write_bytes(b"x" * 4096)
    assert V.VideoBackend._planned_bytes(plan) == 4096
