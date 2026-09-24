# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The NVFP4 switch (``UNSLOTH_NVFP4_DIFFUSION``), default OFF.

Off means NVFP4 is unavailable for image and video generation in general: ``auto`` never picks it,
an explicit transformer or text-encoder request is refused with a clear message, no hosted
``*-NVFP4`` repo is resolved, planned or asked about, and nothing advertises it. The modules that
test the NVFP4 behaviour itself run with the switch ON (see the conftest fixture); this one runs
with the environment cleared, so it sees the shipped default, and turns it on only where it proves
the enabled path is still reachable end to end.
"""

from __future__ import annotations

import asyncio
import inspect
import types

import pytest

from core.inference import diffusion_nvfp4_flag as flag
from core.inference import diffusion_transformer_quant as tq

ENV = "UNSLOTH_NVFP4_DIFFUSION"
DISABLED = "NVFP4 is disabled in this build"


@pytest.fixture(autouse = True)
def _default_off(monkeypatch):
    monkeypatch.delenv(ENV, raising = False)
    yield


def _enable(monkeypatch):
    monkeypatch.setenv(ENV, "1")


# ---------------------------------------------------------------------------------------------
# Parsing


def test_the_default_is_off_and_lives_in_one_constant():
    assert flag.NVFP4_DIFFUSION_DEFAULT is False
    assert flag.NVFP4_DIFFUSION_ENV == ENV
    assert flag.nvfp4_diffusion_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on", "On"])
def test_truthy_values_enable(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    assert flag.nvfp4_diffusion_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "  ", "garbage", "2", "enable"])
def test_everything_else_disables(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    assert flag.nvfp4_diffusion_enabled() is False


def test_flipping_the_default_is_the_one_line_change(monkeypatch):
    """Unset (and unrecognised) follow the constant; an explicit 0 still turns it off."""
    monkeypatch.setattr(flag, "NVFP4_DIFFUSION_DEFAULT", True)
    assert flag.nvfp4_diffusion_enabled() is True
    monkeypatch.setenv(ENV, "0")
    assert flag.nvfp4_diffusion_enabled() is False
    monkeypatch.setenv(ENV, "garbage")
    assert flag.nvfp4_diffusion_enabled() is True


def test_the_switch_is_read_at_call_time(monkeypatch):
    assert flag.nvfp4_blocked("nvfp4")
    _enable(monkeypatch)
    assert not flag.nvfp4_blocked("nvfp4")
    monkeypatch.setenv(ENV, "0")
    assert flag.nvfp4_blocked("NVFP4")


def test_only_nvfp4_is_blocked():
    for scheme in ("int8", "fp8", "mxfp8", "fp8_dynamic", "auto", None, ""):
        assert not flag.nvfp4_blocked(scheme)
    assert flag.without_nvfp4(("int8", "nvfp4", "fp8")) == ("int8", "fp8")
    assert flag.nvfp4_repo_blocked("unsloth/Wan2.2-TI2V-5B-NVFP4")
    assert not flag.nvfp4_repo_blocked("unsloth/Wan2.2-TI2V-5B-FP8")
    assert not flag.nvfp4_repo_blocked(None)


# ---------------------------------------------------------------------------------------------
# Dense transformer quant: auto and explicit


def _blackwell(monkeypatch, *, supported = ("int8", "fp8", "nvfp4", "mxfp8")):
    """A datacenter sm_100 host whose smoke probe passes ``supported``, nothing allocated."""
    monkeypatch.setattr(tq, "_capability", lambda: (10, 0))
    monkeypatch.setattr(tq, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(tq, "_is_consumer_gpu", lambda device = None: False)
    monkeypatch.setattr(tq, "_child_probe_table", lambda device: None)
    monkeypatch.setattr(
        tq, "_smoke_probe", lambda scheme, device, unproven_ok = False: scheme in supported
    )
    monkeypatch.setattr(tq, "_SMOKE_CACHE", {})
    # nvfp4 in the ladder AND at the head of a family prefer row, the two ways auto could reach it.
    monkeypatch.setattr(
        tq,
        "_AUTO_LADDER",
        (((10, 0), (tq.TQ_INT8, tq.TQ_FP8, tq.TQ_NVFP4, tq.TQ_MXFP8)),),
    )
    monkeypatch.setattr(
        tq,
        "_FAMILY_AUTO_PREFER",
        {"wan2.2-t2v-a14b": tq._AutoPrefer(floor = (10, 0), schemes = (tq.TQ_NVFP4,))},
    )
    return types.SimpleNamespace(device = "cuda", dtype = None)


def _auto(target, family):
    """``auto`` on ``family`` as a load asks it, with a hosted checkpoint for every scheme where
    the selector requires one for nvfp4, so only the switch stands between auto and nvfp4."""
    kwargs = {}
    if "has_prequant" in inspect.signature(tq.select_transformer_quant_scheme).parameters:
        kwargs["has_prequant"] = lambda scheme: True
    return tq.select_transformer_quant_scheme(target, "auto", family = family, **kwargs)


def test_auto_never_offers_nvfp4(monkeypatch):
    target = _blackwell(monkeypatch)
    fam = "wan2.2-t2v-a14b"
    assert tq.TQ_NVFP4 not in tq._auto_scheme_order(fam, "cuda", (10, 0))
    assert tq.TQ_NVFP4 not in tq.auto_scheme_candidates(target, fam)
    assert tq.TQ_NVFP4 not in tq.auto_scheme_candidates_cached(target, fam)
    assert _auto(target, fam) == tq.TQ_INT8


def test_a_host_that_only_runs_nvfp4_has_nothing_automatic(monkeypatch):
    target = _blackwell(monkeypatch, supported = ("nvfp4",))
    assert _auto(target, "wan2.2-t2v-a14b") is None
    probed = []
    monkeypatch.setattr(tq, "_smoke_probe", lambda scheme, *a, **k: probed.append(scheme) or True)
    assert tq._scheme_supported(tq.TQ_NVFP4, "cuda") is False
    assert probed == []


def test_the_child_probe_is_not_asked_about_nvfp4(monkeypatch):
    asked = []

    class _Stop(Exception):
        pass

    def _spawn(*args, **kwargs):
        asked.append(kwargs.get("args") or args)
        raise _Stop

    import multiprocessing as mp

    class _Ctx:
        def Queue(self):
            return types.SimpleNamespace(close = lambda: None, join_thread = lambda: None)

        def Process(self, *args, **kwargs):
            return _spawn(*args, **kwargs)

    monkeypatch.setattr(mp, "get_context", lambda kind: _Ctx())
    monkeypatch.setattr(tq, "_CHILD_PROBE_UNAVAILABLE", False)
    monkeypatch.setattr(tq, "_close_probe_child", lambda proc, queue: True)
    assert tq._child_probe_table("cuda") is None
    assert asked, "the spawn was never attempted"
    schemes = asked[0][4]
    assert tq.TQ_NVFP4 not in schemes and tq.TQ_INT8 in schemes


def test_an_explicit_nvfp4_transformer_quant_is_refused_not_swapped(monkeypatch):
    target = _blackwell(monkeypatch)
    with pytest.raises(ValueError, match = DISABLED):
        tq.normalize_transformer_quant("nvfp4")
    with pytest.raises(ValueError, match = DISABLED):
        tq.normalize_transformer_quant(" NVFP4 ")
    with pytest.raises(ValueError, match = DISABLED):
        tq.select_transformer_quant_scheme(target, "nvfp4", family = "z-image")
    assert DISABLED in tq.explain_unusable_scheme("z-image", "nvfp4")
    # Every other scheme is untouched.
    assert tq.normalize_transformer_quant("fp8") == tq.TQ_FP8
    assert tq.select_transformer_quant_scheme(target, "mxfp8", family = "z-image") == tq.TQ_MXFP8


def test_the_precision_gates_refuse_nvfp4_even_under_the_silent_fallback(monkeypatch):
    """The opt-in fallback swaps a declined scheme for another one; a disabled scheme is refused."""
    from core.inference.diffusion import DiffusionBackend
    from core.inference.video import assert_video_precision_available

    monkeypatch.setenv("UNSLOTH_DIFFUSION_ALLOW_PRECISION_FALLBACK", "1")
    with pytest.raises(ValueError, match = DISABLED):
        DiffusionBackend.assert_precision_available(
            object(), None, model_kind = "pipeline", transformer_quant = "nvfp4"
        )
    with pytest.raises(ValueError, match = DISABLED):
        DiffusionBackend.assert_precision_available(
            object(), None, model_kind = "pipeline", text_encoder_quant = "nvfp4"
        )
    with pytest.raises(ValueError, match = DISABLED):
        assert_video_precision_available(None, model_kind = "pipeline", transformer_quant = "nvfp4")
    from routes.inference import _assert_native_precision_unset

    with pytest.raises(ValueError, match = DISABLED):
        _assert_native_precision_unset(transformer_quant = "nvfp4")


def _run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize(
    "field,control",
    [("transformer_quant", "transformer_quant"), ("text_encoder_quant", "text_encoder_quant")],
)
def test_every_load_and_plan_route_400s_an_nvfp4_request(monkeypatch, field, control):
    from fastapi import HTTPException

    from models.inference import DiffusionLoadRequest, VideoLoadRequest
    from routes import inference as image_routes
    from routes import video as video_routes

    touched = []
    # Nothing past the refusal may run: not the account checks, not the backend.
    monkeypatch.setattr(
        image_routes.account_access, "managed_account", lambda: touched.append("acct") or False
    )
    monkeypatch.setattr(
        video_routes.account_access, "managed_account", lambda: touched.append("acct") or False
    )
    image = DiffusionLoadRequest(model_path = "Tongyi-MAI/Z-Image-Turbo", **{field: "nvfp4"})
    video = VideoLoadRequest(model_path = "Wan-AI/Wan2.2-TI2V-5B-Diffusers", **{field: "nvfp4"})
    calls = [
        image_routes.diffusion_download_plan(image, current_subject = "u"),
        image_routes.load_diffusion_model_gated(image, "u"),
        video_routes.video_download_plan(video, current_subject = "u"),
        video_routes.load_video_model_gated(video, "u"),
    ]
    for call in calls:
        with pytest.raises(HTTPException) as exc:
            _run(call)
        assert exc.value.status_code == 400
        assert DISABLED in exc.value.detail
        assert f"{control}='nvfp4'" in exc.value.detail
    assert touched == []


# ---------------------------------------------------------------------------------------------
# Text encoder


def test_text_encoder_nvfp4_is_refused():
    from core.inference import diffusion_precision as dp

    with pytest.raises(ValueError, match = DISABLED):
        dp.normalize_te_quant("nvfp4")
    with pytest.raises(ValueError, match = DISABLED):
        dp.resolve_te_quant_request("nvfp4", None)
    # A family default of nvfp4 is simply no default while it is off, never a refused load.
    assert dp.resolve_te_quant_request(None, "nvfp4") == (None, False)
    assert dp.resolve_te_quant_request("auto", "fp8") == ("fp8", True)
    target = types.SimpleNamespace(device = "cuda", dtype = None)
    assert dp.te_quant_supported(target, dp.TE_QUANT_NVFP4) is False
    assert dp.normalize_te_quant("fp8_dynamic") == dp.TE_QUANT_FP8_DYNAMIC


def test_no_hosted_nvfp4_text_encoder_is_resolved():
    from core.inference.diffusion_te_prequant import family_te_prequant_repo
    fam = types.SimpleNamespace(
        te_prequant_repos = (("nvfp4", "text_encoder", "unsloth/Some-Model-NVFP4"),)
    )
    assert family_te_prequant_repo(fam, "nvfp4", "text_encoder") is None


# ---------------------------------------------------------------------------------------------
# Hosted prequant: no lookup, no seed, no Hub request


_WAN_T2V = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"


def _wan_a14b():
    from core.inference.video_families import detect_video_family
    return detect_video_family(_WAN_T2V)


def test_no_hosted_nvfp4_repo_is_resolved_for_any_family():
    from core.inference.diffusion_families import family_prequant_repo
    from core.inference.diffusion_prequant import resolve_prequant_source
    from core.inference.video_families import (
        _FAMILIES as VIDEO_FAMILIES,
        video_family_prequant_available,
        video_family_prequant_repo,
        video_family_prequant_resident_gb,
        video_family_prequant_schemes,
    )
    from core.inference.video_denoiser_prequant import denoiser_prequant_sources

    hosted = [f for f in VIDEO_FAMILIES if any(s == "nvfp4" for s, _r in f.prequant_repos)]
    assert hosted, "the registry no longer hosts an NVFP4 denoiser; retarget this test"
    for fam in hosted:
        assert video_family_prequant_repo(fam, "nvfp4") is None
        assert family_prequant_repo(fam, "nvfp4") is None
        assert resolve_prequant_source(fam, "nvfp4") is None
        assert resolve_prequant_source(fam, "nvfp4", path_override = "/x/local-NVFP4.pt") is None
        assert denoiser_prequant_sources(fam, "nvfp4", fam.base_repo) is None
        assert not video_family_prequant_available(fam, "nvfp4")
        assert video_family_prequant_resident_gb(fam, "nvfp4") is None
        assert "nvfp4" not in video_family_prequant_schemes(fam)


class _RecordingApi:
    def __init__(self, repos):
        self.repos = repos
        self.asked: list[str] = []

    def model_info(
        self,
        repo_id,
        files_metadata = False,
        token = None,
    ):
        self.asked.append(repo_id)
        return types.SimpleNamespace(siblings = self.repos[repo_id])


def _sib(name, size):
    return types.SimpleNamespace(rfilename = name, size = size)


_WAN_SIBLINGS = {
    _WAN_T2V: [
        _sib("model_index.json", 1000),
        _sib("transformer/config.json", 1000),
        _sib("transformer/diffusion_pytorch_model.safetensors", 28_000_000_000),
        _sib("transformer_2/config.json", 1000),
        _sib("transformer_2/diffusion_pytorch_model.safetensors", 28_000_000_000),
        _sib("text_encoder/model-00001-of-00001.safetensors", 11_000_000_000),
        _sib("vae/diffusion_pytorch_model.safetensors", 500_000_000),
    ],
    "unsloth/Wan2.2-T2V-A14B-NVFP4": [
        _sib("Wan2.2-T2V-A14B-NVFP4.pt", 8_000_000_000),
        _sib("Wan2.2-T2V-A14B-transformer_2-NVFP4.pt", 8_100_000_000),
    ],
}


def _wan_plan(monkeypatch):
    """A Wan2.2-A14B pipeline plan on a host whose auto selector lands on nvfp4, Hub mocked."""
    import core.inference.video as video_mod
    from core.inference import diffusion_prequant
    from core.inference.diffusion import DiffusionBackend

    api = _RecordingApi(_WAN_SIBLINGS)
    monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **k: api)
    monkeypatch.setattr(
        DiffusionBackend,
        "_hub_file_is_cached",
        staticmethod(lambda repo_id, filename, revision = None, expected_size = None, **kw: False),
    )
    # The selector is what the switch filters; forcing its answer proves the resolver below it
    # holds on its own too.
    monkeypatch.setattr(video_mod, "select_transformer_quant_scheme", lambda *a, **k: "nvfp4")
    monkeypatch.setattr(
        diffusion_prequant, "restricted_prequant_load_supported", lambda *a, **k: True
    )
    monkeypatch.setattr(video_mod, "_video_seed_stays_resident", lambda fam, **kw: True)
    monkeypatch.setattr(
        video_mod,
        "resolve_diffusion_device_target",
        lambda *a, **k: types.SimpleNamespace(backend = "cuda", device = "cuda", dtype = None),
    )
    from core.inference.video import VideoBackend

    plan = VideoBackend().download_plan(_WAN_T2V, model_kind = "pipeline")
    staged = {f for e in plan["entries"] for f in e["files"]}
    return plan, staged, api


def test_the_video_download_plan_asks_no_nvfp4_repo_and_keeps_the_dense_shards(monkeypatch):
    plan, staged, api = _wan_plan(monkeypatch)
    assert not any(repo.lower().endswith("-nvfp4") for repo in api.asked), api.asked
    assert not any("NVFP4" in f for f in staged)
    assert "transformer/diffusion_pytorch_model.safetensors" in staged
    assert "transformer_2/diffusion_pytorch_model.safetensors" in staged


def test_the_video_seed_hub_probe_never_reaches_an_nvfp4_repo():
    from core.inference.video import VideoBackend

    api = _RecordingApi(_WAN_SIBLINGS)
    repo, files = VideoBackend._denoiser_prequant_hub_files(_wan_a14b(), "nvfp4", _WAN_T2V, api)
    assert (repo, files) == (None, [])
    assert api.asked == []


def test_the_image_hub_entry_never_asks_about_an_nvfp4_repo(monkeypatch):
    from core.inference.diffusion import DiffusionBackend

    asked = []
    monkeypatch.setattr(
        "huggingface_hub.HfApi", lambda *a, **k: types.SimpleNamespace(model_info = asked.append)
    )
    source = types.SimpleNamespace(
        kind = "repo", location = "unsloth/Z-Image-Turbo-NVFP4", filename = "x-NVFP4.pt"
    )
    assert DiffusionBackend._prequant_source_hub_entry(source, None, scheme = "nvfp4") is None
    # The repo-name backstop holds even when a caller forgets the scheme.
    assert DiffusionBackend._prequant_source_hub_entry(source, None) is None
    assert asked == []


# ---------------------------------------------------------------------------------------------
# Capability and info payloads


def test_the_info_payload_and_estimates_omit_nvfp4(monkeypatch):
    from core.inference.diffusion_auto_policy import estimate_dense_quant
    from core.inference.diffusion_families import detect_family
    from core.inference.diffusion_inference_info import family_inference_infos

    infos = family_inference_infos()
    assert infos
    for info in infos:
        assert "nvfp4" not in info["estimated_resident_gb"]
        assert "int8" in info["estimated_resident_gb"]
    assert estimate_dense_quant(detect_family("Tongyi-MAI/Z-Image-Turbo"), "nvfp4") is None
    assert estimate_dense_quant(detect_family("Tongyi-MAI/Z-Image-Turbo"), "int8") is not None


def test_api_system_reports_the_switch(monkeypatch):
    import main

    assert main._nvfp4_diffusion_enabled() is False
    _enable(monkeypatch)
    assert main._nvfp4_diffusion_enabled() is True


# ---------------------------------------------------------------------------------------------
# Enabled: the same entry points reach the real NVFP4 code, so an internal run with the switch on
# exercises exactly today's behaviour.


def test_enabled_the_nvfp4_path_is_reachable_end_to_end(monkeypatch):
    _enable(monkeypatch)
    from core.inference import diffusion_precision as dp
    from core.inference.diffusion_inference_info import family_inference_infos
    from core.inference.video_denoiser_prequant import denoiser_prequant_sources
    from core.inference.video_families import video_family_prequant_repo

    # auto: a prefer row that leads with nvfp4 is honoured again.
    target = _blackwell(monkeypatch)
    assert _auto(target, "wan2.2-t2v-a14b") == "nvfp4"
    # explicit: accepted, validated and selected like any other scheme.
    assert tq.normalize_transformer_quant("nvfp4") == "nvfp4"
    assert tq.select_transformer_quant_scheme(target, "nvfp4", family = "z-image") == "nvfp4"
    assert dp.normalize_te_quant("nvfp4") == "nvfp4"
    # prequant lookup: the hosted repo resolves and every expert seeds.
    fam = _wan_a14b()
    assert video_family_prequant_repo(fam, "nvfp4") == "unsloth/Wan2.2-T2V-A14B-NVFP4"
    sources = denoiser_prequant_sources(fam, "nvfp4", _WAN_T2V)
    assert sources and set(sources) == {"transformer", "transformer_2"}
    assert all(s.location == "unsloth/Wan2.2-T2V-A14B-NVFP4" for s in sources.values())
    assert all("nvfp4" in info["estimated_resident_gb"] for info in family_inference_infos())


def test_enabled_the_video_plan_stages_the_hosted_nvfp4_denoisers(monkeypatch):
    _enable(monkeypatch)
    plan, staged, api = _wan_plan(monkeypatch)
    assert "unsloth/Wan2.2-T2V-A14B-NVFP4" in api.asked
    assert {"Wan2.2-T2V-A14B-NVFP4.pt", "Wan2.2-T2V-A14B-transformer_2-NVFP4.pt"} <= staged
    assert "transformer/diffusion_pytorch_model.safetensors" not in staged


def test_enabled_the_routes_let_nvfp4_through_the_switch(monkeypatch):
    _enable(monkeypatch)
    from routes.inference import _refuse_disabled_nvfp4_request
    _refuse_disabled_nvfp4_request(
        types.SimpleNamespace(transformer_quant = "nvfp4", text_encoder_quant = "nvfp4")
    )


# ---------------------------------------------------------------------------------------------
# Per-layer image policies, gated auto rows and the flashinfer backend (studio-nvfp4-image)


def _record_nvfp4_probes(monkeypatch):
    """Record every gate-record read and flashinfer backend probe instead of running them."""
    from core.inference import diffusion_nvfp4_gate as gate
    from core.inference import diffusion_nvfp4_ops as ops

    calls: list[str] = []
    monkeypatch.setattr(gate, "nvfp4_gate_passed", lambda *a, **k: calls.append("gate") or True)
    monkeypatch.setattr(
        gate, "nvfp4_gate_backends", lambda *a, **k: calls.append("gate") or ("flashinfer",)
    )
    monkeypatch.setattr(
        ops, "select_nvfp4_backend", lambda *a, **k: calls.append("backend") or "flashinfer"
    )
    return calls


@pytest.mark.parametrize("family", ["wan2.2-t2v-a14b", "z-image", "flux.1", "qwen-image"])
def test_the_nvfp4_auto_rows_are_dropped_before_any_gate_or_flashinfer_probe(monkeypatch, family):
    calls = _record_nvfp4_probes(monkeypatch)
    order = tq._auto_scheme_order(family, "cuda", (10, 0), "some/base")
    assert tq.TQ_NVFP4 not in order
    assert order == (tq.TQ_INT8, tq.TQ_FP8, tq.TQ_MXFP8)
    assert calls == []


def test_enabled_the_flashinfer_auto_row_leads_again(monkeypatch):
    _enable(monkeypatch)
    calls = _record_nvfp4_probes(monkeypatch)
    order = tq._auto_scheme_order("wan2.2-t2v-a14b", "cuda", (10, 0), "Wan-AI/Wan2.2-T2V-A14B")
    assert order[0] == tq.TQ_NVFP4
    assert "backend" in calls


def test_no_flashinfer_import_or_preflight_is_triggered(monkeypatch):
    from core.inference import diffusion_nvfp4_linear as linear
    from core.inference import diffusion_nvfp4_ops as ops

    touched = []
    monkeypatch.setattr(
        ops, "_flashinfer_available", lambda: touched.append("import") or (True, "x")
    )
    monkeypatch.setattr(ops, "_preflight_probe", lambda dev: touched.append("preflight") or True)
    backend, reason = ops._resolve_backend(0)
    assert backend == ops.BACKEND_TORCHAO and DISABLED in reason
    assert ops.select_nvfp4_backend(0) == ops.BACKEND_TORCHAO
    assert ops.nvfp4_preflight(0)["ok"] is False
    assert linear.nvfp4_prewarm(object(), (16,)) == 0
    assert touched == []
    assert tq._nvfp4_gate_passed("z-image", "Tongyi-MAI/Z-Image-Turbo") is False
    assert tq._nvfp4_backend_is("cuda", "flashinfer") is False


def test_the_hosted_image_nvfp4_rows_resolve_to_nothing(monkeypatch):
    from core.inference.diffusion_families import detect_family, family_prequant_repo
    from core.inference.diffusion_prequant import usable_prequant_source

    schnell = "black-forest-labs/FLUX.1-schnell"
    zimage = "Tongyi-MAI/Z-Image-Turbo"
    for base in (schnell, zimage):
        fam = detect_family(base)
        assert family_prequant_repo(fam, "nvfp4", base_repo = base) is None
        assert usable_prequant_source(fam, "nvfp4", base_repo = base) is None
    _enable(monkeypatch)
    assert family_prequant_repo(detect_family(schnell), "nvfp4", base_repo = schnell) == (
        "unsloth/FLUX.1-schnell-NVFP4"
    )
    assert family_prequant_repo(detect_family(zimage), "nvfp4", base_repo = zimage) == (
        "unsloth/Z-Image-Turbo-NVFP4"
    )


def test_the_per_layer_policy_factor_is_not_applied(monkeypatch):
    from core.inference.diffusion_auto_policy import policy_steady_factor

    assert policy_steady_factor("z-image", "Tongyi-MAI/Z-Image-Turbo") is None
    _enable(monkeypatch)
    assert policy_steady_factor("z-image", "Tongyi-MAI/Z-Image-Turbo") is not None


# ---------------------------------------------------------------------------------------------
# An NVFP4 checkpoint loaded directly as the model


def _prequant_dir(root, name, scheme):
    """A directory holding one tiny safetensors pre-quant artifact that records ``scheme``."""
    import json

    import torch
    from safetensors.torch import save_file

    from core.inference.diffusion_prequant import PREQUANT_FORMAT
    from core.inference.prequant_safetensors import UNSLOTH_FORMAT_KEY, UNSLOTH_METADATA_KEY

    folder = root / name
    folder.mkdir()
    save_file(
        {"w": torch.zeros(2)},
        str(folder / "model.safetensors"),
        metadata = {
            UNSLOTH_FORMAT_KEY: PREQUANT_FORMAT,
            UNSLOTH_METADATA_KEY: json.dumps({"scheme": scheme}),
        },
    )
    return folder


def _checkpoint_route_calls(image_path, video_path, monkeypatch):
    from models.inference import DiffusionLoadRequest, VideoLoadRequest
    from routes import inference as image_routes
    from routes import video as video_routes

    monkeypatch.setattr(image_routes.account_access, "managed_account", lambda: False)
    monkeypatch.setattr(video_routes.account_access, "managed_account", lambda: False)
    monkeypatch.setattr(image_routes.account_access, "require_idle_other_accounts", lambda: None)
    monkeypatch.setattr(video_routes.account_access, "require_idle_other_accounts", lambda: None)
    image = DiffusionLoadRequest(model_path = image_path)
    video = VideoLoadRequest(model_path = video_path)
    return [
        lambda: image_routes.diffusion_download_plan(image, current_subject = "u"),
        lambda: image_routes.load_diffusion_model_gated(image, "u"),
        lambda: video_routes.video_download_plan(video, current_subject = "u"),
        lambda: video_routes.load_video_model_gated(video, "u"),
    ]


HOSTED_NVFP4 = [
    "unsloth/Z-Image-Turbo-NVFP4",
    "unsloth/FLUX.1-schnell-NVFP4",
    "unsloth/Qwen-Image-2512-NVFP4",
    "unsloth/Wan2.2-TI2V-5B-NVFP4",
    "unsloth/Wan2.2-T2V-A14B-NVFP4",
    "unsloth/HunyuanVideo-1.5-NVFP4",
]


@pytest.mark.parametrize("repo", HOSTED_NVFP4)
def test_a_hosted_nvfp4_repo_as_the_model_is_refused(repo):
    with pytest.raises(ValueError, match = DISABLED):
        flag.refuse_disabled_nvfp4_checkpoint(repo)


def test_every_family_registered_nvfp4_repo_is_recognised_by_the_table():
    from core.inference.diffusion_prequant import hosted_nvfp4_repo_ids
    from core.inference.video_families import _FAMILIES

    registered = {
        repo.lower()
        for fam in _FAMILIES
        for scheme, repo in fam.prequant_repos
        if scheme == "nvfp4"
    }
    assert registered and registered <= hosted_nvfp4_repo_ids()


def test_the_image_family_nvfp4_repos_are_recognised_by_the_table():
    from core.inference.diffusion_prequant import hosted_nvfp4_repo_ids

    ids = hosted_nvfp4_repo_ids()
    # A (scheme, repo) row and a (base, scheme, repo) variant row.
    assert "unsloth/z-image-turbo-nvfp4" in ids
    assert "unsloth/flux.1-schnell-nvfp4" in ids
    assert not any("fp8" in repo and "nvfp4" not in repo for repo in ids)


def test_a_cached_repo_is_judged_by_its_metadata_not_its_name(tmp_path, monkeypatch):
    from core.inference import diffusion_prequant as dpq

    snap = _prequant_dir(tmp_path, "snap", "nvfp4")
    monkeypatch.setattr(dpq, "_cached_snapshot_dirs", lambda repo: [str(snap)])
    assert dpq.declares_nvfp4_checkpoint("someone/renamed-checkpoint") is True
    with pytest.raises(ValueError, match = DISABLED):
        flag.refuse_disabled_nvfp4_checkpoint("someone/renamed-checkpoint")


def test_a_local_dir_whose_metadata_declares_nvfp4_is_refused(tmp_path):
    # No NVFP4 in the name: the recorded scheme is the evidence.
    folder = _prequant_dir(tmp_path, "my-checkpoint", "nvfp4")
    with pytest.raises(ValueError, match = DISABLED):
        flag.refuse_disabled_nvfp4_checkpoint(str(folder))
    with pytest.raises(ValueError, match = DISABLED):
        flag.refuse_disabled_nvfp4_checkpoint(str(folder / "model.safetensors"))


def test_the_routes_400_an_nvfp4_checkpoint_as_the_model(tmp_path, monkeypatch):
    from fastapi import HTTPException
    folder = _prequant_dir(tmp_path, "local-z-image", "nvfp4")
    for image_path, video_path in (
        ("unsloth/Z-Image-Turbo-NVFP4", "unsloth/Wan2.2-TI2V-5B-NVFP4"),
        (str(folder), str(folder)),
    ):
        for call in _checkpoint_route_calls(image_path, video_path, monkeypatch):
            with pytest.raises(HTTPException) as exc:
                _run(call())
            assert exc.value.status_code == 400
            assert DISABLED in exc.value.detail
            assert "NVFP4 checkpoint" in exc.value.detail
            assert str(tmp_path) not in exc.value.detail


@pytest.mark.parametrize(
    "path",
    [
        "Tongyi-MAI/Z-Image-Turbo",
        "unsloth/Z-Image-Turbo-FP8",
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "unsloth/Qwen-Image-2512-GGUF",
        "",
        None,
    ],
)
def test_non_nvfp4_models_are_not_refused(path):
    flag.refuse_disabled_nvfp4_checkpoint(path)


def test_a_local_dir_with_another_scheme_is_not_refused(tmp_path):
    folder = _prequant_dir(tmp_path, "my-fp8-checkpoint", "fp8")
    flag.refuse_disabled_nvfp4_checkpoint(str(folder))
    plain = tmp_path / "plain-pipeline"
    plain.mkdir()
    (plain / "model_index.json").write_text("{}")
    flag.refuse_disabled_nvfp4_checkpoint(str(plain))
    # A pipeline loads from its component folders, so a stray root artifact is not the model.
    pipeline = _prequant_dir(tmp_path, "pipeline-with-extra", "nvfp4")
    (pipeline / "model_index.json").write_text("{}")
    flag.refuse_disabled_nvfp4_checkpoint(str(pipeline))


def test_enabled_an_nvfp4_checkpoint_as_the_model_proceeds(tmp_path, monkeypatch):
    _enable(monkeypatch)
    folder = _prequant_dir(tmp_path, "my-checkpoint", "nvfp4")
    for path in (*HOSTED_NVFP4, str(folder)):
        flag.refuse_disabled_nvfp4_checkpoint(path)

    reached = []

    def _stop(*args, **kwargs):
        reached.append(True)
        raise RuntimeError("reached the backend")

    monkeypatch.setattr("core.inference.diffusion.get_diffusion_backend", _stop)
    calls = _checkpoint_route_calls("unsloth/Z-Image-Turbo-NVFP4", str(folder), monkeypatch)
    with pytest.raises(RuntimeError, match = "reached the backend"):
        _run(calls[0]())
    assert reached == [True]
