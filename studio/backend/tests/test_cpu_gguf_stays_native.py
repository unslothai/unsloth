# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A GGUF on a CPU-only torch runs on sd.cpp only: a native failure, an explicit Diffusers pick and the server's own
diffusers routes are refused with an actionable reason, never swapped for a diffusers load that downloads the dense
encoder the Hub plan never staged (FLUX.3 Klein: 7.2 GB staged, 11.8 GB loaded). Only the developer kill switch
UNSLOTH_DIFFUSION_CPU_GGUF_DIFFUSERS_FALLBACK=1 restores it."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.inference import diffusion_engine_router as r
from core.inference.diffusion_families import detect_family
from core.inference.sd_cpp_engine import ENGINE_DIFFUSERS, ENGINE_SD_CPP

_ENVS = (
    "UNSLOTH_DIFFUSION_ENGINE",
    "UNSLOTH_DIFFUSION_SD_CPP",
    "UNSLOTH_DIFFUSION_SD_CPP_MPS",
    "UNSLOTH_DIFFUSION_SD_CPP_INSTALL",
    r.CPU_GGUF_DIFFUSERS_FALLBACK_ENV,
)


@pytest.fixture(autouse = True)
def _host(monkeypatch):
    r._supported_family_capabilities.cache_clear()
    for e in _ENVS:
        monkeypatch.delenv(e, raising = False)
    monkeypatch.setattr(
        r,
        "get_active_diffusion_engine",
        lambda: SimpleNamespace(
            status = lambda: {"loaded": False, "repo_id": None}, unload = lambda: None
        ),
    )
    monkeypatch.setattr(r, "off_torch_sd_cpp_device", lambda *_a, **_k: None)
    monkeypatch.setattr(r, "image_install_accelerator", lambda backend: "auto")
    monkeypatch.setattr(r, "_server_binary_runnable", lambda *_a, **_k: True)
    monkeypatch.setattr(r, "sd_cpp_binary_runs_family", lambda *_a, **_k: True)
    monkeypatch.setattr(r, "usable_or_recorded_failure", lambda b, *_a, **_k: b)
    # FLUX.3 needs a newer diffusers than some test venvs carry; the pick refusal is not under test here.
    monkeypatch.setattr(r, "family_pipeline_available", lambda fam: True)
    from core.inference import sd_cpp_backend as b

    b.clear_native_start_failure()
    saved = (r._active_engine_name, r._fallback_reason, r._engine_pick, r._pending_engine_pick)
    try:
        yield
    finally:
        r._active_engine_name, r._fallback_reason, r._engine_pick, r._pending_engine_pick = saved
        r._supported_family_capabilities.cache_clear()


def _device(monkeypatch, backend):
    monkeypatch.setattr(
        r,
        "resolve_diffusion_device_target",
        lambda **_: SimpleNamespace(backend = backend, device = backend),
    )
    monkeypatch.setattr(r, "diffusion_target_is_cpu", lambda: backend == "cpu")


def _binaries(monkeypatch, path):
    """No sd.cpp build at all (``path`` None), as after a failed install or an unrunnable one."""
    monkeypatch.setattr(r, "ensure_sd_cpp_binary", lambda **_: path)
    monkeypatch.setattr(r, "ensure_sd_server_binary", lambda **_: path)
    monkeypatch.setattr(
        r, "SdCppEngine", lambda **_: SimpleNamespace(version = lambda: "v" if path else None)
    )


FLUX3 = "flux.3-klein"


def test_a_failed_native_route_refuses_the_cpu_gguf_load(monkeypatch):
    _device(monkeypatch, "cpu")
    _binaries(monkeypatch, None)
    unloads = []
    with pytest.raises(
        ValueError, match = "native engine could not start: native sd.cpp binary unavailable"
    ):
        r.select_and_activate_engine(
            detect_family(FLUX3), model_kind = "gguf", before_fallback = lambda: unloads.append(1)
        )
    assert unloads == []  # refused before anything unloads


def test_the_plan_refuses_what_the_load_refuses(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_INSTALL", "0")
    _device(monkeypatch, "cpu")
    _binaries(monkeypatch, None)
    with pytest.raises(ValueError, match = "automatic install is off"):
        r.predict_engine(detect_family(FLUX3), model_kind = "gguf")


def test_a_runnable_build_still_routes_native(monkeypatch):
    _device(monkeypatch, "cpu")
    _binaries(monkeypatch, "/opt/sd-cli")
    fam = detect_family(FLUX3)
    for engine in (None, "auto", "sd_cpp"):
        assert r.predict_engine(fam, model_kind = "gguf", engine = engine) == ENGINE_SD_CPP
        r.select_and_activate_engine(fam, model_kind = "gguf", engine = engine)
        assert r.active_engine_name() == ENGINE_SD_CPP


@pytest.mark.parametrize("binary", [None, "/opt/sd-cli"])
def test_an_explicit_diffusers_pick_is_refused_on_torch_cpu(monkeypatch, binary):
    _device(monkeypatch, "cpu")
    _binaries(monkeypatch, binary)
    fam = detect_family(FLUX3)
    for call in (r.predict_engine, r.select_and_activate_engine, r.refuse_engine_pick):
        with pytest.raises(ValueError, match = "choose Auto or Native"):
            call(fam, model_kind = "gguf", engine = "diffusers")


@pytest.mark.parametrize(
    "env, value, match",
    [
        ("UNSLOTH_DIFFUSION_ENGINE", "diffusers", "UNSLOTH_DIFFUSION_ENGINE=diffusers; unset it"),
        ("UNSLOTH_DIFFUSION_SD_CPP", "0", "UNSLOTH_DIFFUSION_SD_CPP=0"),
    ],
)
def test_the_server_diffusers_routes_are_refused_too(monkeypatch, env, value, match):
    monkeypatch.setenv(env, value)
    _device(monkeypatch, "cpu")
    _binaries(monkeypatch, "/opt/sd-cli")
    fam = detect_family(FLUX3)
    with pytest.raises(ValueError, match = match):
        r.predict_engine(fam, model_kind = "gguf")
    with pytest.raises(ValueError, match = match):
        r.select_and_activate_engine(fam, model_kind = "gguf")


@pytest.mark.parametrize(
    "env, value, engine",
    [
        (None, None, None),
        (None, None, "diffusers"),
        ("UNSLOTH_DIFFUSION_ENGINE", "diffusers", None),
        ("UNSLOTH_DIFFUSION_SD_CPP", "0", None),
    ],
)
def test_only_the_kill_switch_restores_the_diffusers_route(monkeypatch, env, value, engine):
    monkeypatch.setenv(r.CPU_GGUF_DIFFUSERS_FALLBACK_ENV, "1")
    # Nothing installed and no install: otherwise the plan counts the install the load would attempt.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_INSTALL", "0")
    if env:
        monkeypatch.setenv(env, value)
    _device(monkeypatch, "cpu")
    _binaries(monkeypatch, None)
    fam = detect_family(FLUX3)
    assert r.predict_engine(fam, model_kind = "gguf", engine = engine) == ENGINE_DIFFUSERS
    r.select_and_activate_engine(fam, model_kind = "gguf", engine = engine)
    assert r.active_engine_name() == ENGINE_DIFFUSERS


@pytest.mark.parametrize(
    "backend, kind, fam_name, engine",
    [
        ("cuda", "gguf", FLUX3, None),  # a GPU torch keeps diffusers for a GGUF
        ("cuda", "gguf", FLUX3, "diffusers"),
        ("cpu", "pipeline", "sdxl", None),  # no native route: the pipeline stays on diffusers
        ("cpu", "gguf", "sdxl", None),  # no native asset mapping: diffusers is the only route
        ("cpu", "gguf", "sdxl", "diffusers"),
    ],
)
def test_other_loads_are_unchanged(monkeypatch, backend, kind, fam_name, engine):
    _device(monkeypatch, backend)
    _binaries(monkeypatch, None)
    fam = detect_family(fam_name)
    assert r.predict_engine(fam, model_kind = kind, engine = engine) == ENGINE_DIFFUSERS
    r.select_and_activate_engine(fam, model_kind = kind, engine = engine)
    assert r.active_engine_name() == ENGINE_DIFFUSERS


def test_every_natively_mapped_image_family_is_covered(monkeypatch):
    from core.inference.diffusion_families import _FAMILIES, family_sd_cpp_supported

    _device(monkeypatch, "cpu")
    _binaries(monkeypatch, None)
    mapped = [f for f in _FAMILIES if family_sd_cpp_supported(f)]
    assert mapped
    for fam in mapped:
        with pytest.raises(ValueError):
            r.select_and_activate_engine(fam, model_kind = "gguf")
        with pytest.raises(ValueError):
            r.refuse_engine_pick(fam, model_kind = "gguf", engine = "diffusers")


def _probe_exit(monkeypatch, returncode):
    """sd-server / sd-cli exist but every start dies with ``returncode`` (the real probe; subprocess stubbed)."""
    import subprocess

    from core.inference import sd_cpp_backend as b

    monkeypatch.setattr(r, "_server_binary_runnable", b._server_binary_runnable)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode = returncode, stdout = b"", stderr = b""),
    )
    monkeypatch.setattr(r, "ensure_sd_cpp_binary", lambda **_: "C:/sd/sd-cli.exe")
    monkeypatch.setattr(r, "ensure_sd_server_binary", lambda **_: "C:/sd/sd-server.exe")
    monkeypatch.setattr(r, "SdCppEngine", lambda **_: SimpleNamespace(version = lambda: None))
    monkeypatch.setattr(r, "note_unlaunchable_accelerator_build", lambda *a, **k: None)


def test_a_missing_dll_names_the_vc_redistributable(monkeypatch):
    _device(monkeypatch, "cpu")
    _probe_exit(monkeypatch, 0xC0000135)
    monkeypatch.setattr(r.sys, "platform", "win32")
    with pytest.raises(ValueError) as info:
        r.select_and_activate_engine(detect_family(FLUX3), model_kind = "gguf")
    msg = str(info.value)
    assert "could not start" in msg and "0xC0000135 (STATUS_DLL_NOT_FOUND)" in msg
    assert "Visual C++ Redistributable" in msg and "vc_redist.x64.exe" in msg
    assert "MSVCP140.dll" in msg and "VCOMP140.dll" in msg


def test_a_failed_download_is_told_apart_from_a_dll_failure(monkeypatch):
    from core.inference import sd_cpp_backend as b

    _device(monkeypatch, "cpu")
    _binaries(monkeypatch, None)

    def _install_fails(**_):
        b.note_native_start_failure(
            b.NATIVE_FAILURE_INSTALL, "<urlopen error [Errno 11001] getaddrinfo failed>"
        )
        return None

    monkeypatch.setattr(r, "ensure_sd_server_binary", _install_fails)
    with pytest.raises(ValueError) as info:
        r.select_and_activate_engine(detect_family(FLUX3), model_kind = "gguf")
    msg = str(info.value)
    assert "could not be installed" in msg and "getaddrinfo failed" in msg and "github.com" in msg
    assert "Visual C++" not in msg


def test_a_real_install_failure_is_recorded_as_one(monkeypatch, tmp_path):
    from core.inference import sd_cpp_backend as b

    monkeypatch.setattr(b, "find_sd_server_binary", lambda: None)
    monkeypatch.setattr(b, "find_sd_cpp_binary", lambda: None)

    def _boom(*_a, **_k):
        raise RuntimeError("HTTP Error 403: rate limit exceeded")

    monkeypatch.setattr(b, "_install_down_the_ladder", _boom)
    assert b.ensure_sd_server_binary(allow_install = True, accelerator = "cpu") is None
    assert b.last_native_start_failure() == (
        b.NATIVE_FAILURE_INSTALL,
        "HTTP Error 403: rate limit exceeded",
    )


@pytest.mark.parametrize(
    "returncode, kind",
    [
        (0xC0000135, "dll"),
        (0xC0000139, "dll"),
        (0xC0000142, "dll"),
        (-1073741515, "dll"),  # the signed spelling of 0xC0000135
        (127, "exec"),
        (-4, "exec"),  # SIGILL
    ],
)
def test_the_start_probe_classifies_loader_failures(returncode, kind):
    from core.inference import sd_cpp_backend as b

    assert b.classify_start_status("sd-cli.exe", returncode)[0] == kind
    assert b.classify_start_status("sd-cli.exe", 0) is None
    assert b.classify_start_status("sd-cli.exe", 1) is None


def test_status_tells_the_page_to_hide_the_diffusers_choice(monkeypatch):
    _device(monkeypatch, "cpu")
    assert r.engine_choice_status({"loaded": False})["gguf_diffusers_allowed"] is False
    monkeypatch.setenv(r.CPU_GGUF_DIFFUSERS_FALLBACK_ENV, "1")
    assert r.engine_choice_status({"loaded": False})["gguf_diffusers_allowed"] is True
    monkeypatch.delenv(r.CPU_GGUF_DIFFUSERS_FALLBACK_ENV)
    _device(monkeypatch, "cuda")
    assert r.engine_choice_status({"loaded": False})["gguf_diffusers_allowed"] is True


@pytest.mark.parametrize(
    "studio, cuda, xpu, mps, want",
    [
        ("CUDA", True, False, False, False),
        ("CUDA", False, False, False, True),
        ("XPU", False, True, False, False),
        ("MLX", False, False, True, False),
        ("CPU", False, False, True, False),
        ("CPU", False, False, False, True),
        ("CPU", True, False, False, False),
        ("CPU", False, True, False, False),
    ],
)
def test_the_status_poll_check_agrees_with_the_target_without_its_probes(
    monkeypatch, studio, cuda, xpu, mps, want
):
    """Same answer as ``resolve_diffusion_device_target`` without its dtype probes (they initialise CUDA / MPS)."""
    import torch

    from core.inference import diffusion_device as dd
    from utils import hardware
    from utils.hardware import hardware as hw

    probed = []
    monkeypatch.setattr(hardware, "get_device", lambda: getattr(hw.DeviceType, studio))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda *_a: probed.append("cuda") or (9, 0)
    )
    monkeypatch.setattr(
        torch,
        "xpu",
        SimpleNamespace(is_available = lambda: xpu, is_bf16_supported = lambda: True),
        raising = False,
    )
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    monkeypatch.setattr(
        dd, "_mps_or_cpu_target", lambda t: SimpleNamespace(backend = "mps" if mps else "cpu")
    )
    assert dd.diffusion_target_is_cpu() is want
    assert probed == []
    assert (dd.resolve_diffusion_device_target().backend == "cpu") is want


# ── CPU-only hosts: no FP8 / INT8 / bf16 rows for a family with a GGUF route ──────────────────────────────────────

_CATALOG = (
    __import__("pathlib").Path(__file__).resolve().parents[2]
    / "frontend/src/features/model-picker/components/model-selector/model-catalog.ts"
)


def _catalog_groups(scope):
    """(canonical id, artifact (kind, repo) pairs, nativeGgufRoute) per catalog group of ``scope``."""
    import re

    src = _CATALOG.read_text(encoding = "utf-8")
    out = []
    for chunk in src.split("canonicalId:")[1:]:
        m = re.match(r'\s*"([^"]+)"', chunk)
        if not m:
            continue
        body = chunk.split("canonicalId:")[0]
        found = re.search(r'scope:\s*"(\w+)"', body)
        if not found or found.group(1) != scope:
            continue
        arts = re.findall(r'(\w+)\(\s*"([\w.-]+/[\w.-]+)"', body[body.find("artifacts:") :])
        out.append((m.group(1), arts, "nativeGgufRoute: true" in body))
    return out


@pytest.mark.parametrize("kind", ["pipeline", "single_file"])
def test_a_cpu_dense_pick_of_a_family_with_a_gguf_route_is_refused(monkeypatch, kind):
    _device(monkeypatch, "cpu")
    _binaries(monkeypatch, "/opt/sd-cli")
    fam = detect_family(FLUX3)
    for call in (r.predict_engine, r.select_and_activate_engine):
        with pytest.raises(ValueError, match = "runs only as a GGUF on the native engine"):
            call(fam, model_kind = kind)


def test_a_cpu_dense_pick_without_a_listed_gguf_names_no_missing_rows(monkeypatch):
    """Krea 2 has no GGUF row: the refusal must not tell the user to pick one, the FLUX.3 refusal still does."""
    _device(monkeypatch, "cpu")
    _binaries(monkeypatch, "/opt/sd-cli")
    with pytest.raises(ValueError) as krea:
        r.refuse_cpu_dense_image_pick(detect_family("krea/Krea-2-Turbo"), "pipeline")
    assert "runs only as a GGUF on the native engine" in str(krea.value)
    assert "Pick one of its GGUF files" not in str(krea.value)
    assert "has no Krea 2 GGUF yet" in str(krea.value)
    with pytest.raises(ValueError, match = "Pick one of its GGUF files"):
        r.refuse_cpu_dense_image_pick(detect_family(FLUX3), "pipeline")


def test_no_curated_gguf_set_matches_the_catalog():
    missing = set()
    for canonical, arts, native in _catalog_groups("image"):
        if native and not any(kind == "gguf" for kind, _repo in arts):
            fam = detect_family(canonical)
            assert fam is not None, canonical
            missing.add(fam.name)
    assert missing == set(r.CPU_NO_CURATED_GGUF_IMAGE_FAMILIES)


@pytest.mark.parametrize(
    "backend, fam_name, env",
    [
        ("cpu", "sdxl", None),  # no GGUF route: keeps its rows
        ("cpu", "lumina-2", None),
        ("cuda", FLUX3, None),  # a GPU keeps the FP8 / bf16 rows
        ("cpu", FLUX3, "1"),  # the developer kill switch
    ],
)
def test_dense_picks_elsewhere_are_unchanged(monkeypatch, backend, fam_name, env):
    if env:
        monkeypatch.setenv(r.CPU_GGUF_DIFFUSERS_FALLBACK_ENV, env)
    _device(monkeypatch, backend)
    fam = detect_family(fam_name)
    assert r.predict_engine(fam, model_kind = "pipeline") == ENGINE_DIFFUSERS
    r.select_and_activate_engine(fam, model_kind = "pipeline")
    assert r.active_engine_name() == ENGINE_DIFFUSERS


def test_the_backend_refuses_exactly_the_image_rows_the_cpu_picker_hides():
    from core.inference.diffusion_families import detect_family_for_pick, family_sd_cpp_supported

    groups = _catalog_groups("image")
    assert len(groups) > 15
    for canonical, arts, native_route in groups:
        fam = detect_family_for_pick(canonical, None, None)
        hidden = native_route or any(k == "gguf" for k, _ in arts)
        assert hidden == bool(fam and family_sd_cpp_supported(fam)), canonical


def test_the_video_gguf_only_set_matches_the_catalog():
    from core.inference.video import CPU_GGUF_ONLY_VIDEO_REPOS
    curated = {
        repo.lower() for _c, arts, _n in _catalog_groups("video") for k, repo in arts if k == "gguf"
    }
    assert curated == set(CPU_GGUF_ONLY_VIDEO_REPOS)


def _video_device(monkeypatch, backend):
    import core.inference.diffusion_device as devmod
    monkeypatch.setattr(
        devmod,
        "resolve_diffusion_device_target",
        lambda **_: SimpleNamespace(backend = backend, device = backend),
    )


def test_a_cpu_dense_video_pick_of_a_family_with_a_gguf_row_is_refused(monkeypatch):
    from core.inference.video import refuse_cpu_dense_video_pick
    from core.inference.video_families import _FAMILIES as VIDEO_FAMILIES

    _video_device(monkeypatch, "cpu")
    by_repo = {str(f.gguf_repo or "").lower(): f for f in VIDEO_FAMILIES}
    picks = {
        "unsloth/ltx-2.5-gguf": ("Lightricks/LTX-2.5-Diffusers",),
        "unsloth/ltx-2.3-gguf": ("Lightricks/LTX-2.3", None, "ltx-2.3-22b-distilled.safetensors"),
        "unsloth/minimax-h3-gguf": ("MiniMaxAI/MiniMax-H3",),
    }
    for repo, names in picks.items():
        fam = by_repo[repo]
        for kind in ("pipeline", "single_file"):
            with pytest.raises(ValueError, match = "loads only as a GGUF here"):
                refuse_cpu_dense_video_pick(fam, kind, names)
        refuse_cpu_dense_video_pick(fam, "gguf", names)
    # LTX-2.0 shares the ltx-2 family with 2.3 but has no GGUF row: the picker keeps its pipeline, so the load does too.
    ltx2 = by_repo["unsloth/ltx-2.3-gguf"]
    for names in (
        ("Lightricks/LTX-2",),
        ("Lightricks/LTX-2", "Lightricks/LTX-2", "ltx-2-19b-dev.safetensors"),
    ):
        refuse_cpu_dense_video_pick(ltx2, "pipeline", names)
    for names in (
        ("unsloth/LTX-2.3-FP8",),
        ("x/y", None, "ltx2.3-distilled.safetensors"),
        ("me/ltx23-int8",),
    ):
        with pytest.raises(ValueError, match = "loads only as a GGUF here"):
            refuse_cpu_dense_video_pick(ltx2, "single_file", names)
    # Wan2.2 TI2V-5B has a GGUF repo but no curated GGUF row: its pipeline row stays, so it is not refused.
    refuse_cpu_dense_video_pick(by_repo["unsloth/wan2.2-ti2v-5b-gguf"], "pipeline")
    _video_device(monkeypatch, "cuda")
    refuse_cpu_dense_video_pick(by_repo["unsloth/ltx-2.5-gguf"], "pipeline")
    _video_device(monkeypatch, "cpu")
    monkeypatch.setenv(r.CPU_GGUF_DIFFUSERS_FALLBACK_ENV, "1")
    refuse_cpu_dense_video_pick(by_repo["unsloth/ltx-2.5-gguf"], "pipeline")


def test_the_video_load_validation_runs_the_refusal(monkeypatch):
    from core.inference.video import get_video_backend

    _video_device(monkeypatch, "cpu")
    with pytest.raises(ValueError, match = "loads only as a GGUF here"):
        get_video_backend().validate_load_request(
            "Lightricks/LTX-2.5-Diffusers", model_kind = "pipeline"
        )
    # LTX-2.0's pipeline row is still offered on a CPU-only host, so its load is not refused for having no GGUF.
    try:
        get_video_backend().validate_load_request("Lightricks/LTX-2", model_kind = "pipeline")
    except ValueError as exc:
        assert "loads only as a GGUF here" not in str(exc)
