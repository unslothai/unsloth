# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-request MLX / diffusers routing on Apple Silicon, with fake engines (no mlx, torch or GPU needed)."""

from __future__ import annotations

import inspect
import threading
import time
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from core.inference import diffusion_families as families
from core.inference import mlx_diffusion_routing as routing
from core.inference.diffusion_families import (
    DIFFUSION_CANCELLED_MSG,
    DIFFUSION_NOT_LOADED_MSG,
    DiffusionModelReplacedError,
    load_identity,
)

GIB = 1024**3
_FAM = SimpleNamespace(name = "z-image")


class _Engine:
    """The duck-typed engine surface the routed engine drives."""

    def __init__(self, name: str, log: list, *, base: str) -> None:
        self.name = name
        self.log = log
        self.base = base
        self._state: Any = None
        self._load_error: Optional[str] = None
        self._loading = False
        self.load_kwargs: Optional[dict] = None
        self.fail_next_load: Optional[str] = None
        self.hang_load = False
        self.cancelled = threading.Event()

    def begin_load(self, repo_id: str, **kwargs: Any) -> dict:
        self.log.append((self.name, "load"))
        self.load_kwargs = {"repo_id": repo_id, **kwargs}
        self._load_error = None
        if self.fail_next_load:
            self._load_error, self.fail_next_load = self.fail_next_load, None
            return self.status()
        if self.hang_load:
            self._loading = True
            return self.status()
        self._state = SimpleNamespace(
            family = _FAM,
            repo_id = repo_id,
            timings = {"quant_plan": {"planned_bytes": 6 * GIB}},
            gguf_path = None,
            convert_dir = None,
        )
        return self.status()

    def load_progress(self) -> dict:
        if self._load_error:
            return {"phase": "error", "error": self._load_error}
        if self._loading:
            return {"phase": "downloading"}
        return {"phase": "ready" if self._state is not None else None}

    def unload(self, *, expected_account: Optional[str] = None) -> dict:
        self.log.append((self.name, "unload"))
        self._state = None
        self._loading = False
        return self.status()

    def status(self) -> dict:
        if self._state is None:
            return {"loaded": False, "engine": self.name, "workflows": []}
        return {
            "loaded": True,
            "engine": self.name,
            "repo_id": self._state.repo_id,
            "base_repo": self.base,
            "family": "z-image",
            "workflows": ["txt2img"] if self.name == "mlx" else ["txt2img", "img2img"],
        }

    def generate_progress(self) -> dict:
        return {"active": False, "engine": self.name}

    def cancel_generate(self, expected_account: Optional[str] = None) -> bool:
        self.cancelled.set()
        return True

    def loaded_repo_ids(self) -> tuple:
        return (self._state.repo_id,) if self._state is not None else ()

    def _run(self, expected_load, kwargs) -> dict:
        if self._state is None:
            raise RuntimeError(DIFFUSION_NOT_LOADED_MSG)
        if expected_load is not None and expected_load != load_identity(
            self._state.repo_id, self.base, "z-image"
        ):
            raise DiffusionModelReplacedError(expected_load, expected_load)
        self.log.append((self.name, "generate"))
        return {"images": [f"{self.name}-image"], "seed": kwargs.get("seed"), "kwargs": kwargs}


class _Mlx(_Engine):
    def generate(
        self,
        *,
        prompt: str,
        width: Optional[int] = 1024,
        height: Optional[int] = 1024,
        seed: Optional[int] = None,
        init_image: Optional[str] = None,
        workflow: Optional[str] = None,
        loras: Any = None,
        controlnet: Any = None,
        prompts: Any = None,
        seeds: Any = None,
        upscale: Optional[float] = None,
        expected_load: Any = None,
        hires: Any = None,
    ) -> dict:
        if init_image is not None or workflow not in (None, "txt2img"):
            raise ValueError("The MLX engine runs text-to-image only")
        return self._run(
            expected_load, {"prompt": prompt, "width": width, "height": height, "seed": seed}
        )


class _Diffusers(_Engine):
    def generate(
        self,
        *,
        prompt: str,
        width: Optional[int] = 1024,
        height: Optional[int] = 1024,
        seed: Optional[int] = None,
        init_image: Optional[str] = None,
        mask_image: Optional[str] = None,
        strength: Optional[float] = None,
        workflow: Optional[str] = None,
        loras: Any = None,
        controlnet: Any = None,
        prompts: Any = None,
        seeds: Any = None,
        upscale: Optional[float] = None,
        expected_load: Any = None,
        hires: Any = None,
        live_preview: Optional[bool] = None,
        sampler: Optional[str] = None,
        schedule_type: Optional[str] = None,
        sd_controls: Any = None,
        shift: Optional[float] = None,
        base_shift: Optional[float] = None,
        max_shift: Optional[float] = None,
        upscale_model: Optional[str] = None,
        tile_vae: bool = False,
    ) -> dict:
        return self._run(
            expected_load,
            {"prompt": prompt, "init_image": init_image, "seed": seed, "sampler": sampler},
        )


@pytest.fixture
def rig(monkeypatch):
    monkeypatch.delenv(routing.COEXIST_ENV, raising = False)
    monkeypatch.setattr(families, "family_pipeline_available", lambda _f: True)
    log: list = []
    mlx = _Mlx("mlx", log, base = "Tongyi-MAI/Z-Image-Turbo")
    # diffusers resolves the base its own way: identity checks must not depend on the two agreeing.
    diffusers = _Diffusers("diffusers", log, base = "unsloth/Z-Image-Turbo")
    fit = {"coexist": False}
    engine = routing.RoutedMlxEngine(
        mlx_factory = lambda: mlx,
        diffusers_factory = lambda: diffusers,
        budget_fn = lambda: 5 * GIB,
        capabilities_fn = lambda fam: {
            "workflows": ["txt2img", "img2img", "upscale", "inpaint", "outpaint"],
            "hires": {"available": True},
        },
        poll_s = 0.01,
    )
    monkeypatch.setattr(engine, "_engines_fit_together", lambda: fit["coexist"])
    engine.begin_load(
        "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z-image-turbo-Q4_0.gguf",
        local_files_only = True,
        hf_token = None,
        model_kind = "gguf",
    )
    log.clear()
    return SimpleNamespace(engine = engine, mlx = mlx, diffusers = diffusers, log = log, fit = fit)


def _txt2img(engine, **extra):
    return engine.generate(prompt = "a fox", width = 512, height = 512, seed = 42, **extra)


def _img2img(engine, **extra):
    return engine.generate(
        prompt = "a fox",
        width = 512,
        height = 512,
        seed = 42,
        init_image = "data:image/png;base64,AAAA",
        strength = 0.6,
        workflow = "img2img",
        **extra,
    )


# ── the routing decision ─────────────────────────────────────────────────────


def test_plain_txt2img_runs_on_mlx_and_never_loads_diffusers(rig):
    result = _txt2img(rig.engine, sampler = "default", schedule_type = "automatic", live_preview = True)
    assert result["engine"] == "mlx"
    assert result["images"] == ["mlx-image"]
    assert rig.log == [("mlx", "generate")]
    # Options MLX does not take never reach it.
    assert set(result["kwargs"]) == {"prompt", "width", "height", "seed"}
    routing_status = rig.engine.status()["engine_routing"]
    assert routing_status["resident"] == ["mlx"]
    assert routing_status["last_served_by"] == "mlx"
    assert routing_status["txt2img_engine"] == "mlx"


def test_img2img_after_txt2img_unloads_mlx_first_then_serves_from_diffusers(rig):
    assert _txt2img(rig.engine)["engine"] == "mlx"
    result = _img2img(rig.engine)
    assert result["engine"] == "diffusers"
    assert result["kwargs"]["init_image"] == "data:image/png;base64,AAAA"
    # MLX leaves BEFORE diffusers loads: a 16 GB Mac never holds both.
    assert rig.log == [
        ("mlx", "generate"),
        ("mlx", "unload"),
        ("diffusers", "load"),
        ("diffusers", "generate"),
    ]
    # The fallback loads the same pick with the same load options.
    assert rig.diffusers.load_kwargs == {
        "repo_id": "unsloth/Z-Image-Turbo-GGUF",
        "gguf_filename": "z-image-turbo-Q4_0.gguf",
        "local_files_only": True,
        "hf_token": None,
        "model_kind": "gguf",
    }
    status = rig.engine.status()
    assert status["engine"] == "diffusers"
    assert status["engine_routing"]["resident"] == ["diffusers"]
    assert status["engine_routing"]["last_served_by"] == "diffusers"
    assert "img2img" in status["engine_routing"]["last_reason"]
    assert status["engine_routing"]["engines_fit_together"] is False


def test_after_a_fallback_that_did_not_fit_txt2img_stays_on_diffusers(rig):
    _img2img(rig.engine)
    rig.log.clear()
    result = _txt2img(rig.engine)
    assert result["engine"] == "diffusers"
    # No reload ping-pong.
    assert rig.log == [("diffusers", "generate")]
    routing_status = rig.engine.status()["engine_routing"]
    assert routing_status["txt2img_engine"] == "diffusers"
    assert "load the model again" in routing_status["last_reason"]


def test_when_both_fit_both_stay_resident_and_txt2img_returns_to_mlx(rig):
    rig.fit["coexist"] = True
    assert _img2img(rig.engine)["engine"] == "diffusers"
    assert ("mlx", "unload") not in rig.log
    assert _txt2img(rig.engine)["engine"] == "mlx"
    assert _img2img(rig.engine)["engine"] == "diffusers"
    assert rig.log.count(("diffusers", "load")) == 1
    assert rig.engine.status()["engine_routing"]["resident"] == ["mlx", "diffusers"]


@pytest.mark.parametrize(
    "extra, needle",
    [
        ({"init_image": "x", "workflow": "img2img"}, "img2img"),
        ({"mask_image": "m", "init_image": "x"}, "inpainting"),
        ({"workflow": "edit"}, "edit"),
        ({"hires": object()}, "Hires"),
        ({"upscale_model": "4x"}, "upscaler"),
        ({"upscale": 2.0}, "upscaling"),
        ({"prompts": ["a", "b"]}, "lists"),
        ({"seeds": [1, 2]}, "lists"),
        ({"loras": [("l", 1.0)]}, "LoRA"),
        ({"controlnet": ("c", "img", "canny", 0.8, 0.0, 1.0)}, "ControlNet"),
        ({"sampler": "euler"}, "sampler"),
        ({"schedule_type": "karras"}, "sampler"),
        ({"sd_controls": SimpleNamespace(requested = lambda: True)}, "prompt weighting"),
        ({"shift": 3.0}, "shift"),
        ({"tile_vae": True}, "tile_vae"),
    ],
)
def test_every_feature_mlx_lacks_routes_to_diffusers(rig, extra, needle):
    blocker = routing.request_blocker(
        {"prompt": "p", **extra}, family = "z-image", mlx_params = frozenset({"prompt"})
    )
    assert blocker and needle in blocker
    assert rig.engine.generate(prompt = "p", **extra)["engine"] == "diffusers"


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"workflow": "txt2img"},
        {"upscale": 1.0},
        {"controlnet": ("c", "img", "canny", 0.0, 0.0, 1.0)},
        {"sampler": "default", "schedule_type": "automatic"},
        {"sd_controls": SimpleNamespace(requested = lambda: False)},
        {"live_preview": True, "mask_blur": 4},
        {"tile_vae": False},
    ],
)
def test_neutral_values_stay_on_mlx(rig, extra):
    assert (
        routing.request_blocker({"prompt": "p", **extra}, mlx_params = frozenset({"prompt"})) is None
    )
    assert rig.engine.generate(prompt = "p", **extra)["engine"] == "mlx"


@pytest.mark.parametrize(
    "blocker, mlx_resident, diffusers_resident, fallback, coexist, expected",
    [
        (None, True, False, True, False, ("mlx", False, False)),
        (None, True, True, True, True, ("mlx", False, False)),
        (None, False, True, True, False, ("diffusers", False, False)),
        (None, False, False, True, False, ("mlx", False, False)),
        ("img2img", True, False, True, False, ("diffusers", True, True)),
        ("img2img", True, False, True, True, ("diffusers", True, False)),
        ("img2img", True, True, True, True, ("diffusers", False, False)),
        ("img2img", False, True, True, False, ("diffusers", False, False)),
        # No diffusers pipeline for the family here: MLX answers with its own refusal.
        ("img2img", True, False, False, False, ("mlx", False, False)),
        # A load in flight: nothing to fall back from yet.
        ("img2img", False, False, True, False, ("mlx", False, False)),
    ],
)
def test_decide_route_table(blocker, mlx_resident, diffusers_resident, fallback, coexist, expected):
    route = routing.decide_route(
        blocker = blocker,
        mlx_resident = mlx_resident,
        diffusers_resident = diffusers_resident,
        fallback_available = fallback,
        coexist = coexist,
    )
    assert (route.engine, route.load_diffusers, route.unload_mlx) == expected


def test_a_family_diffusers_cannot_build_keeps_the_mlx_refusal(rig, monkeypatch):
    monkeypatch.setattr(families, "family_pipeline_available", lambda _f: False)
    with pytest.raises(ValueError, match = "text-to-image only"):
        _img2img(rig.engine)
    assert ("diffusers", "load") not in rig.log
    # And the status does not advertise workflows nothing here can run.
    assert rig.engine.status()["workflows"] == ["txt2img"]


def test_status_advertises_the_fallback_workflows_while_mlx_is_resident(rig):
    status = rig.engine.status()
    assert status["engine"] == "mlx"
    assert "img2img" in status["workflows"]
    assert status["hires"] == {"available": True}


# ── memory budget ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "mlx_gib, diffusers_gib, budget_gib, expected",
    [
        (6, 4, 10.7, True),
        (7.5, 12, 10.7, False),  # a 16 GB Mac with Z-Image Q4: never both
        (2, 3, 5.01, True),
        (0, 3, 5.01, False),  # unknown MLX size (RAM-rule plan)
        (3, 0, 5.01, False),  # unreadable diffusers size
        (3, 3, 0, False),  # unknown working set
    ],
)
def test_engines_coexist_only_when_both_fit(
    monkeypatch, mlx_gib, diffusers_gib, budget_gib, expected
):
    monkeypatch.delenv(routing.COEXIST_ENV, raising = False)
    assert (
        routing.engines_coexist(int(mlx_gib * GIB), int(diffusers_gib * GIB), int(budget_gib * GIB))
        is expected
    )


@pytest.mark.parametrize("value, expected", [("1", True), ("0", False)])
def test_coexist_env_overrides(monkeypatch, value, expected):
    monkeypatch.setenv(routing.COEXIST_ENV, value)
    assert routing.engines_coexist(100 * GIB, 100 * GIB, GIB) is expected
    assert routing.engines_coexist(1, 1, 100 * GIB) is expected


def test_diffusers_size_is_the_gguf_plus_bf16_text_encoder_and_vae(tmp_path):
    import json
    import struct

    def _st(folder, shape):
        folder.mkdir()
        header = json.dumps(
            {"w": {"dtype": "BF16", "shape": shape, "data_offsets": [0, 0]}}
        ).encode()
        (folder / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)

    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"\0" * 1000)
    _st(tmp_path / "text_encoder", [10, 10])
    _st(tmp_path / "vae", [5])
    assert routing.diffusers_resident_bytes(str(gguf), str(tmp_path)) == 1000 + 200 + 10
    assert routing.diffusers_resident_bytes(None, str(tmp_path)) == 0
    assert routing.diffusers_resident_bytes(str(tmp_path / "missing.gguf"), str(tmp_path)) == 0


def test_the_real_fit_check_reads_the_mlx_plan_and_the_budget(tmp_path, monkeypatch):
    log: list = []
    mlx = _Mlx("mlx", log, base = "b")
    mlx.begin_load("r")
    mlx._state.gguf_path = "g"
    mlx._state.convert_dir = "d"
    monkeypatch.setattr(routing, "diffusers_resident_bytes", lambda g, d: 4 * GIB)
    monkeypatch.delenv(routing.COEXIST_ENV, raising = False)
    small = routing.RoutedMlxEngine(mlx_factory = lambda: mlx, budget_fn = lambda: 9 * GIB)
    big = routing.RoutedMlxEngine(mlx_factory = lambda: mlx, budget_fn = lambda: 11 * GIB)
    assert small._engines_fit_together() is False  # 6 + 4 > 9
    assert big._engines_fit_together() is True


# ── failure, cancel, replacement, unload ─────────────────────────────────────


def test_a_failed_fallback_reports_the_error_and_restores_mlx(rig):
    rig.diffusers.fail_next_load = "does not fit"
    with pytest.raises(RuntimeError, match = "does not fit"):
        _img2img(rig.engine)
    assert rig.log == [("mlx", "unload"), ("diffusers", "load"), ("mlx", "load")]
    assert rig.engine.status()["engine_routing"]["resident"] == ["mlx"]
    assert _txt2img(rig.engine)["engine"] == "mlx"


def test_cancel_during_the_fallback_load_stops_it(rig):
    rig.diffusers.hang_load = True
    errors = []

    def _run():
        try:
            _img2img(rig.engine)
        except RuntimeError as exc:
            errors.append(str(exc))

    worker = threading.Thread(target = _run)
    worker.start()
    deadline = time.time() + 5
    while rig.engine._fallback_cancel is None and time.time() < deadline:
        time.sleep(0.01)
    progress = rig.engine.generate_progress()
    assert progress["active"] is True and progress["step"] == 0
    assert rig.engine.cancel_generate() is True
    worker.join(5)
    assert errors == [DIFFUSION_CANCELLED_MSG]
    assert ("diffusers", "unload") in rig.log


def test_cancel_during_the_fallback_load_checks_the_owner(rig):
    rig.engine._fallback_cancel = threading.Event()
    rig.engine._active_generate_account = "owner"
    assert rig.engine.cancel_generate(expected_account = "intruder") is False
    assert rig.engine.cancel_generate(expected_account = "owner") is True


def test_a_replaced_model_is_refused(rig):
    stale = load_identity("someone/else", "x", "z-image")
    with pytest.raises(DiffusionModelReplacedError):
        _txt2img(rig.engine, expected_load = stale)


def test_the_original_identity_is_accepted_after_the_fallback(rig):
    mlx_identity = load_identity(
        "unsloth/Z-Image-Turbo-GGUF", "Tongyi-MAI/Z-Image-Turbo", "z-image"
    )
    assert _txt2img(rig.engine, expected_load = mlx_identity)["engine"] == "mlx"
    # The client still holds the MLX status's identity; diffusers resolved a different base.
    assert _img2img(rig.engine, expected_load = mlx_identity)["engine"] == "diffusers"
    assert _txt2img(rig.engine, expected_load = mlx_identity)["engine"] == "diffusers"


def test_a_new_load_drops_the_old_fallback_and_returns_to_mlx(rig):
    _img2img(rig.engine)
    rig.log.clear()
    rig.engine.begin_load("unsloth/Z-Image-Turbo-GGUF", gguf_filename = "z-image-turbo-Q8_0.gguf")
    assert rig.log == [("diffusers", "unload"), ("mlx", "load")]
    assert _txt2img(rig.engine)["engine"] == "mlx"
    assert rig.engine.status()["engine_routing"]["resident"] == ["mlx"]


def test_unload_frees_both_engines(rig):
    rig.fit["coexist"] = True
    _img2img(rig.engine)
    rig.log.clear()
    status = rig.engine.unload()
    assert rig.log == [("mlx", "unload"), ("diffusers", "unload")]
    assert status["loaded"] is False
    with pytest.raises(RuntimeError, match = DIFFUSION_NOT_LOADED_MSG):
        _img2img(rig.engine)
    assert ("diffusers", "load") not in rig.log


def test_a_plain_diffusers_load_is_never_unloaded_as_a_fallback(rig):
    # The diffusers singleton is shared: one that the routed engine did not load is not its to free.
    rig.diffusers.begin_load("someone/else")
    rig.log.clear()
    rig.engine.unload()
    assert ("diffusers", "unload") not in rig.log


def test_repo_ids_cover_both_engines(rig):
    rig.fit["coexist"] = True
    _img2img(rig.engine)
    assert rig.engine.loaded_repo_ids() == ("unsloth/Z-Image-Turbo-GGUF",)


# ── the route's signature checks ─────────────────────────────────────────────


def test_generate_signature_is_the_union_of_both_engines(rig):
    params = inspect.signature(rig.engine.generate).parameters
    for name in (
        "sampler",
        "sd_controls",
        "shift",
        "hires",
        "upscale_model",
        "live_preview",
        "prompt",
    ):
        assert name in params
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params.values())
    # _sd_controls_kwargs reads the engine off the bound callable.
    assert rig.engine.generate.__self__ is rig.engine
    assert rig.engine.parses_prompt_attention is False


def test_load_blocker_matrix():
    assert routing.load_blocker() is None
    assert routing.load_blocker(transformer_quant = "auto", text_encoder_quant = "none") is None
    assert "LoRA" in routing.load_blocker(loras = [("a", 1.0)])
    assert "VAE" in routing.load_blocker(component_files = {"vae_file": "v"})
    assert "int8" in routing.load_blocker(transformer_quant = "int8")
    assert "fp8" in routing.load_blocker(text_encoder_quant = "fp8")
