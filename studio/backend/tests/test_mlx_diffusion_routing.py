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
        # Each engine's own load epoch, as the real engines keep one (bumped by begin_load / unload).
        self._load_token = 0

    def begin_load(self, repo_id: str, **kwargs: Any) -> dict:
        self._load_token += 1
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

    def unload(
        self,
        *,
        expected_account: Optional[str] = None,
        only_if = None,
    ) -> Optional[dict]:
        # As the real engines: a conditional unload rechecks under the engine's lock first.
        if only_if is not None and not only_if():
            return None
        self.log.append((self.name, "unload"))
        self._load_token += 1
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
        guidance: float = 0.0,
        negative_prompt: Optional[str] = None,
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


def test_plain_txt2img_runs_on_mlx_and_never_loads_diffusers(rig):
    result = _txt2img(rig.engine, sampler = "default", schedule_type = "automatic", live_preview = True)
    assert result["engine"] == "mlx"
    assert result["images"] == ["mlx-image"]
    assert rig.log == [("mlx", "generate")]
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
    assert rig.log == [
        ("mlx", "generate"),
        ("mlx", "unload"),
        ("diffusers", "load"),
        ("diffusers", "generate"),
    ]
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
        ("img2img", True, False, False, False, ("mlx", False, False)),
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
    assert rig.engine.status()["workflows"] == ["txt2img"]


def test_status_advertises_the_fallback_workflows_while_mlx_is_resident(rig):
    status = rig.engine.status()
    assert status["engine"] == "mlx"
    assert "img2img" in status["workflows"]
    assert status["hires"] == {"available": True}


@pytest.mark.parametrize(
    "mlx_gib, diffusers_gib, budget_gib, expected",
    [
        (6, 4, 10.7, True),
        (7.5, 12, 10.7, False),
        (2, 3, 5.01, True),
        (0, 3, 5.01, False),
        (3, 0, 5.01, False),
        (3, 3, 0, False),
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
    assert small._engines_fit_together() is False
    assert big._engines_fit_together() is True


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


def test_a_stopped_generation_ticket_stops_the_fallback_load(rig):
    from core.inference.generation_ticket import GenerationTicket, generation_ticket

    rig.diffusers.hang_load = True
    ticket = GenerationTicket()
    errors = []

    def _run():
        with generation_ticket(ticket):
            try:
                _img2img(rig.engine)
            except RuntimeError as exc:
                errors.append(str(exc))

    worker = threading.Thread(target = _run, daemon = True)
    worker.start()
    deadline = time.time() + 5
    while rig.engine._fallback_cancel is None and time.time() < deadline:
        time.sleep(0.01)
    ticket.stop()
    worker.join(5)
    assert not worker.is_alive()
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


def test_a_conditional_unload_the_load_no_longer_holds_tears_nothing_down(rig):
    # A load token whose load ended: refused before the drain, so the fallback keeps running.
    rig.fit["coexist"] = True
    _img2img(rig.engine)
    rig.log.clear()
    cancel = threading.Event()
    rig.engine._fallback_cancel = cancel
    epoch = rig.engine._epoch
    assert rig.engine.unload(only_if = lambda: False) is None
    assert rig.log == [] and not cancel.is_set() and rig.engine._epoch == epoch


def test_a_conditional_unload_of_the_named_load_frees_both_engines(rig):
    rig.fit["coexist"] = True
    _img2img(rig.engine)
    rig.log.clear()
    status = rig.engine.unload(only_if = lambda: True)
    assert rig.log == [("mlx", "unload"), ("diffusers", "unload")]
    assert status["loaded"] is False


def test_a_conditional_unload_refused_by_the_mlx_recheck_keeps_the_fallback_model(rig):
    # The named load ended between the router's check and the MLX engine's recheck under its lock.
    rig.fit["coexist"] = True
    _img2img(rig.engine)
    rig.log.clear()
    answers = iter([True, False])
    assert rig.engine.unload(only_if = lambda: next(answers)) is None
    assert ("diffusers", "unload") not in rig.log


def test_the_routed_load_epoch_survives_the_fallback(rig):
    # media_load_ids reads _load_token as the load's epoch: a diffusers fallback serves the same load,
    # so the epoch must not jump to the other engine's own counter.
    from core.inference import media_load_ids
    from hub.services.models import account_access

    media_load_ids._reset_for_tests()
    try:
        rig.mlx._load_token = 5  # the two engines' own counters differ
        with account_access.media_load_lock:
            prior = media_load_ids.admitting("diffusion", rig.engine, "t1")
            rig.engine.begin_load(
                "unsloth/Z-Image-Turbo-GGUF",
                gguf_filename = "z-image-turbo-Q4_0.gguf",
                local_files_only = True,
                hf_token = None,
                model_kind = "gguf",
            )
            media_load_ids.admitted("diffusion", rig.engine, "t1", prior)
        epoch = rig.engine._load_token
        assert media_load_ids.current_id("diffusion", rig.engine) == "t1"
        _img2img(rig.engine)  # served by the diffusers fallback
        assert (
            rig.engine._primary() is rig.diffusers
            and rig.diffusers._load_token != rig.mlx._load_token
        )
        assert rig.engine._load_token == epoch
        assert media_load_ids.current_id("diffusion", rig.engine) == "t1"
        rig.engine.unload()
        assert rig.engine._load_token == epoch + 1
    finally:
        media_load_ids._reset_for_tests()


def test_a_refused_load_keeps_the_routed_load_epoch(rig, monkeypatch):
    # MLX refuses a second load while one is in flight; that load's admitted epoch must still match.
    epoch = rig.engine._load_token

    def refuse(repo_id, **kwargs):
        raise RuntimeError("A diffusion load is already in progress.")

    monkeypatch.setattr(rig.mlx, "begin_load", refuse)
    with pytest.raises(RuntimeError):
        rig.engine.begin_load(
            "unsloth/Z-Image-Turbo-GGUF",
            gguf_filename = "z-image-turbo-Q4_0.gguf",
            local_files_only = True,
            hf_token = None,
            model_kind = "gguf",
        )
    assert rig.engine._load_token == epoch


def test_a_scoped_unload_never_cancels_another_accounts_fallback(rig):
    from core.inference.gpu_arbiter import GpuBusyForAnotherAccountError

    cancel = threading.Event()
    rig.engine._fallback_cancel = cancel
    rig.engine._active_generate_account = "owner"
    epoch = rig.engine._epoch
    with pytest.raises(GpuBusyForAnotherAccountError):
        rig.engine.unload(expected_account = "intruder")
    assert not cancel.is_set() and rig.engine._epoch == epoch
    assert ("mlx", "unload") not in rig.log


@pytest.mark.parametrize("field", ["transformer_quant", "text_encoder_quant"])
@pytest.mark.parametrize("value", ["none", "off", "NONE"])
def test_a_pinned_original_precision_keeps_the_load_on_diffusers(field, value):
    blocker = routing.load_blocker(**{field: value})
    assert blocker and value in blocker


def test_a_lora_request_stays_on_mlx_for_a_family_whose_build_applies_loras(rig):
    assert routing.mlx_capabilities("z-image").loras is True
    assert (
        routing.request_blocker(
            {"prompt": "p", "loras": [("l", 1.0)]},
            family = "z-image",
            mlx_params = frozenset({"prompt", "loras"}),
        )
        is None
    )
    assert rig.engine.generate(prompt = "p", loras = [("l", 1.0)])["engine"] == "mlx"
    assert "LoRA" in routing.request_blocker({"prompt": "p", "loras": [("l", 1.0)]}, family = "other")


def test_unload_forwards_the_account_scope_to_the_fallback(rig):
    rig.fit["coexist"] = True
    _img2img(rig.engine)
    seen = []
    real = rig.diffusers.unload

    unscoped = object()

    def _scoped_unload(*, expected_account = unscoped):
        seen.append(expected_account)
        return real() if expected_account is unscoped else real(expected_account = expected_account)

    rig.diffusers.unload = _scoped_unload
    rig.engine.unload(expected_account = "owner")
    assert seen == ["owner"]


def test_a_plain_diffusers_load_is_never_unloaded_as_a_fallback(rig):
    rig.diffusers.begin_load("someone/else")
    rig.log.clear()
    rig.engine.unload()
    assert ("diffusers", "unload") not in rig.log


def test_repo_ids_cover_both_engines(rig):
    rig.fit["coexist"] = True
    _img2img(rig.engine)
    assert rig.engine.loaded_repo_ids() == ("unsloth/Z-Image-Turbo-GGUF",)


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
    assert rig.engine.generate.__self__ is rig.engine
    assert rig.engine.parses_prompt_attention is False


def test_load_blocker_matrix():
    assert routing.load_blocker() is None
    assert routing.load_blocker(transformer_quant = "auto", text_encoder_quant = "default") is None
    assert "VAE" in routing.load_blocker(component_files = {"vae_file": "v"})
    assert "int8" in routing.load_blocker(transformer_quant = "int8")
    assert "fp8" in routing.load_blocker(text_encoder_quant = "fp8")


def test_unload_during_the_fallback_begin_load_waits_for_its_teardown(rig):
    entered, release = threading.Event(), threading.Event()
    real_begin_load = rig.diffusers.begin_load

    def _slow_begin_load(repo_id, **kwargs):
        entered.set()
        release.wait(5)
        return real_begin_load(repo_id, **kwargs)

    rig.diffusers.begin_load = _slow_begin_load
    outcome: list = []

    def _run():
        try:
            outcome.append(_img2img(rig.engine)["engine"])
        except RuntimeError as exc:
            outcome.append(str(exc))

    worker = threading.Thread(target = _run)
    worker.start()
    assert entered.wait(5)
    unloaded: list = []
    unloader = threading.Thread(target = lambda: unloaded.append(rig.engine.unload()))
    unloader.start()
    time.sleep(0.05)
    assert unloaded == []
    release.set()
    unloader.join(5)
    worker.join(5)
    assert unloaded and unloaded[0]["loaded"] is False
    assert outcome == [DIFFUSION_CANCELLED_MSG]
    assert rig.diffusers._state is None
    assert ("diffusers", "generate") not in rig.log


def test_a_cancel_is_honoured_even_when_the_fallback_turns_ready(rig):
    rig.diffusers.hang_load = True
    outcome: list = []

    def _run():
        try:
            outcome.append(_img2img(rig.engine)["engine"])
        except RuntimeError as exc:
            outcome.append(str(exc))

    worker = threading.Thread(target = _run)
    worker.start()
    deadline = time.time() + 5
    while rig.engine._fallback_cancel is None and time.time() < deadline:
        time.sleep(0.005)
    rig.engine._fallback_cancel.set()
    rig.diffusers._loading = False
    rig.diffusers._state = SimpleNamespace(
        family = _FAM,
        repo_id = "unsloth/Z-Image-Turbo-GGUF",
        timings = {},
        gguf_path = None,
        convert_dir = None,
    )
    worker.join(5)
    assert outcome == [DIFFUSION_CANCELLED_MSG]
    assert ("diffusers", "generate") not in rig.log


def test_no_fallback_while_a_replacement_mlx_load_is_pending(rig):
    a_identity = load_identity("unsloth/Z-Image-Turbo-GGUF", "Tongyi-MAI/Z-Image-Turbo", "z-image")
    rig.mlx.hang_load = True
    real_begin_load = rig.mlx.begin_load

    def _keep_resident(repo_id, **kwargs):
        resident = rig.mlx._state
        out = real_begin_load(repo_id, **kwargs)
        rig.mlx._state = resident
        return out

    rig.mlx.begin_load = _keep_resident
    rig.engine.begin_load("someone/Other-GGUF", gguf_filename = "other-Q8_0.gguf")
    rig.log.clear()
    with pytest.raises(RuntimeError, match = "load is in progress"):
        _img2img(rig.engine, expected_load = a_identity)
    assert ("diffusers", "load") not in rig.log
    assert ("mlx", "unload") not in rig.log


def test_an_option_mlx_never_receives_is_refused_without_a_fallback(rig, monkeypatch):
    monkeypatch.setattr(families, "family_pipeline_available", lambda _f: False)
    with pytest.raises(ValueError, match = "sampler choice is not supported on the MLX engine"):
        _txt2img(rig.engine, sampler = "euler")
    with pytest.raises(ValueError, match = "shift override"):
        _txt2img(rig.engine, shift = 3.0)
    assert ("mlx", "generate") not in rig.log
    assert _txt2img(rig.engine, sampler = "default")["engine"] == "mlx"


def test_a_load_replacing_the_model_after_the_identity_check_is_refused(rig):
    a_identity = load_identity("unsloth/Z-Image-Turbo-GGUF", "Tongyi-MAI/Z-Image-Turbo", "z-image")
    entered, release = threading.Event(), threading.Event()
    real_accepted = rig.engine._accepted_identities

    def _paused(*args, **kwargs):
        out = real_accepted(*args, **kwargs)
        entered.set()
        assert release.wait(5)
        return out

    rig.engine._accepted_identities = _paused
    outcome: list = []

    def _run():
        try:
            outcome.append(_txt2img(rig.engine, expected_load = a_identity)["engine"])
        except Exception as exc:  # noqa: BLE001
            outcome.append(exc)

    worker = threading.Thread(target = _run)
    worker.start()
    assert entered.wait(5)
    rig.engine.begin_load("someone/Other-GGUF", gguf_filename = "other-Q8_0.gguf")
    release.set()
    worker.join(5)
    assert not worker.is_alive()
    assert len(outcome) == 1 and isinstance(outcome[0], DiffusionModelReplacedError), outcome
    assert ("mlx", "generate") not in rig.log


def test_a_superseded_fallback_never_unloads_the_new_load(rig):
    entered, release = threading.Event(), threading.Event()
    real_fallback = rig.engine._load_fallback

    def _paused(**kwargs):
        entered.set()
        assert release.wait(5)
        return real_fallback(**kwargs)

    rig.engine._load_fallback = _paused
    outcome: list = []

    def _run():
        try:
            outcome.append(_img2img(rig.engine)["engine"])
        except RuntimeError as exc:
            outcome.append(str(exc))

    worker = threading.Thread(target = _run)
    worker.start()
    assert entered.wait(5)
    rig.engine.begin_load("someone/Other-GGUF", gguf_filename = "other-Q8_0.gguf")
    rig.log.clear()
    release.set()
    worker.join(5)
    assert not worker.is_alive()
    assert outcome == [DIFFUSION_CANCELLED_MSG]
    assert rig.log == []
    assert rig.mlx._state is not None and rig.mlx._state.repo_id == "someone/Other-GGUF"
    assert rig.engine.is_loaded


def _pause_mlx_begin_load(rig):
    entered, release = threading.Event(), threading.Event()
    real_begin_load = rig.mlx.begin_load

    def _paused(repo_id, **kwargs):
        entered.set()
        assert release.wait(5)
        return real_begin_load(repo_id, **kwargs)

    rig.mlx.begin_load = _paused
    return entered, release


def _in_thread(fn):
    outcome: list = []

    def _run():
        try:
            outcome.append(fn())
        except Exception as exc:  # noqa: BLE001
            outcome.append(exc)

    worker = threading.Thread(target = _run)
    worker.start()
    return worker, outcome


def test_no_fallback_registers_while_a_replacement_is_being_set_up(rig):
    # Epoch bumped, options not yet registered: a fallback now would load stale options.
    entered, release = _pause_mlx_begin_load(rig)
    replacer, replaced = _in_thread(
        lambda: rig.engine.begin_load("someone/Other-GGUF", gguf_filename = "other-Q8_0.gguf")
    )
    assert entered.wait(5)
    with pytest.raises(RuntimeError, match = DIFFUSION_CANCELLED_MSG):
        _img2img(rig.engine)
    assert ("diffusers", "load") not in rig.log
    release.set()
    replacer.join(5)
    assert not replacer.is_alive() and not isinstance(replaced[0], Exception), replaced
    assert rig.mlx._state.repo_id == "someone/Other-GGUF"
    assert rig.diffusers._state is None
    assert rig.engine._load_kwargs["repo_id"] == "someone/Other-GGUF"


def test_a_route_decided_during_replacement_setup_never_registers_after_it(rig):
    entered, release = _pause_mlx_begin_load(rig)
    routed, go = threading.Event(), threading.Event()
    real_fallback = rig.engine._load_fallback

    def _paused_fallback(**kwargs):
        routed.set()
        assert go.wait(5)
        return real_fallback(**kwargs)

    rig.engine._load_fallback = _paused_fallback
    replacer, _ = _in_thread(
        lambda: rig.engine.begin_load("someone/Other-GGUF", gguf_filename = "other-Q8_0.gguf")
    )
    assert entered.wait(5)
    requester, outcome = _in_thread(lambda: _img2img(rig.engine)["engine"])
    assert routed.wait(5)
    release.set()
    replacer.join(5)
    assert not replacer.is_alive()
    rig.log.clear()
    go.set()
    requester.join(5)
    assert not requester.is_alive()
    assert len(outcome) == 1 and isinstance(outcome[0], RuntimeError), outcome
    assert str(outcome[0]) == DIFFUSION_CANCELLED_MSG
    assert rig.log == []
    assert rig.mlx._state.repo_id == "someone/Other-GGUF"
    assert rig.diffusers._state is None


def _guidance_distilled_mlx(rig, monkeypatch, *, supports_guidance):
    real_status, real_run = rig.mlx.status, rig.mlx._run

    def _status():
        out = real_status()
        if out.get("loaded"):
            out["supports_guidance"] = supports_guidance
        return out

    def _generate(
        *,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = 1024,
        height: Optional[int] = 1024,
        seed: Optional[int] = None,
        guidance: float = 0.0,
        expected_load: Any = None,
    ) -> dict:
        return real_run(expected_load, {"prompt": prompt, "guidance": guidance, "seed": seed})

    monkeypatch.setattr(rig.mlx, "status", _status)
    monkeypatch.setattr(rig.mlx, "generate", _generate)


def test_explicit_guidance_the_mlx_variant_drops_runs_on_diffusers(rig, monkeypatch):
    _guidance_distilled_mlx(rig, monkeypatch, supports_guidance = False)
    assert _txt2img(rig.engine, guidance = 0.0, negative_prompt = "blurry")["engine"] == "mlx"
    assert _txt2img(rig.engine)["engine"] == "mlx"
    assert ("diffusers", "load") not in rig.log
    result = _txt2img(rig.engine, guidance = 4.0, negative_prompt = "blurry")
    assert result["engine"] == "diffusers"
    assert ("diffusers", "load") in rig.log


def test_explicit_guidance_stays_on_an_mlx_variant_that_applies_it(rig, monkeypatch):
    _guidance_distilled_mlx(rig, monkeypatch, supports_guidance = True)
    assert _txt2img(rig.engine, guidance = 4.0, negative_prompt = "blurry")["engine"] == "mlx"
    assert ("diffusers", "load") not in rig.log


def test_explicit_guidance_without_a_fallback_is_refused_not_dropped(rig, monkeypatch):
    _guidance_distilled_mlx(rig, monkeypatch, supports_guidance = False)
    monkeypatch.setattr(families, "family_pipeline_available", lambda _f: False)
    with pytest.raises(ValueError, match = "guidance .* is not supported on the MLX engine"):
        _txt2img(rig.engine, guidance = 4.0)
    assert ("mlx", "generate") not in rig.log
    assert _txt2img(rig.engine, guidance = 0.0)["engine"] == "mlx"


def test_status_offers_the_fallback_guidance_controls_over_a_turbo_mlx_variant(rig, monkeypatch):
    # A guided request routes to diffusers, so the routed status advertises what diffusers serves.
    _guidance_distilled_mlx(rig, monkeypatch, supports_guidance = False)
    z_image = next(f for f in families._FAMILIES if f.name == "z-image")
    monkeypatch.setattr(rig.engine, "_family", lambda: z_image)
    monkeypatch.setattr(rig.engine, "_capabilities_fn", routing._diffusers_capabilities)
    status = rig.engine.status()
    assert status["engine"] == "mlx"
    assert status["supports_guidance"] is True
    assert status["supports_negative_prompt"] is True
    assert status["negative_prompt_min_guidance"] == 0.0
    assert rig.engine._mlx_guidance() is False
    assert _txt2img(rig.engine, guidance = 4.0, negative_prompt = "blurry")["engine"] == "diffusers"


def test_an_mlx_variant_that_applies_guidance_keeps_its_own_thresholds(rig, monkeypatch):
    _guidance_distilled_mlx(rig, monkeypatch, supports_guidance = True)
    real_status = rig.mlx.status

    def _status():
        out = real_status()
        out.update(supports_negative_prompt = True, negative_prompt_min_guidance = 1.0)
        return out

    monkeypatch.setattr(rig.mlx, "status", _status)
    monkeypatch.setattr(
        rig.engine,
        "_capabilities_fn",
        lambda fam: {"workflows": ["txt2img", "img2img"], "negative_prompt_min_guidance": 0.0},
    )
    assert rig.engine.status()["negative_prompt_min_guidance"] == 1.0


def test_an_unload_racing_a_failing_fallback_never_leaves_mlx_resident(rig):
    rig.diffusers.hang_load = True
    worker, outcome = _in_thread(lambda: _img2img(rig.engine))
    deadline = time.time() + 5
    while not rig.diffusers._loading and time.time() < deadline:
        time.sleep(0.01)
    assert rig.diffusers._loading and rig.mlx._state is None
    real_unload = rig.mlx.unload

    def _unload_then_fail_the_fallback(**kwargs):
        result = real_unload(**kwargs)
        rig.diffusers._load_error = "out of memory"
        worker.join(5)
        return result

    rig.mlx.unload = _unload_then_fail_the_fallback
    status = rig.engine.unload()
    worker.join(5)
    assert not worker.is_alive() and isinstance(outcome[0], RuntimeError), outcome
    assert status["loaded"] is False
    assert rig.mlx._state is None
    assert rig.diffusers._state is None


def test_a_unified_memory_refusal_of_the_fallback_keeps_its_error_code(rig):
    from core.inference.diffusion_memory import (
        UNIFIED_OVERSIZE_REFUSAL_CODE,
        unified_memory_refusal_code,
    )

    real_progress = rig.diffusers.load_progress

    def _progress():
        out = real_progress()
        if out.get("phase") == "error":
            out["error_code"] = UNIFIED_OVERSIZE_REFUSAL_CODE
        return out

    rig.diffusers.load_progress = _progress
    rig.diffusers.fail_next_load = "needs about 30 GB of unified memory"
    with pytest.raises(RuntimeError, match = "unified memory") as excinfo:
        _img2img(rig.engine)
    assert unified_memory_refusal_code(excinfo.value) == UNIFIED_OVERSIZE_REFUSAL_CODE
    assert rig.engine.status()["engine_routing"]["resident"] == ["mlx"]


def _refuse_for_unified_memory(rig, *messages):
    from core.inference.diffusion_memory import UNIFIED_OVERSIZE_REFUSAL_CODE

    real_load, real_progress = rig.diffusers.begin_load, rig.diffusers.load_progress
    pending = list(messages)

    def _load(repo_id, **kwargs):
        if pending:
            rig.diffusers.fail_next_load = pending.pop(0)
        return real_load(repo_id, **kwargs)

    def _progress():
        out = real_progress()
        if out.get("phase") == "error":
            out["error_code"] = UNIFIED_OVERSIZE_REFUSAL_CODE
        return out

    rig.diffusers.begin_load = _load
    rig.diffusers.load_progress = _progress


def test_a_coexist_fallback_refused_for_memory_retries_once_with_mlx_unloaded(rig):
    # Estimate says both fit; the loader's real free-memory reading says no.
    rig.fit["coexist"] = True
    _refuse_for_unified_memory(rig, "needs about 12 GB of unified memory")
    assert _img2img(rig.engine)["engine"] == "diffusers"
    assert rig.log == [
        ("diffusers", "load"),
        ("mlx", "unload"),
        ("diffusers", "load"),
        ("diffusers", "generate"),
    ]
    assert rig.engine.status()["engine_routing"]["resident"] == ["diffusers"]
    rig.log.clear()
    assert _img2img(rig.engine)["engine"] == "diffusers"
    assert rig.log == [("diffusers", "generate")]


def test_a_coexist_fallback_refused_twice_restores_mlx_and_keeps_the_error_code(rig):
    from core.inference.diffusion_memory import (
        UNIFIED_OVERSIZE_REFUSAL_CODE,
        unified_memory_refusal_code,
    )

    rig.fit["coexist"] = True
    _refuse_for_unified_memory(rig, "needs about 30 GB of unified memory", "still 30 GB")
    with pytest.raises(RuntimeError, match = "still 30 GB") as excinfo:
        _img2img(rig.engine)
    assert unified_memory_refusal_code(excinfo.value) == UNIFIED_OVERSIZE_REFUSAL_CODE
    assert rig.log == [
        ("diffusers", "load"),
        ("mlx", "unload"),
        ("diffusers", "load"),
        ("mlx", "load"),
    ]
    assert rig.engine.status()["engine_routing"]["resident"] == ["mlx"]
    assert _txt2img(rig.engine)["engine"] == "mlx"


def test_a_coexist_fallback_failing_for_another_reason_keeps_mlx_and_does_not_retry(rig):
    rig.fit["coexist"] = True
    rig.diffusers.fail_next_load = "missing file"
    with pytest.raises(RuntimeError, match = "missing file"):
        _img2img(rig.engine)
    assert rig.log == [("diffusers", "load")]
    assert rig.engine.status()["engine_routing"]["resident"] == ["mlx"]


_Z_TURBO_SCHEDULER = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "_diffusers_version": "0.36.0.dev0",
    "num_train_timesteps": 1000,
    "use_dynamic_shifting": False,
    "shift": 3.0,
}
_Z_BASE_SCHEDULER = {**_Z_TURBO_SCHEDULER, "shift": 6.0}


def _cache_base(monkeypatch, tmp_path, repo: str, files: dict) -> None:
    import json

    from huggingface_hub import constants

    import utils.hf_cache_settings as cache_settings

    root = tmp_path / "hub"
    folder = root / f"models--{repo.replace('/', '--')}"
    snapshot = folder / "snapshots" / ("0" * 40)
    for name, content in files.items():
        path = snapshot / name
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_text(json.dumps(content), encoding = "utf-8")
    (folder / "refs").mkdir(parents = True, exist_ok = True)
    (folder / "refs" / "main").write_text("0" * 40, encoding = "utf-8")
    monkeypatch.setattr(cache_settings, "active_hf_hub_cache", lambda: str(root))
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(root))


def _expected_shift_control(config: dict) -> Optional[dict]:
    from diffusers import FlowMatchEulerDiscreteScheduler

    from core.inference.diffusion_flow_shift import apply_comfy_flow_shift, shift_control

    pipe = SimpleNamespace(scheduler = FlowMatchEulerDiscreteScheduler.from_config(config))
    apply_comfy_flow_shift(pipe, 3.0)
    return shift_control("sd3", pipe.scheduler)


def _z_image_rig(rig, monkeypatch):
    z_image = next(f for f in families._FAMILIES if f.name == "z-image")
    monkeypatch.setattr(rig.engine, "_family", lambda: z_image)
    return z_image


@pytest.mark.parametrize("config", [_Z_TURBO_SCHEDULER, _Z_BASE_SCHEDULER], ids = ["turbo", "base"])
def test_status_offers_the_fallback_shift_control_from_the_cached_scheduler(
    rig, monkeypatch, tmp_path, config
):
    pytest.importorskip("diffusers")
    _z_image_rig(rig, monkeypatch)
    _cache_base(
        monkeypatch,
        tmp_path,
        "Tongyi-MAI/Z-Image-Turbo",
        {
            "scheduler/scheduler_config.json": config,
            "model_index.json": {"_class_name": "ZImagePipeline"},
        },
    )
    status = rig.engine.status()
    assert status["engine"] == "mlx"
    expected = _expected_shift_control(config)
    assert expected == {"node": "sd3", "shift": 3.0}
    assert status["shift_control"] == expected
    assert _txt2img(rig.engine, shift = 3.0)["engine"] == "diffusers"


def test_fallback_shift_control_reads_a_local_base_dir(rig, monkeypatch, tmp_path):
    pytest.importorskip("diffusers")
    import json

    z_image = _z_image_rig(rig, monkeypatch)
    (tmp_path / "scheduler").mkdir()
    (tmp_path / "scheduler" / "scheduler_config.json").write_text(json.dumps(_Z_BASE_SCHEDULER))
    status = {"repo_id": "unsloth/Z-Image-GGUF", "base_repo": str(tmp_path)}
    assert routing.fallback_shift_control(z_image, status) == _expected_shift_control(
        _Z_BASE_SCHEDULER
    )


def test_status_hides_shift_when_the_scheduler_config_is_not_cached(rig, monkeypatch, tmp_path):
    _z_image_rig(rig, monkeypatch)
    _cache_base(monkeypatch, tmp_path, "Tongyi-MAI/Z-Image-Turbo", {"model_index.json": {}})
    assert rig.engine.status().get("shift_control") is None


def test_status_hides_shift_when_the_base_ships_a_sampling_grid(rig, monkeypatch, tmp_path):
    _z_image_rig(rig, monkeypatch)
    _cache_base(
        monkeypatch,
        tmp_path,
        "Tongyi-MAI/Z-Image-Turbo",
        {
            "scheduler/scheduler_config.json": _Z_TURBO_SCHEDULER,
            "model_index.json": {"sample_sigmas": [1.0, 0.75, 0.5, 0.25]},
        },
    )
    assert rig.engine.status().get("shift_control") is None


def test_an_mlx_reported_shift_control_is_kept(rig, monkeypatch, tmp_path):
    _z_image_rig(rig, monkeypatch)
    _cache_base(
        monkeypatch,
        tmp_path,
        "Tongyi-MAI/Z-Image-Turbo",
        {"scheduler/scheduler_config.json": _Z_TURBO_SCHEDULER},
    )
    real_status = rig.mlx.status
    own = {"node": "sd3", "shift": 5.0}
    monkeypatch.setattr(rig.mlx, "status", lambda: {**real_status(), "shift_control": own})
    assert rig.engine.status()["shift_control"] == own


def test_no_fallback_no_shift_control(rig, monkeypatch, tmp_path):
    _z_image_rig(rig, monkeypatch)
    _cache_base(
        monkeypatch,
        tmp_path,
        "Tongyi-MAI/Z-Image-Turbo",
        {"scheduler/scheduler_config.json": _Z_TURBO_SCHEDULER},
    )
    monkeypatch.setattr(families, "family_pipeline_available", lambda _f: False)
    assert rig.engine.status().get("shift_control") is None


def _gated_rig(rig, gate):
    rig.engine.begin_load(
        "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z-image-turbo-Q4_0.gguf",
        local_files_only = True,
        hf_token = None,
        model_kind = "gguf",
        fallback_gate = gate,
    )
    rig.log.clear()
    return rig


def test_the_fallback_gate_wraps_the_diffusers_load_and_never_reaches_an_engine(rig):
    seen = []

    def _gate(start, cancel = None):
        seen.append("gate")
        return start()

    _gated_rig(rig, _gate)
    assert "fallback_gate" not in (rig.mlx.load_kwargs or {})
    assert _img2img(rig.engine)["engine"] == "diffusers"
    assert seen == ["gate"]
    assert "fallback_gate" not in (rig.diffusers.load_kwargs or {})
    assert rig.log == [("mlx", "unload"), ("diffusers", "load"), ("diffusers", "generate")]


def test_a_refusing_fallback_gate_leaves_mlx_loaded_and_keeps_the_refusal(rig):
    from fastapi import HTTPException

    refusal = HTTPException(status_code = 409, detail = "training is running")

    def _gate(start, cancel = None):
        raise routing.FallbackLoadRefusedError(refusal)

    _gated_rig(rig, _gate)
    with pytest.raises(routing.FallbackLoadRefusedError) as excinfo:
        _img2img(rig.engine)
    assert excinfo.value.http is refusal
    assert rig.log == []
    assert rig.engine.status()["engine_routing"]["resident"] == ["mlx"]
    assert _txt2img(rig.engine)["engine"] == "mlx"


def test_a_new_load_and_unload_drop_the_old_fallback_gate(rig):
    _gated_rig(rig, lambda start, cancel = None: (_ for _ in ()).throw(AssertionError("stale gate")))
    rig.engine.unload()
    rig.engine.begin_load("unsloth/Z-Image-Turbo-GGUF", model_kind = "gguf", hf_token = None)
    rig.log.clear()
    assert _img2img(rig.engine)["engine"] == "diffusers"


@pytest.fixture
def route_gate(monkeypatch):
    import routes.inference as inference
    from core.inference import gpu_arbiter as arb
    from core.inference import gpu_residency
    from core.training import diffusion_training_service as dts

    class _Trainer:
        active = False

        def is_active(self):
            return self.active

        def is_training_active(self):
            return self.active

        def gpu_load_admission(self):
            from contextlib import contextmanager

            @contextmanager
            def _cm():
                if self.active:
                    raise dts.TrainingActiveError("Diffusion training is running")
                yield

            return _cm()

    trainer = _Trainer()
    import core.training as training

    monkeypatch.setattr(training, "get_training_backend", lambda: trainer)
    monkeypatch.setattr(dts, "get_diffusion_training_service", lambda: trainer)
    monkeypatch.setattr(inference.account_access, "require_live_account", lambda: None)
    monkeypatch.setattr(inference, "_repo_is_in_the_hub_cache", lambda ref: False)
    noted: list = []
    monkeypatch.setattr(
        inference,
        "_note_load_fetched_with_a_request_token",
        lambda ref, token: noted.append((ref, token)),
    )
    fit = {"kwargs": {}}
    monkeypatch.setattr(gpu_residency, "media_fit_kwargs", lambda *a, **k: dict(fit["kwargs"]))
    residents = {arb.DIFFUSION: None}
    monkeypatch.setattr(arb, "_residents", residents)
    monkeypatch.setattr(arb, "_sync_from_view", lambda: None)
    request = SimpleNamespace(
        model_path = "unsloth/Z-Image-Turbo-GGUF",
        base_repo = "Tongyi-MAI/Z-Image-Turbo",
        hf_token = "hf_request_token",  # secret-scrub: allow
        text_encoder_file = None,
        vae_file = None,
        text_encoder_files = None,
    )
    monkeypatch.setattr(inference.account_access, "media_component_file_references", lambda req: ())
    gate = inference._mlx_fallback_load_gate(request, None, "off")
    return SimpleNamespace(
        gate = gate, trainer = trainer, noted = noted, fit = fit, residents = residents, arb = arb
    )


def test_the_route_gate_runs_a_normal_fallback_and_records_the_token(rig, route_gate):
    _gated_rig(rig, route_gate.gate)
    assert _img2img(rig.engine)["engine"] == "diffusers"
    assert rig.log == [("mlx", "unload"), ("diffusers", "load"), ("diffusers", "generate")]
    assert route_gate.noted == [
        ("unsloth/Z-Image-Turbo-GGUF", "hf_request_token"),
        ("Tongyi-MAI/Z-Image-Turbo", "hf_request_token"),
    ]


def test_the_fallback_is_refused_while_training_runs(rig, route_gate):
    route_gate.trainer.active = True
    _gated_rig(rig, route_gate.gate)
    with pytest.raises(routing.FallbackLoadRefusedError) as excinfo:
        _img2img(rig.engine)
    assert excinfo.value.http.status_code == 409
    assert "training is running" in excinfo.value.http.detail
    assert rig.log == []
    assert route_gate.noted == []
    assert rig.engine.status()["engine_routing"]["resident"] == ["mlx"]


def test_the_fallback_is_refused_when_the_trainer_wins_the_admission(rig, route_gate, monkeypatch):
    import routes.inference as inference

    monkeypatch.setattr(inference, "_guard_diffusion_load_against_training", lambda: None)
    route_gate.trainer.active = True
    _gated_rig(rig, route_gate.gate)
    with pytest.raises(routing.FallbackLoadRefusedError) as excinfo:
        _img2img(rig.engine)
    assert excinfo.value.http.status_code == 409
    assert "Diffusion training is running" in excinfo.value.http.detail
    assert rig.log == []


def test_the_fallback_is_refused_when_it_would_have_to_evict(rig, route_gate):
    route_gate.residents[route_gate.arb.CHAT] = None
    _gated_rig(rig, route_gate.gate)
    with pytest.raises(routing.FallbackLoadRefusedError) as excinfo:
        _img2img(rig.engine)
    assert excinfo.value.http.status_code == 409
    assert rig.log == []
    assert rig.engine.status()["engine_routing"]["resident"] == ["mlx"]


def test_the_fallback_is_refused_when_it_does_not_fit_under_always(rig, route_gate, monkeypatch):
    from core.inference import gpu_residency

    arb = route_gate.arb
    route_gate.residents[arb.CHAT] = None
    route_gate.fit["kwargs"] = {"policy": "always"}
    monkeypatch.setattr(arb, "_view", lambda: {arb.CHAT: None, arb.DIFFUSION: None})
    monkeypatch.setattr(
        arb,
        "resident_slots",
        lambda name: [{"kind": "chat", "model": "unsloth/Qwen3-4B"}] if name == arb.CHAT else [],
    )
    monkeypatch.setattr(gpu_residency, "chat_load_in_flight", lambda: False)
    _gated_rig(rig, route_gate.gate)
    with pytest.raises(routing.FallbackLoadRefusedError) as excinfo:
        _img2img(rig.engine)
    http = excinfo.value.http
    assert http.status_code == 409
    assert http.headers == dict(arb.MULTI_MODEL_REFUSAL_HEADERS)
    assert rig.log == []
    assert rig.engine.status()["engine_routing"]["resident"] == ["mlx"]


def test_only_an_engine_that_takes_a_fallback_gate_is_handed_one():
    import routes.inference as inference

    gate = object()
    assert inference._fallback_gate_kwarg(routing.RoutedMlxEngine().begin_load, gate) == {
        "fallback_gate": gate
    }
    assert inference._fallback_gate_kwarg(lambda repo_id, **kw: None, gate) == {}


def test_the_fallback_fit_check_stops_waiting_for_the_arbiter_once_cancelled():
    # An evictor holding the arbiter lock drains this generation: the wait must end on its cancel, not at the drain timeout.
    import threading
    import time

    from core.inference import gpu_arbiter as arb
    from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG
    from core.inference.gpu_residency import refuse_unless_fits

    cancel = threading.Event()
    with arb._lock:
        threading.Timer(0.2, cancel.set).start()
        started = time.monotonic()
        with pytest.raises(RuntimeError, match = DIFFUSION_CANCELLED_MSG):
            refuse_unless_fits(arb.DIFFUSION, cancel = cancel)
    assert time.monotonic() - started < 5
