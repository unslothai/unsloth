# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The /video/generate shape gate: the API must enforce the rules the interface offers.

The Desktop resolution select is populated from the loaded family's
``resolution_presets`` and its duration select from that family's k*frame_step+1
lattice, but the API accepted anything inside the coarse request bounds and then
SNAPPED it silently. 256x256 is divisible by both 16 and 32, so it survived the
snap untouched and denoised at a size no checkpoint was ever trained for. These
tests pin the family-aware rejection (422) and, just as importantly, the
fallbacks: nothing loaded, or a family declaring no presets, keeps snapping.

The pure-function half needs no torch/GPU; the route half swaps in a fake
backend that INHERITS the real begin_generate / job machinery, so the gate is
exercised where it actually lives (the route, before the worker starts).
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import core.inference.video as video_module
import core.inference.video_families as video_families_module
import core.inference.video_gallery as gallery_module
from auth.authentication import get_current_subject
from core.inference.video_families import (
    _FAMILIES,
    MAX_VIDEO_NUM_FRAMES,
    VIDEO_NOT_LOADED_MSG,
    detect_video_family,
    format_video_resolution_presets,
    snap_num_frames,
    snap_video_size,
    validate_video_request_shape,
)
from core.inference.video_minimax_h3 import h3_conditioning_mode
from models.inference import VideoReferenceVideo
from routes.video import router as video_router

# LTX-2 is the reference family for the single-family cases: 4 presets and frame_step 8.
LTX2 = detect_video_family("Lightricks/LTX-2")
# Wan is the contrast family: 704x1216 is an LTX-2 preset and is not one of Wan's.
WAN_TI2V_5B = detect_video_family("Wan-AI/Wan2.2-TI2V-5B-Diffusers")


# ── the validator itself ──────────────────────────────────────────────────────


@pytest.mark.parametrize("fam", _FAMILIES, ids = lambda f: f.name)
def test_every_declared_preset_is_accepted(fam):
    """Whatever the interface can offer, the API must take: the resolution select
    is built from exactly this tuple, so a rejection here is a dead UI control."""
    for width, height in fam.resolution_presets:
        validate_video_request_shape(
            fam, width = width, height = height, num_frames = fam.default_num_frames
        )


@pytest.mark.parametrize("fam", _FAMILIES, ids = lambda f: f.name)
def test_256x256_is_rejected_and_the_message_names_the_real_presets(fam):
    """The QA report's case. 256 divides both 16 and 32, so the snap left it alone;
    no family lists it, so every family must now refuse it by name."""
    with pytest.raises(ValueError) as excinfo:
        validate_video_request_shape(fam, width = 256, height = 256)
    message = str(excinfo.value)
    assert "256x256" in message
    assert fam.name in message
    # The message must quote sizes that actually exist, not a generic "unsupported".
    for width, height in fam.resolution_presets:
        assert f"{width}x{height}" in message


@pytest.mark.parametrize("fam", _FAMILIES, ids = lambda f: f.name)
def test_the_default_frame_count_is_on_its_own_lattice(fam):
    validate_video_request_shape(fam, num_frames = fam.default_num_frames)


def test_off_lattice_frame_count_is_rejected_with_the_straddling_counts():
    # 100 sits between 97 (12*8+1) and 105 on LTX-2's step-8 lattice.
    with pytest.raises(ValueError) as excinfo:
        validate_video_request_shape(LTX2, num_frames = 100)
    message = str(excinfo.value)
    assert "97" in message and "105" in message
    assert str(LTX2.default_num_frames) in message
    # On-lattice neighbours of the same request are fine.
    validate_video_request_shape(LTX2, num_frames = 97)
    validate_video_request_shape(LTX2, num_frames = 105)


def test_the_refusal_never_suggests_a_count_past_the_request_ceiling():
    """Near the top of the range the upper straddling point falls outside the request
    model's own `le`, so naming it sends the user into a second, differently-shaped 422.
    On LTX-2's step-8 lattice the whole band 1018-1024 is affected (1017 + 8 = 1025)."""
    with pytest.raises(ValueError) as excinfo:
        validate_video_request_shape(LTX2, num_frames = 1024)
    message = str(excinfo.value)
    assert "1017" in message
    assert "1025" not in message
    assert "the nearest supported count is" in message
    # The ceiling itself is on no family's lattice, but the point below it is loadable.
    validate_video_request_shape(LTX2, num_frames = 1017)
    # Away from the ceiling both points are still named.
    with pytest.raises(ValueError) as excinfo:
        validate_video_request_shape(LTX2, num_frames = 100)
    assert "the nearest supported counts are" in str(excinfo.value)


def test_the_request_ceiling_and_the_gate_share_one_constant():
    """A drifted pair would silently reintroduce the dead-end suggestion above."""
    from models.inference import VideoGenerateRequest

    field = VideoGenerateRequest.model_fields["num_frames"]
    ceiling = next(m.le for m in field.metadata if hasattr(m, "le"))
    assert ceiling == MAX_VIDEO_NUM_FRAMES


def test_wan_lattice_is_step_4_not_step_8():
    """Per-family, not one hardcoded rule: 85 is valid on Wan's 4k+1 and invalid on LTX-2's 8k+1."""
    wan = detect_video_family("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    validate_video_request_shape(wan, num_frames = 85)
    with pytest.raises(ValueError):
        validate_video_request_shape(LTX2, num_frames = 85)


def test_the_lattice_reads_frame_offset_not_a_hardcoded_one():
    """MiniMax-H3's lattice is 17k + 5, not 17k + 1. A gate written against k*step+1
    refuses every count H3's own duration select offers, its default of 124 included,
    so the offset has to come from the family the way snap_num_frames reads it."""
    h3 = detect_video_family("MiniMaxAI/MiniMax-H3")
    assert (h3.frame_step, h3.frame_offset) == (17, 5)
    # The three durations the interface offers (5s / 10s / 14.4s at 24 fps, snapped up).
    for count in (124, 243, 345):
        assert (count - h3.frame_offset) % h3.frame_step == 0
        validate_video_request_shape(h3, num_frames = count)
    # A count on the WRONG offset (17k + 1) is what the old rule would have accepted.
    with pytest.raises(ValueError) as excinfo:
        validate_video_request_shape(h3, num_frames = 137)
    message = str(excinfo.value)
    assert "k * 17 + 5" in message
    assert "124" in message and "141" in message


def test_a_suggested_count_never_falls_outside_the_family_range():
    """Naming a lattice point the family cannot load is the same dead end as naming one
    past the request ceiling. H3 starts at 124, so 90 (a real 17k+5 point) is not an answer."""
    h3 = detect_video_family("MiniMaxAI/MiniMax-H3")
    with pytest.raises(ValueError) as excinfo:
        validate_video_request_shape(h3, num_frames = 100)
    message = str(excinfo.value)
    assert "90" not in message
    assert "124" in message
    # Above the family ceiling of 345 there is nothing to suggest, so name the range.
    with pytest.raises(ValueError) as excinfo:
        validate_video_request_shape(h3, num_frames = 400)
    assert "supported counts run from 124 to 345" in str(excinfo.value)


def test_the_frame_gate_enforces_the_range_it_names():
    """A lattice point outside the family's trained window is refused, not snapped.

    The gate already computed the floor and the ceiling to WORD its lattice error, then accepted
    counts outside them: 5, 90 and 107 are all real 17k + 5 points below H3's floor of 124 and were
    snapped up to 124, and 362 and 872 were snapped down to 345. On the native path that turns
    num_frames=5 into a 25x compute surprise, silently.
    """
    h3 = detect_video_family("MiniMaxAI/MiniMax-H3")
    assert (h3.min_num_frames, h3.max_num_frames) == (124, 345)
    for count in (5, 90, 107, 362, 872):
        # Each is genuinely on the lattice, so only the range check can catch it.
        assert (count - h3.frame_offset) % h3.frame_step == 0
        with pytest.raises(ValueError) as excinfo:
            validate_video_request_shape(h3, num_frames = count)
        assert "counts run from 124 to 345" in str(excinfo.value)
    # The three counts the interface offers stay valid.
    for count in (124, 243, 345):
        validate_video_request_shape(h3, num_frames = count)
    # Families that declare no window are untouched: LTX-2 keeps its whole lattice.
    for count in (1, 9, 121, 1017):
        validate_video_request_shape(LTX2, num_frames = count)


def test_omitted_fields_are_always_valid():
    """None means "use the family default", which is valid by construction."""
    validate_video_request_shape(LTX2)
    validate_video_request_shape(LTX2, num_frames = None)


def test_a_half_specified_size_resolves_against_the_default_preset():
    """generate() fills a missing side from presets[0], so the check must judge the
    same pair it will actually denoise."""
    # 768 alone resolves to 768x512, the default preset.
    validate_video_request_shape(LTX2, width = 768)
    validate_video_request_shape(LTX2, height = 512)
    # 1216 alone resolves to 1216x512, which is NOT a preset (1216 only pairs with 704).
    with pytest.raises(ValueError) as excinfo:
        validate_video_request_shape(LTX2, width = 1216)
    assert "1216x512" in str(excinfo.value)


def test_presets_spelled_as_lists_still_match():
    """The status payload hands presets out as lists; a round-trip back in must not
    silently stop matching and start 422-ing every supported size."""
    fam = replace(LTX2, resolution_presets = tuple([w, h] for w, h in LTX2.resolution_presets))
    for width, height in LTX2.resolution_presets:
        validate_video_request_shape(fam, width = width, height = height)


def test_a_family_with_no_presets_is_left_to_the_snap():
    """Backwards compatibility for an unusual/custom family: nothing to enforce
    against, so the old silent snap stays in charge rather than a blanket 422."""
    fam = replace(LTX2, resolution_presets = ())
    validate_video_request_shape(fam, width = 256, height = 256)
    # The frame lattice is intrinsic to the VAE, so it is still enforced.
    with pytest.raises(ValueError):
        validate_video_request_shape(fam, num_frames = 100)


def test_snapping_helpers_are_untouched():
    """The validator is additive: internal callers still get the flooring snap."""
    assert snap_video_size(LTX2, 250, 250) == (224, 224)
    assert snap_num_frames(LTX2, 100) == 97
    assert format_video_resolution_presets(LTX2) == "768x512, 1216x704, 704x1216, 512x768"


def test_reference_video_trim_schema_requires_one_bounded_interval():
    reference = VideoReferenceVideo(
        video = "data:video/mp4;base64,AA==",
        trim_start_seconds = 4.0,
        trim_end_seconds = 19.0,
    )
    assert reference.trim_start_seconds == 4.0
    assert reference.trim_end_seconds == 19.0

    for values, message in (
        ({"trim_start_seconds": 4.0}, "provided together"),
        ({"trim_start_seconds": 4.0, "trim_end_seconds": 5.0}, "2 to 15 seconds"),
        ({"trim_start_seconds": 4.0, "trim_end_seconds": 20.0}, "2 to 15 seconds"),
    ):
        with pytest.raises(ValueError, match = message):
            VideoReferenceVideo(video = "data:video/mp4;base64,AA==", **values)


# ── the route ─────────────────────────────────────────────────────────────────


class _ShapeFakeBackend(video_module.VideoBackend):
    """Real load state + real begin_generate/job machinery over a stub generate().

    ``_state`` is a genuine ``_VideoLoadState`` so ``loaded_family()`` is exercised
    against the object the loader really commits, and generate() mirrors the real
    one's shape resolution (snap + family defaults) so a test can see whether a
    request was snapped or rejected.
    """

    def load_as(self, fam) -> None:
        self._state = video_module._VideoLoadState(
            pipe = object(),
            family = fam,
            repo_id = f"unsloth/{fam.name}",
            base_repo = fam.base_repo,
            device = "cpu",
            dtype = "bfloat16",
            kind = "pipeline",
        )

    def generate(
        self,
        *,
        prompt,
        seed = None,
        cancel_event = None,
        **kwargs,
    ):
        state = self._state
        if state is None:
            raise RuntimeError(VIDEO_NOT_LOADED_MSG)
        fam = state.family
        default = fam.resolution_presets[0] if fam.resolution_presets else (768, 512)
        width, height = snap_video_size(
            fam, kwargs.get("width") or default[0], kwargs.get("height") or default[1]
        )
        frames = snap_num_frames(fam, kwargs.get("num_frames") or fam.default_num_frames)
        fps = int(kwargs.get("fps") or fam.default_fps)
        return {
            "mp4_bytes": b"MP4-FAKE-BYTES",
            "seed": 4242 if seed is None else seed,
            "repo_id": state.repo_id,
            "width": width,
            "height": height,
            "num_frames": frames,
            "fps": fps,
            "duration_s": frames / fps,
            "has_audio": fam.has_audio,
            "steps": int(kwargs.get("steps") or fam.default_steps),
            "guidance": fam.default_guidance,
            # The real generate() records how the clip was conditioned and the job's persist step
            # reads it unconditionally, so the stub has to speak the same contract or every route
            # case here dies in persist with a KeyError instead of exercising the gate. Derived
            # from the shared helper rather than a literal, so a new conditioning mode cannot
            # leave this stub quietly returning a spelling the gallery no longer accepts.
            "conditioning": h3_conditioning_mode(),
            "flow_shift": kwargs.get("flow_shift"),
            "audio_flow_shift": kwargs.get("audio_flow_shift"),
        }


@pytest.fixture
def backend(monkeypatch):
    fake = _ShapeFakeBackend()
    monkeypatch.setattr(video_module, "get_video_backend", lambda: fake)
    return fake


@pytest.fixture
def client(backend, monkeypatch, tmp_path):
    # A real tmp gallery so the completed path runs the actual persist code.
    monkeypatch.setattr(gallery_module, "gallery_dir", lambda: tmp_path)
    app = FastAPI()
    app.include_router(video_router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app)


def _payload(**overrides) -> dict:
    return {"prompt": "a cat", **overrides}


def _wait_terminal(client, timeout = 5.0) -> dict:
    """Generation is asynchronous (the POST only starts the job), so the outcome is
    only observable by polling generate-progress."""
    deadline = time.monotonic() + timeout
    progress: dict = {}
    while time.monotonic() < deadline:
        progress = client.get("/api/inference/video/generate-progress").json()
        if progress.get("phase") in ("completed", "failed"):
            return progress
        time.sleep(0.01)
    raise AssertionError(f"generation never reached a terminal state: {progress}")


def test_generate_rejects_256x256_with_422_naming_the_presets(client, backend):
    """The QA report end to end: the request is in range and parses, but the loaded
    model cannot render it, so it is refused instead of silently denoised."""
    backend.load_as(LTX2)
    resp = client.post("/api/inference/video/generate", json = _payload(width = 256, height = 256))
    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert "256x256" in detail
    assert "768x512" in detail and "1216x704" in detail
    # Rejected AT THE BOUNDARY: no job was started, so the backend is still idle.
    progress = client.get("/api/inference/video/generate-progress").json()
    assert progress["active"] is False and progress.get("phase") is None


@pytest.mark.parametrize("fam", _FAMILIES, ids = lambda f: f.name)
def test_generate_accepts_every_declared_preset_of_the_loaded_family(client, backend, fam):
    """Every size the interface can offer for this family round-trips to a saved clip."""
    backend.load_as(fam)
    for width, height in fam.resolution_presets:
        resp = client.post(
            "/api/inference/video/generate",
            json = _payload(width = width, height = height, num_frames = fam.default_num_frames),
        )
        assert resp.status_code == 200, (fam.name, width, height, resp.text)
        record = _wait_terminal(client)["video"]
        assert (record["width"], record["height"]) == (width, height)


def test_generate_rejects_an_off_lattice_frame_count_with_422(client, backend):
    backend.load_as(LTX2)
    resp = client.post("/api/inference/video/generate", json = _payload(num_frames = 100))
    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert "97" in detail and "105" in detail


def test_generate_with_nothing_loaded_still_reports_not_loaded_not_a_shape_error(client):
    """The gate must not preempt the 409: with no model there is no family whose
    rules could be applied, so the request falls through exactly as before."""
    resp = client.post("/api/inference/video/generate", json = _payload(width = 256, height = 256))
    assert resp.status_code == 409
    assert resp.json()["detail"] == VIDEO_NOT_LOADED_MSG


def test_generate_for_a_family_without_presets_still_snaps(client, backend):
    """Backwards compatibility: an odd size against a family that declares no presets
    is accepted and floored to the family multiple, the pre-change behaviour."""
    backend.load_as(replace(LTX2, resolution_presets = ()))
    resp = client.post("/api/inference/video/generate", json = _payload(width = 250, height = 250))
    assert resp.status_code == 200, resp.text
    record = _wait_terminal(client)["video"]
    # 250 floored to LTX-2's /32 multiple, as snap_video_size has always done.
    assert (record["width"], record["height"]) == (224, 224)


def test_a_family_without_presets_still_enforces_its_frame_lattice(client, backend):
    """The preset escape hatch covers the SIZE only. frame_step is declared whether or
    not a family lists presets, so an off-lattice count is still a 422 here -- pinned at
    the route because the size and frame branches take different paths through the gate."""
    backend.load_as(replace(LTX2, resolution_presets = ()))
    resp = client.post("/api/inference/video/generate", json = _payload(num_frames = 100))
    assert resp.status_code == 422, resp.text
    assert "97" in resp.json()["detail"] and "105" in resp.json()["detail"]
    # And an on-lattice count against the same preset-less family still runs.
    assert (
        client.post("/api/inference/video/generate", json = _payload(num_frames = 97)).status_code
        == 200
    )


def test_generate_omitting_the_shape_uses_the_family_defaults(client, backend):
    """The common API call sends no size at all; it must not be caught by the gate."""
    backend.load_as(LTX2)
    resp = client.post("/api/inference/video/generate", json = _payload())
    assert resp.status_code == 200, resp.text
    record = _wait_terminal(client)["video"]
    assert (record["width"], record["height"]) == LTX2.resolution_presets[0]
    assert record["num_frames"] == LTX2.default_num_frames


def test_the_coarse_pydantic_bounds_still_reject_out_of_range_sizes(client, backend):
    """The outer guard is unchanged: family-agnostic nonsense is still a 422 from the
    request model, before any family is consulted."""
    backend.load_as(LTX2)
    for body in (_payload(width = 16), _payload(height = 4096), _payload(num_frames = 0)):
        assert client.post("/api/inference/video/generate", json = body).status_code == 422


# ── the check has to be atomic with the state it judges ───────────────────────


def test_the_shape_is_judged_under_the_lock_that_reserves_the_state(backend, monkeypatch):
    """A load commits its new ``_state`` under the same lock ``begin_generate`` takes. Reading
    the family separately, before that lock, leaves a window where a size is accepted for the
    family being replaced and then denoised by the new one -- or a size the new family supports
    is rejected. Proven directly: while the validator runs, the lock is unavailable to anyone
    else, so no load can be committing.
    """
    backend.load_as(LTX2)
    held: list[bool] = []
    real = video_families_module.validate_video_request_shape

    def _spy(*args, **kwargs):
        # Another thread, because a Lock says nothing about which thread owns it and this call
        # runs on the one that took it.
        probe: list[bool] = []
        watcher = threading.Thread(target = lambda: probe.append(backend._lock.acquire(False)))
        watcher.start()
        watcher.join(5)
        if probe and probe[0]:
            backend._lock.release()
        held.append(not (probe and probe[0]))
        return real(*args, **kwargs)

    monkeypatch.setattr(video_module, "validate_video_request_shape", _spy)
    backend.begin_generate(prompt = "a cat", width = 768, height = 512)

    assert held == [True], "the shape was judged outside the lock that owns the state"


def test_a_family_swap_cannot_slip_between_the_check_and_the_job(backend):
    """The consequence, end to end rather than by construction. 704x1216 is a real LTX-2 preset
    and is not one of Wan's, so whichever family is resident when the lock is taken decides the
    outcome -- and the job that runs afterwards is the one that was judged."""
    backend.load_as(LTX2)
    backend.begin_generate(prompt = "a cat", width = 704, height = 1216)  # accepted for LTX-2
    backend.cancel_generate()
    # The worker clears the slot asynchronously and this test is about the check, not the job.
    backend._generate_job_active = False

    backend.load_as(WAN_TI2V_5B)
    with pytest.raises(ValueError):
        backend.begin_generate(prompt = "a cat", width = 704, height = 1216)
    assert not backend._generate_job_active, "a refused shape must not reserve the job slot"


# ── the half-specified canvas, with and without a keyframe ────────────────────


def _tiny_png_b64() -> str:
    """A 2x1 PNG, enough for the keyframe branch of the request model."""
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (2, 1), (255, 0, 0)).save(buf, format = "PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.mark.parametrize("axes", [{"width": 768}, {"height": 512}])
def test_a_half_specified_canvas_stays_valid_without_a_keyframe(axes):
    """The regression this pins: the paired-axes rule was written for the keyframe canvas but
    ran as an unconditional request validator, so every existing LTX / Wan / Hunyuan /
    prompt-only H3 client that sends one axis started getting a 422. The backend deliberately
    resolves the missing axis from the family's default preset (validate_video_request_shape
    documents it, and _resolve_keyframes implements it), so these calls must still be accepted.
    """
    from models.inference import VideoGenerateRequest

    req = VideoGenerateRequest(prompt = "a cat", **axes)
    assert (req.width, req.height) == (axes.get("width"), axes.get("height"))


@pytest.mark.parametrize("axes", [{"width": 768}, {"height": 512}])
def test_a_half_specified_canvas_is_refused_with_a_keyframe(axes):
    """The rule still has to hold where it means something: with a keyframe present
    _resolve_keyframes matches the SOURCE aspect whenever either axis is missing, so the axis
    the caller did send would be silently discarded. Refuse rather than draw another recipe."""
    import pydantic
    from models.inference import VideoGenerateRequest

    with pytest.raises(pydantic.ValidationError, match = "sent together"):
        VideoGenerateRequest(prompt = "a cat", first_frame = _tiny_png_b64(), **axes)


def test_both_axes_and_neither_stay_valid_with_a_keyframe():
    """The two shapes the rule exists to allow: an explicit canvas, and "match source"."""
    from models.inference import VideoGenerateRequest

    frame = _tiny_png_b64()
    assert (
        VideoGenerateRequest(prompt = "a cat", first_frame = frame, width = 768, height = 512).width == 768
    )
    assert VideoGenerateRequest(prompt = "a cat", first_frame = frame).width is None


# ── on-lattice but out of the family's range ─────────────────────────────────


@pytest.mark.parametrize("count", [107, 362])
def test_an_on_lattice_count_outside_the_family_range_is_refused(count):
    """The hole this closes: the gate judged the LATTICE only, while snap_num_frames also CLAMPS
    to min/max_num_frames. MiniMax-H3 is 17k + 5 over 124..345, so 107 and 362 both sit exactly
    on the lattice, passed validation, and were then rendered as 124 and 345 -- the API
    accepting one recipe and drawing another, which is the whole reason this check exists."""
    from core.inference.video_families import (
        VideoShapeError,
        detect_video_family,
        snap_num_frames,
        validate_video_request_shape,
    )

    fam = detect_video_family("", override = "minimax-h3")
    assert (count - fam.frame_offset) % fam.frame_step == 0, "the point of the case is on-lattice"
    assert snap_num_frames(fam, count) != count, "and that the snap would have moved it"
    with pytest.raises(VideoShapeError, match = "not a supported frame count"):
        validate_video_request_shape(fam, num_frames = count)


@pytest.mark.parametrize("count", [124, 141, 345])
def test_in_range_lattice_counts_still_pass(count):
    """The endpoints and one interior point stay valid, so the range check did not narrow the
    family to less than it actually offers."""
    from core.inference.video_families import detect_video_family, validate_video_request_shape

    fam = detect_video_family("", override = "minimax-h3")
    validate_video_request_shape(fam, num_frames = count)


def test_a_family_without_a_declared_range_is_unaffected():
    """Every pre-existing family declares min 1 and no max, so the added bound must be inert for
    them: the request model's own ceiling stays the only upper limit."""
    from core.inference.video_families import detect_video_family, validate_video_request_shape

    fam = detect_video_family("", override = "ltx-2")
    assert fam.min_num_frames == 1 and fam.max_num_frames is None
    for k in (0, 1, 5, 20):
        validate_video_request_shape(fam, num_frames = k * fam.frame_step + fam.frame_offset)
