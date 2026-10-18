# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/v1/videos takes the Unsloth Studio controls, draws a seed when none is sent, and keeps it."""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import sys
import time
import types
from pathlib import Path

import pytest

import routes.video as video_routes
from core.inference import audio_gallery, audio_inputs, audio_voices, media_auto_switch
from core.inference.video_families import VIDEO_MODEL_CHANGED_MSG
from utils.account_context import AccountContext, arun_as, is_owner_context

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from test_openai_videos_route import (  # noqa: E402,F401 -- fixtures are found by name
    _create,
    _healthy_diffusers,
    _reference_data_url,
    _reference_image_bytes,
    _save_clip,
    _wait_terminal,
    backend,
    client,
)

# The keys a plain OpenAI request sent begin_generate before the controls existed.
_PLAIN_KWARGS = {"prompt", "width", "height", "duration_s", "input_reference", "video_id"}
# VideoJob's keys before seed was added.
_PLAIN_RESPONSE = {
    "id",
    "object",
    "model",
    "status",
    "progress",
    "created_at",
    "completed_at",
    "expires_at",
    "prompt",
    "size",
    "seconds",
    "error",
    "remixed_from_video_id",
}
_WAV = b"RIFF\x24\x00\x00\x00WAVEfmt fake-wav-bytes"
_OTHER = AccountContext("acct-other", "other", "user")


def _capture(monkeypatch, backend) -> list[dict]:
    calls: list[dict] = []
    monkeypatch.setattr(backend, "begin_generate", lambda **kwargs: calls.append(kwargs))
    return calls


def _decoded(data_url: str) -> bytes:
    return base64.b64decode(data_url.split(",", 1)[1])


def _switch_to(
    monkeypatch,
    model_path: str,
    gguf_filename = None,
) -> dict:
    switched = {"completed": False}

    async def _fake_switch(
        model,
        *,
        before_switch = None,
        **_kwargs,
    ):
        pick = types.SimpleNamespace(
            model_path = model_path, gguf_filename = gguf_filename, model_kind = "auto"
        )
        if before_switch is not None:
            before_switch(pick)
        switched["completed"] = True

    monkeypatch.setattr(media_auto_switch, "maybe_auto_switch_media_model", _fake_switch)
    return switched


def test_a_plain_openai_request_sends_the_old_kwargs_plus_a_seed(client, backend, monkeypatch):
    calls = _capture(monkeypatch, backend)
    png_url = _reference_data_url("PNG", "image/png")
    json_resp = client.post(
        "/v1/videos",
        json = {
            "prompt": "a fox",
            "model": "sora-2",
            "seconds": "4",
            "size": "768x512",
            "input_reference": png_url,
        },
    )
    png = _reference_image_bytes("PNG")
    form_resp = _create(
        client,
        {"prompt": "a fox", "model": "sora-2", "seconds": "4", "size": "768x512"},
        {"input_reference": ("frame.png", png, "image/png")},
    )
    for resp, kwargs in zip((json_resp, form_resp), calls, strict = True):
        assert resp.status_code == 200, resp.json()
        assert set(kwargs) - {"seed"} == _PLAIN_KWARGS
        assert isinstance(kwargs["seed"], int) and 0 <= kwargs["seed"] < 2**53
        job = resp.json()
        assert set(job) - {"seed"} == _PLAIN_RESPONSE
        assert job["seed"] == kwargs["seed"]
        assert job["size"] == "768x512" and job["prompt"] == "a fox"


def test_every_control_reaches_begin_generate_from_json(client, backend, monkeypatch):
    calls = _capture(monkeypatch, backend)
    png_url = _reference_data_url("PNG", "image/png")
    jpeg_url = _reference_data_url("JPEG", "image/jpeg")
    wav_url = "data:audio/wav;base64," + base64.b64encode(_WAV).decode()
    scalars = {
        "seed": 5,
        "negative_prompt": "blur",
        "steps": 12,
        "guidance": 3.5,
        "num_frames": 49,
        "fps": 24,
    }
    resp = client.post(
        "/v1/videos", json = {"prompt": "x", **scalars, "last_frame": {"image_url": png_url}}
    )
    assert resp.status_code == 200, resp.json()
    assert resp.json()["seed"] == 5
    resp = client.post(
        "/v1/videos",
        json = {
            "prompt": "x",
            "reference_images": [png_url, jpeg_url],
            "reference_audios": [wav_url],
        },
    )
    assert resp.status_code == 200, resp.json()
    first, second = calls
    assert {key: first[key] for key in scalars} == scalars
    assert first["last_frame"] == png_url and first["duration_s"] is None
    assert "reference_images" not in first
    assert second["reference_images"] == [png_url, jpeg_url]
    assert second["reference_audios"] == [wav_url]
    assert "last_frame" not in second and "steps" not in second


def test_every_control_reaches_begin_generate_from_multipart(client, backend, monkeypatch):
    calls = _capture(monkeypatch, backend)
    png, jpeg = _reference_image_bytes("PNG"), _reference_image_bytes("JPEG")
    fields = {
        "prompt": "x",
        "seed": "9",
        "steps": "8",
        "guidance": "",
        "fps": "25",
        "num_frames": "33",
    }
    resp = client.post(
        "/v1/videos", data = fields, files = [("last_frame", ("end.png", png, "image/png"))]
    )
    assert resp.status_code == 200, resp.json()
    resp = client.post(
        "/v1/videos",
        data = {"prompt": "x"},
        files = [
            ("reference_images", ("a.png", png, "image/png")),
            ("reference_images[]", ("b.jpg", jpeg, "image/jpeg")),
            ("reference_audios", ("voice.wav", _WAV, "audio/wav")),
            ("reference_audios", ("score.bin", _WAV, "application/x-unknown")),
        ],
    )
    assert resp.status_code == 200, resp.json()
    first, second = calls
    assert (first["seed"], first["steps"], first["fps"], first["num_frames"]) == (9, 8, 25, 33)
    assert "guidance" not in first, "a blank form field means omitted"
    assert first["last_frame"].startswith("data:image/png;base64,")
    assert _decoded(first["last_frame"]) == png
    assert [_decoded(url) for url in second["reference_images"]] == [png, jpeg]
    assert second["reference_images"][1].startswith("data:image/jpeg;base64,")
    assert [url.split(",")[0] for url in second["reference_audios"]] == [
        "data:audio/wav;base64",
        "data:application/octet-stream;base64",
    ]
    assert [_decoded(url) for url in second["reference_audios"]] == [_WAV, _WAV]


def test_an_empty_audio_upload_is_refused(client, backend, monkeypatch):
    calls = _capture(monkeypatch, backend)
    resp = client.post(
        "/v1/videos",
        data = {"prompt": "x"},
        files = [
            ("reference_images", ("a.png", _reference_image_bytes("PNG"), "image/png")),
            ("reference_audios", ("voice.wav", b"", "audio/wav")),
        ],
    )
    assert resp.status_code == 400, resp.json()
    assert resp.json()["error"]["message"] == "reference_audios[0] is empty."
    assert resp.json()["error"]["param"] == "reference_audios"
    assert calls == []


def test_an_omitted_seed_is_drawn_kept_and_survives_a_restart(client, backend):
    backend.gate.clear()
    job = _create(client, {"prompt": "seeded"}).json()
    seed = job["seed"]
    assert isinstance(seed, int) and 0 <= seed < 2**53
    running = client.get(f"/v1/videos/{job['id']}").json()
    assert running["status"] in ("queued", "in_progress") and running["seed"] == seed
    video_routes._jobs = {}
    restored = client.get(f"/v1/videos/{job['id']}").json()
    assert restored["status"] in ("queued", "in_progress") and restored["seed"] == seed
    backend.gate.set()
    done = _wait_terminal(client, job["id"])
    assert done["status"] == "completed" and done["seed"] == seed
    assert backend.last_generate_kwargs["seed"] == seed, "the engine rendered another seed"
    video_routes._jobs = {}
    assert client.get(f"/v1/videos/{job['id']}").json()["seed"] == seed


def test_an_explicit_seed_passes_through_and_omitted_seeds_differ(client, backend, monkeypatch):
    calls = _capture(monkeypatch, backend)
    explicit = client.post("/v1/videos", json = {"prompt": "x", "seed": 2**53 - 1}).json()
    assert explicit["seed"] == calls[0]["seed"] == 2**53 - 1
    seeds = {client.post("/v1/videos", json = {"prompt": "x"}).json()["seed"] for _ in range(2)}
    assert len(seeds) == 2


def test_a_model_changed_retry_reuses_the_seed(client, backend, monkeypatch):
    seeds: list[int] = []

    def _begin(**kwargs):
        seeds.append(kwargs["seed"])
        if len(seeds) == 1:
            raise RuntimeError(VIDEO_MODEL_CHANGED_MSG)
        return None

    monkeypatch.setattr(backend, "begin_generate", _begin)
    resp = client.post("/v1/videos", json = {"prompt": "x"})
    assert resp.status_code == 200, resp.json()
    assert len(seeds) == 2 and seeds[0] == seeds[1] == resp.json()["seed"]


def test_a_completed_clip_reports_the_seed_the_engine_saved(client, backend):
    record = _save_clip("saved", "2026-01-01T00:00:00Z")
    assert client.get(f"/v1/videos/{record['id']}").json()["seed"] == 7


def test_an_old_record_loads_without_a_seed_and_a_new_one_round_trips():
    job = video_routes._VideoJob(
        id = "video_old_record",
        created_at = 100,
        prompt = "old",
        model = "model",
        size = "768x512",
        seconds = "5",
    )
    old = {key: value for key, value in vars(job).items() if key != "seed"}
    loaded = video_routes._job_from_record(old)
    assert loaded is not None and loaded.seed is None
    assert video_routes._job_to_openai(loaded).seed is None
    seeded = video_routes._job_from_record({**old, "seed": 11})
    assert seeded is not None and video_routes._job_to_openai(seeded).seed == 11


@pytest.mark.parametrize(
    "body, param, text",
    [
        (
            {"last_frame": "PNG"},
            "last_frame",
            "takes no last frame",
        ),
        (
            {"reference_images": ["PNG"]},
            "reference_images",
            "takes no reference images",
        ),
    ],
)
@pytest.mark.parametrize("switch", [True, False], ids = ["before-switch", "resident"])
def test_the_family_refuses_controls_it_cannot_take(
    client, backend, monkeypatch, body, param, text, switch
):
    png_url = _reference_data_url("PNG", "image/png")
    body = {key: ([png_url] if isinstance(value, list) else png_url) for key, value in body.items()}
    switched = _switch_to(monkeypatch, "Lightricks/LTX-2") if switch else None
    resp = client.post("/v1/videos", json = {"prompt": "x", "model": "LTX-2", **body})
    assert resp.status_code == 400, resp.json()
    assert text in resp.json()["error"]["message"]
    assert resp.json()["error"]["param"] == param
    if switched is not None:
        assert switched["completed"] is False, "the model was switched before the refusal"


@pytest.mark.parametrize(
    "body, param, text",
    [
        ({"reference_audios": ["WAV"]}, "reference_audios", "reference audio needs at least one"),
        ({"seconds": "4", "num_frames": 33}, None, "Send seconds or num_frames, not both."),
        (
            {"input_reference": "PNG", "reference_images": ["PNG"]},
            "reference_images",
            "input_reference cannot be combined with reference_images",
        ),
        ({"reference_images": ["PNG"] * 10}, "reference_images", "reference_images"),
        (
            {"reference_images": ["PNG"], "reference_audios": ["WAV"] * 4},
            "reference_audios",
            "reference_audios",
        ),
        (
            {"last_frame": "data:image/png;base64,AAAA"},
            "last_frame",
            "last_frame is not a readable",
        ),
    ],
)
def test_shape_refusals_come_before_any_switch(client, backend, monkeypatch, body, param, text):
    calls = _capture(monkeypatch, backend)
    switched = _switch_to(monkeypatch, "Lightricks/LTX-2")
    media = {"PNG": _reference_data_url("PNG", "image/png"), "WAV": "data:audio/wav;base64,AAAA"}
    body = {
        key: [media.get(v, v) for v in value]
        if isinstance(value, list)
        else media.get(value, value)
        for key, value in body.items()
    }
    resp = client.post("/v1/videos", json = {"prompt": "x", **body})
    assert resp.status_code == 400, resp.json()
    error = resp.json()["error"]
    assert text in error["message"]
    if param is not None:
        assert error["param"] == param
    assert switched["completed"] is False and calls == []


@pytest.fixture
def audio_store(monkeypatch, tmp_path):
    """Each account's audio under its own folder of the test's root."""

    def _gallery_dir() -> Path:
        directory = tmp_path / ("audio-owner" if is_owner_context() else "audio-other")
        directory.mkdir(parents = True, exist_ok = True)
        return directory

    monkeypatch.setattr(audio_gallery, "gallery_dir", _gallery_dir)


def _store_clip(prompt: str = "score") -> str:
    meta = {
        "prompt": prompt,
        "model": "audio-model",
        "audio_type": "snac",
        "sample_rate": 24000,
        "duration_s": 1.0,
        "created_at": "2026-10-17T00:00:00Z",
    }
    return audio_gallery.save(_WAV, meta)["id"]


def _store_input() -> str:
    input_id = "input-1"
    directory = audio_inputs.inputs_dir()
    (directory / f"{input_id}.wav").write_bytes(_WAV)
    meta = {
        "name": "voice.wav",
        "duration_s": 1.0,
        "sample_rate": 16000,
        "channels": 1,
        "sha256": "0" * 64,
        "created_at": "2026-10-17T00:00:00Z",
        "touched_at": time.time(),
    }
    (directory / f"{input_id}.json").write_text(json.dumps(meta), encoding = "utf-8")
    return input_id


def _store_voice() -> str:
    voice_id = "voice-1"
    directory = audio_voices.voices_dir()
    (directory / f"{voice_id}.wav").write_bytes(_WAV)
    meta = {"name": "Narrator", "duration_s": 1.0, "sample_rate": 16000, "created_at": "x"}
    (directory / f"{voice_id}.json").write_text(json.dumps(meta), encoding = "utf-8")
    return voice_id


@pytest.mark.parametrize("kind", ["clip_id", "input_id", "voice_id"])
def test_an_audio_id_resolves_to_its_stored_wav_before_the_switch(
    client, backend, monkeypatch, audio_store, kind
):
    ids = {
        "clip_id": _store_clip,
        "input_id": _store_input,
        "voice_id": _store_voice,
    }
    ref = {kind: ids[kind]()}
    order: list[str] = []
    real_resolve = video_routes._audio_ref_data_url

    def _resolve(item):
        order.append("resolve")
        return real_resolve(item)

    async def _fake_switch(model, **_kwargs):
        order.append("switch")

    monkeypatch.setattr(video_routes, "_audio_ref_data_url", _resolve)
    monkeypatch.setattr(media_auto_switch, "maybe_auto_switch_media_model", _fake_switch)
    calls = _capture(monkeypatch, backend)
    png_url = _reference_data_url("PNG", "image/png")
    resp = client.post(
        "/v1/videos", json = {"prompt": "x", "reference_images": [png_url], "reference_audios": [ref]}
    )
    assert resp.status_code == 200, resp.json()
    (audio,) = calls[0]["reference_audios"]
    assert audio.startswith("data:audio/wav;base64,") and _decoded(audio) == _WAV
    assert order == ["resolve", "switch"]


@pytest.mark.parametrize(
    "ref, text",
    [
        ({"clip_id": "missing"}, "That clip is no longer in your history."),
        ({"input_id": "missing"}, "This reference expired. Add it again."),
        ({"voice_id": "missing"}, "That saved voice no longer exists."),
    ],
)
def test_an_unknown_audio_id_is_a_404_and_never_renders(
    client, backend, monkeypatch, audio_store, ref, text
):
    calls = _capture(monkeypatch, backend)
    switched = _switch_to(monkeypatch, "Lightricks/LTX-2")
    png_url = _reference_data_url("PNG", "image/png")
    resp = client.post(
        "/v1/videos", json = {"prompt": "x", "reference_images": [png_url], "reference_audios": [ref]}
    )
    assert resp.status_code == 404, resp.json()
    assert resp.json()["error"]["message"] == text
    assert resp.json()["error"]["param"] == "reference_audios"
    assert calls == [] and switched["completed"] is False


def test_another_accounts_clip_is_a_404(client, backend, monkeypatch, audio_store):
    from utils.account_context import run_as

    calls = _capture(monkeypatch, backend)
    clip_id = run_as(_OTHER, _store_clip, "theirs")
    png_url = _reference_data_url("PNG", "image/png")
    resp = client.post(
        "/v1/videos",
        json = {
            "prompt": "x",
            "reference_images": [png_url],
            "reference_audios": [{"clip_id": clip_id}],
        },
    )
    assert resp.status_code == 404, resp.json()
    assert resp.json()["error"]["message"] == "That clip is no longer in your history."
    assert calls == []


def test_audio_ids_resolve_in_the_callers_account_across_the_thread(audio_store):
    from models.inference import AudioSourceRef

    clip_id = _store_clip("owner's")
    ref = AudioSourceRef(clip_id = clip_id)
    assert _decoded(video_routes._audio_ref_data_url(ref)) == _WAV

    async def _as_other():
        return await asyncio.to_thread(video_routes._audio_ref_data_url, ref)

    with pytest.raises(audio_inputs.AudioInputError) as exc:
        asyncio.run(arun_as(_OTHER, _as_other()))
    assert exc.value.status == 404


def test_create_openai_video_keeps_its_signature():
    assert list(inspect.signature(video_routes._create_openai_video).parameters) == [
        "body",
        "reference",
        "current_subject",
        "hf_token",
    ]


@pytest.mark.parametrize("switch", [True, False], ids = ["before-switch", "resident"])
def test_minimax_h3_refuses_a_custom_fps(client, backend, monkeypatch, switch):
    # H3 always writes its clip at 24 fps, so 30 fps would only mislabel the clip's length.
    calls = _capture(monkeypatch, backend)
    if switch:
        switched = _switch_to(monkeypatch, "MiniMaxAI/MiniMax-H3", "minimax_h3_ref2va-Q4_K_M.gguf")
    else:
        switched = None
        resident = backend.status

        def _h3_status():
            status = resident()
            return {
                **status,
                "family": "minimax-h3",
                "defaults": {**(status.get("defaults") or {}), "fps": 24},
            }

        monkeypatch.setattr(backend, "status", _h3_status)
    resp = client.post("/v1/videos", json = {"prompt": "x", "model": "H3", "fps": 30})
    assert resp.status_code == 400, resp.json()
    error = resp.json()["error"]
    assert error["param"] == "fps" and "fixed 24 fps" in error["message"]
    assert calls == []
    if switched is not None:
        assert switched["completed"] is False
    else:
        resp = client.post("/v1/videos", json = {"prompt": "x", "fps": 24})
        assert resp.status_code == 200, resp.json()


def test_minimax_h3_refuses_a_custom_fps_on_the_pinned_snapshot(client, backend, monkeypatch):
    # With auto-switch on, the route reads generation_snapshot(), which names no family.
    from core.inference.video_families import detect_video_family
    from utils import openai_auto_switch_settings

    calls = _capture(monkeypatch, backend)
    backend._state = types.SimpleNamespace(
        family = detect_video_family("MiniMaxAI/MiniMax-H3"),
        h3_task = None,
        engine = "diffusers",
        repo_id = "MiniMaxAI/MiniMax-H3",
    )

    async def _resident(*_args, **_kwargs):
        return None

    monkeypatch.setattr(media_auto_switch, "maybe_auto_switch_media_model", _resident)
    monkeypatch.setattr(media_auto_switch, "resident_answers_media_request", lambda *a, **k: True)
    monkeypatch.setattr(openai_auto_switch_settings, "get_media_auto_switch_enabled", lambda: True)
    resp = client.post("/v1/videos", json = {"prompt": "x", "model": "MiniMax-H3", "fps": 30})
    assert resp.status_code == 400, resp.json()
    assert resp.json()["error"]["param"] == "fps"
    assert calls == []


def test_multipart_references_keep_their_wire_order_across_both_spellings(
    client, backend, monkeypatch
):
    calls = _capture(monkeypatch, backend)
    png, jpeg = _reference_image_bytes("PNG"), _reference_image_bytes("JPEG")
    resp = client.post(
        "/v1/videos",
        data = {"prompt": "<Picture 1> meets <Picture 2>"},
        files = [
            ("reference_images[]", ("first.jpg", jpeg, "image/jpeg")),
            ("reference_images", ("second.png", png, "image/png")),
        ],
    )
    assert resp.status_code == 200, resp.json()
    assert [_decoded(url) for url in calls[0]["reference_images"]] == [jpeg, png]


def test_a_stored_audio_over_the_cap_is_refused_without_reading_it(audio_store, monkeypatch):
    from models.inference import AudioSourceRef

    ref = AudioSourceRef(clip_id = _store_clip())
    monkeypatch.setattr(video_routes, "_VIDEO_REFERENCE_AUDIO_MAX_BYTES", len(_WAV) - 1)

    def _no_read(self):
        raise AssertionError(f"{self} was read before its size was checked")

    monkeypatch.setattr(Path, "read_bytes", _no_read)
    with pytest.raises(audio_inputs.AudioInputError) as exc:
        video_routes._audio_ref_data_url(ref)
    assert exc.value.status == 400 and "24 MB" in exc.value.detail


def test_a_32_mb_image_fits_last_frame_and_reference_images():
    from pydantic import ValidationError

    from models.inference import VideoJobCreateRequest
    from utils.upload_limits import VIDEO_INPUT_REFERENCE_MAX_BYTES

    # A 32 MB image is about 42.7 MB of base64, past the old 32 MiB character cap.
    header = "data:image/png;base64,"
    full = header + "A" * (((VIDEO_INPUT_REFERENCE_MAX_BYTES + 2) // 3) * 4)
    assert VideoJobCreateRequest(prompt = "x", last_frame = full).last_frame == full
    assert VideoJobCreateRequest(prompt = "x", reference_images = [full]).reference_images == [full]
    over = full + "A" * (64 * 1024)
    with pytest.raises(ValidationError, match = "last_frame"):
        VideoJobCreateRequest(prompt = "x", last_frame = over)
    with pytest.raises(ValidationError, match = "each reference image must be at most 32 MB"):
        VideoJobCreateRequest(prompt = "x", reference_images = [over])
    audio = "data:audio/wav;base64," + "A" * (33 * 1024 * 1024)
    with pytest.raises(ValidationError, match = "each reference audio must be at most 24 MB"):
        VideoJobCreateRequest(prompt = "x", reference_images = [header], reference_audios = [audio])


def test_a_24_mb_audio_data_url_fits_reference_audios():
    from pydantic import ValidationError

    from models.inference import VideoJobCreateRequest

    # The full 24 MB the upload and id paths accept, plus its data URL header.
    audio = "data:audio/wav;base64," + "A" * (((24 * 1024 * 1024 + 2) // 3) * 4)
    image = _reference_data_url("PNG", "image/png")
    body = VideoJobCreateRequest(prompt = "x", reference_images = [image], reference_audios = [audio])
    assert body.reference_audios == [audio]
    with pytest.raises(ValidationError, match = "each reference audio must be at most 24 MB"):
        VideoJobCreateRequest(
            prompt = "x", reference_images = [image], reference_audios = [audio + "A" * 65536]
        )


@pytest.mark.parametrize(
    "audio", ["data:audio/wav;base64,", "data:audio/wav;base64,AAAA"], ids = ["empty", "undecodable"]
)
def test_unreadable_reference_audio_is_refused_before_the_switch(
    client, backend, monkeypatch, audio
):
    from core.inference import video_minimax_h3

    def _decode(blob, **_kwargs):
        raise ValueError("That reference file carries no audio track.")

    monkeypatch.setattr(video_minimax_h3, "decode_h3_reference_audio", _decode)
    calls = _capture(monkeypatch, backend)
    switched = _switch_to(monkeypatch, "MiniMaxAI/MiniMax-H3", "minimax_h3_ref2va-Q4_K_M.gguf")
    body = {
        "prompt": "x",
        "model": "H3",
        "reference_images": [_reference_data_url("PNG", "image/png")],
        "reference_audios": [audio],
    }
    resp = client.post("/v1/videos", json = body)
    assert resp.status_code == 400, resp.json()
    error = resp.json()["error"]
    assert error["param"] == "reference_audios"
    assert "reference_audios[0] is not readable audio" in error["message"]
    assert switched["completed"] is False and calls == []


def test_readable_reference_audio_goes_on_to_the_switch(client, backend, monkeypatch):
    from core.inference import video_minimax_h3

    seen = []
    monkeypatch.setattr(
        video_minimax_h3,
        "decode_h3_reference_audio",
        lambda blob, **_kwargs: seen.append(blob) or (None, 48000),
    )
    _capture(monkeypatch, backend)
    switched = _switch_to(monkeypatch, "MiniMaxAI/MiniMax-H3", "minimax_h3_ref2va-Q4_K_M.gguf")
    wav_url = "data:audio/wav;base64," + base64.b64encode(_WAV).decode()
    body = {
        "prompt": "x",
        "model": "H3",
        "reference_images": [_reference_data_url("PNG", "image/png")],
        "reference_audios": [wav_url],
    }
    client.post("/v1/videos", json = body)
    assert seen == [_WAV] and switched["completed"] is True
