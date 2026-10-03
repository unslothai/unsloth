# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Music studio requests to audiocpp_server, per family and mode, against a recording fake server.

Strict specs (schema_version) refuse undeclared options, so their bodies carry only declared keys:
MiDashengLM and ControlFoley take ``duration_sec``, never ``duration_seconds`` (finding 7).
"""

from __future__ import annotations

import base64
import io
import json
import wave
from dataclasses import replace
from pathlib import Path

import pytest

from core.inference import audio_cpp_backend, audio_cpp_files, audio_cpp_music as cm
from core.inference import audio_cpp_models as acm
from core.inference.audio_cpp_models import AUDIO_CPP_REPO, AudioCppVariant, RepoFile

STRICT_KEYS = {
    "heartmula": frozenset(
        {
            "lyrics",
            "tags",
            "duration_sec",
            "temperature",
            "top_k",
            "guidance_scale",
            "codec_duration_sec",
            "num_inference_steps",
            "codec_guidance_scale",
            "infinite_mode",
            "text_chunk_size",
            "infinite_chunk_audio_duration_ms",
            "seed",
        }
    ),
    "midashenglm_gen": frozenset(
        {"duration_sec", "guidance_scale", "stop_threshold", "min_stop_step", "seed"}
    ),
    "controlfoley": frozenset(
        {
            "duration_sec",
            "num_inference_steps",
            "guidance_scale",
            "negative_prompt",
            "video",
            "mask_away_clip",
            "seed",
        }
    ),
    "minimax_music3": frozenset(
        {
            "lyrics",
            "duration_sec",
            "num_inference_steps",
            "guidance_scale",
            "ar_guidance_scale",
            "top_k",
            "seed",
        }
    ),
    "yue2": frozenset(
        {
            "style",
            "lyrics",
            "cot",
            "guidance_scale",
            "seed",
            "num_inference_steps",
            "semantic_temperature",
            "semantic_top_p",
            "semantic_top_k",
            "semantic_min_tokens",
            "semantic_max_tokens",
        }
    ),
}
FOLDERS = {
    "ace_step": "ACE-Step1.5-GGUF",
    "stable_audio": "Stable-Audio-3-Small-Music-GGUF",
    "heartmula": "HeartMuLa-GGUF",
    "midashenglm_gen": "MiDashengLM-Gen-GGUF",
    "controlfoley": "ControlFoley-GGUF",
    "minimax_music3": "MiniMax-Music3-GGUF",
    "yue2": "Yue2-3B-GGUF",
}


def _wav(
    seconds = 0.1,
    rate = 44100,
    channels = 2,
    fill = 0,
) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes([fill, 0]) * int(seconds * rate) * channels)
    return buf.getvalue()


def _reply(n = 1):
    takes = [_wav(fill = i + 1) for i in range(n)]
    body = {
        "audio": base64.b64encode(takes[0]).decode(),
        "named_audio_outputs": [
            {"id": f"audio_{i}", "audio": base64.b64encode(t).decode()} for i, t in enumerate(takes)
        ],
    }
    return "application/json", json.dumps(body).encode()


def _model(
    family,
    folder = None,
    strict = True,
    spec_options = (),
):
    folder = folder or FOLDERS[family]
    policy = acm.family_policy(family, None, (folder,))
    files = (RepoFile(f"{folder}/{folder.lower()}-q8_0.gguf", 4),)
    variant = AudioCppVariant("Q8_0", files, files[0].path)
    return acm.AudioCppModel(
        id = f"{AUDIO_CPP_REPO}/{folder}",
        repo_id = AUDIO_CPP_REPO,
        folder = folder,
        display_name = folder,
        family = family,
        task = "music",
        server_task = "gen",
        variant = variant,
        variants = (variant,),
        default_variant = "Q8_0",
        options = acm.option_schema(policy, {"options": {"request": list(spec_options)}}, None),
        music = acm.music_with_spec_bounds(policy.music, list(spec_options)),
        request_keys = STRICT_KEYS.get(family) if strict else None,
    )


class _Server:
    def __init__(
        self,
        model,
        replies = None,
        backend = "cuda",
    ):
        self.model = model
        self.model_id = "studio-test"
        self.backend = backend
        self.replies = list(replies or [])
        self.calls: list[tuple[str, dict, dict]] = []

    def alive(self):
        return True

    def post_json(self, path, payload, **kwargs):
        self.calls.append((path, json.loads(json.dumps(payload)), kwargs))
        return self.replies.pop(0) if self.replies else _reply()

    def stop(self):
        pass


@pytest.fixture
def starts(monkeypatch):
    """Server (re)starts, recorded with the session options each was given."""
    started: list[dict] = []

    def start(served, model_path, **_kw):
        started.append(dict((served.model_options or {}).get("session_options") or {}))
        return _Server(served)

    monkeypatch.setattr(audio_cpp_backend.AudioCppServer, "start", staticmethod(start))
    monkeypatch.setattr(audio_cpp_files, "materialize", lambda model: "/models/x.gguf")
    return started


def _backend(
    model,
    replies = None,
    backend = "cuda",
):
    b = audio_cpp_backend.AudioCppBackend()
    b._server = _Server(model, replies, backend)
    b._model = model
    b.models = {model.id: {"is_audio": True, "audio_type": model.audio_type}}
    b.active_model_name = model.id
    return b


def _run(
    b,
    seed = 7,
    output_dir = None,
    options = None,
    source = None,
    **music,
):
    music.setdefault("mode", "song")
    return b.generate_audio_response(
        music.get("text", ""),
        seed = seed,
        workflow = "music",
        music = music,
        audio_options = options,
        audio_inputs = {"source": source} if source else None,
        output_dir = output_dir,
    )


def _request(b, index = 0):
    path, body, _kw = b._server.calls[index]
    assert path == "/v1/tasks/run" and body["model"] == "studio-test"
    return body["request"]


def _keys(node):
    """Every key anywhere in a request body."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _keys(value)


def test_each_music_family_offers_its_modes():
    modes = lambda family, folder = None: [m.id for m in _model(family, folder).music.modes]
    assert modes("ace_step") == ["song", "edit"]
    assert modes("stable_audio") == ["song", "edit"]
    assert modes("stable_audio", "Stable-Audio-3-Small-SFX-GGUF") == ["sfx"]
    assert modes("stable_audio", "Stable-Audio-3-Medium-GGUF") == ["song", "sfx", "edit"]
    assert modes("heartmula") == ["song"]
    assert modes("midashenglm_gen") == ["song", "sfx"]
    assert modes("controlfoley") == ["sfx"]
    assert modes("minimax_music3") == ["song"]
    assert modes("yue2") == ["song"]
    assert acm.family_policy("kokoro_tts").music is None


def test_spec_duration_bounds_narrow_the_table():
    mida = _model(
        "midashenglm_gen",
        spec_options = [{"name": "duration_sec", "type": "float", "min": 0.04, "max": 163.84}],
    )
    assert mida.music.mode("song").duration == (1.0, 80.0, 30.0)
    assert mida.music.mode("sfx").duration == (1.0, 80.0, 10.0)
    foley = _model(
        "controlfoley",
        spec_options = [{"name": "duration_sec", "type": "float", "min": 0.64, "max": 60.0}],
    )
    assert foley.music.mode("sfx").duration == (1.0, 60.0, 8.0)
    narrow = _model("ace_step", spec_options = [{"name": "duration_sec", "type": "float", "max": 20}])
    assert narrow.music.mode("song").duration == (5.0, 20.0, 20.0)


def test_request_keys_come_from_strict_specs_only():
    strict = {
        "schema_version": 1,
        "options": {"request": [{"name": "duration_sec"}, {"name": "seed"}]},
    }
    assert acm._request_keys(strict, None) == frozenset({"duration_sec", "seed"})
    assert acm._request_keys({"tasks": ["music"]}, None) is None
    assert acm._request_keys(None, None) is None
    assert acm._request_keys({"schema_version": 1, "options": {"request": []}}, None) is None
    assert acm._request_keys(None, strict) == frozenset({"duration_sec", "seed"})


def test_status_reports_the_music_studio_rules():
    ace = audio_cpp_backend.model_info_fields(_model("ace_step", strict = False))["audio_music"]
    song, edit = ace["modes"]
    assert song == {
        "id": "song",
        "lyrics": "optional",
        "description": "required",
        "instrumental": "toggle",
        "section_case": "lower",
        "duration": {"min": 5.0, "max": 240.0, "default": 30.0, "approximate": False},
        "variations": None,
    }
    assert edit == {
        "id": "edit",
        "actions": ["repaint", "extend", "cover", "continue"],
        "max_ranges": 1,
        "max_source_s": 240.0,
    }
    stable = cm.music_rules(_model("stable_audio", "Stable-Audio-3-Medium-GGUF", strict = False))
    assert [m["id"] for m in stable["modes"]] == ["song", "sfx", "edit"]
    assert stable["modes"][0]["variations"] == {"max": 4, "how": "batch", "loaded": 1}
    assert stable["modes"][1]["duration"]["default"] == 8.0
    assert stable["modes"][2]["actions"] == ["inpaint", "restyle"]
    assert stable["modes"][2]["max_ranges"] == 8
    # Medium keeps Studio's 240 s; Small cannot return more than its ~120 s window.
    assert stable["modes"][0]["duration"]["max"] == 240.0
    assert stable["modes"][2]["max_source_s"] == 240.0
    small = cm.music_rules(_model("stable_audio", "Stable-Audio-3-Small-Music-GGUF", strict = False))
    assert small["modes"][0]["duration"]["max"] == 120.0
    assert small["modes"][1]["max_source_s"] == 120.0
    assert cm.music_rules(_model("controlfoley"))["modes"][0]["variations"] == {
        "max": 4,
        "how": "sequential",
        "loaded": 4,
    }
    minimax = cm.music_rules(_model("minimax_music3"))["modes"][0]
    assert minimax["duration"]["approximate"] and minimax["lyrics"] == "required"
    yue = cm.music_rules(_model("yue2"))["modes"][0]
    assert (yue["section_case"], yue["instrumental"]) == ("title", "toggle")
    assert cm.music_rules(_model("heartmula"))["modes"][0]["instrumental"] == "never"
    speech = replace(_model("ace_step"), task = "tts", music = None)
    assert audio_cpp_backend.model_info_fields(speech)["audio_music"] is None


def test_midashenglm_and_controlfoley_send_duration_sec_never_duration_seconds():
    for family, mode in (
        ("midashenglm_gen", "sfx"),
        ("controlfoley", "sfx"),
        ("midashenglm_gen", "song"),
    ):
        b = _backend(_model(family))
        _run(b, mode = mode, text = "rain on a tin roof", duration_s = 12, lyrics = "la la")
        request = _request(b)
        assert "duration_seconds" not in set(_keys(request)), family
        assert "lyrics" not in set(_keys(request)), family
        assert request == {
            "text": "rain on a tin roof",
            "options": {"duration_sec": 12.0},
            "seed": "7",
        }, family


def test_undeclared_and_edit_only_options_never_reach_a_strict_spec():
    b = _backend(_model("controlfoley"))
    _run(
        b,
        mode = "sfx",
        text = "glass",
        options = {"foo": 1, "route": "repaint", "batch_size": "4", "guidance_scale": 3.0},
    )
    assert _request(b) == {"text": "glass", "options": {"duration_sec": 8.0}, "seed": "7"}


def test_heartmula_takes_its_description_as_tags():
    b = _backend(_model("heartmula"))
    _run(b, text = "pop, female vocal", lyrics = "[verse] hi", duration_s = 20)
    assert _request(b) == {
        "text": "pop, female vocal",
        "options": {"tags": "pop, female vocal", "duration_sec": 20.0, "lyrics": "[verse] hi"},
        "seed": "7",
    }


def test_yue2_instrumental_sends_the_instrumental_tag_not_its_style():
    b = _backend(_model("yue2"))
    _run(b, text = "ambient piano", lyrics = "[Verse] words", instrumental = True, duration_s = 20)
    request = _request(b)
    assert request["text"] == "[Instrumental]"
    assert request["options"]["lyrics"] == "[Instrumental]"
    assert request["options"]["style"] == "ambient piano"
    assert request["options"]["semantic_max_tokens"] == 500


def test_minimax_gguf_keeps_its_body_through_the_studio():
    b = _backend(_model("minimax_music3"))
    _run(b, text = "lofi", lyrics = "[verse] hi", duration_s = 20, seed = 3)
    request = _request(b)
    assert request == {
        "text": "lofi",
        "lyrics": "[verse] hi",
        "duration_seconds": 20.0,
        "options": {"lyrics": "[verse] hi"},
        "seed": "3",
    }
    assert "duration_sec" not in set(_keys(request))


def test_ace_step_song_and_instrumental():
    b = _backend(_model("ace_step", strict = False))
    _run(
        b,
        text = "synth pop",
        lyrics = "[verse] la",
        duration_s = 45,
        options = {"bpm": 100, "timesignature": "3"},
    )
    assert _request(b) == {
        "text": "synth pop",
        "duration_seconds": 45.0,
        "lyrics": "[verse] la",
        "options": {"bpm": 100, "timesignature": "3"},
        "seed": "7",
    }
    _run(b, text = "synth pop", lyrics = "[verse] la", instrumental = True)
    request = _request(b, 1)
    assert request["lyrics"] == "[Instrumental]" and request["duration_seconds"] == 30.0


def test_durations_clamp_to_the_mode():
    b = _backend(_model("stable_audio", strict = False))
    _run(b, text = "house", duration_s = 500)
    assert _request(b)["duration_seconds"] == 120.0


def test_stable_audio_batches_variations_as_a_string(starts, tmp_path):
    model = _model("stable_audio", strict = False)
    b = _backend(model)
    run = tmp_path / "run"
    run.mkdir()
    wav, rate = _run(b, text = "house", duration_s = 6, variations = 3, seed = None, output_dir = str(run))
    request = b._server.calls[0][1]["request"]
    assert request["options"] == {"batch_size": "3"} and "seed" not in request
    assert rate == 44100 and wav[:4] == b"RIFF"
    manifest = json.loads((run / "outputs.json").read_text())
    assert [m["id"] for m in manifest] == ["audio_0"]  # the fake answers one take
    assert manifest[0]["file"] == "00.wav" and manifest[0]["sample_rate"] == 44100


def test_max_batch_reloads_once_and_only_when_needed(starts):
    model = _model("stable_audio", strict = False)
    b = _backend(model)
    _run(b, text = "house", variations = 1)
    assert starts == [] and b.take_status_patch() is None
    _run(b, text = "house", variations = 3)
    assert starts == [{"stable_audio.max_batch": "4"}]
    patch = b.take_status_patch()
    assert patch["audio_music"]["modes"][0]["variations"]["loaded"] == 4
    assert b.models[model.id]["audio_music"] == patch["audio_music"]
    assert b.take_status_patch() is None
    _run(b, text = "house", variations = 2)
    _run(b, text = "house", variations = 1)
    assert len(starts) == 1 and b.take_status_patch() is None
    b._server.alive = lambda: False
    _run(b, text = "house", variations = 4)
    assert starts[-1] == {"stable_audio.max_batch": "4"}
    assert b.take_status_patch() is None


def test_too_many_variations_fail_before_any_call(starts):
    b = _backend(_model("stable_audio", strict = False))
    with pytest.raises(RuntimeError, match = "at most 4"):
        _run(b, text = "house", variations = 5)
    assert b._server.calls == [] and starts == []
    one = _backend(_model("ace_step", strict = False))
    with pytest.raises(RuntimeError, match = "at most 1"):
        _run(one, text = "x", variations = 2)


def test_sequential_variations_call_once_per_take_with_consecutive_seeds(tmp_path):
    b = _backend(_model("controlfoley"))
    run = tmp_path / "run"
    _run(b, mode = "sfx", text = "door", variations = 3, seed = 40, output_dir = str(run))
    assert [_request(b, i)["seed"] for i in range(3)] == ["40", "41", "42"]
    manifest = json.loads((run / "outputs.json").read_text())
    assert [(m["id"], m["seed"], m["file"]) for m in manifest] == [
        ("take_0", 40, "00.wav"),
        ("take_1", 41, "01.wav"),
        ("take_2", 42, "02.wav"),
    ]
    assert all((run / m["file"]).is_file() for m in manifest)
    # The top valid seed wraps instead of passing MiDashengLM's 2**31 - 1 limit.
    top = _backend(_model("midashenglm_gen"))
    _run(top, mode = "sfx", text = "rain", variations = 2, seed = 2**31 - 2)
    assert [_request(top, i)["seed"] for i in range(2)] == [str(2**31 - 2), "0"]


def test_a_fixed_seed_family_gets_a_recorded_random_seed(tmp_path):
    b = _backend(_model("heartmula"))
    _run(b, text = "pop", lyrics = "la", seed = None, output_dir = str(tmp_path))
    seed = int(_request(b)["seed"])
    assert 0 <= seed < 2**31
    assert json.loads((tmp_path / "outputs.json").read_text())[0]["seed"] == seed
    sa = _backend(_model("stable_audio", strict = False))
    _run(sa, text = "house", seed = None)
    assert "seed" not in _request(sa)


def test_the_wait_grows_with_the_audio_asked_for_and_is_capped():
    t = lambda s, n = 1, cpu = False: cm.timeout_seconds("yue2", s, n, cpu)
    assert t(10) < t(60) < t(60, 3) < t(60, 3, cpu = True)
    assert t(0) == 300.0 and t(240, 4, cpu = True) == 4 * 3600.0
    assert cm.timeout_seconds("stable_audio", 10, 1, False) == 310.0
    b = _backend(_model("yue2"))
    _run(b, text = "rock", lyrics = "x", duration_s = 60)
    assert b._server.calls[0][2]["timeout"] == t(60)
    # The route's budget covers work the backend cannot see (extend, continue length).
    b = _backend(_model("yue2"))
    _run(b, text = "rock", lyrics = "x", duration_s = 60, timeout_s = 1234.0)
    assert b._server.calls[0][2]["timeout"] == 1234.0


@pytest.fixture
def source(tmp_path) -> str:
    path = tmp_path / "source.48000.stereo.wav"
    path.write_bytes(_wav(seconds = 10.0, rate = 48000))
    return str(path)


def test_ace_step_repaint_has_no_duration_and_may_pass_the_end(source):
    b = _backend(_model("ace_step", strict = False))
    _run(
        b,
        mode = "edit",
        text = "brighter",
        source = source,
        edit = {"action": "repaint", "ranges": [{"start_s": 5.0, "end_s": 14.0}], "strength": 0.4},
    )
    request = _request(b)
    assert request == {
        "text": "brighter",
        "audio": source,
        "options": {
            "route": "repaint",
            "repainting_start": 5.0,
            "repainting_end": 14.0,
            "repaint_strength": 0.4,
        },
        "seed": "7",
    }
    assert "duration_seconds" not in set(_keys(request))


def test_ace_step_extend_cover_and_continue(source):
    b = _backend(_model("ace_step", strict = False))
    _run(b, mode = "edit", text = "outro", source = source, edit = {"action": "extend", "extend_s": 12.0})
    assert _request(b, 0)["options"] == {
        "route": "repaint",
        "repainting_start": 10.0,
        "repainting_end": 22.0,
    }
    # strength is how much to change, so a light cover keeps most source-conditioned steps.
    _run(b, mode = "edit", text = "jazz", source = source, edit = {"action": "cover", "strength": 0.25})
    assert _request(b, 1)["options"] == {"route": "cover", "audio_cover_strength": 0.75}
    _run(
        b, mode = "edit", text = "add drums", source = source, duration_s = 30, edit = {"action": "continue"}
    )
    assert _request(b, 2)["options"] == {"route": "complete", "duration_seconds": 30.0}


def test_stable_audio_inpaint_sorts_and_merges_ranges(source):
    b = _backend(_model("stable_audio", strict = False))
    ranges = [
        {"start_s": 5.0, "end_s": 6.5},
        {"start_s": 1.0, "end_s": 2.0},
        {"start_s": 1.5, "end_s": 1.8},
        {"start_s": 9.5, "end_s": 30.0},
    ]
    _run(
        b,
        mode = "edit",
        text = "add birds",
        source = source,
        edit = {"action": "inpaint", "ranges": ranges},
    )
    request = _request(b)
    assert request["options"] == {
        "audio_input_kind": "inpaint_audio",
        "inpaint_mask_start_seconds": "1,5,9.5",
        "inpaint_mask_end_seconds": "2,6.5,10",
    }
    assert request["duration_seconds"] == 10.0 and request["audio"] == source


def test_stable_audio_restyle_sends_init_noise(source):
    b = _backend(_model("stable_audio", strict = False))
    _run(b, mode = "edit", text = "lofi", source = source, edit = {"action": "restyle", "strength": 0.45})
    assert _request(b)["options"] == {"audio_input_kind": "init_audio", "init_noise_level": 0.45}


def test_an_edit_the_family_does_not_offer_fails_before_any_call(source):
    b = _backend(_model("stable_audio", strict = False))
    with pytest.raises(RuntimeError, match = "cannot repaint"):
        _run(b, mode = "edit", text = "x", source = source, edit = {"action": "repaint", "ranges": []})
    with pytest.raises(RuntimeError, match = "does not offer"):
        _run(_backend(_model("heartmula")), mode = "edit", text = "x", source = source, edit = {})
    assert b._server.calls == []


def test_merge_ranges():
    assert cm.merge_ranges([(3, 4), (1, 2), (1.5, 3.5)], 10) == [(1.0, 4.0)]
    assert cm.merge_ranges([(-1, 2), (8, 12), (12, 13)], 10) == [(0.0, 2.0), (8.0, 10.0)]


def test_the_runtime_refusal_is_typed_for_the_route():
    from core.inference.audio_cpp_server import AudioCppRequestError
    from core.inference.audio_errors import AudioRuntimeError

    b = _backend(_model("controlfoley"))

    def refuse(path, payload, **kwargs):
        raise AudioCppRequestError(400, "unknown controlfoley request option: foo")

    b._server.post_json = refuse
    with pytest.raises(AudioRuntimeError, match = "unknown controlfoley"):
        _run(b, mode = "sfx", text = "door")


def test_legacy_requests_keep_their_bodies_with_the_finding_7_fixes():
    b = _backend(_model("midashenglm_gen"))
    b.generate_audio_response("wind", max_new_tokens = 250)
    assert _request(b) == {"text": "wind", "options": {"duration_sec": 10.0}}
    hm = _backend(_model("heartmula"))
    hm.generate_audio_response("[verse] hi", instructions = "pop", max_new_tokens = 500)
    assert _request(hm) == {
        "text": "pop",
        "options": {"tags": "pop", "duration_sec": 20.0, "lyrics": "[verse] hi"},
    }
    yue = _backend(_model("yue2"))
    yue.generate_audio_response("", instructions = "ambient")
    assert _request(yue)["options"]["lyrics"] == "[Instrumental]"
    assert "ambient" not in _request(yue)["text"]


def test_paths_in_the_manifest_are_bare_names(tmp_path):
    b = _backend(_model("controlfoley"))
    _run(b, mode = "sfx", text = "door", variations = 2, output_dir = str(tmp_path))
    for entry in json.loads((tmp_path / "outputs.json").read_text()):
        assert Path(entry["file"]).name == entry["file"]


def test_legacy_music_never_asks_past_the_page_maximum():
    b = _backend(_model("midashenglm_gen"))
    b.generate_audio_response("relaxing piano", instructions = "relaxing piano", max_new_tokens = 2048)
    assert _request(b)["options"]["duration_sec"] == 80.0
    medium = _backend(_model("stable_audio", "Stable-Audio-3-Medium-GGUF", strict = False))
    medium.generate_audio_response("ambient", instructions = "ambient", max_new_tokens = 4500)
    assert _request(medium)["duration_seconds"] == 180.0
