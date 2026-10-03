# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The exact requests audio.cpp receives for an Edit speech run, per family (spike S2's bodies).

The golden cases are the shared fixture the frontend's Request preview is tested against, so the
preview and the server agree call for call. A fake server records every ``post_json``; no runtime
runs.
"""

from __future__ import annotations

import base64
import json
import os
import re
import threading
import wave
from pathlib import Path

import pytest

from core.inference import audio_cpp_backend as acb
from core.inference import audio_cpp_files
from core.inference import audio_cpp_models as acm
from core.inference import audio_edit
from core.inference.audio_cpp_models import (
    AUDIO_CPP_REPO,
    AudioCppModel,
    AudioCppVariant,
    RepoFile,
)
from core.inference.audio_errors import AudioGenerationCancelledError

FIXTURE = Path(__file__).resolve().parents[2] / "frontend/tests/fixtures/audio-edit-requests.json"
SOURCE = "/srv/accounts/a/audio/inputs/0123.24000.mono.wav"
REF = "/srv/accounts/a/audio/inputs/4567.24000.mono.m30.wav"
ORIGINAL = "Okay, I'm Cemo and what you just heard wasn't a human voice."
EDITED = "Okay, I'm Sam and what you just heard wasn't a robot voice."
MODEL_ID = "studio-test"

_NAMES = {
    "dots_tts": ("DotTTS-Edit-GGUF", "dots-tts-edit-q8_0.gguf"),
    "vevo2": ("Vevo2-GGUF", "vevo2-q8_0.gguf"),
    "firered_audio": ("FireRedAudio-GGUF", "firered-audio-q8_0.gguf"),
    "voxcpm2": ("VoxCPM2-GGUF", "voxcpm2-q8_0.gguf"),
}


def _wav(frames: int, rate = 24000) -> bytes:
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * frames)
    return buf.getvalue()


def _model(family, options = ()) -> AudioCppModel:
    folder, filename = _NAMES[family]
    policy = acm.family_policy(family, None, [f"{AUDIO_CPP_REPO}/{folder}", folder, filename])
    main = f"{folder}/{filename}"
    variant = AudioCppVariant("Q8_0", (RepoFile(main, 1),), main)
    return AudioCppModel(
        id = f"{AUDIO_CPP_REPO}/{folder}",
        repo_id = AUDIO_CPP_REPO,
        folder = folder,
        display_name = folder,
        family = family,
        task = policy.task,
        server_task = policy.default_server_task,
        variant = variant,
        variants = (variant,),
        default_variant = "Q8_0",
        options = tuple(options),
        speaks = policy.speaks,
        clone = policy.clone,
        edit = policy.edit,
    )


class _Recorder:
    """A fake server: each call answers with a distinct WAV (FireRed's JSON shape, with a ``text``
    that is not the edited transcript), and checks a chained call reads the previous answer."""

    model_id = MODEL_ID
    backend = "cpu"

    def __init__(
        self,
        model,
        field = None,
        cancel_after = None,
        cancel_event = None,
    ):
        self.model = model
        self.field = field
        self.calls: list[tuple[str, dict]] = []
        self.answers: list[bytes] = []
        self.step_files: list[str] = []
        self.cancel_after = cancel_after
        self.cancel_event = cancel_event

    def alive(self):
        return True

    def stop(self):
        pass

    def post_json(self, path, payload, **_kwargs):
        body = json.loads(json.dumps(payload))
        self.calls.append((path, body))
        read = body.get(self.field) if self.field else None
        if read and re.search(r"step\d+\.wav$", str(read)):
            # A chained call reads the previous answer from the temp WAV the server wrote.
            assert Path(read).read_bytes() == self.answers[-1]
            self.step_files.append(read)
        answer = _wav(240 * (len(self.calls) + 1))
        self.answers.append(answer)
        if self.cancel_after is not None and len(self.calls) >= self.cancel_after:
            self.cancel_event.set()
        if path == "/v1/audio/speech":
            return "audio/wav", answer
        out = {
            "text": "Okay, I'm Cemo and what you just heard wasn't a  voice.",
            "audio": base64.b64encode(answer).decode(),
            "sample_rate": 24000,
        }
        return "application/json", json.dumps(out).encode()


@pytest.fixture
def started(monkeypatch, tmp_path):
    """Records the task each server start asks for, and keeps temp files under tmp_path."""
    starts: list[str] = []
    monkeypatch.setattr(audio_cpp_files, "materialize", lambda m: str(tmp_path / "m.gguf"))

    def start(served, path, **_kwargs):
        starts.append(served.server_task)
        field = served.edit.source_field if served.edit else None
        return _Recorder(served, field)

    monkeypatch.setattr(acb.AudioCppServer, "start", start)
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr("tempfile.tempdir", str(scratch))
    return starts


def _backend(model, server = None):
    backend = acb.AudioCppBackend()
    backend._server = (
        server
        if server is not None
        else _Recorder(model, model.edit.source_field if model.edit else None)
    )
    backend._model = model
    backend.models = {model.id: {"is_audio": True}}
    backend.active_model_name = model.id
    return backend


def _placeholders(body: dict, server: _Recorder) -> dict:
    out = {}
    for key, value in body.items():
        if value == MODEL_ID:
            value = "<model>"
        elif value == SOURCE:
            value = "<recording>"
        elif isinstance(value, str) and value in server.step_files:
            index = int(re.search(r"step(\d+)\.wav$", value).group(1))
            value = f"<result of call {index}>"
        out[key] = value
    return out


_CASES = json.loads(FIXTURE.read_text(encoding = "utf-8"))["cases"]


@pytest.mark.parametrize("case", _CASES, ids = [c["name"] for c in _CASES])
def test_the_runtime_bodies_match_the_shared_fixture(case, started, tmp_path):
    model = _model(case["family"])
    backend = _backend(model)
    run = case["run_body"]
    wav, rate = backend.generate_audio_response(
        run["text"],
        workflow = "edit",
        audio_inputs = {"source": SOURCE},
        reference_text = run["inputs"].get("reference_text"),
        edit = run["edit"],
        audio_options = case["input"]["advanced"] or None,
    )
    server = backend._server
    assert wav[:4] == b"RIFF" and rate == 24000
    # The last call's answer is the result.
    assert wav == server.answers[-1]
    sent = [{"path": path, "body": _placeholders(body, server)} for path, body in server.calls]
    assert sent == case["runtime"]
    # The pure builder the worker uses gives the same calls with the placeholders in place.
    assert (
        acb.edit_request_bodies(
            model,
            run["text"],
            run["inputs"].get("reference_text"),
            run["edit"],
            {},
            None,
        )
        == case["runtime"]
    )
    # Every chain step's temp WAV is gone afterwards.
    assert all(not os.path.exists(step) for step in server.step_files)
    assert list((tmp_path / "tmp").iterdir()) == []


def test_a_firered_chain_reads_each_previous_output(started):
    model = _model("firered_audio")
    backend = _backend(model)
    instructions = [
        "Replace 'Cemo' with 'Sam'.",
        "Replace 'human' with 'robot'.",
        "Delete 'Okay,'.",
    ]
    backend.generate_audio_response(
        "I'm Sam and what you just heard wasn't a robot voice.",
        workflow = "edit",
        audio_inputs = {"source": SOURCE},
        reference_text = ORIGINAL,
        edit = {"mode": "words", "instructions": instructions},
        seed = 7,
    )
    server = backend._server
    assert [body["audio"] for _p, body in server.calls][0] == SOURCE
    assert len(server.step_files) == 2
    assert [Path(p).name for p in server.step_files] == ["step1.wav", "step2.wav"]
    # One shared temp dir, removed with its files.
    assert Path(server.step_files[0]).parent.name.startswith("unsloth-audio-edit-")
    assert not Path(server.step_files[0]).parent.exists()
    # The seed goes top-level, as a string like the clone path, on every call.
    assert [body["seed"] for _p, body in server.calls] == ["7", "7", "7"]
    assert [body["options"]["instruction"] for _p, body in server.calls] == instructions


def test_a_cancel_between_steps_stops_the_chain(started):
    model = _model("firered_audio")
    cancel = threading.Event()
    server = _Recorder(model, "audio", cancel_after = 1, cancel_event = cancel)
    backend = _backend(model, server)
    with pytest.raises(AudioGenerationCancelledError):
        backend.generate_audio_response(
            EDITED,
            workflow = "edit",
            audio_inputs = {"source": SOURCE},
            reference_text = ORIGINAL,
            edit = {
                "mode": "words",
                "instructions": ["Replace 'Cemo' with 'Sam'.", "Replace 'human' with 'robot'."],
            },
            cancel_event = cancel,
        )
    assert len(server.calls) == 1


def test_vevo2_edits_as_s2s_and_the_next_clone_restarts_as_tts(started):
    model = _model("vevo2")
    assert model.server_task == "tts"
    backend = _backend(model)
    backend.generate_audio_response(
        EDITED,
        workflow = "edit",
        audio_inputs = {"source": SOURCE},
        reference_text = ORIGINAL,
        edit = {"mode": "words"},
    )
    assert started == ["s2s"]
    assert backend._server.model.server_task == "s2s"
    # A second edit keeps the s2s session.
    backend.generate_audio_response(
        EDITED, workflow = "edit", audio_inputs = {"source": SOURCE}, edit = {"mode": "words"}
    )
    assert started == ["s2s"]
    backend.generate_audio_response(EDITED, workflow = "clone", audio_inputs = {"reference": REF})
    assert started == ["s2s", "tts"]
    (call,) = backend._server.calls
    assert call[0] == "/v1/audio/speech" and call[1]["voice_ref"] == REF


def test_vevo2_without_a_transcript_leaves_reference_text_out(started):
    backend = _backend(_model("vevo2"))
    backend.generate_audio_response(
        EDITED, workflow = "edit", audio_inputs = {"source": SOURCE}, edit = {"mode": "words"}
    )
    ((_path, body),) = backend._server.calls
    assert "reference_text" not in body and body["target_text"] == EDITED


def test_firered_edits_in_the_clone_session_without_a_restart(started):
    model = _model("firered_audio")
    backend = _backend(model)
    before = backend._server
    backend.generate_audio_response(
        EDITED,
        workflow = "edit",
        audio_inputs = {"source": SOURCE},
        reference_text = ORIGINAL,
        edit = {"mode": "words", "instructions": ["Replace 'human' with 'robot'."]},
    )
    assert started == [] and backend._server is before


_DOTS_OPTIONS = (
    {"name": "template_name", "type": "enum", "values": ["tts", "edit"]},
    {"name": "source_text", "type": "string"},
    {"name": "target_text", "type": "string"},
    {"name": "num_inference_steps", "type": "int", "min": 1, "max": 64},
    {"name": "use_xvector", "type": "bool"},
)


def test_advanced_options_leave_out_the_claimed_ones_and_unknown_ones(started):
    backend = _backend(_model("dots_tts", options = _DOTS_OPTIONS))
    backend.generate_audio_response(
        EDITED,
        workflow = "edit",
        audio_inputs = {"source": SOURCE},
        reference_text = ORIGINAL,
        edit = {"mode": "words", "markup": "x"},
        audio_options = {
            "template_name": "tts",
            "source_text": "ignored",
            "target_text": "ignored",
            "instruction": "ignored",
            "num_inference_steps": 10.0,
            "use_xvector": True,
            "not_in_the_spec": 1,
        },
    )
    ((_path, body),) = backend._server.calls
    assert body["options"] == {
        "template_name": "edit",
        "num_inference_steps": "10",
        "use_xvector": "true",
    }
    firered_options = (
        {"name": "template_name", "type": "enum", "values": ["semantic_edit", "acoustic_edit"]},
        {"name": "num_inference_steps", "type": "int", "min": 1},
    )
    backend = _backend(_model("firered_audio", options = firered_options))
    backend.generate_audio_response(
        EDITED,
        workflow = "edit",
        audio_inputs = {"source": SOURCE},
        edit = {"mode": "delivery", "speed": 0.5},
        audio_options = {"template_name": "semantic_edit", "num_inference_steps": 4},
    )
    ((_path, body),) = backend._server.calls
    assert body["options"] == {
        "template_name": "acoustic_edit",
        "instruction": "adjust the speed to 0.5x",
        "num_inference_steps": "4",
    }


def test_speak_with_a_saved_voice_still_clones_and_edit_never_does(started, monkeypatch):
    # F-6: an edit carries audio_inputs too, which must not make it a clone; Speak in a saved voice
    # relies on exactly that rule.
    backend = _backend(_model("voxcpm2"))
    backend.generate_audio_response(
        "Hello there.", workflow = "speak", audio_inputs = {"reference": REF}
    )
    ((path, body),) = backend._server.calls
    assert path == "/v1/audio/speech" and body["voice_ref"] == REF

    def no_clone(*_args, **_kwargs):
        raise AssertionError("an edit took the clone path")

    monkeypatch.setattr(acb.AudioCppBackend, "_generate_clone", staticmethod(no_clone))
    backend = _backend(_model("firered_audio"))
    backend.generate_audio_response(
        EDITED,
        workflow = "edit",
        audio_inputs = {"source": SOURCE},
        reference_text = ORIGINAL,
        edit = {"mode": "words", "instructions": ["Replace 'human' with 'robot'."]},
    )
    ((path, body),) = backend._server.calls
    assert path == "/v1/tasks/run" and "voice_ref" not in body
    # DotTTS Edit cannot clone; read as a clone, its edit would be refused.
    dots = _model("dots_tts")
    assert dots.clone is None
    backend = _backend(dots)
    backend.generate_audio_response(
        EDITED,
        workflow = "edit",
        audio_inputs = {"source": SOURCE},
        reference_text = ORIGINAL,
        edit = {"mode": "words", "markup": 'a <sub targ="robot">human</sub> voice.'},
    )
    ((path, body),) = backend._server.calls
    assert path == "/v1/tasks/run" and body["source_audio"] == SOURCE


def test_a_model_that_cannot_edit_refuses_before_any_call(started):
    backend = _backend(_model("voxcpm2"))
    with pytest.raises(RuntimeError, match = "cannot edit speech"):
        backend.generate_audio_response(
            EDITED, workflow = "edit", audio_inputs = {"source": SOURCE}, edit = {"mode": "words"}
        )
    assert backend._server.calls == [] and started == []


def test_status_fields_carry_the_edit_rules():
    dots = acb.model_info_fields(_model("dots_tts"))
    assert dots["audio_workflows"] == ["speak", "edit"]
    assert dots["audio_edit"] == {
        "style": "markup",
        "delivery": False,
        "max_changes": None,
        "input_rate": 24000,
    }
    vevo = acb.model_info_fields(_model("vevo2"))
    assert vevo["audio_workflows"] == ["clone", "edit"]
    assert vevo["audio_edit"]["style"] == "sentence" and vevo["audio_edit"]["delivery"] is False
    firered = acb.model_info_fields(_model("firered_audio"))
    assert firered["audio_edit"] == {
        "style": "instructions",
        "delivery": True,
        "max_changes": 5,
        "input_rate": 24000,
    }
    assert acb.model_info_fields(_model("voxcpm2"))["audio_edit"] is None


# The pure checks.


def test_markup_sides_and_check_markup():
    markup = (
        "Okay, I'm <sub targ=\"Sam\">Cemo</sub> and what you just heard wasn't a "
        '<sub targ="robot">human</sub> voice.'
    )
    assert audio_edit.markup_sides(markup) == (ORIGINAL, EDITED)
    assert audio_edit.check_markup(markup, ORIGINAL, EDITED) is None
    deleted = "what you <del>just</del> heard <ins>really</ins> wasn't"
    assert audio_edit.markup_sides(deleted) == (
        "what you just heard wasn't",
        "what you heard really wasn't",
    )
    # Anything but the three tags, or a tag left open, is refused.
    for bad in (
        "a <b>human</b> voice",
        'a <sub targ="robot">human voice',
        'a <sub targ="x" onload="y">human</sub>',
        "a > b",
    ):
        assert audio_edit.markup_sides(bad) is None
        assert audio_edit.check_markup(bad, ORIGINAL, EDITED) == audio_edit.MISMATCH
    assert audio_edit.check_markup(ORIGINAL, ORIGINAL, ORIGINAL) == audio_edit.NO_CHANGE
    assert audio_edit.check_markup(markup, ORIGINAL, ORIGINAL) == audio_edit.MISMATCH
    assert audio_edit.check_markup(markup, "Something else.", EDITED) == audio_edit.MISMATCH


def test_check_instructions():
    check = audio_edit.check_instructions
    two = ["Replace 'Cemo' with 'Sam'.", "Replace 'human' with 'robot'."]
    assert check(two, ORIGINAL, EDITED, 5) is None
    inserted = ORIGINAL.replace("heard", "heard really")
    assert check(["Insert 'really' before 'wasn't'."], ORIGINAL, inserted, 5) is None
    assert check(["Delete 'Okay,'."], ORIGINAL, ORIGINAL[6:], 5) is None
    assert check([], ORIGINAL, EDITED, 5) == audio_edit.NO_CHANGE
    assert check(["Replace 'human' with 'robot'."] * 6, ORIGINAL, EDITED, 5, "FireRedAudio") == (
        "FireRedAudio applies at most 5 changes. Make fewer changes, or use DotTTS Edit."
    )
    for bad in (
        "Replace 'human' with 'robot'",
        "replace 'human' with 'robot'.",
        "Speak faster.",
        "Replace 'human' with 'robot'. Replace 'Cemo' with 'Sam'.x",
    ):
        assert check([bad], ORIGINAL, EDITED, 5) == audio_edit.UNKNOWN_INSTRUCTION
    # Words that are not in the transcripts.
    assert check(["Replace 'alien' with 'robot'."], ORIGINAL, EDITED, 5) == audio_edit.MISMATCH
    assert check(["Replace 'human' with 'cat'."], ORIGINAL, EDITED, 5) == audio_edit.MISMATCH


def test_delivery_instructions():
    assert audio_edit.delivery_instructions(1.5, None) == ["adjust the speed to 1.5x"]
    assert audio_edit.delivery_instructions(2.0, None) == ["adjust the speed to 2x"]
    assert audio_edit.delivery_instructions(1.0, 3) == ["shift the pitch by 3 steps"]
    assert audio_edit.delivery_instructions(0.5, 2) == [
        "adjust the speed to 0.5x",
        "shift the pitch by 2 steps",
    ]
    assert audio_edit.delivery_instructions(None, None) == []


def test_request_problem_per_style():
    dots = {"style": "markup", "delivery": False, "max_changes": None}
    vevo = {"style": "sentence", "delivery": False, "max_changes": None}
    firered = {"style": "instructions", "delivery": True, "max_changes": 5}
    problem = audio_edit.request_problem
    assert problem(dots, {"mode": "words"}, EDITED, ORIGINAL, "DotTTS-Edit") == (
        "DotTTS-Edit needs the marked-up changes."
    )
    assert problem(dots, {"mode": "words"}, ORIGINAL, ORIGINAL, "x") == audio_edit.NO_CHANGE
    assert problem(dots, {"mode": "delivery", "speed": 1.5}, EDITED, ORIGINAL, "x") == (
        "Delivery changes need FireRedAudio."
    )
    assert problem(vevo, {"mode": "words"}, ORIGINAL, ORIGINAL, "x") == audio_edit.NO_CHANGE
    assert problem(vevo, {"mode": "words"}, EDITED, None, "x") is None
    assert problem(firered, {"mode": "delivery", "speed": 1.0}, EDITED, ORIGINAL, "x") == (
        "Pick a speed or a pitch change."
    )
    assert problem(firered, {"mode": "delivery", "pitch_steps": 3}, ORIGINAL, ORIGINAL, "x") is None
    assert problem(firered, {"mode": "words"}, EDITED, ORIGINAL, "x") == audio_edit.NO_CHANGE
