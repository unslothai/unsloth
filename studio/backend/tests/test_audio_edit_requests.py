# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The exact requests audio.cpp receives for an Edit run, per family. The golden cases share a
fixture with the frontend, which builds each case's run body; a fake server records the calls."""

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
    backend._server = server or _Recorder(model, model.edit.source_field if model.edit else None)
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


def _edit(
    backend,
    edit,
    text = EDITED,
    reference_text = ORIGINAL,
    **kwargs,
):
    return backend.generate_audio_response(
        text,
        workflow = "edit",
        audio_inputs = {"source": SOURCE},
        reference_text = reference_text,
        edit = edit,
        **kwargs,
    )


_TWO = ["Replace 'Cemo' with 'Sam'.", "Replace 'human' with 'robot'."]


@pytest.mark.parametrize("case", _CASES, ids = [c["name"] for c in _CASES])
def test_the_runtime_bodies_match_the_shared_fixture(case, started, tmp_path):
    backend = _backend(_model(case["family"]))
    run = case["run_body"]
    wav, rate = _edit(
        backend,
        run["edit"],
        run["text"],
        run["inputs"].get("reference_text"),
        audio_options = case["input"]["advanced"] or None,
    )
    server = backend._server
    assert rate == 24000 and wav == server.answers[-1]
    sent = [{"path": path, "body": _placeholders(body, server)} for path, body in server.calls]
    assert sent == case["runtime"]
    assert list((tmp_path / "tmp").iterdir()) == []


def test_a_firered_chain_reads_each_previous_output(started):
    backend = _backend(_model("firered_audio"))
    instructions = [*_TWO, "Delete 'Okay,'."]
    _edit(backend, {"mode": "words", "instructions": instructions}, EDITED[6:], seed = 7)
    server = backend._server
    assert server.calls[0][1]["audio"] == SOURCE
    assert [Path(p).name for p in server.step_files] == ["step1.wav", "step2.wav"]
    assert not Path(server.step_files[0]).parent.exists()
    assert [body["seed"] for _p, body in server.calls] == ["7", "7", "7"]
    assert [body["options"]["instruction"] for _p, body in server.calls] == instructions


def test_a_cancel_between_steps_stops_the_chain(started):
    model = _model("firered_audio")
    cancel = threading.Event()
    server = _Recorder(model, "audio", cancel_after = 1, cancel_event = cancel)
    with pytest.raises(AudioGenerationCancelledError):
        _edit(
            _backend(model, server),
            {"mode": "words", "instructions": _TWO},
            cancel_event = cancel,
        )
    assert len(server.calls) == 1


def test_vevo2_edits_as_s2s_and_the_next_clone_restarts_as_tts(started):
    backend = _backend(_model("vevo2"))
    _edit(backend, {"mode": "words"})
    _edit(backend, {"mode": "words"}, reference_text = None)
    assert started == ["s2s"]
    ((_path, body),) = backend._server.calls[-1:]
    assert "reference_text" not in body and body["target_text"] == EDITED
    backend.generate_audio_response(EDITED, workflow = "clone", audio_inputs = {"reference": REF})
    assert started == ["s2s", "tts"]
    ((path, body),) = backend._server.calls
    assert path == "/v1/audio/speech" and body["voice_ref"] == REF


def test_firered_edits_in_the_clone_session_without_a_restart(started):
    backend = _backend(_model("firered_audio"))
    before = backend._server
    _edit(backend, {"mode": "words", "instructions": _TWO[1:]})
    assert started == [] and backend._server is before


def test_advanced_options_leave_out_the_claimed_ones_and_unknown_ones(started):
    dots_options = (
        {"name": "template_name", "type": "enum", "values": ["tts", "edit"]},
        {"name": "source_text", "type": "string"},
        {"name": "num_inference_steps", "type": "int", "min": 1, "max": 64},
        {"name": "use_xvector", "type": "bool"},
    )
    backend = _backend(_model("dots_tts", options = dots_options))
    claimed = {name: "ignored" for name in ("source_text", "target_text", "instruction")}
    _edit(
        backend,
        {"mode": "words", "markup": "x"},
        audio_options = {
            "template_name": "tts",
            **claimed,
            "num_inference_steps": 10.0,
            "use_xvector": True,
            "not_in_the_spec": 1,
        },
    )
    assert backend._server.calls[0][1]["options"] == {
        "template_name": "edit",
        "num_inference_steps": "10",
        "use_xvector": "true",
    }
    firered_options = (
        {"name": "template_name", "type": "enum", "values": ["semantic_edit", "acoustic_edit"]},
        {"name": "num_inference_steps", "type": "int", "min": 1},
    )
    backend = _backend(_model("firered_audio", options = firered_options))
    _edit(
        backend,
        {"mode": "delivery", "speed": 0.5},
        audio_options = {"template_name": "semantic_edit", "num_inference_steps": 4},
    )
    assert backend._server.calls[0][1]["options"] == {
        "template_name": "acoustic_edit",
        "instruction": "adjust the speed to 0.5x",
        "num_inference_steps": "4",
    }


def test_speak_with_a_saved_voice_still_clones_and_edit_never_does(started, monkeypatch):
    # An edit carries audio_inputs too; that must not make it a clone.
    backend = _backend(_model("voxcpm2"))
    backend.generate_audio_response(
        "Hello there.", workflow = "speak", audio_inputs = {"reference": REF}
    )
    ((path, body),) = backend._server.calls
    assert path == "/v1/audio/speech" and body["voice_ref"] == REF

    def no_clone(*_args, **_kwargs):
        raise AssertionError("an edit took the clone path")

    monkeypatch.setattr(acb.AudioCppBackend, "_generate_clone", staticmethod(no_clone))
    for family, edit in (
        ("firered_audio", {"mode": "words", "instructions": _TWO[1:]}),
        ("dots_tts", {"mode": "words", "markup": 'a <sub targ="robot">human</sub> voice.'}),
    ):
        backend = _backend(_model(family))
        _edit(backend, edit)
        ((path, body),) = backend._server.calls
        assert path == "/v1/tasks/run" and "voice_ref" not in body


def test_a_model_that_cannot_edit_refuses_before_any_call(started):
    backend = _backend(_model("voxcpm2"))
    with pytest.raises(RuntimeError, match = "cannot edit speech"):
        _edit(backend, {"mode": "words"})
    assert backend._server.calls == [] and started == []


@pytest.mark.parametrize(
    "family, workflows, rules",
    [
        ("dots_tts", ["speak", "edit"], ("markup", False, None)),
        ("vevo2", ["clone", "edit"], ("sentence", False, None)),
        ("firered_audio", None, ("instructions", True, 5)),
        ("voxcpm2", None, None),
    ],
)
def test_status_fields_carry_the_edit_rules(family, workflows, rules):
    fields = acb.model_info_fields(_model(family))
    if workflows:
        assert fields["audio_workflows"] == workflows
    expected = rules and dict(zip(("style", "delivery", "max_changes"), rules))
    assert fields["audio_edit"] == expected


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


@pytest.mark.parametrize(
    "speed, pitch, expected",
    [
        (1.5, None, ["adjust the speed to 1.5x"]),
        (2.0, None, ["adjust the speed to 2x"]),
        (1.0, 3, ["shift the pitch by 3 steps"]),
        (0.5, 2, ["adjust the speed to 0.5x", "shift the pitch by 2 steps"]),
        (None, None, []),
    ],
)
def test_delivery_instructions(speed, pitch, expected):
    assert audio_edit.delivery_instructions(speed, pitch) == expected


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
