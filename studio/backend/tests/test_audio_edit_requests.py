# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The exact requests audio.cpp receives for an Edit run, per family. The golden cases share a
fixture with the frontend, which builds each case's run body; a fake server records the calls."""

from __future__ import annotations

import base64
import io
import json
import re
import threading
import wave
from pathlib import Path

import pytest

from core.inference import audio_cpp_backend as acb
from core.inference import audio_cpp_files
from core.inference import audio_cpp_models as acm
from core.inference import audio_edit
from core.inference.audio_cpp_models import AUDIO_CPP_REPO, AudioCppModel, AudioCppVariant, RepoFile
from core.inference.audio_errors import AudioGenerationCancelledError

FIXTURE = Path(__file__).resolve().parents[2] / "frontend/tests/fixtures/audio-edit-requests.json"
_CASES = json.loads(FIXTURE.read_text(encoding = "utf-8"))["cases"]
SOURCE = "/srv/accounts/a/audio/inputs/0123.24000.mono.wav"
REF = "/srv/accounts/a/audio/inputs/4567.24000.mono.m30.wav"
ORIGINAL = "Okay, I'm Cemo and what you just heard wasn't a human voice."
EDITED = "Okay, I'm Sam and what you just heard wasn't a robot voice."
MARKUP = (
    "Okay, I'm <sub targ=\"Sam\">Cemo</sub> and what you just heard wasn't a "
    '<sub targ="robot">human</sub> voice.'
)
TWO = ["Replace 'Cemo' with 'Sam'.", "Replace 'human' with 'robot'."]
MODEL_ID = "studio-test"
_NAMES = {
    "dots_tts": ("DotTTS-Edit-GGUF", "dots-tts-edit-q8_0.gguf"),
    "vevo2": ("Vevo2-GGUF", "vevo2-q8_0.gguf"),
    "firered_audio": ("FireRedAudio-GGUF", "firered-audio-q8_0.gguf"),
    "voxcpm2": ("VoxCPM2-GGUF", "voxcpm2-q8_0.gguf"),
}
_RULES = {
    "dots_tts": {"style": "markup", "delivery": False, "max_changes": None},
    "vevo2": {"style": "sentence", "delivery": False, "max_changes": None},
    "firered_audio": {"style": "instructions", "delivery": True, "max_changes": 5},
}


def _wav(frames):
    with wave.open(buf := io.BytesIO(), "wb") as w:
        w.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        w.writeframes(b"\x01\x00" * frames)
    return buf.getvalue()


# fmt: off
def _model(family, options = ()):
    folder, filename = _NAMES[family]
    policy = acm.family_policy(family, None, [f"{AUDIO_CPP_REPO}/{folder}", folder, filename])
    variant = AudioCppVariant("Q8_0", (RepoFile(f"{folder}/{filename}", 1),), f"{folder}/{filename}")
    return AudioCppModel(
        id = f"{AUDIO_CPP_REPO}/{folder}", repo_id = AUDIO_CPP_REPO, folder = folder, display_name = folder,
        family = family, server_task = policy.default_server_task, variant = variant, variants = (variant,),
        default_variant = "Q8_0", options = tuple(options), task = policy.task, speaks = policy.speaks,
        clone = policy.clone, edit = policy.edit,
    )
# fmt: on


class _Recorder:
    """A fake server: each call answers with a distinct WAV (FireRed's JSON shape, with a ``text``
    that is not the edited transcript), and checks a chained call reads the previous answer."""

    model_id, backend, cancel = MODEL_ID, "cpu", None
    alive = staticmethod(lambda: True)
    stop = staticmethod(lambda: None)

    def __init__(self, model):
        self.model = model
        self.field = model.edit.source_field if model.edit else None
        self.calls, self.answers, self.step_files = [], [], []

    def post_json(self, path, payload, **_kwargs):
        body = json.loads(json.dumps(payload))
        self.calls.append((path, body))
        read = body.get(self.field) if self.field else None
        if read and re.search(r"step\d+\.wav$", str(read)):
            assert Path(read).read_bytes() == self.answers[-1]
            self.step_files.append(read)
        answer = _wav(240 * (len(self.calls) + 1))
        self.answers.append(answer)
        if self.cancel:
            self.cancel.set()
        if path == "/v1/audio/speech":
            return "audio/wav", answer
        text = "Okay, I'm Cemo and what you just heard wasn't a  voice."
        out = {"text": text, "audio": base64.b64encode(answer).decode(), "sample_rate": 24000}
        return "application/json", json.dumps(out).encode()


@pytest.fixture
def started(monkeypatch, tmp_path):
    """Records the task each server start asks for, and keeps temp files under tmp_path."""
    starts = []
    monkeypatch.setattr(audio_cpp_files, "materialize", lambda m: str(tmp_path / "m.gguf"))
    start = lambda served, path, **_k: starts.append(served.server_task) or _Recorder(served)
    monkeypatch.setattr(acb.AudioCppServer, "start", start)
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path / "tmp"))
    return starts


def _backend(model, server = None):
    backend = acb.AudioCppBackend()
    backend._server, backend._model = server or _Recorder(model), model
    backend.models = {model.id: {"is_audio": True}}
    backend.active_model_name = model.id
    return backend


def _edit(backend, edit, **kwargs):
    kwargs = {
        "text": EDITED,
        "reference_text": ORIGINAL,
        "audio_inputs": {"source": SOURCE},
        **kwargs,
    }
    return backend.generate_audio_response(workflow = "edit", edit = edit, **kwargs)


@pytest.mark.parametrize("case", _CASES, ids = [c["name"] for c in _CASES])
def test_the_runtime_bodies_match_the_shared_fixture(case, started, tmp_path):
    backend = _backend(_model(case["family"]))
    run = case["run_body"]
    reference_text = run["inputs"].get("reference_text")
    options = case["input"]["advanced"] or None
    kwargs = {"text": run["text"], "reference_text": reference_text, "audio_options": options}
    wav, rate = _edit(backend, run["edit"], **kwargs)
    server = backend._server
    assert rate == 24000 and wav == server.answers[-1]
    names = {MODEL_ID: "<model>", SOURCE: "<recording>"}
    names |= {p: f"<result of call {i}>" for i, p in enumerate(server.step_files, 1)}
    swap = lambda body: {k: names.get(v, v) if isinstance(v, str) else v for k, v in body.items()}
    assert [{"path": path, "body": swap(body)} for path, body in server.calls] == case["runtime"]
    assert list((tmp_path / "tmp").iterdir()) == []


def test_a_firered_chain_reads_each_previous_output_and_a_cancel_stops_it(started):
    backend = _backend(_model("firered_audio"))
    instructions = [*TWO, "Delete 'Okay,'."]
    _edit(backend, {"mode": "words", "instructions": instructions}, text = EDITED[6:], seed = 7)
    server = backend._server
    assert server.calls[0][1]["audio"] == SOURCE
    assert [Path(p).name for p in server.step_files] == ["step1.wav", "step2.wav"]
    assert not Path(server.step_files[0]).parent.exists()
    sent = [(body["seed"], body["options"]["instruction"]) for _path, body in server.calls]
    assert sent == [("7", i) for i in instructions]
    server = _Recorder(_model("firered_audio"))
    server.cancel = cancel = threading.Event()
    with pytest.raises(AudioGenerationCancelledError):
        _edit(
            _backend(server.model, server),
            {"mode": "words", "instructions": TWO},
            cancel_event = cancel,
        )
    assert len(server.calls) == 1


def test_vevo2_edits_in_an_s2s_session_and_firered_in_its_clone_session(started):
    backend = _backend(_model("vevo2"))
    _edit(backend, {"mode": "words"})
    _edit(backend, {"mode": "words"}, reference_text = None)
    assert started == ["s2s"]
    body = backend._server.calls[-1][1]
    assert "reference_text" not in body and body["target_text"] == EDITED
    backend.generate_audio_response(EDITED, workflow = "clone", audio_inputs = {"reference": REF})
    assert started == ["s2s", "tts"]
    ((path, body),) = backend._server.calls
    assert path == "/v1/audio/speech" and body["voice_ref"] == REF
    backend = _backend(_model("firered_audio"))
    before = backend._server
    _edit(backend, {"mode": "words", "instructions": TWO[1:]})
    assert started == ["s2s", "tts"] and backend._server is before


# fmt: off
DOTS_OPTIONS = (
    {"name": "template_name", "type": "enum", "values": ["tts", "edit"]},
    {"name": "source_text", "type": "string"},
    {"name": "num_inference_steps", "type": "int", "min": 1, "max": 64},
    {"name": "use_xvector", "type": "bool"},
)
FIRERED_OPTIONS = (
    {"name": "template_name", "type": "enum", "values": ["semantic_edit", "acoustic_edit"]},
    {"name": "num_inference_steps", "type": "int", "min": 1},
)
CLAIMED = {name: "ignored" for name in ("source_text", "target_text", "instruction")}


@pytest.mark.parametrize("family, options, edit, sent, expected", [
    ("dots_tts", DOTS_OPTIONS, {"mode": "words", "markup": "x"},
     {"template_name": "tts", **CLAIMED, "num_inference_steps": 10.0, "use_xvector": True, "not_in_the_spec": 1},
     {"template_name": "edit", "num_inference_steps": "10", "use_xvector": "true"}),
    ("firered_audio", FIRERED_OPTIONS, {"mode": "delivery", "speed": 0.5},
     {"template_name": "semantic_edit", "num_inference_steps": 4},
     {"template_name": "acoustic_edit", "instruction": "adjust the speed to 0.5x", "num_inference_steps": "4"}),
])
# fmt: on
def test_advanced_options_leave_out_the_claimed_ones_and_unknown_ones(
    started, family, options, edit, sent, expected
):
    backend = _backend(_model(family, options))
    _edit(backend, edit, audio_options = sent)
    assert backend._server.calls[0][1]["options"] == expected


def test_a_saved_voice_clones_on_speak_and_an_edit_never_clones(started, monkeypatch):
    backend = _backend(_model("voxcpm2"))
    backend.generate_audio_response("Hi.", workflow = "speak", audio_inputs = {"reference": REF})
    ((path, body),) = backend._server.calls
    assert path == "/v1/audio/speech" and body["voice_ref"] == REF
    with pytest.raises(RuntimeError, match = "cannot edit speech"):
        _edit(backend, {"mode": "words"})
    assert len(backend._server.calls) == 1 and started == []
    no_clone = staticmethod(lambda *_a, **_k: pytest.fail("an edit took the clone path"))
    monkeypatch.setattr(acb.AudioCppBackend, "_generate_clone", no_clone)
    for family, edit in (
        ("firered_audio", {"mode": "words", "instructions": TWO[1:]}),
        ("dots_tts", {"mode": "words", "markup": 'a <sub targ="robot">human</sub> voice.'}),
    ):
        backend = _backend(_model(family))
        _edit(backend, edit)
        ((path, body),) = backend._server.calls
        assert path == "/v1/tasks/run" and "voice_ref" not in body


def test_status_fields_carry_the_edit_rules():
    for family in _NAMES:
        fields = acb.model_info_fields(_model(family))
        assert fields["audio_edit"] == _RULES.get(family), family
        assert ("edit" in fields["audio_workflows"]) == (family in _RULES), family
    assert acb.model_info_fields(_model("dots_tts"))["audio_workflows"] == [
        "speak",
        "clone",
        "edit",
    ]
    assert acb.model_info_fields(_model("vevo2"))["audio_workflows"] == ["clone", "edit"]


# fmt: off
def test_markup_sides_and_check_markup():
    assert audio_edit.markup_sides(MARKUP) == (ORIGINAL, EDITED)
    assert audio_edit.markup_sides("what you <del>just</del> heard <ins>really</ins> wasn't") == (
        "what you just heard wasn't", "what you heard really wasn't")
    for bad in ("a <b>human</b> voice", 'a <sub targ="robot">human voice', 'a <sub targ="x" onload="y">human</sub>', "a > b"):
        assert audio_edit.markup_sides(bad) is None
        assert audio_edit.check_markup(bad, ORIGINAL, EDITED) == audio_edit.MISMATCH
    for markup, original, edited, problem in (
        (MARKUP, ORIGINAL, EDITED, None),
        (ORIGINAL, ORIGINAL, ORIGINAL, audio_edit.NO_CHANGE),
        (MARKUP, ORIGINAL, ORIGINAL, audio_edit.MISMATCH),
        (MARKUP, "Something else.", EDITED, audio_edit.MISMATCH),
    ):
        assert audio_edit.check_markup(markup, original, edited) == problem


def test_check_instructions():
    unknown, mismatch = audio_edit.UNKNOWN_INSTRUCTION, audio_edit.MISMATCH
    for instructions, edited, problem in (
        (TWO, EDITED, None),
        (["Insert 'really' before 'wasn't'."], ORIGINAL.replace("heard", "heard really"), None),
        (["Delete 'Okay,'."], ORIGINAL[6:], None),
        ([], EDITED, audio_edit.NO_CHANGE),
        (["Replace 'human' with 'robot'."] * 6, EDITED,
         "FireRedAudio applies at most 5 changes. Make fewer changes, or use DotTTS Edit."),
        (["Replace 'human' with 'robot'"], EDITED, unknown),
        (["replace 'human' with 'robot'."], EDITED, unknown),
        (["Speak faster."], EDITED, unknown),
        (["Replace 'human' with 'robot'. Replace 'Cemo' with 'Sam'.x"], EDITED, unknown),
        (["Replace 'alien' with 'robot'."], EDITED, mismatch),
        (["Replace 'human' with 'cat'."], EDITED, mismatch),
        (["Replace 'Cemo' with 'human'."],
         ORIGINAL.replace("Cemo", "human").replace("a human voice", "a Cemo voice"), mismatch),
        (["Replace 'human' with 'robot'."], EDITED, mismatch),
        (["Delete 'Okay,'."], ORIGINAL.replace("Okay, I'm", "I'm Okay,"), mismatch),
        (["Replace 'human' with 'robot'."], ORIGINAL.replace("human", "robot"), None),
        (["Delete 'human'.", "Delete 'human'."], ORIGINAL.replace("a human voice", "a voice"), mismatch),
        (["Delete 'human'.", "Insert 'robot' before 'human'."], ORIGINAL.replace("human", "robot"), mismatch),
    ):
        check = audio_edit.check_instructions(instructions, ORIGINAL, edited, 5, "FireRedAudio")
        assert check == problem, instructions


def test_delivery_instructions():
    for speed, pitch, expected in (
        (1.5, None, ["adjust the speed to 1.5x"]),
        (2.0, None, ["adjust the speed to 2x"]),
        (1.0, 3, ["shift the pitch by 3 steps"]),
        (0.5, 2, ["adjust the speed to 0.5x", "shift the pitch by 2 steps"]),
        (None, None, []),
    ):
        assert audio_edit.delivery_instructions(speed, pitch) == expected


def test_request_problem_per_style():
    dots, vevo, firered = _RULES.values()
    words, no_change = {"mode": "words"}, audio_edit.NO_CHANGE
    for rules, edit, text, reference, problem in (
        (dots, words, EDITED, ORIGINAL, "DotTTS-Edit needs the marked-up changes."),
        (dots, words, ORIGINAL, ORIGINAL, no_change),
        (dots, {"mode": "delivery", "speed": 1.5}, EDITED, ORIGINAL, "Delivery changes need FireRedAudio."),
        (vevo, words, ORIGINAL, ORIGINAL, no_change),
        (vevo, words, EDITED, None, None),
        (firered, {"mode": "delivery", "speed": 1.0}, EDITED, ORIGINAL, "Pick a speed or a pitch change."),
        (firered, {"mode": "delivery", "pitch_steps": 3}, ORIGINAL, ORIGINAL, None),
        (firered, words, EDITED, ORIGINAL, no_change),
        (dots, {"mode": "words", "markup": "<del>" + ORIGINAL + "</del>"}, " ", ORIGINAL, audio_edit.EMPTY_TARGET),
        (vevo, words, "", ORIGINAL, audio_edit.EMPTY_TARGET),
        (firered, {"mode": "words", "instructions": ["Delete 'a'."]}, "  ", ORIGINAL, audio_edit.EMPTY_TARGET),
        (dots, {"mode": "words", "markup": ORIGINAL.replace("Cemo", '<sub targ="Cemo">Cemo</sub>')}, ORIGINAL, ORIGINAL, no_change),
        (firered, {"mode": "words", "instructions": ["Replace 'Cemo' with 'Cemo'."]}, ORIGINAL, ORIGINAL, no_change),
    ):
        assert audio_edit.request_problem(rules, edit, text, reference, "DotTTS-Edit") == problem
# fmt: on
