# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which Audio pages a family lands on, from the FAMILIES table or its spec's tasks, and which of
its GGUFs is the default."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from core.inference import audio_cpp_models as acm
from core.inference.audio_cpp_models import AudioCppVariant, RepoFile

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_audio_cpp_clone_requests import REF, SPEECH, T, TEXT, _backend  # noqa: E402
from test_audio_cpp_models import _gguf_bytes, _put, _snapshot, hub  # noqa: E402, F401


def _req(*names: str) -> dict:
    return {"request": [{"name": n, "type": "string", "required": False} for n in names]}


# Trimmed from audio.cpp's model_specs (HEAD).
DRAMA = {
    "tasks": ["tts", "clone"],
    "options": _req("reference_text", "references", "seed"),
    "ui": {"recommended_package": "fish_drama3_q4_experts_q8_backbone"},
    "packages": [
        {
            "id": f"fish_drama3_{p}",
            "default": p == "q4_experts_q8_backbone",
            "format": "gguf",
            "files": [f"drama-3-backbone-{f}.gguf"],
        }
        for p, f in (
            ("q4_experts_q8_backbone", "8bit-experts-Q4_K_S"),
            ("q3_experts_q8_backbone", "8bit-experts-Q3_K_S"),
            ("q2_experts_q8_backbone", "8bit-experts-Q2_K"),
            ("q4_backbone", "4bit-experts-Q4_K_S"),
        )
    ],
}
KITTEN2 = {
    "tasks": ["tts", "clone"],
    "options": _req("voice_id", "seed", "reference_text"),
    "ui": {"recommended_package": "kitten_tts2_q8_v2"},
    "packages": [
        {
            "id": f"kitten_tts2_{q}",
            "default": q == "q8_v2",
            "format": "gguf",
            "files": [f"kitten-tts2-{q.replace('_', '-')}-multilingual.gguf", "LICENSE"],
        }
        for q in ("bf16", "q8_v2", "q4")
    ],
}
SOPRO = {
    "tasks": ["tts", "clone"],
    "ui": {"recommended_package": "sopro_v2_turbo_safetensors"},
    "packages": [
        {
            "id": "sopro_v2_turbo_f16",
            "default": True,
            "format": "gguf",
            "files": ["sopro-v2-turbo-f16.gguf"],
        },
        {
            "id": "sopro_v2_turbo_safetensors",
            "format": "safetensors",
            "files": ["model.safetensors"],
        },
    ],
}
LFM2 = {
    "tasks": ["asr", "tts", "s2s"],
    "options": _req("language", "seed"),
    "ui": {"recommended_package": "lfm2_audio_1_5b_q8_0"},
    "packages": [
        {
            "id": "lfm2_audio_1_5b_q8_0",
            "default": True,
            "format": "gguf",
            "files": [
                f"{p}LFM2.5-Audio-1.5B-Q8_0.gguf" for p in ("", "mmproj-", "vocoder-", "tokenizer-")
            ],
        }
    ],
}
MUSIC = {"tasks": ["music"], "options": _req("lyrics", "duration_sec")}
SFX = {
    "tasks": ["sfx"],
    "options": {"request": [{"name": "duration_sec", "min": 0.64, "default": 8}, {"name": "seed"}]},
}

# family, spec, expected pages
PLACEMENTS = [
    ("fish_drama3", DRAMA, {"speak", "clone"}),
    ("kitten_tts2", KITTEN2, {"speak", "clone"}),
    ("new_tts_unused", {"tasks": ["tts", "clone"], "options": _req("seed")}, {"speak", "clone"}),
    ("new_tts_legacy", {"tasks": ["tts", "clone"]}, {"speak", "clone"}),
    ("mira_tts", {"tasks": ["tts", "clone"], "options": _req("seed")}, {"clone"}),
    ("sopro_tts", SOPRO, {"clone"}),
    ("lfm2_audio", LFM2, set()),
    ("new_lfm2", {"tasks": ["asr", "tts", "s2s"]}, {"speak"}),
    ("chatterbox", {"tasks": ["clone", "vc"]}, {"clone", "convert"}),
    ("rvc", {"tasks": ["vc"]}, {"convert"}),
    *((f, {"tasks": ["sep"]}, {"separate"}) for f in ("htdemucs", "htdemucs_6stems")),
    *((f, {"tasks": ["sep"]}, {"separate"}) for f in ("bs_roformer", "mel_band_roformer")),
    ("mossformer2", {"tasks": ["sep"]}, {"separate"}),
    ("tf_gridnet", {"tasks": ["sep"]}, {"separate"}),
    ("new_music", MUSIC, {"music"}),
    ("new_sfx", SFX, {"music"}),
    ("new_vc", {"tasks": ["vc"]}, set()),
    ("new_speech_edit", {"tasks": ["edit"]}, set()),
    ("new_tts_edit", {"tasks": ["tts", "edit"]}, {"speak"}),
    ("new_design", {"tasks": ["design"]}, {"speak"}),
    ("new_clone_music", {"tasks": ["clone", "music"]}, {"music"}),
    ("new_clone_asr", {"tasks": ["clone", "asr"]}, {"transcribe"}),
]


def _resolve(
    hub,
    family,
    spec,
    files = ("m-q8_0.gguf",),
):
    repo = f"someone/{family}-GGUF"
    snap = _snapshot(hub, repo)
    for name in files:
        _put(snap, name, _gguf_bytes(family = family, spec = spec))
    return acm.resolve(repo, network = False)


@pytest.mark.parametrize("family, spec, pages", PLACEMENTS, ids = [p[0] for p in PLACEMENTS])
def test_spec_tasks_place_a_family_on_its_pages(hub, family, spec, pages):
    model = _resolve(hub, family, spec)
    assert set(model.workflows) == pages
    assert (model.unsupported is None) == bool(pages)


@pytest.mark.parametrize(
    "family, spec, reference_text",
    [
        ("fish_drama3", DRAMA, "required"),
        ("kitten_tts2", KITTEN2, "required"),
        ("new_tts_unused", {"tasks": ["tts", "clone"], "options": _req("seed")}, "unused"),
        ("new_tts_legacy", {"tasks": ["tts", "clone"]}, "optional"),
    ],
)
def test_a_spec_that_clones_speaks_and_clones_on_one_tts_session(hub, family, spec, reference_text):
    model = _resolve(hub, family, spec)
    assert model.speaks and model.server_task == "tts"
    assert model.clone.reference_text == reference_text
    assert model.workflows["clone"].server_task == "tts"


@pytest.mark.parametrize("family", ["mira_tts", "sopro_tts"])
def test_reference_only_families_offer_clone_alone(hub, family):
    model = _resolve(hub, family, {"tasks": ["tts", "clone"]})
    assert not model.speaks and model.clone.reference_text == "unused"
    assert model.server_task == "tts"


def test_a_spec_derived_clone_runs_on_the_speech_endpoint(hub):
    model = _resolve(hub, "fish_drama3", DRAMA)
    backend, server = _backend(model)
    wav, _rate = backend.generate_audio_response(
        TEXT, workflow = "clone", audio_inputs = {"reference": REF}, reference_text = T, seed = 7
    )
    assert wav[:4] == b"RIFF"
    ((path, body),) = server.calls
    assert path == SPEECH and body["voice_ref"] == REF and body["reference_text"] == T
    assert {b.server_task for b in model.workflows.values()} == {"tts"}


def test_spec_derived_music_and_sfx_get_modes(hub):
    music = _resolve(hub, "new_music", MUSIC).music
    song = music.mode("song")
    assert song.lyrics == "optional" and music.fixed_seed is False and music.mode("sfx") is None
    sfx = _resolve(hub, "new_sfx", SFX).music
    assert [m.id for m in sfx.modes] == ["sfx"] and sfx.mode("sfx").duration == (1.0, 60.0, 8.0)
    assert sfx.fixed_seed is True
    sep = _resolve(hub, "tf_gridnet", {"tasks": ["sep"]})
    assert sep.separation is not None


def test_design_only_spec_runs_voice_design(hub):
    model = _resolve(hub, "new_design", {"tasks": ["design"]})
    assert model.server_task == "vdes" and model.clone is None


def test_a_multi_file_package_spec_is_refused(hub):
    model = _resolve(hub, "lfm2_audio", LFM2)
    assert model.unsupported == acm._MULTI_FILE_PACKAGE
    assert "multi-file package Unsloth Studio cannot load yet" in model.unsupported
    assert acm.recommended_gguf_files(LFM2)[0] == "LFM2.5-Audio-1.5B-Q8_0.gguf"


@pytest.mark.parametrize(
    "family, spec", [(f, s) for f, s, pages in PLACEMENTS if pages and f not in acm.FAMILIES]
)
def test_a_family_the_runtime_lacks_is_still_refused(hub, monkeypatch, family, spec):
    monkeypatch.setattr(acm, "runtime_knows_family", lambda name: False)
    model = _resolve(hub, family, spec)
    assert model.workflows == {} and "needs a newer audio runtime" in model.unsupported


@pytest.mark.parametrize("family", ["mira_tts", "sopro_tts"])
def test_a_reference_only_table_family_the_runtime_lacks_is_refused(hub, monkeypatch, family):
    monkeypatch.setattr(acm, "runtime_knows_family", lambda name: False)
    model = _resolve(hub, family, {"tasks": ["tts", "clone"]})
    assert model.workflows == {} and "needs a newer audio runtime" in model.unsupported
    monkeypatch.setattr(acm, "runtime_knows_family", lambda name: None)
    acm.forget()
    assert set(_resolve(hub, family, {"tasks": ["tts", "clone"]}).workflows) == {"clone"}


def test_unsupported_messages_say_unsloth_studio():
    messages = [f.unsupported for f in acm.FAMILIES.values() if f.unsupported]
    for spec in ({"tasks": t} for t in (["vc"], ["svc"], ["s2s"], ["turn"], ["odd"], [])):
        messages.append(acm._family_from_spec_tasks("new_x", spec).unsupported)
    messages.append(acm._family_from_spec_tasks("lfm2_audio", LFM2).unsupported)
    assert "Unsloth Studio doesn't support voice conversion models yet." in messages
    for message in messages:
        assert message and re.search(r"(?<!Unsloth )Studio", message) is None, message


def test_recommended_gguf_files():
    assert acm.recommended_gguf_files(KITTEN2) == ("kitten-tts2-q8-v2-multilingual.gguf",)
    # The recommended package is safetensors, so the default GGUF package stands in.
    assert acm.recommended_gguf_files(SOPRO) == ("sopro-v2-turbo-f16.gguf",)
    assert acm.recommended_gguf_files(None) == () == acm.recommended_gguf_files({"tasks": ["tts"]})


def _rank(*files: str) -> list[str]:
    variants = [AudioCppVariant(acm.quant_label(f), (RepoFile(f, 1),), f) for f in files]
    return [v.primary for v in sorted(variants, key = acm._variant_rank)]


def test_k_and_iq_quants_rank_after_the_established_order():
    assert acm.quant_label("drama-3-backbone-8bit-experts-Q4_K_S.gguf") == "Q4_K_S"
    assert _rank("m-Q2_K.gguf", "m-Q3_K_S.gguf", "m-Q4_K_S.gguf") == [
        "m-Q4_K_S.gguf",
        "m-Q3_K_S.gguf",
        "m-Q2_K.gguf",
    ]
    assert _rank("m-Q3_K_M.gguf", "m-IQ4_XS.gguf") == ["m-IQ4_XS.gguf", "m-Q3_K_M.gguf"]
    assert _rank("m-Q4_K.gguf", "m-F32.gguf")[0] == "m-F32.gguf"


def test_the_recommended_package_is_the_default_variant(hub):
    drama = [
        f"drama-3-backbone-{b}bit-experts-{q}.gguf"
        for b in (8, 4)
        for q in ("Q4_K_S", "Q3_K_S", "Q2_K")
    ]
    model = _resolve(hub, "fish_drama3", DRAMA, drama)
    assert len(model.variants) == 6
    assert model.variant.primary == "drama-3-backbone-8bit-experts-Q4_K_S.gguf"
    no_pick = _resolve(hub, "fish_drama3_bare", {"tasks": ["tts", "clone"]}, drama)
    assert no_pick.variant.primary != "drama-3-backbone-8bit-experts-Q4_K_S.gguf"
    kittens = [f"kitten-tts2-{q}-multilingual.gguf" for q in ("bf16", "q8-v2", "q4")]
    kitten = _resolve(hub, "kitten_tts2", KITTEN2, kittens)
    assert kitten.variant.primary == "kitten-tts2-q8-v2-multilingual.gguf"
    sopros = ["sopro-v2-turbo-q8_0.gguf", "sopro-v2-turbo-f16.gguf"]
    sopro = _resolve(hub, "sopro_tts", SOPRO, sopros)
    assert sopro.variant.primary == "sopro-v2-turbo-f16.gguf"
