# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Options audio.cpp v0.9.2 renamed: one control under the legacy name, one name on the wire, and
the spec each runtime checks deciding which."""

import sys
from pathlib import Path

import pytest

from core.inference import audio_cpp_models as acm
from core.inference import audio_cpp_server as srv
from core.inference.audio_cpp_models import AUDIO_CPP_REPO
from core.inference.audio_cpp_option_table import OPTION_RENAMES

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_audio_cpp_models import _fake_runtime, _gguf_bytes, _put, _snapshot, hub  # noqa: E402, F401


def _float(name, default, description = "", **kw):
    return {"name": name, "type": "float", "description": description, "required": False, "default": default, **kw}


def _spec(family, *options, strict = False):
    spec = {"family": family, "options": {"request": list(options)}}
    return {**spec, "schema_version": 1} if strict else spec


# Request options trimmed from model_specs/ at v0.9.1 and v0.9.2.
_STEPS = {"name": "num_inference_steps", "type": "int", "description": "", "required": False, "min": 1}
CHATTERBOX_OLD = _spec("chatterbox", _float("s3gen_cfg_rate", 0.7), {**_STEPS, "default": 10})
CHATTERBOX_NEW = _spec("chatterbox", _float("flow_guidance_scale", 0.7), {**_STEPS, "default": 10})
SEED_VC_OLD = _spec("seed_vc", _float("length_adjust", 1.0, "Output duration multiplier.", min = 0.0), {**_STEPS, "default": 30}, strict = True)
SEED_VC_NEW = _spec("seed_vc", _float("duration_scale", 1.0, "V1 output duration multiplier.", min = 0.0), {**_STEPS, "default": 30}, strict = True)

FOLDERS = {
    "chatterbox": ("Chatterbox-GGUF", ("chatterbox-q8_0.gguf",)),
    "seed_vc": ("SeedVC-MLX-GGUF", ("seed-vc-mlx-q8_0.gguf",)),
    "ace_step": (
        "ACE-Step1.5-GGUF",
        ("base/ace-step-1.5-base-q8_0.gguf", "turbo/ace-step-1.5-turbo-q8_0.gguf"),
    ),
}


def _resolve(hub, monkeypatch, tmp_path, family, embedded, runtime = None, tag = None):
    """``family``'s umbrella folder with ``embedded`` in its GGUF, ``runtime`` as its model_specs/
    entry, and an install record for ``tag`` (None: a custom build)."""
    folder, files = FOLDERS[family]
    snap = _snapshot(hub)
    for rel in files:
        _put(snap, f"{folder}/{rel}", _gguf_bytes(family = family, spec = embedded))
    _fake_runtime(monkeypatch, tmp_path / "runtime", tag = tag)
    monkeypatch.setattr(acm, "runtime_spec", lambda name: runtime if name == family else None)
    acm.forget()
    model = acm.resolve(f"{AUDIO_CPP_REPO}/{folder}", network = False)
    assert model is not None and model.unsupported is None
    return model


def _names(model):
    return [o["name"] for o in model.options]


def test_the_table_lists_three_pairs():
    assert dict(OPTION_RENAMES) == {
        "chatterbox": (("s3gen_cfg_rate", "flow_guidance_scale"),),
        "seed_vc": (("length_adjust", "duration_scale"),),
        "ace_step": (("lm_cfg_scale", "lm_guidance_scale"),),
    }


def test_refreshed_chatterbox_shows_one_control_under_the_legacy_name(hub, monkeypatch, tmp_path):
    model = _resolve(hub, monkeypatch, tmp_path, "chatterbox", CHATTERBOX_NEW)
    assert model.request_keys is None
    assert _names(model).count("s3gen_cfg_rate") == 1 and "flow_guidance_scale" not in _names(model)
    (option,) = (o for o in model.options if o["name"] == "s3gen_cfg_rate")
    assert option["default"] == 0.7
    convert = acm.filled_options(model, "convert", {"s3gen_cfg_rate": 1.0})
    assert convert["s3gen_cfg_rate"] == 1.0 and "flow_guidance_scale" not in convert
    clone = acm.filled_options(model, "clone", {})
    assert clone["s3gen_cfg_rate"] == 0.7 and "flow_guidance_scale" not in clone


def test_the_schema_drops_a_canonical_entry_beside_its_legacy_one(hub, monkeypatch, tmp_path):
    both = _spec("chatterbox", _float("s3gen_cfg_rate", 0.7), _float("flow_guidance_scale", 0.9))
    model = _resolve(hub, monkeypatch, tmp_path, "chatterbox", both)
    assert [o["default"] for o in model.options if o["name"] == "s3gen_cfg_rate"] == [0.7]
    assert "flow_guidance_scale" not in _names(model)


def test_an_api_caller_may_use_the_canonical_name(hub, monkeypatch, tmp_path):
    model = _resolve(hub, monkeypatch, tmp_path, "chatterbox", CHATTERBOX_NEW)
    out = acm.filled_options(model, "convert", {"flow_guidance_scale": 1.0})
    assert out["s3gen_cfg_rate"] == 1.0 and "flow_guidance_scale" not in out
    # Both given: the legacy value wins, whatever the order.
    for values in (
        {"flow_guidance_scale": 2.0, "s3gen_cfg_rate": 1.0},
        {"s3gen_cfg_rate": 1.0, "flow_guidance_scale": 2.0},
    ):
        out = acm.filled_options(model, "convert", values)
        assert out["s3gen_cfg_rate"] == 1.0 and "flow_guidance_scale" not in out


def test_strict_refreshed_seed_vc_sends_the_canonical_name(hub, monkeypatch, tmp_path):
    model = _resolve(hub, monkeypatch, tmp_path, "seed_vc", SEED_VC_NEW)
    assert "duration_scale" in model.request_keys and "length_adjust" in _names(model)
    out = acm.filled_options(model, "convert", {"length_adjust": 1.5})
    assert out["duration_scale"] == 1.5 and "length_adjust" not in out
    # Left unset, the default is still filled, under the name the contract declares.
    out = acm.filled_options(model, "convert", {})
    assert out["duration_scale"] == 1.0 and "length_adjust" not in out


def test_old_seed_vc_sends_the_legacy_name(hub, monkeypatch, tmp_path):
    model = _resolve(hub, monkeypatch, tmp_path, "seed_vc", SEED_VC_OLD)
    for values in ({"length_adjust": 1.5}, {"duration_scale": 1.5}):
        out = acm.filled_options(model, "convert", values)
        assert out["length_adjust"] == 1.5 and "duration_scale" not in out


def test_ace_step_takes_the_canonical_planner_guidance_as_input(hub, monkeypatch, tmp_path):
    model = _resolve(hub, monkeypatch, tmp_path, "ace_step", {})
    out = acm.filled_options(model, "music", {"lm_guidance_scale": 3.0})
    assert out["lm_cfg_scale"] == 3.0 and "lm_guidance_scale" not in out


def test_a_spec_first_runtime_takes_its_own_spec_over_an_old_strict_gguf(hub, monkeypatch, tmp_path):
    model = _resolve(
        hub, monkeypatch, tmp_path, "seed_vc", SEED_VC_OLD, runtime = SEED_VC_NEW, tag = "v0.9.2-unsloth.1"
    )
    assert "length_adjust" in _names(model) and "duration_scale" not in _names(model)
    assert "duration_scale" in model.request_keys and "length_adjust" not in model.request_keys
    out = acm.filled_options(model, "convert", {"length_adjust": 1.5})
    assert out["duration_scale"] == 1.5 and "length_adjust" not in out


def test_an_older_runtime_ignores_a_backfilled_model_specs(hub, monkeypatch, tmp_path):
    model = _resolve(
        hub, monkeypatch, tmp_path, "seed_vc", SEED_VC_OLD, runtime = SEED_VC_NEW, tag = "v0.9.1-unsloth.1"
    )
    assert model.request_keys == frozenset({"length_adjust", "num_inference_steps"})
    assert [o["description"] for o in model.options if o["name"] == "length_adjust"] == [
        "Output duration multiplier."
    ]
    out = acm.filled_options(model, "convert", {"length_adjust": 1.5})
    assert out["length_adjust"] == 1.5 and "duration_scale" not in out


def test_a_custom_build_keeps_the_embedded_contract_first(hub, monkeypatch, tmp_path):
    model = _resolve(hub, monkeypatch, tmp_path, "seed_vc", SEED_VC_OLD, runtime = SEED_VC_NEW)
    assert srv.runtime_release_key() is None
    assert model.request_keys == frozenset({"length_adjust", "num_inference_steps"})
    out = acm.filled_options(model, "convert", {"length_adjust": 1.5})
    assert out["length_adjust"] == 1.5 and "duration_scale" not in out


def _loose(spec):
    return {k: v for k, v in spec.items() if k != "schema_version"}


# fmt: off
@pytest.mark.parametrize("family, workflow, embedded, runtime", [
    ("chatterbox", "convert", CHATTERBOX_OLD, CHATTERBOX_NEW),
    ("chatterbox", "convert", CHATTERBOX_NEW, CHATTERBOX_NEW),
    ("seed_vc", "convert", _loose(SEED_VC_OLD), SEED_VC_NEW),
    ("seed_vc", "convert", _loose(SEED_VC_NEW), SEED_VC_NEW),
    ("ace_step", "music", {}, None),
])
# fmt: on
def test_a_v091_runtime_gets_only_legacy_names(hub, monkeypatch, tmp_path, family, workflow, embedded, runtime):
    model = _resolve(hub, monkeypatch, tmp_path, family, embedded, runtime = runtime, tag = "v0.9.1-unsloth.1")
    assert model.request_keys is None
    ((old, canonical),) = OPTION_RENAMES[family]
    for values in ({}, {old: 1.5}, {canonical: 1.5}):
        out = acm.filled_options(model, workflow, values)
        assert canonical not in out, (values, out)
        if values:
            assert out[old] == 1.5
