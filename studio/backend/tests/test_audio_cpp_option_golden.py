# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Each catalog model's Advanced options against a golden written separately from the option table.

``fixtures/audio_cpp_option_golden.json`` was derived from the audio.cpp webui parameter table and
the C++ runtime defaults, without reading ``audio_cpp_option_table.py``, so a mistake has to be made
twice to slip through. Per model and workflow it names the options that must be offered (and the
value each starts from), the ones that may be offered (and then with which value), and the ones
that must not be. A case id is ``<folder>[/variant]::<workflow>``."""

import json
import sys
from pathlib import Path

import pytest

from core.inference import audio_cpp_models as acm

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_audio_cpp_option_table import SPECS, _resolved, catalog_hub  # noqa: E402, F401

GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures" / "audio_cpp_option_golden.json").read_text(
        encoding = "utf-8"
    )
)["models"]


def _cases():
    out = []
    for model_id, cases in sorted(GOLDEN.items()):
        folder = model_id.rsplit("/", 1)[-1]
        for case in cases:
            label = folder + (f"/{case['variant']}" if case.get("variant") else "")
            out.append(pytest.param(model_id, case, id = f"{label}::{case['workflow']}"))
    return out


def _variants(model_id, case):
    models = _resolved(model_id)
    prefix = case.get("variant")
    if prefix:
        models = [m for m in models if m.variant.key.lower().startswith(prefix.lower() + "/")]
    assert models, (model_id, prefix)
    return models


def _same(got, want):
    if isinstance(want, bool) or isinstance(got, bool):
        return type(got) is type(want) and got == want
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        return abs(float(got) - float(want)) <= 1e-9
    return got == want


def _pick(by_name, name, spec):
    """The offered option for a golden entry (one of its accepted names), or None."""
    names = [n for n in spec.get("names", [name]) if n in by_name]
    assert len(names) <= 1, f"{name}: more than one alias offered {names}"
    return by_name[names[0]] if names else None


@pytest.mark.parametrize("model_id, case", _cases())
def test_options_match_the_golden(catalog_hub, model_id, case):
    workflow = case["workflow"]
    for model in _variants(model_id, case):
        where = f"{model_id} [{model.variant.key}] {workflow}"
        assert workflow in model.workflows, f"{where}: workflow not offered"
        schema = acm.workflow_options(model, workflow)
        by_name = {o["name"]: o for o in schema}
        sent = acm.filled_options(model, workflow, {})
        # A tool panel sends its own options; the request fills only the model's.
        own = {o["name"] for o in model.options}
        problems = []
        checked = [(n, s, True) for n, s in case["include"].items()]
        checked += [(n, s, False) for n, s in case["optional"].items()]
        for name, spec, required in checked:
            option = _pick(by_name, name, spec)
            if option is None:
                if required:
                    problems.append(f"{name}: not offered (want default {spec['default']!r})")
                continue
            want = spec["default"]
            if not _same(option.get("default"), want):
                problems.append(f"{name}: default {option.get('default')!r}, want {want!r}")
            if "values" in spec and sorted(option.get("values") or []) != sorted(spec["values"]):
                problems.append(f"{name}: values {option.get('values')!r}, want {spec['values']!r}")
            real = option["name"]
            if real not in own:
                if real in sent:
                    problems.append(f"{real}: a tool panel's option was filled")
                continue
            if want is None or real == "voice":
                if real in sent:
                    problems.append(f"{real}: sent {sent[real]!r} though nothing should be")
            elif model.request_keys is not None and real not in model.request_keys:
                problems.append(
                    f"{real}: recommended {want!r} but the strict spec does not declare it"
                )
            elif real not in sent or not _same(sent[real], want):
                problems.append(
                    f"{real}: request sends {sent.get(real, '<nothing>')!r}, want {want!r}"
                )
        for name in case["exclude"]:
            if name in by_name:
                problems.append(f"{name}: offered but must not be")
        if workflow in ("clone", "edit", "convert") and "voice" in sent:
            problems.append("voice: sent on a workflow that speaks in a given voice")
        assert not problems, where + "\n  " + "\n  ".join(problems)


def test_the_golden_covers_every_tts_and_music_model(catalog_hub):
    wanted = {model_id for model_id, entry in SPECS.items() if entry["task"] != "sep"}
    assert set(GOLDEN) == wanted
    for model_id, cases in GOLDEN.items():
        for model in _resolved(model_id):
            covered = {
                c["workflow"]
                for c in cases
                if not c.get("variant")
                or model.variant.key.lower().startswith(c["variant"].lower() + "/")
            }
            assert covered == set(model.workflows), (model_id, model.variant.key)
