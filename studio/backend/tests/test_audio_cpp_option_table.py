# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every audio.cpp model in the Audio catalog, one at a time: the options each workflow offers, the
recommended value each one starts from, and that a request with nothing set sends those values.

The specs are the ones embedded in the published GGUFs (``fixtures/audio_cpp_catalog_specs.json``),
read with no runtime ``model_specs/``, as a prebuilt runtime runs them."""

import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.inference import audio_cpp_backend, audio_cpp_files
from core.inference import audio_cpp_models as acm
from core.inference.audio_cpp_option_table import (
    AUDIO_CPP_OPTION_TABLE_TAGS,
    OPTION_RULES,
    STUDIO_OPTION_LISTS,
)
from core.inference.audio_cpp_spec_families import AUDIO_CPP_SPEC_FAMILIES_TAGS

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_audio_cpp_models import _gguf_bytes, _put, _snapshot  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "audio_cpp_catalog_specs.json"
CATALOG = Path(__file__).resolve().parents[2] / "frontend/src/features/audio/audio-cpp-catalog.ts"
SPECS = json.loads(FIXTURE.read_text(encoding = "utf-8"))["models"]
WORKFLOWS = ("speak", "clone", "edit", "convert", "music")


@pytest.fixture
def catalog_hub(tmp_path, monkeypatch):
    """A fake HF cache holding every catalog model's files, each GGUF carrying its embedded spec."""
    root = tmp_path / "hub"
    root.mkdir()
    monkeypatch.setattr(acm, "_hub_cache", lambda: root)
    monkeypatch.setattr(audio_cpp_files, "_hub_cache", lambda: root)
    monkeypatch.setattr(acm, "runtime_spec", lambda family: None)
    monkeypatch.setattr(acm, "runtime_knows_family", lambda family: None)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    acm.forget()
    with acm._resolve_lock:
        acm._downloaded_cache.clear()
    snaps = {}
    for model_id, entry in SPECS.items():
        repo, _, folder = model_id.partition("/audio.cpp-gguf/")
        repo = acm.AUDIO_CPP_REPO if folder else model_id
        snap = snaps.get(repo) or snaps.setdefault(repo, _snapshot(root, repo))
        header = _gguf_bytes(family = entry["family"], spec = entry["spec"])
        for rel, _size in entry["files"]:
            path = f"{folder}/{rel}" if folder else rel
            _put(snap, path, header if rel.endswith(".gguf") else b"{}")
    yield root
    acm.forget()


def _models():
    """(model id, variant key) for every variant of every TTS and music catalog model."""
    out = []
    for model_id, entry in sorted(SPECS.items()):
        if entry["task"] == "sep":
            continue
        out.append(model_id)
    return out


def _resolved(model_id):
    model = acm.resolve(model_id, network = False)
    assert model is not None and model.unsupported is None, (model_id, model and model.unsupported)
    return [model, *(model.with_variant(v) for v in model.variants if v.key != model.variant.key)]


def _wanted(model, workflow):
    return workflow in model.workflows


def test_the_table_is_written_for_the_pinned_runtime():
    assert AUDIO_CPP_OPTION_TABLE_TAGS == AUDIO_CPP_SPEC_FAMILIES_TAGS


def test_the_fixture_covers_the_catalog():
    source = CATALOG.read_text(encoding = "utf-8")
    seeded = {
        (f"{acm.AUDIO_CPP_REPO}/{folder}" if folder else repo)
        for folder, repo in re.findall(
            r'id: (?:folder\("([^"]+)"\)|"(audio-cpp/[^"]+)"),\s*task: "(?:tts|music|sep)"', source
        )
    }
    assert seeded == set(SPECS)


def test_tables_name_known_families():
    for family in (*STUDIO_OPTION_LISTS, *OPTION_RULES):
        assert family in acm.FAMILIES, family


@pytest.mark.parametrize("model_id", _models())
def test_each_option_starts_from_a_value_it_accepts(catalog_hub, model_id):
    for model in _resolved(model_id):
        for workflow in WORKFLOWS:
            if not _wanted(model, workflow):
                continue
            for option in acm.workflow_options(model, workflow):
                default = option.get("default")
                where = (model_id, model.variant.key, workflow, option["name"])
                if option["type"] == "bool":
                    # A switch has no "unset" look: it must show what runs.
                    assert isinstance(default, bool), where
                if default is None:
                    continue
                assert acm.validate_options((option,), {option["name"]: default}) == {
                    option["name"]: default
                }, where


@pytest.mark.parametrize("model_id", _models())
def test_a_request_with_nothing_set_sends_the_shown_values(catalog_hub, model_id):
    for model in _resolved(model_id):
        for workflow in WORKFLOWS:
            if not _wanted(model, workflow):
                continue
            schema = acm.workflow_options(model, workflow)
            sent = acm.filled_options(model, workflow, {})
            # A tool panel's own options are the panel's to send.
            own = {o["name"] for o in model.options}
            shown = {
                o["name"]: o["default"]
                for o in schema
                if o.get("default") is not None and o["name"] != "voice" and o["name"] in own
            }
            if model.request_keys is not None:
                assert set(sent) <= model.request_keys, (model_id, workflow)
                shown = {k: v for k, v in shown.items() if k in model.request_keys}
            assert sent == shown, (model_id, model.variant.key, workflow)
            if workflow in ("clone", "edit", "convert"):
                assert "voice" not in {o["name"] for o in schema}


@pytest.mark.parametrize("model_id", _models())
def test_user_values_win_over_recommended_ones(catalog_hub, model_id):
    model = _resolved(model_id)[0]
    for workflow in WORKFLOWS:
        if not _wanted(model, workflow):
            continue
        schema = acm.workflow_options(model, workflow)
        numeric = next((o for o in schema if o["type"] in ("int", "float")), None)
        if numeric is None:
            continue
        value = numeric["max"] if numeric.get("max") is not None else (numeric.get("min") or 0) + 1
        sent = acm.request_options(schema, {numeric["name"]: value}, None)
        assert (
            sent[numeric["name"]]
            == acm.validate_options(schema, {numeric["name"]: value})[numeric["name"]]
        )


def test_status_lists_each_workflow_where_it_differs(catalog_hub):
    for model_id in _models():
        model = _resolved(model_id)[0]
        fields = audio_cpp_backend.model_info_fields(model)
        primary = "speak" if "speak" in model.workflows else next(iter(model.workflows))
        assert fields["audio_options"] == [dict(o) for o in acm.workflow_options(model, primary)]
        by_workflow = fields["audio_options_by_workflow"] or {}
        for workflow in model.workflows:
            listed = by_workflow.get(workflow, fields["audio_options"])
            assert listed == [dict(o) for o in acm.workflow_options(model, workflow)], (
                model_id,
                workflow,
            )


MAGPIE = f"{acm.AUDIO_CPP_REPO}/MagpieTTS-Multilingual-357M-GGUF"


def _stale_magpie():
    """MagpieTTS as resolved before its download: options read from the ``orig`` file's header,
    which names the speaker option ``speaker`` where the served Q8_0 says ``voice_id``."""
    served = acm.resolve(MAGPIE, network = False)
    speaker = {
        "name": "speaker",
        "type": "int",
        "description": "",
        "required": False,
        "default": 0,
        "min": None,
        "max": None,
        "values": None,
    }
    own = tuple(o for o in served.spec_options if o["name"] != "voice_id")
    keys = (served.request_keys or frozenset()) - {"voice_id"} | {"speaker"}
    stale = replace(
        served, options = (speaker, *own), spec_options = (speaker, *own), request_keys = keys
    )
    return served, stale


@pytest.mark.parametrize("downloaded", [True, False])
def test_a_load_that_downloads_fills_the_served_files_options(catalog_hub, monkeypatch, downloaded):
    from core.inference import audio_cpp_server

    served, stale = _stale_magpie()
    backend = audio_cpp_backend.AudioCppBackend()
    monkeypatch.setattr(backend, "_ensure_downloaded", lambda model, token: downloaded)
    monkeypatch.setattr(backend, "_start_server", lambda model, *a, **k: None)
    monkeypatch.setattr(audio_cpp_server, "model_runtime_problem", lambda model: None)
    backend.load_model(SimpleNamespace(identifier = MAGPIE, audio_cpp = stale))
    filled = acm.filled_options(backend._model, "speak", {})
    if downloaded:
        assert "voice_id" in filled and "speaker" not in filled
        assert filled == acm.filled_options(served, "speak", {})
    else:
        # Nothing was fetched, so the files on disk did not change: the model is kept as given.
        assert "speaker" in filled


def test_each_cached_quantization_offers_its_own_files_options(catalog_hub):
    """With both MagpieTTS files cached, each variant's options come from the file it serves."""
    q8 = SPECS[MAGPIE]["spec"]
    orig = json.loads(json.dumps(q8))
    for option in orig["options"]["request"]:
        if option["name"] == "voice_id":
            option["name"] = "speaker"
    snap = _snapshot(catalog_hub)
    folder = "MagpieTTS-Multilingual-357M-GGUF"
    _put(
        snap,
        f"{folder}/magpie-tts-multilingual-357m-orig.gguf",
        _gguf_bytes(family = "magpie_tts", spec = orig),
    )
    _put(
        snap,
        f"{folder}/magpie-tts-multilingual-357m-q8_0.gguf",
        _gguf_bytes(family = "magpie_tts", spec = q8),
    )
    acm.forget()
    served = acm.resolve(MAGPIE, network = False)
    assert served.variant.key == "Q8_0"
    filled = acm.filled_options(served, "speak", {})
    assert "voice_id" in filled and "speaker" not in filled
    assert "speaker" not in (served.request_keys or ())
    acm.forget()
    other = acm.resolve(MAGPIE, "orig", network = False)
    assert "speaker" in acm.filled_options(other, "speak", {})


ACE_STEP = f"{acm.AUDIO_CPP_REPO}/ACE-Step1.5-GGUF"


@pytest.mark.parametrize(
    "name, base",
    [
        ("ace-step-1.5-base-q8_0.gguf", True),
        ("acestep-v15-xl-sft-q8_0.gguf", True),
        ("ace-step-1.5-turbo-q8_0.gguf", False),
        # The runtime reads the file name only, so a base/ folder alone is turbo.
        ("base/model-q8_0.gguf", False),
        ("ace step base v1.gguf", True),
    ],
)
def test_ace_step_variant_follows_the_file_name_like_the_runtime(catalog_hub, tmp_path, name, base):
    """audio.cpp picks base/SFT from "base" or "sft" in the GGUF name, wherever it is stored."""
    path = tmp_path / name
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(_gguf_bytes(family = "ace_step", spec = SPECS[ACE_STEP]["spec"]))
    model = acm.resolve(str(path), network = False)
    options = {o["name"]: o for o in acm.workflow_options(model, "music")}
    filled = acm.filled_options(model, "music", {})
    if base:
        assert options["num_inference_steps"]["max"] == 200
        assert filled["num_inference_steps"] == 32 and filled["guidance_scale"] == 7.0
        assert filled["shift"] == 1.0
    else:
        assert options["num_inference_steps"]["max"] == 8 and "guidance_scale" not in options
        assert filled["shift"] == 3.0
