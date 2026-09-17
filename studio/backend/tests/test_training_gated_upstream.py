# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for training gated models through public Unsloth mirrors."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

import routes.training as tr
from models import TrainingStartRequest
from utils.models import unsloth_mirror


_STUBBED: list[str] = []


def _stub_if_missing(name, attrs):
    # Stub training dependencies absent from the backend pytest environment.
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:  # noqa: BLE001
        pass
    _STUBBED.append(name)
    mod = types.ModuleType(name)
    mod.__spec__ = None
    for attr in attrs:
        setattr(mod, attr, MagicMock())
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, mod)


_stub_if_missing("unsloth", ("FastLanguageModel", "FastVisionModel", "is_bfloat16_supported"))
_stub_if_missing("unsloth.chat_templates", ("get_chat_template",))
_stub_if_missing("trl", ("SFTTrainer", "SFTConfig"))

import core.training.trainer as trainer_mod  # noqa: E402

for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)

import transformers  # noqa: E402

transformers.AutoTokenizer  # noqa: B018
transformers = sys.modules["transformers"]


_TABLES = (
    {"unsloth/gemma-3-270m-it-unsloth-bnb-4bit": "unsloth/gemma-3-270m-it"},
    {
        "google/gemma-3-270m-it": "unsloth/gemma-3-270m-it-unsloth-bnb-4bit",
        "meta-llama/meta-llama-3-70b-instruct": "unsloth/llama-3-70b-instruct-bnb-4bit",
    },
    {
        "google/gemma-3-270m-it": "unsloth/gemma-3-270m-it",
        "google/gemma-4-26b-a4b": "unsloth/gemma-4-26B-A4B",
    },
)


@pytest.fixture
def mapper(monkeypatch):
    monkeypatch.setattr(unsloth_mirror, "_mapper_tables", lambda: _TABLES)


@pytest.mark.parametrize(
    "name,load_in_4bit,mirror",
    [
        ("google/gemma-3-270m-it", False, "unsloth/gemma-3-270m-it"),
        ("Google/Gemma-3-270M-it", False, "unsloth/gemma-3-270m-it"),
        # A 4-bit load resolves through the 4-bit table, which can map a different model.
        ("google/gemma-3-270m-it", True, "unsloth/gemma-3-270m-it-unsloth-bnb-4bit"),
        ("meta-llama/Meta-Llama-3-70B-Instruct", True, "unsloth/llama-3-70b-instruct-bnb-4bit"),
        ("meta-llama/Meta-Llama-3-70B-Instruct", False, None),
        ("google/gemma-4-26B-A4B", False, "unsloth/gemma-4-26B-A4B"),
        ("google/gemma-4-26B-A4B", True, None),
        # Do not remap public Unsloth repos.
        ("unsloth/gemma-3-270m-it-unsloth-bnb-4bit", False, None),
        ("google/gemma-2-2b-jpn-it", True, None),
        ("google/gemma-2-2b-jpn-it", False, None),
        ("/models/google/gemma-3-270m-it", True, None),
        ("", True, None),
        (None, True, None),
    ],
)
def test_mirror_is_the_repo_the_loader_substitutes(mapper, name, load_in_4bit, mirror):
    assert unsloth_mirror.unsloth_public_mirror(name, load_in_4bit) == mirror


def test_mapper_is_read_from_the_file_without_importing_unsloth(monkeypatch, tmp_path):
    package = tmp_path / "unsloth"
    (package / "models").mkdir(parents = True)
    (package / "__init__.py").write_text("raise RuntimeError('the package must not be imported')\n")
    (package / "models" / "mapper.py").write_text(
        "INT_TO_FLOAT_MAPPER = {}\n"
        "FLOAT_TO_INT_MAPPER = {}\n"
        "MAP_TO_UNSLOTH_16bit = {'google/gemma-3-270m-it': 'unsloth/gemma-3-270m-it'}\n"
    )
    spec = importlib.util.spec_from_file_location(
        "unsloth", package / "__init__.py", submodule_search_locations = [str(package)]
    )
    monkeypatch.setattr(unsloth_mirror.importlib.util, "find_spec", lambda name: spec)
    unsloth_mirror._mapper_tables.cache_clear()
    try:
        assert unsloth_mirror.unsloth_public_mirror("google/gemma-3-270m-it", False) == (
            "unsloth/gemma-3-270m-it"
        )
    finally:
        unsloth_mirror._mapper_tables.cache_clear()


def test_an_unreadable_mapper_redirects_nothing(monkeypatch):
    monkeypatch.setattr(unsloth_mirror.importlib.util, "find_spec", lambda name: None)
    unsloth_mirror._mapper_tables.cache_clear()
    try:
        assert unsloth_mirror.unsloth_public_mirror("google/gemma-3-270m-it", False) is None
    finally:
        unsloth_mirror._mapper_tables.cache_clear()


@pytest.mark.parametrize(
    "load_name,local_only,revision,expected",
    [
        (None, False, None, "unsloth/gemma-3-270m-it-unsloth-bnb-4bit"),
        # A pinned revision loads with use_exact_model_name, so the loader keeps the upstream.
        (None, False, "0123456789abcdef0123456789abcdef01234567", "google/gemma-3-270m-it"),
        ("/hf/models--google--gemma-3-270m-it/snapshots/abc", True, None, None),
        (None, True, None, "google/gemma-3-270m-it"),
    ],
)
def test_pre_detect_reads_the_repo_the_loader_fetches(
    mapper, monkeypatch, load_name, local_only, revision, expected
):
    expected = expected or load_name
    reads = []

    def probe(name, *a, **kw):
        reads.append(name)
        return None, True

    def vision(name, **kw):
        reads.append(name)
        return False

    class _Tokenizer:
        @classmethod
        def from_pretrained(cls, name, **kw):
            reads.append(name)
            return object()

    monkeypatch.setattr(trainer_mod, "detect_audio_type_checked", probe)
    monkeypatch.setattr(trainer_mod, "is_vision_model", vision)
    monkeypatch.setattr(transformers, "AutoTokenizer", _Tokenizer, raising = False)

    trainer = trainer_mod.UnslothTrainer()
    trainer.pre_detect_and_load_tokenizer(
        "google/gemma-3-270m-it",
        model_load_name = load_name,
        local_files_only = local_only,
        model_revision = revision,
    )

    assert reads == [expected, expected, expected]
    assert trainer.model_name == "google/gemma-3-270m-it"


def test_pre_detect_follows_the_requested_load_mode(mapper, monkeypatch):
    reads = []

    class _Tokenizer:
        @classmethod
        def from_pretrained(cls, name, **kw):
            reads.append(name)
            return object()

    def probe(name, *a, **kw):
        reads.append(name)
        return None, True

    def vision(name, **kw):
        reads.append(name)
        return False

    monkeypatch.setattr(trainer_mod, "detect_audio_type_checked", probe)
    monkeypatch.setattr(trainer_mod, "is_vision_model", vision)
    monkeypatch.setattr(transformers, "AutoTokenizer", _Tokenizer, raising = False)

    trainer_mod.UnslothTrainer().pre_detect_and_load_tokenizer(
        "google/gemma-3-270m-it", load_in_4bit = False
    )

    assert reads == ["unsloth/gemma-3-270m-it"] * 3


class _Session:
    def __init__(self, outcome):
        self.outcome = outcome
        self.urls = []

    def get(self, url, **kwargs):
        self.urls.append(url)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return SimpleNamespace(status_code = self.outcome)


def _http_error(status):
    error = Exception(f"{status} Client Error")
    error.response = SimpleNamespace(status_code = status)
    return error


def _route(monkeypatch, *, gated, session):
    import huggingface_hub
    import huggingface_hub.utils

    info = SimpleNamespace(
        gated = gated,
        siblings = [
            SimpleNamespace(rfilename = "config.json"),
            SimpleNamespace(rfilename = "model.safetensors"),
        ],
    )
    monkeypatch.setattr(huggingface_hub, "model_info", lambda *a, **kw: info)
    monkeypatch.setattr(huggingface_hub.utils, "get_session", lambda: session)
    monkeypatch.setattr(huggingface_hub.utils, "hf_raise_for_status", lambda response: None)


def test_gated_repo_without_access_or_a_public_copy_is_refused(mapper, monkeypatch):
    session = _Session(_http_error(401))
    _route(monkeypatch, gated = "manual", session = session)

    with pytest.raises(HTTPException) as error:
        tr._remote_untrainable_model_format("google/gemma-2-2b-jpn-it", None)

    assert error.value.status_code == 422
    assert error.value.detail["code"] == "hf_model_access_denied"
    assert session.urls and session.urls[0].endswith(
        "/api/models/google/gemma-2-2b-jpn-it/auth-check"
    )


@pytest.mark.parametrize(
    "model,gated,outcome",
    [
        ("google/gemma-3-270m-it", "manual", _http_error(401)),
        ("google/gemma-2-2b-jpn-it", "manual", 200),
        # Timeouts and server errors are inconclusive.
        ("google/gemma-2-2b-jpn-it", "auto", TimeoutError("timed out")),
        ("google/gemma-2-2b-jpn-it", "manual", _http_error(500)),
        ("org/public-model", False, _http_error(401)),
    ],
)
def test_gated_check_admits_what_the_worker_can_load(mapper, monkeypatch, model, gated, outcome):
    session = _Session(outcome)
    _route(monkeypatch, gated = gated, session = session)

    assert tr._remote_untrainable_model_format(model, None) is None
    if model == "google/gemma-3-270m-it" or not gated:
        assert session.urls == []


def test_start_request_surfaces_the_refusal_code(mapper, monkeypatch):
    session = _Session(_http_error(403))
    _route(monkeypatch, gated = "manual", session = session)
    monkeypatch.setattr(tr, "hf_env_offline", lambda: False)
    monkeypatch.setattr(tr, "_hub_unreachable", lambda: False)
    monkeypatch.setattr(tr, "cached_read_refused", lambda *a, **kw: False)

    request = TrainingStartRequest(
        model_name = "google/gemma-2-2b-jpn-it",
        training_type = "LoRA/QLoRA",
        format_type = "alpaca",
    )
    with pytest.raises(HTTPException) as error:
        tr._reject_untrainable_model_request(request, hf_token = None)
    assert error.value.detail["code"] == "hf_model_access_denied"


def test_a_4bit_public_copy_admits_a_gated_upstream(mapper, monkeypatch):
    session = _Session(_http_error(401))
    _route(monkeypatch, gated = "manual", session = session)

    model = "meta-llama/Meta-Llama-3-70B-Instruct"
    assert tr._remote_untrainable_model_format(model, None) is None
    assert session.urls == []


@pytest.mark.parametrize(
    "model,load_in_4bit",
    [
        # Mapped for 4-bit only, loaded 16-bit; and mapped for 16-bit only, loaded 4-bit.
        ("meta-llama/Meta-Llama-3-70B-Instruct", False),
        ("google/gemma-4-26B-A4B", True),
    ],
)
def test_a_mirror_for_the_other_load_mode_does_not_skip_the_check(
    mapper, monkeypatch, model, load_in_4bit
):
    session = _Session(_http_error(401))
    _route(monkeypatch, gated = "manual", session = session)

    with pytest.raises(HTTPException) as error:
        tr._remote_untrainable_model_format(model, None, load_in_4bit)

    assert error.value.detail["code"] == "hf_model_access_denied"
    assert session.urls


def test_the_start_route_passes_the_requested_load_mode(mapper, monkeypatch):
    seen = {}

    def probe(model_name, hf_token, load_in_4bit = True):
        seen["load_in_4bit"] = load_in_4bit
        return None

    monkeypatch.setattr(tr, "_remote_untrainable_model_format", probe)
    monkeypatch.setattr(tr, "hf_env_offline", lambda: False)
    monkeypatch.setattr(tr, "_hub_unreachable", lambda: False)
    monkeypatch.setattr(tr, "cached_read_refused", lambda *a, **kw: False)

    request = TrainingStartRequest(
        model_name = "google/gemma-3-270m-it",
        training_type = "LoRA/QLoRA",
        format_type = "alpaca",
        load_in_4bit = False,
    )
    tr._reject_untrainable_model_request(request, hf_token = None)
    assert seen == {"load_in_4bit": False}


def test_load_model_gate_checks_the_repo_the_loader_fetches(mapper, monkeypatch):
    import huggingface_hub

    checked = []

    def probe(name, *a, **kw):
        return None, True

    def model_info(name, **kw):
        checked.append(name)
        return SimpleNamespace(gated = "manual")

    monkeypatch.setattr(trainer_mod, "detect_audio_type_checked", probe)
    monkeypatch.setattr(trainer_mod, "is_vision_model", lambda name, **kw: False)
    monkeypatch.setattr(huggingface_hub, "model_info", model_info)

    trainer = trainer_mod.UnslothTrainer()
    assert trainer.load_model("google/gemma-3-270m-it", load_in_4bit = False) is False
    assert checked == ["unsloth/gemma-3-270m-it"]


@pytest.mark.parametrize("sidecar,checked", [(False, False), (True, True)])
def test_the_latest_sidecar_flip_is_folded_into_the_mode(mapper, monkeypatch, sidecar, checked):
    import utils.transformers_version as tv

    session = _Session(_http_error(401))
    _route(monkeypatch, gated = "manual", session = session)
    monkeypatch.setattr(tv, "latest_tier_active_for", lambda *a, **kw: sidecar)

    model = "meta-llama/Meta-Llama-3-70B-Instruct"
    if checked:
        with pytest.raises(HTTPException) as error:
            tr._remote_untrainable_model_format(model, None)
        assert error.value.detail["code"] == "hf_model_access_denied"
    else:
        assert tr._remote_untrainable_model_format(model, None) is None
    assert bool(session.urls) is checked
