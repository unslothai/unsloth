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

# core.training.trainer imports torch at module scope; skip rather than error.
pytest.importorskip("torch", reason = "core.training.trainer imports torch at module scope")

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


@pytest.fixture(autouse = True)
def _clear_mirror_caches():
    # Both lookups are lru_cached; clear between tests. getattr: the mapper fixture swaps
    # _mapper_tables for an uncached lambda.
    def clear():
        for lookup in (unsloth_mirror._mapper_tables, unsloth_mirror._bad_mappings):
            getattr(lookup, "cache_clear", lambda: None)()

    clear()
    yield
    clear()


@pytest.fixture(autouse = True)
def _torch_trainer(monkeypatch):
    # On Apple Silicon UnslothTrainer.__new__ returns _MLXTrainerAdapter; these tests target
    # the Torch path.
    import core.training.training as training_mod

    # Patch the definition AND trainer.py's copy; a single patch leaves one on the real answer.
    for module in (training_mod, trainer_mod):
        monkeypatch.setattr(
            module, "should_use_mlx_training_backend", lambda *a, **kw: False, raising = False
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
        # get_model_name maps this back through INT_TO_FLOAT_MAPPER for a 16-bit load.
        ("unsloth/gemma-3-270m-it-unsloth-bnb-4bit", False, "unsloth/gemma-3-270m-it"),
        ("google/gemma-2-2b-jpn-it", True, None),
        ("google/gemma-2-2b-jpn-it", False, None),
        ("/models/google/gemma-3-270m-it", True, None),
        ("", True, None),
        (None, True, None),
    ],
)
def test_mirror_is_the_repo_the_loader_substitutes(mapper, name, load_in_4bit, mirror):
    assert unsloth_mirror.unsloth_public_mirror(name, load_in_4bit) == mirror


@pytest.mark.parametrize(
    "name,mirror",
    [
        # get_model_name corrects these; the uncorrected name does not exist on the Hub.
        ("Qwen/Qwen3-30B-A3B", "unsloth/Qwen3-30B-A3B"),
        ("Qwen/Qwen3-30B-A3B-Base", "unsloth/Qwen3-30B-A3B-Base"),
        ("Qwen/Qwen3-32B", "unsloth/Qwen3-32B-bnb-4bit"),
    ],
)
def test_the_loader_corrections_are_applied_to_the_mirror(name, mirror):
    assert unsloth_mirror.unsloth_public_mirror(name, True).lower() == mirror.lower()


def test_corrections_are_read_from_the_real_loader_source():
    # Parsed out of loader_utils.py, so the table cannot drift from the loader.
    corrections = unsloth_mirror._bad_mappings()
    assert corrections
    assert all(key == key.lower() for key in corrections)
    assert "unsloth/qwen3-30b-a3b-unsloth-bnb-4bit" in corrections


@pytest.mark.parametrize(
    "source",
    [
        "BAD_MAPPINGS = {compute(): 'x'}\n",  # an entry that is not readable as data
        "BAD_MAPPINGS = build()\n",  # not a dict literal at all
        "",  # no BAD_MAPPINGS in the file
    ],
)
def test_unreadable_corrections_redirect_nothing(mapper, monkeypatch, tmp_path, source):
    # None, not {}: an uncorrected lookup can name a nonexistent repo.
    (tmp_path / "loader_utils.py").write_text(source)
    monkeypatch.setattr(unsloth_mirror, "_unsloth_models_dir", lambda: tmp_path)
    unsloth_mirror._bad_mappings.cache_clear()
    try:
        assert unsloth_mirror._bad_mappings() is None
        assert unsloth_mirror.unsloth_public_mirror("google/gemma-3-270m-it", False) is None
    finally:
        unsloth_mirror._bad_mappings.cache_clear()


def test_mapper_is_read_from_the_file_without_importing_unsloth(monkeypatch, tmp_path):
    package = tmp_path / "unsloth"
    (package / "models").mkdir(parents = True)
    (package / "__init__.py").write_text("raise RuntimeError('the package must not be imported')\n")
    (package / "models" / "mapper.py").write_text(
        "INT_TO_FLOAT_MAPPER = {}\n"
        "FLOAT_TO_INT_MAPPER = {}\n"
        "MAP_TO_UNSLOTH_16bit = {'google/gemma-3-270m-it': 'unsloth/gemma-3-270m-it'}\n"
    )
    (package / "models" / "loader_utils.py").write_text("BAD_MAPPINGS = {}\n")
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


def test_a_transient_auth_check_failure_is_retried_then_fails_open(mapper, monkeypatch):
    # A single timeout must not admit the run.
    class _Flaky(_Session):
        def get(self, url, **kwargs):
            self.urls.append(url)
            if len(self.urls) == 1:
                raise TimeoutError("first attempt timed out")
            raise _http_error(401)

    session = _Flaky(None)
    _route(monkeypatch, gated = "manual", session = session)

    with pytest.raises(HTTPException) as error:
        tr._remote_untrainable_model_format("google/gemma-2-2b-jpn-it", None)

    assert error.value.detail["code"] == "hf_model_access_denied"
    assert len(session.urls) == 2


def test_an_auth_check_that_never_answers_admits_the_run(mapper, monkeypatch):
    session = _Session(TimeoutError("timed out"))
    _route(monkeypatch, gated = "manual", session = session)

    assert tr._remote_untrainable_model_format("google/gemma-2-2b-jpn-it", None) is None
    assert len(session.urls) == 2


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


def test_a_mirror_in_either_load_mode_admits_the_model(mapper, monkeypatch):
    # Mapping exists in only one mode; this process cannot tell which mode the worker uses,
    # so either mapping admits.
    session = _Session(_http_error(401))
    _route(monkeypatch, gated = "manual", session = session)

    for model in ("meta-llama/Meta-Llama-3-70B-Instruct", "google/gemma-4-26B-A4B"):
        assert tr._remote_untrainable_model_format(model, None) is None
    assert session.urls == []


def test_a_gated_model_with_no_mirror_in_any_mode_is_still_refused(mapper, monkeypatch):
    session = _Session(_http_error(401))
    _route(monkeypatch, gated = "manual", session = session)

    with pytest.raises(HTTPException) as error:
        tr._remote_untrainable_model_format("google/gemma-2-2b-jpn-it", None)
    assert error.value.detail["code"] == "hf_model_access_denied"
    assert session.urls


def test_the_start_route_does_not_pass_a_guessed_load_mode(mapper, monkeypatch):
    # Signature guard: a 4th positional argument broke stubs in other tests.
    import inspect

    params = list(inspect.signature(tr._remote_untrainable_model_format).parameters)
    assert params == ["model_name", "hf_token", "is_embedding"]

    seen = {}

    def probe(
        model_name,
        hf_token,
        is_embedding = False,
    ):
        seen["args"] = (model_name, is_embedding)
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
    assert seen == {"args": ("google/gemma-3-270m-it", False)}


def test_a_full_finetune_of_a_gated_unmirrored_model_is_still_refused(mapper, monkeypatch):
    # The route cannot derive the 16-bit flips without torch; a model with no mirror in
    # EITHER mode is refused regardless.
    session = _Session(_http_error(401))
    _route(monkeypatch, gated = "manual", session = session)
    monkeypatch.setattr(tr, "hf_env_offline", lambda: False)
    monkeypatch.setattr(tr, "_hub_unreachable", lambda: False)
    monkeypatch.setattr(tr, "cached_read_refused", lambda *a, **kw: False)

    request = TrainingStartRequest(
        model_name = "google/gemma-2-2b-jpn-it",
        training_type = "Full Finetuning",
        format_type = "alpaca",
    )
    with pytest.raises(HTTPException) as error:
        tr._reject_untrainable_model_request(request, hf_token = None)

    assert error.value.detail["code"] == "hf_model_access_denied"
    assert request.load_in_4bit is True


@pytest.mark.parametrize(
    "config_4bit,sidecar,expected",
    [
        (True, False, True),
        # Load flipped to 16-bit, so pre-detect must read the 16-bit mirror.
        (True, True, False),
        (False, False, False),
        (False, True, False),
    ],
)
def test_pre_detect_follows_the_sidecar_flip_like_the_load_does(
    monkeypatch, config_4bit, sidecar, expected
):
    import core.training.worker as worker_mod
    import utils.transformers_version as tv

    monkeypatch.setattr(tv, "latest_tier_active_for", lambda *a, **kw: sidecar)
    config = {"load_in_4bit": config_4bit}
    assert worker_mod._pre_detect_load_in_4bit(config, "google/gemma-3-270m-it", None) is expected

    from core.training.provenance import effective_training_load_in_4bit

    assert effective_training_load_in_4bit(config, "google/gemma-3-270m-it", None) is expected


@pytest.mark.parametrize(
    "bnb_ok,expected",
    [(True, "unsloth/gemma-3-270m-it-unsloth-bnb-4bit"), (False, "unsloth/gemma-3-270m-it")],
)
def test_pre_detect_follows_the_bitsandbytes_fallback(mapper, monkeypatch, bnb_ok, expected):
    # from_pretrained clears load_in_4bit when bitsandbytes is unusable, before get_model_name.
    monkeypatch.setattr(trainer_mod, "_bitsandbytes_allows_4bit", lambda: bnb_ok)
    assert (
        trainer_mod._metadata_lookup_name(
            "google/gemma-3-270m-it", "google/gemma-3-270m-it", False, None, True
        )
        == expected
    )


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


def test_mlx_has_no_mirror_so_the_check_still_runs(mapper, monkeypatch):
    # The MLX loader never consults the Torch mapper, so a Torch mapping proves nothing there.
    import core.training.training as training_mod

    session = _Session(_http_error(401))
    _route(monkeypatch, gated = "manual", session = session)
    monkeypatch.setattr(training_mod, "should_use_mlx_training_backend", lambda **kw: True)

    with pytest.raises(HTTPException) as error:
        tr._remote_untrainable_model_format("google/gemma-3-270m-it", None)
    assert error.value.detail["code"] == "hf_model_access_denied"
    assert session.urls


def test_mlx_pre_detect_reads_the_repo_the_mlx_loader_fetches(mapper, monkeypatch):
    import core.training.training as training_mod
    monkeypatch.setattr(training_mod, "should_use_mlx_training_backend", lambda **kw: True)
    assert (
        trainer_mod._metadata_lookup_name(
            "google/gemma-3-270m-it", "google/gemma-3-270m-it", False, None, True
        )
        == "google/gemma-3-270m-it"
    )


def test_an_unreadable_mapper_admits_instead_of_refusing(monkeypatch):
    # find_spec can find no models/mapper.py; unknown must admit, like an unanswered auth check.
    monkeypatch.setattr(unsloth_mirror.importlib.util, "find_spec", lambda name: None)
    unsloth_mirror._mapper_tables.cache_clear()
    unsloth_mirror._bad_mappings.cache_clear()
    session = _Session(_http_error(401))
    _route(monkeypatch, gated = "manual", session = session)
    try:
        assert unsloth_mirror.mirror_lookup_available() is False
        assert tr._remote_untrainable_model_format("google/gemma-3-270m-it", None) is None
        assert session.urls == []
    finally:
        unsloth_mirror._mapper_tables.cache_clear()
        unsloth_mirror._bad_mappings.cache_clear()


def test_an_embedding_run_has_no_mirror_so_the_check_still_runs(mapper, monkeypatch):
    # Embedding training loads the name as given, so the mapper never runs.
    session = _Session(_http_error(401))
    _route(monkeypatch, gated = "manual", session = session)

    with pytest.raises(HTTPException) as error:
        tr._remote_untrainable_model_format("google/gemma-3-270m-it", None, is_embedding = True)
    assert error.value.detail["code"] == "hf_model_access_denied"
    assert session.urls


def test_the_start_route_passes_the_embedding_flag(mapper, monkeypatch):
    seen = {}

    def probe(
        model_name,
        hf_token,
        load_in_4bit = True,
        is_embedding = False,
    ):
        seen["is_embedding"] = is_embedding
        return None

    monkeypatch.setattr(tr, "_remote_untrainable_model_format", probe)
    monkeypatch.setattr(tr, "hf_env_offline", lambda: False)
    monkeypatch.setattr(tr, "_hub_unreachable", lambda: False)
    monkeypatch.setattr(tr, "cached_read_refused", lambda *a, **kw: False)

    request = TrainingStartRequest(
        model_name = "google/gemma-3-270m-it",
        training_type = "LoRA/QLoRA",
        format_type = "alpaca",
        is_embedding = True,
    )
    tr._reject_untrainable_model_request(request, hf_token = None)
    assert seen == {"is_embedding": True}


def test_the_security_scan_covers_the_repo_the_loader_substitutes(mapper, monkeypatch):
    # Scan the mapped download target, not just the picked name.
    import core.training.worker as worker_mod

    scanned: list[str] = []

    class _Decision:
        blocked = False

        def response_payload(self):
            return {}

    monkeypatch.setattr(worker_mod, "_model_local_files_only", lambda config: False, raising = False)
    import utils.security as security_mod

    monkeypatch.setattr(security_mod, "security_load_subdirs", lambda *a, **kw: ())
    monkeypatch.setattr(security_mod, "load_scan_target", lambda t, s: (t, s))
    monkeypatch.setattr(
        security_mod,
        "evaluate_file_security",
        lambda target, **kw: (scanned.append(target), _Decision())[1],
    )

    config = {
        "model_name": "google/gemma-3-270m-it",
        "load_in_4bit": True,
        "trust_remote_code": False,
    }
    assert worker_mod._model_load_security_error(config, "google/gemma-3-270m-it", None) is None
    assert "unsloth/gemma-3-270m-it-unsloth-bnb-4bit" in scanned


def test_the_security_scan_covers_both_load_modes(mapper, monkeypatch):
    # The sidecar and bnb fallback cannot be read here (runs before torch); scan both candidates.
    import core.training.worker as worker_mod
    import utils.security as security_mod

    scanned: list[str] = []

    class _Decision:
        blocked = False

        def response_payload(self):
            return {}

    monkeypatch.setattr(worker_mod, "_model_local_files_only", lambda config: False, raising = False)
    monkeypatch.setattr(security_mod, "security_load_subdirs", lambda *a, **kw: ())
    monkeypatch.setattr(security_mod, "load_scan_target", lambda t, s: (t, s))
    monkeypatch.setattr(
        security_mod,
        "evaluate_file_security",
        lambda target, **kw: (scanned.append(target), _Decision())[1],
    )

    assert (
        worker_mod._model_load_security_error(
            {
                "model_name": "google/gemma-3-270m-it",
                "load_in_4bit": True,
                "trust_remote_code": False,
            },
            "google/gemma-3-270m-it",
            None,
        )
        is None
    )
    assert "unsloth/gemma-3-270m-it" in scanned
    assert "unsloth/gemma-3-270m-it-unsloth-bnb-4bit" in scanned


def test_the_security_scan_imports_nothing_heavy():
    # Must not import core.training.trainer: it would initialise torch before the patches.
    import ast
    import pathlib as _pathlib

    import core.training.worker as worker_mod

    # From the module, not a relative path: pytest runs from different cwds.
    src = _pathlib.Path(worker_mod.__file__).read_text()
    fn = next(
        node
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.FunctionDef) and node.name == "_model_load_security_error"
    )
    imported = {n.module for n in ast.walk(fn) if isinstance(n, ast.ImportFrom) and n.module} | {
        alias.name for n in ast.walk(fn) if isinstance(n, ast.Import) for alias in n.names
    }
    assert not any(
        m == "torch"
        or m.startswith(("torch.", "unsloth.", "core.training.trainer"))
        or m in {"unsloth", "unsloth_zoo"}
        for m in imported
    ), sorted(imported)


@pytest.mark.parametrize(
    "name,load_in_4bit,mirror",
    [
        # Verified against get_model_name: 16-bit resolves through INT_TO_FLOAT_MAPPER...
        ("unsloth/gemma-3-270m-it-unsloth-bnb-4bit", False, "unsloth/gemma-3-270m-it"),
        # ...while 4-bit keeps it, and BAD_MAPPINGS lands on a different repo.
        ("unsloth/Qwen3-30B-A3B-unsloth-bnb-4bit", True, "unsloth/Qwen3-30B-A3B"),
        # An id the loader really does leave alone still answers None.
        ("unsloth/gemma-3-270m-it-unsloth-bnb-4bit", True, None),
        ("unsloth/gemma-3-270m-it", False, None),
    ],
)
def test_an_unsloth_id_is_not_a_fixed_point(mapper, name, load_in_4bit, mirror):
    assert unsloth_mirror.unsloth_public_mirror(name, load_in_4bit) == mirror


@pytest.mark.parametrize("config_key", ["is_embedding", "mlx"])
def test_the_scan_does_not_expand_mirrors_off_the_torch_path(mapper, monkeypatch, config_key):
    # MLX and embedding runs never use this mapper, so scanning a mirror there is wrong.
    import core.training.training as training_mod
    import core.training.worker as worker_mod
    import utils.security as security_mod

    scanned: list[str] = []

    class _Decision:
        blocked = False

        def response_payload(self):
            return {}

    config = {
        "model_name": "google/gemma-3-270m-it",
        "load_in_4bit": True,
        "trust_remote_code": False,
    }
    if config_key == "is_embedding":
        config["is_embedding"] = True
    else:
        monkeypatch.setattr(training_mod, "should_use_mlx_training_backend", lambda **kw: True)

    monkeypatch.setattr(worker_mod, "_model_local_files_only", lambda config: False, raising = False)
    monkeypatch.setattr(security_mod, "security_load_subdirs", lambda *a, **kw: ())
    monkeypatch.setattr(security_mod, "load_scan_target", lambda t, s: (t, s))
    monkeypatch.setattr(
        security_mod,
        "evaluate_file_security",
        lambda target, **kw: (scanned.append(target), _Decision())[1],
    )

    assert worker_mod._model_load_security_error(config, "google/gemma-3-270m-it", None) is None
    assert scanned == ["google/gemma-3-270m-it"]


def test_a_configured_16bit_load_does_not_scan_the_4bit_mirror(mapper, monkeypatch):
    # The mode fallbacks only run downwards, so a configured 16-bit load cannot become 4-bit
    # and the 4-bit mirror is a repo it will never fetch.
    import core.training.worker as worker_mod
    import utils.security as security_mod

    scanned: list[str] = []

    class _Decision:
        blocked = False

        def response_payload(self):
            return {}

    monkeypatch.setattr(worker_mod, "_model_local_files_only", lambda config: False, raising = False)
    monkeypatch.setattr(security_mod, "security_load_subdirs", lambda *a, **kw: ())
    monkeypatch.setattr(security_mod, "load_scan_target", lambda t, s: (t, s))
    monkeypatch.setattr(
        security_mod,
        "evaluate_file_security",
        lambda target, **kw: (scanned.append(target), _Decision())[1],
    )

    assert (
        worker_mod._model_load_security_error(
            {
                "model_name": "google/gemma-3-270m-it",
                "load_in_4bit": False,
                "trust_remote_code": False,
            },
            "google/gemma-3-270m-it",
            None,
        )
        is None
    )
    assert scanned == ["google/gemma-3-270m-it", "unsloth/gemma-3-270m-it"]


def test_the_scan_covers_the_repo_left_after_the_4bit_suffix_is_stripped(mapper, monkeypatch):
    # Without ALLOW_PREQUANTIZED_MODELS (ROCm, bnb < 0.49.2) loader.py strips the suffix off
    # the mapped name and downloads that repo.
    import core.training.worker as worker_mod
    import utils.security as security_mod

    scanned: list[str] = []

    class _Decision:
        blocked = False

        def response_payload(self):
            return {}

    monkeypatch.setattr(worker_mod, "_model_local_files_only", lambda config: False, raising = False)
    monkeypatch.setattr(security_mod, "security_load_subdirs", lambda *a, **kw: ())
    monkeypatch.setattr(security_mod, "load_scan_target", lambda t, s: (t, s))
    monkeypatch.setattr(
        security_mod,
        "evaluate_file_security",
        lambda target, **kw: (scanned.append(target), _Decision())[1],
    )

    model = "meta-llama/Meta-Llama-3-70B-Instruct"
    assert (
        worker_mod._model_load_security_error(
            {"model_name": model, "load_in_4bit": True, "trust_remote_code": False}, model, None
        )
        is None
    )
    # Lower case as the mapper stores it; HF cache dirs are case sensitive.
    assert "unsloth/llama-3-70b-instruct-bnb-4bit" in scanned
    assert "unsloth/llama-3-70b-instruct" in scanned


@pytest.mark.parametrize(
    "name,stripped",
    [
        ("unsloth/gemma-3-270m-it-unsloth-bnb-4bit", "unsloth/gemma-3-270m-it"),
        ("unsloth/llama-3-70b-Instruct-bnb-4bit", "unsloth/llama-3-70b-Instruct"),
        ("unsloth/gemma-3-270m-it", "unsloth/gemma-3-270m-it"),
    ],
)
def test_the_suffix_strip_matches_the_loader(name, stripped):
    import ast
    from pathlib import Path

    import core.training.worker as worker_mod

    # load the pure helper without importing the gpu stack into backend-only tests.
    loader_path = Path(__file__).resolve().parents[3] / "unsloth" / "models" / "loader.py"
    loader_tree = ast.parse(loader_path.read_text(encoding = "utf-8"))
    helper = next(
        node
        for node in loader_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_strip_unsloth_bnb_4bit_suffix"
    )
    namespace = {}
    exec(compile(ast.Module(body = [helper], type_ignores = []), str(loader_path), "exec"), namespace)
    loader_strip = namespace["_strip_unsloth_bnb_4bit_suffix"]

    assert worker_mod._strip_unsloth_bnb_4bit_suffix(name) == stripped
    assert loader_strip(name) == stripped


@pytest.mark.parametrize("backend", ["mlx", "embedding"])
def test_an_unreadable_mapper_does_not_admit_a_non_torch_backend(monkeypatch, backend):
    # MLX and embedding fetch the picked repo, so the auth check is the only guard there.
    import core.training.training as training_mod
    import utils.models.unsloth_mirror as mirror_mod

    monkeypatch.setattr(mirror_mod, "mirror_lookup_available", lambda: False)
    session = _Session(_http_error(401))
    _route(monkeypatch, gated = "manual", session = session)

    kwargs = {}
    if backend == "mlx":
        monkeypatch.setattr(training_mod, "should_use_mlx_training_backend", lambda **kw: True)
    else:
        kwargs["is_embedding"] = True

    with pytest.raises(HTTPException) as error:
        tr._remote_untrainable_model_format("google/gemma-3-270m-it", None, **kwargs)
    assert error.value.detail["code"] == "hf_model_access_denied"
    assert session.urls


def test_a_lora_adapter_base_gets_its_own_mirror_scanned(mapper, monkeypatch):
    # loader.py maps the adapter's base_model_name_or_path, so the base's mirror is fetched.
    import core.training.worker as worker_mod
    import utils.models.model_config as model_config_mod
    import utils.security as security_mod

    scanned: list[str] = []

    class _Decision:
        blocked = False

        def response_payload(self):
            return {}

    monkeypatch.setattr(
        model_config_mod,
        "get_base_model_from_lora_identifier",
        lambda target, token: "google/gemma-3-270m-it",
    )
    monkeypatch.setattr(worker_mod, "_model_local_files_only", lambda config: False, raising = False)
    monkeypatch.setattr(security_mod, "security_load_subdirs", lambda *a, **kw: ())
    monkeypatch.setattr(security_mod, "load_scan_target", lambda t, s: (t, s))
    monkeypatch.setattr(
        security_mod,
        "evaluate_file_security",
        lambda target, **kw: (scanned.append(target), _Decision())[1],
    )

    adapter = "someone/my-lora-adapter"
    assert (
        worker_mod._model_load_security_error(
            {"model_name": adapter, "load_in_4bit": True, "trust_remote_code": False},
            adapter,
            None,
        )
        is None
    )
    assert "google/gemma-3-270m-it" in scanned
    assert "unsloth/gemma-3-270m-it-unsloth-bnb-4bit" in scanned
