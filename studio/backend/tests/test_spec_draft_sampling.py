# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MTP drafting at T > 0: Studio passes --spec-draft-sampling probabilistic (the draft is sampled and the target
verifies it by rejection sampling, so the output distribution is unchanged) when the build lists the flag, and
never overrides a choice the user made in the extras or the LLAMA_ARG_SPEC_DRAFT_SAMPLING env."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_llama_cpp_mtp_detection import (  # noqa: F401 (stubs are installed on import)
    LlamaCppBackend,
    _NEEDS_BASH,
    _clear_caps_cache,
    _flags_dict,
    _make_fake_llama_server,
)

from core.inference.llama_cpp import _spec_draft_sampling_flags

_MTP_MODEL = "unsloth/Qwen3.6-27B-MTP-GGUF"
_NON_MTP_MODEL = "unsloth/Qwen3-7B-Instruct-GGUF"
_PROB = ["--spec-draft-sampling", "probabilistic"]

_HELP_WITH = """\
--spec-draft-n-max N                    number of tokens to draft for speculative decoding (default: 3)
--spec-draft-sampling {greedy,probabilistic}
                                        how the draft is sampled: greedy takes its argmax, probabilistic samples it and has the target
                                        verify by rejection sampling (default: greedy)
                                        (env: LLAMA_ARG_SPEC_DRAFT_SAMPLING)
--spec-type none,draft-simple,draft-mtp,ngram-mod                                        comma-separated list of types of speculative decoding to use
"""

_HELP_WITHOUT = """\
--spec-draft-n-max N                    number of tokens to draft for speculative decoding (default: 3)
--spec-type none,draft-simple,draft-mtp,ngram-mod                                        comma-separated list of types of speculative decoding to use
"""


@pytest.fixture(autouse = True)
def _no_inherited_choice(monkeypatch):
    monkeypatch.delenv("LLAMA_ARG_SPEC_DRAFT_SAMPLING", raising = False)
    monkeypatch.delenv("UNSLOTH_SPEC_DRAFT_SAMPLING", raising = False)


def _backend(monkeypatch, *, sampling = True):
    caps = {
        "found": True,
        "mtp_token": "draft-mtp",
        "supports_mtp": True,
        "supports_dspark": True,
        "mtp_probe_inconclusive": False,
        "ngram_mod_flavor": "new",
        "supports_ngram_mod": True,
        "spec_draft_n_max_flag": "--spec-draft-n-max",
        "supports_spec_draft_sampling": sampling,
    }
    monkeypatch.setattr(
        LlamaCppBackend, "probe_server_capabilities", classmethod(lambda cls, binary = None: caps)
    )
    backend = LlamaCppBackend()
    backend._nextn_predict_layers = None
    return backend


def _flags(
    backend,
    mode,
    model = _MTP_MODEL,
    extra_args = None,
    gpus = True,
    mtp_draft_path = None,
):
    return backend._build_speculative_flags(
        speculative_type = mode,
        spec_draft_n_max = None,
        extra_args = extra_args,
        model_identifier = model,
        model_path = None,
        gpus = gpus,
        binary = "/fake/llama-server",
        mtp_draft_path = mtp_draft_path,
    )


@_NEEDS_BASH
def test_probe_detects_spec_draft_sampling(tmp_path):
    fake = _make_fake_llama_server(tmp_path / "llama-server", _HELP_WITH)
    _clear_caps_cache()
    assert (
        LlamaCppBackend.probe_server_capabilities(str(fake))["supports_spec_draft_sampling"] is True
    )


@_NEEDS_BASH
def test_probe_without_the_flag_fails_closed(tmp_path):
    fake = _make_fake_llama_server(tmp_path / "llama-server", _HELP_WITHOUT)
    _clear_caps_cache()
    assert (
        LlamaCppBackend.probe_server_capabilities(str(fake))["supports_spec_draft_sampling"]
        is False
    )


def test_missing_binary_reports_no_support(tmp_path):
    _clear_caps_cache()
    caps = LlamaCppBackend.probe_server_capabilities(str(tmp_path / "no-such-llama-server"))
    assert caps["supports_spec_draft_sampling"] is False


@pytest.mark.parametrize(
    "mode, gpus", [("auto", True), ("auto", False), ("mtp", True), ("mtp+ngram", True)]
)
def test_mtp_emits_probabilistic_after_spec_type(monkeypatch, mode, gpus):
    flags = _flags(_backend(monkeypatch), mode, gpus = gpus)
    assert flags[-2:] == _PROB or flags[flags.index("--spec-draft-sampling") :][:2] == _PROB
    assert flags.index("--spec-type") < flags.index("--spec-draft-sampling")
    assert flags.count("--spec-draft-sampling") == 1


def test_separate_mtp_drafter_emits_it_too(monkeypatch, tmp_path):
    drafter = tmp_path / "mtp-gemma-4-26B-A4B-it-Q8_0.gguf"
    drafter.write_bytes(b"GGUF")
    flags = _flags(
        _backend(monkeypatch),
        "mtp",
        model = "unsloth/gemma-4-26B-A4B-it-GGUF",
        mtp_draft_path = str(drafter),
    )
    parsed = _flags_dict(flags)
    assert parsed.get("--model-draft") == str(drafter)
    assert parsed.get("--spec-draft-sampling") == "probabilistic"


def test_build_without_the_flag_emits_nothing(monkeypatch):
    flags = _flags(_backend(monkeypatch, sampling = False), "mtp")
    assert "--spec-type" in flags
    assert "--spec-draft-sampling" not in flags


@pytest.mark.parametrize(
    "mode, model", [("ngram", _MTP_MODEL), ("off", _MTP_MODEL), ("auto", _NON_MTP_MODEL)]
)
def test_no_mtp_drafter_no_flag(monkeypatch, mode, model):
    assert "--spec-draft-sampling" not in _flags(_backend(monkeypatch), mode, model = model)


def test_user_extras_choice_wins(monkeypatch):
    assert "--spec-draft-sampling" not in _flags(
        _backend(monkeypatch), "mtp", extra_args = ["--spec-draft-sampling", "greedy"]
    )
    assert "--spec-draft-sampling" not in _flags(
        _backend(monkeypatch), "mtp", extra_args = ["--spec-draft-sampling=greedy"]
    )


def test_user_spec_type_extras_emit_nothing(monkeypatch):
    assert _flags(_backend(monkeypatch), "mtp", extra_args = ["--spec-type", "draft-mtp"]) == []


def test_inherited_env_choice_wins(monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_SPEC_DRAFT_SAMPLING", "greedy")
    assert "--spec-draft-sampling" not in _flags(_backend(monkeypatch), "mtp")


def test_operator_opt_out(monkeypatch):
    monkeypatch.setenv("UNSLOTH_SPEC_DRAFT_SAMPLING", "greedy")
    assert "--spec-draft-sampling" not in _flags(_backend(monkeypatch), "mtp")
    monkeypatch.setenv("UNSLOTH_SPEC_DRAFT_SAMPLING", "nonsense")
    assert (
        _flags_dict(_flags(_backend(monkeypatch), "mtp")).get("--spec-draft-sampling")
        == "probabilistic"
    )


def test_helper_contract():
    caps = {"supports_spec_draft_sampling": True}
    assert _spec_draft_sampling_flags(caps, None) == _PROB
    assert _spec_draft_sampling_flags(None, None) == []
    assert _spec_draft_sampling_flags({}, None) == []
    assert _spec_draft_sampling_flags(caps, ["--spec-draft-sampling", "probabilistic"]) == []
