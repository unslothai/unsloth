# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An automatic text-encoder pick whose hosted pre-cast encoder cannot be fetched (HunyuanVideo-1.5: a private repo
answers 401) says the dense bf16 encoder loads, in the log and the resolved status, instead of "selected"."""

from __future__ import annotations

import threading
import types

import pytest

pytest.importorskip("huggingface_hub")
from huggingface_hub.errors import RepositoryNotFoundError  # noqa: E402

import core.inference.diffusion_te_prequant as tpq  # noqa: E402
from core.inference.video import VideoBackend, auto_te_fetch_failure  # noqa: E402

REPO = "unsloth/HunyuanVideo-1.5-FP8"


def _te():
    return tpq.TePrequantSource(
        kind = "repo", location = REPO, filename = "HunyuanVideo-1.5-text_encoder-FP8.safetensors"
    )


def _refused(monkeypatch):
    import utils.hf_xet_fallback as xet

    def _dl(repo, name, token, **kw):
        response = types.SimpleNamespace(status_code = 401, headers = {}, request = None, url = "u")
        raise RepositoryNotFoundError("401 Client Error. Repository Not Found", response = response)

    monkeypatch.setattr(xet, "hf_hub_download_with_xet_fallback", _dl)
    monkeypatch.setattr(tpq, "te_candidate_is_readable", lambda name: True)


def test_a_refused_hosted_encoder_is_reported_per_component(monkeypatch):
    _refused(monkeypatch)
    failures: dict = {}
    got = VideoBackend._fetch_te_prequant(
        VideoBackend.__new__(VideoBackend),
        {"text_encoder": _te()},
        None,
        cancel_event = threading.Event(),
        failures = failures,
    )
    assert got == ()
    assert set(failures) == {"text_encoder"}
    assert (
        REPO in failures["text_encoder"] and "RepositoryNotFoundError" in failures["text_encoder"]
    )


def test_the_failure_maps_to_the_automatic_scheme_only():
    fam = types.SimpleNamespace(te_prequant_repos = (("fp8", "text_encoder", REPO),))
    failures = {"text_encoder": f"{REPO}: RepositoryNotFoundError: 401"}
    assert auto_te_fetch_failure(fam, "fp8", failures) == failures["text_encoder"]
    assert (
        auto_te_fetch_failure(fam, "int8", failures) is None
    )  # another scheme was never asked for
    assert auto_te_fetch_failure(fam, None, failures) is None  # not an automatic pick
    assert auto_te_fetch_failure(fam, "fp8", {}) is None  # fetched fine
    assert auto_te_fetch_failure(types.SimpleNamespace(), "fp8", failures) is None


def test_a_fetched_encoder_records_nothing(monkeypatch):
    import utils.hf_xet_fallback as xet

    monkeypatch.setattr(
        xet, "hf_hub_download_with_xet_fallback", lambda repo, name, tok, **kw: "/x/" + name
    )
    monkeypatch.setattr(tpq, "te_candidate_is_readable", lambda name: True)
    failures: dict = {}
    got = VideoBackend._fetch_te_prequant(
        VideoBackend.__new__(VideoBackend),
        {"text_encoder": _te()},
        None,
        cancel_event = threading.Event(),
        failures = failures,
    )
    assert got == ("text_encoder",) and failures == {}
