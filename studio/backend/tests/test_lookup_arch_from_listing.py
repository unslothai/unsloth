# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The listing's own model_info answers the lookup-table architecture: no second Hub read."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import test_kv_cache_estimation  # noqa: E402,F401
import test_llama_cpp_placement  # noqa: E402,F401 -- stubs jwt for routes.inference

import huggingface_hub  # noqa: E402
from hub.schemas.inventory import GgufVariantDetail, GgufVariantsResponse  # noqa: E402
from hub.services.models import gguf_variants as gv  # noqa: E402
from hub.utils import gguf as hub_gguf  # noqa: E402
from hub.utils import gguf_lookup_tables as glt  # noqa: E402
from hub.utils import hf_tokens  # noqa: E402

GIB = 1024**3
REPO = "unsloth/Flash-Next-GGUF"
TOKEN = "hf_oauth_" + "a" * 40
TABLE = 123456


class RepositoryNotFoundError(Exception):
    """Matched by type name, as the helpers do."""

    def __init__(self):
        super().__init__("401 Client Error")
        self.response = SimpleNamespace(
            status_code = 401, headers = {"X-Error-Message": "OAuth token verification failed"}
        )


@pytest.fixture(autouse = True)
def _isolated(monkeypatch):
    monkeypatch.setattr(gv, "_host_streams_lookup_tables", lambda: True)
    monkeypatch.setattr(gv, "_listing_lazy_mode", lambda: "auto")
    monkeypatch.setattr(gv, "_lookup_arch_cache", {})
    monkeypatch.setattr(hub_gguf, "_hub_token_answers", {})
    monkeypatch.setattr(hf_tokens, "_ambient_hf_token", lambda: (True, None))
    monkeypatch.setattr(hf_tokens, "_wire_hf_token", lambda: None)
    monkeypatch.setattr(glt, "remote_lookup_info", lambda *a, **k: ("qwen4exp", TABLE))


class _ModelInfo(SimpleNamespace):
    """Weakly referenceable, as huggingface_hub's ModelInfo is."""

    __slots__ = ("__weakref__",)


def _info(repo = REPO, arch = "qwen4exp"):
    return _ModelInfo(id = repo, siblings = [], gguf = {"architecture": arch} if arch else None)


def _listed(
    monkeypatch,
    info,
    *,
    refuse_token = False,
    token = TOKEN,
):
    """``info`` as _hub_model_info returns it, with the reads it sent."""
    sent = []

    class FakeApi:
        def __init__(self, token = None):
            self.token = token

        def model_info(self, repo_id, **kwargs):
            sent.append(self.token)
            if refuse_token and self.token is not False:
                raise RepositoryNotFoundError()
            return info

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    assert hub_gguf._hub_model_info(REPO, token) is info
    return sent


def _priced(info, token = TOKEN):
    answer = gv.VariantsAnswer(
        GgufVariantsResponse(
            repo_id = REPO,
            variants = [GgufVariantDetail(filename = "a.gguf", quant = "Q4", size_bytes = 40 * GIB)],
        ),
        None,
    )
    out = asyncio.run(
        gv._with_lookup_tables(
            answer,
            REPO,
            hf_token = token,
            offline = False,
            prefer_local_cache = False,
            listing_info = info,
        )
    )
    return [v.lookup_table_bytes for v in out.response.variants]


def _fallback(monkeypatch, arch = "qwen4exp"):
    asked = []
    monkeypatch.setattr(
        gv, "_hub_repo_architecture", lambda repo, token: asked.append((repo, token)) or arch
    )
    return asked


@pytest.mark.parametrize("token", [TOKEN, None], ids = ["own_token", "saved_token"])
def test_the_listing_architecture_skips_the_second_hub_read(monkeypatch, token):
    info = _info()
    assert _listed(monkeypatch, info, token = token) == [token]
    monkeypatch.setattr(gv, "_hub_repo_architecture", lambda *a: pytest.fail("no Hub call"))
    assert _priced(info, token) == [TABLE]
    # Cached as the separate read caches it, so the next listing asks no one either.
    assert gv._cached_hub_repo_architecture(REPO.upper(), token) == (True, "qwen4exp")


def test_the_seeded_answer_matches_the_separate_read(monkeypatch):
    info = _info()
    _listed(monkeypatch, info)
    monkeypatch.setattr(gv, "_hub_repo_architecture", lambda *a: pytest.fail("no Hub call"))
    seeded = _priced(info)
    monkeypatch.setattr(gv, "_lookup_arch_cache", {})
    asked = _fallback(monkeypatch)
    assert _priced(None) == seeded and asked == [(REPO, TOKEN)]


def test_an_anonymous_retry_after_a_refused_token_keeps_the_separate_read(monkeypatch):
    """The anonymous answer must not stand in for a read the user's own token makes."""
    info = _info()
    assert _listed(monkeypatch, info, refuse_token = True) == [TOKEN, False]
    assert not hub_gguf.hub_model_info_from_token(info)
    asked = _fallback(monkeypatch, None)
    assert _priced(info) == [0]
    assert asked == [(REPO, TOKEN)]


@pytest.mark.parametrize(
    "info",
    [None, _info(arch = None), _info(repo = "other/Repo-GGUF")],
    ids = ["no_listing", "no_gguf_metadata", "another_repo"],
)
def test_without_a_usable_listing_the_separate_read_runs(monkeypatch, info):
    if info is not None:
        _listed(monkeypatch, info)
    asked = _fallback(monkeypatch)
    assert _priced(info) == [TABLE]
    assert asked == [(REPO, TOKEN)]


@pytest.mark.parametrize(
    "info",
    [_info(), SimpleNamespace(id = REPO, siblings = [], gguf = {"architecture": "qwen4exp"})],
    ids = ["never_fetched", "not_weakly_referenceable"],
)
def test_an_unrecorded_listing_is_not_trusted(monkeypatch, info):
    """A model_info this module never recorded as the token's answer proves nothing."""
    if isinstance(info, SimpleNamespace) and not isinstance(info, _ModelInfo):
        _listed(monkeypatch, info)
    asked = _fallback(monkeypatch)
    assert _priced(info) == [TABLE]
    assert asked == [(REPO, TOKEN)]


def test_an_empty_token_keeps_the_separate_read(monkeypatch):
    """The separate read sends ``hf_token or None``; an empty string listed differently."""
    info = _info()
    _listed(monkeypatch, info, token = "")
    asked = _fallback(monkeypatch)
    assert _priced(info, "") == [TABLE]
    assert asked == [(REPO, "")]


def test_a_small_repo_still_asks_no_one(monkeypatch):
    info = _info()
    _listed(monkeypatch, info)
    monkeypatch.setattr(gv, "_hub_repo_architecture", lambda *a: pytest.fail("no Hub call"))
    answer = gv.VariantsAnswer(
        GgufVariantsResponse(
            repo_id = REPO,
            variants = [GgufVariantDetail(filename = "a.gguf", quant = "Q4", size_bytes = GIB)],
        ),
        None,
    )
    out = asyncio.run(
        gv._with_lookup_tables(
            answer,
            REPO,
            hf_token = TOKEN,
            offline = False,
            prefer_local_cache = False,
            listing_info = info,
        )
    )
    assert out is answer
    assert gv._lookup_arch_cache == {}
