# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The llama.cpp banner shows a delta, never the cumulative release body."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from utils import llama_cpp_changelog as changes  # noqa: E402


OLD_BODY = """Automated prebuild, merged with:

- DiffusionGemma ([#24423](https://github.com/ggml-org/llama.cpp/pull/24423), commit [old0000](https://github.com/ggml-org/llama.cpp/pull/24423/commits/old0000))
- Add TML Inkling architecture ([#25731](https://github.com/ggml-org/llama.cpp/pull/25731), commit [old1111](https://github.com/ggml-org/llama.cpp/pull/25731/commits/old1111))
- kimi-k3 loading fixes ([unslothai/llama.cpp#70](https://github.com/unslothai/llama.cpp/pull/70), commit [old2222](https://github.com/unslothai/llama.cpp/pull/70/commits/old2222))
"""

LATEST_BODY = """Automated prebuild, merged with:

- Carry ggml-org#24423 (DiffusionGemma) ([unslothai/llama.cpp#107](https://github.com/unslothai/llama.cpp/pull/107), commit [new0000](https://github.com/unslothai/llama.cpp/pull/107/commits/new0000))
- Add TML Inkling architecture, rebased ([#25731](https://github.com/ggml-org/llama.cpp/pull/25731), commit [new1111](https://github.com/ggml-org/llama.cpp/pull/25731/commits/new1111))
- kimi-k3 loading fixes ([unslothai/llama.cpp#70](https://github.com/unslothai/llama.cpp/pull/70), commit [new2222](https://github.com/unslothai/llama.cpp/pull/70/commits/new2222))
- model: add GLM-5-Next (GLM-5.3-Flash) ([#27754](https://github.com/ggml-org/llama.cpp/pull/27754), commit [949f7ef](https://github.com/ggml-org/llama.cpp/pull/27754/commits/949f7ef))
- MTP for Qwen3.8-Flash-Next ([unslothai/llama.cpp#144](https://github.com/unslothai/llama.cpp/pull/144), commit [586b15e](https://github.com/unslothai/llama.cpp/pull/144/commits/586b15e))
"""


def test_delta_excludes_old_prs_after_rebase(monkeypatch):
    releases = {
        "b10698-mix-old": {"body": OLD_BODY},
        "b10715-mix-new": {
            "body": LATEST_BODY,
            "html_url": "https://github.com/unslothai/llama.cpp/releases/tag/b10715-mix-new",
        },
    }
    monkeypatch.setattr(
        changes,
        "_release_for_tag",
        lambda _repo, tag, *, force_refresh = False: releases.get(tag),
    )

    result = changes.changelog_for_update(
        "unslothai/llama.cpp", "b10698-mix-old", "b10715-mix-new"
    )

    assert result is not None
    assert [item["summary"] for item in result["changes"]] == [
        "model: add GLM-5-Next (GLM-5.3-Flash)",
        "MTP for Qwen3.8-Flash-Next",
    ]
    assert result["changes"][0]["links"] == [
        {"label": "#27754", "url": "https://github.com/ggml-org/llama.cpp/pull/27754"},
        {
            "label": "commit 949f7ef",
            "url": "https://github.com/ggml-org/llama.cpp/pull/27754/commits/949f7ef",
        },
    ]
    assert result["total_changes"] == 2
    assert result["truncated"] is False


def test_delta_fails_closed_when_either_release_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        changes,
        "_release_for_tag",
        lambda _repo, tag, *, force_refresh = False: (
            {"body": LATEST_BODY} if tag == "latest" else None
        ),
    )

    assert changes.changelog_for_update("unslothai/llama.cpp", "installed", "latest") is None


def test_text_identity_filters_old_unlinked_bullets(monkeypatch):
    releases = {
        "old": {"body": "- Fix portable build\n"},
        "new": {"body": "- Fix portable build\n- Add a new backend\n"},
    }
    monkeypatch.setattr(
        changes,
        "_release_for_tag",
        lambda _repo, tag, *, force_refresh = False: releases[tag],
    )

    result = changes.changelog_for_update("unslothai/llama.cpp", "old", "new")

    assert result is not None
    assert result["changes"] == [{"summary": "Add a new backend", "links": []}]


def test_invalid_repo_never_reaches_github(monkeypatch):
    called = False

    def fail(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(changes.urllib.request, "urlopen", fail)

    assert changes._fetch_release("owner/repository/extra", "b1") is None
    assert called is False


def test_cpp_identifiers_keep_literal_underscores():
    entry = changes._entry(
        "ggml-cuda: keep ROCm_Host and GGML_CUDA_ENABLE_UNIFIED_MEMORY=0"
    )

    assert entry["summary"] == (
        "ggml-cuda: keep ROCm_Host and GGML_CUDA_ENABLE_UNIFIED_MEMORY=0"
    )
