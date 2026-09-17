# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The llama.cpp banner shows a delta, never the cumulative release body."""

from __future__ import annotations

import http.client
import sys
import time
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

    result = changes.changelog_for_update("unslothai/llama.cpp", "b10698-mix-old", "b10715-mix-new")

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


def test_delta_fails_closed_when_installed_notes_have_no_bullets(monkeypatch):
    # Pre-b9625-mix-2d6bd50 (2026-06-14) releases name carried PRs in prose, so the
    # bullet list is empty for a build that does carry #24423.
    releases = {
        "b9596-mix-e6f2453": {
            "body": (
                "Automated Unsloth llama.cpp prebuild for upstream b9596 "
                "+ PR #24423 @ 10a2613 (unslothai/llama.cpp@7bbfff8)."
            )
        },
        "b10715-mix-new": {"body": LATEST_BODY},
    }
    monkeypatch.setattr(
        changes,
        "_release_for_tag",
        lambda _repo, tag, *, force_refresh = False: releases[tag],
    )

    result = changes.changelog_for_update(
        "unslothai/llama.cpp", "b9596-mix-e6f2453", "b10715-mix-new"
    )

    assert result is None


def test_delta_reports_no_changes_when_target_drops_every_carried_pr(monkeypatch):
    # Not a failure: the target is always newest, so no bullets means no carries.
    releases = {
        "old": {"body": OLD_BODY},
        "new": {"body": "Automated Unsloth llama.cpp prebuild for upstream b10800."},
    }
    monkeypatch.setattr(
        changes,
        "_release_for_tag",
        lambda _repo, tag, *, force_refresh = False: releases[tag],
    )

    result = changes.changelog_for_update("unslothai/llama.cpp", "old", "new")

    assert result is not None
    assert result["changes"] == []
    assert result["total_changes"] == 0


def test_invalid_repo_never_reaches_github(monkeypatch):
    called = False

    def fail(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(changes.urllib.request, "urlopen", fail)

    assert changes._fetch_release("owner/repository/extra", "b1") is None
    assert called is False


def test_cpp_identifiers_keep_literal_underscores():
    entry = changes._entry("ggml-cuda: keep ROCm_Host and GGML_CUDA_ENABLE_UNIFIED_MEMORY=0")

    assert entry["summary"] == ("ggml-cuda: keep ROCm_Host and GGML_CUDA_ENABLE_UNIFIED_MEMORY=0")


def test_repo_with_a_dot_segment_never_reaches_github(monkeypatch):
    # "." is in the repo character class, so "owner/.." passed the shape check.
    called = False

    def fail(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(changes.urllib.request, "urlopen", fail)

    for repo in ("../etc", "owner/..", "..", "../rate_limit"):
        assert changes._fetch_release(repo, "b1") is None
    assert called is False


def test_pull_and_issue_urls_share_one_identity_namespace():
    # One number space, so the same reference written three ways must match itself.
    from_pull = changes._identities(
        "Fix a crash ([ggml-org/llama.cpp#900](https://github.com/ggml-org/llama.cpp/pull/900))"
    )
    from_issue = changes._identities(
        "Fix a crash ([the issue](https://github.com/ggml-org/llama.cpp/issues/900))"
    )
    from_text = changes._identities("Fix a crash, ggml-org#900")

    assert from_pull & from_issue
    assert from_issue & from_text


def test_a_title_containing_the_metadata_delimiter_is_not_truncated():
    entry = changes._entry(
        "vulkan: handle ([a],[b]) tuples "
        "([unslothai/llama.cpp#5](https://github.com/unslothai/llama.cpp/pull/5))"
    )

    assert entry["summary"] == "vulkan: handle ([a],[b]) tuples"
    assert [link["url"] for link in entry["links"]] == [
        "https://github.com/unslothai/llama.cpp/pull/5"
    ]


def test_a_bullet_with_no_summary_does_not_suppress_a_later_real_one(monkeypatch):
    releases = {
        "old": {"body": "Automated prebuild, merged with:\n\n- Unrelated carry\n"},
        "new": {
            "body": (
                "Automated prebuild, merged with:\n\n"
                "- Unrelated carry\n"
                "- [ ](https://github.com/unslothai/llama.cpp/pull/900)\n"
                "- Real change for PR 900 "
                "([unslothai/llama.cpp#900](https://github.com/unslothai/llama.cpp/pull/900))\n"
            ),
        },
    }
    monkeypatch.setattr(
        changes,
        "_release_for_tag",
        lambda _repo, tag, *, force_refresh = False: releases[tag],
    )

    result = changes.changelog_for_update("unslothai/llama.cpp", "old", "new")

    assert [item["summary"] for item in result["changes"]] == ["Real change for PR 900"]


def test_a_missing_target_body_fails_closed(monkeypatch):
    # Distinct from a prose-only target, which legitimately means "carries nothing".
    releases = {"old": {"body": OLD_BODY}, "new": {}}
    monkeypatch.setattr(
        changes,
        "_release_for_tag",
        lambda _repo, tag, *, force_refresh = False: releases[tag],
    )

    assert changes.changelog_for_update("unslothai/llama.cpp", "old", "new") is None


def test_forced_refresh_is_floored_to_one_fetch_per_interval(monkeypatch):
    calls = []
    monkeypatch.setattr(changes, "_release_memo", {})
    monkeypatch.setattr(changes, "_release_failed_at", {})
    monkeypatch.setattr(changes, "_release_forced_at", {})
    monkeypatch.setattr(
        changes,
        "_fetch_release",
        lambda _repo, tag, timeout = 5.0: calls.append(tag) or {"body": "- x"},
    )

    for _ in range(10):
        changes._release_for_tag("unslothai/llama.cpp", "b1", force_refresh = True)

    # Without the floor this was ten uncached GitHub round trips per ten clicks.
    assert len(calls) == 1


def test_an_expired_body_is_not_resurrected_by_a_recent_failure(monkeypatch):
    key = ("unslothai/llama.cpp", "b1")
    stale = time.monotonic() - (changes.RELEASE_CACHE_TTL_SECONDS * 2)
    monkeypatch.setattr(changes, "_release_memo", {key: (stale, {"body": "- ancient"})})
    monkeypatch.setattr(changes, "_release_failed_at", {key: time.monotonic()})
    monkeypatch.setattr(changes, "_release_forced_at", {})

    assert changes._release_for_tag(*key) is None


def test_release_page_url_is_github_or_nothing():
    assert changes.release_page_url("unslothai/llama.cpp", "b1/2") == (
        "https://github.com/unslothai/llama.cpp/releases/tag/b1%2F2"
    )
    assert changes.release_page_url("owner/..", "b1") is None
    assert changes.release_page_url("unslothai/llama.cpp", "") is None


def test_unavailable_reason_separates_permanent_from_transient(monkeypatch):
    # Predating the bullet format is permanent, so the banner offers no Retry.
    releases = {
        "prose": {"body": "Automated Unsloth llama.cpp prebuild for upstream b9000."},
        "itemised": {"body": OLD_BODY},
    }
    monkeypatch.setattr(
        changes,
        "_release_for_tag",
        lambda _repo, tag, *, force_refresh = False: releases.get(tag),
    )

    assert (
        changes.unavailable_reason("unslothai/llama.cpp", "prose", "itemised")
        == "notes_not_itemised"
    )
    assert (
        changes.unavailable_reason("unslothai/llama.cpp", "missing", "itemised")
        == "release_notes_unavailable"
    )


def test_a_noncumulative_repo_is_never_compared(monkeypatch):
    # Measured on real upstream releases: b10721 -> b10734 reported 5 "changes"
    # (commit-message lines and an attestation URL) and dropped the 324 bullets in
    # the 9 releases between them, because per-release notes are not cumulative.
    upstream = {
        "b10721": {"body": "<details open>\n\n- webgpu : avoid a crash (#28045)\n"},
        "b10734": {"body": "<details open>\n\n- metal : enable Metal 4.0 (#27461)\n"},
    }
    monkeypatch.setattr(
        changes,
        "_release_for_tag",
        lambda _repo, tag, *, force_refresh = False: upstream.get(tag),
    )

    assert changes.changelog_for_update("ggml-org/llama.cpp", "b10721", "b10734") is None
    # Permanent: that repo will never publish cumulative notes, so no Retry.
    assert (
        changes.unavailable_reason("ggml-org/llama.cpp", "b10721", "b10734")
        == "notes_not_comparable"
    )


def test_a_case_variant_of_the_official_repo_is_still_official(monkeypatch):
    # GitHub resolves this to the same repository and the installer persists the
    # spelling, so a case-sensitive check would label the official repo custom.
    releases = {"old": {"body": OLD_BODY}, "new": {"body": LATEST_BODY}}
    monkeypatch.setattr(
        changes,
        "_release_for_tag",
        lambda _repo, tag, *, force_refresh = False: releases.get(tag),
    )

    result = changes.changelog_for_update("UnslothAI/Llama.cpp", "old", "new")

    assert result is not None
    assert [item["summary"] for item in result["changes"]] == [
        "model: add GLM-5-Next (GLM-5.3-Flash)",
        "MTP for Qwen3.8-Flash-Next",
    ]


def test_a_bodyless_target_is_transient_not_permanent(monkeypatch):
    # The target is the newest release, so a missing body is a publishing gap that
    # may be filled in: keep the Retry.
    releases = {"itemised": {"body": OLD_BODY}, "bodyless": {}}
    monkeypatch.setattr(
        changes,
        "_release_for_tag",
        lambda _repo, tag, *, force_refresh = False: releases.get(tag),
    )

    assert (
        changes.unavailable_reason("unslothai/llama.cpp", "itemised", "bodyless")
        == "release_notes_unavailable"
    )


def test_a_truncated_response_does_not_escape_to_the_caller(monkeypatch):
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, size = -1):
            raise http.client.IncompleteRead(b"partial")

    monkeypatch.setattr(changes.urllib.request, "urlopen", lambda *_a, **_k: _Response())

    # IncompleteRead is an HTTPException, not an OSError.
    assert changes._fetch_release_blocking("unslothai/llama.cpp", "b1", 5.0) is None


def test_an_oversized_release_body_is_rejected(monkeypatch):
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, size = -1):
            return b"x" * (changes.MAX_RELEASE_BYTES + 1)

    monkeypatch.setattr(changes.urllib.request, "urlopen", lambda *_a, **_k: _Response())

    assert changes._fetch_release_blocking("unslothai/llama.cpp", "b1", 5.0) is None
