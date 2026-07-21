# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Trust-anchor tests for install_whisper_prebuilt.py.

Whisper verifies each download against the release's own
whisper-prebuilt-sha256.json checksum index (the same model as
install_llama_prebuilt.py), not a committed pins file. These pin the index
parser, the fail-closed behaviour when an asset is not covered, the
tampered-manifest guard, and the newest-release resolution.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_studio = Path(__file__).resolve().parent.parent.parent
if str(_studio) not in sys.path:
    sys.path.insert(0, str(_studio))

iwp = importlib.import_module("install_whisper_prebuilt")

if not hasattr(iwp, "parse_release_checksums"):
    pytest.skip("checksum-model symbols not present - check branch", allow_module_level = True)

_A = "0" * 64
_B = "1" * 64
_TAG = "v1.9.1-unsloth.1"
_REPO = "unslothai/whisper.cpp"


def _index(**overrides) -> dict:
    payload = {
        "schema_version": 1,
        "component": "whisper.cpp",
        "release_tag": _TAG,
        "upstream_tag": "v1.9.1",
        "artifacts": {
            "whisper-v1.9.1-unsloth.1-linux-x64-cpu.tar.gz": {"sha256": _A},
            "whisper-v1.9.1-unsloth.1-linux-x64-cuda12-portable.tar.gz": {"sha256": _B},
        },
    }
    payload.update(overrides)
    return payload


# parse_release_checksums.


def test_parse_release_checksums_valid():
    out = iwp.parse_release_checksums(_REPO, _TAG, _index())
    assert out["whisper-v1.9.1-unsloth.1-linux-x64-cpu.tar.gz"] == _A
    assert out["whisper-v1.9.1-unsloth.1-linux-x64-cuda12-portable.tar.gz"] == _B


def test_parse_release_checksums_rejects_wrong_component():
    with pytest.raises(iwp.PrebuiltFallback):
        iwp.parse_release_checksums(_REPO, _TAG, _index(component = "llama.cpp"))


def test_parse_release_checksums_rejects_wrong_schema():
    with pytest.raises(iwp.PrebuiltFallback):
        iwp.parse_release_checksums(_REPO, _TAG, _index(schema_version = 999))


def test_parse_release_checksums_rejects_release_tag_mismatch():
    # A redirected/renamed release whose index names a different tag is refused.
    with pytest.raises(iwp.PrebuiltFallback):
        iwp.parse_release_checksums(_REPO, _TAG, _index(release_tag = "v1.9.1-unsloth.2"))


def test_parse_release_checksums_rejects_no_usable_entries():
    with pytest.raises(iwp.PrebuiltFallback):
        iwp.parse_release_checksums(_REPO, _TAG, _index(artifacts = {"x": {"sha256": "nope"}}))


def test_parse_release_checksums_rejects_non_dict():
    with pytest.raises(iwp.PrebuiltFallback):
        iwp.parse_release_checksums(_REPO, _TAG, ["not", "a", "dict"])


# expected_sha256_for.


def test_expected_sha256_for_covered_asset():
    checks = {"a.tar.gz": _A}
    assert iwp.expected_sha256_for(checks, "a.tar.gz") == _A


def test_expected_sha256_for_uncovered_fails_closed():
    with pytest.raises(iwp.PrebuiltFallback):
        iwp.expected_sha256_for({"a.tar.gz": _A}, "b.tar.gz")


def test_expected_sha256_for_manifest_agreement_ok():
    checks = {"a.tar.gz": _A}
    assert iwp.expected_sha256_for(checks, "a.tar.gz", manifest_sha256 = _A) == _A


def test_expected_sha256_for_manifest_disagreement_refused():
    checks = {"a.tar.gz": _A}
    with pytest.raises(iwp.PrebuiltFallback):
        iwp.expected_sha256_for(checks, "a.tar.gz", manifest_sha256 = _B)


# release tag resolution.


def test_resolve_release_tag_explicit_override_passthrough():
    assert iwp.resolve_release_tag(_REPO, published_release_tag = "v1.9.1-unsloth.2") == (
        "v1.9.1-unsloth.2"
    )


def test_resolve_release_tag_resolves_newest_when_no_override(monkeypatch):
    monkeypatch.setattr(iwp, "resolve_newest_release_tag", lambda repo: "v9.9.9-unsloth.9")
    assert iwp.resolve_release_tag(_REPO, published_release_tag = None) == "v9.9.9-unsloth.9"


def test_resolve_newest_release_tag_picks_latest_published(monkeypatch):
    releases = [
        {"tag_name": "v1.9.1-unsloth.1", "published_at": "2026-01-01T00:00:00Z"},
        {"tag_name": "v1.9.1-unsloth.3", "published_at": "2026-03-01T00:00:00Z"},
        {"tag_name": "v1.9.1-unsloth.2", "published_at": "2026-02-01T00:00:00Z"},
        {"tag_name": "draft", "published_at": "2026-09-01T00:00:00Z", "draft": True},
        {"tag_name": "pre", "published_at": "2026-09-01T00:00:00Z", "prerelease": True},
    ]
    monkeypatch.setattr(iwp, "fetch_json", lambda url: releases)
    assert iwp.resolve_newest_release_tag(_REPO) == "v1.9.1-unsloth.3"


def test_resolve_newest_release_tag_none_published_fails_closed(monkeypatch):
    monkeypatch.setattr(iwp, "fetch_json", lambda url: [{"tag_name": "d", "draft": True}])
    with pytest.raises(iwp.PrebuiltFallback):
        iwp.resolve_newest_release_tag(_REPO)


def test_pins_symbols_are_gone():
    # The committed-pins trust model was removed in favour of llama's runtime index.
    for gone in ("load_pins", "pins_path", "resolve_expected_sha256", "PINS_FILENAME"):
        assert not hasattr(iwp, gone), f"{gone} should have been removed"
