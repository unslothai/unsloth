# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Windows ARM64 CUDA bundle was the one archive this installer would install unchecked.

Every other prebuilt goes through apply_approved_hashes, which drops an attempt our own
checksum manifest does not cover. The ARM64 CUDA branch could not: we publish no
windows-arm64-cuda artifact yet, so nothing covers upstream's zip, and refusing outright
would mean no CUDA llama.cpp on this hardware at all. It logged that it was installing
without a hash and did so.

GitHub's release API now reports a `digest` for every asset (checked against
ggml-org/llama.cpp b10853: 27 of 27, including llama-*-bin-win-cuda-13.4-arm64.zip and its
paired cudart archive). That is weaker than our manifest, which we compute ourselves rather
than read from the same host that serves the bytes, but it pins the download to what the API
listed, and it is enough to stop being the exception. An asset GitHub states no digest for is
now refused, so no path installs an unverified archive.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "studio"))

import install_llama_prebuilt as ip  # noqa: E402
import prebuilt_core  # noqa: E402

SHA_A = "a" * 64
SHA_B = "b" * 64


def _release(*assets: dict) -> dict:
    return {"tag_name": "b10853", "assets": list(assets)}


def _asset(name: str, digest: str | None) -> dict:
    out = {"name": name, "browser_download_url": f"https://example.invalid/{name}"}
    if digest is not None:
        out["digest"] = digest
    return out


class TestTheDigestMapReadsOnlyWhatGitHubStated:
    def test_a_sha256_digest_is_read_without_its_prefix(self):
        release = _release(_asset("bundle.zip", f"sha256:{SHA_A}"))
        assert prebuilt_core.release_asset_digests(release) == {"bundle.zip": SHA_A}

    @pytest.mark.parametrize(
        ("digest", "why"),
        [
            (None, "no digest field at all"),
            ("", "empty"),
            (SHA_A, "bare hex: GitHub did not say which algorithm produced it"),
            (f"sha512:{'c' * 128}", "a different algorithm"),
            ("sha256:nothex" + "0" * 58, "not hex"),
            ("sha256:" + "a" * 63, "too short"),
            ("sha256:" + "a" * 65, "too long"),
        ],
        ids = ["absent", "empty", "unprefixed", "sha512", "not-hex", "short", "long"],
    )
    def test_anything_else_is_left_out_rather_than_guessed_at(self, digest, why):
        release = _release(_asset("bundle.zip", digest))
        assert prebuilt_core.release_asset_digests(release) == {}, why

    def test_an_upper_case_digest_still_matches_a_lower_case_file_hash(self):
        """The verifier compares hex strings, so the case has to be normalised here."""
        release = _release(_asset("bundle.zip", "SHA256:" + SHA_A.upper()))
        assert prebuilt_core.release_asset_digests(release) == {"bundle.zip": SHA_A}

    def test_a_release_with_no_assets_is_empty_not_an_error(self):
        assert prebuilt_core.release_asset_digests({"tag_name": "b1"}) == {}
        assert prebuilt_core.release_asset_digests({"assets": "not a list"}) == {}


class TestAnAttemptWithNoDigestIsDropped:
    """The rule that makes this safe. [] is what the caller reads as "refuse"."""

    def _choice(
        self,
        name = "bundle.zip",
        runtime = None,
    ):
        return ip.AssetChoice(
            repo = "ggml-org/llama.cpp",
            tag = "b10853",
            name = name,
            url = f"https://example.invalid/{name}",
            source_label = "upstream",
            install_kind = "windows-arm64-cuda",
            runtime_name = runtime,
            runtime_url = f"https://example.invalid/{runtime}" if runtime else None,
        )

    def test_a_covered_attempt_carries_the_digest_forward(self):
        kept = ip._apply_release_digests([self._choice()], {"bundle.zip": SHA_A})
        assert [c.expected_sha256 for c in kept] == [SHA_A]

    def test_an_uncovered_attempt_is_dropped_not_installed(self):
        assert ip._apply_release_digests([self._choice()], {"other.zip": SHA_A}) == []

    def test_an_empty_digest_map_refuses_everything(self):
        assert ip._apply_release_digests([self._choice()], {}) == []

    def test_a_paired_runtime_gets_its_own_digest(self):
        kept = ip._apply_release_digests(
            [self._choice(runtime = "cudart.zip")],
            {"bundle.zip": SHA_A, "cudart.zip": SHA_B},
        )
        assert kept[0].runtime_sha256 == SHA_B

    def test_an_uncovered_runtime_unpairs_rather_than_riding_along(self):
        """Dropping the whole attempt would lose CUDA over a missing side archive, and
        keeping the pair would install one unverified. Unpairing is neither."""
        kept = ip._apply_release_digests(
            [self._choice(runtime = "cudart.zip")], {"bundle.zip": SHA_A}
        )
        assert len(kept) == 1
        assert kept[0].expected_sha256 == SHA_A
        assert kept[0].runtime_name is None
        assert kept[0].runtime_url is None
        assert kept[0].runtime_sha256 is None


class TestTheFetcherFailsClosed:
    def test_an_api_failure_is_an_empty_map_not_an_exception(self, monkeypatch):
        """A rate limit or an outage has to cost the CUDA bundle, not the whole install,
        and it must not be mistaken for "no digest needed"."""

        def _boom(repo, tag):
            raise RuntimeError("429 rate limited")

        monkeypatch.setattr(ip, "github_release", _boom)
        assert ip.github_release_asset_digests("ggml-org/llama.cpp", "b10853") == {}

    def test_it_reads_the_digests_off_the_release_it_fetched(self, monkeypatch):
        monkeypatch.setattr(
            ip, "github_release", lambda repo, tag: _release(_asset("b.zip", f"sha256:{SHA_A}"))
        )
        assert ip.github_release_asset_digests("r", "t") == {"b.zip": SHA_A}


def test_no_branch_returns_attempts_that_were_never_hash_gated():
    """The property this file exists for, asserted against the source.

    Every `return` of asset choices in the release-plan resolver goes through
    apply_approved_hashes or _apply_release_digests. A future branch that returns a bare
    list would reintroduce exactly the hole this closes.
    """
    source = (REPO_ROOT / "studio" / "install_llama_prebuilt.py").read_text(encoding = "utf-8")
    start = source.index("def resolve_release_asset_choice(")
    end = source.index("\ndef ", start + 1)
    body = source[start:end]
    returns = [
        line.strip()
        for line in body.splitlines()
        if line.strip().startswith("return ") and "attempts" not in line
    ]
    for line in returns:
        assert (
            "apply_approved_hashes" in line
            or "_apply_release_digests" in line
            or "verified" in line
            or "published_choice" in line
        ), f"a return that is not hash gated: {line}"
