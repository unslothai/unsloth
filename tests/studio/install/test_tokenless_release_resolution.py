# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for tokenless (no GH_TOKEN) llama.cpp release resolution.

Cover the github.com fallback that resolves upstream releases, their assets and
their sha256 digests when the rate-limited api.github.com REST surface is
unavailable. All I/O is monkeypatched.
"""

import sys
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
STUDIO_DIR = ROOT / "studio"
if str(STUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIO_DIR))

import install_llama_prebuilt as INSTALL_LLAMA_PREBUILT  # noqa: E402
import prebuilt_core as PREBUILT_CORE  # noqa: E402
from install_llama_prebuilt import HostInfo  # noqa: E402

MOD = INSTALL_LLAMA_PREBUILT
CORE = PREBUILT_CORE
UPSTREAM = MOD.UPSTREAM_REPO

DIGEST_A = "8a3a5d19721cbb5cfef58a28eb197d39ffc19a8cbf8a9a9927ecbff36d5c5806"
DIGEST_B = "758b9df442bf57eb68f1aee566a9c6582601386703dc993ecc68b8fa91a5e3bc"


def _host(**overrides) -> HostInfo:
    base = dict(
        system = "Darwin",
        machine = "arm64",
        is_windows = False,
        is_linux = False,
        is_macos = True,
        is_x86_64 = False,
        is_arm64 = True,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
        has_rocm = False,
        rocm_gfx_target = None,
    )
    base.update(overrides)
    return HostInfo(**base)


def _linux_cpu(**overrides) -> HostInfo:
    return _host(
        system = "Linux",
        machine = "x86_64",
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        **overrides,
    )


def _asset_row(repo: str, tag: str, name: str, digest: str | None) -> str:
    """One asset row shaped like the fragment github.com serves for a release."""
    row = (
        f'<a href="/{repo}/releases/download/{tag}/{name}" rel="nofollow" '
        f'class="wb-break-all"><span class="text-bold">{name}</span></a>'
    )
    if digest is not None:
        row += (
            f'<span class="Truncate-text">sha256:{digest}</span>'
            f'<clipboard-copy id="clipboard-button-sha256:{digest}" '
            f'aria-label="Copy to clipboard digest for {name}" type="button" '
            f'value="sha256:{digest}" class="Button--invisible">copy</clipboard-copy>'
        )
    return row


def _expanded_assets(repo: str, tag: str, assets: dict[str, str | None]) -> str:
    rows = "".join(_asset_row(repo, tag, name, digest) for name, digest in assets.items())
    return f'<div class="Box">{rows}</div>'


def _atom(repo: str, tags: list[str]) -> str:
    entries = "".join(
        f'<entry><link rel="alternate" type="text/html" '
        f'href="https://github.com/{repo}/releases/tag/{tag}"/><title>{tag}</title></entry>'
        for tag in tags
    )
    return f'<?xml version="1.0"?><feed>{entries}</feed>'


class _Web:
    """Stands in for the github.com web host, recording every URL fetched."""

    def __init__(self, pages: dict[str, str]):
        self.pages = pages
        self.urls: list[str] = []

    def __call__(self, url: str, **kwargs) -> bytes:
        self.urls.append(url)
        for suffix, body in self.pages.items():
            if url.endswith(suffix):
                return body.encode("utf-8")
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)


def _tag_page(prerelease: bool = True) -> str:
    """A release page, labelled the way github.com labels one.

    Prerelease by default because ggml-org marks every bNNNN build release that way.
    """
    label = (
        '<span class="Label Label--warning Label--large">Pre-release</span>'
        if prerelease
        else '<span class="Label Label--success Label--large">Latest</span>'
    )
    return f"<html><h1>release</h1>{label}</html>"


def _install_web(
    monkeypatch,
    pages: dict[str, str],
    *,
    prerelease: bool = True,
) -> _Web:
    # Every release whose assets are served needs its release page served too, since
    # that is where the prerelease status is read from.
    pages = dict(pages)
    for suffix in [key for key in pages if "expanded_assets/" in key]:
        tag = suffix.rsplit("/", 1)[-1]
        pages.setdefault(f"releases/tag/{tag}", _tag_page(prerelease))
    web = _Web(pages)
    monkeypatch.setattr(CORE, "download_bytes", lambda ops, url, **kw: web(url, **kw))
    return web


def _rest_403(*args, **kwargs):
    raise RuntimeError(
        "GitHub API returned 403 for "
        "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=100&page=1"
        "; set GH_TOKEN or GITHUB_TOKEN to avoid GitHub API rate limits"
    )


def _boom(*args, **kwargs):
    raise AssertionError("the tokenless path must not be used here")


# ── the release feed ──


class TestWebReleaseTags:
    def test_parses_tags_newest_first(self, monkeypatch):
        _install_web(monkeypatch, {"releases.atom": _atom(UPSTREAM, ["b11070", "b11069"])})
        assert MOD.web_release_tags(UPSTREAM) == ["b11070", "b11069"]

    def test_skips_the_versioned_pointer_release(self, monkeypatch):
        # Upstream's /releases/latest points at v0.4.1, whose only asset is a text
        # file naming a nightly. Selecting it would build 404-bound archive URLs.
        _install_web(
            monkeypatch, {"releases.atom": _atom(UPSTREAM, ["v0.4.1", "b11070", "b11069"])}
        )
        assert MOD.web_release_tags(UPSTREAM)[0] == "v0.4.1"
        assert MOD.upstream_web_release_tags(UPSTREAM) == ["b11070", "b11069"]

    def test_honours_the_limit(self, monkeypatch):
        _install_web(
            monkeypatch,
            {"releases.atom": _atom(UPSTREAM, [f"b{n}" for n in range(11070, 11060, -1)])},
        )
        assert MOD.web_release_tags(UPSTREAM, limit = 3) == ["b11070", "b11069", "b11068"]


# ── the release page ──


class TestWebReleasePayload:
    def test_binds_each_digest_to_its_own_asset(self, monkeypatch):
        _install_web(
            monkeypatch,
            {
                "expanded_assets/b11070": _expanded_assets(
                    UPSTREAM,
                    "b11070",
                    {
                        "llama-b11070-bin-macos-arm64.tar.gz": DIGEST_A,
                        "llama-b11070-bin-ubuntu-x64.tar.gz": DIGEST_B,
                    },
                )
            },
        )
        release = MOD.web_release_payload(UPSTREAM, "b11070")
        assert release["tag_name"] == "b11070"
        assert MOD.release_asset_digests(release) == {
            "llama-b11070-bin-macos-arm64.tar.gz": DIGEST_A,
            "llama-b11070-bin-ubuntu-x64.tar.gz": DIGEST_B,
        }

    def test_urls_are_the_deterministic_release_asset_urls(self, monkeypatch):
        _install_web(
            monkeypatch,
            {
                "expanded_assets/b11070": _expanded_assets(
                    UPSTREAM, "b11070", {"llama-b11070-bin-macos-arm64.tar.gz": DIGEST_A}
                )
            },
        )
        release = MOD.web_release_payload(UPSTREAM, "b11070")
        assert release["assets"][0]["browser_download_url"] == (
            "https://github.com/ggml-org/llama.cpp/releases/download/b11070/"
            "llama-b11070-bin-macos-arm64.tar.gz"
        )

    def test_drops_an_asset_with_no_published_digest(self, monkeypatch):
        _install_web(
            monkeypatch,
            {
                "expanded_assets/b11070": _expanded_assets(
                    UPSTREAM,
                    "b11070",
                    {
                        "llama-b11070-bin-macos-arm64.tar.gz": None,
                        "llama-b11070-bin-ubuntu-x64.tar.gz": DIGEST_B,
                    },
                )
            },
        )
        release = MOD.web_release_payload(UPSTREAM, "b11070")
        assert [asset["name"] for asset in release["assets"]] == [
            "llama-b11070-bin-ubuntu-x64.tar.gz"
        ]

    def test_drops_an_asset_whose_digest_is_stated_twice_and_differs(self, monkeypatch):
        page = _expanded_assets(
            UPSTREAM, "b11070", {"llama-b11070-bin-macos-arm64.tar.gz": DIGEST_A}
        ) + (
            '<clipboard-copy aria-label="Copy to clipboard digest for '
            f'llama-b11070-bin-macos-arm64.tar.gz" value="sha256:{DIGEST_B}"></clipboard-copy>'
        )
        _install_web(monkeypatch, {"expanded_assets/b11070": page})
        with pytest.raises(RuntimeError, match = "no digest-bearing assets"):
            MOD.web_release_payload(UPSTREAM, "b11070")

    def test_ignores_a_link_belonging_to_another_release(self, monkeypatch):
        page = _expanded_assets(
            UPSTREAM, "b11069", {"llama-b11069-bin-macos-arm64.tar.gz": DIGEST_A}
        )
        _install_web(monkeypatch, {"expanded_assets/b11070": page})
        with pytest.raises(RuntimeError, match = "no digest-bearing assets"):
            MOD.web_release_payload(UPSTREAM, "b11070")

    def test_a_digest_without_a_published_asset_is_not_invented(self, monkeypatch):
        page = (
            '<clipboard-copy aria-label="Copy to clipboard digest for ghost.tar.gz" '
            f'value="sha256:{DIGEST_A}"></clipboard-copy>'
        )
        _install_web(monkeypatch, {"expanded_assets/b11070": page})
        with pytest.raises(RuntimeError, match = "no digest-bearing assets"):
            MOD.web_release_payload(UPSTREAM, "b11070")

    def test_refuses_an_implausibly_large_page(self, monkeypatch):
        monkeypatch.setattr(CORE, "download_bytes", lambda ops, url, **kw: b"x" * (8 * 1024 * 1024))
        with pytest.raises(RuntimeError, match = "implausibly large"):
            MOD.web_release_payload(UPSTREAM, "b11070")

    def test_never_sends_an_authorization_header(self, monkeypatch):
        seen = {}

        def capture(ops, url, **kwargs):
            seen[url] = kwargs.get("headers") or {}
            return _expanded_assets(
                UPSTREAM, "b11070", {"llama-b11070-bin-macos-arm64.tar.gz": DIGEST_A}
            ).encode("utf-8")

        monkeypatch.setattr(CORE, "download_bytes", capture)
        MOD.web_release_payload(UPSTREAM, "b11070")
        assert seen
        for headers in seen.values():
            assert not any(key.lower() == "authorization" for key in headers)


# ── latest_upstream_release_tag ──


class TestLatestUpstreamReleaseTag:
    def test_uses_the_rest_api_first(self, monkeypatch):
        monkeypatch.setattr(MOD, "fetch_json", lambda url: {"tag_name": "b500"})
        monkeypatch.setattr(MOD, "web_release_tags", _boom)
        assert MOD.latest_upstream_release_tag() == "b500"

    def test_falls_back_to_the_feed_on_a_rate_limit(self, monkeypatch):
        monkeypatch.setattr(MOD, "fetch_json", _rest_403)
        _install_web(monkeypatch, {"releases.atom": _atom(UPSTREAM, ["v0.4.1", "b11070"])})
        assert MOD.latest_upstream_release_tag() == "b11070"

    def test_falls_back_when_rest_states_no_tag(self, monkeypatch):
        monkeypatch.setattr(MOD, "fetch_json", lambda url: {})
        _install_web(monkeypatch, {"releases.atom": _atom(UPSTREAM, ["b11070"])})
        assert MOD.latest_upstream_release_tag() == "b11070"

    def test_both_paths_failing_reports_both_causes(self, monkeypatch):
        monkeypatch.setattr(MOD, "fetch_json", _rest_403)
        _install_web(monkeypatch, {})
        with pytest.raises(RuntimeError) as excinfo:
            MOD.latest_upstream_release_tag()
        assert "403" in str(excinfo.value)
        assert "feed fallback also failed" in str(excinfo.value)


# ── iter_release_payloads_by_time ──


class TestIterReleasePayloads:
    def test_rest_listing_wins_when_it_answers(self, monkeypatch):
        rest = {"tag_name": "b1234", "assets": []}
        monkeypatch.setattr(MOD, "github_releases", lambda repo, **kw: [rest])
        monkeypatch.setattr(MOD, "web_release_payload", _boom)
        monkeypatch.setattr(MOD, "web_release_tags", _boom)
        assert list(MOD.iter_release_payloads_by_time(UPSTREAM, "", "latest")) == [rest]

    def test_rate_limited_listing_walks_the_feed_newest_first(self, monkeypatch):
        monkeypatch.setattr(MOD, "github_releases", _rest_403)
        _install_web(
            monkeypatch,
            {
                "releases.atom": _atom(UPSTREAM, ["b11070", "b11069"]),
                "expanded_assets/b11070": _expanded_assets(
                    UPSTREAM, "b11070", {"llama-b11070-bin-macos-arm64.tar.gz": DIGEST_A}
                ),
                "expanded_assets/b11069": _expanded_assets(
                    UPSTREAM, "b11069", {"llama-b11069-bin-macos-arm64.tar.gz": DIGEST_B}
                ),
            },
        )
        got = list(MOD.iter_release_payloads_by_time(UPSTREAM, "", "latest"))
        assert [release["tag_name"] for release in got] == ["b11070", "b11069"]

    def test_the_walk_is_lazy(self, monkeypatch):
        monkeypatch.setattr(MOD, "github_releases", _rest_403)
        web = _install_web(
            monkeypatch,
            {
                "releases.atom": _atom(UPSTREAM, ["b11070", "b11069"]),
                "expanded_assets/b11070": _expanded_assets(
                    UPSTREAM, "b11070", {"llama-b11070-bin-macos-arm64.tar.gz": DIGEST_A}
                ),
                "expanded_assets/b11069": _expanded_assets(
                    UPSTREAM, "b11069", {"llama-b11069-bin-macos-arm64.tar.gz": DIGEST_B}
                ),
            },
        )
        releases = MOD.iter_release_payloads_by_time(UPSTREAM, "", "latest")
        assert next(iter(releases))["tag_name"] == "b11070"
        assert not any("b11069" in url for url in web.urls)

    def test_an_unreadable_release_is_skipped_not_fatal(self, monkeypatch):
        monkeypatch.setattr(MOD, "github_releases", _rest_403)
        _install_web(
            monkeypatch,
            {
                "releases.atom": _atom(UPSTREAM, ["b11070", "b11069"]),
                "expanded_assets/b11069": _expanded_assets(
                    UPSTREAM, "b11069", {"llama-b11069-bin-macos-arm64.tar.gz": DIGEST_B}
                ),
            },
        )
        got = list(MOD.iter_release_payloads_by_time(UPSTREAM, "", "latest"))
        assert [release["tag_name"] for release in got] == ["b11069"]

    def test_a_pinned_tag_resolves_through_its_release_page(self, monkeypatch):
        # The macOS floor pin (b9415) arrives here as requested_tag, and is exactly the
        # population the pin exists to serve, so it must not be left on the source build.
        monkeypatch.setattr(MOD, "github_release", _rest_403)
        _install_web(
            monkeypatch,
            {
                "expanded_assets/b9415": _expanded_assets(
                    UPSTREAM, "b9415", {"llama-b9415-bin-macos-arm64.tar.gz": DIGEST_A}
                )
            },
        )
        got = list(MOD.iter_release_payloads_by_time(UPSTREAM, "", "b9415"))
        assert [release["tag_name"] for release in got] == ["b9415"]

    def test_the_fork_never_uses_the_upstream_web_path(self, monkeypatch):
        monkeypatch.setattr(MOD, "github_releases", _rest_403)
        monkeypatch.setattr(MOD, "web_release_tags", _boom)
        with pytest.raises(RuntimeError, match = "403"):
            list(MOD.iter_release_payloads_by_time(MOD.DEFAULT_PUBLISHED_REPO, "", "latest"))

    def test_the_escape_hatch_disables_the_web_path(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_LLAMA_DISABLE_DOWNLOAD_HOST_RESOLVE", "1")
        monkeypatch.setattr(MOD, "github_releases", _rest_403)
        monkeypatch.setattr(MOD, "web_release_tags", _boom)
        with pytest.raises(RuntimeError, match = "403"):
            list(MOD.iter_release_payloads_by_time(UPSTREAM, "", "latest"))

    def test_both_paths_failing_reports_both_causes(self, monkeypatch):
        monkeypatch.setattr(MOD, "github_releases", _rest_403)
        _install_web(monkeypatch, {})
        with pytest.raises(RuntimeError) as excinfo:
            list(MOD.iter_release_payloads_by_time(UPSTREAM, "", "latest"))
        assert "403" in str(excinfo.value)
        assert "feed fallback also failed" in str(excinfo.value)


# ── end to end, through the real planner ──


def _plans(host, requested = "latest"):
    return MOD.resolve_simple_install_release_plans(requested, host, UPSTREAM, "")


class TestEndToEnd:
    def test_macos_installs_a_digest_verified_prebuilt_when_rest_is_rate_limited(self, monkeypatch):
        monkeypatch.setattr(MOD, "github_releases", _rest_403)
        _install_web(
            monkeypatch,
            {
                "releases.atom": _atom(UPSTREAM, ["v0.4.1", "b11070"]),
                "expanded_assets/b11070": _expanded_assets(
                    UPSTREAM, "b11070", {"llama-b11070-bin-macos-arm64.tar.gz": DIGEST_A}
                ),
            },
        )
        requested, plans = _plans(_host())
        assert requested == "latest"
        assert len(plans) == 1
        attempt = plans[0].attempts[0]
        assert attempt.name == "llama-b11070-bin-macos-arm64.tar.gz"
        assert attempt.url == (
            "https://github.com/ggml-org/llama.cpp/releases/download/b11070/"
            "llama-b11070-bin-macos-arm64.tar.gz"
        )
        # The point of the whole path: the archive is still bound to a published digest.
        assert attempt.expected_sha256 == DIGEST_A

    def test_linux_x64_installs_a_digest_verified_prebuilt(self, monkeypatch):
        monkeypatch.setattr(MOD, "github_releases", _rest_403)
        _install_web(
            monkeypatch,
            {
                "releases.atom": _atom(UPSTREAM, ["b11070"]),
                "expanded_assets/b11070": _expanded_assets(
                    UPSTREAM, "b11070", {"llama-b11070-bin-ubuntu-x64.tar.gz": DIGEST_B}
                ),
            },
        )
        _requested, plans = _plans(_linux_cpu())
        attempt = plans[0].attempts[0]
        assert attempt.name == "llama-b11070-bin-ubuntu-x64.tar.gz"
        assert attempt.expected_sha256 == DIGEST_B

    def test_an_asset_with_no_digest_is_refused_not_installed(self, monkeypatch):
        monkeypatch.setattr(MOD, "github_releases", _rest_403)
        _install_web(
            monkeypatch,
            {
                "releases.atom": _atom(UPSTREAM, ["b11070"]),
                "expanded_assets/b11070": _expanded_assets(
                    UPSTREAM, "b11070", {"llama-b11070-bin-macos-arm64.tar.gz": None}
                ),
            },
        )
        with pytest.raises(MOD.PrebuiltFallback):
            _plans(_host())

    def test_a_release_without_this_hosts_archive_falls_back_to_an_older_one(self, monkeypatch):
        monkeypatch.setattr(MOD, "github_releases", _rest_403)
        _install_web(
            monkeypatch,
            {
                "releases.atom": _atom(UPSTREAM, ["b11070", "b11069"]),
                # b11070 published Linux only, so a macOS host must walk back.
                "expanded_assets/b11070": _expanded_assets(
                    UPSTREAM, "b11070", {"llama-b11070-bin-ubuntu-x64.tar.gz": DIGEST_B}
                ),
                "expanded_assets/b11069": _expanded_assets(
                    UPSTREAM, "b11069", {"llama-b11069-bin-macos-arm64.tar.gz": DIGEST_A}
                ),
            },
        )
        _requested, plans = _plans(_host())
        assert plans[0].attempts[0].name == "llama-b11069-bin-macos-arm64.tar.gz"


# ── prerelease parity between the two paths ──


class TestPrereleaseStatus:
    def test_the_payload_states_the_status_it_read(self, monkeypatch):
        _install_web(
            monkeypatch,
            {
                "expanded_assets/b11070": _expanded_assets(
                    UPSTREAM, "b11070", {"llama-b11070-bin-macos-arm64.tar.gz": DIGEST_A}
                )
            },
            prerelease = True,
        )
        assert MOD.web_release_payload(UPSTREAM, "b11070")["prerelease"] is True

    def test_a_release_without_the_label_is_not_reported_as_prerelease(self, monkeypatch):
        _install_web(
            monkeypatch,
            {
                "expanded_assets/b11070": _expanded_assets(
                    UPSTREAM, "b11070", {"llama-b11070-bin-macos-arm64.tar.gz": DIGEST_A}
                )
            },
            prerelease = False,
        )
        assert MOD.web_release_payload(UPSTREAM, "b11070")["prerelease"] is False

    def test_an_upstream_build_release_stays_selectable_when_marked_prerelease(self):
        # ggml-org marks every bNNNN build prerelease; excluding them strands the
        # upstream path on the newest release that is not one, which ships no prebuilt.
        assert MOD.release_is_selectable(UPSTREAM, {"tag_name": "b11070", "prerelease": True})

    def test_a_non_build_prerelease_is_still_refused(self):
        assert not MOD.release_is_selectable(
            UPSTREAM, {"tag_name": "v0.5.0-rc1", "prerelease": True}
        )

    def test_the_fork_keeps_the_plain_rule(self):
        assert not MOD.release_is_selectable(
            MOD.DEFAULT_PUBLISHED_REPO, {"tag_name": "b11070", "prerelease": True}
        )

    def test_a_draft_is_never_selectable(self):
        assert not MOD.release_is_selectable(UPSTREAM, {"tag_name": "b11070", "draft": True})

    def test_the_rest_path_keeps_a_prerelease_build(self, monkeypatch):
        rest = {"tag_name": "b11070", "prerelease": True, "draft": False, "assets": []}
        monkeypatch.setattr(MOD, "github_releases", lambda repo, **kw: [rest])
        monkeypatch.setattr(MOD, "web_release_payload", _boom)
        monkeypatch.setattr(MOD, "web_release_tags", _boom)
        got = list(MOD.iter_release_payloads_by_time(UPSTREAM, "", "latest"))
        assert [release["tag_name"] for release in got] == ["b11070"]

    def test_the_web_path_keeps_the_same_prerelease_build(self, monkeypatch):
        monkeypatch.setattr(MOD, "github_releases", _rest_403)
        _install_web(
            monkeypatch,
            {
                "releases.atom": _atom(UPSTREAM, ["b11070"]),
                "expanded_assets/b11070": _expanded_assets(
                    UPSTREAM, "b11070", {"llama-b11070-bin-macos-arm64.tar.gz": DIGEST_A}
                ),
            },
            prerelease = True,
        )
        got = list(MOD.iter_release_payloads_by_time(UPSTREAM, "", "latest"))
        assert [release["tag_name"] for release in got] == ["b11070"]
        assert got[0]["prerelease"] is True

    def test_the_web_walk_skips_a_release_the_rest_path_would_filter(self, monkeypatch):
        monkeypatch.setattr(MOD, "github_releases", _rest_403)
        _install_web(
            monkeypatch,
            {
                "releases.atom": _atom(UPSTREAM, ["b11070"]),
                "expanded_assets/b11070": _expanded_assets(
                    UPSTREAM, "b11070", {"llama-b11070-bin-macos-arm64.tar.gz": DIGEST_A}
                ),
            },
        )
        monkeypatch.setattr(MOD, "release_is_selectable", lambda repo, release: False)
        assert list(MOD.iter_release_payloads_by_time(UPSTREAM, "", "latest")) == []


# ── a pinned published release ──


class TestPinnedPublishedRelease:
    def test_a_pinned_published_release_resolves_through_its_release_page(self, monkeypatch):
        monkeypatch.setattr(MOD, "github_release", _rest_403)
        _install_web(
            monkeypatch,
            {
                "expanded_assets/b9415": _expanded_assets(
                    UPSTREAM, "b9415", {"llama-b9415-bin-macos-arm64.tar.gz": DIGEST_A}
                )
            },
        )
        got = list(MOD.iter_release_payloads_by_time(UPSTREAM, "b9415", "latest"))
        assert [release["tag_name"] for release in got] == ["b9415"]

    def test_a_pinned_published_release_on_the_fork_still_re_raises(self, monkeypatch):
        monkeypatch.setattr(MOD, "github_release", _rest_403)
        monkeypatch.setattr(MOD, "web_release_payload", _boom)
        with pytest.raises(RuntimeError, match = "403"):
            list(MOD.iter_release_payloads_by_time(MOD.DEFAULT_PUBLISHED_REPO, "b9415", "latest"))

    def test_rest_still_wins_for_a_pinned_published_release(self, monkeypatch):
        rest = {"tag_name": "b9415", "assets": []}
        monkeypatch.setattr(MOD, "github_release", lambda repo, tag: rest)
        monkeypatch.setattr(MOD, "web_release_payload", _boom)
        assert list(MOD.iter_release_payloads_by_time(UPSTREAM, "b9415", "latest")) == [rest]


class TestPinnedTagHttpFailures:
    """HTTPError subclasses URLError, so the HTTPError clause must route the fallback."""

    def _http(self, code):
        def call(*args, **kwargs):
            raise urllib.error.HTTPError("https://api.github.com/x", code, "boom", None, None)

        return call

    @pytest.mark.parametrize("code", [500, 502, 503, 504])
    def test_a_retryable_http_failure_reaches_the_release_page(self, monkeypatch, code):
        monkeypatch.setattr(MOD, "github_release", self._http(code))
        _install_web(
            monkeypatch,
            {
                "expanded_assets/b9415": _expanded_assets(
                    UPSTREAM, "b9415", {"llama-b9415-bin-macos-arm64.tar.gz": DIGEST_A}
                )
            },
        )
        got = list(MOD.iter_release_payloads_by_time(UPSTREAM, "", "b9415"))
        assert [release["tag_name"] for release in got] == ["b9415"]

    def test_a_404_still_scans_rather_than_reading_the_page(self, monkeypatch):
        # The tag does not exist, so its release page cannot answer either.
        monkeypatch.setattr(MOD, "github_release", self._http(404))
        monkeypatch.setattr(MOD, "github_releases", lambda repo, **kw: [])
        monkeypatch.setattr(MOD, "web_release_payload", _boom)
        monkeypatch.setattr(MOD, "web_release_tags", _boom)
        assert list(MOD.iter_release_payloads_by_time(UPSTREAM, "", "b9415")) == []

    def test_the_fork_still_re_raises_a_retryable_http_failure(self, monkeypatch):
        monkeypatch.setattr(MOD, "github_release", self._http(503))
        monkeypatch.setattr(MOD, "web_release_payload", _boom)
        with pytest.raises(urllib.error.HTTPError):
            list(MOD.iter_release_payloads_by_time(MOD.DEFAULT_PUBLISHED_REPO, "", "b9415"))


class TestFreshnessAgreesWithSelection:
    """The update-currency check and the planner must answer from the same rule.

    They disagreed live before this was threaded: the check reported v0.4.1 newest
    while the planner installed b11071, so every update run would have reinstalled.
    """

    RELEASES = [
        {
            "tag_name": "v0.4.1",
            "prerelease": False,
            "draft": False,
            "published_at": "2026-09-14T00:00:00Z",
        },
        {
            "tag_name": "b11071",
            "prerelease": True,
            "draft": False,
            "published_at": "2026-09-21T00:00:00Z",
        },
        {
            "tag_name": "b11070",
            "prerelease": True,
            "draft": False,
            "published_at": "2026-09-20T00:00:00Z",
        },
    ]

    def test_upstream_freshness_names_the_build_the_planner_installs(self, monkeypatch):
        monkeypatch.setattr(MOD, "github_releases", lambda repo, **kw: self.RELEASES)
        assert MOD._api_newest_release_tag(UPSTREAM) == "b11071"

        monkeypatch.setattr(MOD, "web_release_payload", _boom)
        monkeypatch.setattr(MOD, "web_release_tags", _boom)
        planned = list(MOD.iter_release_payloads_by_time(UPSTREAM, "", "latest"))
        assert planned[0]["tag_name"] == "b11071"

    def test_the_fork_still_drops_prereleases_in_the_freshness_check(self, monkeypatch):
        monkeypatch.setattr(MOD, "github_releases", lambda repo, **kw: self.RELEASES)
        assert MOD._api_newest_release_tag(MOD.DEFAULT_PUBLISHED_REPO) == "v0.4.1"

    def test_a_draft_is_never_newest(self, monkeypatch):
        releases = self.RELEASES + [
            {
                "tag_name": "b11072",
                "prerelease": True,
                "draft": True,
                "published_at": "2026-09-22T00:00:00Z",
            }
        ]
        monkeypatch.setattr(MOD, "github_releases", lambda repo, **kw: releases)
        assert MOD._api_newest_release_tag(UPSTREAM) == "b11071"


class TestLatestTagNamesABuild:
    """The source build must compile the version the prebuilt path would install.

    /releases/latest resolves by make_latest, and upstream points it at v0.4.1, a
    pointer release that packages no prebuilt.
    """

    def test_a_non_build_rest_tag_consults_the_feed(self, monkeypatch):
        monkeypatch.setattr(MOD, "fetch_json", lambda url: {"tag_name": "v0.4.1"})
        _install_web(monkeypatch, {"releases.atom": _atom(UPSTREAM, ["v0.4.1", "b11071"])})
        assert MOD.latest_upstream_release_tag() == "b11071"

    def test_a_build_rest_tag_is_taken_without_consulting_the_feed(self, monkeypatch):
        monkeypatch.setattr(MOD, "fetch_json", lambda url: {"tag_name": "b11071"})
        monkeypatch.setattr(MOD, "web_release_tags", _boom)
        assert MOD.latest_upstream_release_tag() == "b11071"

    def test_the_rest_tag_is_kept_when_the_feed_cannot_answer(self, monkeypatch):
        # Better a released version than no source ref at all.
        monkeypatch.setattr(MOD, "fetch_json", lambda url: {"tag_name": "v0.4.1"})
        _install_web(monkeypatch, {})
        assert MOD.latest_upstream_release_tag() == "v0.4.1"

    def test_the_rest_tag_is_kept_when_the_feed_lists_no_build(self, monkeypatch):
        monkeypatch.setattr(MOD, "fetch_json", lambda url: {"tag_name": "v0.4.1"})
        _install_web(monkeypatch, {"releases.atom": _atom(UPSTREAM, ["v0.4.1", "v0.4.0"])})
        assert MOD.latest_upstream_release_tag() == "v0.4.1"
