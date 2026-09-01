# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the advisory VirusTotal release asset scan.

The scan is a sweep of the bundles `publish-release` uploaded, run in the
`virustotal-scan` job after it. Those bundles are attached to a draft on the
default dispatch and to a published release otherwise, which is why neither the
job nor the summary heading claims a publication. It is not a gate and cannot
hold a release back; Defender in the build job is the fail-closed check.

Offline by design: every test injects a fake transport, so the suite never spends
the account's 500/day quota and never uploads a build. The two behaviours worth
protecting are the ones a release depends on:

  - a missing API key must skip, never fail, or a contributor without the org
    secret cannot publish at all,
  - the bundles are 41-46 MB, over the 32 MB cap on `POST /files`, so the upload
    must go through `GET /files/upload_url`. A regression to the plain endpoint
    would fail on every asset.
"""

from __future__ import annotations

import fnmatch
import importlib.util
import itertools
import pathlib
import shlex
import sys
import time

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "virustotal_scan.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("virustotal_scan", MODULE_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"cannot import {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["virustotal_scan"] = module
    spec.loader.exec_module(module)
    return module


vt = _load_module()


class FakeTransport:
    """Records every call and replays a queued (status, body) per URL fragment."""

    def __init__(self, routes: dict[str, tuple[int, bytes]]):
        self.routes = routes
        self.calls: list[tuple[str, str, dict, int]] = []
        self.timeouts: list[float | None] = []

    def __call__(
        self,
        method,
        url,
        headers,
        body,
        timeout = None,
    ):
        self.timeouts.append(timeout)
        self.calls.append((method, url, headers, len(body or b"")))
        for fragment, response in self.routes.items():
            if fragment in url:
                return response
        raise AssertionError(f"unrouted request: {method} {url}")


def _client(routes):
    transport = FakeTransport(routes)
    client = vt.VirusTotalClient(
        "fake-key",
        transport = transport,
        request_interval = 0.0,
        sleep = lambda _seconds: None,
    )
    return client, transport


class TestParseStats:
    def test_missing_keys_default_to_zero(self):
        stats = vt.parse_stats({"malicious": 2})
        assert (stats.malicious, stats.suspicious, stats.undetected) == (2, 0, 0)

    def test_non_dict_is_tolerated(self):
        assert vt.parse_stats(None) == vt.ScanStats()
        assert vt.parse_stats([1, 2]) == vt.ScanStats()

    def test_confirmed_timeout_folds_into_timeout(self):
        assert vt.parse_stats({"timeout": 1, "confirmed-timeout": 2}).timeout == 3

    def test_booleans_are_not_counted_as_ints(self):
        # bool is a subclass of int; True must not silently become 1 detection.
        assert vt.parse_stats({"malicious": True}).malicious == 0

    def test_flagged_sums_malicious_and_suspicious(self):
        assert vt.parse_stats({"malicious": 3, "suspicious": 4}).flagged == 7


class TestParseDetections:
    def test_only_malicious_and_suspicious_are_reported(self):
        names = vt.parse_detections(
            {
                "AlphaAV": {"category": "malicious", "result": "Trojan.Gen"},
                "BetaAV": {"category": "undetected"},
                "GammaAV": {"category": "suspicious", "result": None},
                "DeltaAV": {"category": "harmless"},
            }
        )
        assert names == ["AlphaAV (Trojan.Gen)", "GammaAV"]

    def test_non_dict_is_tolerated(self):
        assert vt.parse_detections("nope") == []


class TestThreshold:
    def _reports(self, flagged):
        return [vt.FileReport(name = "a.exe", stats = vt.ScanStats(malicious = flagged))]

    def test_zero_threshold_is_advisory_only(self):
        # The shipped default. Detections must never fail the release.
        assert vt.exceeds_threshold(self._reports(50), 0) is False
        assert vt.exceeds_threshold(self._reports(50), -1) is False

    def test_positive_threshold_fails_at_or_above(self):
        assert vt.exceeds_threshold(self._reports(3), 3) is True
        assert vt.exceeds_threshold(self._reports(2), 3) is False

    def test_rows_without_stats_never_trip_the_gate(self):
        assert vt.exceeds_threshold([vt.FileReport(name = "a.exe")], 1) is False


class TestSelectScanTargets:
    def test_sig_sidecars_are_skipped(self, tmp_path):
        for name in (
            "Unsloth-Desktop-0_1_1-Windows.exe",
            "Unsloth-Desktop-0_1_1-Windows.exe.sig",
            "Unsloth-Desktop-0_1_1-Linux.AppImage",
            "Unsloth-Desktop-0_1_1-Linux.AppImage.sig",
        ):
            (tmp_path / name).write_bytes(b"x")
        names = [path.name for path in vt.collect_paths([tmp_path])]
        assert names == [
            "Unsloth-Desktop-0_1_1-Linux.AppImage",
            "Unsloth-Desktop-0_1_1-Windows.exe",
        ]

    def test_directories_are_expanded_and_files_passed_through(self, tmp_path):
        (tmp_path / "a.dmg").write_bytes(b"x")
        assert [p.name for p in vt.collect_paths([tmp_path / "a.dmg"])] == ["a.dmg"]


class TestMissingKey:
    def test_missing_key_skips_without_failing(self, tmp_path, monkeypatch, capsys):
        monkeypatch.delenv(vt.API_KEY_ENV, raising = False)
        (tmp_path / "a.exe").write_bytes(b"x")
        summary = tmp_path / "summary.md"
        rc = vt.main([str(tmp_path), "--output-markdown", str(summary)])
        assert rc == 0
        assert "Skipped: no API key" in summary.read_text()
        out = capsys.readouterr().out
        assert "skipping the scan" in out
        # The message spells VT_API_KEY out literally to avoid a CodeQL false positive, so pin that it still matches
        assert vt.API_KEY_ENV in out

    def test_whitespace_only_key_is_treated_as_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv(vt.API_KEY_ENV, "   ")
        (tmp_path / "a.exe").write_bytes(b"x")
        assert vt.main([str(tmp_path)]) == 0


class TestLargeFileUploadFlow:
    def test_upload_uses_the_signed_url_not_the_32mb_endpoint(self, tmp_path):
        bundle = tmp_path / "big.exe"
        bundle.write_bytes(b"payload")
        signed = "https://upload.virustotal.example/receive?sig=secret"
        client, transport = _client(
            {
                "/files/upload_url": (200, b'{"data": "' + signed.encode() + b'"}'),
                "upload.virustotal.example": (200, b'{"data": {"id": "analysis-1"}}'),
            }
        )

        assert client.upload(bundle) == "analysis-1"

        methods_urls = [(m, u) for m, u, _h, _n in transport.calls]
        assert methods_urls[0] == ("GET", f"{vt.API_ROOT}/files/upload_url")
        assert methods_urls[1][0] == "POST"
        assert methods_urls[1][1] == signed
        # The plain 32 MB-capped endpoint must never be used for a bundle.
        assert all(u.rstrip("/") != f"{vt.API_ROOT}/files" for _m, u in methods_urls)

    def test_upload_body_is_multipart_with_the_file_field(self, tmp_path):
        bundle = tmp_path / "big.exe"
        bundle.write_bytes(b"payload")
        body, content_type = vt._build_multipart(bundle)
        assert content_type.startswith("multipart/form-data; boundary=")
        assert b'name="file"' in body
        assert b'filename="big.exe"' in body
        assert b"payload" in body

    def test_api_key_is_sent_as_a_header_never_in_the_url(self, tmp_path):
        client, transport = _client({"/files/": (200, b"{}")})
        client.lookup_hash("a" * 64)
        _method, url, headers, _n = transport.calls[0]
        assert headers["x-apikey"] == "fake-key"
        assert "fake-key" not in url


class TestHashLookupFirst:
    def test_known_hash_short_circuits_the_upload(self, tmp_path):
        bundle = tmp_path / "known.exe"
        bundle.write_bytes(b"payload")
        client, transport = _client(
            {
                "/files/": (
                    200,
                    b'{"data": {"attributes": {"last_analysis_stats": '
                    b'{"malicious": 1}, "last_analysis_results": '
                    b'{"AlphaAV": {"category": "malicious", "result": "X"}}}}}',
                ),
            }
        )
        report = vt.scan_file(client, bundle, deadline = float("inf"))
        assert report.source == "known to VirusTotal (no upload)"
        assert report.stats.malicious == 1
        assert report.detections == ["AlphaAV (X)"]
        # Exactly one call: the lookup.
        # No upload_url, no upload, no polling.
        assert len(transport.calls) == 1

    def test_unknown_hash_falls_through_to_upload(self, tmp_path):
        bundle = tmp_path / "new.exe"
        bundle.write_bytes(b"payload")
        client, transport = _client(
            {
                "/files/upload_url": (200, b'{"data": "https://up.example/x"}'),
                "up.example": (200, b'{"data": {"id": "an-1"}}'),
                "/analyses/": (
                    200,
                    b'{"data": {"attributes": {"status": "completed", '
                    b'"stats": {"malicious": 0}, "results": {}}}}',
                ),
                "/files/": (404, b"{}"),
            }
        )
        report = vt.scan_file(client, bundle, deadline = float("inf"))
        assert report.source == "uploaded"
        assert report.stats.malicious == 0


class TestFailureDegradation:
    def test_transport_failure_degrades_to_a_note_not_an_exception(self, tmp_path):
        bundle = tmp_path / "a.exe"
        bundle.write_bytes(b"payload")
        client, _transport = _client({"/files/": (500, b"")})
        report = vt.scan_file(client, bundle, deadline = float("inf"))
        assert report.source == "unavailable"
        assert report.note
        assert report.stats is None

    def test_redact_url_strips_the_signed_query_string(self):
        assert vt._redact_url("https://up.example/x?sig=secret") == "https://up.example/x"


class TestSignedUrlMasking:
    """The signed upload URL is a credential and is NOT a registered GitHub secret,
    so the runner will not mask it unless we register it with ::add-mask::."""

    def test_upload_registers_the_signed_url_with_add_mask(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        bundle = tmp_path / "big.exe"
        bundle.write_bytes(b"payload")
        signed = "https://upload.virustotal.example/receive?sig=secret-credential"
        client, _transport = _client(
            {
                "/files/upload_url": (200, b'{"data": "' + signed.encode() + b'"}'),
                "upload.virustotal.example": (200, b'{"data": {"id": "an-1"}}'),
            }
        )
        client.upload(bundle)
        out = capsys.readouterr().out
        assert f"::add-mask::{signed}" in out
        # Masking must happen before the URL is used, not after.
        assert out.index("::add-mask::") == 0

    def test_no_workflow_commands_off_the_runner(self, tmp_path, monkeypatch, capsys):
        monkeypatch.delenv("GITHUB_ACTIONS", raising = False)
        bundle = tmp_path / "big.exe"
        bundle.write_bytes(b"payload")
        client, _transport = _client(
            {
                "/files/upload_url": (200, b'{"data": "https://up.example/x?sig=s"}'),
                "up.example": (200, b'{"data": {"id": "an-1"}}'),
            }
        )
        client.upload(bundle)
        assert "::add-mask::" not in capsys.readouterr().out

    def test_empty_value_is_not_registered(self, monkeypatch, capsys):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        vt._mask_in_actions("")
        assert capsys.readouterr().out == ""


class TestSingleUseUploadUrl:
    """A signed upload URL is single use, so replaying one can only ever be
    rejected. A failed upload must go back for a fresh URL instead."""

    def test_upload_post_is_not_retried_on_the_same_url(self, tmp_path):
        bundle = tmp_path / "big.exe"
        bundle.write_bytes(b"payload")
        seen_upload_urls = []

        class Transport:
            def __init__(self):
                self.posts = 0

            def __call__(
                self,
                method,
                url,
                headers,
                body,
                timeout = None,
            ):
                if url.endswith("/files/upload_url"):
                    token = f"https://up.example/{len(seen_upload_urls)}"
                    seen_upload_urls.append(token)
                    return 200, b'{"data": "' + token.encode() + b'"}'
                self.posts += 1
                if self.posts == 1:
                    return 500, b""
                return 200, b'{"data": {"id": "an-2"}}'

        transport = Transport()
        client = vt.VirusTotalClient(
            "k", transport = transport, request_interval = 0.0, sleep = lambda _s: None
        )
        assert client.upload(bundle) == "an-2"
        # Two distinct signed URLs were fetched: the failed POST was not replayed.
        assert len(seen_upload_urls) == 2
        assert transport.posts == 2

    def test_max_attempts_one_disables_retry(self, tmp_path):
        calls = []

        def transport(
            method,
            url,
            headers,
            body,
            timeout = None,
        ):
            calls.append(url)
            return 500, b""

        client = vt.VirusTotalClient(
            "k", transport = transport, request_interval = 0.0, sleep = lambda _s: None
        )
        with pytest.raises(RuntimeError):
            client.request("POST", "https://up.example/x", max_attempts = 1)
        assert len(calls) == 1


class TestDeadlineEnforcement:
    """One attempt can block for the full socket timeout, so the deadline has to be
    checked BEFORE a request, not after, or the step timeout kills the process
    before any summary is written."""

    def test_request_checks_deadline_before_issuing(self):
        calls = []

        def transport(
            method,
            url,
            headers,
            body,
            timeout = None,
        ):
            calls.append(url)
            return 200, b"{}"

        client = vt.VirusTotalClient(
            "k",
            transport = transport,
            request_interval = 0.0,
            sleep = lambda _s: None,
            clock = lambda: 1000.0,
        )
        with pytest.raises(TimeoutError):
            client.request("GET", "https://api.example/x", deadline = 999.0)
        assert calls == []

    def test_wait_for_analysis_stops_at_the_deadline(self):
        def transport(
            method,
            url,
            headers,
            body,
            timeout = None,
        ):
            return 200, b'{"data": {"attributes": {"status": "queued"}}}'

        now = [0.0]
        client = vt.VirusTotalClient(
            "k",
            transport = transport,
            request_interval = 0.0,
            sleep = lambda _s: None,
            clock = lambda: now[0],
        )
        with pytest.raises(TimeoutError):
            now[0] = 100.0
            client.wait_for_analysis("an-1", deadline = 50.0)

    def test_scan_file_reports_a_timeout_row_rather_than_raising(self, tmp_path):
        bundle = tmp_path / "a.exe"
        bundle.write_bytes(b"payload")
        client = vt.VirusTotalClient(
            "k",
            transport = lambda *a: (200, b"{}"),
            request_interval = 0.0,
            sleep = lambda _s: None,
            clock = lambda: 1000.0,
        )
        report = vt.scan_file(client, bundle, deadline = 0.0)
        assert report.source == "timed out"
        assert report.note

    def test_summary_is_still_written_when_every_asset_times_out(self, tmp_path, monkeypatch):
        monkeypatch.setenv(vt.API_KEY_ENV, "k")
        (tmp_path / "a.exe").write_bytes(b"x")
        summary = tmp_path / "s.md"
        rc = vt.main(
            [
                str(tmp_path),
                "--output-markdown",
                str(summary),
                "--timeout-seconds",
                "0",
                "--request-interval",
                "0",
            ]
        )
        assert rc == 0
        assert vt.SUMMARY_HEADING in summary.read_text()


class TestRenderMarkdown:
    def test_advisory_footer_when_threshold_disabled(self):
        text = vt.render_markdown(
            [vt.FileReport(name = "a.exe", stats = vt.ScanStats(), sha256 = "ab")], 0
        )
        assert "Advisory only" in text
        assert "never fail the release" in text

    def test_threshold_footer_when_enabled(self):
        text = vt.render_markdown([vt.FileReport(name = "a.exe", stats = vt.ScanStats())], 4)
        assert "Failure threshold: 4" in text

    def test_flagging_engines_are_listed(self):
        text = vt.render_markdown(
            [
                vt.FileReport(
                    name = "a.exe", stats = vt.ScanStats(malicious = 1), detections = ["AlphaAV (Trojan)"]
                )
            ],
            0,
        )
        assert "Flagging engines" in text
        assert "AlphaAV (Trojan)" in text

    def test_a_flagged_asset_gets_a_submission_packet(self):
        # The build job only ever assembled a packet for the Windows -setup.exe, so the one detection that actually
        # arrived
        # Trojan:Script/Wacatac.B!ml on the Linux AppImage
        text = vt.render_markdown(
            [
                vt.FileReport(
                    name = "Unsloth-Desktop-Linux.AppImage",
                    sha256 = "e3aa9b36",
                    size = 46193144,
                    stats = vt.ScanStats(malicious = 1, undetected = 62),
                    detections = ["Microsoft (Trojan:Script/Wacatac.B!ml)"],
                )
            ],
            0,
        )
        assert "False-positive submission packet" in text
        assert "Unsloth-Desktop-Linux.AppImage" in text
        assert "e3aa9b36" in text
        assert "46193144 bytes" in text
        assert "wdsi/filesubmission" in text

    def test_a_flagged_asset_with_no_readable_engine_list_still_gets_a_packet(self):
        # stats and results are separate fields of the same response.
        text = vt.render_markdown(
            [
                vt.FileReport(
                    name = "a.exe",
                    sha256 = "ab",
                    size = 10,
                    stats = vt.ScanStats(malicious = 1, undetected = 60),
                    detections = [],
                )
            ],
            0,
        )
        assert "False-positive submission packet" in text
        assert "Flagging engines" not in text

    def test_a_clean_run_gets_no_submission_packet(self):
        text = vt.render_markdown(
            [vt.FileReport(name = "a.exe", sha256 = "ab", stats = vt.ScanStats(undetected = 60))], 0
        )
        assert "False-positive submission packet" not in text


class TestFailClosedOnMalformedLookup:
    """A 200 whose body does not parse must not be read as 'never seen'.

    Returning None there is indistinguishable from a 404 and uploads the bundle,
    which is an unnecessary disclosure of an unreleased build.
    """

    def test_malformed_200_does_not_upload(self, tmp_path):
        bundle = tmp_path / "draft.exe"
        bundle.write_bytes(b"unreleased build")
        client, transport = _client(
            {
                "/files/upload_url": (200, b'{"data": "https://up.example/x"}'),
                "up.example": (200, b'{"data": {"id": "an-1"}}'),
                "/files/": (200, b"<html>proxy error page</html>"),
            }
        )
        report = vt.scan_file(client, bundle, deadline = float("inf"))
        assert report.source == "unavailable"
        assert "malformed" in report.note
        assert not any("up.example" in url for _m, url, _h, _n in transport.calls)

    def test_malformed_200_is_distinguishable_from_404(self, tmp_path):
        client, _ = _client({"/files/": (200, b"not json")})
        with pytest.raises(RuntimeError, match = "malformed"):
            client.lookup_hash("a" * 64)

        client, _ = _client({"/files/": (404, b"{}")})
        assert client.lookup_hash("a" * 64) is None


class TestDeadlineIsNotOverrunByThrottling:
    """Pacing sleeps between the deadline check and the network call."""

    def _clocked_client(
        self,
        routes,
        interval = 20.0,
    ):
        now = [1000.0]
        transport = FakeTransport(routes)

        def sleep(seconds):
            now[0] += seconds

        client = vt.VirusTotalClient(
            "k",
            transport = transport,
            request_interval = interval,
            sleep = sleep,
            clock = lambda: now[0],
        )
        return client, transport, now

    def test_transport_never_starts_after_the_deadline(self):
        client, transport, now = self._clocked_client({"x.example": (200, b"{}")})
        client._last_request_at = now[0]
        deadline = now[0] + 5.0
        with pytest.raises(TimeoutError, match = "pacing"):
            client.request("GET", "https://x.example/y", deadline = deadline)
        assert transport.calls == []

    def test_throttle_sleep_is_capped_by_the_deadline(self):
        client, _transport, now = self._clocked_client({"x.example": (200, b"{}")})
        client._last_request_at = now[0]
        deadline = now[0] + 5.0
        client._throttle(deadline)
        # Capped at the 5s of remaining budget, not the full 20s interval.
        assert now[0] == pytest.approx(1005.0)

    def test_a_request_with_budget_left_still_proceeds(self):
        client, transport, now = self._clocked_client({"x.example": (200, b"{}")})
        client._last_request_at = now[0]
        status, _payload = client.request("GET", "https://x.example/y", deadline = now[0] + 600.0)
        assert status == 200
        assert len(transport.calls) == 1


class TestSocketBudgetIsClampedToTheDeadline:
    """The per-call socket timeout has to respect the scan deadline.

    Otherwise a call that starts just before the deadline still blocks for the
    full socket timeout and eats the cushion the step needs to write its summary.
    """

    def test_socket_timeout_is_clamped_to_remaining_budget(self):
        client, transport = _client({"x.example": (200, b"{}")})
        client.request("GET", "https://x.example/y", deadline = time.monotonic() + 30.0)
        assert transport.timeouts[0] <= 30.0

    def test_socket_timeout_is_the_default_when_budget_is_large(self):
        client, transport = _client({"x.example": (200, b"{}")})
        client.request("GET", "https://x.example/y", deadline = time.monotonic() + 100000.0)
        assert transport.timeouts[0] == vt._SOCKET_TIMEOUT

    def test_socket_timeout_without_a_deadline_is_the_default(self):
        client, transport = _client({"x.example": (200, b"{}")})
        client.request("GET", "https://x.example/y")
        assert transport.timeouts[0] == vt._SOCKET_TIMEOUT

    def test_clamp_never_goes_to_zero_or_negative(self):
        # A non-positive urlopen timeout would fail instantly rather than try.
        client, transport = _client({"x.example": (200, b"{}")})
        client.request("GET", "https://x.example/y", deadline = time.monotonic() + 0.001)
        assert transport.timeouts[0] >= 1.0


class TestMalformedUploadAcknowledgement:
    """An accepted upload whose ack did not parse is a failed attempt.

    Raising straight out reports the asset unavailable after we already paid the
    disclosure cost of sending the bundle.
    """

    def test_malformed_ack_retries_with_a_fresh_signed_url(self, tmp_path):
        bundle = tmp_path / "big.exe"
        bundle.write_bytes(b"payload")
        state = {"n": 0}

        def transport(
            method,
            url,
            headers,
            body,
            timeout = None,
        ):
            if "/files/upload_url" in url:
                state["n"] += 1
                return (200, b'{"data": "https://up.example/%d"}' % state["n"])
            if state["n"] == 1:
                return (200, b"not json")
            return (200, b'{"data": {"id": "an-2"}}')

        client = vt.VirusTotalClient(
            "k", transport = transport, request_interval = 0.0, sleep = lambda _s: None
        )
        assert client.upload(bundle) == "an-2"
        assert state["n"] == 2

    def test_malformed_ack_on_the_last_attempt_raises(self, tmp_path):
        bundle = tmp_path / "big.exe"
        bundle.write_bytes(b"payload")

        def transport(
            method,
            url,
            headers,
            body,
            timeout = None,
        ):
            if "/files/upload_url" in url:
                return (200, b'{"data": "https://up.example/x"}')
            return (200, b"not json")

        client = vt.VirusTotalClient(
            "k", transport = transport, request_interval = 0.0, sleep = lambda _s: None
        )
        with pytest.raises(RuntimeError, match = "analysis id"):
            client.upload(bundle)


class TestNoCompletedAnalysis:
    """A known hash with no finished analysis must not read as clean. Reporting
    zero detections when no engine ran is the worst outcome available here."""

    def _report(
        self,
        attributes,
        completed = False,
    ):
        report = vt.FileReport(name = "a.exe")
        vt._record(report, "known to VirusTotal (no upload)", *attributes, completed = completed)
        return report

    def test_a_completed_analysis_is_trusted_without_engine_counts(self):
        # The upload path polls until status == "completed", so a stats dict is authoritative there even if the counts
        report = self._report(({"malicious": 0}, {}), completed = True)
        assert report.stats is not None
        assert report.source == "known to VirusTotal (no upload)"

    def test_a_completed_analysis_still_needs_a_stats_object(self):
        assert self._report((None, {}), completed = True).stats is None

    def test_missing_stats_is_not_reported_as_clean(self):
        report = self._report((None, None))
        assert report.stats is None
        assert report.source == "no completed analysis"
        assert "unscanned rather than clean" in report.note

    def test_all_zero_stats_is_not_reported_as_clean(self):
        # A stats dict where no engine reported anything means nothing ran.
        report = self._report(({"malicious": 0, "undetected": 0}, {}))
        assert report.stats is None
        assert report.source == "no completed analysis"

    def test_a_real_verdict_is_kept(self):
        report = self._report(
            (
                {"malicious": 0, "undetected": 70},
                {"AlphaAV": {"category": "undetected"}},
            )
        )
        assert report.stats is not None
        assert report.stats.undetected == 70
        assert report.source == "known to VirusTotal (no upload)"
        assert report.note == ""

    def test_an_unanalysed_row_never_trips_the_gate(self):
        # stats=None rows are ignored by the threshold, so this stays advisory.
        assert vt.exceeds_threshold([self._report((None, None))], 1) is False


class TestMarkdownEscaping:
    """The summary is a second sink for third-party text, appended to
    $GITHUB_STEP_SUMMARY and rendered as Markdown."""

    def _summary(self, report):
        return vt.render_markdown([report], 0)

    def test_a_newline_cannot_break_out_of_a_table_row(self):
        report = vt.FileReport(
            name = "a.exe",
            stats = vt.ScanStats(malicious = 1, undetected = 1),
            detections = ["Evil\n| fake | row |"],
        )
        body = self._summary(report)
        bullet = [line for line in body.splitlines() if "Evil" in line]
        # The newline is flattened, so the detection stays on its own bullet.
        assert len(bullet) == 1
        assert "\\|" in bullet[0]
        assert "| fake | row |" not in body

    def test_html_is_neutralised(self):
        report = vt.FileReport(name = "a.exe", note = "<img src=x onerror=alert(1)>")
        body = self._summary(report)
        assert "&lt;img" in body
        assert "<img" not in body

    def test_a_backtick_cannot_close_the_code_span(self):
        report = vt.FileReport(name = "a`.exe")
        assert "`a'.exe`" in self._summary(report)

    def test_clean_text_renders_unchanged(self):
        report = vt.FileReport(
            name = "a.exe",
            stats = vt.ScanStats(undetected = 70),
            detections = ["AlphaAV (Trojan.Gen)"],
        )
        assert "- `a.exe`: AlphaAV (Trojan.Gen)" in self._summary(report)


class TestAnnotationEscaping:
    """Engine names, detection labels and error strings are third-party data.
    Actions truncates an annotation at the first newline, which would drop the
    engine list exactly when the scan is trying to alert a maintainer."""

    def test_percent_is_escaped_before_the_newlines(self):
        # Order matters: escaping % last would double-encode %0A into %250A.
        assert vt._gha_escape("100%\nnext") == "100%25%0Anext"
        assert vt._gha_escape("a\r\nb") == "a%0D%0Ab"

    def test_clean_text_is_untouched(self):
        assert vt._gha_escape("AlphaAV (Trojan.Gen)") == "AlphaAV (Trojan.Gen)"

    def test_detection_annotation_stays_on_one_line(self, capsys):
        report = vt.FileReport(
            name = "a.exe",
            stats = vt.ScanStats(malicious = 1),
            detections = ["Evil\nAV (Tro%jan)"],
        )
        vt._emit(report)
        annotation = [
            line for line in capsys.readouterr().out.splitlines() if line.startswith("::warning")
        ]
        assert len(annotation) == 1
        assert "Evil%0AAV (Tro%25jan)" in annotation[0]

    def test_note_annotation_is_escaped(self, capsys):
        vt._emit(vt.FileReport(name = "a.exe", note = "HTTP 500\r\nbody: 50%"))
        annotation = [
            line for line in capsys.readouterr().out.splitlines() if line.startswith("::warning")
        ]
        assert len(annotation) == 1
        assert "HTTP 500%0D%0Abody: 50%25" in annotation[0]


class TestRetryBackoffRespectsTheDeadline:
    """Retry sleeps grow exponentially, so a late 429 could otherwise sleep well
    past --timeout-seconds before the loop notices and writes its summary."""

    def _client(self, status, now, slept, interval):
        def transport(
            method,
            url,
            headers,
            body,
            timeout = None,
        ):
            return status, b""

        # The retry backoff is seeded from the request interval.
        return vt.VirusTotalClient(
            "k",
            transport = transport,
            request_interval = interval,
            sleep = slept.append,
            clock = lambda: now[0],
        )

    @pytest.mark.parametrize("status", [429, 503])
    def test_backoff_never_sleeps_past_the_deadline(self, status):
        slept = []
        client = self._client(status, [0.0], slept, interval = 20.0)
        with pytest.raises((RuntimeError, TimeoutError)):
            client.request("GET", "https://api.example/x", deadline = 5.0)
        assert slept, "expected the retry path to sleep at all"
        assert max(slept) <= 5.0, slept

    def test_no_remaining_budget_means_no_sleep_at_all(self):
        slept = []
        client = self._client(429, [10.0], slept, interval = 20.0)
        # 5s of budget left, but an uncapped backoff would sleep 20s, then 40s.
        with pytest.raises((RuntimeError, TimeoutError)):
            client.request("GET", "https://api.example/x", deadline = 10.0)
        assert slept == []

    def test_backoff_is_unbounded_when_no_deadline_is_set(self):
        slept = []
        client = self._client(429, [0.0], slept, interval = 2.0)
        with pytest.raises(RuntimeError):
            client.request("GET", "https://api.example/x")
        # Full exponential backoff is preserved when there is no budget to respect.
        assert {2.0, 4.0, 8.0} <= set(slept), slept


class TestWorkflowOrdering:
    """The scan is a post-publish sweep, and the wiring that makes it run must hold.

    There is no pre-publish gate here and there never was one that could block a
    release: the scan is advisory by design (Defender in the build job is the
    fail-closed check for Windows), and since #8194 it runs in its own
    `virustotal-scan` job after `publish-release` rather than inline before the
    upload. The bundles are already public by the time it runs.

    What is still worth pinning is that the sweep cannot be quietly lost. A
    deleted job, a dropped `needs`, an `if:` that never fires, a missing script
    checkout or a `|| true` around the invocation would each leave the release
    scanned by nothing while the workflow stayed green. The tests below assert
    each of those against the workflow YAML.
    """

    def _workflow(self):
        yaml = pytest.importorskip("yaml")
        workflow = REPO_ROOT / ".github" / "workflows" / "release-desktop.yml"
        return yaml.safe_load(workflow.read_text(encoding = "utf-8"))

    def _publish_step_list(self):
        return self._workflow()["jobs"]["publish-release"]["steps"]

    def _publish_steps(self):
        return [step.get("name") for step in self._publish_step_list()]

    def _publish_step_map(self):
        return {step.get("name"): step for step in self._publish_step_list()}

    def _scan_job(self):
        jobs = self._workflow()["jobs"]
        assert "virustotal-scan" in jobs, (
            "the virustotal-scan job is gone; the release would ship unscanned by "
            "anything but Defender"
        )
        return jobs["virustotal-scan"]

    def _scan_step_map(self):
        return {step.get("name"): step for step in self._scan_job()["steps"]}

    def _scan_step_names(self):
        return [step.get("name") for step in self._scan_job()["steps"]]

    @staticmethod
    def _runner_temp(path):
        """Normalise the three spellings of the runner temp dir to one token.

        `with:` uses `${{ runner.temp }}` and `run:` uses `$RUNNER_TEMP`, so two
        paths can name one directory and still compare unequal.
        """
        normalised = " ".join(str(path).split())
        for spelling in ("${{ runner.temp }}", "${RUNNER_TEMP}", "$RUNNER_TEMP"):
            normalised = normalised.replace(spelling, "<RUNNER_TEMP>")
        return normalised.rstrip("/")

    def _scan_script_argv(self):
        """The argv the `VirusTotal scan` step hands to virustotal_scan.py."""
        run = self._scan_step_map()["VirusTotal scan"]["run"]
        command = run.replace("\\\n", " ")
        line = next(
            text for text in command.split("\n") if "python3 scripts/virustotal_scan.py" in text
        )
        argv = shlex.split(line)
        return argv[argv.index("scripts/virustotal_scan.py") + 1 :]

    def test_the_scan_is_its_own_job_gated_on_publish_release(self):
        job = self._scan_job()
        assert job["needs"] == ["publish-release"]

    def test_the_scan_job_is_not_conditioned_away(self):
        # `needs:` alone carries GitHub's default `success()` gating, so whether the scan runs is decided by
        # publish-release and nothing else.
        # A job-level `if:` fails in both directions: `always()` or `success() || inputs.scan_anyway` sends build
        job = self._scan_job()
        assert "if" not in job, (
            f"virustotal-scan carries `if: {job.get('if')}`; a job-level condition "
            "either runs the scan without a successful publish-release or skips it "
            "after one, and `needs:` already gates it correctly"
        )

        # Nor may the individual steps be skipped, except the summary, which is `if:
        for step in job["steps"]:
            condition = step.get("if")
            if step.get("name") == "Publish VirusTotal summary":
                assert condition == "always()"
            else:
                assert condition is None, step.get("name")

    def test_the_scan_scans_the_bundles_that_were_published(self):
        # The job has no build outputs of its own, so it re-downloads the very artifacts the build matrix uploaded and
        build = self._workflow()["jobs"]["build"]
        upload_names = {
            step.get("with", {}).get("name")
            for step in build["steps"]
            if step.get("uses", "").startswith("actions/upload-artifact@")
        }
        assert "desktop-release-${{ matrix.artifact }}" in upload_names

        download = self._scan_step_map()["Download published assets"]
        assert download["uses"].startswith("actions/download-artifact@")
        assert download["with"]["merge-multiple"] is True

        # Tie the scan's input to publish-release's own download rather than to a literal repeated in both places:
        publish_download = next(
            step
            for step in self._publish_step_list()
            if step.get("uses", "").startswith("actions/download-artifact@")
        )
        for key in ("pattern", "merge-multiple"):
            assert download["with"][key] == publish_download["with"][key], (
                key,
                download["with"].get(key),
                publish_download["with"].get(key),
            )
        assert self._runner_temp(download["with"]["path"]) == self._runner_temp(
            publish_download["with"]["path"]
        ), (download["with"]["path"], publish_download["with"]["path"])

        # And the scan has to be pointed at that same directory.
        argv = self._scan_script_argv()
        scan_paths = list(itertools.takewhile(lambda argument: not argument.startswith("-"), argv))
        assert scan_paths, argv
        assert [self._runner_temp(path) for path in scan_paths] == [
            self._runner_temp(download["with"]["path"])
        ], (argv, download["with"]["path"])

        # And that shared pattern has to match what the matrix actually uploads, or both jobs would agree on a set that
        template = "desktop-release-${{ matrix.artifact }}"
        for entry in build["strategy"]["matrix"]["include"]:
            artifact = template.replace("${{ matrix.artifact }}", entry["artifact"])
            assert fnmatch.fnmatch(artifact, download["with"]["pattern"]), (
                artifact,
                download["with"]["pattern"],
            )

        names = self._scan_step_names()
        assert names.index("Download published assets") < names.index("VirusTotal scan")

    def test_the_publish_job_no_longer_runs_the_scan(self):
        # #8194 moved the scan out wholesale.
        for step in self._publish_step_list():
            assert "virustotal" not in (step.get("name") or "").lower()
            assert "virustotal_scan.py" not in (step.get("run") or "")

    def test_nothing_slow_sits_between_validation_and_the_upload(self):
        # The v{version} release already exists and is published, so the window worth minimising is now between
        names = self._publish_steps()
        validate = names.index("Validate versioned release state")
        assert names[validate + 1] == "Generate versioned updater metadata"
        assert names[validate + 2] == "Publish release assets"

    def test_release_notes_are_written_unconditionally(self):
        # Validation writes the notes;
        steps = self._publish_step_map()
        assert "desktop-release-notes.md" in steps["Validate versioned release state"]["run"]
        metadata = steps["Generate versioned updater metadata"]["run"]
        assert "desktop-release-notes.md" in metadata

    def test_a_missing_release_stops_the_publish(self):
        # Nothing is created here any more, so an absent release is a dispatch mistake:
        run = self._publish_step_map()["Validate versioned release state"]["run"]
        assert "gh release create" not in run
        missing = run.split("does not exist.", 1)[1]
        assert "Tag main and publish it first" in missing
        assert "exit 1" in missing

    def test_every_public_mutation_is_gated_on_a_real_release(self):
        yaml = pytest.importorskip("yaml")
        workflow = REPO_ROOT / ".github" / "workflows" / "release-desktop.yml"
        data = yaml.safe_load(workflow.read_text(encoding = "utf-8"))
        steps = data["jobs"]["publish-release"]["steps"]
        by_name = {step.get("name"): step for step in steps}
        assert by_name["Validate versioned release state"]["id"] == "versioned_release_state"
        assert "Create versioned release" not in by_name

        # Validation runs on every dispatch;
        for name in ("Publish release assets", "Publish versioned updater metadata"):
            assert by_name[name]["if"] == "${{ !inputs.draft }}"

    def test_the_scan_step_does_not_swallow_its_own_failure(self):
        # cannot retroactively fail a release that already published; the step
        # must still surface its exit status, or a broken invocation reads as a
        # The advisory posture is a property of the job, not of the step.
        step = self._scan_step_map()["VirusTotal scan"]
        assert "continue-on-error" not in step

        run = step["run"]
        assert "set -euo pipefail" in run

        invocation = run.split("python3 scripts/virustotal_scan.py", 1)[1]
        for swallow in ("|| true", "|| :", "exit 0", "; true"):
            assert swallow not in invocation, swallow

        # pin the threshold to something the script treats as "never fail" while
        # The script signals a detection by returning 1 from main() once `--fail-threshold` is met (see TestThreshold)
        assert "--fail-threshold" not in run

    def test_the_advisory_escape_hatch_is_confined_to_the_scan_job(self):
        # `continue-on-error` anywhere else would let a genuine release failure pass as success.
        jobs = self._workflow()["jobs"]
        assert self._scan_job()["continue-on-error"] is True

        tolerant_jobs = [name for name, job in jobs.items() if "continue-on-error" in job]
        assert tolerant_jobs == ["virustotal-scan"]

        # every job that touches a bundle must not be.
        # free-capacity only asks CI to release runners and is allowed to fail;
        tolerant_steps = [
            (name, step.get("name"))
            for name, job in jobs.items()
            if name != "free-capacity"
            for step in job.get("steps", [])
            if "continue-on-error" in step
        ]
        assert tolerant_steps == []

    def test_the_scan_job_makes_the_scan_script_available(self):
        # The job publishes nothing and so has no source tree of its own;
        # the sparse checkout is the only thing that puts scripts/virustotal_scan.py on disk.
        checkouts = [
            step
            for step in self._scan_job()["steps"]
            if step.get("uses", "").startswith("actions/checkout@")
        ]
        assert len(checkouts) == 1
        checkout = checkouts[0]
        assert checkout["with"]["sparse-checkout"] == "scripts/virustotal_scan.py"
        assert checkout["with"]["persist-credentials"] is False

        names = self._scan_step_names()
        assert names.index(checkout["name"]) < names.index("VirusTotal scan")

        # And if it ever does not, the scan step says so loudly and exits 1 rather than reporting a clean sweep of
        guard = self._scan_step_map()["VirusTotal scan"]["run"]
        assert "if [ ! -f scripts/virustotal_scan.py ]; then" in guard
        assert "exit 1" in guard.split("if [ ! -f scripts/virustotal_scan.py ]; then", 1)[1]

    def test_the_scan_verdict_is_always_reported(self):
        summary = self._scan_step_map()["Publish VirusTotal summary"]
        assert summary["if"] == "always()"
        assert "$GITHUB_STEP_SUMMARY" in summary["run"]
        assert "virustotal-summary.md" in summary["run"]

    def test_the_placeholder_summary_matches_the_real_one(self):
        # continue-on-error means nobody is forced to look at the job result, so the step summary is the report.
        # The placeholder stands in when the scan produced no summary, so a heading that drifts from the script's
        summary = self._scan_step_map()["Publish VirusTotal summary"]
        assert vt.SUMMARY_HEADING in summary["run"], summary["run"]

    def test_the_summary_heading_holds_for_a_validation_only_run_too(self):
        # `inputs.draft` defaults to true, and every uploading step is gated on it, so the ordinary dispatch validates
        workflow = self._workflow()
        draft = workflow.get("on", workflow.get(True))["workflow_dispatch"]["inputs"]["draft"]
        assert draft["default"] is True

        upload = self._publish_step_map()["Publish release assets"]
        assert upload["if"] == "${{ !inputs.draft }}"

        heading = vt.SUMMARY_HEADING.lower()
        for claim in ("post-publish", "published", "pre-flight"):
            assert claim not in heading, (vt.SUMMARY_HEADING, claim)
