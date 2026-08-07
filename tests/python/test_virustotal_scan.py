# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the advisory VirusTotal pre-flight scan.

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

import importlib.util
import pathlib
import sys

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

    def __call__(self, method, url, headers, body):
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
        assert "skipping the scan" in capsys.readouterr().out

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
        # Exactly one call: the lookup. No upload_url, no upload, no polling.
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
        client, _transport = _client({
            "/files/upload_url": (200, b'{"data": "' + signed.encode() + b'"}'),
            "upload.virustotal.example": (200, b'{"data": {"id": "an-1"}}'),
        })
        client.upload(bundle)
        out = capsys.readouterr().out
        assert f"::add-mask::{signed}" in out
        # Masking must happen before the URL is used, not after.
        assert out.index("::add-mask::") == 0

    def test_no_workflow_commands_off_the_runner(self, tmp_path, monkeypatch, capsys):
        monkeypatch.delenv("GITHUB_ACTIONS", raising = False)
        bundle = tmp_path / "big.exe"
        bundle.write_bytes(b"payload")
        client, _transport = _client({
            "/files/upload_url": (200, b'{"data": "https://up.example/x?sig=s"}'),
            "up.example": (200, b'{"data": {"id": "an-1"}}'),
        })
        client.upload(bundle)
        assert "::add-mask::" not in capsys.readouterr().out

    def test_empty_value_is_not_registered(self, monkeypatch, capsys):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        vt._mask_in_actions("")
        assert capsys.readouterr().out == ""


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
