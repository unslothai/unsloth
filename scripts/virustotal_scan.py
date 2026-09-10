#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Advisory VirusTotal scan of the desktop release bundles.

Runs after `publish-release` has uploaded the assets, which for the default
`draft: true` dispatch means they are attached to a draft rather than published.

Stdlib only on purpose: this runs on a bare `ubuntu-latest` runner right after the
artifacts are downloaded, before any Python environment is provisioned, so it cannot
assume `requests` is importable.

The scan is advisory by default. VirusTotal aggregates roughly 70 engines and
Tauri/NSIS installers routinely trip one or two obscure heuristics, so a hard gate
here would be flaky. `--fail-threshold` exists to make it blocking later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence


API_ROOT = "https://www.virustotal.com/api/v3"
API_KEY_ENV = "VT_API_KEY"

# Signature sidecars are a few hundred bytes of base64 produced by the Tauri updater signer.
SKIPPED_SUFFIXES = (".sig",)

# The public API allows 4 requests/minute.
# 20s between requests keeps us just inside that even if the runner clock and VirusTotal's window disagree slightly.
DEFAULT_REQUEST_INTERVAL = 20.0
DEFAULT_TIMEOUT_SECONDS = 1500.0

# 0 disables the gate entirely; any positive N fails when malicious + suspicious >= N.
DEFAULT_FAIL_THRESHOLD = 0

_MAX_ATTEMPTS = 4
# A failed upload means fetching a fresh signed URL, so this is deliberately small: each retry re-sends 40+ MB and
# spends two more requests of quota.
_UPLOAD_ATTEMPTS = 2
_SOCKET_TIMEOUT = 300.0

# Transport contract: (method, url, headers, body, timeout) -> (status, response_bytes).
Transport = Callable[[str, str, "dict[str, str]", "bytes | None", float], "tuple[int, bytes]"]


@dataclass(frozen = True)
class ScanStats:
    """The subset of a VirusTotal stats dict we report on."""

    malicious: int = 0
    suspicious: int = 0
    undetected: int = 0
    harmless: int = 0
    timeout: int = 0

    @property
    def flagged(self) -> int:
        return self.malicious + self.suspicious

    @property
    def total(self) -> int:
        """Engine verdicts of any kind. Zero means nothing has actually run."""
        return self.malicious + self.suspicious + self.undetected + self.harmless + self.timeout


@dataclass
class FileReport:
    """One row of the job summary table."""

    name: str
    sha256: str = ""
    size: int = 0
    source: str = "skipped"
    stats: ScanStats | None = None
    detections: list[str] = field(default_factory = list)
    note: str = ""


def parse_stats(raw: object) -> ScanStats:
    """Coerce a VirusTotal stats dict into ScanStats.

    VirusTotal omits keys that are zero and has added categories over time
    (`confirmed-timeout`, `type-unsupported`, `failure`), so every field is read
    defensively rather than indexed.
    """
    if not isinstance(raw, dict):
        return ScanStats()

    def _count(key: str) -> int:
        value = raw.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return 0
        return int(value)

    return ScanStats(
        malicious = _count("malicious"),
        suspicious = _count("suspicious"),
        undetected = _count("undetected"),
        harmless = _count("harmless"),
        # `confirmed-timeout` is a separate bucket that means the same thing to us.
        timeout = _count("timeout") + _count("confirmed-timeout"),
    )


def parse_detections(raw: object) -> list[str]:
    """Return the sorted engine names whose verdict was malicious or suspicious.

    The engine list is what makes a warning actionable: `3 malicious` is noise until
    you can see whether it is three no-name heuristics or Microsoft plus Kaspersky.
    """
    if not isinstance(raw, dict):
        return []
    names: list[str] = []
    for engine, result in raw.items():
        if not isinstance(result, dict):
            continue
        if result.get("category") in ("malicious", "suspicious"):
            label = result.get("result")
            if isinstance(label, str) and label:
                names.append(f"{engine} ({label})")
            else:
                names.append(str(engine))
    return sorted(names)


def select_scan_targets(paths: Iterable[Path]) -> list[Path]:
    """Filter a directory listing down to the bundles worth spending quota on."""
    targets = [
        path for path in paths if path.is_file() and not path.name.endswith(SKIPPED_SUFFIXES)
    ]
    return sorted(targets, key = lambda path: path.name)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exceeds_threshold(reports: Sequence[FileReport], threshold: int) -> bool:
    """True when the run should fail. threshold <= 0 means advisory-only."""
    if threshold <= 0:
        return False
    return any(report.stats is not None and report.stats.flagged >= threshold for report in reports)


def _md_code(text: str) -> str:
    """Flatten a value that gets wrapped in a Markdown code span.

    A backslash is literal inside a code span, so a backtick cannot be escaped
    there; swap it for an apostrophe rather than let it close the span.
    """
    return " ".join(text.split()).replace("`", "'")


def _md_text(text: str) -> str:
    """Neutralise third-party text rendered as Markdown in the job summary.

    Engine names and detection labels are third-party data, and the summary is
    appended to `$GITHUB_STEP_SUMMARY`. A newline ends the table row or bullet,
    `|` opens a new cell, and `<` starts raw HTML, which GitHub renders, so an
    engine list could otherwise break out and obscure the report.
    """
    flattened = " ".join(text.split())
    escaped = flattened.replace("\\", "\\\\")
    for char in ("`", "|", "*", "_", "[", "]", "#"):
        escaped = escaped.replace(char, "\\" + char)
    return escaped.replace("<", "&lt;").replace(">", "&gt;")


# Also written by the workflow's placeholder step, so a reader sees one heading whether or not the scan produced a
# summary.
# Says neither "pre-flight" nor "post-publish": the scan runs after `publish-release`, but `inputs.draft` defaults to
# true, so the ordinary run has uploaded the assets to a draft rather than published them. Naming the assets is the
# only wording true of both dispatches.
SUMMARY_HEADING = "### VirusTotal release asset scan"


def submission_packet_lines(detected: Sequence[FileReport]) -> list[str]:
    """Build a false-positive submission packet for every flagged asset."""
    lines = [
        "",
        "#### False-positive submission packet",
        "",
        "Submit each flagged asset before announcing this release.",
        "",
        "- Microsoft: <https://www.microsoft.com/en-us/wdsi/filesubmission>, "
        "**Software developer** -> **Incorrectly detected as malware/malicious** "
        "(50 MB cap; use <https://security.microsoft.com/reportsubmission> for larger bundles).",
        "- Any other flagging vendor: use that vendor's own false-positive form. "
        "Microsoft clearance does not carry across engines.",
        "",
        "| Asset | SHA-256 | Size |",
        "| --- | --- | ---: |",
    ]
    for report in detected:
        lines.append(
            f"| `{_md_code(report.name)}` | `{_md_code(report.sha256 or 'n/a')}` | "
            f"{report.size} bytes |"
        )
    lines += [
        "",
        "> Clearance is per hash. It fixes the release you submit and nothing after it, so this",
        "> is a complement to signing, not a substitute.",
    ]
    return lines


def render_markdown(reports: Sequence[FileReport], threshold: int) -> str:
    """Render the job-summary table. Kept pure so it is unit testable."""
    lines = [
        SUMMARY_HEADING,
        "",
        "| Asset | Malicious | Suspicious | Undetected | Harmless | Timeout | Source |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for report in reports:
        stats = report.stats
        name = _md_code(report.name)
        source = _md_text(report.source)
        if stats is None:
            lines.append(f"| `{name}` | - | - | - | - | - | {source} |")
        else:
            lines.append(
                f"| `{name}` | {stats.malicious} | {stats.suspicious} | "
                f"{stats.undetected} | {stats.harmless} | {stats.timeout} | {source} |"
            )

    detected = [report for report in reports if report.detections]
    if detected:
        lines += ["", "#### Flagging engines", ""]
        for report in detected:
            engines = ", ".join(_md_text(detection) for detection in report.detections)
            lines.append(f"- `{_md_code(report.name)}`: {engines}")

    notes = [report for report in reports if report.note]
    if notes:
        lines += ["", "#### Notes", ""]
        for report in notes:
            lines.append(f"- `{_md_code(report.name)}`: {_md_text(report.note)}")

    # Use the flagged count because the results map may be absent.
    flagged = [report for report in reports if report.stats is not None and report.stats.flagged]
    if flagged:
        lines += submission_packet_lines(flagged)

    lines += ["", "<details><summary>SHA-256</summary>", ""]
    for report in reports:
        lines.append(f"- `{_md_code(report.name)}`: `{_md_code(report.sha256 or 'n/a')}`")
    lines += ["</details>", ""]

    if threshold > 0:
        lines.append(f"Failure threshold: {threshold} malicious + suspicious detections.")
    else:
        lines.append(
            "Advisory only: detections are reported as warnings and never fail the release. "
            "Windows Defender remains the authoritative gate."
        )
    return "\n".join(lines) + "\n"


def _default_transport(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
    timeout: float = _SOCKET_TIMEOUT,
) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data = body, headers = headers, method = method)
    context = ssl.create_default_context()
    try:
        with urllib.request.urlopen(request, timeout = timeout, context = context) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        # 404 on a hash lookup is an expected control-flow signal, not a failure.
        return error.code, error.read()


class VirusTotalClient:
    """Thin, rate-limited VirusTotal v3 client."""

    def __init__(
        self,
        api_key: str,
        transport: Transport | None = None,
        request_interval: float = DEFAULT_REQUEST_INTERVAL,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._api_key = api_key
        self._transport = transport or _default_transport
        self._request_interval = max(0.0, request_interval)
        self._sleep = sleep
        self._clock = clock
        self._last_request_at: float | None = None

    def _throttle(self, deadline: float | None = None) -> None:
        """Pace requests, without ever sleeping past the caller's budget.

        Capping matters because the sleep sits between the caller's deadline check
        and the network call: an uncapped sleep can carry execution past the
        deadline and start a request that then blocks for the full socket timeout.
        The caller re-checks the deadline once this returns.
        """
        if self._last_request_at is None:
            return
        elapsed = self._clock() - self._last_request_at
        remaining = self._request_interval - elapsed
        if remaining <= 0:
            return
        if deadline is not None:
            remaining = min(remaining, max(0.0, deadline - self._clock()))
        if remaining > 0:
            self._sleep(remaining)

    def _backoff(
        self,
        seconds: float,
        deadline: float | None = None,
    ) -> None:
        """Sleep between retry attempts, clamped to the remaining budget.

        The retry sleep grows exponentially, so a 429 arriving shortly before the
        deadline could otherwise sleep well past `--timeout-seconds` before the
        next iteration notices, delaying the summary the release step depends on.
        """
        if deadline is None:
            self._sleep(seconds)
            return
        remaining = min(seconds, deadline - self._clock())
        if remaining > 0:
            self._sleep(remaining)

    def request(
        self,
        method: str,
        url: str,
        body: bytes | None = None,
        extra_headers: dict[str, str] | None = None,
        allow_status: Sequence[int] = (),
        max_attempts: int = _MAX_ATTEMPTS,
        deadline: float | None = None,
    ) -> tuple[int, object]:
        """Issue one API call, retrying 429s and transient network errors.

        Returns (status, decoded_json_or_None). Raises RuntimeError once the retry
        budget is spent so the caller can degrade to a warning row.

        `max_attempts = 1` disables retrying, which is mandatory for a POST to a
        single-use signed upload URL: replaying it can only ever be rejected.

        `deadline` is checked BEFORE each attempt. One attempt can block for the
        full socket timeout, so a loop that only checks afterwards can overrun the
        caller's budget by minutes and get the whole step killed before it writes
        a summary.
        """
        headers = {"x-apikey": self._api_key, "accept": "application/json"}
        if extra_headers:
            headers.update(extra_headers)

        backoff = self._request_interval if self._request_interval > 0 else 1.0
        last_error = ""
        for attempt in range(1, max_attempts + 1):
            if deadline is not None and self._clock() >= deadline:
                raise TimeoutError(f"deadline reached before {method} {_redact_url(url)}")
            self._throttle(deadline)
            # Re-check: pacing sleeps between the check above and the call below, so without this a request could start
            # after the deadline and then block for the full socket timeout, overrunning the step's own budget.
            if deadline is not None and self._clock() >= deadline:
                raise TimeoutError(
                    f"deadline reached while pacing before {method} {_redact_url(url)}"
                )
            try:
                # Clamp the socket budget to what is left.
                socket_timeout = _SOCKET_TIMEOUT
                if deadline is not None:
                    socket_timeout = max(1.0, min(socket_timeout, deadline - self._clock()))
                status, payload = self._transport(method, url, headers, body, socket_timeout)
            except Exception as error:
                last_error = f"{type(error).__name__}: {error}"
                status, payload = 0, b""
            finally:
                self._last_request_at = self._clock()

            if status == 429:
                # Quota or minute-rate exhaustion.
                last_error = "429 rate limited"
                if attempt < max_attempts:
                    self._backoff(backoff * (2 ** (attempt - 1)), deadline)
                continue
            if status == 0 or status >= 500:
                last_error = last_error or f"HTTP {status}"
                if attempt < max_attempts:
                    self._backoff(backoff * (2 ** (attempt - 1)), deadline)
                continue
            if status >= 400 and status not in allow_status:
                raise RuntimeError(f"VirusTotal returned HTTP {status} for {_redact_url(url)}")

            if not payload:
                return status, None
            try:
                return status, json.loads(payload.decode("utf-8", "replace"))
            except json.JSONDecodeError:
                return status, None

        raise RuntimeError(
            f"VirusTotal request failed after {max_attempts} attempt(s): {last_error}"
        )

    def lookup_hash(
        self,
        sha256: str,
        deadline: float | None = None,
    ) -> object | None:
        """Return the existing file report, or None when VirusTotal has never seen it.

        Doing this first is both a quota saving and a disclosure saving: a bundle that
        VirusTotal already holds gains nothing from being uploaded again.
        """
        status, payload = self.request(
            "GET",
            f"{API_ROOT}/files/{sha256}",
            allow_status = (404,),
            deadline = deadline,
        )
        if status == 404:
            return None
        if not isinstance(payload, dict):
            # A 200 whose body did not parse (a proxy error page, a truncated read) proves nothing about whether
            # VirusTotal holds this file. Returning None would be indistinguishable from a 404 and would upload the
            # bundle, disclosing an unreleased build. Fail closed instead.
            raise RuntimeError("VirusTotal hash lookup returned a malformed body")
        return payload

    def upload(
        self,
        path: Path,
        deadline: float | None = None,
    ) -> str:
        """Upload via the large-file flow and return the analysis id.

        Every desktop bundle is 41-46 MB, which is over the 32 MB cap on
        `POST /files`, so the signed upload URL is the only path that works here.

        Each signed URL is SINGLE USE, so the POST is issued with retries disabled.
        Replaying one after, say, the response body failed to read would be rejected
        no matter how many times we tried, and would report the asset as unavailable
        while an analysis was in fact already running. Retrying instead means going
        back for a fresh URL, which is what the loop below does.
        """
        attempts = max(1, _UPLOAD_ATTEMPTS)
        last_error: Exception | None = None
        body, content_type = _build_multipart(path)

        for attempt in range(1, attempts + 1):
            _, payload = self.request("GET", f"{API_ROOT}/files/upload_url", deadline = deadline)
            upload_url = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(upload_url, str) or not upload_url:
                raise RuntimeError("VirusTotal did not return an upload URL")
            # Mask before the URL is ever used, so anything that later echoes it -- a traceback, a future debug
            # print, a library error string -- is scrubbed.
            _mask_in_actions(upload_url)

            try:
                _, payload = self.request(
                    "POST",
                    upload_url,
                    body = body,
                    extra_headers = {"content-type": content_type},
                    max_attempts = 1,
                    deadline = deadline,
                )
            except TimeoutError:
                raise
            except Exception as error:
                last_error = error
                if attempt < attempts:
                    continue
                raise

            analysis_id = None
            if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
                analysis_id = payload["data"].get("id")
            if not isinstance(analysis_id, str) or not analysis_id:
                # An accepted upload whose acknowledgement did not parse is a failed attempt, not a dead end:
                # VirusTotal may well be analysing the file already. Raising straight out would report the asset
                # unavailable after we had paid the disclosure cost of sending it, so spend the remaining attempt
                # on a fresh signed URL instead.
                last_error = RuntimeError("VirusTotal upload did not return an analysis id")
                if attempt < attempts:
                    continue
                raise last_error
            return analysis_id

        raise RuntimeError(f"VirusTotal upload failed after {attempts} attempt(s): {last_error}")

    def wait_for_analysis(self, analysis_id: str, deadline: float) -> object:
        """Poll until the analysis completes or the caller's deadline passes."""
        while True:
            # Checked inside request() too, but raising the analysis-specific message here keeps the summary row
            # readable.
            if self._clock() >= deadline:
                raise TimeoutError(f"analysis {analysis_id} did not complete before the deadline")
            _, payload = self.request(
                "GET", f"{API_ROOT}/analyses/{analysis_id}", deadline = deadline
            )
            attributes = _attributes(payload)
            if attributes.get("status") == "completed":
                return payload
            if self._request_interval <= 0:
                # With throttling disabled (premium key) the loop would otherwise spin.
                self._sleep(1.0)


def _build_multipart(path: Path) -> tuple[bytes, str]:
    """Encode `path` as a single-part multipart/form-data body under the field `file`."""
    boundary = f"----UnslothDesktopScan{uuid.uuid4().hex}"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return head + path.read_bytes() + tail, f"multipart/form-data; boundary={boundary}"


def _redact_url(url: str) -> str:
    """Strip the query string before logging: signed upload URLs carry credentials."""
    return url.split("?", 1)[0]


def _mask_in_actions(value: str) -> None:
    """Register `value` with the runner's log scrubber.

    `VT_API_KEY` comes from a repository secret, so Actions masks it everywhere
    automatically. The signed upload URL does not: VirusTotal mints it per call and
    its query string is a credential, which makes it exactly the "sensitive
    information that is not a GitHub secret" the Actions docs say to pass through
    `::add-mask::`. Without this, the only thing keeping it out of the log is us
    remembering to call `_redact_url` at every print site, which is a rule that
    holds right up until someone adds a print or a traceback escapes.

    No-ops off the runner so local runs are not littered with workflow commands.
    """
    if value and os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::add-mask::{value}", flush = True)


def _attributes(payload: object) -> dict:
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        attributes = payload["data"].get("attributes")
        if isinstance(attributes, dict):
            return attributes
    return {}


def _record(
    report: FileReport, source: str, raw_stats: object, raw_results: object, *, completed: bool
) -> None:
    """Attach a verdict, but only when an analysis has actually completed.

    A hash can be known to VirusTotal with no finished analysis, in which case
    `last_analysis_stats` is missing and `parse_stats` yields an all-zero
    ScanStats. Reporting that as a clean row is the worst failure mode this
    script has: it looks like 70 engines cleared the bundle when none ran. Leave
    `stats` as None instead, which renders as dashes and never trips the gate.

    `completed` says whether the caller already has an authoritative completion
    signal. The upload path polls until `status == "completed"`, so a stats dict
    is enough there. The hash lookup carries no such field, so it additionally
    has to see at least one engine verdict before believing the result.
    """
    stats = parse_stats(raw_stats)
    if not isinstance(raw_stats, dict) or (not completed and stats.total == 0):
        report.source = "no completed analysis"
        report.note = (
            "VirusTotal returned no completed analysis for this hash, "
            "so the bundle is unscanned rather than clean"
        )
        return
    report.source = source
    report.stats = stats
    report.detections = parse_detections(raw_results)


def scan_file(client: VirusTotalClient, path: Path, deadline: float) -> FileReport:
    """Scan one bundle, degrading to an annotated row rather than raising."""
    report = FileReport(name = path.name, size = path.stat().st_size)
    report.sha256 = sha256_of(path)

    try:
        existing = client.lookup_hash(report.sha256, deadline = deadline)
        if existing is not None:
            attributes = _attributes(existing)
            _record(
                report,
                "known to VirusTotal (no upload)",
                attributes.get("last_analysis_stats"),
                attributes.get("last_analysis_results"),
                completed = False,
            )
            return report

        analysis_id = client.upload(path, deadline = deadline)
        # wait_for_analysis only returns once status == "completed".
        attributes = _attributes(client.wait_for_analysis(analysis_id, deadline))
        _record(
            report,
            "uploaded",
            attributes.get("stats"),
            attributes.get("results"),
            completed = True,
        )
    except TimeoutError as error:
        report.source = "timed out"
        report.note = str(error)
    except Exception as error:
        report.source = "unavailable"
        report.note = f"{type(error).__name__}: {error}"
    return report


def _gha_escape(text: str) -> str:
    """Escape a string for a GH Actions `::warning::` message.

    Engine names, detection labels and error strings are third-party data, so
    they can contain anything. GH Actions truncates an annotation at the first
    newline unless `\\n`/`\\r` are escaped as `%0A`/`%0D`. `%` must be replaced
    first to avoid double-encoding the subsequent escapes.

    Mirrors `_gha_escape` in scripts/lockfile_supply_chain_audit.py.
    """
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _emit(report: FileReport) -> None:
    """One deterministic, greppable line per asset."""
    stats = report.stats or ScanStats()
    print(
        f"virustotal_scan: asset={report.name} sha256={report.sha256 or 'n/a'} "
        f"bytes={report.size} source={report.source!r} malicious={stats.malicious} "
        f"suspicious={stats.suspicious} undetected={stats.undetected} "
        f"harmless={stats.harmless} timeout={stats.timeout}",
        flush = True,
    )
    if report.detections:
        # ::warning:: and not ::error:: so the release still ships; see the module docstring.
        print(
            f"::warning title=VirusTotal detection::{_gha_escape(report.name)}: "
            f"{stats.malicious} malicious, {stats.suspicious} suspicious "
            f"({_gha_escape(', '.join(report.detections))})",
            flush = True,
        )
    if report.note:
        print(
            f"::warning title=VirusTotal scan incomplete::"
            f"{_gha_escape(report.name)}: {_gha_escape(report.note)}",
            flush = True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "paths",
        nargs = "+",
        type = Path,
        help = "release bundles to scan, or a directory containing them",
    )
    parser.add_argument(
        "--output-markdown",
        type = Path,
        default = None,
        help = "write the summary table here (for $GITHUB_STEP_SUMMARY)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type = float,
        default = DEFAULT_TIMEOUT_SECONDS,
        help = "overall wall-clock cap for the whole scan",
    )
    parser.add_argument(
        "--request-interval",
        type = float,
        default = DEFAULT_REQUEST_INTERVAL,
        help = "minimum seconds between API calls (free tier allows 4/min)",
    )
    parser.add_argument(
        "--fail-threshold",
        type = int,
        default = DEFAULT_FAIL_THRESHOLD,
        help = "exit non-zero when malicious + suspicious >= N (0 disables)",
    )
    return parser


def collect_paths(paths: Sequence[Path]) -> list[Path]:
    candidates: list[Path] = []
    for path in paths:
        if path.is_dir():
            candidates.extend(sorted(path.iterdir()))
        else:
            candidates.append(path)
    return select_scan_targets(candidates)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # The key never comes from argv: CLI arguments are world-readable in /proc.
    api_key = os.environ.get(API_KEY_ENV, "").strip()

    def _write_markdown(text: str) -> None:
        if args.output_markdown is not None:
            args.output_markdown.parent.mkdir(parents = True, exist_ok = True)
            args.output_markdown.write_text(text, encoding = "utf-8")

    if not api_key:
        # A missing secret must never break a release: forks and re-runs by contributors without the org secret still
        # have to be able to publish.
        # The env var NAME is written out literally rather than interpolated from API_KEY_ENV.
        # test_missing_key_skips_without_failing asserts the two stay in step.
        print(
            "virustotal_scan: VT_API_KEY is unset or empty; skipping the scan.",
            flush = True,
        )
        _write_markdown(f"{SUMMARY_HEADING}\n\nSkipped: no API key configured for this run.\n")
        return 0

    targets = collect_paths(args.paths)
    if not targets:
        print("virustotal_scan: no scannable bundles found.", flush = True)
        _write_markdown(f"{SUMMARY_HEADING}\n\nSkipped: no scannable bundles found.\n")
        return 0

    client = VirusTotalClient(api_key, request_interval = args.request_interval)
    deadline = time.monotonic() + max(0.0, args.timeout_seconds)

    reports: list[FileReport] = []
    for path in targets:
        if time.monotonic() >= deadline:
            report = FileReport(name = path.name, size = path.stat().st_size)
            report.source = "skipped"
            report.note = "overall scan timeout reached before this asset was submitted"
            reports.append(report)
            _emit(report)
            continue
        report = scan_file(client, path, deadline)
        reports.append(report)
        _emit(report)

    _write_markdown(render_markdown(reports, args.fail_threshold))

    if exceeds_threshold(reports, args.fail_threshold):
        print(
            f"virustotal_scan: detections reached the --fail-threshold of {args.fail_threshold}.",
            file = sys.stderr,
            flush = True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
