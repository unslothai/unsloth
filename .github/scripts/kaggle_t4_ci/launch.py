# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Push the kernels to Kaggle, wait for them, and bring the evidence back.

One invocation handles every kernel of the run. They are pushed FIRST, all of
them, and only then waited on: pushing one, waiting for it and pushing the
next would serialise two sessions that Kaggle is happy to run at once and
double the wall clock of the job for nothing. It also matters for the
control/canary pair, whose whole value is that they ran at the same time on
the same account.

A kernel that could not be pushed does not stop the others. The verdict is
computed from the reports that came back against the number expected, so a
half-launched run reports as ``partial`` -- which is a warning, not a failure,
because half a comparison is not evidence of a regression.


Failure semantics are the whole point of this file, so they are stated up
front. It distinguishes two kinds of bad outcome and exits differently for
each, because conflating them is how a flaky external service ends up
blocking merges:

* ``exit 0`` with ``verdict=infra`` -- the test never got to run, or its
  result never got back. Push throttled, account at its concurrency cap, a
  kernel that died on Kaggle's side, a download that would not complete, our
  own wall-clock ceiling. Nothing was learned about the code under test, so
  nothing should turn red.
* ``exit 0`` with ``verdict=pass`` / ``verdict=fail`` -- the payload ran and
  reached a conclusion. Judging that conclusion is ``report.py``'s job, not
  this one's; this file only transports it.

The only nonzero exit is a usage error.

Wall-clock is bounded three times over, and the order of trust here is the
opposite of what it looks like:

* **Deleting the kernel.** This is the control that has been observed to
  work. Every kernel this process pushed is deleted on the way out, on every
  path including the failure ones, and deletion stops the billing (measured:
  the account's used-hours figure went DOWN when a wedged kernel was
  deleted).
* **Our polling deadline** (``--max-wait``), which decides when to give up
  and therefore when to delete.
* **Kaggle's own kernel timeout**, passed at push time. It is a backstop and
  it is NOT sufficient on its own: on 2026-08-11 a kernel pushed with
  ``-t 5400`` whose own nbconvert crashed at t=406s sat in RUNNING for over
  two hours, past that ceiling and past this process's deadline, and stopped
  only when it was deleted by hand. So the value is still passed, and
  nothing is left resting on it.

A socket timeout is set globally for the same reason. Without one, a single
status call that never returns stalls the poll loop past every deadline
above -- which is exactly how that two-hour kernel went unnoticed.

No credential is printed. The token is read from the environment by the
Kaggle client and never echoed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path, PurePosixPath, PureWindowsPath

API_ROOT = "https://www.kaggle.com/api/v1"
RESULT_PREFIX = "T4_SMOKE_REPORT "
OUTPUT_SUFFIX = "_output.ipynb"

TERMINAL_OK = {"COMPLETE"}
TERMINAL_BAD = {"ERROR", "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED"}

_STATUS_RE = re.compile(r"KernelWorkerStatus\.(?P<status>[A-Z_]+)")

PUSH_ATTEMPTS = 4
PUSH_BACKOFF_SEC = 45

# What a throttled or briefly unavailable push looks like coming back. The
# JSON-decode message is the common face: Kaggle answers 429 and 503 with an
# HTML error page, the client decodes it as JSON regardless, and the
# throttling is never named. Everything else -- a bad slug, a rejected
# accelerator, missing credentials -- is deterministic and must not be
# retried.
THROTTLED_PUSH = (
    "expecting value: line 1 column 1",
    "429",
    "too many requests",
    "502",
    "503",
    "service unavailable",
    "bad gateway",
    "timed out",
    "connection reset",
    "connection aborted",
)

# Kaggle's concurrency cap, reported as a push rejection rather than a queue.
CAPACITY_MARKERS = (
    "maximum batch gpu session count",
    "session count of 2 reached",
    "toomanyassignments",
    "precondition failed",
    "412",
    "no accelerator quota",
    "no quota for",
)

# Ceiling on any single network call. urllib takes an explicit timeout and
# the Kaggle client does not, so this is the only bound available on its
# status and quota calls. Without it one call that never returns outlasts
# every deadline in this file, and the kernel it was watching keeps billing.
SOCKET_TIMEOUT_SEC = 120

# Consecutive unreadable statuses before we stop waiting. One is not enough:
# the API returns transient 5xx and the client prints them the same way as a
# permanent refusal, and giving up on a blip abandons a kernel that is still
# doing the work.
MAX_CONSECUTIVE_UNKNOWN = 10


def _log(msg: str) -> None:
    print(f"[launch] {msg}", flush = True)


def _out(key: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding = "utf-8") as fh:
            if "\n" in value:
                delim = f"ghadelim{uuid.uuid4().hex}"
                fh.write(f"{key}<<{delim}\n{value}\n{delim}\n")
            else:
                fh.write(f"{key}={value}\n")


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return re.sub(r"-{2,}", "-", s)[:50].strip("-")


def _api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _pushed(ok: bool, reason: str, out: str, attempted: list[str]) -> dict:
    """A push outcome, always carrying the slugs the call filed.

    A failed attempt may still have landed -- Kaggle answers a committed
    push with a 5xx often enough for it to be a known issue -- so the slug
    is reported rather than forgotten, and the caller can reconcile it.
    """
    return {"ok": ok, "reason": reason, "detail": out.strip()[:400], "attempts": attempted}


def push(
    notebook: Path,
    user: str,
    kernel_timeout_sec: int,
    accelerator: str = "NvidiaTeslaT4",
) -> dict:
    """Push as a fresh private kernel. Every attempt gets its own slug.

    A fresh slug per attempt is not cosmetic, and the reason is Kaggle's own
    versioning. Pushing to an id that ALREADY exists does not replace or
    supersede what is there: it files a new version and starts a SECOND
    batch session, and the running one keeps running. Meanwhile
    ``kernels status`` and ``kernels/output`` send no version label, so they
    answer for the newest session only. Reuse a slug across a retry and the
    evidence collected belongs to whichever execution happened to be latest,
    while the other consumes a session slot and its quota unseen.

    That matters here because the retried failures are exactly the ambiguous
    ones -- a reset connection, an aborted transfer, a 5xx -- where Kaggle
    may well have accepted the push whose response never arrived. So each
    attempt also DELETES the previous attempt's slug before pushing again:
    deletion is kernel-level, it costs one call, and it is what frees the
    session slot the retry is likely waiting on.

    Returns the slug of the accepted attempt, plus ``attempts``: every slug
    this call filed, newest last.
    """
    base = _slugify("unsloth t4 ci")[:32]
    attempted: list[str] = []

    def _discard(slug: str) -> None:
        """Best effort. The attempt usually created nothing at all."""
        try:
            subprocess.run(
                ["kaggle", "kernels", "delete", slug, "-y"],
                capture_output = True,
                text = True,
                timeout = 180,
            )
        except Exception:  # noqa: BLE001
            _log(f"could not discard the previous push attempt {slug}")

    workdir = Path(tempfile.mkdtemp(prefix = "kaggle-t4-ci-"))
    try:
        out = ""
        for attempt in range(PUSH_ATTEMPTS):
            if attempted:
                _discard(attempted[-1])
            slug_name = f"{base}-{uuid.uuid4().hex[:8]}"
            # The slug is derived from the TITLE, not from the metadata id. A
            # mismatch files the kernel at an unexpected address and every later
            # status/output call 403s, so assert the round trip.
            title = slug_name.replace("-", " ")
            assert _slugify(title) == slug_name, f"title {title!r} slugifies to {_slugify(title)!r}"
            attempted.append(f"{user}/{slug_name}")

            for stale in workdir.glob("*.ipynb"):
                stale.unlink()
            code_file = workdir / f"{slug_name}.ipynb"
            shutil.copy(notebook, code_file)
            (workdir / "kernel-metadata.json").write_text(
                json.dumps(
                    {
                        "id": f"{user}/{slug_name}",
                        "title": title,
                        "code_file": code_file.name,
                        "language": "python",
                        "kernel_type": "notebook",
                        "is_private": "true",
                        "enable_gpu": "true",
                        "enable_internet": "true",
                        "machine_shape": accelerator,
                        "dataset_sources": [],
                        "competition_sources": [],
                        "kernel_sources": [],
                        "model_sources": [],
                    },
                    indent = 2,
                ),
                encoding = "utf-8",
            )

            proc = subprocess.run(
                [
                    "kaggle",
                    "kernels",
                    "push",
                    "-p",
                    str(workdir),
                    "--accelerator",
                    accelerator,
                    "-t",
                    str(kernel_timeout_sec),
                ],
                capture_output = True,
                text = True,
                timeout = 600,
            )
            out = proc.stdout + proc.stderr
            lowered = out.lower()
            if "successfully pushed" in lowered:
                if "does not resolve to the specified id" in lowered:
                    return _pushed(False, "slug_mismatch", out, attempted)
                return {"ok": True, "slug": attempted[-1], "attempts": attempted}
            if any(m in lowered for m in CAPACITY_MARKERS):
                return _pushed(False, "at_capacity", out, attempted)
            if attempt + 1 == PUSH_ATTEMPTS or not any(m in lowered for m in THROTTLED_PUSH):
                return _pushed(False, "push_failed", out, attempted)
            delay = PUSH_BACKOFF_SEC * (2**attempt)
            _log(
                f"push looks throttled, retrying in {delay}s "
                f"(attempt {attempt + 1}/{PUSH_ATTEMPTS})"
            )
            time.sleep(delay)
        return _pushed(False, "push_failed", out, attempted)
    finally:
        shutil.rmtree(workdir, ignore_errors = True)


def poll(api, slug: str) -> str:
    try:
        raw = str(getattr(api.kernels_status(slug), "status", ""))
    except Exception as exc:  # noqa: BLE001
        _log(f"status unreadable: {type(exc).__name__}")
        return "UNKNOWN"
    match = _STATUS_RE.search(raw)
    return match.group("status") if match else (raw.strip().upper() or "UNKNOWN")


def wait(api, slug: str, poll_every: int, max_wait: int) -> str:
    """Poll to a terminal state.

    An unreadable status must NOT count as "still running" forever. When the
    kernel cannot be seen at all there is no status to match, and sitting in
    this loop for the full ceiling with the kernel long finished is an hour
    of wall clock spent learning nothing. Bound the consecutive failures and
    hand back a state the caller can act on -- UNREADABLE, never COMPLETE,
    because we genuinely do not know how it ended.
    """
    deadline = time.time() + max_wait
    unknowns = 0
    last = "UNKNOWN"
    while time.time() < deadline:
        state = poll(api, slug)
        if state != last:
            _log(f"state: {state}")
            last = state
        if state in TERMINAL_OK or state in TERMINAL_BAD:
            return state
        if state == "UNKNOWN":
            unknowns += 1
            if unknowns >= MAX_CONSECUTIVE_UNKNOWN:
                return "UNREADABLE"
        else:
            unknowns = 0
        time.sleep(poll_every)
    return "HARNESS_TIMEOUT"


def _bearer() -> str:
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        raise RuntimeError("KAGGLE_API_TOKEN unset; REST path unavailable")
    return token


def list_outputs(slug: str, timeout: int = 120) -> dict:
    user, _, name = slug.partition("/")
    params = {"userName": user, "kernelSlug": name}
    files: list[dict] = []
    log = ""
    for _ in range(20):
        url = f"{API_ROOT}/kernels/output?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers = {
                "Authorization": f"Bearer {_bearer()}",
                "User-Agent": "unsloth-kaggle-t4-ci/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout = timeout) as resp:
            data = json.loads(resp.read())
        files.extend(f for f in data.get("files") or [] if f.get("fileName"))
        log = log or (data.get("log") or "")
        token = data.get("nextPageToken") or ""
        if not (data.get("hasNextPageToken") and token):
            break
        params = dict(params, pageToken = token)
    return {"files": files, "log": log}


def _dest_name(file_name: str) -> str:
    """Basename of a listed output, safe on every platform.

    Kaggle lists nested outputs with POSIX separators. Joining a listed name
    onto the output directory unexamined would let ``../`` walk out of it,
    and ``Path`` alone answers for the HOST rather than for the name, so
    peel POSIX first and Windows second with both pure flavours.
    """
    name = PureWindowsPath(PurePosixPath(file_name).name).name
    return name or PurePosixPath(file_name).name


def fetch_evidence(
    slug: str,
    outdir: Path,
    timeout: int = 300,
) -> dict:
    """Pull the executed notebooks and the kernel log by direct URL.

    Evidence first, and by direct URL rather than the bulk download: the
    bulk call returns the WHOLE of /kaggle/working as one stream, and a
    previous incident lost two PASSING notebooks because a multi-GB saved
    model sorted alphabetically ahead of them and the stream broke partway
    through.
    """
    outdir.mkdir(parents = True, exist_ok = True)
    listing = list_outputs(slug, timeout = min(timeout, 120))
    fetched = []
    for entry in listing["files"]:
        name = _dest_name(entry["fileName"])
        if not name.endswith(OUTPUT_SUFFIX):
            continue
        url = entry.get("url") or entry.get("urlNullable")
        if not url:
            continue
        dest = outdir / name
        part = dest.with_suffix(dest.suffix + ".part")
        try:
            req = urllib.request.Request(url, headers = {"User-Agent": "unsloth-kaggle-t4-ci/1.0"})
            with urllib.request.urlopen(req, timeout = timeout) as resp:
                part.write_bytes(resp.read())
            # Only publish once it parses: a download killed mid-write leaves
            # a file of plausible size, and the whole point here is not to
            # produce evidence that looks present and is not.
            json.loads(part.read_text(encoding = "utf-8", errors = "replace"))
            part.replace(dest)
            fetched.append(dest.name)
        except Exception as exc:  # noqa: BLE001
            _log(f"could not fetch {name}: {type(exc).__name__}")
            part.unlink(missing_ok = True)
    log_path = outdir / "kernel.log"
    if listing.get("log"):
        log_path.write_text(listing["log"], encoding = "utf-8")
    return {"notebooks": fetched, "log": log_path.name if log_path.exists() else None}


def _flatten_log(raw: str) -> str:
    """A kernel log as flat text, whichever shape Kaggle returned it in.

    ``kernels/output`` hands the log back as a JSON array of
    ``{stream_name, time, data}`` records rather than as text, and the record
    boundaries are not line boundaries. Scanning that file as-is finds no
    line beginning with the report prefix, so a payload whose executed
    notebook never came back -- the exact case this fallback exists for --
    was read as no report at all and its verdict downgraded to ``infra``.

    ``report.kernel_log_text`` does the same thing for the summary; both are
    kept because the two scripts run as separate processes and neither
    imports the other.
    """
    try:
        records = json.loads(raw)
    except ValueError:
        return raw
    if not isinstance(records, list):
        return raw
    return "".join(r.get("data", "") for r in records if isinstance(r, dict))


def extract_reports(outdir: Path) -> list[dict]:
    """Every T4_SMOKE_REPORT payload found in the collected evidence.

    Looks in the executed notebooks' cell outputs first and the flat kernel
    log second. The notebook is the better source (one cell, unambiguous
    ownership), but the log survives cases where the notebook never got
    written back.
    """
    reports: list[dict] = []
    seen: set[str] = set()

    def _consume(text: str) -> None:
        for line in text.splitlines():
            if not line.startswith(RESULT_PREFIX):
                continue
            blob = line[len(RESULT_PREFIX) :].strip()
            try:
                parsed = json.loads(blob)
            except json.JSONDecodeError:
                continue
            key = f"{parsed.get('label')}|{parsed.get('model')}"
            if key in seen:
                continue
            seen.add(key)
            reports.append(parsed)

    # rglob, not glob: each kernel of a run collects into its own
    # subdirectory so two kernels cannot overwrite each other's kernel.log.
    for nb_path in sorted(outdir.rglob(f"*{OUTPUT_SUFFIX}")):
        try:
            nb = json.loads(nb_path.read_text(encoding = "utf-8", errors = "replace"))
        except Exception:  # noqa: BLE001
            continue
        for cell in nb.get("cells", []):
            for output in cell.get("outputs", []):
                text = output.get("text") or ""
                if isinstance(text, list):
                    text = "".join(text)
                _consume(text)
    for log_path in sorted(outdir.rglob("kernel.log")):
        _consume(_flatten_log(log_path.read_text(encoding = "utf-8", errors = "replace")))
    return reports


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--notebook",
        required = True,
        action = "append",
        help = "kernel notebook to push. Repeatable: all of them "
        "are pushed before any of them is waited on",
    )
    ap.add_argument("--user", required = True)
    ap.add_argument("--outdir", required = True)
    ap.add_argument(
        "--expect", type = int, default = 2, help = "payload reports this kernel should produce"
    )
    ap.add_argument(
        "--kernel-timeout-sec",
        type = int,
        default = 3600,
        help = "hard ceiling enforced by KAGGLE on the session",
    )
    ap.add_argument(
        "--max-wait", type = int, default = 4200, help = "how long this process waits before giving up"
    )
    ap.add_argument("--poll-every", type = int, default = 60)
    ap.add_argument(
        "--keep-kernel", action = "store_true", help = "do not delete the kernel after collecting"
    )
    args = ap.parse_args()

    # Before the first network call, and globally: the Kaggle client offers
    # no per-call timeout of its own. See SOCKET_TIMEOUT_SEC.
    socket.setdefaulttimeout(SOCKET_TIMEOUT_SEC)

    outdir = Path(args.outdir)
    outdir.mkdir(parents = True, exist_ok = True)
    result: dict = {
        "verdict": "infra",
        "reason": "",
        "slug": None,
        "kernel_state": None,
        "reports": [],
    }

    def release() -> None:
        """Delete every kernel this process pushed. Idempotent, and on every
        path out of main().

        This is the budget control, not a tidy-up. A kernel left behind bills
        to its own ceiling with nobody reading the result, and Kaggle's
        push-time timeout has been observed not to stop one that wedged. The
        measurement that settles it: deleting a two-hour-old stuck kernel
        took the account's used-hours figure DOWN.
        """
        if args.keep_kernel:
            return
        for entry in result.get("kernels") or []:
            slug = entry.get("slug")
            if not slug or entry.get("released"):
                continue
            try:
                subprocess.run(
                    ["kaggle", "kernels", "delete", slug, "-y"],
                    capture_output = True,
                    text = True,
                    timeout = 180,
                )
                entry["released"] = True
            except Exception:  # noqa: BLE001
                _log(f"could not delete {slug}; it may keep billing")

    def finish(code: int = 0) -> int:
        release()
        (outdir / "launch_result.json").write_text(json.dumps(result, indent = 2), encoding = "utf-8")
        _out("verdict", result["verdict"])
        _out("reason", result["reason"])
        _out("slug", result["slug"] or "")
        _log(f"verdict={result['verdict']} reason={result['reason']}")
        return code

    try:
        api = _api()
    except BaseException as exc:  # noqa: BLE001
        if isinstance(exc, KeyboardInterrupt):
            raise
        result["reason"] = f"kaggle auth failed: {type(exc).__name__}"
        return finish()

    # Push everything first. See this file's docstring: waiting between
    # pushes would serialise sessions Kaggle runs happily in parallel, and
    # would put an hour between the control leg and the canary leg.
    kernels: list[dict] = []
    for notebook in args.notebook:
        _log(f"pushing {notebook} (kernel ceiling {args.kernel_timeout_sec}s)")
        pushed = push(Path(notebook), args.user, args.kernel_timeout_sec)
        entry = {
            "notebook": notebook,
            "slug": pushed.get("slug"),
            # Every slug the push filed, accepted or not. A push that
            # reported an error may still have landed, and this is the only
            # record of what to reconcile against the account afterwards.
            "attempted": pushed.get("attempts") or [],
            "state": None,
            "push_error": None
            if pushed["ok"]
            else f"{pushed['reason']}: {pushed.get('detail', '')}",
        }
        kernels.append(entry)
        if pushed["ok"]:
            _log(f"pushed as {pushed['slug']}")
        else:
            _log(f"push failed for {notebook}: {entry['push_error']}")
    result["kernels"] = kernels

    live = [k for k in kernels if k["slug"]]
    if not live:
        result["reason"] = "; ".join(k["push_error"] for k in kernels if k["push_error"])
        return finish()
    # Kept for the summary and for anything reading the previous single-kernel
    # shape of this file.
    result["slug"] = live[0]["slug"]

    # One deadline for the whole run, not one per kernel. They are running
    # concurrently, so consuming the ceiling once per kernel would let a
    # two-kernel run wait twice as long as its own stated bound.
    deadline = time.time() + args.max_wait
    for entry in live:
        remaining = max(0, int(deadline - time.time()))
        entry["state"] = wait(api, entry["slug"], args.poll_every, remaining)
        _log(f"{entry['slug']} terminal state: {entry['state']}")
    result["kernel_state"] = ",".join(k["state"] or "?" for k in live)

    for entry in live:
        # Each kernel gets its own directory. Kaggle names the executed
        # notebooks after the payloads, and two kernels of the same run
        # would otherwise overwrite each other's kernel.log.
        try:
            entry["evidence"] = fetch_evidence(
                entry["slug"], outdir / entry["slug"].rsplit("/", 1)[-1]
            )
            _log(f"collected {entry['slug']}: {entry['evidence']}")
        except Exception as exc:  # noqa: BLE001
            entry["evidence"] = None
            entry["collect_error"] = type(exc).__name__
            _log(f"could not collect {entry['slug']}: {type(exc).__name__}")

    reports = extract_reports(outdir)
    result["reports"] = reports
    _log(f"extracted {len(reports)} payload report(s) of {args.expect} expected")

    if not reports:
        result["reason"] = (
            f"kernel(s) ended {result['kernel_state']} but produced no payload "
            f"report; nothing was learned about the code under test"
        )
        return finish()

    # A kernel that ended badly but still produced reports is worth reading:
    # the payload deliberately does not propagate a nonzero exit, so ERROR
    # here usually means the SESSION died (timeout, OOM of the box, Kaggle
    # side), which is infra unless a report says otherwise.
    failing = [r for r in reports if not r.get("passed")]
    if failing:
        result["verdict"] = "fail"
        result["reason"] = (
            f"{len(failing)} of {len(reports)} payload(s) failed their " f"assertions"
        )
    elif len(reports) < args.expect:
        result["verdict"] = "partial"
        result["reason"] = (
            f"only {len(reports)} of {args.expect} payload(s) reported back "
            f"(kernel state {result['kernel_state']}); the ones that did, "
            f"passed"
        )
    else:
        result["verdict"] = "pass"
        result["reason"] = f"all {len(reports)} payload(s) passed"

    # The kernels are released by finish(), on this path and on every other.
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
