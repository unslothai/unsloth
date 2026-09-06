# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Collect the kernels an earlier job dispatched and left running.

``launch.py --dispatch`` pushes a kernel and exits within minutes instead of
holding a GitHub runner for the forty it takes Kaggle to finish. That trade is
only sound if something else finishes the job, and this is that something.

THE ACCOUNT IS THE QUEUE. No database, artifact or branch sits between the
dispatching job and this one, because each is a second place for the truth to
go stale. ``kernels list --mine`` enumerates the kernels and the SLUG carries
the commit and workflow (``launch.slug_name``), so this script rediscovers
everything from the account and can run from any job, in any order, any number
of times.

Three outcomes per kernel, and they are not symmetric:

* **terminal** -- download the evidence, judge the reports, DELETE, and emit a
  commit status for the sha in the slug. The only path that produces a verdict.
* **still running, within its age ceiling** -- left completely alone; a kernel
  doing its job is not a problem to report.
* **still running, past the ceiling** -- deleted, and reported as a failure to
  collect rather than of the code. This is the reaper, and why the script runs
  on a SCHEDULE: a dispatched kernel bills accelerator quota to its ceiling and
  a quiet repo would leave it there.

This script never talks to GitHub. It emits the statuses it wants posted into
``collect_result.json`` and the workflow posts them (``post_statuses.py``): a
process holding both a Kaggle and a GitHub credential is one bug away from
sending one to the other, and a verdict needing GitHub access cannot be tested
on CPU.

DELETION COMES LAST, after the status is delivered. The workflows run
``--no-delete``, post, then ``--delete-collected``. A kernel whose status did
not reach GitHub is left up for the next pass; deleting first would turn a
transient API failure into the permanent loss of a verdict.

SAFETY above all. This script enumerates a whole ACCOUNT shared with a human
and DELETES what it collects, so nothing is touched unless
``launch.parse_slug`` recognises its name. An unrecognised kernel is not
skipped-with-a-warning, it is invisible.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import launch  # noqa: E402
from gate import GONE_MARKERS, _as_naive_utc  # noqa: E402

# How long a dispatched kernel may stay in flight before the reaper takes it.
# Kaggle's own 12h kill has been caught failing here: a kernel pushed with
# `-t 5400` whose nbconvert crashed sat RUNNING two hours past it until deleted
# by hand. So the ceiling is ours, and well under Kaggle's.
#
# Generous rather than tight: the wired notebook kernel measured 2101.8s (35
# min), the two-account run took 41.5 min of job wall clock, and queueing behind
# another of our own adds most of that again. Reaping a kernel that would have
# finished costs a whole run and reports a failure nobody can act on.
DEFAULT_MAX_AGE_HOURS = 3.0

# Paging for the account walk. Bounded by PAGES, not by age: a kernel far past
# the reap ceiling is exactly the one the reaper exists for, and an old terminal
# kernel still owes its commit a status. Everything pushed is collected or
# reaped once and then gone, so the walk does not grow.
PAGE_SIZE = 100
MAX_PAGES = 5

# The driver's own record of how many payload reports the kernel was BUILT to
# produce, and the only one that survives dispatch: judging a five-payload
# kernel against `--expect 1` turns a run that lost four of them into a pass.
EXPECT_SENTINEL = "KAGGLE_T4_CI_DRIVER_EXPECT "

# Wall clock for the whole collection (evidence downloads dominate). This runs
# on a schedule beside a 600-deep queue, so it must not become the job that
# never ends: a kernel missed this pass is reached five minutes later.
BUDGET_SEC = 900

# Per-kernel evidence budget, well under the total, so one slow download cannot
# consume the pass and starve every kernel behind it.
EVIDENCE_BUDGET_SEC = 300

SOCKET_TIMEOUT_SEC = 120

# The commit status context per workflow kind. A PUBLIC interface: branch
# protection is configured against these strings, so renaming one silently
# stops requiring the check it names.
STATUS_CONTEXTS = {
    "notebook": "kaggle-t4-notebook",
    "studio": "kaggle-studio-gpu",
}


def _log(msg: str) -> None:
    print(f"[collect] {msg}", flush = True)


def _out(key: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    try:
        with open(path, "a", encoding = "utf-8") as fh:
            fh.write(f"{key}={value}\n")
    except OSError:
        pass


def kernel_age_hours(kernel, now: datetime) -> float | None:
    """How long ago this kernel last started, or None if Kaggle did not say.

    For an UNFINISHED kernel ``last_run_time`` is when the run started, which is
    what the reap ceiling measures against. None rather than a default, and the
    caller treats it as too young to reap: guessing old deletes a live session.
    """
    last_run = _as_naive_utc(getattr(kernel, "last_run_time", None))
    if last_run is None:
        return None
    return (now - last_run).total_seconds() / 3600.0


def find_ours(
    api,
    now: datetime | None = None,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    page_size: int = PAGE_SIZE,
    max_pages: int = MAX_PAGES,
) -> list[dict]:
    """Every kernel on this account that WE pushed, newest first.

    The filter is ``launch.parse_slug`` and nothing else: an unrecognised kernel
    is not ours to read, let alone delete. ``max_age_hours`` is the reap ceiling
    the caller applies later, not a listing cutoff, because the kernel that most
    needs collecting is the one a delayed pass finds hours past it.
    """
    now = now or datetime.now(timezone.utc).replace(tzinfo = None)
    found: list[dict] = []
    for page in range(1, max_pages + 1):
        kernels = (
            api.kernels_list(mine = True, page = page, page_size = page_size, sort_by = "dateRun") or []
        )
        if not kernels:
            break
        for kernel in kernels:
            ref = getattr(kernel, "ref", None)
            if not ref:
                continue
            parsed = launch.parse_slug(ref)
            if parsed is None:
                continue
            found.append(
                {
                    "slug": ref,
                    "sha": parsed["sha"],
                    "kind": parsed["kind"],
                    "legacy": parsed["legacy"],
                    "age_hours": kernel_age_hours(kernel, now),
                }
            )
    return found


def _status_of(api, slug: str) -> str:
    """The kernel's state, with a deleted kernel distinguished from an unknown one.

    A kernel deleted moments ago still lists but answers its status call with a
    404 -- definitively not running, not unreadable. Conflating the two makes
    the reaper attack gone kernels and read a genuine outage as idle.
    """
    try:
        raw = str(getattr(api.kernels_status(slug), "status", ""))
    except Exception as exc:  # noqa: BLE001
        text = f"{exc}".lower()
        if any(m in text for m in GONE_MARKERS):
            return "GONE"
        _log(f"status unreadable for {slug}: {type(exc).__name__}")
        return "UNKNOWN"
    match = launch._STATUS_RE.search(raw)
    return match.group("status") if match else (raw.strip().upper() or "UNKNOWN")


def verdict_of(reports: list[dict], expect: int) -> tuple[str, str]:
    """Turn the payload reports into a verdict, exactly as launch.py does inline.

    Kept identical so a dispatched run and a waited-on run reach the same
    conclusion from the same evidence.
    """
    if not reports:
        return "infra", (
            "the kernel finished but produced no payload report; "
            "nothing was learned about the code under test"
        )
    failing = [r for r in reports if not r.get("passed")]
    if failing:
        names = ", ".join(str(r.get("payload") or "?") for r in failing)
        return (
            "fail",
            f"{len(failing)} of {len(reports)} payload(s) failed their assertions: {names}",
        )
    if len(reports) < expect:
        return "partial", (
            f"only {len(reports)} of {expect} payload(s) reported back, "
            "so this is half a comparison rather than a result"
        )
    return "pass", f"all {len(reports)} payload(s) passed"


def _gone(exc: BaseException) -> bool:
    """Did this failure say the kernel is no longer there?

    The notebook job, the Studio job and the scheduled pass all collect and
    nothing serialises them, so another can delete a kernel between this pass
    seeing it terminal and downloading it. Not an infra failure: a result
    somebody else already posted.
    """
    text = f"{exc}".lower()
    return any(m in text for m in GONE_MARKERS)


def _evidence_lines(dest: Path):
    """Every line of driver output in the collected evidence, notebooks first."""
    for nb_path in sorted(dest.rglob(f"*{launch.OUTPUT_SUFFIX}")):
        try:
            nb = json.loads(nb_path.read_text(encoding = "utf-8", errors = "replace"))
        except Exception:  # noqa: BLE001
            continue
        # Valid JSON that is not a notebook (`[]`, a bare string) is skipped,
        # not raised on: this runs outside the report guard, and an exception
        # here writes no result file and wedges every later pass on this kernel.
        if not isinstance(nb, dict):
            continue
        for cell in nb.get("cells") or []:
            if not isinstance(cell, dict):
                continue
            for output in cell.get("outputs") or []:
                if not isinstance(output, dict):
                    continue
                text = output.get("text") or ""
                if isinstance(text, list):
                    text = "".join(str(t) for t in text)
                if isinstance(text, str):
                    yield from text.splitlines()
    for log_path in sorted(dest.rglob("kernel.log")):
        raw = log_path.read_text(encoding = "utf-8", errors = "replace")
        yield from launch.flatten_kernel_log(raw).splitlines()


def expected_reports(dest: Path, default: int) -> int:
    """How many payload reports this kernel was built to produce.

    Read off the kernel's own evidence, the only record that survives dispatch.
    A kernel predating the sentinel, or a Studio kernel (always one report),
    answers ``default``.
    """
    for line in _evidence_lines(dest):
        if not line.startswith(EXPECT_SENTINEL):
            continue
        try:
            parsed = json.loads(line[len(EXPECT_SENTINEL) :].strip())
        except ValueError:
            continue
        reports = parsed.get("reports") if isinstance(parsed, dict) else None
        if isinstance(reports, int) and reports > 0:
            return reports
    return default


# How a verdict is reported to GitHub. `infra` and `partial` are deliberately
# NOT failures: nothing was learned about the code, and a red the author cannot
# act on is how a required check gets dropped from branch protection. They post
# `success` with a description saying so.
VERDICT_STATE = {
    "pass": "success",
    "fail": "failure",
    "partial": "success",
    "infra": "success",
    "reaped": "failure",
}


def collect_one(
    api,
    entry: dict,
    outdir: Path,
    expect: int,
    max_age_hours: float,
    delete: bool = True,
    deadline: float | None = None,
) -> dict:
    """Read one kernel to a conclusion. Returns the record for the result file.

    ``deadline`` is the whole pass's; the evidence download gets its own budget
    clamped to what is left, so the last kernel cannot overrun the job.
    """
    slug = entry["slug"]
    state = _status_of(api, slug)
    record = {
        **entry,
        "state": state,
        "verdict": None,
        "reason": "",
        "deleted": False,
        "reports": 0,
    }

    if state in ("QUEUED", "RUNNING"):
        age = entry.get("age_hours")
        if age is not None and age > max_age_hours:
            # Reaped, and reported as a FAILURE of collection rather than
            # dropped: it has been billing throughout and will never produce a
            # result, and silence leaves the commit reading "not run" forever.
            record["verdict"] = "reaped"
            record["reason"] = (
                f"the kernel was still {state} after {age:.1f}h, past the {max_age_hours}h "
                "ceiling, so it was deleted. It was billing accelerator quota and would "
                "not have produced a result"
            )
            if delete:
                record["deleted"] = launch.delete_kernel(slug)
            _log(f"reaped {slug} ({state}, {age:.1f}h)")
            return record
        record["verdict"] = "pending"
        record["reason"] = f"still {state}"
        _log(f"pending {slug} ({state}, age {age if age is None else round(age, 2)}h)")
        return record

    if state == "UNKNOWN":
        # Left alone: an unreadable status says nothing, and both actions here
        # are destructive (delete may kill a live session, report may fail a
        # run that was fine). The next pass asks again.
        record["verdict"] = "pending"
        record["reason"] = "status unreadable this pass"
        return record

    if state == "GONE":
        record["verdict"] = "gone"
        record["reason"] = "the kernel no longer exists; nothing to collect"
        return record

    if state not in launch.TERMINAL_OK | launch.TERMINAL_BAD:
        # Only a KNOWN terminal state may download, judge, post and delete.
        # Kaggle's enum has states this file never sees (NEW_SCRIPT) and can
        # grow more; judging one would post a green `infra` for a run still to
        # come and then delete it.
        record["verdict"] = "pending"
        record["reason"] = f"state {state} is not one this collector judges; asked again next pass"
        _log(f"unrecognised state {state} for {slug}; kept")
        return record

    # Terminal. Evidence FIRST, delete second: a delete before the download
    # turns a finished run into a result nobody can read.
    dest = outdir / slug.rsplit("/", 1)[-1]
    try:
        evidence_deadline = time.time() + EVIDENCE_BUDGET_SEC
        if deadline is not None:
            evidence_deadline = min(evidence_deadline, deadline)
        evidence = launch.fetch_evidence(slug, dest, deadline = evidence_deadline)
    except Exception as exc:  # noqa: BLE001
        record["evidence"] = None
        if _gone(exc):
            # Deleted by another pass between our status call and this
            # download. Their verdict stands; ours would be an `infra` on top.
            record["verdict"] = "gone"
            record["reason"] = "another collector finished this kernel first"
            _log(f"{slug} was collected by another pass")
            return record
        # Kept and NOT judged; the next pass tries again. `pending` rather than
        # `infra` because `infra` posts green and releases the kernel, turning
        # one transient listing failure into the loss of a real result.
        record["verdict"] = "pending"
        record["reason"] = (
            f"the kernel finished but its evidence would not download this pass "
            f"({type(exc).__name__}); kept for the next one"
        )
        _log(f"could not collect {slug}: {type(exc).__name__}")
        return record

    record["evidence"] = evidence
    if evidence.get("truncated"):
        # fetch_evidence flags a spent budget or failed download instead of
        # raising, so same answer as the except above: nothing judged, nothing
        # deleted. Judging a short set reads a run that lost half its notebooks
        # as whatever the surviving half says.
        record["verdict"] = "pending"
        record["reason"] = (
            "the evidence download was incomplete this pass; the kernel is kept for the next one"
        )
        _log(f"incomplete evidence for {slug}; left for the next pass")
        return record

    try:
        reports = launch.extract_reports(dest)
    except Exception as exc:  # noqa: BLE001
        # Evidence that downloaded but cannot be read will not read better next
        # pass, and raising here writes no result and wedges every later pass.
        # Nothing was learned, so `infra`, and the kernel is released.
        reports = []
        record["report_error"] = f"{type(exc).__name__}: {exc}"[:200]
        _log(f"unreadable report in {slug}: {type(exc).__name__}")
    record["reports"] = len(reports)
    expect = expected_reports(dest, expect)
    record["expected"] = expect
    verdict, reason = verdict_of(reports, expect)
    if record.get("report_error") and verdict == "infra":
        reason = (
            f"the evidence downloaded but its report could not be read ({record['report_error']})"
        )
    if state in launch.TERMINAL_BAD and verdict == "infra":
        reason = (
            f"the kernel ended {state} without reporting; the session died rather than the code"
        )
    record["verdict"] = verdict
    record["reason"] = reason
    if delete:
        record["deleted"] = launch.delete_kernel(slug)
        if not record["deleted"]:
            _log(f"could not delete {slug}; it may keep billing")
    _log(f"collected {slug}: {verdict} ({reason})")
    return record


def statuses_from(records: list[dict], target_url: str = "") -> list[dict]:
    """The commit statuses to post, one per collected kernel that names a commit.

    THIS is the check that means "the GPU tests passed": the dispatching job
    succeeds by dispatching, so branch protection pointing at it would require
    only that Kaggle accepted a push. A legacy slug with no sha yields nothing;
    there is no commit to attribute it to.
    """
    out: dict[tuple[str, str], dict] = {}
    for record in records:
        if record.get("verdict") in (None, "pending", "gone"):
            continue
        sha = record.get("sha")
        kind = record.get("kind")
        if not sha or kind not in STATUS_CONTEXTS:
            continue
        context = STATUS_CONTEXTS[kind]
        state = VERDICT_STATE.get(record["verdict"], "success")
        # One line: GitHub shows the description on one line and caps it at 140.
        description = " ".join(f"{record['verdict']}: {record['reason']}".split())[:140]
        status = {
            "sha": sha,
            "context": context,
            "state": state,
            "description": description,
            "target_url": target_url,
            "slug": record["slug"],
            "slugs": [record["slug"]],
        }
        # Two kernels for one commit under one context (notebook slots 1 and 2
        # of the same sha) must not race to post last: a failure wins whichever
        # kernel found it, and both are named as owners so the delete step
        # releases both.
        key = (sha, context)
        prior = out.get(key)
        if prior is None:
            out[key] = status
            continue
        prior["slugs"].append(record["slug"])
        if state == "failure" and prior["state"] != "failure":
            status["slugs"] = prior["slugs"]
            out[key] = status
    return list(out.values())


# Verdicts whose kernel is finished with once the status is delivered. `pending`
# and `gone` own nothing to delete; `None` never got as far as a verdict.
DELETABLE = {"pass", "fail", "partial", "infra", "reaped"}


def delete_collected(result_path: Path, posted_path: Path | None) -> int:
    """Release the kernels a ``--no-delete`` pass judged, now that their
    statuses are posted.

    ``posted.json`` names the slugs delivered, refused, and whose commit is
    gone. A refused post KEEPS its kernel, which is the only retry there is once
    the runner is gone. A gone commit releases it; holding it only bills.
    """
    data = json.loads(result_path.read_text(encoding = "utf-8"))
    posted = (
        json.loads(posted_path.read_text(encoding = "utf-8"))
        if posted_path and posted_path.exists()
        else {}
    )
    failed = set(posted.get("failed") or [])
    if posted_path and not posted_path.exists():
        # No record of delivery: the poster never ran. Keep every kernel that
        # had something to post, or their verdicts are lost.
        failed = {s["slug"] for s in data.get("statuses", [])} | {
            slug for s in data.get("statuses", []) for slug in s.get("slugs", [])
        }
    outcome = {"deleted": [], "kept": [], "failed_delete": []}
    for record in data.get("kernels") or []:
        slug = record.get("slug")
        if not slug or record.get("verdict") not in DELETABLE:
            continue
        if record.get("deleted"):
            continue
        if slug in failed:
            outcome["kept"].append(slug)
            _log(f"kept {slug}: its status did not post; the next pass retries")
            continue
        if launch.delete_kernel(slug):
            outcome["deleted"].append(slug)
            _log(f"deleted {slug}")
        else:
            outcome["failed_delete"].append(slug)
            print(f"::warning title=Kaggle kernel not deleted::{slug} may keep billing")
    result_path.with_name("delete_result.json").write_text(
        json.dumps(outcome, indent = 2), encoding = "utf-8"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required = True, help = "where evidence is downloaded")
    ap.add_argument(
        "--expect",
        type = int,
        default = 1,
        help = "payload reports a collected kernel should carry",
    )
    ap.add_argument("--max-age-hours", type = float, default = DEFAULT_MAX_AGE_HOURS)
    ap.add_argument(
        "--sha",
        default = "",
        help = "when given, report whether a kernel for THIS commit is already in "
        "flight, so the caller can skip dispatching a second one",
    )
    ap.add_argument(
        "--kind",
        default = "",
        choices = ("", *launch.KIND_CODES),
        help = "narrow --sha to one workflow's kernels",
    )
    ap.add_argument("--target-url", default = "", help = "run URL to attach to each status")
    ap.add_argument(
        "--no-delete",
        action = "store_true",
        help = "collect and report without deleting. The workflows always pass "
        "this and delete afterwards with --delete-collected, once the statuses "
        "are posted; a kernel left up after that keeps billing",
    )
    ap.add_argument(
        "--require-auth",
        action = "store_true",
        help = "fail rather than skip when Kaggle authentication fails. The "
        "scheduled reaper passes this: it runs with the repository's own "
        "secrets, so an auth failure there is a broken token, not a fork",
    )
    ap.add_argument(
        "--delete-collected",
        default = "",
        metavar = "COLLECT_RESULT_JSON",
        help = "delete the kernels a previous --no-delete pass collected, except "
        "those whose status did not post (see --posted). Nothing else runs",
    )
    ap.add_argument(
        "--posted",
        default = "",
        metavar = "POSTED_JSON",
        help = "post_statuses.py's record of which statuses were delivered",
    )
    args = ap.parse_args()

    if args.delete_collected:
        return delete_collected(
            Path(args.delete_collected), Path(args.posted) if args.posted else None
        )

    socket.setdefaulttimeout(SOCKET_TIMEOUT_SEC)
    outdir = Path(args.outdir)
    outdir.mkdir(parents = True, exist_ok = True)
    result: dict = {"owner": None, "kernels": [], "statuses": [], "in_flight_for_sha": False}

    def finish(code: int = 0) -> int:
        (outdir / "collect_result.json").write_text(json.dumps(result, indent = 2), encoding = "utf-8")
        _out("in_flight", "true" if result["in_flight_for_sha"] else "false")
        _out(
            "collected",
            str(
                sum(
                    1
                    for k in result["kernels"]
                    if k.get("verdict") not in (None, "pending", "gone")
                )
            ),
        )
        _out("pending", str(sum(1 for k in result["kernels"] if k.get("verdict") == "pending")))
        return code

    try:
        api = launch._api()
    except BaseException as exc:  # noqa: BLE001
        if isinstance(exc, KeyboardInterrupt):
            raise
        if args.require_auth:
            # The scheduled reaper runs on the repository's own token, so this
            # is an expired credential. A green empty pass would hide it while
            # kernels bill to their ceiling and commits stay pending.
            _log(f"kaggle auth failed ({type(exc).__name__}) and this pass requires it")
            print(
                "::error title=Kaggle authentication failed::the collector could not "
                f"authenticate ({type(exc).__name__}). Nothing was collected or reaped; "
                "dispatched kernels keep billing and their commits stay pending until "
                "the token is repaired."
            )
            return finish(1)
        # A skip, not a failure, for the same reason gate.py skips: on a fork
        # pull request the secret is withheld and there is nothing to fix.
        _log(f"kaggle auth failed ({type(exc).__name__}); nothing collected")
        return finish()

    owner = launch.username_of(api)
    result["owner"] = owner
    _log(f"authenticated as {owner}")

    ours = find_ours(api, max_age_hours = args.max_age_hours)
    _log(f"{len(ours)} kernel(s) of ours on this account")

    deadline = time.time() + BUDGET_SEC
    for entry in ours:
        if time.time() >= deadline:
            # Safe by construction: nothing was deleted that was not first
            # collected, and the next pass sees the same account. Logged so a
            # partial pass is not read as an empty one.
            _log("collection budget spent; the rest is left for the next pass")
            break
        result["kernels"].append(
            collect_one(
                api,
                entry,
                outdir,
                args.expect,
                args.max_age_hours,
                delete = not args.no_delete,
                deadline = deadline,
            )
        )

    result["statuses"] = statuses_from(result["kernels"], args.target_url)

    if args.sha:
        want = args.sha.strip().lower()
        # Prefix match either way: slugs carry SLUG_SHA_LEN characters now and
        # carried eight before, and both can be on the account at once.
        result["in_flight_for_sha"] = any(
            k.get("sha")
            and (want.startswith(k["sha"]) or k["sha"].startswith(want))
            and k.get("verdict") == "pending"
            and (not args.kind or k.get("kind") == args.kind)
            for k in result["kernels"]
        )
        _log(f"in flight for {want}: {result['in_flight_for_sha']}")

    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
