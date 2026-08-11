# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Decide whether this invocation is allowed to spend Kaggle GPU quota.

The account behind KAGGLE_ACCESS_TOKEN_GH has a WEEKLY accelerator budget,
shared with every other use of that account. A workflow that launched a
kernel on every push would drain a week of it in a day and lock out every
other consumer, so the default answer here is "no" and the job has to earn
a "yes" through four independent checks, in this order:

1. **Override.** ``workflow_dispatch`` with ``force=true``, or a pull request
   carrying the opt-in label. A human asked for it; skip the dice.
2. **Sampling.** Roughly one invocation in ten. Derived from the run id, so
   re-running the same workflow run gives the same answer (a re-run must not
   be a fresh roll of the dice, or anyone could reroll until it fires) while
   different runs are independent.
3. **Remaining quota.** Refuses to start when what is left would not cover
   the worst case this invocation could cost, plus a reserve so the account
   is never drained to zero by CI.
4. **Concurrency.** Kaggle caps concurrent batch GPU kernels at 2, and that
   cap is per ACCOUNT, not per workflow. The survey therefore separates
   kernels this workflow pushed from everything else, because the two call
   for opposite answers: a FOREIGN kernel means a human is using the shared
   account and this invocation stands down entirely, while this workflow's
   own two kernels are what it is here to launch. Both still occupy slots,
   so the gate also refuses when fewer than KERNELS_PER_INVOCATION of them
   are free. The search for an in-flight kernel is bounded by how long a
   session is allowed to last rather than by a kernel count, which is what
   makes it exhaustive; see LOOKBACK_HOURS.

Every negative answer is a SKIP, and a skip exits 0. Not spending quota is
the designed behaviour, not a fault, and must never colour a pull request
red. The only nonzero exit here is a real error in the gate itself, and even
that is converted to a skip by ``--soft-fail``.

No credential is ever printed. The token is read from the environment,
handed to the Kaggle client, and never echoed, logged, or written to an
output file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone

# Kaggle's cap on simultaneous batch (committed) GPU kernels per account.
# Measured, not documented: exceeding it fails the push with
# "Maximum batch GPU session count of 2 reached."
MAX_CONCURRENT_GPU_KERNELS = 2

# How many FOREIGN kernels of this account may already be in flight and this
# job still launch. ZERO IS DELIBERATE, and it is a policy choice rather than
# a technical limit, so it is named here instead of being implied by the code.
#
# The cap is per ACCOUNT and the account is shared with human use. A person
# who starts a notebook and finds the push rejected has no way to tell that
# CI took the slot, and CI has no way to give it back. The cost of standing
# down is a few minutes until the next commit draws again -- the sampling
# gate means this job has no deadline of its own -- and the cost of being
# wrong is somebody else's session. Yielding is cheap here and expensive
# there, so a single foreign kernel stands this job down entirely, whatever
# the arithmetic below would otherwise allow.
ALLOWED_IN_FLIGHT_FOREIGN_KERNELS = 0

# How many kernels one invocation of this workflow pushes. Both of Kaggle's
# concurrency slots, which is a change from the single kernel this workflow
# used to run, and the reasoning for taking the second one is narrow:
#
# The second slot is not "spare capacity to be grabbed", it is capacity this
# job takes only when the account is otherwise IDLE, which the survey
# establishes immediately beforehand. What changed is not the willingness to
# compete with a human -- that is still zero -- but the payload: four legs
# are now worth running and two kernels x two T4s is the only shape that
# fits them. Legs that were split across two sessions would be compared
# across two images and two hours, and for the control/canary pair that
# comparison is the entire instrument.
#
# The residual risk is a foreign kernel that starts BETWEEN the survey and
# the push. That is not new and it is already handled where it lands: the
# launcher recognises Kaggle's capacity rejection (CAPACITY_MARKERS) and
# reports it as infra, exiting 0. A human's push is never the one rejected,
# because ours is the one that arrives second.
KERNELS_PER_INVOCATION = 2

# Kernel states that mean a session is occupying one of those slots.
BUSY_STATES = {"QUEUED", "RUNNING"}

# How this job recognises its own kernels. `launch.py` pushes every kernel as
# `<user>/<OWN_KERNEL_PREFIX><8 hex>`, a fresh slug per attempt, and nothing
# else on the account uses that prefix.
#
# The distinction matters because "the account is busy" and "this workflow is
# busy" call for opposite answers. A foreign kernel means a human is using
# the account and this job must yield. One of our own means a previous run of
# this workflow is still in flight, which the job-level concurrency group is
# supposed to prevent; it is reported separately rather than being counted as
# a human, and it still occupies a slot, so it still blocks.
OWN_KERNEL_PREFIX = "unsloth-t4-ci-"

# How far back the in-flight survey has to look, and why that bound is
# COMPLETE rather than merely convenient.
#
# Kaggle exposes no "list my running sessions" call, so the only way to find
# an in-flight kernel is to list kernels and status-check them. The listing
# is sorted by last run time, descending, and for a kernel that has not
# finished, that timestamp is when the run STARTED (measured: a kernel
# pushed at 10:05:19Z listed as last_run_time 10:05:19.297).
#
# Kaggle kills a notebook session at 12 hours (CPU/GPU; 9 for TPU). So a
# kernel that is still QUEUED or RUNNING cannot have started more than 12
# hours ago, and once the walk reaches an entry older than that, every
# remaining entry is older still and none of them can be in flight. Stopping
# there is therefore exhaustive, not a sample -- which the previous fixed
# bound of "the 12 most recent kernels" was not: a kernel that started three
# hours ago and is still running would be missed the moment twelve newer
# ones had since run, and the push would then fail at the capacity cap and
# be reported as infra.
MAX_SESSION_HOURS = 12.0

# Slack on top, for clock skew between Kaggle's timestamps and this runner,
# and for any future raise of the session ceiling.
CLOCK_SKEW_HOURS = 1.0
LOOKBACK_HOURS = MAX_SESSION_HOURS + CLOCK_SKEW_HOURS

# Paging for that walk. The cap on pages exists so a pathological account
# cannot make the gate walk forever; reaching it means the survey did NOT
# cover the whole window, which is reported as an incomplete survey and
# treated as "unknown", never as "idle".
KERNELS_PAGE_SIZE = 100
MAX_KERNEL_PAGES = 5


def _out(key: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding = "utf-8") as fh:
            fh.write(f"{key}={value}\n")
    print(f"[gate] {key}={value}", flush = True)


def _summary(text: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding = "utf-8") as fh:
            fh.write(text + "\n")


def _decide(run: bool, reason: str) -> int:
    _out("should_run", "true" if run else "false")
    _out("reason", reason)
    verdict = "RUN" if run else "SKIP"
    print(f"[gate] {verdict}: {reason}", flush = True)
    _summary(f"### Kaggle T4 gate: {verdict}\n\n{reason}\n")
    return 0


def sampled_in(run_id: str, percent: int) -> tuple[bool, int]:
    """Stable pseudo-random draw in [0, 100) derived from the run id."""
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    draw = int(digest[:8], 16) % 100
    return draw < percent, draw


def kaggle_client():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def remaining_gpu_hours(api) -> dict:
    """Exact remaining accelerator hours from Kaggle's own quota API."""
    resp = api.quota_view()
    quota = getattr(resp, "gpu_quota", None)
    if quota is None:
        return {"ok": False, "error": "no gpu_quota in quota response"}
    used = quota.time_used.total_seconds() / 3600.0
    total = quota.total_time_allowed.total_seconds() / 3600.0
    refresh = getattr(resp, "quota_refresh_time", None)
    return {
        "ok": True,
        "used_hours": round(used, 3),
        "total_hours": round(total, 3),
        "remaining_hours": round(max(0.0, total - used), 3),
        "refresh_at": refresh.isoformat() if refresh else None,
    }


def _as_naive_utc(value):
    """Kaggle's timestamps, normalised so they can be compared at all."""
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo = None)


def survey_kernels(
    api,
    now: datetime | None = None,
    lookback_hours: float = LOOKBACK_HOURS,
    page_size: int = KERNELS_PAGE_SIZE,
    max_pages: int = MAX_KERNEL_PAGES,
) -> dict:
    """Status-check every kernel that could still be in flight.

    Walks the account's kernels most-recently-run first and stops at the
    first one that started longer ago than any session is allowed to last.
    See LOOKBACK_HOURS for why that makes the walk exhaustive.

    Returns the busy refs, split by ownership, plus enough bookkeeping for
    the caller to tell "nothing is running" from "the question could not be
    answered":

    ``busy`` / ``own`` / ``foreign``
        every in-flight kernel, and the two disjoint halves of that list.
        A kernel is ours when its slug carries OWN_KERNEL_PREFIX, which is
        what ``launch.py`` pushes under and nothing else uses.
    ``complete``
        the walk either ran off the end of the listing or reached an entry
        outside the window. False means the page cap was hit first and some
        candidate kernels were never looked at.
    ``surveyed`` / ``unreadable``
        how many in-window kernels were status-checked, and how many of
        those answers came back as an error. All of them unreadable means
        the status API is down, which is not evidence of an idle account.
    """
    # Naive UTC, to match what Kaggle returns. utcnow() would do the same
    # thing and is deprecated from 3.12.
    now = now or datetime.now(timezone.utc).replace(tzinfo = None)
    cutoff = now - timedelta(hours = lookback_hours)
    busy: list[str] = []
    own: list[str] = []
    foreign: list[str] = []
    surveyed = 0
    unreadable = 0
    complete = False

    for page in range(1, max_pages + 1):
        kernels = (
            api.kernels_list(mine = True, page = page, page_size = page_size, sort_by = "dateRun") or []
        )
        for kernel in kernels:
            ref = getattr(kernel, "ref", None)
            if not ref:
                continue
            last_run = _as_naive_utc(getattr(kernel, "last_run_time", None))
            # A missing timestamp cannot end the walk (it says nothing about
            # age) but it can still be checked, so check it.
            if last_run is not None and last_run < cutoff:
                complete = True
                break
            surveyed += 1
            try:
                status = str(getattr(api.kernels_status(ref), "status", ""))
            except Exception as exc:  # noqa: BLE001
                # An unreadable status is not evidence of an idle account.
                # Count it, say so, and keep going.
                unreadable += 1
                print(f"[gate] status unreadable for {ref}: " f"{type(exc).__name__}", flush = True)
                continue
            state = status.rsplit(".", 1)[-1].upper()
            if state in BUSY_STATES:
                entry = f"{ref} ({state})"
                busy.append(entry)
                # Ownership is read off the SLUG, which is the part after the
                # username: a foreign kernel on this account belongs to the
                # same user, so the user half says nothing.
                slug = ref.rsplit("/", 1)[-1]
                (own if slug.startswith(OWN_KERNEL_PREFIX) else foreign).append(entry)
        if complete:
            break
        if len(kernels) < page_size:
            # Ran off the end of the account's kernels; nothing is left to
            # miss, so the survey covered everything it needed to.
            complete = True
            break

    return {
        "busy": busy,
        "own": own,
        "foreign": foreign,
        "surveyed": surveyed,
        "unreadable": unreadable,
        "complete": complete,
        "window_hours": lookback_hours,
    }


def concurrency_verdict(
    survey: dict,
    kernels_needed: int = KERNELS_PER_INVOCATION,
    allowed_foreign: int = ALLOWED_IN_FLIGHT_FOREIGN_KERNELS,
) -> tuple[bool, str]:
    """Is the account idle enough? Returns (clear_to_launch, why not).

    Two separate questions, in this order, because they have different
    answers and conflating them is how a workflow ends up either competing
    with a human or refusing to use capacity nobody wants.

    1. **Is anyone else using the account?** Any foreign kernel at all and
       this job stands down, whatever the slot arithmetic says. That is
       stricter than Kaggle's cap requires and it is the policy; see
       ALLOWED_IN_FLIGHT_FOREIGN_KERNELS.
    2. **Are there enough free slots for the kernels this job pushes?** It
       pushes ``kernels_needed`` of them and the account cap is
       MAX_CONCURRENT_GPU_KERNELS, so its own leftovers count against it
       exactly as a stranger's would. A run that launched half its kernels
       would report half its legs and the control/canary comparison, which
       is the whole point of the pairing, would have nothing to compare.

    "No busy kernel was found" is only worth acting on if the search could
    actually have found one. An unanswerable question is a skip here, never
    a go-ahead: the cost of standing down is a few minutes until the next
    commit draws again, and the cost of guessing wrong is a push rejected at
    the capacity cap.
    """
    foreign = survey.get("foreign", survey["busy"])
    own = survey.get("own", [])
    if len(foreign) > allowed_foreign:
        return False, (
            f"the Kaggle account has {len(foreign)} kernel(s) in flight that "
            f"are not this workflow's, and this job tolerates "
            f"{allowed_foreign}: {', '.join(foreign)}. The account is shared "
            f"with human use and CI yields to it; standing down rather than "
            f"queueing."
        )
    free = MAX_CONCURRENT_GPU_KERNELS - len(foreign) - len(own)
    if free < kernels_needed:
        return False, (
            f"this job pushes {kernels_needed} kernel(s) and only {free} of "
            f"the account's {MAX_CONCURRENT_GPU_KERNELS} slot(s) are free "
            f"({len(own)} already held by this workflow, {len(foreign)} by "
            f"something else). A partial launch would report a subset of the "
            f"legs, and the control and canary legs are only worth anything "
            f"as a pair."
        )
    if not survey["complete"]:
        return False, (
            "the in-flight survey did not reach the end of its "
            f"{survey['window_hours']}h window within {MAX_KERNEL_PAGES} "
            "pages, so an older kernel of this account could still be "
            "running unseen"
        )
    if survey["surveyed"] and survey["unreadable"] == survey["surveyed"]:
        return False, (
            f"no kernel status could be read at all ({survey['unreadable']} "
            f"of {survey['surveyed']} unreadable), so whether the account is "
            "busy is unknown"
        )
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--percent", type = int, default = 10, help = "sampling rate when no override is present"
    )
    ap.add_argument("--run-id", default = os.environ.get("GITHUB_RUN_ID", "0"))
    ap.add_argument("--run-attempt", default = os.environ.get("GITHUB_RUN_ATTEMPT", "1"))
    ap.add_argument("--force", default = "false", help = "workflow_dispatch force input")
    ap.add_argument("--labels", default = "", help = "comma or newline separated PR labels")
    ap.add_argument("--label-name", default = "kaggle-t4-ci")
    ap.add_argument(
        "--budget-hours",
        type = float,
        required = True,
        help = "worst-case GPU hours this invocation can spend",
    )
    ap.add_argument(
        "--reserve-hours",
        type = float,
        default = 6.0,
        help = "quota CI refuses to dip into, left for humans",
    )
    ap.add_argument(
        "--kernels",
        type = int,
        default = KERNELS_PER_INVOCATION,
        help = "how many Kaggle kernels this invocation will push. "
        "The gate refuses unless that many slots are free",
    )
    ap.add_argument(
        "--allow-foreign-in-flight",
        type = int,
        default = ALLOWED_IN_FLIGHT_FOREIGN_KERNELS,
        help = "kernels NOT belonging to this workflow that may "
        "already be running and this job still launch. "
        "Default 0: the account is shared with human use and "
        "CI yields to it. See "
        "ALLOWED_IN_FLIGHT_FOREIGN_KERNELS before raising it",
    )
    ap.add_argument(
        "--soft-fail",
        action = "store_true",
        default = True,
        help = "treat a gate error as a skip rather than a failure",
    )
    ap.add_argument("--no-soft-fail", dest = "soft_fail", action = "store_false")
    args = ap.parse_args()

    override = args.force.strip().lower() in ("true", "1", "yes")
    labels = [l.strip().lower() for l in args.labels.replace("\n", ",").split(",") if l.strip()]
    if args.label_name.lower() in labels:
        override = True
        print(f"[gate] override: label {args.label_name!r} present", flush = True)

    # The draw is reported even when overridden, so the log always shows what
    # the unforced answer would have been.
    # Re-runs of the same run must not reroll, so run_attempt is excluded.
    picked, draw = sampled_in(str(args.run_id), args.percent)
    print(
        f"[gate] sampling draw={draw} threshold={args.percent} "
        f"picked={picked} (run {args.run_id}, attempt {args.run_attempt})",
        flush = True,
    )

    if not override and not picked:
        return _decide(
            False,
            f"not sampled this time (draw {draw} of 100, "
            f"threshold {args.percent}); this is the normal "
            f"outcome for roughly {100 - args.percent}% of "
            f"invocations",
        )

    if not os.environ.get("KAGGLE_API_TOKEN"):
        return _decide(
            False,
            "KAGGLE_API_TOKEN is not available to this "
            "context (expected on a fork pull request, "
            "where secrets are withheld)",
        )

    try:
        api = kaggle_client()
    except BaseException as exc:  # noqa: BLE001
        if isinstance(exc, KeyboardInterrupt):
            raise
        msg = f"could not authenticate to Kaggle: {type(exc).__name__}"
        if not args.soft_fail:
            print(f"[gate] {msg}", flush = True)
            return 1
        return _decide(False, msg)

    try:
        quota = remaining_gpu_hours(api)
    except Exception as exc:  # noqa: BLE001
        quota = {"ok": False, "error": type(exc).__name__}
    print("[gate] quota " + json.dumps(quota), flush = True)
    _out("quota", json.dumps(quota))

    if quota.get("ok"):
        need = args.budget_hours + args.reserve_hours
        if quota["remaining_hours"] < need:
            return _decide(
                False,
                f"insufficient weekly GPU quota: {quota['remaining_hours']}h "
                f"remaining of {quota['total_hours']}h, and this run needs up "
                f"to {args.budget_hours}h on top of a {args.reserve_hours}h "
                f"reserve. Quota refreshes at {quota.get('refresh_at')}",
            )
    else:
        # An unreadable quota is not permission to spend it.
        return _decide(
            False,
            "could not read the Kaggle accelerator quota, so the remaining budget is unknown",
        )

    try:
        survey = survey_kernels(api)
    except Exception as exc:  # noqa: BLE001
        return _decide(
            False,
            "could not list this account's kernels "
            f"({type(exc).__name__}), so concurrency cannot "
            "be established",
        )
    print(
        "[gate] concurrency "
        + json.dumps(
            {k: v for k, v in survey.items() if k not in ("busy", "own", "foreign")}
            | {
                "busy": len(survey["busy"]),
                "own": len(survey["own"]),
                "foreign": len(survey["foreign"]),
            }
        ),
        flush = True,
    )

    clear, why_not = concurrency_verdict(survey, args.kernels, args.allow_foreign_in_flight)
    if not clear:
        return _decide(False, why_not)

    why = (
        "forced by override"
        if override
        else f"sampled in (draw {draw} of 100, threshold {args.percent})"
    )
    return _decide(
        True,
        f"{why}; {quota['remaining_hours']}h of GPU quota "
        f"remaining and {args.kernels} of the account's "
        f"{MAX_CONCURRENT_GPU_KERNELS} kernel slots are free",
    )


if __name__ == "__main__":
    raise SystemExit(main())
