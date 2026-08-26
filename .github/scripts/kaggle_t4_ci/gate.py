# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Decide whether this invocation is allowed to spend Kaggle GPU quota.

The account behind KAGGLE_ACCESS_TOKEN_GH has a WEEKLY accelerator budget shared
with every other use of that account, and a kernel on every push would drain a
week of it in a day. So the default answer is "no" and a job earns "yes" through
four checks, in this order:

0. **Was this event a request at all?** A run started by APPLYING a label is a
   request only when that label is the opt-in one. The trigger fires on all
   labels and the budget counts none of them, so every other ``labeled`` event
   stands down first.
1. **Override.** ``workflow_dispatch`` with ``force=true``, or a pull request
   carrying the opt-in label. A human asked; skip the dice.
2. **Sampling.** Roughly one invocation in ten, derived from the run id so a
   re-run of the same run gives the same answer (otherwise anyone could reroll
   until it fires) while different runs stay independent.
3. **Remaining quota.** Refuses when what is left would not cover this
   invocation's worst case plus a reserve, so CI never drains the account. This
   is the ONE stand-down that is a failure rather than a skip; see below.
4. **Concurrency.** Kaggle caps concurrent batch GPU kernels at 2, per ACCOUNT
   rather than per workflow. The survey splits kernels this workflow pushed
   from the rest because they call for opposite answers: a FOREIGN kernel means
   a human is on the shared account and this invocation stands down entirely,
   while our own two are what it is here to launch. Both occupy slots, so the
   gate also refuses when fewer than KERNELS_PER_INVOCATION are free. The search
   is bounded by how long a session may last rather than by a kernel count,
   which is what makes it exhaustive; see LOOKBACK_HOURS.

Every negative answer is a SKIP, and a skip exits 0: not spending quota is
designed behaviour and must never colour a pull request red. The ONE exception
is an EXHAUSTED weekly quota, which exits nonzero carrying
QUOTA_EXHAUSTED_MESSAGE: that is not a dice roll going the usual way, it is the
whole account out of accelerator hours until the refresh, and a reader who sees
nothing at all cannot tell that from a workflow nobody wired up. It is decided
before the concurrency survey and before any kernel is pushed, so it costs one
API call rather than a Kaggle session. An UNREADABLE quota is still a skip:
"unknown" is not "exhausted". A real error in the gate itself is a skip too, and
``--no-soft-fail`` turns that back into a failure.

``--soft-fail`` is a REQUEST, not the default state. Passed explicitly it turns
the exhausted-quota failure back into a skip, for a caller that has already been
approved and is only re-asking (the workflow's recheck step, which runs with the
account slot in hand); passed nowhere, the failure stands.

No credential is ever printed: the token is read from the environment, handed to
the Kaggle client, and never echoed, logged or written to an output file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from datetime import datetime, timedelta, timezone

# Ceiling on any single network call, set globally before the first one. The
# Kaggle client takes no timeout of its own and Python's default is to block
# FOREVER, so without this a stalled connection returns to nobody: the `try`
# around each call never sees an exception, --soft-fail has nothing to convert
# into a skip, and the job's own timeout-minutes kills the runner and reports a
# red infrastructure failure on a workflow whose contract is that only a failed
# assertion on a T4 is red. launch.py sets one for the same reason.
SOCKET_TIMEOUT_SEC = 60

# Wall clock the in-flight survey may spend, whatever the account holds. The
# survey status-checks every kernel in a 13h window and pages up to
# MAX_KERNEL_PAGES x KERNELS_PAGE_SIZE of them, so slow (not hung: those are
# capped above) responses multiply. Running out of budget is NOT read as an idle
# account: the walk stops with complete=False, which concurrency_verdict already
# turns into a skip, so the gate answers within the job deadline rather than
# being killed by it.
SURVEY_BUDGET_SEC = 180

# Kaggle's cap on simultaneous batch (committed) GPU kernels per account.
# Measured, not documented: exceeding it fails the push with
# "Maximum batch GPU session count of 2 reached."
MAX_CONCURRENT_GPU_KERNELS = 2

# How many FOREIGN kernels may be in flight and this job still launch. ZERO IS
# DELIBERATE: a policy choice rather than a technical limit, so it is named
# rather than implied.
#
# The cap is per ACCOUNT and the account is shared with human use. Someone whose
# push is rejected cannot tell that CI took the slot, and CI cannot give it
# back. Standing down costs a few minutes until the next commit draws again (the
# sampling gate leaves this job no deadline of its own); being wrong costs
# somebody else's session. So a single foreign kernel stands this job down,
# whatever the arithmetic below would allow.
ALLOWED_IN_FLIGHT_FOREIGN_KERNELS = 0

# How many of Kaggle's slots one invocation takes. ONE, and it is the default
# rather than a per-workflow override because it is now true of both callers.
#
# This was 2 while the notebook leg ran four legs as two kernels of two, one leg
# per T4. That shape took BOTH of the account's slots, which had two costs. The
# account is shared with human use, so a run held every seat there was. And
# kaggle-t4-studio-gpu-ci.yml is on the same account: with no slot left it could
# not push at all, and the shared concurrency group meant it did not even try
# until the notebook job had finished (measured: Unsloth's run 32607617804 queued
# about 40 minutes behind notebook run 32607621452).
#
# The legs did not have to be split to fit. They now queue INSIDE one kernel --
# one worker per card, a card taking its next leg when the previous one exits --
# so four legs fit in one session. That is strictly better for the thing the old
# comment here worried about: it said legs "split across two sessions would be
# compared across two images and two hours, and for control/canary that
# comparison is the entire instrument", and one kernel makes such a split
# impossible rather than merely avoided by careful pairing.
#
# The residual risk, a foreign kernel starting BETWEEN the survey and the push,
# is not new and is handled where it lands: the launcher recognises Kaggle's
# capacity rejection (CAPACITY_MARKERS) and reports it as infra, exiting 0.
# Ours is the push that arrives second, so a human's is never the one rejected.
KERNELS_PER_INVOCATION = 1

# Kernel states that mean a session is occupying one of those slots.
BUSY_STATES = {"QUEUED", "RUNNING"}

# How this job recognises its own kernels: `launch.py` pushes every kernel as
# `<user>/<OWN_KERNEL_PREFIX><8 hex>`, a fresh slug per attempt, and nothing else
# on the account uses that prefix. "The account is busy" and "this workflow is
# busy" call for opposite answers -- a foreign kernel means a human is using the
# account and this job must yield, while one of our own means a previous run is
# still in flight, which the concurrency group is supposed to prevent. Ours is
# reported separately rather than counted as a human, and still blocks, since it
# still occupies a slot.
OWN_KERNEL_PREFIX = "unsloth-t4-ci-"

# How far back the in-flight survey looks, and why that bound is COMPLETE
# rather than merely convenient.
#
# Kaggle exposes no "list my running sessions" call, so finding an in-flight
# kernel means listing kernels and status-checking them. The listing is sorted
# by last run time descending, and for an unfinished kernel that timestamp is
# when the run STARTED (measured: a kernel pushed at 10:05:19Z listed as
# last_run_time 10:05:19.297).
#
# Kaggle kills a notebook session at 12 hours (CPU/GPU; 9 for TPU), so a kernel
# still QUEUED or RUNNING cannot have started more than 12 hours ago, and once
# the walk reaches an older entry every remaining one is older still. Stopping
# there is exhaustive rather than a sample, which the previous fixed bound of
# "the 12 most recent kernels" was not: a kernel running for three hours would
# be missed once twelve newer ones had run, and the push would then fail at the
# capacity cap and be reported as infra.
MAX_SESSION_HOURS = 12.0

# Slack on top, for clock skew between Kaggle's timestamps and this runner,
# and for any future raise of the session ceiling.
CLOCK_SKEW_HOURS = 1.0
LOOKBACK_HOURS = MAX_SESSION_HOURS + CLOCK_SKEW_HOURS

# Paging for that walk. The page cap stops a pathological account making the
# gate walk forever; reaching it means the survey did NOT cover the whole
# window, reported as incomplete and treated as "unknown", never as "idle".
KERNELS_PAGE_SIZE = 100
MAX_KERNEL_PAGES = 5

# The two ways a status lookup can fail, only one of them benign. A deleted
# kernel stays in the listing for a while and answers its status call with a
# 404, which is not an unknown state but a kernel definitively not running;
# blocking on it would wedge the gate shut, since the launcher deletes every
# kernel it pushes. Anything else (a 5xx, a socket timeout, a client bug) says
# nothing about the kernel, which may be a human's RUNNING session, and
# proceeding could take the last slot from the person the zero-foreign policy
# protects. So 404 counts as "gone", everything else as "unreadable", and only
# the second stands the job down.
GONE_MARKERS = ("404", "not found", "notfound", "does not exist")

# What an exhausted weekly quota says, verbatim, in the log line and in the job
# summary. It is addressed to whoever opened the pull request, who did not cause
# it and cannot clear it: nothing was learned about their code, the hours come
# back on Kaggle's own refresh, and there is nothing for them to fix. The
# measured numbers are appended rather than replacing it, so the reader can see
# WHEN it clears; this sentence stays intact.
QUOTA_EXHAUSTED_MESSAGE = (
    "GPU capacity exhausted - please wait until next week - you can ignore this CI failure"
)


def _looks_gone(exc: BaseException) -> bool:
    """Did this status lookup fail because the kernel no longer exists?"""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in GONE_MARKERS)


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


def _decide(
    run: bool,
    reason: str,
    exit_code: int = 0,
) -> int:
    """Publish the answer and return the process exit code.

    ``exit_code`` defaults to 0 because almost every answer here is a skip. The
    one caller that passes 1 is the exhausted-quota branch; ``should_run`` is
    still written first either way, so a step reading the output sees "false"
    rather than an empty string.
    """
    _out("should_run", "true" if run else "false")
    _out("reason", reason)
    verdict = "RUN" if run else ("FAIL" if exit_code else "SKIP")
    print(f"[gate] {verdict}: {reason}", flush = True)
    _summary(f"### Kaggle T4 gate: {verdict}\n\n{reason}\n")
    return exit_code


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
    budget_sec: float = SURVEY_BUDGET_SEC,
    clock = time.monotonic,
) -> dict:
    """Status-check every kernel that could still be in flight.

    Walks the account's kernels most-recently-run first and stops at the first
    that started longer ago than any session may last; see LOOKBACK_HOURS for
    why that is exhaustive.

    Returns the busy refs split by ownership, plus enough bookkeeping to tell
    "nothing is running" from "the question could not be answered":

    ``busy`` / ``own`` / ``foreign``
        every in-flight kernel, and the two disjoint halves of it. A kernel is
        ours when its slug carries OWN_KERNEL_PREFIX, which ``launch.py`` pushes
        under and nothing else uses.
    ``complete``
        the walk ran off the end of the listing or reached an entry outside the
        window. False means the page cap or ``budget_sec`` stopped it first and
        some candidates were never looked at.
    ``out_of_budget``
        the walk stopped because it ran out of wall clock. Slow responses over
        hundreds of status calls are how the survey outlives the job's own
        timeout, and being killed there costs the runner and reports red; giving
        up inside the deadline reports an incomplete survey, which is a skip.
    ``surveyed`` / ``unreadable`` / ``gone``
        how many in-window kernels were status-checked; how many left their
        state genuinely unknown; and how many answered 404, a deleted kernel
        rather than an unknown one. One unreadable status is not evidence of an
        idle account, so it is counted apart from the benign kind. See
        GONE_MARKERS.
    """
    # Naive UTC, matching what Kaggle returns. utcnow() is the same and is
    # deprecated from 3.12.
    now = now or datetime.now(timezone.utc).replace(tzinfo = None)
    cutoff = now - timedelta(hours = lookback_hours)
    busy: list[str] = []
    own: list[str] = []
    foreign: list[str] = []
    surveyed = 0
    unreadable = 0
    gone = 0
    complete = False
    out_of_budget = False
    deadline = clock() + budget_sec

    for page in range(1, max_pages + 1):
        if clock() >= deadline:
            out_of_budget = True
            break
        kernels = (
            api.kernels_list(mine = True, page = page, page_size = page_size, sort_by = "dateRun") or []
        )
        for kernel in kernels:
            ref = getattr(kernel, "ref", None)
            if not ref:
                continue
            if clock() >= deadline:
                out_of_budget = True
                break
            last_run = _as_naive_utc(getattr(kernel, "last_run_time", None))
            # A missing timestamp says nothing about age, so it cannot end the
            # walk, but it can still be checked.
            if last_run is not None and last_run < cutoff:
                complete = True
                break
            surveyed += 1
            try:
                status = str(getattr(api.kernels_status(ref), "status", ""))
            except Exception as exc:  # noqa: BLE001
                # A 404 is a deleted kernel, so the slot is free. Any other
                # error leaves the state unknown, which is not evidence of an
                # idle account. See GONE_MARKERS.
                if _looks_gone(exc):
                    gone += 1
                    print(f"[gate] status 404 for {ref}: already deleted", flush = True)
                    continue
                unreadable += 1
                print(f"[gate] status unreadable for {ref}: " f"{type(exc).__name__}", flush = True)
                continue
            state = status.rsplit(".", 1)[-1].upper()
            if state in BUSY_STATES:
                entry = f"{ref} ({state})"
                busy.append(entry)
                # Ownership is read off the SLUG, the part after the username:
                # a foreign kernel belongs to the same user, so the user half
                # says nothing.
                slug = ref.rsplit("/", 1)[-1]
                (own if slug.startswith(OWN_KERNEL_PREFIX) else foreign).append(entry)
        if complete or out_of_budget:
            break
        if len(kernels) < page_size:
            # Ran off the end of the account's kernels, so nothing is left to
            # miss.
            complete = True
            break

    if out_of_budget:
        print(
            f"[gate] survey gave up after {budget_sec}s with {surveyed} kernel(s) checked",
            flush = True,
        )

    return {
        "busy": busy,
        "own": own,
        "foreign": foreign,
        "surveyed": surveyed,
        "unreadable": unreadable,
        "gone": gone,
        # An abandoned walk is never complete, whatever it saw on the way.
        "complete": complete and not out_of_budget,
        "out_of_budget": out_of_budget,
        "window_hours": lookback_hours,
    }


def concurrency_verdict(
    survey: dict,
    kernels_needed: int = KERNELS_PER_INVOCATION,
    allowed_foreign: int = ALLOWED_IN_FLIGHT_FOREIGN_KERNELS,
) -> tuple[bool, str]:
    """Is the account idle enough? Returns (clear_to_launch, why not).

    Two questions, in this order, because conflating them is how a workflow
    ends up either competing with a human or refusing capacity nobody wants.

    1. **Is anyone else using the account?** Any foreign kernel stands this job
       down, whatever the slot arithmetic says. Stricter than Kaggle's cap
       requires, and deliberate; see ALLOWED_IN_FLIGHT_FOREIGN_KERNELS.
    2. **Are there enough free slots?** It pushes ``kernels_needed`` against a
       cap of MAX_CONCURRENT_GPU_KERNELS, so its own leftovers count against it
       exactly as a stranger's would. A half launch would report half its legs,
       and the control/canary pair would have nothing to compare.

    "No busy kernel was found" is worth acting on only if the search could have
    found one, so an unanswerable question is a skip: standing down costs a few
    minutes until the next commit draws, guessing wrong costs a push rejected at
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
        ran_out = (
            "its wall-clock budget" if survey.get("out_of_budget") else f"{MAX_KERNEL_PAGES} pages"
        )
        return False, (
            "the in-flight survey did not reach the end of its "
            f"{survey['window_hours']}h window within {ran_out}, so an older "
            "kernel of this account could still be running unseen"
        )
    # ANY unreadable candidate, not just all of them: the one in-window kernel
    # that could not be read may be the human session this job yields to, and
    # "the ones we could read were idle" is no answer about it. Deleted kernels
    # answer 404 and count as `gone`, so the routine case does not wedge the
    # gate shut; see GONE_MARKERS.
    if survey.get("unreadable"):
        return False, (
            f"{survey['unreadable']} of {survey['surveyed']} in-window kernel "
            "status(es) could not be read, so whether the account is busy is "
            "unknown; standing down rather than assuming it is idle"
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
        "--event-action",
        default = "",
        help = "the pull_request action that started this run, if any",
    )
    ap.add_argument(
        "--event-label",
        default = "",
        help = "for a `labeled` action, the ONE label that was just applied",
    )
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
    # THREE states, not two, which is why this is store_const against a default
    # of None rather than store_true. An error in the gate is a skip whether or
    # not anyone asked (that has always been the default and stays it), but an
    # exhausted quota is a failure UNLESS a caller asked for soft failure, and
    # "the flag defaults to on" would make that request unaskable: every
    # invocation would look like it had been made and the red would never
    # appear. So None means "nobody said", True means "asked", False means
    # "--no-soft-fail", and the two questions read the value separately below.
    ap.add_argument(
        "--soft-fail",
        dest = "soft_fail",
        action = "store_const",
        const = True,
        default = None,
        help = "stand down rather than fail even when the weekly GPU quota is "
        "exhausted. For a caller already past the gate that is only "
        "re-asking; see the workflow's recheck step",
    )
    ap.add_argument(
        "--no-soft-fail",
        dest = "soft_fail",
        action = "store_const",
        const = False,
        help = "treat an error in the gate itself as a failure too",
    )
    args = ap.parse_args()

    # An error in the gate says nothing about the code under test, so it stays a
    # skip unless --no-soft-fail was passed.
    errors_are_skips = args.soft_fail is not False
    # Exhaustion is a fact about the account, and only an explicit request
    # softens it.
    exhaustion_is_soft = args.soft_fail is True

    label_name = args.label_name.strip().lower()

    # A LABEL EVENT IS A REQUEST ONLY IF IT IS THE OPT-IN LABEL, checked first
    # because it is what keeps the budget arithmetic true.
    #
    # The workflow subscribes to `labeled` so the opt-in label can start a run,
    # but GitHub fires that action for EVERY label. Without this, each unrelated
    # label (triage, size, whatever a bot applies) is a fresh run and a fresh
    # sampling draw, while the estimate at the top of the workflow counts only
    # pull request opens and pushes. Worse, once the opt-in label is present it
    # stays in the label list below, so every LATER label of any kind arrives as
    # an override and FORCES a session: two bot labels would spend two more.
    #
    # So a `labeled` run stands down unless the label that started it means "run
    # this". Every other action (opened, synchronize, reopened, a push, a
    # dispatch) is unaffected, and an opted-in pull request still forces on each.
    action = args.event_action.strip().lower()
    if action == "labeled":
        applied = args.event_label.strip().lower()
        if applied != label_name:
            return _decide(
                False,
                f"this run was started by applying the label {applied or '(unnamed)'!r}, "
                f"which is not the opt-in label {args.label_name!r}. Labelling a pull "
                f"request is not a request to spend a Kaggle session, and every label "
                f"would otherwise be one more draw -- or, once the opt-in label is "
                f"present, one more forced run",
            )
        print(f"[gate] started by the opt-in label {args.label_name!r}", flush = True)

    override = args.force.strip().lower() in ("true", "1", "yes")
    labels = [l.strip().lower() for l in args.labels.replace("\n", ",").split(",") if l.strip()]
    if label_name in labels:
        override = True
        print(f"[gate] override: label {args.label_name!r} present", flush = True)

    # Reported even when overridden, so the log shows what the unforced answer
    # would have been. run_attempt is excluded so a re-run cannot reroll.
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

    # Before the first network call, and globally: authenticate(), quota_view()
    # and every status call below go through a client with no timeout of its
    # own, and a stalled one would outlive this job's timeout-minutes. With it,
    # a stall raises and the handlers below turn it into a skip. See
    # SOCKET_TIMEOUT_SEC.
    socket.setdefaulttimeout(SOCKET_TIMEOUT_SEC)

    try:
        api = kaggle_client()
    except BaseException as exc:  # noqa: BLE001
        if isinstance(exc, KeyboardInterrupt):
            raise
        msg = f"could not authenticate to Kaggle: {type(exc).__name__}"
        if not errors_are_skips:
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
            # THE ONE RED STAND-DOWN, and it is answered here rather than after
            # the survey on purpose: one quota call, no kernel pushed, no
            # session spent to report that there are no sessions left. The
            # required sentence comes first and whole; the numbers behind it
            # follow so the reader can see when the hours come back.
            return _decide(
                False,
                f"{QUOTA_EXHAUSTED_MESSAGE}. "
                f"{quota['remaining_hours']}h of the weekly {quota['total_hours']}h "
                f"accelerator quota is left, and this run needs up to "
                f"{args.budget_hours}h on top of a {args.reserve_hours}h reserve. "
                f"Quota refreshes at {quota.get('refresh_at')}",
                exit_code = 0 if exhaustion_is_soft else 1,
            )
    else:
        # An unreadable quota is not permission to spend it -- and it is not
        # evidence of exhaustion either, so it stays a green skip.
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
