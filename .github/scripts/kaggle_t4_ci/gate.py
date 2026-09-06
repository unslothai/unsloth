# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Decide whether this invocation is allowed to spend Kaggle GPU quota.

Each Kaggle account CI may spend has a WEEKLY accelerator budget shared with
every other use of that account, and a kernel on every push would drain a week
of it in a day. So the default answer is "no" and a job earns "yes" through five
checks, in this order:

0. **Was this event a request at all?** A run started by APPLYING a label is a
   request only when that label is the opt-in one. The trigger fires on all
   labels and the budget counts none of them, so every other ``labeled`` event
   stands down first.
1. **Override.** ``workflow_dispatch`` with ``force=true``, or a pull request
   carrying the opt-in label. A human asked; skip the dice.
2. **Sampling.** Roughly one invocation in ten, derived from the run id so a
   re-run of the same run gives the same answer (otherwise anyone could reroll
   until it fires) while different runs stay independent.
3. **Which account.** There is more than one, with different weekly
   allowances, and traffic is split in proportion to them: an account with
   twice the hours takes twice the runs. The weights are READ from each
   account's own ``quota_view()`` rather than written down here, so a plan
   changed on Kaggle's side changes the split without anyone editing this file.
   The draw is keyed on the run id ALONE -- not the attempt -- so a re-run
   picks the SAME account as the attempt whose kernels may still be in flight.
   An account that cannot take the run (no quota, no free session, no readable
   answer) hands over to the other rather than standing the run down; see
   FALLBACK_ELIGIBLE.
4. **Remaining quota.** Refuses when what is left would not cover this
   invocation's worst case plus a reserve, so CI never drains the account. This
   is the ONE stand-down that is a failure rather than a skip; see below. The
   reserve SCALES with the account: a flat 20h held back from a 30h account is
   two thirds of it and one third of a 60h one, which would quietly make the
   smaller account the stricter one. See ``scaled_reserve``.
5. **Concurrency.** Kaggle caps concurrent batch GPU kernels at 2, per ACCOUNT
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

# The accounts CI may spend, in order. The env var names are the SECRET names
# the workflows pass through; the account id is the position, which is what the
# concurrency group and the job summary use.
#
# Deliberately not carrying each account's weekly hours: they are read from
# `quota_view()` per run, so a plan changed on Kaggle's side changes the traffic
# split with nothing here to update and nothing here to go stale.
DEFAULT_ACCOUNT_ENVS = ("KAGGLE_API_TOKEN", "KAGGLE_API_TOKEN_2")

# `--reserve-hours` was calibrated against an account of this size. See
# `scaled_reserve`: the reserve is a FRACTION of the plan, not a fixed number of
# hours, or the smaller account keeps back proportionally more.
DEFAULT_RESERVE_BASIS_HOURS = 60.0

# An account answering with one of these hands over to the next one instead of
# standing the whole run down: none of them says anything about the code under
# test, and the other account may be perfectly able to run it.
#
# `credential_absent` is here so a repo holding one secret still works, which is
# also what a fork sees. What is NOT here is every decision made ABOVE the
# account -- not sampled, wrong label, bad input -- because those are answers
# about the run, and trying a second account would be answering a question
# nobody asked.
FALLBACK_ELIGIBLE = frozenset(
    {
        "credential_absent",
        "auth_failed",
        "username_unreadable",
        "quota_unreadable",
        "insufficient_quota",
        "capacity_occupied",
        "capacity_unreadable",
    }
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


def kaggle_client(token: str | None = None):
    """An authenticated client, optionally for a NAMED account's token.

    `authenticate()` reads the credential from the environment, so selecting an
    account means putting that account's token there for the length of the call
    and putting back whatever was there before. The restore is not tidiness: the
    launcher further down this job reads `KAGGLE_API_TOKEN` for itself, and a
    probe that left the last-tried account's token behind would hand the pushing
    step a different account from the one this gate cleared.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    if token is None:
        api = KaggleApi()
        api.authenticate()
        return api

    previous = os.environ.get("KAGGLE_API_TOKEN")
    os.environ["KAGGLE_API_TOKEN"] = token
    try:
        api = KaggleApi()
        api.authenticate()
    finally:
        if previous is None:
            os.environ.pop("KAGGLE_API_TOKEN", None)
        else:
            os.environ["KAGGLE_API_TOKEN"] = previous
    return api


def client_username(api) -> str | None:
    """The account this client authenticated AS, from the client's own record.

    `_authenticate_with_access_token` introspects the token and stores the name
    it comes back with, so this is Kaggle's answer rather than ours. It is not a
    nicety: a kernel id carries its owner, so the pushing step needs the real
    username, and a hardcoded one silently belongs to whichever account happened
    to be first when it was written.
    """
    try:
        return api.config_values.get(api.CONFIG_NAME_USER) or None
    except Exception:  # noqa: BLE001
        return None


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


def scaled_reserve(reserve_hours: float, total_hours: float, basis_hours: float) -> float:
    """The reserve this account keeps for humans, in proportion to its size.

    `--reserve-hours` was calibrated against one account. Applied flat to a
    smaller one it is a much bigger bite: 20h held out of a 30h plan is two
    thirds of it against one third of a 60h plan, so the SMALLER account would
    silently become the stricter one and take even less traffic than its weight
    already gives it. Scaling keeps "what fraction is kept back" the constant,
    which is what the number was chosen to express.
    """
    if basis_hours <= 0 or total_hours <= 0:
        return reserve_hours
    return round(reserve_hours * (total_hours / basis_hours), 3)


def in_flight_for_commit(own_busy: list[str], head_sha: str, kind: str) -> str | None:
    """The ref of a busy kernel of ours already running THIS commit for THIS
    workflow, or None.

    Asked of every account the gate surveys, not only the one it picks: the
    draw is keyed on the commit, but a handover (the preferred account full,
    or unreadable) lands a retry on the other account, whose GPU job collects
    with only its own token and would see nothing in flight.
    """
    sha = (head_sha or "").strip().lower()
    if not sha or not kind:
        return None
    import launch  # noqa: PLC0415  (sibling script; loaded lazily to keep the gate importable alone)

    for entry in own_busy:
        ref = entry.split(" (", 1)[0]
        parsed = launch.parse_slug(ref)
        if not parsed or parsed.get("kind") != kind or not parsed.get("sha"):
            continue
        if sha.startswith(parsed["sha"]) or parsed["sha"].startswith(sha):
            return ref
    return None


def weighted_pick(key: str, weights: dict[str, float]) -> tuple[str, float]:
    """Deterministic weighted choice of account, keyed on the COMMIT under test
    (the run id when no commit is known).

    The commit rather than the run: a second Actions run on one commit (a
    label added while a sampled run is still out, a forced dispatch) must land
    on the account already holding that commit's kernel, or the collector on
    the other account sees nothing in flight and dispatches a duplicate.

    Not the attempt, for the same reason `sampled_in` excludes it and a sharper
    one: a re-run of a run whose kernels are still in flight must return to the
    SAME account, or the second attempt pushes to an account the first one is
    not watching and the first one's kernels are reaped by nobody.

    Salted differently from `sampled_in` so the two draws off one run id are
    independent -- without the salt, whether a run is sampled in would correlate
    with which account it lands on, and one account would quietly get a
    different SHARE of the forced runs than of the sampled ones.
    """
    ids = sorted(weights)
    total = sum(max(0.0, weights[i]) for i in ids)
    digest = hashlib.sha256(("account:" + key).encode("utf-8")).hexdigest()
    draw = (int(digest[:8], 16) % 1_000_000) / 1_000_000.0
    if not ids or total <= 0:
        return (ids[0] if ids else ""), draw
    cumulative = 0.0
    for account_id in ids:
        cumulative += max(0.0, weights[account_id]) / total
        if draw < cumulative:
            return account_id, draw
    return ids[-1], draw


def probe_account(
    account_id: str,
    env_name: str,
    *,
    budget_hours: float,
    reserve_hours: float,
    reserve_basis_hours: float,
) -> tuple[dict, object]:
    """What this account can tell us before a single kernel is pushed.

    Returns (record, client). The record is JSON-safe and goes in the log and
    the job summary; the client is kept only in memory, for the survey that runs
    on whichever account is actually chosen.

    Every unusable answer gets a CODE rather than a sentence, because the
    fallback branches on it. Parsing the human-readable reason would make the
    wording load bearing, and the wording is written for a person who did not
    cause the problem and cannot fix it.
    """
    record: dict = {"account": account_id, "env": env_name, "outcome": "ok"}
    token = os.environ.get(env_name)
    if not token:
        record["outcome"] = "credential_absent"
        record["detail"] = (
            f"{env_name} is not available to this context (expected on a fork "
            "pull request, where secrets are withheld)"
        )
        return record, None

    try:
        api = kaggle_client(token)
    except BaseException as exc:  # noqa: BLE001
        if isinstance(exc, KeyboardInterrupt):
            raise
        record["outcome"] = "auth_failed"
        record["error"] = type(exc).__name__
        record["detail"] = f"could not authenticate to Kaggle: {type(exc).__name__}"
        return record, None

    # Refused rather than defaulted. The launcher names the kernel's owner, and
    # a run that cannot say who it authenticated as cannot name it correctly --
    # which is exactly the state a hardcoded username hides.
    username = client_username(api)
    if not username:
        record["outcome"] = "username_unreadable"
        record["detail"] = (
            "could not determine which Kaggle account this token belongs to, so "
            "the kernel could not be pushed under an owner that can delete it"
        )
        return record, None
    record["user"] = username

    try:
        quota = remaining_gpu_hours(api)
    except Exception as exc:  # noqa: BLE001
        quota = {"ok": False, "error": type(exc).__name__}
    if not quota.get("ok"):
        record["outcome"] = "quota_unreadable"
        record["error"] = quota.get("error")
        record["detail"] = (
            "could not read the Kaggle accelerator quota, so the remaining budget is unknown"
        )
        return record, api

    record["quota"] = quota
    record["total_hours"] = quota["total_hours"]
    record["remaining_hours"] = quota["remaining_hours"]
    reserve = scaled_reserve(reserve_hours, quota["total_hours"], reserve_basis_hours)
    record["reserve_hours"] = reserve
    if quota["remaining_hours"] < budget_hours + reserve:
        record["outcome"] = "insufficient_quota"
    return record, api


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
    ap.add_argument(
        "--head-sha",
        default = "",
        help = "the commit under test; the account draw is keyed on it so every run of one "
        "commit lands on the account holding its kernel",
    )
    ap.add_argument(
        "--kind",
        default = "",
        help = "notebook or studio: with --head-sha, a kernel of this kind already running "
        "this commit on ANY account stands the run down instead of dispatching a duplicate",
    )
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
        help = "quota CI refuses to dip into, left for humans. Scaled to each "
        "account's own weekly total against --reserve-basis-hours",
    )
    ap.add_argument(
        "--reserve-basis-hours",
        type = float,
        default = DEFAULT_RESERVE_BASIS_HOURS,
        help = "the account size --reserve-hours was calibrated against, so the "
        "reserve stays the same FRACTION of a smaller plan",
    )
    ap.add_argument(
        "--account-env",
        action = "append",
        default = None,
        help = "env var holding an account's token; repeat for each account, in "
        f"order. Default: {', '.join(DEFAULT_ACCOUNT_ENVS)}",
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

    account_envs = [e.strip() for e in (args.account_env or DEFAULT_ACCOUNT_ENVS) if e.strip()]
    if not any(os.environ.get(e) for e in account_envs):
        return _decide(
            False,
            "no Kaggle credential is available to this context (expected on a "
            "fork pull request, where secrets are withheld). Looked for: "
            + ", ".join(account_envs),
        )

    # Before the first network call, and globally: authenticate(), quota_view()
    # and every status call below go through a client with no timeout of its
    # own, and a stalled one would outlive this job's timeout-minutes. With it,
    # a stall raises and the handlers below turn it into a skip. See
    # SOCKET_TIMEOUT_SEC.
    socket.setdefaulttimeout(SOCKET_TIMEOUT_SEC)

    # Every account's quota is read BEFORE the draw, because the draw is
    # weighted by what those calls report. Two quota calls, no session, and the
    # weights are then a measurement rather than a number somebody typed.
    probes: dict[str, dict] = {}
    clients: dict[str, object] = {}
    for index, env_name in enumerate(account_envs, start = 1):
        account_id = str(index)
        try:
            record, api = probe_account(
                account_id,
                env_name,
                budget_hours = args.budget_hours,
                reserve_hours = args.reserve_hours,
                reserve_basis_hours = args.reserve_basis_hours,
            )
        except BaseException as exc:  # noqa: BLE001
            if isinstance(exc, KeyboardInterrupt):
                raise
            record, api = (
                {
                    "account": account_id,
                    "env": env_name,
                    "outcome": "auth_failed",
                    "error": type(exc).__name__,
                },
                None,
            )
        probes[account_id] = record
        clients[account_id] = api
    print("[gate] accounts " + json.dumps(list(probes.values())), flush = True)
    _out("accounts", json.dumps(list(probes.values())))

    # An account whose quota could not be read has no weight, because a weight
    # is what its plan says and we did not get to hear it. It stays a CANDIDATE
    # -- the order below still reaches it -- so an unreadable answer costs the
    # account its share of the traffic and not its place in the queue.
    weights = {i: p["total_hours"] for i, p in probes.items() if p.get("total_hours")}
    account_key = (args.head_sha or "").strip().lower() or str(args.run_id)
    sampled_account, account_draw = weighted_pick(account_key, weights)
    if weights:
        share = max(0.0, weights.get(sampled_account, 0.0)) / sum(weights.values())
        print(
            f"[gate] account draw={account_draw:.6f} sampled={sampled_account} "
            f"p={share:.3f} weights=" + json.dumps({i: weights[i] for i in sorted(weights)}),
            flush = True,
        )
    else:
        print(
            f"[gate] account draw={account_draw:.6f} sampled={sampled_account or '(none)'} "
            "with NO readable weights, so this is the declaration order rather "
            "than a weighted choice",
            flush = True,
        )

    # The sampled account first, then the rest in declaration order. Only the
    # account actually being considered pays for a survey, which is the
    # expensive call here.
    order = [sampled_account] + [i for i in probes if i != sampled_account]
    handovers: list[str] = []
    surveys: dict[str, dict] = {}

    def _survey(account_id: str) -> dict:
        if account_id not in surveys:
            surveys[account_id] = survey_kernels(clients[account_id])
        return surveys[account_id]

    # EVERY candidate account first, for THIS commit, before any is chosen. An
    # earlier run can have handed this commit to the other account; a retry
    # that finds the preferred account free again would otherwise pick it and
    # never look at the account whose kernel is still running.
    if args.head_sha and args.kind:
        for account_id in order:
            if not account_id or probes[account_id]["outcome"] != "ok":
                continue
            try:
                survey = _survey(account_id)
            except Exception:  # noqa: BLE001
                continue  # reported below, where the account is considered
            already = in_flight_for_commit(survey["own"], args.head_sha, args.kind)
            if already:
                return _decide(
                    False,
                    f"a {args.kind} kernel for this commit is already running on account "
                    f"{account_id} ({already}); its result arrives as the commit status, so "
                    "nothing is dispatched",
                )

    for account_id in order:
        if not account_id:
            continue
        record = probes[account_id]
        if record["outcome"] != "ok":
            handovers.append(f"account {account_id} {record['outcome']}")
            continue

        try:
            survey = _survey(account_id)
        except Exception as exc:  # noqa: BLE001
            record["outcome"] = "capacity_unreadable"
            record["error"] = type(exc).__name__
            handovers.append(
                f"account {account_id} kernels could not be listed ({type(exc).__name__})"
            )
            continue
        print(
            f"[gate] concurrency account={account_id} "
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
            record["outcome"] = "capacity_occupied"
            record["detail"] = why_not
            handovers.append(f"account {account_id} {why_not}")
            continue

        quota = record["quota"]
        _out("quota", json.dumps(quota))
        _out("account", account_id)
        _out("account_env", record["env"])
        _out("account_user", record["user"])
        # A ONE-ELEMENT matrix, and the token is not in it. The GPU job indexes
        # the secrets context with `secret_name`, which is the only shape that
        # cannot end up holding a different account's token than the metadata
        # beside it claims. See the workflow.
        _out(
            "matrix",
            json.dumps(
                {
                    "include": [
                        {
                            "account_id": account_id,
                            "secret_name": record["env"],
                            "kaggle_user": record["user"],
                            "weekly_hours": quota["total_hours"],
                            "reserve_hours": record["reserve_hours"],
                        }
                    ]
                }
            ),
        )

        why = (
            "forced by override"
            if override
            else f"sampled in (draw {draw} of 100, threshold {args.percent})"
        )
        moved = ""
        if account_id != sampled_account and handovers:
            moved = f"; sampled account {sampled_account} handed over ({'; '.join(handovers)})"
        return _decide(
            True,
            f"{why}; running on account {account_id} ({record['user']}) with "
            f"{quota['remaining_hours']}h of its weekly {quota['total_hours']}h "
            f"left and {args.kernels} of that account's "
            f"{MAX_CONCURRENT_GPU_KERNELS} kernel slots free{moved}",
        )

    # Nobody could run. WHICH stand-down this is depends on what every account
    # said, and only one of the answers is a failure: an account out of hours is
    # a fact about the week, while an unreadable one is a fact about the minute.
    # "Unknown" is not "exhausted", so a single unreadable account keeps the
    # whole verdict green -- the same rule the one-account gate applied, now over
    # a set.
    #
    # An account with no credential is not a candidate at all. It is what a fork
    # sees, and what a repo holding one of the two secrets sees, so counting it
    # as "not exhausted" would turn the exhausted RED into a green skip for
    # everyone with a single account configured.
    candidates = {i: p for i, p in probes.items() if p["outcome"] != "credential_absent"}
    outcomes = {i: p["outcome"] for i, p in candidates.items()}

    if outcomes and all(o == "insufficient_quota" for o in outcomes.values()):
        detail = "; ".join(
            f"account {i} has {candidates[i]['remaining_hours']}h of its weekly "
            f"{candidates[i]['total_hours']}h left against a "
            f"{candidates[i]['reserve_hours']}h reserve"
            for i in sorted(candidates)
        )
        first = candidates[sorted(candidates)[0]]
        return _decide(
            False,
            f"{QUOTA_EXHAUSTED_MESSAGE}. "
            f"{detail}, and this run needs up to {args.budget_hours}h on top of "
            f"that reserve. Quota refreshes at {first['quota'].get('refresh_at')}",
            exit_code = 0 if exhaustion_is_soft else 1,
        )

    if not errors_are_skips and any(
        o in ("auth_failed", "username_unreadable") for o in outcomes.values()
    ):
        print("[gate] no account could be authenticated: " + json.dumps(outcomes), flush = True)
        return 1

    # The per-account sentences, not the codes: these are read by whoever opened
    # the pull request, who did not cause any of this and cannot fix it. With one
    # account configured this reads exactly as the single-account gate did.
    details = [
        probes[i].get("detail") or probes[i]["outcome"]
        for i in sorted(probes)
        if probes[i]["outcome"] != "ok"
    ]
    return _decide(False, "; ".join(details) or "; ".join(handovers))


if __name__ == "__main__":
    raise SystemExit(main())
