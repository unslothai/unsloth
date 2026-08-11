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
   cap is per ACCOUNT, not per workflow. If anything of this account's is
   already QUEUED or RUNNING, this invocation stands down rather than
   queueing behind it or failing on a push rejection.

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

# Kaggle's cap on simultaneous batch (committed) GPU kernels per account.
# Measured, not documented: exceeding it fails the push with
# "Maximum batch GPU session count of 2 reached."
MAX_CONCURRENT_GPU_KERNELS = 2

# Kernel states that mean a session is occupying one of those slots.
BUSY_STATES = {"QUEUED", "RUNNING"}

# How many of the account's most recently run kernels to status-check. The
# listing is sorted by last run time, so anything still running is at the
# top; this bounds the API calls without bounding the correctness in
# practice.
RECENT_KERNELS_TO_CHECK = 12


def _out(key: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"{key}={value}\n")
    print(f"[gate] {key}={value}", flush=True)


def _summary(text: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


def _decide(run: bool, reason: str) -> int:
    _out("should_run", "true" if run else "false")
    _out("reason", reason)
    verdict = "RUN" if run else "SKIP"
    print(f"[gate] {verdict}: {reason}", flush=True)
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


def busy_kernels(api, limit: int = RECENT_KERNELS_TO_CHECK) -> list[str]:
    """Refs of this account's kernels that are QUEUED or RUNNING.

    Kaggle exposes no "list my running sessions" call, so this walks the
    account's kernels most-recently-run first and asks each for its status.
    """
    busy: list[str] = []
    kernels = api.kernels_list(mine=True, page_size=limit, sort_by="dateRun")
    for kernel in kernels[:limit]:
        ref = getattr(kernel, "ref", None)
        if not ref:
            continue
        try:
            status = str(getattr(api.kernels_status(ref), "status", ""))
        except Exception as exc:  # noqa: BLE001
            # An unreadable status is not evidence of an idle account. Say so
            # and keep going; the caller treats any busy hit as blocking.
            print(f"[gate] status unreadable for {ref}: "
                  f"{type(exc).__name__}", flush=True)
            continue
        state = status.rsplit(".", 1)[-1].upper()
        if state in BUSY_STATES:
            busy.append(f"{ref} ({state})")
    return busy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--percent", type=int, default=10,
                    help="sampling rate when no override is present")
    ap.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "0"))
    ap.add_argument("--run-attempt",
                    default=os.environ.get("GITHUB_RUN_ATTEMPT", "1"))
    ap.add_argument("--force", default="false",
                    help="workflow_dispatch force input")
    ap.add_argument("--labels", default="",
                    help="comma or newline separated PR labels")
    ap.add_argument("--label-name", default="kaggle-t4-ci")
    ap.add_argument("--budget-hours", type=float, required=True,
                    help="worst-case GPU hours this invocation can spend")
    ap.add_argument("--reserve-hours", type=float, default=6.0,
                    help="quota CI refuses to dip into, left for humans")
    ap.add_argument("--soft-fail", action="store_true", default=True,
                    help="treat a gate error as a skip rather than a failure")
    ap.add_argument("--no-soft-fail", dest="soft_fail", action="store_false")
    args = ap.parse_args()

    override = args.force.strip().lower() in ("true", "1", "yes")
    labels = [l.strip().lower() for l in
              args.labels.replace("\n", ",").split(",") if l.strip()]
    if args.label_name.lower() in labels:
        override = True
        print(f"[gate] override: label {args.label_name!r} present", flush=True)

    # The draw is reported even when overridden, so the log always shows what
    # the unforced answer would have been.
    # Re-runs of the same run must not reroll, so run_attempt is excluded.
    picked, draw = sampled_in(str(args.run_id), args.percent)
    print(f"[gate] sampling draw={draw} threshold={args.percent} "
          f"picked={picked} (run {args.run_id}, attempt {args.run_attempt})",
          flush=True)

    if not override and not picked:
        return _decide(False, f"not sampled this time (draw {draw} of 100, "
                              f"threshold {args.percent}); this is the normal "
                              f"outcome for roughly {100 - args.percent}% of "
                              f"invocations")

    if not os.environ.get("KAGGLE_API_TOKEN"):
        return _decide(False, "KAGGLE_API_TOKEN is not available to this "
                              "context (expected on a fork pull request, "
                              "where secrets are withheld)")

    try:
        api = kaggle_client()
    except BaseException as exc:  # noqa: BLE001
        if isinstance(exc, KeyboardInterrupt):
            raise
        msg = f"could not authenticate to Kaggle: {type(exc).__name__}"
        if not args.soft_fail:
            print(f"[gate] {msg}", flush=True)
            return 1
        return _decide(False, msg)

    try:
        quota = remaining_gpu_hours(api)
    except Exception as exc:  # noqa: BLE001
        quota = {"ok": False, "error": type(exc).__name__}
    print("[gate] quota " + json.dumps(quota), flush=True)
    _out("quota", json.dumps(quota))

    if quota.get("ok"):
        need = args.budget_hours + args.reserve_hours
        if quota["remaining_hours"] < need:
            return _decide(
                False,
                f"insufficient weekly GPU quota: {quota['remaining_hours']}h "
                f"remaining of {quota['total_hours']}h, and this run needs up "
                f"to {args.budget_hours}h on top of a {args.reserve_hours}h "
                f"reserve. Quota refreshes at {quota.get('refresh_at')}")
    else:
        # An unreadable quota is not permission to spend it.
        return _decide(False, "could not read the Kaggle accelerator quota, "
                              "so the remaining budget is unknown")

    try:
        busy = busy_kernels(api)
    except Exception as exc:  # noqa: BLE001
        return _decide(False, "could not list this account's kernels "
                              f"({type(exc).__name__}), so concurrency cannot "
                              "be established")
    if busy:
        return _decide(
            False,
            f"the Kaggle account already has {len(busy)} kernel(s) in flight "
            f"and the cap is {MAX_CONCURRENT_GPU_KERNELS}: "
            f"{', '.join(busy)}. Standing down rather than queueing.")

    why = "forced by override" if override else \
        f"sampled in (draw {draw} of 100, threshold {args.percent})"
    return _decide(True, f"{why}; {quota['remaining_hours']}h of GPU quota "
                         f"remaining and no kernel of this account is running")


if __name__ == "__main__":
    raise SystemExit(main())
