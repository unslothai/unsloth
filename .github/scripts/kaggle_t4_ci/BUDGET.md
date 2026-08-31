# What the Kaggle GPU budget buys, and where the sampling percentage comes from

One workflow spends this account's weekly GPU quota today:
`.github/workflows/kaggle-t4-notebook-ci.yml`. The percentage in it is not a
preference; it is the output of the arithmetic below. This file is here so the
next person to change it changes it against measurements rather than against a
guess.

**The workflow header is the source of truth.** The BUDGET block at the top of
`kaggle-t4-notebook-ci.yml` carries the same arithmetic beside the settings it
justifies, so it cannot drift from them silently the way this file can. If the
two ever disagree, the workflow is right and this file is stale; fix this file.

Measured 2026-08-11 against `unslothai/unsloth`, 7-day window, or on real
Kaggle sessions. None of it is estimated.

## The demand side

| quantity | measured |
|---|---|
| commits to `main` | 479/week |
| ... touching the paths filter | 45 (9.4%) |
| PRs opened | 567/week |
| ... touching the paths filter | 7.5% (9 of a 120 sample), so 43/week |
| commits carried by those PRs | 4.33 each |

"Watched paths" is what the notebook CI is actually about, and it is exactly
the `paths:` list in the workflow: `unsloth/**`, `tests/kaggle/**`,
`.github/scripts/kaggle_t4_ci/**`, the workflow file itself, and
`pyproject.toml` (the payloads install the commit under test as a
distribution, so how it is built and what it depends on is part of what this
tests). A PR that only edits Unsloth frontend or docs cannot regress a T4
training run, and spending a GPU session on it buys nothing.

Eligible invocations per week:

| event | count |
|---|---|
| push to main | 45 |
| pull_request opened | 43 |
| pull_request synchronize | 0 .. 143 |
| **total** | **88 .. 231** |

`synchronize` is one event per push after the first and is bounded above by
the commit count those PRs carry.

`labeled` is a trigger too and contributes nothing to this table, which is a
property of the gate rather than of the trigger. GitHub fires that action for
EVERY label, so without the check it would add one draw per label applied to
an eligible PR -- and once `kaggle-t4-ci` is present, one FORCED session per
label, since the label stays in the list the override reads. The gate stands a
`labeled` run down unless the label that arrived is the opt-in one, so the
only label activity that costs anything is the label that is a request for it.
Do not subscribe to another activity type without asking what it does to this
table.

## The supply side

Kaggle gives this account **60 GPU-hours/week** at time of writing. Kaggle's
documented baseline is 30h; the surplus is a discretionary "floating"
allowance that can be withdrawn, so treat 30h as the number that is
guaranteed. This workflow is allotted **40 GPU-h/week** of it.

Measured per-leg durations, on run `32607621452`:

| leg | duration |
|---|---|
| gptoss | 384.1 s |
| frontier | 312.2 s |
| canary | 265.3 s |
| control | 262.2 s |

All four now ride in ONE kernel, two at a time, one worker per card taking its
next leg when the previous one exits. Packed longest-first that is
384.1 + 262.2 = 646.3 s on one card against 312.2 + 265.3 = 577.5 s on the
other, so:

| kernel | wall clock |
|---|---|
| one kernel, four legs | 0.18 h |
| **one invocation** | **~0.25 h** (envelope, see below) |

A session bills its wall clock once, not per card. That used to mean the second
T4 of each of two kernels was free, and it is why `frontier` was described as
costing nothing to carry. With one kernel it means something narrower: the two
cards are free relative to each other while both are busy, and the 68.8 s tail
where only one card still has work costs no more than the rest of the session.

**This shape is not a quota optimisation.** Two kernels of two legs measured
0.10 h + 0.13 h = 0.23 h against 0.18 h here, which is within the rounding.
What two kernels cost was the whole ACCOUNT: they took both of Kaggle's
concurrent sessions, so `kaggle-t4-studio-gpu-ci.yml`, which shares this
account, could not push at all and queued behind the entire notebook job
(measured: Unsloth run `32607617804` waited ~40 minutes on notebook run
`32607621452`). One kernel leaves the second session for Unsloth, and the two
workflows now hold separate GitHub concurrency groups so they can use it.

The **~0.25 h** envelope below is deliberately NOT lowered to the measured
0.18 h. Every figure in this document is derived from it, and the real cost
moved down, so it remains a true upper bound; re-deriving the whole budget to
book a 0.07 h saving would only make the reserve thinner.

## The sampling rate

Solve at the pessimistic end of the eligible range, targeting 30h rather than
the full 40 so an unusually busy week does not spend the allowance before the
quota floor has to intervene:

```
231 x r x 0.25 h = 30 h   ->   r = 0.52, set to 40%
```

So the workflow runs `--percent 40` with `--reserve-hours 20`. Expected spend
at 40%:

| week | invocations | spend |
|---|---|---|
| quiet | 88 x 0.40 | 8.8 GPU-h |
| busy | 231 x 0.40 | 23.1 GPU-h |

against the 40 GPU-h allowance: 15% to 39% of the 60h account.

## Why the reserve, and not just the rate

The rate sets the EXPECTED spend. The reserve sets the CEILING. Against a 60h
account, refusing to start below 20h remaining means CI can never have spent
more than 40h in a week, whatever the arithmetic above got wrong. Raise
`--reserve-hours` to throttle CI harder; do not raise it above roughly 45 or
CI will never run at all on a week with any other usage.

The worst case, if every sampled launch ran to the kernel ceiling, is far
above the allowance and is not what controls the spend. The reserve is.

`--budget-hours` is that worst case, and it is DERIVED rather than chosen:
`launch.py`'s constants bound one invocation at about 13800s of wall clock
(the push retries and the `_discard()` each one pays, the shared polling
deadline, `EVIDENCE_BUDGET_SEC`, and `release()` reconciling every slug
filed), and this workflow pushes ONE session, billing its wall clock once. So
1 x 13800s = 3.8 GPU-h, set to 4.

It was `2 x 13800s = 7.7 GPU-h, set to 8` while the legs travelled as two
kernels. The multiplier is how many sessions THIS invocation pushes, not
Kaggle's per-account cap of 2: the other slot may be Unsloth's, and Unsloth
reserves against its own budget rather than this one.
`test_the_reserved_budget_covers_every_billable_launcher_phase` recomputes it
from `launch.py` and `--kernels`; do not edit the number here or in the
workflow without changing what it is derived from.

## What would change these numbers

* A second Kaggle account doubles supply and the percentage could roughly
  double with it.
* The path filter is the biggest lever on demand. Widening the watched set is
  what makes the sampling rate feel too low.
* **The rate is set for THIS payload set and does not survive a change to it.**
  Wiring the `grpo` leg would roughly double kernel 2 and put a busy week over
  the allowance, so that change comes with a recomputation of this file and of
  the workflow header, not just a line in `legs.KERNELS`.
* If a second workflow ever starts spending this account, the split has to be
  derived here first. There is no second consumer today, and the reserve is
  sized on that.
