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
tests). A PR that only edits Studio frontend or docs cannot regress a T4
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

## The supply side

Kaggle gives this account **60 GPU-hours/week** at time of writing. Kaggle's
documented baseline is 30h; the surplus is a discretionary "floating"
allowance that can be withdrawn, so treat 30h as the number that is
guaranteed. This workflow is allotted **40 GPU-h/week** of it.

Measured session cost, on kernels `066cd463` (control + canary, 263s of
payload) and `8161ceb9` / `7ab727f1` (gpt-oss, 375s of payload):

| kernel | wall clock |
|---|---|
| kernel 1 (control + canary) | 0.10 h |
| kernel 2 (gptoss + frontier) | 0.13 h |
| **one invocation** | **~0.25 h** |

A session bills its wall clock once, not per card, so the second T4 of each
kernel is free. That is why `frontier` costs nothing to carry.

## The sampling rate

Solve at the pessimistic end of the eligible range, targeting 30h rather than
the full 40 so an unusually busy week does not spend the allowance before the
quota floor has to intervene:

```
231 x r x 0.25 h = 30 h   ->   r = 0.52, set to 40%
```

So the workflow runs `--percent 40` with `--reserve-hours 20` and
`--budget-hours 2`. Expected spend at 40%:

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

The worst case, if every sampled launch ran to both kernel ceilings, is far
above the allowance and is not what controls the spend. The reserve is.

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
