# What the Kaggle GPU budget buys, and where the sampling percentages come from

The two workflows both spend one shared Kaggle account's weekly GPU quota. The
percentages in them are not preferences; they are the output of the arithmetic
below. This file is here so the next person to change a percentage changes it
against measurements rather than against a guess.

Measured 2026-08-11 against `unslothai/unsloth`, 28-day window from 2026-07-14.

## The demand side

| quantity | measured |
|---|---|
| commits to `main` | 985 in 28 days, ~35/day, ~246/week |
| PRs created | 1157 in 28 days, ~41/day, ~289/week |
| PRs touching watched paths | 8 of 60 sampled, **13%** |

"Watched paths" is what the notebook CI is actually about: `unsloth/`,
`unsloth_zoo`, `pyproject.toml`, `tests/kaggle/`,
`.github/workflows/kaggle*`. A PR that only edits Studio frontend or docs
cannot regress a T4 training run, and spending a GPU session on it buys
nothing.

So the events that COULD justify a run are roughly `289 * 0.13 ~= 38
PRs/week`, plus pushes to main at a similar filtered rate. Call it ~75
candidate events per week.

## The supply side

Kaggle gives one account **30 GPU-hours/week**, and the quota is shared
between the two workflows and any manual probing. Measured session costs:

| kernel | wall clock |
|---|---|
| notebook CI kernel (2 legs, 2xT4) | ~35-50 min |
| Studio GPU kernel | ~30-40 min |

A session bills its wall clock once, not per card, so a 2-leg kernel is one
charge. Round to **0.75 GPU-h per launch**.

Reserving ~5h/week for manual probing and reruns leaves ~25h, which is
**~33 launches/week** across both workflows.

## The split, and why the percentages look inverted

33 launches against ~75 candidate events is a **~44% ceiling** if everything
went to one workflow. It does not: the notebook CI takes the larger share
because it covers more surface (training, inference, export, four library
sets), and Studio takes the smaller.

The percentages in the workflows are per-event sampling rates applied AFTER
the path filter, which is why they read lower than 44% looks:

* notebook CI at `--percent 15` with `--reserve-hours 25`
* Studio at `--percent 5` with `--reserve-hours 10`

15% of ~75 filtered events is ~11 launches/week, ~8 GPU-h. 5% is ~4
launches/week, ~3 GPU-h. Together ~11 GPU-h/week against a 25h allowance,
which leaves headroom for the reruns that a red run always causes and for the
weeks when merge volume spikes.

The reserve-hours figures differ because they are floors, not shares: each
workflow stands itself down when the account's remaining quota drops below
its own reserve, and the notebook CI reserves more because its sessions are
longer and it is the one that must not be starved.

## What would change these numbers

* A second Kaggle account doubles supply and the percentages could roughly
  double with it.
* The path filter is the biggest lever on demand. It is at 13% now; widening
  the watched set is what makes the sampling rate feel too low.
* If session cost rises above ~1 GPU-h (more legs per kernel, bigger models),
  re-derive rather than assuming the percentages still hold.
