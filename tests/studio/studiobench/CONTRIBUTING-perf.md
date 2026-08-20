# Proving a Studio performance change

This is the protocol for turning "it feels faster" into a number a reviewer can trust. It exists
because an audit of 40 frontend pull requests found that **30 of them had no effect distinguishable
from a null control**, and several of those had looked like clear wins right up until a detection
floor was applied to them.

Read this before you quote a result. It is short, and every rule in it was written after a wrong
answer was produced by not having it.

## The loop

```
1. Screen    python -m tests.studio.studiobench --tier fast    ~5 min, direction only
2. Confirm   python -m tests.studio.studiobench --tier standard --reps 4    ~20 min
3. Gate      per-metric floor, sign consistency, stability
4. Parity    prove you did not change what is rendered
5. Read      python -m tests.studio.studiobench --report <out>/payload.jsonl --tier standard
```

Every step above except the last drives a real Studio and therefore needs credentials. The
commands are written short here; see **"You need a Studio, and you need its password"** in
[README.md](README.md) before you run the first one, because a missing `--password` fails as an
HTTP 401 only after the browser has started, and `--doctor` reports PASS on that configuration.
Step 5 needs neither Studio nor network.

Steps 3 and 4 are not optional extras. A number that has not cleared them is not evidence.

## 1. Screen on the fast tier

`--tier fast` is one rung (100K), a 57.3 second film, about 5 minutes, or about 9 minutes for a full
A/B wave. Its detection floor is wider than the standard tier's, so it can tell you a direction and
it cannot tell you a magnitude.

Use it to kill bad ideas cheaply. Do not use it to decide anything.

## 2. Confirm on the standard tier

`--tier standard --reps 4`. Three rungs, a 243 second film.

**Use at least 4 repetitions.** At `--reps 2` the null control reported a 38.1% delta on
`settings.open_ms`. At `--reps 4` the same metric collapsed to 1.2% with 48% spread. Two
repetitions is not enough to separate a mean from an order effect, because the harness flips arm
order on odd repetitions and a 1-rep or 2-rep session bakes in whichever order it happened to run.

If you shard a run, shard into **2 shards of 2 repetitions, not 4 shards of 1**. A single-repetition
session always runs base first.

## 3. The three gates

A result must clear all three. Failing any one makes it **void**, not "a small win".

### Gate 1: its own per-metric detection floor

You need a **null control**: the same build against itself, base versus base, producing a delta that
you know is exactly zero in truth. Whatever it reports is your noise.

Two rules about it, both learned the hard way:

- **The floor is per metric, not global.** A single global floor derived from the frame tail says
  nothing about whether `settings.open_ms` can resolve 5%. Each metric gets its own floor from the
  null control.
- **The floor must be measured in band**, concurrently with your A/B, under the same machine load.
  A floor measured on a quiet machine does not transfer to a busy one. Session-to-session drift on
  this metric set is about 8%, which is larger than most real effects, so a floor from yesterday's
  run is not a floor.

The floor for a metric is `max(|null delta|, spread)`. The null control's own systematic bias has to
be cleared as well as its scatter.

If you run waves, **each wave carries its own null control.** Wave 0's floor is not wave 1's floor.

### Gate 2: sign consistency

Every paired ratio must fall on the same side of 1.0. If three repetitions say faster and one says
slower, you have scatter that happens to average in your favour, not an effect.

### Gate 3: stability

The effect must exceed its own scatter. This gate rejected 12 rows that had passed gates 1 and 2 and
were junk.

## 4. Parity: prove you did not change what is rendered

A performance change that alters what the user sees is a different pull request with a different
review. Run the parity digest, which takes a normalised structural DOM signature per action window
on both arms and compares them.

What the digest **can** see: tag structure, `data-slot`, `data-state`, `data-role`, classes, text.
It normalises generated ids, rendered durations, relative times, scroll state, backend-minted record
ids, absolute URLs and blob/data URLs, so those do not read as false differences.

What it **cannot** see: stylesheet CSS, computed layout, colour, typography, or anything rastered.
So a change to `content-visibility`, `contain-intrinsic-size`, transforms or paint can be perfectly
parity-clean and still visibly wrong. **If you touched CSS, take screenshots as well.**

`NOT EXERCISED` is not a pass. An action that never ran is not evidence that it is stable.

## Before you trust any number at all

### Check that the thing you measured actually ran

The single most common way this harness has produced a wrong answer, three separate times in three
separate subsystems, is code that could never fire reporting as "no effect":

- Four scene actions recorded NOT RUN on 312 of 312 attempts, because their slots opened while a
  follow-up turn was still streaming. They read as fast, stable and meaningless.
- A surface crawler walked 53 surfaces that would all have digested the same persistently mounted
  chat root.
- An overlay walk was written in a way that could never fire at all.

Check `ran` before you read a timing. Check that a potency counter moved before you believe an
ablation.

### Do not measure a stale branch against current main

If your branch does not merge cleanly and the harness falls back to your raw head, your treatment
arm predates every commit that landed since you branched, and you are measuring branch age. One
pull request in the audit showed a +246% regression on the message menu that **vanished entirely**
when the branch was rebased. The regression was another PR's improvement missing from the old side.

For a change that has **already merged**, do not merge it onto main again: that is a no-op producing
a byte-identical bundle and a void that looks exactly like "this changed nothing". Measure it as
`merge commit` against `merge commit^1`.

### Know what the rung ladder can and cannot see

The rung varies the **seeded thread size**. The streamed reply is pinned at 6,000 characters on
every rung on purpose. A mechanism whose cost scales with **reply length** is therefore held
constant by this ladder and will read as a flat floor on it. If that is your mechanism, you need to
build a reply-length axis rather than quote a rung comparison that cannot see your effect.

Related: rank frames by **slope, not size**. `findNextMatchSync` is the second-largest frame at the
100K rung at 1,365 ms, and it is a floor, not a cause: it grows at 3.52x against a 3.83x aggregate.
Optimising it would shave a constant off a curve whose shape is set elsewhere.

## Ablation: correlation to causation

A hot frame with a steep slope is a lead. To make it a finding, remove the mechanism and watch the
slope go with it. `arms/knobs.js` injects knobs into the shipped build through `add_init_script`, so
no recompile is needed.

Every arm reports two things or its reading is discarded:

- **INVARIANCE**: evidence the rendered output is unchanged. An arm claiming exactness that drifts
  is void, not quoted with a caveat.
- **POTENCY**: evidence the knob fired, through a counter that must move. An arm that is exact but
  whose potency counter did not move reads **NOT RUN**, never "no effect".

Prefer a dose-response to an on/off arm where you can build one. Varying the dose and requiring a
straight line makes a *null* informative, which an on/off arm cannot do.

Which knob removes the slope names the fix. If none does, your hypothesis was wrong, and saying so
is a better contribution than a fix aimed at the wrong mechanism.

## If you have many cores

Running several comparisons at once is fine and is how the 40-PR audit was done, but contention is a
confound unless you contain it:

- Pin **whole physical cores**. Never split SMT siblings across two jobs, or two jobs share an
  execution unit and neither timing means anything.
- Give each job **one NUMA node**, memory included.
- Make the slices **equal**, sized by the most crowded node, and hold back about 20% for the kernel.
- Read the topology from `/sys/devices/system/node` rather than hard-coding a CPU list.
- Run the null control **as one of the parallel jobs**, so the floor sees the same contention the
  results do.

## What to put in your pull request

- The metric, both arms' values, the ratio, the spread, and **the floor it cleared**.
- Which tier, how many repetitions, and confirmation that a null control ran in the same session.
- The parity result, plus screenshots if you touched CSS.
- Anything that did not run, said plainly.

A result quoted without its floor will be asked for its floor.
