# Proving a Studio performance change

This is the protocol for turning "it feels faster" into a number a reviewer can trust. It exists
because an audit of 40 frontend pull requests found that **30 of them had no effect distinguishable
from a null control**, and several of those had looked like clear wins right up until a detection
floor was applied to them.

Read this before you quote a result. It is short, and every rule in it was written after a wrong
answer was produced by not having it.

## The loop

```
# 1. Screen: about 5 minutes, direction only
python -m tests.studio.studiobench --tier fast --ab YOUR_REF --out outputs/mine

# 2. Confirm: about 20 minutes
python -m tests.studio.studiobench --tier standard --reps 4 --ab YOUR_REF --out outputs/mine

# ... and the same command with --ab pointing at the BASE, as your null control
python -m tests.studio.studiobench --tier standard --reps 4 --ab BASE_REF --out outputs/null

# 3. Gate: per-metric floor, sign consistency, stability
python -m tests.studio.studiobench.sweep.floor_table --floor outputs/null outputs/mine

# 4. Parity: prove you did not change what is rendered
python -m tests.studio.studiobench.sweep.ui_parity --null outputs/null outputs/mine

# and, before you read any timing at all, that the run actually ran
python -m tests.studio.studiobench --assert-liveness outputs/mine/payload.jsonl

# 5. Read the payload back as a scored report. No Studio, no browser, no network.
python -m tests.studio.studiobench --report outputs/mine/payload.jsonl --tier standard
```

Every command above except the last drives a real Studio and therefore needs credentials.
See **"You need a Studio, and you need its password"** in [README.md](README.md) before you
run the first one: a missing `--password` fails as an HTTP 401 only after the browser has
already started, and `--doctor` reports PASS on that exact configuration. If you drive the wave
against Studios you started yourself rather than letting studiobench install them, `--ab` needs
`--attach` **and** `--attach-b`, one URL per arm, or it exits before measuring anything.

`--assert-liveness` is strict on purpose and an action that could not run is a real finding, not
noise. But two of the eighteen are unreachable on a fixture that loads no model (`image_upload`
has no attachments button, `message_menu` no More button), and a small tier can leave others with
nothing to act on. Read the reasons it prints before you reach for `--allow-not-run`, and name
only the actions you have understood.

Steps 3 and 4 are not optional extras. A number that has not cleared them is not evidence, and
`floor_table` will tell you so: run without `--floor` and it prints the deltas followed by a
refusal to call any of them a result.

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

### The policy, and the two exemptions

**UI and UX idempotency is required for every change, with exactly two exemptions:**

1. **A deliberate difference accepted for a dramatic performance improvement.** This is a decision
   somebody makes on the record with the number attached, not a default. If you are relying on it,
   say so in the pull request and say what the difference is.
2. **A difference that exists only OFF SCREEN.** This one is a definition rather than a judgement:
   rendering only what is visible is an accepted technique, not a parity violation.

Neither exemption removes the need for a floor. An exemption changes what counts as a pass; it does
not make an unmeasured claim into a measured one, so the null control still has to run on the same
scale.

**A bare "PARITY OK" reads far stronger than any of the three modes can support**, which is why
each mode prints the CLAIM it is making and the POLICY it is judging against, and why you should
read both before quoting a pass:

| mode | what a pass means | can it grant the off-screen exemption? |
|---|---|---|
| `--mode digest` | the thread root and declared overlays serialise identically, on screen and off | no, it does not know what was on screen |
| `--mode visible` | every message the viewport showed is present and identical, and every difference lies off screen | **yes** -- this is the mode that can say so |
| `--mode behaviour` | scroll extent matches and the invariants a windowed mount breaks first still hold | no, and it says nothing about appearance |

The digest is **sidebar-blind and layout-blind**: run against a real sidebar-drag change it reported
0 of 34 differing pairs, and so did its own null control. So a digest pass is not a statement that
the UI is unchanged.

### The digest itself

Run the parity digest, which takes a normalised structural DOM signature per action window
on both arms and compares them.

What the digest **can** see: tag structure, `data-slot`, `data-state`, `data-role`, classes, text.
It normalises generated ids, rendered durations, relative times, scroll state, backend-minted record
ids, absolute URLs and blob/data URLs, so those do not read as false differences.

What it **cannot** see: stylesheet CSS, computed layout, colour, typography, or anything rastered.
So a change to `content-visibility`, `contain-intrinsic-size`, transforms or paint can be perfectly
parity-clean and still visibly wrong. **If you touched CSS, take screenshots as well.**

`NOT EXERCISED` is not a pass. An action that never ran is not evidence that it is stable.

### The digest cannot judge a change that alters the DOM on purpose

Virtualization, windowing and progressive mount all change what is mounted by design, so the digest
reports differences everywhere and proves nothing. For those changes parity means screenshots at
matched scroll positions plus the behavioural invariants below.

## 5. Invariant counts: the metrics whose sign means the opposite

Rows named `action.count.key` are correctness invariants, not timings, and the table scores them
with the same paired arithmetic and the opposite sign. A timing falling is the result a change is
trying to produce. A count falling is a regression, and a count is often the only thing that can see
it.

`select_all_copy.count.selected_chars` is the case this exists for. The selection is taken over the
viewport's DOM, and the action's own assertion is only that the character count is above zero, so
anything that stops mounting the whole thread truncates the clipboard from roughly 400,000
characters to a few thousand while every timing improves and the action still reports `expect_ok`.
Scored as a timing that reads `faster`; scored as an invariant it reads `LOST (invariant fell)`.

The comparison is against the other arm rather than an absolute threshold, so no per-rung
calibration is needed: both arms seed a byte-identical thread. A count on an action that did not run
is dropped like any other reading from an action that did not run.

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

`--assert-liveness` separates two things that both print as NOT RUN and mean opposite things:

- A **scene problem** is the harness lying: the action was never planned, the button was not in the
  DOM, the thread was shorter than the viewport. That always fails, on any machine.
- A **missed slot** is a fact about the machine. The scene is a fixed-duration film on the wall
  clock, so a machine too slow to reach a slot records the miss and the film rolls on by design,
  rather than quietly taking a different path through a shorter session.

`--allow-slot-misses` tolerates the second and never the first. It defaults to **0**, which is the
setting a measurement is taken under. Raise it only where the gate is proving the plumbing works
rather than that the machine is fast, which is what CI does on a two-core runner. A tolerated miss
still prints, and the tool still says the payload is not quotable, because a missed slot leaves a
hole in the table whether or not the exit code is 0.

### Both sides must be the same film

The floor table refuses two payloads from different tiers, and it refuses two payloads built on
different corpora. The second one catches the case that looks fine: you take a floor one week, the
corpus changes, you score this week's run against last week's floor, and the corpus change reads as
your change. The corpus hash covers every generated byte and every generator parameter, so any edit
to the generator is caught, not just a version bump someone remembered to make.

If you hit that refusal, re-run the older side. There is no flag to override it.

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

### A stress action is not a user journey

`reasoning_toggle` runs at **2.2 fps** at the 100K rung, with a p95 frame of 2,084 ms. It is the
worst number this harness produces, and it has been quoted as though it described what a user feels.
It does not.

The action opens **every reasoning pane in the thread in one gesture**: 10 panes, materialising
74,917 highlight spans, 2,143 ms to open and 805 ms to close. No user does that. A user expands one
pane. So 2.2 fps is a legitimate measurement of a deliberate worst case and an illegitimate answer
to "how slow is opening a reasoning pane". **We do not currently have that second number**, and a
single-pane variant is the cheapest way to get it: the blocker is not the action, it is that the
film is a fixed-duration slot schedule, so a new slot shifts every existing window and voids
comparability against every payload already on disk. Add it as a registered action first and run it
in a purpose-built film; fold it into the standard scene only at the next corpus or tier bump, when
the payloads are being invalidated anyway.

The general rule: before quoting any action's number as a user-facing latency, read what the action
actually does. Several of them are stress gestures by design, because a stress gesture is the only
way to make a mechanism's slope visible above the noise.

### What the fixture over- and under-states

The corpus is not a real thread and it is shaped differently from one in two directions that do not
cancel.

- **Span density is 4.69 characters per highlight span against the generator's 5.6 target**, so the
  fixture carries about **19% more highlight spans per character of code** than it is meant to.
  Every syntax-highlighting number in this campaign, including the fence-deferral arm's `pre span`
  and LayoutObjects reductions, is measured against that inflated span count and **overstates the
  win by roughly that much**. Read those numbers with this beside them.
- The corpus reaches its rung with **too few, uniformly large messages**: at the 100K rung the nine
  assistant turns run 11,374 to 126,203 characters with a median of 36,215, and none is small. Real
  threads carry many short turns. The total size is right; the distribution is not.
- Because there are 18 messages rather than the hundreds a real thread of that size would hold,
  **per-message chrome is about 1.5% of the fixture's DOM against roughly 14% of a real one**. So
  the fixture **understates** total DOM size, which is the conservative direction for any change
  that reduces element counts.

Anything **per-message** (menus, hover targets, avatars, action bars, virtualization row overhead)
is therefore measured on a document that has far too few of them, and should not be extrapolated
from this corpus without saying so.

### One home per arm, or the A/B compares a build against itself

`install_studio` derives its repo checkout from the home directory, so **two arms sharing one
`--home` share one checkout**. The second install overwrites the first and both arms then serve
whichever build was installed last. Every number downstream is then a comparison of a build with
itself: parity matches, invariants agree, timings sit on top of each other, and none of it means
anything.

It is invisible in the payload, and it looks like the result you were hoping not to get. Two runs
of the same pair reported **716 ms and 718 ms** for base and treatment in one, and **2,583 ms and
2,614 ms** in the other -- nearly equal *within* each run and 3.6x apart *between* them, because
each run was internally uniform and the two runs were serving different builds. Read as a
within-run comparison it says the change does nothing. The difference was sitting between the runs
the whole time.

`--ab` now refuses `--home` outright. Left to itself the harness gives each arm its own
`studio_home_<label>` under `--out`, which is what you want.

**The general rule this is an instance of:** before believing a null result, confirm the two arms
could have differed. A comparison that was structurally incapable of showing a difference will
report agreement with total confidence.

### One engine per comparison

Nothing in the payload format stops you from scoring a Chromium arm against a WebKit one. The engine
is recorded in the run-level `platform` header and **not** on the cell rows, and both `sweep/ab.py`
and `sweep/floor_table.py` pool shards by glob without looking at it. A directory glob such as
`outputs/sbench_v*` will happily merge the two.

This has produced exactly one wrong kind of number so far, and it is the worst kind: Playwright's
WebKit never performs a clipboard copy on Control+C, so `select_all_copy` measured the harness's own
250 ms settle and reported it as `copy_ms` -- 258.5 ms at 1K, 258.8 ms at 10K, 263.9 ms at 100K
across 43 rows. A hundredfold change in the quantity under study moved the "measurement" by 2%,
because it was measuring a `sleep`. Chromium reads about 1,538 ms for the same action at 100K. A
stable wrong number is much harder to catch than a missing one.

The action now refuses instead: a clipboard sentinel is written before the keystroke, and if it
survives the row is **NOT RUN** with a reason, on any engine, with no engine list to maintain. Until
`ab.py` and `floor_table.py` check the header too, **check it yourself** before pooling shards.

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

## Recorded negative result: do not virtualize the message list

Somebody proposes this roughly once a campaign, and the argument is good: a 100K thread stands
64,707 elements in the DOM, most of them off screen, so mount a window instead. It was built, it
was measured at the 100K rung on the standard tier with a concurrent in-band null control, and it
was rejected. The measurements are here so the next person can decide from evidence rather than
rebuild it to find out.

**It buys what it says it buys.** 54,015 elements against the base arm's 64,707. `select_all_copy`
went from 6.1 to 31.6 fps, a p95 frame of 2,284 ms down to 58 ms. Headline 28.2% faster, weighted.

**It costs the thing the campaign exists to fix.** `send_turn` went from 61.1 to 37.3 fps, with the
p95 frame rising from 34.0 ms to 374.8 ms, while the concurrent null moved -3.9% on the same
window. Reproduced on two independent runs. Streaming at long context is the whole complaint, and
this made it worse by more than anything else on the page got better.

The mechanism, measured rather than guessed. It is NOT per-chunk: every `stream:gap*` window, the
windows in which the reply is actually streaming into a growing last row, is flat (+0.2%, -0.1%,
+0.6%). It is a ONE-TIME cost at the first append into a freshly opened thread: the eight treatment
readings alternate 13.2, 58.9, 13.7, 62.0, 14.6, 59.8, 14.7, 60.9 fps, two sends per cell, first
expensive and second free. A direct probe put the cost in RECALC STYLE (60 ms against the base
arm's 14 ms) and not in layout (2 ms on both) and not in measurement callbacks (2 ResizeObserver
callbacks, 3 style writes). Appending to a windowed, end-anchored list re-renders and re-positions
every mounted row; the unvirtualized path appends one node and touches nothing that already exists.
The first append also resolves size estimates that are badly wrong: `estimateSize` is 460 px and
the fixture's mounted rows measure a median of 2,379 px, a mean of 5,146 px and a maximum of
21,132 px.

**It loses the thread.** On the treatment arm the census goes from 12 mounted messages to 0 at the
`model_change` slot and never recovers: 2,107 elements for the remaining four slots of the film,
all four reps, both runs, while the concurrent null keeps all 25 messages through the same action.
Visible-region parity reports it as a difference rather than a refusal. Two caveats kept with the
finding: `model_change` is the weakest selector in the suite by its own docstring, and in these
runs it selected an option labelled "Search Hub", which is not a chat model, so this is "the
windowed arm did not survive an interaction the base arm survived" and not "changing the model
empties a virtualized thread". It was not chased further because the arm was closed.

**Three of eighteen actions could not be measured on it.** `delete_message` and `thread_reopen` ran
four times on the base arm and zero times on the treatment; `image_upload` ran on neither. Any
per-window comparison must drop those rather than compare a busy window against an empty one.

**And two accessibility and correctness obligations come with it.** A windowed list must publish
`aria-setsize` and `aria-posinset` or the accessibility tree reports a forty-turn conversation as
the six messages that happen to be mounted. Select-all copy must be served from the message store,
because a windowed DOM cannot select what it has not mounted: measured at 0.61 of the thread before
the fix. The obvious store serialiser is the "save this reply" one, which emits reasoning and
tool-call payloads that no user can select, and it lands at 2.16 of the thread instead. Both
failure directions are now bounds on `clipboard_carries_the_whole_thread`.

If you want the structural win without any of this, defer off-screen work instead of unmounting it.
Score it with `sweep/ui_parity.py --mode visible --null OUTDIR`: an off-screen-only difference is
exempt by policy, and that mode is the one that can say so.
