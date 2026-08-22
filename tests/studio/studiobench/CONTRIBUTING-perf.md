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
python -m tests.studio.studiobench --tier fast --ab YOUR_REF --out outputs/screen

# 2. Confirm: about 20 minutes. A NEW --out: the payload is append-only, so writing a standard
#    run into the screen's directory leaves one file holding two films.
python -m tests.studio.studiobench --tier standard --reps 4 --ab YOUR_REF --out outputs/mine

# ... and the same film with BOTH arms on the base, as your null control. --ab sets the
#    TREATMENT ref only; --branch is the base arm and defaults to main, so passing --ab BASE_REF
#    alone measures main against BASE_REF and calls the difference noise.
python -m tests.studio.studiobench --tier standard --reps 4 \
    --branch BASE_REF --ab BASE_REF --out outputs/null

# ... and, before you read any timing at all, that the runs actually ran. Here rather than at
#    the end: floor_table DROPS a repetition whose action did not run instead of refusing it, so
#    a missed slot leaves the paired table quietly and the repetitions that survive print as a
#    clean win on a smaller n.
python -m tests.studio.studiobench --assert-liveness outputs/mine/payload.jsonl

# ... BOTH payloads. The null control is the one people forget, and a hole there is worse: the
#    floor is max(|null delta|, null spread) over the repetitions that SURVIVED pairing, so a
#    slot the null run missed drops its noisiest repetition and tightens the bar your result is
#    then measured against. Your own payload passes this gate while it happens.
python -m tests.studio.studiobench --assert-liveness outputs/null/payload.jsonl

# 3. Gate: per-metric floor, sign consistency, stability
python -m tests.studio.studiobench.sweep.floor_table --floor outputs/null outputs/mine

# 4. Parity: prove you did not change what is rendered
python -m tests.studio.studiobench.sweep.ui_parity --null outputs/null outputs/mine

# 5. Read the payload back as a scored report. No Studio, no browser, no network.
python -m tests.studio.studiobench --report outputs/mine/payload.jsonl --tier standard
```

The three commands in steps 1 and 2 drive a real Studio and therefore need credentials. The rest
read a payload that already exists: `--assert-liveness`, `floor_table`, `ui_parity` and `--report`
run offline, with no Studio, no browser and no network.
See **"You need a Studio, and you need its password"** in [README.md](README.md) before you
run the first one: a missing `--password` fails as an HTTP 401 only after the browser has
already started, and `--doctor` reports PASS on that exact configuration. If you drive the wave
against Studios you started yourself rather than letting studiobench install them, `--ab` needs
`--attach` **and** `--attach-b`, one URL per arm, or it exits before measuring anything. It also
needs `--password` **and** `--password-b`, one per arm: two separately booted Studios mint two
different bootstrap passwords, so a single `--password` is a 401 on the treatment alone, after
the browser is already up.

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

Three rules about it, all learned the hard way:

- **The floor is per metric, not global.** A single global floor derived from the frame tail says
  nothing about whether `settings.open_ms` can resolve 5%. Each metric gets its own floor from the
  null control.
- **The floor must be measured in band**, concurrently with your A/B, under the same machine load.
  A floor measured on a quiet machine does not transfer to a busy one. Session-to-session drift on
  this metric set is about 8%, which is larger than most real effects, so a floor from yesterday's
  run is not a floor.
- **A hole in the null control tightens the floor**, so the null gets `--assert-liveness` too. The
  floor is computed from the repetitions that survived pairing, and the repetition a null run drops
  is the one that was slow enough to miss a slot, which is the noisiest one it had. Four null
  repetitions reading 1.000, 1.002, 0.999 and 1.300 set a 30.1% floor on
  `message_menu.open_close_ms` and void a 10% result; lose the fourth to a missed slot and the same
  three survivors set a 0.3% floor and the same 10% prints as `faster`.

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

### The digest cannot judge a change that alters the DOM on purpose

Virtualization, windowing and progressive mount all change what is mounted by design, so the digest
reports differences everywhere and proves nothing. For those changes parity means screenshots at
matched scroll positions plus the behavioural invariants below.

### When the equivalence is proved by a differential, the fixture is the claim

Some optimisations are not "the DOM still matches" but "this cheaper path returns the same bytes as
the expensive one it replaces". You prove those with a randomized differential: generate inputs,
run both paths, compare. The number of frames it clears is then quoted as the evidence, so the
generator, not the run, is what the claim actually rests on.

Two ways that goes wrong, both paid for on unslothai/unsloth#9517, where the cheap path repaired a
bounded window of an open code fence instead of the whole tail:

- **A fixture that generates only short lines cannot find a bug about long ones.** The first sweep
  reported byte-identical output on 17,436 adversarial open-fence frames and was quoted in the pull
  request. Every generated piece was a short line, so the window never elided a whole line, and the
  bug lived exactly there: a whitespace-only line longer than the window moved the cut past it, and
  the repair then appended a zero-width space into code that the copy button would put on the
  clipboard. A reviewer found it by reading. Adding lines longer than the window, whitespace-only
  and not, immediately produced 60 divergences, and then a second, unrelated family of 57.
- **A clean result from a fixture you built yourself is weaker evidence than a clean result from one
  designed to break you.** The two are not the same measurement and should not carry the same
  weight in a pull request. Before quoting a differential, write down the dimension your generator
  varies and the dimension it holds constant, and go looking for a bug in the constant one.

A corollary about hunting the second bug: **a check that comes back clean against code you believe
is broken is usually a hole in the check.** The first probe for that second family reported the
production code unaffected, and the reason was that its head set contained no stray backtick before
the fence. It was reachable. Prove reachability against the real code path with the real input
before deciding a class of bug does not apply.

### Getting a guard backwards costs a sweep, so state its direction in words first

The same round produced two guards that were written the wrong way round and only caught by
re-sweeping:

- The refusal gated the cut **from below** when the risk is a cut landing **past** the marker.
  The condition and its inverse both read plausibly at the call site.
- It then tracked the **last** marker in the body when the one that matters is the **first**: a
  marker sitting in the retained window masks an earlier one sitting in the elided middle, which is
  the half that gets dropped. First is also monotone, so it is the cheaper thing to carry anyway.

Write the invariant as a sentence about which region must be clean before writing the comparison,
and keep the shrunk reproducers verbatim in the test with a comment saying so. Both of these needed
a stray backtick before the fence, a longer opener and an escaped fence inside it; a tidier-looking
case stops covering the bug.

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

## The `content-visibility` trap, in full

This mechanism has cost this project three separate investigations and produced two nulls that were
not nulls. It is written out here so nobody spends a fourth day on it. **Do not open another
`content-visibility: auto` arm on the message roots.** The arm is dead, for the reason in the last
subsection, and it is dominated: deferring off-screen fence highlighting monotonically reaches
LayoutObjects -74.0% and `pre span` -96.9% at the same rung with a viewport `scrollHeight` delta of
**0 px**, byte-identical `selected_chars`, and `select_all_ms` at -67.5% instead of +33%.

Everything below generalises past this one property, which is why it is here and not in a commit
message.

### How to prove an arm fired

`contentvisibilityautostatechange` is emitted by Blink's own relevance machinery and by nothing
else. No stylesheet, no selector and no author code can produce one. That makes it a potency
counter of a strictly stronger kind than anything you can read out of `getComputedStyle`.

The method: before writing any property, attach a listener to every candidate element exactly once,
and count events by `event.skipped`. Then run the same probe on a control arm that has no rule at
all. On a 100K thread this read **193 `skipped === true` events on the armed side against exactly 0
on the control**, with 384 state changes in total and 22 roots simultaneously in the skipped state.

A computed-value read cannot give you that. Computed `content-visibility: auto` proves only that
the declaration won the cascade. An element can compute to `auto` and be painted on every frame
because it never stopped being relevant to the user, and then your arm reads null for a reason that
has nothing to do with rendering.

`arms/content_visibility_probe.js` is the working probe. Install it with
`SBENCH_EXTRA_INIT_SCRIPT`, read it back with `SBENCH_PAGE_CONSOLE="CVPOT "`. Both are unset by
default, so a scored run is unaffected. A probe run drives a real Studio like everything else in
the loop, so it needs the credentials above; the file's own header carries the full command.

A run carrying a probe is **not scorable**, because the probe samples the DOM and forces layout on
its own schedule. That is a gate, not a convention. The init script's path is written into
`run_meta.probe_init_script`, the run records a `probe_free` gate, and **all three scoring entry
points refuse the payload on that evidence**: the session prints no `ab.md` at the end, `--report`
refuses, and `floor_table` refuses. There is no flag to override any of them. The payload is still
kept, because `--assert-liveness` and the probe's own output are exactly what you wanted from that
run. Re-run with the variable unset to get a number.

Into a **fresh `--out`**, though, because that refusal is whole-file. `run_meta.probe_init_script`
is on the payload identity, so `--resume` refuses before anything is installed when the variable
was set for the payload and not for this run or the other way round. Without that refusal a
variable still set in the shell from an earlier experiment turned a half-finished clean ladder into
a file no reader will ever accept, cells already recorded included, and a payload is append-only.

A probe must be **self-contained**. Playwright does not define the evaluation order of separate
init scripts, and the scene scripts are deliberately kept as separate scripts so that a throw in
one cannot stop the others, so a probe cannot assume `window.__sb` exists when it is installed.
A probe that fails to install is **reported, never silent**: one that did not parse is named on the
console, and one that parsed and then threw arrives as a `pageerror`, both of which the run log
carries. On chromium and firefox a probe that does not parse also leaves the scene scripts alone.
On **webkit**, the default engine on Linux and macOS, it does not: Playwright hands webkit its init
scripts as one bootstrap unit, so a syntax error in the probe stops dom.js, parity.js and
surfaces.js as well. The probe is installed as plain source rather than through `eval` because
Studio serves `script-src 'self'` with no `'unsafe-eval'`, and webkit enforces that against an init
script, so an eval-installed probe never ran there at all.

Attach your listeners at **insertion**, from a `MutationObserver` installed before the app boots,
not on your sampling tick. `contentvisibilityautostatechange` fires on a *change* of state, and a
root inserted off screen becomes skipped once and then never changes again while the user stays
put. A whole seeded thread mounts inside two seconds, so a probe that adopts roots on a two-second
tick misses every one of their only events and reports zero.

### The geometry route is self-defeating

This is the subtle one, and it is the first thing everyone reaches for.

The reasoning is sound: a skipped subtree is not laid out, so its element descendants generate no
layout boxes, so counting boxes should tell you whether skipping happened. The measurement is still
useless, because **asking is what breaks it**. `getClientRects()` on content inside a locked subtree
makes Chromium render that subtree in order to answer. The probe unlocks exactly what it came to
observe.

Measured side by side in one session, the geometry route reported **0 off-screen unrendered roots
while the event counter recorded 22 roots in the skipped state**. Read on its own it is a clean,
confident, wrong "the arm did not fire". Use the events.

The same forced render is the mechanism behind the select-all cost below, so this is not a quirk of
the instrument. It is the property under test biting the instrument.

### The cascade rule that silently voids an arm

`index.css` opens `@layer utilities` at **line 1051** and closes it at **line 2572** (check by brace
depth, not by eye). Inside it, at **lines 2565-2567**:

```css
.aui-thread-root [data-streamdown="code-block"] {
	content-visibility: visible !important;
	contain-intrinsic-size: none !important;
}
```

Per css-cascade-5 the sorting order is Origin and Importance, then Context, then **Element-Attached
Styles**, then **Layers**, then Specificity, then Order of Appearance, and for important rules the
declaration in the **earliest** layer wins while unlayered declarations sit in an implicit **final**
layer. So:

- an **unlayered** `content-visibility: auto !important` on code blocks **loses** to that rule at
  any specificity, however many `html body` prefixes you pile on;
- an **inline** `!important` **wins**, because Element-Attached Styles are checked before Layers;
- a rule targeting the **message roots** never contests it at all, because that selector matches
  code blocks only.

An arm that loses this way reports a perfectly clean null. If you inject CSS, emit it into the same
layer or inline, and then verify with `getComputedStyle` before you believe any timing.

Worth knowing separately: a descendant's `content-visibility: visible` does **not** stop an ancestor
from skipping. In the session above the code blocks kept computing to `visible` on both arms and
their message roots skipped anyway.

### You cannot opt out of the last remembered size

Chromium ships `content-visibility: auto` **implies** `contain-intrinsic-size: auto` (Blink intent
"content-visibility: auto implies contain-intrinsic-size: auto", CSSWG issue #8407). Declaring a
plain `<length>` does not opt out: on Chromium 151, `contain-intrinsic-size: 300px` on such an
element computes back as `auto 300px`. A variant built with the keyword deliberately removed
produced byte-identical geometry to one with it, which is what that behaviour predicts and which
wastes a build and a run if you have not read this.

### The actual killer: a remembered size of zero

`contain-intrinsic-size: auto <length>` uses the **last remembered size** and falls back to the
`<length>` only if one does not yet exist (css-sizing-4 5.2, 5.2.1). The fallback being too small is
the failure everyone anticipates. It is not the one that happens here.

A message root is mounted **before its content arrives**, so the remembered size is recorded while
the root is empty, and it is recorded as zero. The root is then skipped and frozen at zero forever.
Measured at rest on a freshly loaded 100K thread, 13 of 18 roots reported their **padding alone**
(18 px and 40 px), and the raw height list carried no root at 318 px or 100 px, which is where the
declared 300 px and 60 px fallbacks would have put one: `getBoundingClientRect()` is the border box,
so a fallback-sized root measures the fallback *plus* the root's padding. The thread's scroll height
read **58,355 px on the control against 11,949 px armed** (min 5,101). The concurrent null control
read 58,355 against 58,355, to the pixel.

It self-heals: once the user has scrolled the whole thread, real sizes get remembered and the armed
side climbs back to ~58,000 px. So the damage is on load and after every remount, which is the worst
possible shape for it to have. **Static CSS has no way to override a remembered size.**

UI parity records the same thing from the other side: **0 of 64 action pairs matched** on the armed
comparison against **47 of 64** for the null in the same wave. Every timing from an arm in that
state is a bound at best.

### The select-all cost

Skipped content has to be rendered before it can be selected, so select-all pays for the unlock.
Three independent measurements, all at the 100K rung:

| run | base | armed | delta | n | floor | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| `u6i_` sweep | | | **+28.8%** | 4 | 16.7% | cleared |
| probe run, fast tier | 176.2 ms | 236.1 ms | **+34%** | 2 | none | direction only |
| `cvs1_` standard tier | 196.2 ms | 250.8 ms | **+33.2%** | 4 | 37.5% | **void**, spread 71.5% |

Stated honestly: the third run's own scatter was wider than its effect, so gate 3 voided it. Three
measurements agreeing on sign and magnitude is not the same as one that cleared the gates, and the
row above is the one to quote. The mechanism is not in doubt; the certification is.

`select_all_copy.count.selected_chars` did **not** fall (194,992 against 195,006), which is the
expected direction: the selection forces the content back into existence, so the clipboard is intact
and only the clock suffers.

### The 60.2 versus 64.8 figure is not a message-roots null

This number has circulated as evidence that `content-visibility: auto` on message roots does
nothing. It is not that measurement. It was `content-visibility: auto` on the **reasoning pane's
code blocks**, PR 9073, **n=1, no floor**, and PR 9073's own investigation notes had already
withdrawn it for the cascade reason two subsections above: the rule lost to `index.css:2566` and the
arm never fired. Please stop citing it. The message-roots arm was never null; it fires hard and dies
on geometry instead.

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
