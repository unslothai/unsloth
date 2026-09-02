# studiobench

A performance benchmark, profiler and A/B simulator for Unsloth Studio that runs the **real path**.

## Run it yourself

```
pip install playwright psutil
playwright install webkit          # chromium on Windows; see "Which engine" below

python -m tests.studio.studiobench --doctor
```

`--doctor` is always the first command. It works on a machine with nothing installed, because every
heavy import is lazy, and it names what is missing and what each missing piece costs you. It does
NOT check that it can log in to an Unsloth you point it at, so read the next section before you
conclude that a `--doctor: PASS` means a run will start.

### You need an Unsloth, and you need its password

Every command below other than `--doctor` drives a real Unsloth, and Unsloth requires
authentication. There are two ways to give it one.

**Attach to an Unsloth you are already running.** This is the cheap path, and it needs the
credentials, which are NOT optional and default to nothing:

```
python -m tests.studio.studiobench --tier fast \
    --attach http://127.0.0.1:5401 \
    --password "$(cat ~/.unsloth/studio/auth/.bootstrap_password)"
```

`--username` defaults to `unsloth`, `--password` defaults to empty. Without a correct password the
run aborts with `HTTP 401 from .../api/auth/login` after the browser has already started. If you
launched Unsloth with a non-default `UNSLOTH_STUDIO_HOME`, the password file is
`$UNSLOTH_STUDIO_HOME/auth/.bootstrap_password`.

Note that studiobench **rotates** the password to its own bench value on first login, so a second
run against the same Unsloth does not need `--password` again.

**Or let studiobench install and launch its own Unsloth** with `--branch REF`. No password needed,
because it owns the instance, but the first run clones this repository and runs `install.sh`, which
is a multi-gigabyte download budgeted at up to 45 minutes. The wall-clock figures in the table
below are the measurement only and do not include that install.

Then pick a path. There are two that matter:

| path | tier | rungs | film | wall clock | what it is for |
|---|---|---|---|---|---|
| **fast** | `--tier fast` | 100K only | 57.3 s | about 5 min, or 9 min for an A/B wave | iteration. You are trying a fix and want direction. |
| **slow** | `--tier standard` | 1K, 10K, 100K | 243 s | about 20 min | confirmation. You believe a number and want it to hold. |

`--tier quick` (1K and 10K) is a wiring check, and `--tier full` adds the 500K and 1M rungs for a
ceiling hunt. Neither is the loop you work in.

```
# iterate against an Unsloth you are already running (see the password note above)
python -m tests.studio.studiobench --tier fast \
    --attach http://127.0.0.1:5401 \
    --password "$(cat ~/.unsloth/studio/auth/.bootstrap_password)" \
    --out outputs/iterate

# confirm, letting studiobench install and launch its own Unsloth from a ref
python -m tests.studio.studiobench --tier standard --branch main --reps 4 --out outputs/confirm

# read the payload back as a scored report; writes <out>/summary.md beside it
python -m tests.studio.studiobench --report outputs/confirm/payload.jsonl --tier standard
```

Pass `--out` explicitly. Without it the run invents a `studiobench-<tier>-<timestamp>/` directory
in whatever your working directory is, which for someone standing in a clone means an untracked
directory in the repository root.

**One `--out` holds one run.** A fresh run pointed at a directory that already has a
`payload.jsonl` moves the old one aside to `payload-<timestamp>.jsonl` and starts its own, and says
so. Nothing is deleted, and nothing is mixed: a cell id is the rung, the arm and the repetition, so
two runs sharing one file are two builds under two films in one ladder with no way to tell them
apart. `--resume` is the one reuse that appends, and it refuses a payload recorded under a
different tier, cadence, browser engine, instrument level, corpus or ref rather than skipping cells
that measured something else.

**The fast tier is a screen, not a result.** One rung, a wider detection floor, direction only. It
exists so that someone trying a fix does not wait 20 minutes to learn they were wrong. Nothing goes
in a pull request until `--tier standard` agrees with it.

### Why 100K, and why the fast tier has only that rung

The 10K rung was run across six pull requests and could not separate any of them from a null
control: at that size the UI work disappears underneath the scene's own scripted timings, and
`copy_markdown` read 204 ms on all fourteen arms. 100K is the smallest rung that carries real load,
with a jank index of 29 against 0.6 and a worst frame of 1,855 ms against 214. It is the only rung
worth an iteration loop.

### Proving a change actually did something

A single number from a single build is not evidence. Read
[CONTRIBUTING-perf.md](CONTRIBUTING-perf.md) before you quote a result. The short version: run a
null control alongside your A/B, derive a per-metric detection floor from it, and clear all three
verdict gates. In an audit of 40 frontend pull requests, 30 had no effect distinguishable from that
null control, and several of those had looked like clear wins before the floor was applied.

## Why this exists

A full day of measurement failed to name what makes long generations slow, and the reason turned
out to be the fixture rather than the analysis: **the old harness did not run the code that is
slow.** It measured a backend-free smoke page driven by a local `ChatModelAdapter`, so two whole
mechanisms never executed.

1. **The cumulative `<think>` re-parse.** Real reasoning arrives as `delta.reasoning_content`, is
   wrapped into `<think>...</think>`, appended to a single cumulative buffer, and then
   `parseAssistantContent(cumulativeText)` re-parses **the whole growing buffer on every delta**.
   O(n) per chunk, O(n^2) per reply. Not one line of it ran.
2. **The autoscroll observer.** `use-intent-aware-autoscroll.tsx` installs a `MutationObserver`
   with `subtree: true, characterData: true` over the entire viewport. Its callback synchronously
   reads `scrollHeight`, writes an inherited CSS custom property on the scroll container
   (invalidating style for every descendant), and calls `scrollTo`, on every streamed character, at
   a cost proportional to the whole thread. `LayoutDuration` read as a flat floor, which is exactly
   what you would see if this never ran.

So Layer 1 runs the shipped app, through its own backend, over real SSE bytes.

## How the real path is arranged

```
studiobench pacer  --SSE-->  Unsloth backend relay  --SSE-->  the SPA's own TextDecoder,
(ThreadingHTTPServer)         (external provider,             SSE parser, delta accumulation,
                               provider_type "custom")        parseAssistantContent, Streamdown,
                                                              the autoscroll observer
```

There is **no `page.route` on the primary transport** and no local adapter. A `--transport direct`
mode is reserved purely as a cadence-fidelity ablation.

### The pacer

`pacer.py` is a stdlib `ThreadingHTTPServer` speaking OpenAI-compatible `text/event-stream`.

- **Threaded**, not single-threaded: a single-threaded server previously lost 11 cells of a matrix
  to `goto` timeouts, because the browser opens the SPA's own requests while a stream is in flight
  and one blocked handler stalls all of them.
- **The exact chunk shapes the app parses.** `delta.reasoning_content` carries `content: ""`
  alongside, which is what the backend's own `_gguf_chat_delta_line` emits. The terminal chunk
  carries `finish_reason: "stop"` **and** is followed by `data: [DONE]`: `streamChatCompletions`
  throws `StreamInterruptedError` at EOF unless it saw one of them, so a harness that omits both
  measures the error path. A usage chunk follows, because the app always sends
  `stream_options: {include_usage: true}`.
- **Chunked transfer encoding.** Verified: an HTTP/1.1 response with neither `Content-Length` nor
  `Transfer-Encoding` is a keep-alive body of unknown length, and the reader blocks forever having
  already received every byte.
- **Deficit-scheduled cadence.** Each tick computes `floor((now - t0) / gap)` and sends the
  shortfall in one burst, rather than sleeping a gap per chunk. Stream duration then depends on
  wall clock alone, so a tier's time budget is honest on any machine, and a renderer that jams gets
  a burst when it recovers, which is what a real backend does. Default cadence is the captured
  reply's own: **24 characters every 73 ms**.

`python -m tests.studio.studiobench.pacer` checks all of this on the wire with no browser.

### The corpus

`fixture/corpus/frozen/` **ships**. The text is generated once from a seed and frozen, with a
sha256 per unit; a generator that drifts is refused rather than quietly measured. Every fence is
unique, because Shiki caches highlighted output keyed on the source string and a repeated fence is
free, which is how a harness measures a 300K-character thread and finds no highlighting cost in
it. The escalating cycle is reasoning@10K, code@8K, reasoning@20K, code@16K, and so on.

**Corpus v2 added math**, about 6% of characters, as 60 display blocks and 370 inline spans across
the shipped units. Both delimiter families are present: `$...$` and `$$...$$`, which remark-math
consumes directly, and `\(...\)` and `\[...\]`, which reach the renderer only if `preprocessLaTeX`
rewrites them first. v1 had not one dollar sign in 519,859 characters, which is why a `preprocessLaTeX`
cost that was real in isolation measured as an exact NULL in the browser. Math takes the prose slot
rather than being added alongside it, so the fence share is unchanged, 0.4754 to 0.4779, and the
span-density calibration above still describes this film. The preamble stays free of both math and
fences, because it is the only stretch the film has that builds nothing.

A v1 number and a v2 number are measurements of two different films. The corpus hash covers every
generated byte and every generator parameter, and `sweep/floor_table.py` refuses to pool payloads
built on different corpora, or to score one against a floor from another, rather than reading the
corpus change as a performance change.

Rungs are **1K / 10K / 100K / 500K / 1M tokens**, and the characters-per-token ratio is
**measured** per rung (tiktoken, else Unsloth's own counter, else a labelled estimate) rather than
assumed at 4.0.

Bulk thread mass is **seeded** over `PUT /api/chat/threads/{id}/messages`; only the last reply
streams, because a million tokens at field cadence is three and a half hours. Seeded and streamed
are not the same path, so the equivalence is **checked at the 10K rung** and higher rungs are
labelled `fidelity: seeded_only` when it fails.

The rung varies the **seeded thread**, and the streamed reply is held constant at
`STREAM_TAIL_CHARS = 6_000` on every rung by design, so that the tail is comparable across rungs. A
consequence worth knowing before you design an experiment: a mechanism whose cost scales with
**reply length** rather than thread size is held constant by this ladder and will read as a flat
floor on it. Measuring one of those needs an axis you build yourself.

### The scene

A **fixed-duration film**, not a task list. Every action has a fixed `(t_start_ms, budget_ms)` on
wall clock. A machine too slow to reach a slot records `slot_missed: true` and the film rolls on.
A sequential script would make a slow machine take a different path through a different-length
session, and nothing would be comparable.

Fifteen actions, each with an `expect` assertion that proves it happened:

| action | what proves it |
|---|---|
| keystroke | the composer's controlled value grew by the characters typed |
| scroll during generation / after | the viewport travelled at least 90% of what was commanded |
| reasoning expand/collapse | every pane's `data-state` went open, then all closed |
| stop generation | the run ended, and the character count stopped growing |
| settings open/scroll/close | the dialog appeared, its body travelled, it closed |
| model change | an option was clicked and the menu closed |
| composer short/medium/very long | the composer held every length it was given |
| copy markdown | the clipboard was non-empty afterwards |
| select text | the selection covered at least half the **visible** characters |
| select-all + copy | the selection was non-empty |
| image upload | the composer's attachment count rose |
| thread reopen | the thread came back with the same message count |
| message menu | it opened **and** closed **and** had a non-zero item count |
| delete | the `[data-role]` count dropped |

**An action that did not happen is `ran: false`, never a fast timing.** The Radix menu trigger
opens on `pointerdown`, so `element.click()` leaves it shut and the column reads a tidy small
number. A jump scroll from the bottom is read by the intent-aware autoscroll as programmatic and
snapped back, so the viewport lands where it started and the timing is precise and about nothing.

This is the most common way this harness has produced a wrong answer, and it has happened three
separate times in three separate subsystems. Each time the shape was identical: code that could
never fire, reported as "no effect". Check `ran` before you read a timing.

### The instruments

| name | level | what it reads |
|---|---|---|
| `frames` | 0 | one self-rescheduling rAF loop, timer lag, long tasks, CDP presented frames |
| `input` | 0 | keystroke-to-paint from the page side of a real key event |
| `rss` | 0 | the whole browser tree's RSS, on a thread |
| `glass` | 1 | `scrollHeight` reads, `scrollTop` writes, the stabilizer property, mutation records |

Headline numbers come from level 0 only. Higher levels buy naming at the cost of overhead, and
`overhead_growth_with_length` is a gate: overhead correlated with the treatment disqualifies that
level for that comparison.

The rAF loop **counts and does not pump**, and `requestAnimationFrame` is not wrapped as the frame
counter: a wrapper counts the page's frame once for the loop and once more for every rAF the app
scheduled in it, and reported **888 fps on a 60 Hz page**.

The timer clamp is calibrated **inside an enforced idle window** before each measured window, not
from the first ticks of a page that already has 31,637 elements standing. If the calibrated clamp
exceeds 10 ms, `busy_pct` is `null` **with a reason**, never `0.2%`.

## Gates

- **A dev server is refused.** React's development build inflates the axis under investigation by
  about 3.2x, so a measurement there would confirm any hypothesis. Two checks:
  `/@vite/client` must not serve a JavaScript module, and `bundleType: 0` must appear in the same
  chunk as `rendererPackageName: "react-dom"`.
  - The `/@vite/client` probe checks **what came back**, not just the status: Unsloth serves its SPA
    for any unknown path, so a production build answers 200 with `index.html`.
  - The marker regex accepts **backticks**: the production bundle is minified with a pass that
    rewrites short string literals as template literals.
  - **`jsxDEV` is never grepped.** `hast-util-to-jsx-runtime`, which Streamdown pulls in, ships its
    own option guard naming `jsxDEV`, so the grep fails a perfectly good production build.
- **No bare zeros.** Every numeric key that can legitimately be zero carries a sibling
  `<key>_attempted`. An unmeasurable quantity is `null` with a `<key>_reason`.
- **Three clocks.** rAF, a 1 ms timer, and CDP presented frames. More than 20% disagreement marks
  the window `clocks_agree: false` and the report layer excludes it from scoring.
- **No cross-session comparison.** Every slope, ratio and A/B pair is read within one session, and
  the report layer refuses anything else. Measured session-to-session drift on this metric set is
  about 8%, which is larger than most real effects.

## Ablation: proving a cause rather than correlating with one

A hot frame with a steep slope is a lead, not a finding. `arms/knobs.js` carries runtime-injected
knobs that are applied to the **shipped build** through `add_init_script`, so an ablation needs no
recompile: hide content, undo a `content-visibility` override, detach the autoscroll observer,
neutralise the scroll stabilizer property, freeze React while keeping the DOM.

Every arm must declare and report two things or its reading is worthless:

- **INVARIANCE**: evidence the rendered output is unchanged by the knob. An arm that claims
  exactness and then drifts is **void**, not quoted with a caveat.
- **POTENCY**: evidence the knob actually fired, through a counter that must move. An arm that is
  exact but whose potency counter did not move reads **NOT RUN**, never "no effect".

Which knob removes the slope names the fix. If no knob does, the hypothesis was wrong, and that
negative result is worth more than a fix aimed at the wrong mechanism.

## Output

`<out>/payload.jsonl`, one JSON object per line, flushed and fsynced as it is produced, so a
renderer crash at rung 4 still ships rungs 1 to 3 plus the crash record. A cell that could not
complete emits a `cell` row with `completed: false`, its failure mode and its RSS at death.
(`<out>` is the `--out` directory. `report/` is the source package that renders the payload, not
an output path.)

`--report <out>/payload.jsonl` scores that file and prints the summary, writing it to
`<out>/summary.md` as well. It needs no browser, no Unsloth and no network, which is the point of
shipping a single-file benchmark: the numbers come back as a file and the analysis happens
wherever the analyst is. Pass the same `--tier` (or `--rungs`) the run used, or the ladder will
report rungs you never declared as incomplete.

## Portability

`python -m tests.studio.studiobench.build` produces `dist/studiobench.pyz`, one file. The bootstrap
is stdlib-only and Playwright is imported lazily, so `--help` and `--doctor` work on a machine with
nothing installed, which is the machine where `--doctor` has to say what is missing.

The default engine matches the tester's desktop webview family: Windows to Chromium via
`channel=msedge` (WebView2), macOS to WebKit (WKWebView), Linux to WebKit, **labelled a proxy for
WebKitGTK** rather than presented as the real thing.

## Layers

Layer 1 (this) owns the real-path session. Layer 2 owns tracing and analysis, Layer 3 owns the
ablation arms and the report. `INTERFACES.md` is the contract between them.
