# studiobench

A performance benchmark for Unsloth Studio that runs the **real path**.

```
python -m tests.studio.studiobench --doctor
python -m tests.studio.studiobench --tier quick --attach http://127.0.0.1:5401
```

## Why this exists

A full day of measurement failed to name what makes long generations slow, and the reason turned
out to be the fixture rather than the analysis: **the old harness did not run the code that is
slow.** It measured a backend-free smoke page driven by a local `ChatModelAdapter`, so two whole
mechanisms never executed.

1. **The cumulative `<think>` re-parse.** Real reasoning arrives as `delta.reasoning_content`, is
   wrapped into `<think>…</think>`, appended to a single cumulative buffer, and then
   `parseAssistantContent(cumulativeText)` re-parses **the whole growing buffer on every delta**.
   O(n) per chunk, O(n²) per reply. Not one line of it ran.
2. **The autoscroll observer.** `use-intent-aware-autoscroll.tsx` installs a `MutationObserver`
   with `subtree: true, characterData: true` over the entire viewport. Its callback synchronously
   reads `scrollHeight`, writes an inherited CSS custom property on the scroll container
   (invalidating style for every descendant), and calls `scrollTo` — on every streamed character,
   at a cost proportional to the whole thread. `LayoutDuration` read as a flat floor, which is
   exactly what you would see if this never ran.

So Layer 1 runs the shipped app, through its own backend, over real SSE bytes.

## How the real path is arranged

```
studiobench pacer  --SSE-->  Studio backend relay  --SSE-->  the SPA's own TextDecoder,
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
  a burst when it recovers — which is what a real backend does. Default cadence is the captured
  reply's own: **24 characters every 73 ms**.

`python -m tests.studio.studiobench.pacer` checks all of this on the wire with no browser.

### The corpus

`fixture/corpus/frozen/` **ships**. The text is generated once from a seed and frozen, with a
sha256 per unit; a generator that drifts is refused rather than quietly measured. Every fence is
unique, because Shiki caches highlighted output keyed on the source string and a repeated fence is
free — which is how a harness measures a 300K-character thread and finds no highlighting cost in
it. The escalating cycle is reasoning@10K, code@8K, reasoning@20K, code@16K, and so on.

Rungs are **1K / 10K / 100K / 500K / 1M tokens**, and the characters-per-token ratio is
**measured** per rung (tiktoken, else Studio's own counter, else a labelled estimate) rather than
assumed at 4.0.

Bulk thread mass is **seeded** over `PUT /api/chat/threads/{id}/messages`; only the last reply
streams, because a million tokens at field cadence is three and a half hours. Seeded and streamed
are not the same path, so the equivalence is **checked at the 10K rung** and higher rungs are
labelled `fidelity: seeded_only` when it fails.

### The scene

A **fixed-duration film**, not a task list. Every action has a fixed `(t_start_ms, budget_ms)` on
wall clock. A machine too slow to reach a slot records `slot_missed: true` and the film rolls on.
A sequential script would make a slow machine take a different path through a different-length
session, and nothing would be comparable.

Fifteen actions, each with an `expect` assertion that proves it happened:

| action | what proves it |
|---|---|
| keystroke | the composer's controlled value grew by the characters typed |
| scroll during generation / after | the viewport travelled ≥ 90% of what was commanded |
| reasoning expand/collapse | every pane's `data-state` went open, then all closed |
| stop generation | the run ended, and the character count stopped growing |
| settings open/scroll/close | the dialog appeared, its body travelled, it closed |
| model change | an option was clicked and the menu closed |
| composer short/medium/very long | the composer held every length it was given |
| copy markdown | the clipboard was non-empty afterwards |
| select text | the selection covered ≥ half the **visible** characters |
| select-all + copy | the selection was non-empty |
| image upload | the composer's attachment count rose |
| thread reopen | the thread came back with the same message count |
| message menu | it opened **and** closed **and** had a non-zero item count |
| delete | the `[data-role]` count dropped |

**An action that did not happen is `ran: false`, never a fast timing.** The Radix menu trigger
opens on `pointerdown`, so `element.click()` leaves it shut and the column reads a tidy small
number. A jump scroll from the bottom is read by the intent-aware autoscroll as programmatic and
snapped back, so the viewport lands where it started and the timing is precise and about nothing.

### The instruments

| name | level | what it reads |
|---|---|---|
| `frames` | 0 | one self-rescheduling rAF loop, timer lag, long tasks, CDP presented frames |
| `input` | 0 | keystroke-to-paint from the page side of a real key event |
| `rss` | 0 | the whole browser tree's RSS, on a thread |
| `glass` | 1 | `scrollHeight` reads, `scrollTop` writes, the stabilizer property, mutation records |

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
  - The `/@vite/client` probe checks **what came back**, not just the status: Studio serves its SPA
    for any unknown path, so a production build answers 200 with `index.html`.
  - The marker regex accepts **backticks**: the production bundle is minified with a pass that
    rewrites short string literals as template literals.
  - **`jsxDEV` is never grepped.** `hast-util-to-jsx-runtime`, which Streamdown pulls in, ships its
    own option guard naming `jsxDEV`, so the grep fails a perfectly good production build.
- **No bare zeros.** Every numeric key that can legitimately be zero carries a sibling
  `<key>_attempted`. An unmeasurable quantity is `null` with a `<key>_reason`.
- **Three clocks.** rAF, a 1 ms timer, and CDP presented frames. More than 20% disagreement marks
  the window `clocks_agree: false` and the report layer excludes it from scoring.

## Output

`report/payload.jsonl` — one JSON object per line, flushed and fsynced as it is produced, so a
renderer crash at rung 4 still ships rungs 1 to 3 plus the crash record. A cell that could not
complete emits a `cell` row with `completed: false`, its failure mode and its RSS at death.

## Portability

`python -m tests.studio.studiobench.build` produces `dist/studiobench.pyz`, one file. The bootstrap
is stdlib-only and Playwright is imported lazily, so `--help` and `--doctor` work on a machine with
nothing installed — which is the machine where `--doctor` has to say what is missing.

The default engine matches the tester's desktop webview family: Windows → Chromium via
`channel=msedge` (WebView2), macOS → WebKit (WKWebView), Linux → WebKit, **labelled a proxy for
WebKitGTK** rather than presented as the real thing.

## Layers

Layer 1 (this) owns the real-path session. Layer 2 owns tracing and analysis, Layer 3 owns the
ablation arms and the report. `INTERFACES.md` is the contract between them.
