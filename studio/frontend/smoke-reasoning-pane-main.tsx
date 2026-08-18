// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Harness page for tests/studio/playwright_reasoning_pane.py: the real Thread STREAMING a long
// reasoning part, so what is measured is what a user watching a thinking model actually pays.
//
// WHY THIS EXISTS SEPARATELY FROM smoke-heavy-thread
//
// smoke-heavy-thread seeds a FINISHED thread through `thread.import` and its content mix is
// text, code fences, tool calls and images. It has no reasoning part at all, and it could not
// usefully have one: a finished reasoning group resolves to CLOSED
// (features/chat/utils/reasoning-visibility.ts -> resolveReasoningOpen returns false when
// `isStreaming` is false and the user has not opened it by hand), and Radix unmounts a closed
// CollapsibleContent, so a seeded thread carries ZERO reasoning DOM. Every number that harness
// has ever published was taken on a thread with no thinking block in it.
//
// The cost lives on the streaming path, so this fixture streams. The reasoning group is open
// for the whole run because that is what `resolveReasoningOpen` returns while a group is
// receiving deltas, which is what a user sees.
//
// WHAT IT REPRODUCES
//
// A field trace from Unsloth Desktop on Arch / Wayland / WebKitGTK 2.52.5, one long generation,
// sampled every 5 seconds:
//
//     t=5s     fps 59.4   reasoningChars  2,256   reasoningCodeSpans      0   elements    578
//     t=130s   fps 39.7   reasoningChars 45,943   reasoningCodeSpans  3,520   elements  4,755
//     t=216s   fps 20.5   reasoningChars 73,178   reasoningCodeSpans 11,875   elements 13,688
//     t=241s   fps 18.2   reasoningChars 80,434   reasoningCodeSpans 14,433   elements 16,513
//     t=271s   fps 28.3   reasoningChars 90,262   reasoningCodeSpans 16,186   elements 18,536
//     t=276s   run COMPLETED, reasoningCodeSpans 0, elements 621, fps recovers
//
// So the fixture's shape is fixed by the trace: ONE reasoning part reaching ~90,000 characters,
// dense enough in fenced code to reach ~16,000 `pre code span` inside `.aui-reasoning-text`.
// `reasoningCodeSpans` is the trace's own selector and it is reproduced verbatim below.
//
// The completion row is part of the fixture, not an afterthought: the drop from 18,536 elements
// to 621 is what proves the cost is IN the reasoning pane rather than merely correlated with it.
// So the adapter always emits a short final text answer after the reasoning, which flips
// `message.status.type` off `running`, closes the group and unmounts its content.
//
// Content is UNIQUE per unit on purpose. The highlighter caches tokens keyed on the exact source
// string, so a fixture that repeats one fence would highlight it once and hand back the cached
// result for every copy, and the Shiki cost would vanish from the curve.

import { Thread } from "@/components/assistant-ui/thread";
import { TooltipProvider } from "@/components/ui/tooltip";
import {
  AssistantRuntimeProvider,
  type ChatModelAdapter,
  useAui,
  useLocalRuntime,
} from "@assistant-ui/react";
import {
  RouterProvider,
  createMemoryHistory,
  createRootRoute,
  createRouter,
} from "@tanstack/react-router";
import { type ReactElement, useEffect } from "react";
import { createRoot } from "react-dom/client";
import "./src/index.css";

// The local runtime's thread list item reports a synthetic `__LOCALID_...` remoteId, which is
// truthy, so the fork-count badge really does fire one GET per assistant message. Answering them
// here, before anything mounts, keeps that off the wire entirely.
const realFetch = window.fetch.bind(window);
window.fetch = (input, init) => {
  const url =
    typeof input === "string" ? input : ((input as Request).url ?? String(input));
  if (url.includes("/api/")) {
    return Promise.resolve(
      new Response("{}", {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
  }
  return realFetch(input, init);
};

// ── the reasoning fixture ───────────────────────────────────────────
//
// Thinking text from a model that is working through code looks like this: a line or two of
// deliberation, then a fence it is drafting, then more deliberation. The trace's ratio of
// characters to highlight spans (90,262 / 16,186 = 5.6 characters per span) says the content is
// overwhelmingly fenced code, because prose contributes characters and no spans at all. So the
// units below are mostly fence.

const DELIBERATION = [
  "Let me reconsider the scoring step; the weighting is not obviously right yet.",
  "Wait, that misses the empty rows, so the denominator is wrong. Try again.",
  "I should check the boundary where the batch is smaller than the window.",
  "That reads better. Now the client half has to agree with it exactly.",
  "Hold on, the label filter has to run before the sum, not after.",
  "Writing it out makes the off-by-one obvious, so let me redo the loop.",
];

/**
 * A python fence unique to `index`, sized to about `targetChars`.
 *
 * Dense on purpose: identifiers, punctuation, strings and numbers are each a separate token, and
 * the trace's span density is what the fixture has to hit.
 */
function pythonFence(index: number, targetChars: number): string {
  const lines = [
    "```python",
    `# draft ${index}: weighted batch scorer`,
    "from dataclasses import dataclass",
    "",
    "@dataclass",
    `class Row${index}:`,
    "    weight: float",
    "    count: int",
    "    label: str",
    "",
  ];
  let step = 0;
  while (lines.join("\n").length < targetChars) {
    lines.push(
      `def score_${index}_${step}(rows: list[Row${index}], cap: float = ${step + 1}.${(index * 7 + step) % 97}) -> float:`,
      `    """Draft ${step} of ${index}: sum weighted counts, skip the empties."""`,
      "    total, seen = 0.0, 0",
      "    for row in rows:",
      `        if row.count == 0 or row.label == "skip-${step}":`,
      "            continue",
      `        total += min(cap, row.weight * row.count * ${(step % 13) + 1}.${(index + step) % 89})`,
      "        seen += 1",
      "    return total / seen if seen else 0.0",
      "",
    );
    step += 1;
  }
  lines.push("```");
  return lines.join("\n");
}

function typescriptFence(index: number, targetChars: number): string {
  const lines = [
    "```typescript",
    `// draft ${index}: the client half, so one grammar is not the whole story`,
    `export interface Row${index} {`,
    "  readonly weight: number;",
    "  readonly count: number;",
    "  readonly label: string;",
    "}",
    "",
  ];
  let step = 0;
  while (lines.join("\n").length < targetChars) {
    lines.push(
      `export function score${index}_${step}(rows: readonly Row${index}[], cap = ${step + 1}.${(index * 5 + step) % 89}): number {`,
      "  let total = 0, seen = 0;",
      "  for (const row of rows) {",
      `    if (row.count === 0 || row.label === "skip-${step}") continue;`,
      `    total += Math.min(cap, row.weight * row.count * ${(step % 11) + 1}.${(index + step) % 73});`,
      "    seen += 1;",
      "  }",
      "  return seen === 0 ? 0 : total / seen;",
      "}",
      "",
    );
    step += 1;
  }
  lines.push("```");
  return lines.join("\n");
}

function jsonFence(index: number, targetChars: number): string {
  // One very long line on purpose. A single-line document is the shape that behaves worst in a
  // highlighter and in any code that scans for line boundaries.
  const entries: string[] = [];
  let step = 0;
  let body = "";
  while (body.length < targetChars) {
    entries.push(
      `{"id":"row-${index}-${step}","weight":${(step % 17) + 0.5},"count":${step * 3 + index},"label":"batch-${index}-${step}"}`,
    );
    body = `{"draft":${index},"rows":[${entries.join(",")}]}`;
    step += 1;
  }
  return ["```json", body, "```"].join("\n");
}

const FENCES = [pythonFence, typescriptFence, pythonFence, jsonFence];

/** Deliberation prose unique to `index`, about `targetChars` long. */
function deliberation(index: number, targetChars: number): string {
  if (targetChars <= 0) return DELIBERATION[index % DELIBERATION.length];
  const parts: string[] = [];
  let length = 0;
  let cursor = index;
  while (length < targetChars) {
    const line = `${DELIBERATION[cursor % DELIBERATION.length]} (pass ${cursor - index + 1} of draft ${index})`;
    parts.push(line);
    length += line.length + 2;
    cursor += 1;
  }
  return parts.join("\n\n");
}

/**
 * The whole reasoning part, as one string of about `totalChars`.
 *
 * Two knobs, because length and span DENSITY are separate variables and the trace pins both.
 * `fenceChars` sets how much fenced code each unit carries, `proseChars` how much prose sits
 * between the fences. Prose contributes characters and no highlight spans at all, so the ratio
 * of the two is what moves characters-per-span.
 *
 * The trace's 90,262 characters carried 16,186 spans, i.e. 5.6 characters per span. An earlier
 * all-fence version of this fixture measured 3.47, which would have put ~26,000 spans on a
 * 90,000-character run: a harder workload than the user's, which would have overstated every
 * number taken from it. The defaults below are set from that measurement, and the driver prints
 * the density it achieved so the calibration can be checked rather than trusted.
 */
export function buildReasoning(
  totalChars: number,
  fenceChars: number,
  proseChars: number,
  preambleChars: number,
): string {
  const parts: string[] = [];
  let length = 0;
  let index = 0;
  // A stretch of thinking with NO completed fence in it, first, because that is where the
  // measured onset is.
  //
  // In the field capture the page held 60.0 fps and 0 highlight spans for the first 90 seconds
  // and 33,348 characters of reasoning, and lost a frame only once the first fence CLOSED. Prose
  // costs 8.3 pane elements per 1000 characters; fenced code costs 169, twenty times as much. A
  // fixture whose first unit is a fence starts on the expensive side of that step and cannot show
  // where the step is.
  while (length < preambleChars) {
    const unit = `${deliberation(index, 900)}\n\n`;
    parts.push(unit);
    length += unit.length;
    index += 1;
  }
  while (length < totalChars) {
    const lead = deliberation(index, proseChars);
    const fence = FENCES[index % FENCES.length](index, fenceChars);
    const unit = `${lead}\n\n${fence}\n\n`;
    parts.push(unit);
    length += unit.length;
    index += 1;
  }
  return parts.join("").slice(0, totalChars);
}

const ANSWER =
  "Here is the scorer, with the empty rows skipped and the cap applied before the mean.";

type RunOptions = {
  totalChars?: number;
  fenceChars?: number;
  proseChars?: number;
  preambleChars?: number;
  chunkChars?: number;
  gapMs?: number;
  footnoteAtChars?: number;
};

/**
 * A GFM footnote REFERENCE, appended once the stream passes `footnoteAtChars` and kept for the
 * rest of the run.
 *
 * This is the case that closed PR #9073, reproduced live rather than argued about. A footnote
 * reference makes this renderer treat the WHOLE document as one block, so the moment it arrives
 * the block list collapses from hundreds of entries to one, retroactively, behind content that
 * has already been rendered and measured. Any windowing scheme that carries a position across
 * that moment carries it into a document that no longer has the divisions the position was
 * expressed in.
 */
const LATE_FOOTNOTE = "\n\nA late reference[^z] appears.\n\n[^z]: the definition.";

let config: Required<RunOptions> = {
  totalChars: 90_000,
  fenceChars: 1_800,
  proseChars: 1_250,
  // A quarter of the body, matching the field capture's 33,348 of 131,350 characters before the
  // first fence closed.
  preambleChars: 22_500,
  chunkChars: 24,
  gapMs: 2,
  // Off by default: every measurement in the matrix runs without it, so it cannot quietly change
  // a number that is compared against another arm.
  footnoteAtChars: 0,
};

const stream = {
  startedAt: 0,
  sentChars: 0,
  arrivals: 0,
  reasoningEndedAtMs: null as number | null,
  endedAtMs: null as number | null,
  footnoteAtMs: null as number | null,
  totalChars: 0,
};

const sleep = (ms: number) =>
  new Promise((resolve) => {
    setTimeout(resolve, ms);
  });

// Stands in for the network, not the renderer: the same growing cumulative reasoning text a
// thinking model's adapter yields, at a fixed rate.
//
// Cumulative, not delta: that is what assistant-ui's local runtime contract wants and what the
// app's own adapter sends, and it is also the shape that makes a whole-subtree re-render
// possible, which is one of the things under investigation here.
const adapter: ChatModelAdapter = {
  async *run() {
    const reasoning = buildReasoning(
      config.totalChars,
      config.fenceChars,
      config.proseChars,
      config.preambleChars,
    );
    stream.totalChars = reasoning.length;
    stream.startedAt = performance.now();
    let cursor = 0;
    while (cursor < reasoning.length) {
      cursor = Math.min(reasoning.length, cursor + config.chunkChars);
      stream.arrivals += 1;
      stream.sentChars = cursor;
      const late =
        config.footnoteAtChars > 0 && cursor >= config.footnoteAtChars
          ? LATE_FOOTNOTE
          : "";
      if (late && stream.footnoteAtMs === null) {
        stream.footnoteAtMs = performance.now() - stream.startedAt;
      }
      yield {
        content: [
          { type: "reasoning" as const, text: reasoning.slice(0, cursor) + late },
        ],
      };
      await sleep(config.gapMs);
    }
    stream.reasoningEndedAtMs = performance.now() - stream.startedAt;
    // The final answer is what ends the reasoning round. Without it the group's
    // `isReasoningStreaming` would go false only because the message stopped running, and the
    // completion row of the trace -- the collapse and the fps recovery -- would still happen but
    // would not be the shape a real reply has.
    yield {
      content: [
        {
          type: "reasoning" as const,
          text: reasoning + (config.footnoteAtChars > 0 ? LATE_FOOTNOTE : ""),
        },
        { type: "text" as const, text: ANSWER },
      ],
    };
    stream.endedAtMs = performance.now() - stream.startedAt;
  },
};

// ── the page API ────────────────────────────────────────────────────

function ReasoningPaneApi(): null {
  const aui = useAui();

  useEffect(() => {
    const api = {
      /** Start the generation. Returns immediately; poll `snapshot()`. */
      run(options: RunOptions = {}): void {
        config = { ...config, ...options };
        stream.startedAt = 0;
        stream.sentChars = 0;
        stream.arrivals = 0;
        stream.reasoningEndedAtMs = null;
        stream.endedAtMs = null;
        stream.footnoteAtMs = null;
        // From a later task, so the runtime's own startup does not land inside a measurement
        // window that the driver opened in this one.
        setTimeout(() => {
          void aui.thread().append({
            role: "user",
            content: [{ type: "text", text: "think this through and show the code" }],
          });
        }, 0);
      },
      /**
       * The trace's columns, and nothing else in the hot path.
       *
       * `reasoningCodeSpans` is the trace's selector verbatim: `pre code span` scoped to
       * `.aui-reasoning-text`. Scoping matters -- the same selector unscoped would also count the
       * answer's fences and every other message's, and the whole point of the completion row is
       * that the reasoning ones go to zero while the rest of the thread does not.
       */
      sample(): Record<string, number | boolean> {
        const at = performance.now();
        const pane = document.querySelector(".aui-reasoning-text");
        const reasoningChars = pane ? (pane.textContent ?? "").length : 0;
        const reasoningCodeSpans = document.querySelectorAll(
          ".aui-reasoning-text pre code span",
        ).length;
        const totalElements = document.getElementsByTagName("*").length;
        // The field capture reports `totalElements - reasoningElements` as a CONSTANT 578 for
        // every sample the pane is mounted, i.e. every node the page gains is the thinking pane
        // and nothing else in the app grows at all. Counting both is what lets that be checked
        // here rather than assumed.
        const reasoningElements = document.querySelectorAll(".aui-reasoning-text *").length;
        return {
          sampleCostMs: Math.round((performance.now() - at) * 100) / 100,
          reasoningChars,
          reasoningCodeSpans,
          reasoningElements,
          elementsOutsideReasoning: totalElements - reasoningElements,
          reasoningPanes: document.querySelectorAll(".aui-reasoning-text").length,
          reasoningOpen: Boolean(
            document.querySelector('[data-slot="reasoning-root"][data-state="open"]'),
          ),
          totalElements,
          allCodeSpans: document.querySelectorAll("pre code span").length,
          sentChars: stream.sentChars,
          arrivals: stream.arrivals,
          streamDone: stream.endedAtMs !== null,
        };
      },
      /** Stream bookkeeping the DOM cannot answer. */
      streamState(): Record<string, number | null | boolean> {
        return {
          startedAt: stream.startedAt,
          sentChars: stream.sentChars,
          arrivals: stream.arrivals,
          totalChars: stream.totalChars,
          reasoningEndedAtMs: stream.reasoningEndedAtMs,
          endedAtMs: stream.endedAtMs,
          footnoteAtMs: stream.footnoteAtMs,
          done: stream.endedAtMs !== null,
        };
      },
      /** What the fixture would build, without building the page. Used to calibrate density. */
      preview(
        totalChars: number,
        fenceChars: number,
        proseChars: number,
        preambleChars: number,
      ): number {
        return buildReasoning(totalChars, fenceChars, proseChars, preambleChars).length;
      },
    };
    (window as unknown as { __reasoningPane: typeof api }).__reasoningPane = api;
  }, [aui]);

  return null;
}

function Harness(): ReactElement {
  const runtime = useLocalRuntime(adapter);
  return (
    <TooltipProvider>
      <AssistantRuntimeProvider runtime={runtime}>
        <ReasoningPaneApi />
        {/* Thread is flex-1 basis-0 min-h-0, so it needs a bounded flex parent to scroll. */}
        <div
          data-smoke="reasoning-pane"
          style={{ display: "flex", flexDirection: "column", height: "100vh" }}
        >
          <Thread hideWelcome={true} />
        </div>
      </AssistantRuntimeProvider>
    </TooltipProvider>
  );
}

// Thread reaches useNavigate (the fork action, the composer tools menu). Without a router in
// context tanstack's useRouter still works, but console.warns on every render of every action
// bar, which scales with the thread and is serialised over the debugging channel.
const rootRoute = createRootRoute({ component: Harness });
const router = createRouter({
  routeTree: rootRoute,
  history: createMemoryHistory({ initialEntries: ["/"] }),
});

const root = document.getElementById("root");
if (!root) {
  throw new Error("missing #root");
}
createRoot(root).render(<RouterProvider router={router as unknown as never} />);
