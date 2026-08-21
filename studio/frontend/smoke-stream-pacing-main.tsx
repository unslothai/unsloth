// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Harness for tests/studio/playwright_stream_pacing.py: the real MarkdownText inside a real
// assistant-ui runtime, fed a fixed reply at a fixed rate, so the measured main-thread cost
// is the chat renderer's own. Same shape as smoke-autoscroll.html and smoke-research.html:
// a vite entry, no backend, no auth, no model.
//
// Four merged PRs moved this path (#7892 transition starvation, #8750 incremental Markdown,
// #8845 publish coalescing, #8935 incremental fence tokenization) and each rebuilt a
// throwaway harness to prove it. The number they all reported is `longestStallMs`: the
// longest the bubble stops growing while text is still arriving.
//
// A local runtime rather than a bare component: MarkdownText reads its message part from
// assistant-ui context and its BlockComponent reads `useAuiState`, so mounting it outside a
// provider throws. The real runtime also puts assistant-ui's own update scheduling inside
// the measurement, which is where #7892's starvation lived.
//
// The reply is a fixed string, not a model's output. #8845's first measurement attempts
// failed because free-form sampling gave the two sides different essays, and the renderer's
// cost is superlinear in length, so comparing across different text says nothing.

/* eslint-disable no-restricted-imports -- a measurement entry point, not app code. */
// This store first, deliberately. MarkdownText's import graph reaches the chat barrel and the
// settings General tab reaches back for SIDEBAR_ORGANIZATION_STORAGE_KEY, so entering that
// cycle from the renderer leaves the constant in its temporal dead zone and the harness
// renders nothing. Evaluating the store first breaks the tie, as the app's entry order does.
import "@/features/chat/stores/sidebar-organization-store";
// And the chat barrel before MarkdownText, as the app's entry does. thread.tsx builds
// ASSISTANT_PART_COMPONENTS at module scope with `Text: MarkdownText`, so entering the
// markdown-text -> features/chat -> chat-page -> thread cycle from markdown-text runs that
// object literal while the MarkdownText binding is still in its temporal dead zone and the
// page dies with "Cannot access 'MarkdownText' before initialization".
import "@/features/chat";
/* eslint-enable no-restricted-imports */
import {
  MarkdownText,
  type MarkdownCodeHighlighting,
  observeStreamingCodeHighlights,
} from "@/components/assistant-ui/markdown-text";
import { OVERSIZED_OPEN_CODE_CHARS } from "@/components/assistant-ui/streaming-code-policy";

import { observeIncrementalMarkdownRenders } from "@/components/assistant-ui/streaming-render-schedule";
import {
  AssistantRuntimeProvider,
  type ChatModelAdapter,
  MessagePrimitive,
  ThreadPrimitive,
  useAui,
  useLocalRuntime,
} from "@assistant-ui/react";
import { type ReactElement, useEffect } from "react";
import { createRoot } from "react-dom/client";
import { stallInProgress } from "./smoke-stream-pacing-stall.ts";
import "./src/index.css";

/**
 * Deterministic, and shaped like what the renderer is slow on: prose to lay out, a fenced
 * block for Shiki, display math for KaTeX. Repeated so length is a knob, not a new fixture.
 */
export function buildReply(totalChars: number): string {
  const units = [
    "The printing press did not arrive as a single invention so much as a " +
      "convergence: movable type, a workable oil-based ink, and a screw press " +
      "already used for wine and olives. Each existed before Mainz; what changed " +
      "was that one workshop held all three at once.\n\n",
    "```ts\nfunction paginate(sheets: number, perSheet: number): number[] {\n" +
      "  const out: number[] = [];\n  for (let i = 0; i < sheets; i += 1) {\n" +
      "    out.push(i * perSheet);\n  }\n  return out;\n}\n```\n\n",
    "Setting a single page took a compositor the better part of a day, so the " +
      "economics only worked above a threshold print run. That threshold is what " +
      "the arithmetic below expresses.\n\n",
    "$$\nc_{\\text{unit}} = \\frac{F + vn}{n} = v + \\frac{F}{n}\n$$\n\n",
    "- Fixed cost dominates at small runs\n- Variable cost dominates at large ones\n" +
      "- The crossover moved with paper prices, not with the press\n\n",
  ];
  let text = "";
  let index = 0;
  while (text.length < totalChars) {
    text += units[index % units.length];
    index += 1;
  }
  return text.slice(0, totalChars);
}

type FenceMarker = "`" | "~";
const MODERATE_FENCE_SOURCE_CODE_UNITS = 7_000;

const OPEN_FENCE_RICH_PREFIX =
  "## Mixed terminal-fence fixture\n\n" +
  "The stable prefix keeps **bold text**, *emphasis*, `inline code`, and a " +
  "[safe link](https://example.com) while code grows[^stable].\n\n" +
  "- first stable item\n- second stable item\n\n" +
  "| phase | renderer |\n| --- | --- |\n| open | plain code |\n\n" +
  "> This quote must not remount during terminal-fence growth.\n\n" +
  "$$x^2 + y^2 = z^2$$\n\n" +
  "[^stable]: The footnote keeps this provider block globally scoped.\n\n";


const OPEN_FENCE_COMPACT_GLOBAL_PREFIX =
  "## Mixed terminal-fence fixture\n\n" +
  "A compact global claim keeps provider-wide definitions active[^stable].\n\n" +
  "[^stable]: stable note\n\n";

export function buildOpenFence(
  marker: FenceMarker,
  sourceCodeUnits = MODERATE_FENCE_SOURCE_CODE_UNITS,
  sourceEndsWithLineEnding = true,
  followedByProse = false,
  globalScoped = true,

  richPrefix = true,
): {
  closedMarkdown: string;
  openMarkdown: string;
  source: string;
  prefix: string;
} {
  const lineEnding = marker === "~" ? "\r\n" : "\n";
  const markerLength = marker === "~" ? 4 : 3;
  const fence = marker.repeat(markerLength);
  const prefix = (
    globalScoped && !richPrefix
      ? OPEN_FENCE_COMPACT_GLOBAL_PREFIX
      : globalScoped
        ? OPEN_FENCE_RICH_PREFIX
        : OPEN_FENCE_RICH_PREFIX.replace(" grows[^stable].", " grows.").replace(
            "[^stable]: The footnote keeps this provider block globally scoped.\n\n",
            "",
          )
  ).replaceAll("\n", lineEnding);
  const opening = `${fence}typescript title="source-fidelity"${lineEnding}`;
  const line =
    'const mixedRow0000: number = 0000; // stable terminal fence payload';
  const canonicalSourceLine = `${line}\n`;
  let canonicalSource: string;
  if (sourceEndsWithLineEnding) {
    canonicalSource = canonicalSourceLine
      .repeat(Math.ceil(sourceCodeUnits / canonicalSourceLine.length))
      .slice(0, sourceCodeUnits - 1);
    canonicalSource += "\n";
  } else {
    canonicalSource = canonicalSourceLine
      .repeat(Math.ceil(sourceCodeUnits / canonicalSourceLine.length))
      .slice(0, sourceCodeUnits)
      .replace(/\n+$/, (ending) => " ".repeat(ending.length));
  }
  const source = canonicalSource.replaceAll("\n", lineEnding);
  const openMarkdown = prefix + opening + source;
  return {
    // The source already owns its final line ending, matching the iteration-4
    // 7,000-character fixture; the closer begins on the following line.
    closedMarkdown: `${openMarkdown}${
      sourceEndsWithLineEnding ? "" : lineEnding
    }${fence}${
      followedByProse
        ? `${lineEnding}${lineEnding}## After the oversized fence${lineEnding}${lineEnding}${Array.from(
            { length: globalScoped ? 1 : 12 },
            (_, index) =>
              `Following block ${index} keeps same-update promotion honest.${lineEnding}${lineEnding}`,
          ).join("")}`
        : ""
    }`,
    openMarkdown,
    prefix,
    source,
  };
}

type OpenFenceFixture = ReturnType<typeof buildOpenFence>;

type Results = {
  startedAt: number;
  arrivals: number;
  sentChars: number;
  paintedChars: number;
  longestStallMs: number;
  timeToFullyPaintedMs: number | null;
  streamEndedAtMs: number | null;
  longTaskMs: number;
  longTasks: number;
  longTaskSupported: boolean;
  framesOver33ms: number;
  settledChars: number;
  codeHighlightCalls: number;

  renderPlans: Array<{
    codeFenceSourceLengths: readonly number[];
    isStreaming: boolean;
    sourceLength: number;
    terminalSourceLength: number | null;
  }>;
  done: boolean;
  paused: boolean;
};

type OpenFenceRunOptions = {
  codeHighlighting?: MarkdownCodeHighlighting;
  followedByProse?: boolean;
  globalScoped?: boolean;
  richPrefix?: boolean;
  marker: FenceMarker;
  sourceCodeUnits?: number;
  sourceEndsWithLineEnding?: boolean;
};

type RunOptions = {
  codeHighlighting?: MarkdownCodeHighlighting;

  paginateReasoning?: boolean;
  totalChars?: number;
  chunkChars?: number;
  gapMs?: number;
};

declare global {
  interface Window {
    __stream: {
      ready: boolean;
      run(options?: RunOptions): void;
      runOpenFence(options: OpenFenceRunOptions): void;
      completeOpenFence(): void;
      expectedOpenCode(): string;
      expectedOpenPrefix(): string;
      results(): Results;
    };
  }
}

const state: Results = {
  startedAt: 0,
  arrivals: 0,
  sentChars: 0,
  paintedChars: 0,
  longestStallMs: 0,
  timeToFullyPaintedMs: null,
  streamEndedAtMs: null,
  longTaskMs: 0,
  longTasks: 0,
  longTaskSupported: false,
  framesOver33ms: 0,
  settledChars: 0,
  codeHighlightCalls: 0,

  renderPlans: [],
  done: false,
  paused: false,
};

let activeCodeHighlighting: MarkdownCodeHighlighting = "syntax";

let activePaginateReasoning = false;

let openFenceFixture: OpenFenceFixture | null = null;
let releaseOpenFence: (() => void) | null = null;

/** Consecutive frames without growth that count as settled. */
const SETTLED_FRAMES = 30;

let config: Required<
  Omit<RunOptions, "codeHighlighting" | "paginateReasoning">
> = {
  totalChars: 24_000,
  chunkChars: 24,
  gapMs: 2,
};

const sleep = (ms: number) =>
  new Promise((resolve) => {
    setTimeout(resolve, ms);
  });

// Stands in for the network, not the renderer: the same growing cumulative text the chat
// adapter yields, at a fixed rate.
const adapter: ChatModelAdapter = {
  async *run() {
    const fixture = openFenceFixture;
    const reply = fixture?.openMarkdown ?? buildReply(config.totalChars);
    let cursor = 0;
    state.startedAt = performance.now();
    while (cursor < reply.length) {
      cursor = Math.min(reply.length, cursor + config.chunkChars);
      state.arrivals += 1;
      state.sentChars = cursor;
      yield {
        content: [{ type: "text" as const, text: reply.slice(0, cursor) }],
      };
      await sleep(config.gapMs);
    }

    if (fixture) {
      state.paused = true;
      await new Promise<void>((resolve) => {
        releaseOpenFence = resolve;
      });
      releaseOpenFence = null;
      state.paused = false;
      state.arrivals += 1;
      state.sentChars = fixture.closedMarkdown.length;
      yield {
        content: [{ type: "text" as const, text: fixture.closedMarkdown }],
      };
    }
    state.streamEndedAtMs = performance.now() - state.startedAt;
  },
};
function PolicyMarkdownText(props: { status?: unknown; text?: string }): ReactElement {
  return (
    <MarkdownText
      {...props}
      codeHighlighting={activeCodeHighlighting}
      paginateReasoning={activePaginateReasoning}
    />
  );
}



function AssistantMessage(): ReactElement {
  return (
    <div className="min-w-0 max-w-full">
      <MessagePrimitive.Parts components={{ Text: PolicyMarkdownText }} />
    </div>
  );
}

function NoUserMessage(): null {
  return null;
}

function Harness(): ReactElement {
  const runtime = useLocalRuntime(adapter);
  const aui = useAui({});

  useEffect(() => {
    const stopObservingRenderPlans = observeIncrementalMarkdownRenders(
      (observation) => {
        state.renderPlans.push(observation);
        if (state.renderPlans.length > 40) state.renderPlans.shift();
      },
    );

    const stopObservingCodeHighlights = observeStreamingCodeHighlights(
      (source) => {
        if (
          source.length >= OVERSIZED_OPEN_CODE_CHARS &&
          source.includes("mixedRow0000")
        ) {
          state.codeHighlightCalls += 1;
        }
      },
    );
    // Painted, not published: "the bubble stopped growing" is the complaint, and a store
    // update that never reaches paint does not answer it. So read the DOM.
    const paintedChars = (): number => {
      const node = document.querySelector("[data-status]");
      return node ? (node.textContent ?? "").length : 0;
    };

    let lastGrowthAt = 0;
    let lastFrameAt = 0;
    let settledChars = 0;
    let quietFrames = 0;
    let handle = requestAnimationFrame(function watch(now: number) {
      // Windowed like the long tasks: a frame dropped while loading is not the renderer's.
      // 0 of 286 here, but a slower box need not be 0 and the number must compare across both.
      if (now >= measureFrom && lastFrameAt && now - lastFrameAt > 33) {
        state.framesOver33ms += 1;
      }
      lastFrameAt = now;
      if (state.startedAt) {
        if (!lastGrowthAt) {
          lastGrowthAt = state.startedAt;
        }
        const painted = paintedChars();
        if (painted > state.paintedChars) {
          const stall = now - lastGrowthAt;
          if (stall > state.longestStallMs) {
            state.longestStallMs = stall;
          }
          state.paintedChars = painted;
          lastGrowthAt = now;
        } else {
          // Measure the stall in progress: a stall closed only by a LATER paint misses a
          // freeze that runs to the end of the stream, whose lost tail can hide inside the
          // 90% floor. stallInProgress caps it at stream end, so a freeze spanning that
          // moment is still recorded in full while the settle check's own quiet frames are
          // not counted as one.
          const stall = stallInProgress(
            lastGrowthAt,
            now,
            state.startedAt,
            state.streamEndedAtMs,
          );
          if (stall > state.longestStallMs) {
            state.longestStallMs = stall;
          }
        }
        // Settled is counted in FRAMES without growth, not wall clock: #8845's last
        // failed attempt used a 1.5s quiet window and called a reply finished mid
        // freeze. A freeze blocks the frame loop too, so a frame counter cannot tick
        // through one. Rendered length is compared against itself, not the bytes sent,
        // because Markdown syntax (fences, list markers, math delimiters) never
        // reaches textContent.
        if (
          state.streamEndedAtMs !== null &&
          state.timeToFullyPaintedMs === null
        ) {
          if (painted > settledChars) {
            settledChars = painted;
            quietFrames = 0;
          } else {
            quietFrames += 1;
            if (quietFrames >= SETTLED_FRAMES) {
              // On screen at settlement, not the peak: paintedChars only climbs, so a
              // completion render that truncates the bubble would still pass the workload
              // floor. This is the length the driver checks. Equal to the peak (24,033
              // both) today, so it is a guard, not a live discrepancy.
              state.settledChars = painted;
              state.timeToFullyPaintedMs = now - state.startedAt;
              state.done = true;
            }
          }
        }
      }
      handle = requestAnimationFrame(watch);
    });

    // `observe({type})` aborts silently on an unsupported entry type instead of throwing,
    // so a try/catch never fires and the long-task total stays 0, a perfect score on the
    // budget that matters most. Only supportedEntryTypes answers the question, and the
    // answer is recorded so the driver can fail the run instead of reporting the zero.
    // Chromium alone ships longtask (Gecko bug 1348405 open, WebKit never shipped it).
    // run() moves measureFrom to the moment the stream is asked for, so nothing the page
    // did while loading lands in the budget.
    let measureFrom = Number.POSITIVE_INFINITY;
    let observer: PerformanceObserver | null = null;
    state.longTaskSupported =
      typeof PerformanceObserver !== "undefined" &&
      (PerformanceObserver.supportedEntryTypes ?? []).includes("longtask");
    if (state.longTaskSupported) {
      observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          // Only tasks belonging to the stream. `buffered: true` replays what the timeline
          // already held, on a dev server module evaluation and the first React render:
          // ~140ms in one entry, ~2.6% of the total, and larger on a cold or loaded runner.
          // That is page startup, and budgeting it makes a cold run look like a slow renderer.
          if (entry.startTime < measureFrom) continue;
          state.longTasks += 1;
          state.longTaskMs += entry.duration;
        }
      });
      observer.observe({ type: "longtask", buffered: true });
    }

    window.__stream = {
      ready: true,

      run(options: RunOptions = {}) {
        const {
          codeHighlighting = "syntax",
          paginateReasoning = false,
          ...streamOptions
        } = options;
        activeCodeHighlighting = codeHighlighting;
        activePaginateReasoning = paginateReasoning;
        config = { ...config, ...streamOptions };
        // Open the window and drop what the buffered replay banked. A straddling entry is
        // discarded, not prorated: it began before the stream, so it is not the renderer's.
        measureFrom = performance.now();
        state.longTaskMs = 0;
        state.longTasks = 0;
        state.framesOver33ms = 0;
        state.renderPlans = [];
        state.codeHighlightCalls = 0;
        // Append from a later task. In this same task, a long task is stamped with its
        // task's start, so runtime startup and the first publish would sort before
        // measureFrom and be dropped as page load. A fresh task starts after it.
        setTimeout(() => {
          void runtime.thread.append({
            role: "user",
            content: [{ type: "text", text: "stream the fixture" }],
          });
        }, 0);
      },
      runOpenFence({
        codeHighlighting = "syntax",
        followedByProse,
        globalScoped,
        marker,
        richPrefix,
        sourceCodeUnits,
        sourceEndsWithLineEnding,
      }: OpenFenceRunOptions) {
        activeCodeHighlighting = codeHighlighting;
        openFenceFixture = buildOpenFence(
          marker,
          sourceCodeUnits,
          sourceEndsWithLineEnding,
          followedByProse,
          globalScoped,
          richPrefix,
        );
        state.renderPlans = [];
        state.codeHighlightCalls = 0;
        config = {
          totalChars: openFenceFixture.openMarkdown.length,
          chunkChars: 512,
          gapMs: 0,
        };
        measureFrom = performance.now();
        setTimeout(() => {
          void runtime.thread.append({
            role: "user",
            content: [{ type: "text", text: "stream the open fence fixture" }],
          });
        }, 0);
      },
      completeOpenFence() {
        releaseOpenFence?.();
      },
      expectedOpenCode: () =>
        openFenceFixture?.source.replaceAll("\r\n", "\n") ?? "",
      expectedOpenPrefix: () => openFenceFixture?.prefix ?? "",
      results: () => ({ ...state }),
    };

    return () => {
      cancelAnimationFrame(handle);
      observer?.disconnect();

      stopObservingRenderPlans();

      stopObservingCodeHighlights();
    };
  }, [runtime]);

  return (
    <AssistantRuntimeProvider runtime={runtime} aui={aui}>
      <ThreadPrimitive.Root
        style={{ width: 900, margin: "0 auto", padding: 16 }}
      >
        <ThreadPrimitive.Viewport>
          <ThreadPrimitive.Messages
            components={{ AssistantMessage, UserMessage: NoUserMessage }}
          />
        </ThreadPrimitive.Viewport>
      </ThreadPrimitive.Root>
    </AssistantRuntimeProvider>
  );
}

createRoot(document.getElementById("root") as HTMLElement).render(<Harness />);
