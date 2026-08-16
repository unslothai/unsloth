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
// assistant-ui context, and its custom BlockComponent reads `useAuiState`, so mounting it
// outside a provider throws instead of rendering. Driving the real runtime also puts
// assistant-ui's own update scheduling inside the measurement, which is where #7892's
// starvation actually lived.
//
// The reply is a fixed string rather than a model's output. #8845's first measurement
// attempts failed because free-form sampling gave the two sides different essays, and the
// renderer's cost is superlinear in length, so a comparison across different text says
// nothing.

/* eslint-disable no-restricted-imports -- a measurement entry point, not app code. */
// This store first, and deliberately. MarkdownText's import graph reaches the chat barrel,
// and the settings General tab reaches back into it for SIDEBAR_ORGANIZATION_STORAGE_KEY,
// so entering that cycle from the renderer leaves the constant in its temporal dead zone
// and the harness renders nothing at all. Evaluating the store module first breaks the tie
// the same way the app's own entry order does.
import "@/features/chat/stores/sidebar-organization-store";
/* eslint-enable no-restricted-imports */

import { MarkdownText } from "@/components/assistant-ui/markdown-text";
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
import "./src/index.css";

/**
 * Deterministic, and shaped like what the renderer is actually slow on: prose to lay
 * out, a fenced block for Shiki, and display math for KaTeX. Built by repetition so the
 * length is a knob rather than a new fixture per size.
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
  done: boolean;
};

type RunOptions = { totalChars?: number; chunkChars?: number; gapMs?: number };

declare global {
  interface Window {
    __stream: {
      ready: boolean;
      run(options?: RunOptions): void;
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
  done: false,
};

/** Consecutive frames without growth that count as settled. */
const SETTLED_FRAMES = 30;

let config: Required<RunOptions> = {
  totalChars: 24_000,
  chunkChars: 24,
  gapMs: 2,
};

const sleep = (ms: number) =>
  new Promise((resolve) => {
    setTimeout(resolve, ms);
  });

// Stands in for the network, not for the renderer: it yields the same growing
// cumulative text the chat adapter yields, at a fixed rate.
const adapter: ChatModelAdapter = {
  async *run() {
    const reply = buildReply(config.totalChars);
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
    state.streamEndedAtMs = performance.now() - state.startedAt;
  },
};

function AssistantMessage(): ReactElement {
  return (
    <div className="min-w-0 max-w-full">
      <MessagePrimitive.Parts components={{ Text: MarkdownText }} />
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
    // Painted, not published: read the DOM, because "the bubble stopped growing" is
    // the complaint, and a store update that never reaches paint does not answer it.
    const paintedChars = (): number => {
      const node = document.querySelector("[data-status]");
      return node ? (node.textContent ?? "").length : 0;
    };

    let lastGrowthAt = 0;
    let lastFrameAt = 0;
    let settledChars = 0;
    let quietFrames = 0;
    let handle = requestAnimationFrame(function watch(now: number) {
      if (lastFrameAt && now - lastFrameAt > 33) {
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
        }
        // Settled is counted in FRAMES without growth, not in wall-clock time.
        // #8845's last failed attempt used a 1.5s quiet window and declared a
        // reply finished in the middle of a freeze. A freeze blocks the frame
        // loop as well, so a frame counter cannot tick through one, which is
        // exactly the property that window lacked. Rendered length is compared
        // against itself rather than against the bytes sent, because Markdown
        // syntax (fences, list markers, math delimiters) never reaches
        // textContent.
        if (state.streamEndedAtMs !== null && state.timeToFullyPaintedMs === null) {
          if (painted > settledChars) {
            settledChars = painted;
            quietFrames = 0;
          } else {
            quietFrames += 1;
            if (quietFrames >= SETTLED_FRAMES) {
              state.settledChars = settledChars;
              state.timeToFullyPaintedMs = now - state.startedAt;
              state.done = true;
            }
          }
        }
      }
      handle = requestAnimationFrame(watch);
    });

    // `observe({type})` is specified to abort silently on an entry type the engine does not
    // support, not to throw, so a try/catch around it never fires and the long-task total
    // stays at its initial 0 -- a perfect score on the budget that matters most. Only
    // supportedEntryTypes actually answers the question, and the answer is recorded so the
    // driver can fail the run rather than report the zero. Chromium is the only engine that
    // ships longtask (Gecko bug 1348405 open, WebKit never shipped it).
    let observer: PerformanceObserver | null = null;
    state.longTaskSupported =
      typeof PerformanceObserver !== "undefined" &&
      (PerformanceObserver.supportedEntryTypes ?? []).includes("longtask");
    if (state.longTaskSupported) {
      observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          state.longTasks += 1;
          state.longTaskMs += entry.duration;
        }
      });
      observer.observe({ type: "longtask", buffered: true });
    }

    window.__stream = {
      ready: true,
      run(options: RunOptions = {}) {
        config = { ...config, ...options };
        void runtime.thread.append({
          role: "user",
          content: [{ type: "text", text: "stream the fixture" }],
        });
      },
      results: () => ({ ...state }),
    };

    return () => {
      cancelAnimationFrame(handle);
      observer?.disconnect();
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
