// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Harness for tests/studio/playwright_stream_pacing.py: real MarkdownText in a local runtime fed
// a fixed reply at a fixed rate; reports longestStallMs. Fixed text since cost is superlinear.

/* eslint-disable no-restricted-imports -- a measurement entry point, not app code. */
// Import this store first: entering the cycle from the renderer leaves a constant in TDZ.
import "@/features/chat/stores/sidebar-organization-store";
// Then the chat barrel, else MarkdownText is read in TDZ by thread.tsx.
import "@/features/chat";
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
import { stallInProgress } from "./smoke-stream-pacing-stall.ts";
import "./src/index.css";

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
    // Read the DOM: a store update that never paints does not answer "stopped growing".
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
          // Measure the in-progress stall so a freeze running to stream end is recorded.
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
        // Settled is counted in frames, not wall clock: a freeze blocks the frame loop too.
        if (state.streamEndedAtMs !== null && state.timeToFullyPaintedMs === null) {
          if (painted > settledChars) {
            settledChars = painted;
            quietFrames = 0;
          } else {
            quietFrames += 1;
            if (quietFrames >= SETTLED_FRAMES) {
              // Length at settlement, not the peak, so a truncating completion render fails.
              state.settledChars = painted;
              state.timeToFullyPaintedMs = now - state.startedAt;
              state.done = true;
            }
          }
        }
      }
      handle = requestAnimationFrame(watch);
    });

    // observe() silently ignores unsupported types (longtask is Chromium only), so record support.
    let measureFrom = Number.POSITIVE_INFINITY;
    let observer: PerformanceObserver | null = null;
    state.longTaskSupported =
      typeof PerformanceObserver !== "undefined" &&
      (PerformanceObserver.supportedEntryTypes ?? []).includes("longtask");
    if (state.longTaskSupported) {
      observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          // Skip buffered startup tasks before the stream window.
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
        config = { ...config, ...options };
        // Straddling entries are discarded, not prorated.
        measureFrom = performance.now();
        state.longTaskMs = 0;
        state.longTasks = 0;
        state.framesOver33ms = 0;
        // Append from a later task so its long tasks start after measureFrom.
        setTimeout(() => {
          void runtime.thread.append({
            role: "user",
            content: [{ type: "text", text: "stream the fixture" }],
          });
        }, 0);
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
