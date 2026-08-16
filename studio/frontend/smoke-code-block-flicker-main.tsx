// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Harness page for tests/studio/playwright_code_block_flicker.py: the real Thread, streaming a
// real reply that ends in code fences, with every code block's rendered HEIGHT sampled on every
// frame.
//
// What it exists to catch. Streamdown sets `content-visibility: auto` with
// `contain-intrinsic-size: auto 200px` inline on every code-block wrapper. An element with
// `content-visibility: auto` that has never been rendered has no LAST REMEMBERED SIZE, so it
// lays out at the 200px fallback until the engine decides it is relevant to the user and renders
// it. A code block whose DOM node is REPLACED -- which is what the re-render at the end of a
// stream does -- is a brand new element with no last remembered size, so it can lay out at 200px
// for a frame before snapping back to its real height. That one-frame height change is the
// "reload"-style flicker studio/frontend/src/index.css describes.
//
// So the measurement here is not a timing. It is: did any code block that was TALL become
// SHORT and then tall again, and did the thread's own scroll height dip while that happened.
// Both are recorded per frame, and a frame is the resolution a flicker is visible at.
//
// Same shape as smoke-heavy-thread.html and smoke-autoscroll.html: a vite entry, no backend, no
// auth, no GPU, no model. Thread itself is real on purpose, because `.aui-thread-root` is where
// the override lives and a bare ThreadPrimitive.Root does not carry that class -- a fixture
// mounted on the primitive would measure a page the override never applied to and report no
// flicker on every tree.
//
// useLocalRuntime with a generator adapter rather than a seeded import: the flicker is at stream
// FINALIZATION, so the fixture has to actually finalize a stream. `thread.import` produces
// finished messages and never passes through the running -> complete transition at all.

/* eslint-disable no-restricted-imports -- a measurement entry point, not app code. */
// This store first, deliberately, for the reason smoke-stream-pacing-main.tsx gives: the
// renderer's import graph reaches the chat barrel and back, and entering that cycle from the
// renderer leaves a constant in its temporal dead zone and the harness renders nothing.
import "@/features/chat/stores/sidebar-organization-store";
/* eslint-enable no-restricted-imports */

import { Thread } from "@/components/assistant-ui/thread";
import { TooltipProvider } from "@/components/ui/tooltip";
import {
  AssistantRuntimeProvider,
  type ChatModelAdapter,
  ExportedMessageRepository,
  type ThreadMessageLike,
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

// Same reason as smoke-heavy-thread-main.tsx: the fork-count badge fires one GET per assistant
// message against a backend that is not here. Answering before anything mounts keeps those off
// the wire rather than putting a round trip inside a sampled region.
const realFetch = window.fetch.bind(window);
window.fetch = (input, init) => {
  const url =
    typeof input === "string"
      ? input
      : ((input as Request).url ?? String(input));
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

// ── CSS variants ────────────────────────────────────────────────────
//
// The point of the harness is to compare what the tree ships against the states either side of
// it, so every variant is expressed as a stylesheet appended AFTER src/index.css rather than by
// editing the tree between runs. `?css=tree` measures exactly what the tree ships and is what
// the pass/fail run uses; the others exist so that "no flicker" can be shown to be a property of
// the tree rather than of the fixture.
//
// Every variant selector is written against a deliberately OVERSPECIFIC prefix. The tree's own
// rules are scoped (`.aui-thread-root[data-status="running"] ...` and friends), so a variant
// written at the obvious specificity loses to them exactly where it matters and quietly measures
// the tree under another name: that produced a run in which the pre-override variant reported
// zero flickers, which reads as "there was never anything to fix".
const HERE = ".aui-thread-root.aui-thread-root.aui-thread-root";
const BLOCK = '[data-streamdown="code-block"]';

const CSS_VARIANTS: Record<string, string> = {
  // Whatever src/index.css says, untouched.
  tree: "",
  // Streamdown's own defaults, i.e. the tree BEFORE the override was added. This is the state
  // the override exists to prevent, so a run in this mode that reports no flicker means the
  // fixture is not reproducing anything and nothing measured against it means a thing.
  streamdown: `${HERE} ${BLOCK} {
      content-visibility: auto !important;
      contain-intrinsic-size: auto 200px !important;
    }`,
  // The override released for every block at all times, streaming included. This is the shape
  // of the mistake the scoping is there to avoid.
  released: `${HERE} ${BLOCK} {
      content-visibility: auto !important;
      contain-intrinsic-size: auto 200px !important;
    }`,
  // The override as main shipped it: visible, and contain-intrinsic-size clobbered to none.
  legacy: `${HERE} ${BLOCK} {
      content-visibility: visible !important;
      contain-intrinsic-size: none !important;
    }`,
  // Scoped by the streaming status alone, with NO settle window: held while the message part is
  // running, released the instant it is not. This is the shape the fix would take if the node
  // replacement at fence close did not land in the same commit as the status flip.
  statusonly: `${HERE} ${BLOCK} {
      content-visibility: auto !important;
      contain-intrinsic-size: auto 200px !important;
    }
    ${HERE} [data-status="running"] ${BLOCK} {
      content-visibility: visible !important;
    }`,
  // Scoped to the last message in the thread, the CSS-only alternative: it needs no JavaScript
  // and it does survive finalization, because the message being finalized is the last one. What
  // it cannot do is give an earlier message's blocks a first render, so a freshly opened thread
  // holds every off-screen block at the 200px fallback.
  lastmessage: `${HERE} ${BLOCK} {
      content-visibility: auto !important;
      contain-intrinsic-size: auto 200px !important;
    }
    ${HERE} [data-message-id]:not(:has(~ [data-message-id])) ${BLOCK} {
      content-visibility: visible !important;
    }`,
};

const params = new URLSearchParams(window.location.search);
const CSS_MODE = params.get("css") ?? "tree";
const variantCss = CSS_VARIANTS[CSS_MODE];
if (variantCss === undefined) {
  throw new Error(`unknown css variant ${CSS_MODE}`);
}
if (variantCss) {
  const style = document.createElement("style");
  style.dataset.smokeVariant = CSS_MODE;
  // Inside `@layer utilities`, and that is not cosmetic. src/index.css puts the override in
  // that layer, and for IMPORTANT declarations the cascade reverses layer order: a layered
  // `!important` beats an unlayered one however late it appears. A variant appended as plain
  // CSS therefore loses to the tree's own rule, every variant computes identically, and the
  // run reports "no flicker anywhere" while having measured one stylesheet four times.
  // Same layer, later in document order, so ordinary precedence applies.
  style.textContent = `@layer utilities { ${variantCss} }`;
  document.head.append(style);
}

// ── content ─────────────────────────────────────────────────────────

const PROSE = [
  "The reception of a long thread is decided by what the renderer does on every interaction rather than by what it did once at load.",
  "A reply that arrives quickly can still leave a thread that answers a keystroke slowly, because the two costs are paid in different places.",
  "Anything that walks the whole message list on each frame turns a pleasant session into an unpleasant one somewhere around the twentieth long answer.",
  "Layout that is not contained propagates upward, so a change inside one message can force the entire column to be measured again.",
];

/** Unique per index: the highlighter caches on the exact source string. */
function fence(index: number, lines: number): string {
  const body = [
    `# block ${index}: a scorer long enough for the highlighter to have real work`,
  ];
  body.push("from dataclasses import dataclass", "");
  for (let i = 0; i < lines; i += 1) {
    body.push(
      `def step_${index}_${i}(rows: list[dict], scale: float = ${(index + i) / 7}) -> float:`,
      `    total = sum(row.get("weight_${i}", 0.0) for row in rows)`,
      `    return total * scale + ${i}.0`,
      "",
    );
  }
  return ["```python", ...body, "```"].join("\n");
}

function prose(index: number, paragraphs: number): string {
  const out: string[] = [];
  for (let i = 0; i < paragraphs; i += 1) {
    out.push(
      `${PROSE[(index + i) % PROSE.length]} (paragraph ${i + 1} of reply ${index})`,
    );
  }
  return out.join("\n\n");
}

/** Settled history: the part of the thread that is NOT streaming while the sample runs. */
function history(messages: number): ThreadMessageLike[] {
  const out: ThreadMessageLike[] = [];
  for (let i = 0; i < messages; i += 1) {
    out.push({
      role: "user",
      content: [{ type: "text", text: `Question ${i}?` }],
    });
    out.push({
      role: "assistant",
      content: [
        {
          type: "text",
          text: [prose(i, 3), fence(i, 14), prose(i + 1, 2)].join("\n\n"),
        },
      ],
    });
  }
  return out;
}

/**
 * The streamed reply. It ENDS in a fence on purpose: the index.css comment is about "trailing
 * code blocks the moment streaming ends", and a reply whose last block is prose finalizes with
 * the fence already settled several hundred milliseconds earlier.
 */
function reply(fences: number, linesPerFence: number): string {
  const parts: string[] = [prose(100, 2)];
  for (let i = 0; i < fences; i += 1) {
    parts.push(fence(100 + i, linesPerFence));
    if (i < fences - 1) parts.push(prose(200 + i, 2));
  }
  return parts.join("\n\n");
}

// ── sampling ────────────────────────────────────────────────────────

type Frame = {
  t: number;
  /** Height of every [data-streamdown="code-block"] in DOM order. */
  heights: number[];
  /** Distance of each block's top from the top of the scroll container's content. */
  tops: number[];
  scrollTop: number;
  scrollHeight: number;
  clientHeight: number;
  /** Viewport-relative top of the last SETTLED assistant message, which must not move. */
  anchorTop: number | null;
  running: boolean;
};

type RunOptions = {
  historyMessages?: number;
  fences?: number;
  linesPerFence?: number;
  chunkChars?: number;
  gapMs?: number;
  /** "bottom" leaves autoscroll pinned; "edge" parks the stream tail at the viewport edge. */
  park?: "bottom" | "edge";
};

const state = {
  cssMode: CSS_MODE,
  frames: [] as Frame[],
  sampling: false,
  streamStartedAt: null as number | null,
  streamEndedAt: null as number | null,
  sentChars: 0,
  done: false,
  error: null as string | null,
};

let config: Required<RunOptions> = {
  historyMessages: 8,
  fences: 3,
  linesPerFence: 22,
  chunkChars: 96,
  gapMs: 8,
  park: "bottom",
};

const sleep = (ms: number) =>
  new Promise((resolve) => {
    setTimeout(resolve, ms);
  });

const adapter: ChatModelAdapter = {
  async *run() {
    const text = reply(config.fences, config.linesPerFence);
    state.streamStartedAt = performance.now();
    let cursor = 0;
    while (cursor < text.length) {
      cursor = Math.min(text.length, cursor + config.chunkChars);
      state.sentChars = cursor;
      yield {
        content: [{ type: "text" as const, text: text.slice(0, cursor) }],
      };
      await sleep(config.gapMs);
    }
    state.streamEndedAt = performance.now();
  },
};

function viewport(): HTMLElement | null {
  return document.querySelector<HTMLElement>(".aui-thread-viewport");
}

function codeBlocks(): HTMLElement[] {
  return Array.from(
    document.querySelectorAll<HTMLElement>(
      '.aui-thread-root [data-streamdown="code-block"]',
    ),
  );
}

/**
 * The last assistant message that existed BEFORE the stream started. Nothing about it changes
 * while the stream runs, so any movement of its top edge is the page shifting under the user.
 */
let anchor: HTMLElement | null = null;

function sample(now: number): void {
  const view = viewport();
  const blocks = codeBlocks();
  const viewRect = view?.getBoundingClientRect();
  const heights: number[] = [];
  const tops: number[] = [];
  for (const block of blocks) {
    heights.push(block.offsetHeight);
    const rect = block.getBoundingClientRect();
    tops.push(
      viewRect ? rect.top - viewRect.top + (view?.scrollTop ?? 0) : rect.top,
    );
  }
  state.frames.push({
    t: now,
    heights,
    tops,
    scrollTop: view?.scrollTop ?? -1,
    scrollHeight: view?.scrollHeight ?? -1,
    clientHeight: view?.clientHeight ?? -1,
    anchorTop: anchor ? anchor.getBoundingClientRect().top : null,
    running: state.streamStartedAt !== null && state.streamEndedAt === null,
  });
}

function FlickerApi(): null {
  const aui = useAui();

  useEffect(() => {
    let handle = 0;
    const loop = (now: number) => {
      if (state.sampling) sample(now);
      // `thread.append` returns void here, so completion is read from the runtime rather than
      // from a promise. Both halves matter: the generator returning is not the end of the
      // render, and the runtime clearing isRunning is what flips the message to complete.
      if (
        !state.done &&
        state.streamEndedAt !== null &&
        !aui.thread().getState().isRunning
      ) {
        state.done = true;
      }
      handle = requestAnimationFrame(loop);
    };
    handle = requestAnimationFrame(loop);

    const api = {
      cssMode: CSS_MODE,
      /** Settled history only. Returns how many code blocks it produced. */
      seed(messages: number): number {
        aui
          .thread()
          .import(ExportedMessageRepository.fromArray(history(messages)));
        return messages;
      },
      /** Park the viewport where the flicker is meant to be visible. */
      park(mode: "bottom" | "edge"): {
        scrollTop: number;
        scrollHeight: number;
      } {
        const view = viewport();
        if (!view) return { scrollTop: -1, scrollHeight: -1 };
        view.style.scrollBehavior = "auto";
        view.scrollTop =
          mode === "bottom"
            ? view.scrollHeight
            : Math.max(0, view.scrollHeight - view.clientHeight - 240);
        return { scrollTop: view.scrollTop, scrollHeight: view.scrollHeight };
      },
      /** Fix the anchor and start recording. Called once the history has settled. */
      startSampling(): number {
        const assistants = document.querySelectorAll<HTMLElement>(
          '[data-role="assistant"]',
        );
        anchor = assistants[assistants.length - 1] ?? null;
        state.frames = [];
        state.sampling = true;
        return codeBlocks().length;
      },
      stopSampling(): number {
        state.sampling = false;
        return state.frames.length;
      },
      /**
       * Scroll from the bottom of the thread to the top, a step per frame pair, while the
       * sampler runs.
       *
       * This is the half of the question the stream does not answer. A block that has never
       * been rendered is skipped at the `contain-intrinsic-size` fallback rather than at its
       * real height, and the difference does not show while the user sits at the bottom: it
       * shows when they scroll back up, because every block that expands as it is reached
       * pushes everything below it down. The frame log records each block's DOCUMENT-space top,
       * so a block whose top moves is content above it having been relaid out.
       */
      async sweepUp(
        steps: number,
        stepPx: number,
      ): Promise<Record<string, number>> {
        const view = viewport();
        if (!view) return { steps: 0 };
        view.style.scrollBehavior = "auto";
        view.scrollTop = view.scrollHeight;
        const start = view.scrollTop;
        const twoFrames = () =>
          new Promise<void>((resolve) => {
            requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
          });
        await twoFrames();
        for (let i = 0; i < steps && view.scrollTop > 0; i += 1) {
          view.scrollTop = Math.max(0, view.scrollTop - stepPx);
          await twoFrames();
        }
        await twoFrames();
        return {
          steps,
          scrollTopStart: start,
          scrollTopEnd: view.scrollTop,
          scrollHeight: view.scrollHeight,
        };
      },
      run(options: RunOptions = {}): void {
        config = { ...config, ...options };
        state.streamStartedAt = null;
        state.streamEndedAt = null;
        state.sentChars = 0;
        state.done = false;
        state.error = null;
        // From a later task, so the append is not attributed to the caller's task.
        setTimeout(() => {
          try {
            aui
              .thread()
              .append({
                role: "user",
                content: [{ type: "text", text: "stream the fixture" }],
              });
          } catch (err: unknown) {
            state.error = String(err);
            state.done = true;
          }
        }, 0);
      },
      counts(): Record<string, number> {
        return {
          messages: document.querySelectorAll("[data-role]").length,
          codeBlocks: codeBlocks().length,
          preElements: document.querySelectorAll("pre").length,
          highlightedTokens: document.querySelectorAll("pre code span").length,
          domNodes: document.getElementsByTagName("*").length,
        };
      },
      /** What the tree actually computed for a block, so a run says which CSS it measured. */
      computedFor(index: number): Record<string, string> {
        const block = codeBlocks()[index];
        if (!block) return {};
        const style = getComputedStyle(block);
        return {
          contentVisibility: style.contentVisibility,
          containIntrinsicSize: style.containIntrinsicSize,
          layoutAttr:
            document
              .querySelector(".aui-thread-root")
              ?.getAttribute("data-code-block-layout") ?? "(absent)",
        };
      },
      results() {
        return {
          cssMode: state.cssMode,
          frames: state.frames,
          streamStartedAt: state.streamStartedAt,
          streamEndedAt: state.streamEndedAt,
          sentChars: state.sentChars,
          done: state.done,
          error: state.error,
        };
      },
    };
    (window as unknown as { __flicker: typeof api }).__flicker = api;

    return () => {
      cancelAnimationFrame(handle);
    };
  }, [aui]);

  return null;
}

function Harness(): ReactElement {
  const runtime = useLocalRuntime(adapter);
  return (
    <TooltipProvider>
      <AssistantRuntimeProvider runtime={runtime}>
        <FlickerApi />
        <div
          data-smoke="code-block-flicker"
          style={{ display: "flex", flexDirection: "column", height: "100vh" }}
        >
          <Thread hideWelcome={true} />
        </div>
      </AssistantRuntimeProvider>
    </TooltipProvider>
  );
}

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
