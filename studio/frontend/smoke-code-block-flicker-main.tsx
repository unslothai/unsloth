// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Harness page for tests/studio/playwright_code_block_flicker.py: the real Thread, streaming a
// reply that ends in code fences, sampling every code block's rendered HEIGHT per frame.
//
// The mechanism it catches: streamdown sets `content-visibility: auto` with
// `contain-intrinsic-size: auto 200px` inline on every code-block wrapper. Such an element has no
// LAST REMEMBERED SIZE until it has rendered once, so it lays out at the 200px fallback. The
// re-render at the end of a stream REPLACES the node, and a replaced node is new, so it can lay
// out at 200px for a frame before snapping back. That is the "reload" flicker src/index.css
// describes. So the measurement is not a timing: did a TALL block go SHORT and back, and did the
// thread's scrollHeight dip with it -- per frame, the resolution a flicker is visible at.
//
// Same shape as smoke-heavy-thread.html: vite entry, no backend, auth, GPU or model. Thread is
// real because `.aui-thread-root` is where the override lives; a bare ThreadPrimitive.Root lacks
// that class and would report no flicker on every tree.
//
// useLocalRuntime with a generator adapter, not a seeded import: the flicker is at stream
// FINALIZATION, and `thread.import` never passes through the running -> complete transition.

/* eslint-disable no-restricted-imports -- a measurement entry point, not app code. */
// This store first, per smoke-stream-pacing-main.tsx: the renderer's import graph cycles through
// the chat barrel, and entering the cycle from the renderer leaves a constant in its temporal
// dead zone and the harness renders nothing.
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

// Per smoke-heavy-thread-main.tsx: the fork-count badge fires one GET per assistant message at a
// backend that is not here. Answer before mount, so no round trip lands inside a sampled region.
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
// Each variant is a stylesheet appended AFTER src/index.css, not an edit to the tree. `?css=tree`
// is the pass/fail run; the others exist so "no flicker" can be shown to be a property of the
// tree rather than of the fixture.
//
// The prefix is deliberately OVERSPECIFIC. The tree's rules are scoped
// (`.aui-thread-root[data-status="running"] ...`), so a variant at the obvious specificity loses
// to them exactly where it matters and measures the tree under another name -- that once made
// the pre-override variant report zero flickers, reading as "nothing to fix".
const HERE = ".aui-thread-root.aui-thread-root.aui-thread-root";
const BLOCK = '[data-streamdown="code-block"]';

const CSS_VARIANTS: Record<string, string> = {
  // Whatever src/index.css says, untouched.
  tree: "",
  // Streamdown's defaults, i.e. the tree BEFORE the override. Positive control: this is the state
  // the override prevents, so if it reports no flicker the fixture reproduces nothing and no
  // other row means anything.
  streamdown: `${HERE} ${BLOCK} {
      content-visibility: auto !important;
      contain-intrinsic-size: auto 200px !important;
    }`,
  // The override released for every block at all times, streaming included: the mistake the
  // scoping avoids, and the second positive control, so it MUST flicker too.
  released: `${HERE} ${BLOCK} {
      content-visibility: auto !important;
      contain-intrinsic-size: auto 200px !important;
    }`,
  // The override as main shipped it: visible, and contain-intrinsic-size clobbered to none.
  legacy: `${HERE} ${BLOCK} {
      content-visibility: visible !important;
      contain-intrinsic-size: none !important;
    }`,
  // Streaming status alone, NO settle window: held while the part runs, released the instant it
  // is not. The fix's shape if node replacement at fence close did not land in the same commit
  // as the status flip.
  statusonly: `${HERE} ${BLOCK} {
      content-visibility: auto !important;
      contain-intrinsic-size: auto 200px !important;
    }
    ${HERE} [data-status="running"] ${BLOCK} {
      content-visibility: visible !important;
    }`,
  // Last message only, the CSS-only alternative: no JavaScript, and it survives finalization
  // since the message being finalized is the last. It cannot give an EARLIER message's blocks a
  // first render, so a freshly opened thread holds every off-screen block at the 200px fallback.
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
  // `@layer utilities` is not cosmetic: src/index.css puts the override in that layer, and for
  // IMPORTANT declarations the cascade REVERSES layer order, so a layered `!important` beats an
  // unlayered one however late. Unlayered variants would all lose to the tree and the run would
  // report "no flicker anywhere" having measured one stylesheet four times. Same layer, later in
  // document order, so ordinary precedence applies.
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
 * The streamed reply. It ENDS in a fence on purpose: the flicker is in "trailing code blocks the
 * moment streaming ends", and a reply ending in prose finalizes with the fence long settled.
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
  /** Document-space top of each block, i.e. offset within the scroll container's content. */
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
 * Read in DOCUMENT space (see the analysis module): scrolling must not count as movement.
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
      // `thread.append` returns void, so completion is read from the runtime. Both halves
      // matter: the generator returning is not the end of the render, and the runtime clearing
      // isRunning is what flips the message to complete.
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
       * Scroll bottom to top, a step per frame pair, while the sampler runs.
       *
       * The half the stream does not answer: a never-rendered block is skipped at the
       * `contain-intrinsic-size` fallback, not at its real height. That shows only on the way
       * back up, as each block expands when reached and pushes what is below it down. Tops are
       * DOCUMENT-space, so a top that moves is content above it being relaid out.
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
