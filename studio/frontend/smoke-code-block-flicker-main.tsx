// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Harness for tests/studio/playwright_code_block_flicker.py: samples code block heights per frame
// to catch the 200px contain-intrinsic-size fallback flicker at stream finalization.

/* eslint-disable no-restricted-imports -- a measurement entry point, not app code. */
// Import this store first: entering the import cycle from the renderer leaves a constant in TDZ.
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

// Stub fork-count GETs before mount so no round trip lands in a sampled region.
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

// CSS variants appended after src/index.css. Prefix is overspecific to beat the tree's scoped rules.
const HERE = ".aui-thread-root.aui-thread-root.aui-thread-root";
const BLOCK = '[data-streamdown="code-block"]';

const CSS_VARIANTS: Record<string, string> = {
  tree: "",
  // Positive control: streamdown defaults must flicker or the fixture reproduces nothing.
  streamdown: `${HERE} ${BLOCK} {
      content-visibility: auto !important;
      contain-intrinsic-size: auto 200px !important;
    }`,
  // Second positive control: override released always, must flicker too.
  released: `${HERE} ${BLOCK} {
      content-visibility: auto !important;
      contain-intrinsic-size: auto 200px !important;
    }`,
  legacy: `${HERE} ${BLOCK} {
      content-visibility: visible !important;
      contain-intrinsic-size: none !important;
    }`,
  statusonly: `${HERE} ${BLOCK} {
      content-visibility: auto !important;
      contain-intrinsic-size: auto 200px !important;
    }
    ${HERE} [data-status="running"] ${BLOCK} {
      content-visibility: visible !important;
    }`,
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
  // Same layer as the tree's override: layered !important beats unlayered regardless of order.
  style.textContent = `@layer utilities { ${variantCss} }`;
  document.head.append(style);
}

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

/** Ends in a fence on purpose: the flicker is in trailing code blocks when streaming ends. */
function reply(fences: number, linesPerFence: number): string {
  const parts: string[] = [prose(100, 2)];
  for (let i = 0; i < fences; i += 1) {
    parts.push(fence(100 + i, linesPerFence));
    if (i < fences - 1) parts.push(prose(200 + i, 2));
  }
  return parts.join("\n\n");
}

type Frame = {
  t: number;
  heights: number[];
  tops: number[];
  scrollTop: number;
  scrollHeight: number;
  clientHeight: number;
  anchorTop: number | null;
  running: boolean;
};

type RunOptions = {
  historyMessages?: number;
  fences?: number;
  linesPerFence?: number;
  chunkChars?: number;
  gapMs?: number;
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
      // `thread.append` returns void; completion is when the runtime clears isRunning.
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
      seed(messages: number): number {
        aui
          .thread()
          .import(ExportedMessageRepository.fromArray(history(messages)));
        return messages;
      },
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
      /** Scroll up while sampling: never-rendered blocks expand only when reached. */
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
