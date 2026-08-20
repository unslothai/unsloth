// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Harness page for tests/studio/playwright_code_block_remount.py: the real Thread, SETTLED, with
// each of the three known ways of re-creating a code block on a quiet thread driven for real and
// every code block's rendered HEIGHT sampled on every frame.
//
// Sibling of smoke-code-block-flicker.html, which covers the other half of the same question. That
// one streams a reply and measures finalization; this one never streams at all. Its subject is the
// state AFTER the thread has gone quiet and `data-code-block-layout` has flipped to `settled`, at
// which point `content-visibility: auto` is live on every block. An element that has never been
// rendered has no LAST REMEMBERED SIZE and is laid out at streamdown's 200px
// `contain-intrinsic-size` fallback, so anything that mounts a BRAND NEW block element from here
// paints one frame at 200px and moves everything below it.
//
// The three paths, none of which changes `thread.isRunning`:
//
//   edit       Leaving the edit textarea on a completed reply. thread.tsx renders an editing
//              message as a <textarea> and any other one as its rendered parts, so the two
//              subtrees are different element types and React unmounts one and mounts the other.
//   branch     Switching response branches. markdown-text.tsx keys <Streamdown> on the message
//              id, and sibling branches are distinct messages, so the whole markdown subtree is
//              re-created.
//   reasoning  Expanding a collapsed reasoning section. ui/collapsible.tsx wraps Radix
//              CollapsibleContent with no `forceMount`, so closed content is not in the tree at
//              all, and reasoning bodies render through MarkdownText like any other part.
//
// The fixture drives them through the real components -- the store field the editor reads, the
// runtime's own switchToBranch, an actual click on the disclosure -- rather than by mutating the
// DOM, because the question is whether React re-creates those elements and a fixture that
// re-created them itself would answer its own question.

/* eslint-disable no-restricted-imports -- a measurement entry point, not app code. */
// This store first, deliberately, for the reason smoke-stream-pacing-main.tsx gives: the
// renderer's import graph reaches the chat barrel and back, and entering that cycle from the
// renderer leaves a constant in its temporal dead zone and the harness renders nothing.
import "@/features/chat/stores/sidebar-organization-store";
// The store thread.tsx reads `editingMessageId` from. Reached directly rather than through the
// feature index because the point of the `edit` scenario is to move that exact field, and the
// barrel does not re-export the store hook.
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
/* eslint-enable no-restricted-imports */

import { Thread } from "@/components/assistant-ui/thread";
import { TooltipProvider } from "@/components/ui/tooltip";
import {
  AssistantRuntimeProvider,
  type ChatModelAdapter,
  ExportedMessageRepository,
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

// Same reason as smoke-code-block-flicker-main.tsx: the fork-count badge fires one GET per
// assistant message against a backend that is not here.
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
// Same contract as the flicker fixture, and the same overspecific prefix for the same reason: the
// tree's own rules are scoped, so a variant written at the obvious specificity loses to them
// exactly where it matters and quietly measures the tree under another name.
const HERE = ".aui-thread-root.aui-thread-root.aui-thread-root";
const BLOCK = '[data-streamdown="code-block"]';

const CSS_VARIANTS: Record<string, string> = {
  // Whatever src/index.css says, untouched. This is the pass/fail mode.
  tree: "",
  // Streamdown's own defaults, forced past the hold. This is what the thread looks like with no
  // hold at all, and a run in this mode that reports no collapse means the fixture is not
  // reproducing anything and nothing measured against it means a thing.
  released: `${HERE} ${BLOCK} {
      content-visibility: auto !important;
      contain-intrinsic-size: auto 200px !important;
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
  // Inside `@layer utilities`, for the reason spelled out in smoke-code-block-flicker-main.tsx:
  // for IMPORTANT declarations the cascade reverses layer order, so an unlayered override loses
  // to a layered one however late it appears.
  style.textContent = `@layer utilities { ${variantCss} }`;
  document.head.append(style);
}

// ── content ─────────────────────────────────────────────────────────

const PROSE = [
  "The reception of a long thread is decided by what the renderer does on every interaction rather than by what it did once at load.",
  "A reply that arrives quickly can still leave a thread that answers a keystroke slowly, because the two costs are paid in different places.",
  "Anything that walks the whole message list on each frame turns a pleasant session into an unpleasant one somewhere around the twentieth long answer.",
];

function fence(index: number, lines: number): string {
  const body = [`# block ${index}`, "from dataclasses import dataclass", ""];
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
      `${PROSE[(index + i) % PROSE.length]} (para ${i + 1} of ${index})`,
    );
  }
  return out.join("\n\n");
}

const TARGET_LINES = 22;

/** The reply the measurement is about, in every scenario: prose, a tall fence, prose. */
function targetBody(index: number): string {
  return [
    prose(index, 2),
    fence(index, TARGET_LINES),
    prose(index + 1, 1),
  ].join("\n\n");
}

type Seeded = { id: string; repository: ExportedMessageRepository };

/**
 * A short settled thread ending in the reply the scenario acts on.
 *
 * Short on purpose. The flicker fixture needs length because it measures scrolling cost; here the
 * subject is one reply's blocks, and a long thread only makes it harder to say which block moved.
 */
function seedEdit(): Seeded {
  const items = [
    {
      message: {
        id: "u0",
        role: "user" as const,
        content: [{ type: "text" as const, text: "Question?" }],
      },
      parentId: null,
    },
    {
      message: {
        id: "a0",
        role: "assistant" as const,
        content: [{ type: "text" as const, text: targetBody(1) }],
      },
      parentId: "u0",
    },
  ];
  return {
    id: "a0",
    repository: ExportedMessageRepository.fromBranchableArray(items, {
      headId: "a0",
    }),
  };
}

/**
 * Two sibling replies under one user turn.
 *
 * The fence is IDENTICAL in both, and only the prose differs. If the two branches had different
 * blocks, a height change on the switch would be ambiguous -- the new block really is a different
 * height -- and the measurement would prove nothing. With the fence held equal, the settled height
 * is the same on both sides and any frame at the 200px fallback is unambiguous.
 */
function seedBranch(): Seeded {
  const withFence = (tag: string) =>
    [prose(2, 1), `(${tag})`, fence(1, TARGET_LINES), prose(3, 1)].join("\n\n");
  const items = [
    {
      message: {
        id: "u0",
        role: "user" as const,
        content: [{ type: "text" as const, text: "Question?" }],
      },
      parentId: null,
    },
    {
      message: {
        id: "b1",
        role: "assistant" as const,
        content: [{ type: "text" as const, text: withFence("first response") }],
      },
      parentId: "u0",
    },
    {
      message: {
        id: "b2",
        role: "assistant" as const,
        content: [
          { type: "text" as const, text: withFence("second response") },
        ],
      },
      parentId: "u0",
    },
  ];
  return {
    id: "b1",
    repository: ExportedMessageRepository.fromBranchableArray(items, {
      headId: "b1",
    }),
  };
}

/**
 * A reply whose reasoning section carries the fence.
 *
 * Reasoning renders through MarkdownText (reasoning.tsx: `ReasoningImpl = () => <MarkdownText />`)
 * and the disclosure defaults to CLOSED, and ui/collapsible.tsx passes no `forceMount`, so on a
 * settled thread that fence exists nowhere in the document until the disclosure is opened.
 */
function seedReasoning(): Seeded {
  const items = [
    {
      message: {
        id: "u0",
        role: "user" as const,
        content: [{ type: "text" as const, text: "Question?" }],
      },
      parentId: null,
    },
    {
      message: {
        id: "a0",
        role: "assistant" as const,
        content: [
          {
            type: "reasoning" as const,
            text: [prose(4, 1), fence(1, TARGET_LINES)].join("\n\n"),
          },
          { type: "text" as const, text: prose(5, 2) },
        ],
      },
      parentId: "u0",
    },
  ];
  return {
    id: "a0",
    repository: ExportedMessageRepository.fromBranchableArray(items, {
      headId: "a0",
    }),
  };
}

const SEEDS: Record<string, () => Seeded> = {
  edit: seedEdit,
  branch: seedBranch,
  reasoning: seedReasoning,
};

// ── sampling ────────────────────────────────────────────────────────

type Frame = {
  /** Frames since sampling started. */
  n: number;
  /** Height of every [data-streamdown="code-block"] in DOM order. */
  heights: number[];
  /** Viewport-relative top of the element that FOLLOWS each block, i.e. what moves. */
  nextTops: number[];
  scrollHeight: number;
  /** What the thread root said this frame, so a run reports which state it measured. */
  layoutAttr: string;
  /** Set on the frame a driver fired, so "before" and "after" are readable in the log. */
  mark: string | null;
};

const state = {
  cssMode: CSS_MODE,
  frames: [] as Frame[],
  sampling: false,
  mark: null as string | null,
  error: null as string | null,
};

function threadRoot(): HTMLElement | null {
  return document.querySelector<HTMLElement>(".aui-thread-root");
}

function viewport(): HTMLElement | null {
  return document.querySelector<HTMLElement>(".aui-thread-viewport");
}

function codeBlocks(): HTMLElement[] {
  return Array.from(
    document.querySelectorAll<HTMLElement>(`.aui-thread-root ${BLOCK}`),
  );
}

function sample(n: number): void {
  const blocks = codeBlocks();
  const heights: number[] = [];
  const nextTops: number[] = [];
  for (const block of blocks) {
    heights.push(block.offsetHeight);
    const next =
      block.nextElementSibling ?? block.parentElement?.nextElementSibling;
    nextTops.push(
      next ? Math.round(next.getBoundingClientRect().top * 100) / 100 : -1,
    );
  }
  state.frames.push({
    n,
    heights,
    nextTops,
    scrollHeight: viewport()?.scrollHeight ?? -1,
    layoutAttr:
      threadRoot()?.getAttribute("data-code-block-layout") ?? "(absent)",
    mark: state.mark,
  });
  state.mark = null;
}

function RemountApi(): null {
  const aui = useAui();

  useEffect(() => {
    let handle = 0;
    let n = 0;
    const loop = () => {
      if (state.sampling) {
        sample(n);
        n += 1;
      }
      handle = requestAnimationFrame(loop);
    };
    handle = requestAnimationFrame(loop);

    const api = {
      cssMode: CSS_MODE,
      /** Import one of the three seeds. Returns the id of the reply the scenario acts on. */
      seed(name: keyof typeof SEEDS): string {
        const build = SEEDS[name];
        if (!build) throw new Error(`unknown seed ${String(name)}`);
        const seeded = build();
        aui.thread().import(seeded.repository);
        return seeded.id;
      },
      /** Park the viewport so the target reply is on screen: an off-screen block never renders. */
      park(): { scrollTop: number; scrollHeight: number } {
        const view = viewport();
        if (!view) return { scrollTop: -1, scrollHeight: -1 };
        view.style.scrollBehavior = "auto";
        view.scrollTop = 0;
        return { scrollTop: view.scrollTop, scrollHeight: view.scrollHeight };
      },
      startSampling(): number {
        state.frames = [];
        state.mark = null;
        state.sampling = true;
        n = 0;
        return codeBlocks().length;
      },
      stopSampling(): number {
        state.sampling = false;
        return state.frames.length;
      },

      // ── the three drivers ───────────────────────────────────────────
      //
      // Each goes through the real component path, and each is called from a fresh task so the
      // work is not attributed to the caller's.

      /** Put the reply into the edit textarea. Nothing is measured here; the return is. */
      enterEdit(messageId: string): void {
        useChatRuntimeStore.getState().setEditingMessageId(messageId);
      },
      /** Leave the edit textarea. This is what re-creates the blocks. */
      leaveEdit(): void {
        state.mark = "leaveEdit";
        useChatRuntimeStore.getState().setEditingMessageId(null);
      },
      /** Switch the displayed reply to its sibling branch. */
      switchBranch(): Record<string, unknown> {
        const thread = aui.thread();
        const count = thread.getState().messages.length;
        const message = thread.message({ index: count - 1 });
        const before = message.getState();
        state.mark = "switchBranch";
        message.switchToBranch({ position: "next" });
        return {
          from: before.id,
          branchNumber: before.branchNumber,
          branchCount: before.branchCount,
        };
      },
      /** Open the collapsed reasoning disclosure by clicking it, as a user would. */
      expandReasoning(): boolean {
        const trigger = document.querySelector<HTMLElement>(
          '[data-slot="reasoning-trigger"], [data-slot="collapsible-trigger"]',
        );
        if (!trigger) return false;
        state.mark = "expandReasoning";
        trigger.click();
        return true;
      },

      counts(): Record<string, number | string> {
        return {
          messages: document.querySelectorAll("[data-role]").length,
          codeBlocks: codeBlocks().length,
          highlightedTokens: document.querySelectorAll("pre code span").length,
          layoutAttr:
            threadRoot()?.getAttribute("data-code-block-layout") ?? "(absent)",
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
            threadRoot()?.getAttribute("data-code-block-layout") ?? "(absent)",
        };
      },
      results() {
        return {
          cssMode: state.cssMode,
          frames: state.frames,
          error: state.error,
        };
      },
    };
    (window as unknown as { __remount: typeof api }).__remount = api;

    return () => {
      cancelAnimationFrame(handle);
    };
  }, [aui]);

  return null;
}

// The thread never runs here; the adapter exists only because useLocalRuntime requires one.
const adapter: ChatModelAdapter = {
  async *run() {
    yield { content: [{ type: "text" as const, text: "" }] };
  },
};

function Harness(): ReactElement {
  const runtime = useLocalRuntime(adapter);
  return (
    <TooltipProvider>
      <AssistantRuntimeProvider runtime={runtime}>
        <RemountApi />
        <div
          data-smoke="code-block-remount"
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
