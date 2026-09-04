// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Harness for tests/studio/playwright_thread_weight.py: the real Thread at N messages, so the
// measured cost of a keystroke, a scroll, a menu and a delete is the app's own and grows with the
// thread the way a user's does (#8977).
// Same shape as smoke-autoscroll.html and smoke-research.html: a vite entry, no backend, no auth.
//
// Two things are real on purpose and cannot be mocked away without deleting the measurement:
//   - Thread itself, from src/components/assistant-ui/thread.tsx, with its per-message action
//     bars, tooltips and markdown blocks.
//   - The message bodies, which carry prose plus one code fence plus one KaTeX block each, so
//     Streamdown, Shiki and KaTeX all pay their per-message price.
// The runtime is synthetic: a local runtime whose model adapter never runs, seeded through
// `thread.import`.
//
// useLocalRuntime rather than useExternalStoreRuntime: the delete path under measurement is
// `thread.export()` -> MessageRepository -> `thread.import()`, which only the local runtime backs.

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
import { type ReactElement, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./src/index.css";

// The local runtime's thread list item reports a synthetic `__LOCALID_...` remoteId, which is
// truthy, so ForkCountBadge really does fire one GET per assistant message. Measured: seeding 20
// messages issues 10 requests. Answering them here, before anything mounts, keeps that off the
// wire entirely. Answering them from the Playwright side instead would put a CDP round trip to
// another process inside a region this harness is timing, once per assistant message.
const realFetch = window.fetch.bind(window);
window.fetch = (input, init) => {
  const url = typeof input === "string" ? input : (input as Request).url ?? String(input);
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

const PROSE =
  "The reception of a long thread is decided by what the renderer does on every " +
  "interaction, not by what it did once at load. This paragraph exists to give each " +
  "message real prose to lay out, to wrap over several lines at a chat column's width, " +
  "and to make the message tall enough that a seeded thread overflows the viewport.";

const CLOSING =
  "A second paragraph, so each message has more than one block-level child and the " +
  "layout pass inside it is not trivially small.";

function codeFence(index: number): string {
  return [
    "```python",
    `def step_${index}(rows):`,
    '    """One fence per message, so every message pays a highlighter pass."""',
    "    total = 0",
    "    for row in rows:",
    "        total += row.weight * row.count",
    "    return total",
    "```",
  ].join("\n");
}

function katexBlock(index: number): string {
  return `$$\n\\sum_{k=1}^{${index + 1}} \\frac{1}{k^2} \\le \\frac{\\pi^2}{6}\n$$`;
}

function assistantMarkdown(index: number): string {
  return [
    `Reply ${index}. ${PROSE}`,
    codeFence(index),
    katexBlock(index),
    CLOSING,
  ].join("\n\n");
}

/**
 * A reply that is prose and nothing else: no fence, no math, no link, no image.
 *
 * The default body above carries a code fence, and Streamdown gives every fence its own Copy
 * button, so every assistant message in the default fixture happens to hold a focusable control
 * whether or not its action bar is mounted. Most real replies are not like that. This variant
 * exists so the harness can seed a reply whose focusable count is ZERO and a keyboard walk has
 * nothing to land on (#8992, review item on unreachable plain-text replies).
 */
function plainAssistantMarkdown(index: number): string {
  return [`Reply ${index}. ${PROSE}`, CLOSING].join("\n\n");
}

function userMarkdown(index: number): string {
  return `Prompt ${index}. ${PROSE}`;
}

/**
 * Which assistant replies are plain prose, by their ordinal among assistant messages.
 * `"all"` makes every reply plain; omitting it keeps the default fenced body everywhere, so
 * the existing weight measurements are untouched.
 */
type SeedOptions = { plainAssistants?: readonly number[] | "all" };

function isPlain(assistantOrdinal: number, options: SeedOptions | undefined): boolean {
  const plain = options?.plainAssistants;
  if (!plain) return false;
  if (plain === "all") return true;
  return plain.includes(assistantOrdinal);
}

/** Alternating prompts and replies, oldest first, as a loaded thread arrives. */
function buildMessages(
  count: number,
  options?: SeedOptions,
): ThreadMessageLike[] {
  return Array.from({ length: count }, (_, index) =>
    index % 2 === 0
      ? {
          role: "user" as const,
          content: [{ type: "text" as const, text: userMarkdown(index) }],
        }
      : {
          role: "assistant" as const,
          content: [
            {
              type: "text" as const,
              text: isPlain((index - 1) / 2, options)
                ? plainAssistantMarkdown(index)
                : assistantMarkdown(index),
            },
          ],
        },
  );
}

// A run would need a backend. Seeding goes through `thread.import`, which does not use this.
const NEVER_RUNS: ChatModelAdapter = {
  run: () => {
    throw new Error("smoke-thread-weight does not run the model");
  },
};

function ThreadWeightApi({
  setThreadMounted,
}: {
  setThreadMounted: (mounted: boolean) => void;
}): null {
  const aui = useAui();

  useEffect(() => {
    const api = {
      /** Replace the thread with `count` messages, oldest first. */
      seed(count: number, options?: SeedOptions): void {
        aui
          .thread()
          .import(
            ExportedMessageRepository.fromArray(buildMessages(count, options)),
          );
      },
      /**
       * Unmount or remount the Thread while the RUNTIME stays alive, which is what a sidebar
       * thread switch does to a message's React subtree. The per-message `isHovering` flag
       * lives in the runtime's message client, not in React, so it outlives this.
       */
      setThreadMounted(mounted: boolean): void {
        setThreadMounted(mounted);
      },
      /**
       * The runtime's own `isHovering` per message, oldest first. Reading the flag rather than
       * counting mounted bars is what makes a leak test able to see a flag that is set while
       * nothing is rendered.
       */
      hoverFlags(): { id: string; isHovering: boolean }[] {
        return aui
          .thread()
          .getState()
          .messages.map((message) => ({
            id: message.id,
            isHovering: Boolean(message.isHovering),
          }));
      },
      /**
       * One selector pass. Polling for a deletion has to read this, not counts(): counts() is
       * eight document-wide queries including a walk of every element, so at 500 messages a
       * poll loop built on it would spend more time measuring than the delete itself takes.
       */
      messageCount(): number {
        return document.querySelectorAll("[data-role]").length;
      },
      /**
       * Everything a caller might use to prove the seed landed. A harness that seeds 500 and
       * silently renders 0 measures nothing, so the Python side prints every one of these.
       */
      counts(): {
        messages: number;
        assistantMessages: number;
        userMessages: number;
        domNodes: number;
        codeBlocks: number;
        katexNodes: number;
        actionBars: number;
        tooltipTriggers: number;
      } {
        return {
          messages: document.querySelectorAll("[data-role]").length,
          assistantMessages: document.querySelectorAll('[data-role="assistant"]')
            .length,
          userMessages: document.querySelectorAll('[data-role="user"]').length,
          domNodes: document.getElementsByTagName("*").length,
          codeBlocks: document.querySelectorAll("pre").length,
          katexNodes: document.querySelectorAll(".katex").length,
          actionBars: document.querySelectorAll(".aui-assistant-action-bar-root")
            .length,
          tooltipTriggers: document.querySelectorAll(
            '[data-slot="tooltip-trigger"]',
          ).length,
        };
      },
      viewportMetrics(): {
        scrollHeight: number;
        scrollTop: number;
        clientHeight: number;
      } {
        const element = api.viewport();
        if (!element) {
          return { scrollHeight: -1, scrollTop: -1, clientHeight: -1 };
        }
        return {
          scrollHeight: element.scrollHeight,
          scrollTop: element.scrollTop,
          clientHeight: element.clientHeight,
        };
      },
      viewport(): HTMLElement | null {
        return document.querySelector<HTMLElement>(".aui-thread-viewport");
      },
      composer(): HTMLTextAreaElement | null {
        return document.querySelector<HTMLTextAreaElement>(
          ".aui-composer-input",
        );
      },
      /**
       * What the RUNTIME thinks the composer holds. Reading the textarea back instead would
       * only echo the value the caller just wrote, so a keystroke that never reached React
       * would still look like it landed.
       */
      composerText(): string {
        return aui.composer().getState().text;
      },
      /** One selector pass, for the seed gate; counts() is far too heavy to poll. */
      katexCount(): number {
        return document.querySelectorAll(".katex").length;
      },
      /** Highlighted tokens. Shiki runs after the <pre> exists, so counting <pre> gates nothing. */
      highlightedTokenCount(): number {
        return document.querySelectorAll("pre code span").length;
      },
      /** Items in the open action menu. An empty popover satisfies "the menu opened". */
      openMenuItemCount(): number {
        return document.querySelectorAll(".aui-action-bar-more-item").length;
      },
      lastAssistantMessage(): HTMLElement | null {
        const messages = document.querySelectorAll<HTMLElement>(
          '[data-role="assistant"]',
        );
        return messages[messages.length - 1] ?? null;
      },
      /**
       * The last assistant message's action-bar button with accessible name `label`.
       * TooltipIconButton puts that name in an `sr-only` span rather than an aria-label, so
       * this matches on text and stays correct if the styling classes are renamed.
       */
      actionButton(label: string): HTMLButtonElement | null {
        const last = api.lastAssistantMessage();
        if (!last) return null;
        const buttons = Array.from(last.querySelectorAll("button"));
        return (
          buttons.find(
            (button) => (button.textContent ?? "").trim() === label,
          ) ?? null
        );
      },
    };
    (window as unknown as { __threadWeight: typeof api }).__threadWeight = api;
  }, [aui, setThreadMounted]);

  return null;
}

function Harness(): ReactElement {
  const runtime = useLocalRuntime(NEVER_RUNS);
  const [threadMounted, setThreadMounted] = useState(true);
  return (
    <TooltipProvider>
      <AssistantRuntimeProvider runtime={runtime}>
        <ThreadWeightApi setThreadMounted={setThreadMounted} />
        {/* Thread is flex-1 basis-0 min-h-0, so it needs a bounded flex parent to scroll. */}
        <div
          data-smoke="thread"
          style={{ display: "flex", flexDirection: "column", height: "100vh" }}
        >
          {threadMounted ? <Thread hideWelcome={true} /> : null}
        </div>
      </AssistantRuntimeProvider>
    </TooltipProvider>
  );
}

// Thread reaches useNavigate (the fork action, the composer tools menu). Without a router in
// context tanstack's useRouter still works, but console.warns on every render of every action
// bar: measured at 12 warnings for 10 assistant messages, which scales with N and is serialised
// over CDP. A memory router with one route removes that without pulling in the app shell.
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
