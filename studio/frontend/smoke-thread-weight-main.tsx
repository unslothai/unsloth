// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Harness for tests/studio/playwright_thread_weight.py: the real Thread and real markdown bodies,
// seeded via thread.import on a local runtime (only it backs export/import used by delete).

import { Thread } from "@/components/assistant-ui/thread";
import { TooltipProvider } from "@/components/ui/tooltip";
import {
  AssistantRuntimeProvider,
  type ChatModelAdapter,
  ExportedMessageRepository,
  SimpleImageAttachmentAdapter,
  type ThreadMessageLike,
  useAui,
  unstable_useRemoteThreadListRuntime,
  type unstable_RemoteThreadListAdapter,
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

// Answer ForkCountBadge GETs in-page: a Playwright-side answer would add a CDP round trip per
// assistant message inside the timed region.
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

/** Plain prose reply with zero focusable controls (fenced bodies always carry a Copy button). */
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
type SeedOptions = {
  plainAssistants?: readonly number[] | "all";
  /** `rounds` x reasoning, tool call, answer per reply; truncation every `compactionEvery`th. */
  agent?: { rounds: number; compactionEvery?: number };
};

const AGENT_TOOLS = ["web_search", "terminal", "python"] as const;

function agentContent(index: number, rounds: number) {
  const parts: Exclude<ThreadMessageLike["content"], string>[number][] = [];
  for (let round = 0; round < rounds; round++) {
    const tag = `${index}_${round}`;
    const toolName = AGENT_TOOLS[(index + round) % AGENT_TOOLS.length];
    const args: Record<string, string> =
      toolName === "web_search"
        ? { query: `step ${tag} reference for the scheduler buffer` }
        : { code: `print("step ${tag}")` };
    parts.push({
      type: "reasoning" as const,
      text: `Round ${tag}. ${PROSE} ${CLOSING}`,
    });
    parts.push({
      type: "tool-call" as const,
      toolCallId: `call_${tag}`,
      toolName,
      args,
      argsText: JSON.stringify(args),
      result: `Result ${tag}. ${PROSE}\n${PROSE}\n${CLOSING}`,
    });
    parts.push({
      type: "text" as const,
      text: `Answer ${tag}. ${PROSE}\n\n- first item ${tag}\n- second item ${tag}`,
    });
  }
  return parts;
}

function isPlain(assistantOrdinal: number, options: SeedOptions | undefined): boolean {
  const plain = options?.plainAssistants;
  if (!plain) return false;
  if (plain === "all") return true;
  return plain.includes(assistantOrdinal);
}

function buildMessages(
  count: number,
  options?: SeedOptions,
): ThreadMessageLike[] {
  const agent = options?.agent;
  return Array.from({ length: count }, (_, index): ThreadMessageLike => {
    if (index % 2 === 0) {
      return {
        role: "user" as const,
        content: [{ type: "text" as const, text: userMarkdown(index) }],
      };
    }
    if (agent) {
      const ordinal = (index - 1) / 2;
      const every = agent.compactionEvery ?? 0;
      const compacted = every > 0 && ordinal > 0 && ordinal % every === 0;
      return {
        role: "assistant" as const,
        content: agentContent(index, agent.rounds),
        ...(compacted
          ? {
              metadata: {
                custom: {
                  contextTruncation: {
                    dropped_messages: Math.max(1, index - 4),
                    boundary_messages: Math.max(1, index - 4),
                    fits: true,
                  },
                },
              },
            }
          : {}),
      };
    }
    return {
      role: "assistant" as const,
      content: [
        {
          type: "text" as const,
          text: isPlain((index - 1) / 2, options)
            ? plainAssistantMarkdown(index)
            : assistantMarkdown(index),
        },
      ],
    };
  });
}

// A run would need a backend. Seeding goes through `thread.import`, which does not use this.
const STREAM_CHUNKS = Number(
  new URLSearchParams(window.location.search).get("stream") ?? "0",
);
const NEVER_RUNS: ChatModelAdapter = STREAM_CHUNKS > 0
  ? {
      async *run({ abortSignal }) {
        let text = "";
        for (let chunk = 0; chunk < STREAM_CHUNKS; chunk++) {
          await new Promise((resolve) => setTimeout(resolve, 20));
          if (abortSignal.aborted) return;
          text += `word${chunk} `;
          yield { content: [{ type: "text" as const, text }] };
        }
      },
    }
  : {
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
      send(text: string): void {
        aui.thread().append({
          role: "user",
          content: [{ type: "text", text }],
        });
      },
      isRunning(): boolean {
        return aui.thread().getState().isRunning;
      },
      /** Replace the thread with `count` messages, oldest first. */
      seed(count: number, options?: SeedOptions): void {
        aui
          .thread()
          .import(
            ExportedMessageRepository.fromArray(buildMessages(count, options)),
          );
      },
      /** Toggle the Thread while the runtime lives, like a sidebar switch; isHovering outlives it. */
      setThreadMounted(mounted: boolean): void {
        setThreadMounted(mounted);
      },
      /** Reads the runtime flag so a leak test sees isHovering even when nothing is rendered. */
      hoverFlags(): { id: string; isHovering: boolean }[] {
        return aui
          .thread()
          .getState()
          .messages.map((message) => ({
            id: message.id,
            isHovering: Boolean(message.isHovering),
          }));
      },
      /** Cheap poll target; counts() is too heavy to poll at 500 messages. */
      messageCount(): number {
        return document.querySelectorAll("[data-role]").length;
      },
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
      /** Runtime's composer text; reading the textarea would just echo what the caller wrote. */
      composerText(): string {
        return aui.composer().getState().text;
      },
      katexCount(): number {
        return document.querySelectorAll(".katex").length;
      },
      /** Shiki runs after the <pre> exists, so counting <pre> gates nothing. */
      highlightedTokenCount(): number {
        return document.querySelectorAll("pre code span").length;
      },
      /** An empty popover would satisfy "the menu opened", so count items. */
      openMenuItemCount(): number {
        return document.querySelectorAll(".aui-action-bar-more-item").length;
      },
      lastAssistantMessage(): HTMLElement | null {
        const messages = document.querySelectorAll<HTMLElement>(
          '[data-role="assistant"]',
        );
        return messages[messages.length - 1] ?? null;
      },
      /** TooltipIconButton puts the name in an sr-only span, not aria-label, so match on text. */
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

// `?remote=1`: the remote thread list + attachments adapter the app uses, for the same scopes.
const REMOTE = new URLSearchParams(window.location.search).get("remote") === "1";

const MEMORY_THREAD_LIST: unstable_RemoteThreadListAdapter = {
  list: async () => ({ threads: [] }),
  rename: async () => {},
  archive: async () => {},
  unarchive: async () => {},
  delete: async () => {},
  initialize: async (threadId) => ({ remoteId: threadId, externalId: undefined }),
  generateTitle: async () =>
    new ReadableStream({ start: (controller) => controller.close() }) as never,
  fetch: async (threadId) => ({
    status: "regular",
    remoteId: threadId,
    externalId: undefined,
    title: undefined,
  }),
};

const ATTACHMENTS = new SimpleImageAttachmentAdapter();

function useSmokeLocalRuntime() {
  return useLocalRuntime(NEVER_RUNS, { adapters: { attachments: ATTACHMENTS } });
}

function RemoteHarness(): ReactElement {
  const runtime = unstable_useRemoteThreadListRuntime({
    runtimeHook: useSmokeLocalRuntime,
    adapter: MEMORY_THREAD_LIST,
  });
  return <HarnessBody runtime={runtime} />;
}

function LocalHarness(): ReactElement {
  const runtime = useLocalRuntime(NEVER_RUNS);
  return <HarnessBody runtime={runtime} />;
}

function Harness(): ReactElement {
  return REMOTE ? <RemoteHarness /> : <LocalHarness />;
}

function HarnessBody({
  runtime,
}: {
  runtime: ReturnType<typeof useLocalRuntime>;
}): ReactElement {
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

// Memory router: without one, every action bar console.warns from useNavigate, scaling with N.
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
