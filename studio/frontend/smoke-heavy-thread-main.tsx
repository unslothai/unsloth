// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Harness for tests/studio/playwright_heavy_thread.py: real Thread with a heavy thread built in
// whole cycles of mixed content; local runtime since delete uses thread.export/import.

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
import { type ReactElement, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./src/index.css";

// Explicit allowlist, never a blanket `/api/` match: unlisted requests must trip the stray counter.
const STUBBED_API: ReadonlyArray<readonly [RegExp, string]> = [
  // Fork-count badges, one GET per thread; answer with the real `{"counts":{}}` shape.
  [/\/api\/chat\/threads\/[^/]+\/forks$/, '{"counts":{}}'],
  // The delete action's own persistence (synthetic remoteId is truthy).
  [/\/api\/chat\/threads\/[^/]+\/messages$/, '{"messages":[]}'],
  [/\/api\/chat\/threads\/[^/]+$/, "{}"],
  // App fan-out on reopen, stubbed but recorded in `__stubbedApi`.
  [/\/api\/chat\/projects(\?|$)/, '{"projects":[]}'],
  [/\/api\/rag\/knowledge-bases(\?|$)/, '{"knowledge_bases":[]}'],
];

const stubbedApiCalls: string[] = [];
(window as unknown as { __stubbedApi: string[] }).__stubbedApi = stubbedApiCalls;

const realFetch = window.fetch.bind(window);
window.fetch = (input, init) => {
  const url =
    typeof input === "string" ? input : ((input as Request).url ?? String(input));
  for (const [pattern, body] of STUBBED_API) {
    if (pattern.test(url)) {
      stubbedApiCalls.push(url);
      return Promise.resolve(
        new Response(body, {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      );
    }
  }
  return realFetch(input, init);
};

// Content must be unique per block: code-plugin.ts caches highlighting on the exact source.

const PROSE_SENTENCES = [
  "The reception of a long thread is decided by what the renderer does on every interaction, not by what it did once at load.",
  "A reply that arrives quickly can still leave a thread that answers a keystroke slowly, because the two costs are paid in different places.",
  "Anything that walks the whole message list on each frame turns a pleasant session into an unpleasant one somewhere around the twentieth long answer.",
  "The tell is that generation stays fast while the surrounding interface does not, which points at per-message renderer work rather than at the model.",
  "Layout that is not contained propagates upward, so a change inside one message can force the entire column to be measured again.",
  "None of this is visible on a short thread, which is why the fixture here is sized in characters rather than in messages.",
];

function prose(index: number, targetChars: number): string {
  const parts: string[] = [`Reply ${index}.`];
  let length = parts[0].length;
  let cursor = index;
  while (length < targetChars) {
    const sentence = PROSE_SENTENCES[cursor % PROSE_SENTENCES.length];
    parts.push(`${sentence} (paragraph ${cursor - index + 1} of reply ${index})`);
    length += parts[parts.length - 1].length + 2;
    cursor += 1;
  }
  return parts.join("\n\n");
}

function pythonFence(index: number, targetChars: number): string {
  const lines = [
    "```python",
    `# reply ${index}: a batch scorer, long enough that the highlighter has real work to do`,
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
      `def score_${index}_${step}(rows: list[Row${index}]) -> float:`,
      `    """Step ${step} of reply ${index}: sum the weighted counts, skipping the empties."""`,
      "    total = 0.0",
      "    for row in rows:",
      `        if row.count == 0 or row.label == "skip-${step}":`,
      "            continue",
      `        total += row.weight * row.count * {0}.{1}`.replace("{0}", String(step + 1)).replace("{1}", String((index * 7 + step) % 97)),
      "    return total",
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
    `// reply ${index}: the client half, so the fixture is not one grammar repeated`,
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
      `export function score${index}_${step}(rows: readonly Row${index}[]): number {`,
      "  let total = 0;",
      "  for (const row of rows) {",
      `    if (row.count === 0 || row.label === "skip-${step}") continue;`,
      `    total += row.weight * row.count * ${step + 1}.${(index * 5 + step) % 89};`,
      "  }",
      "  return total;",
      "}",
      "",
    );
    step += 1;
  }
  lines.push("```");
  return lines.join("\n");
}

function jsonFence(index: number, targetChars: number): string {
  const entries: string[] = [];
  let step = 0;
  let body = "";
  while (body.length < targetChars) {
    entries.push(
      `{"id":"row-${index}-${step}","weight":${(step % 17) + 0.5},"count":${step * 3 + index},"label":"batch-${index}-${step}"}`,
    );
    body = `{"reply":${index},"rows":[${entries.join(",")}]}`;
    step += 1;
  }
  return ["```json", body, "```"].join("\n");
}

function htmlArtifact(index: number, targetChars: number): string {
  const head = [
    "<!doctype html>",
    "<html>",
    "  <head>",
    `    <title>Reply ${index} canvas</title>`,
    "    <style>",
    "      body { margin: 0; font-family: system-ui, sans-serif; background: #101014; color: #f4f4f5; }",
    "      canvas { display: block; width: 100%; height: 240px; }",
    "    </style>",
    "  </head>",
    "  <body>",
    `    <canvas id="plot-${index}" width="640" height="240"></canvas>`,
    "    <script>",
    `      const ctx = document.getElementById("plot-${index}").getContext("2d");`,
  ];
  const tail = ["    </script>", "  </body>", "</html>"];
  const body: string[] = [];
  let step = 0;
  while ([...head, ...body, ...tail].join("\n").length < targetChars) {
    body.push(
      `      ctx.fillStyle = "hsl(${(index * 31 + step * 7) % 360}, 70%, 55%)";`,
      `      ctx.fillRect(${step * 12}, ${(step * 5) % 200}, 10, ${20 + (step % 40)});`,
    );
    step += 1;
  }
  return ["```html", ...head, ...body, ...tail, "```"].join("\n");
}

function svgFence(index: number, targetChars: number): string {
  // No <script>, on*= or <foreignObject>: any of those disable the inline svg preview.
  const open = [`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 160" width="320" height="160">`];
  const close = ["</svg>"];
  const body: string[] = [];
  let step = 0;
  while ([...open, ...body, ...close].join("\n").length < targetChars) {
    body.push(
      `  <rect x="${(step * 9) % 300}" y="${(step * 6) % 140}" width="14" height="${8 + (step % 40)}" fill="hsl(${(index * 23 + step * 11) % 360}, 65%, 55%)" />`,
    );
    step += 1;
  }
  return ["```svg", ...open, ...body, ...close, "```"].join("\n");
}

function pythonScript(index: number, targetChars: number): string {
  const lines = [`# tool script for block ${index}`, "import json", ""];
  let step = 0;
  while (lines.join("\n").length < targetChars) {
    lines.push(
      `rows_${step} = [{"weight": ${step + 1}, "count": ${index + step}}]`,
      `print(json.dumps({"step": ${step}, "total": sum(r["weight"] * r["count"] for r in rows_${step})}))`,
    );
    step += 1;
  }
  return lines.join("\n");
}

function toolOutput(index: number, targetChars: number): string {
  const lines: string[] = [];
  let step = 0;
  while (lines.join("\n").length < targetChars) {
    lines.push(
      `[block ${index}] step ${step}: processed ${step * 128 + index} rows, ${((step * 37) % 100) / 10} MB resident`,
    );
    step += 1;
  }
  return lines.join("\n");
}

function bashOutput(index: number, targetChars: number): string {
  const lines: string[] = [`total ${index * 4 + 12}`];
  let step = 0;
  while (lines.join("\n").length < targetChars) {
    lines.push(
      `-rw-r--r--  1 unsloth  staff  ${step * 733 + index}  Jan ${(step % 28) + 1} 09:${String((step * 7) % 60).padStart(2, "0")}  shard-${index}-${step}.safetensors`,
    );
    step += 1;
  }
  return lines.join("\n");
}

const PNG_DATA_URL =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAA50lEQVR42u3PkVIlAABA0VcTBEEQBEGwECwEQRAEwUIQBAtBEARBEFwIgiAIgiAIFoIgCIIgWFi4EARBEARBEAQLQdBXBM3cmfMDZzDAERzFMRzHCZzEKZzGHziDP3EW53AeF3ARl/AXLuMKruJvXMN13MBN3MJt3EFwF/dwHw/wEI/wGE/wFP/gGZ7jBV7iFV7jDf7Ffyje4h3e4wM+4hM+4wu+4hv+x3f8wOHB8NC3VqBAgQIFChQoUKBAgQIFChQoUKBAgQIFChQoUKBAgQIFChQoUKBAgQIFChQoUKBAgQIFCny9T1mrnz8ZmAtoAAAAAElFTkSuQmCC";

const pngCache = new Map<number, string>();

/** Unique PNG per block. assistant-ui drops non png/jpeg/gif/webp data URLs, so no SVG. */
function pngDataUrl(index: number): string {
  const cached = pngCache.get(index);
  if (cached !== undefined) return cached;
  let url = PNG_DATA_URL;
  try {
    const canvas = document.createElement("canvas");
    canvas.width = 96;
    canvas.height = 64;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.fillStyle = `hsl(${(index * 47) % 360}, 60%, 40%)`;
      ctx.fillRect(0, 0, 96, 64);
      ctx.fillStyle = `hsl(${(index * 91) % 360}, 70%, 65%)`;
      ctx.beginPath();
      ctx.arc(24 + (index % 48), 34, 18, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#ffffff";
      ctx.font = "12px sans-serif";
      ctx.fillText(`block ${index}`, 6, 16);
      url = canvas.toDataURL("image/png");
    }
  } catch {
    url = PNG_DATA_URL;
  }
  pngCache.set(index, url);
  return url;
}

type Part = NonNullable<Exclude<ThreadMessageLike["content"], string>>[number];

type Block = { user: string; assistant: ThreadMessageLike[] };

function textPart(text: string): Part {
  return { type: "text", text };
}

const KIND_COUNT = 10;

function buildBlock(index: number): Block {
  const kind = index % KIND_COUNT;
  const ask = `Prompt ${index}. Walk me through step ${index} of the batch scorer, and show the code and the run.`;
  switch (kind) {
    case 0:
      return { user: ask, assistant: [{ role: "assistant", content: [textPart(prose(index, 2600))] }] };
    case 1:
      return {
        user: ask,
        assistant: [
          {
            role: "assistant",
            content: [
              textPart(
                `${prose(index, 300)}\n\n${pythonFence(index, 2900)}\n\nThat is the scoring half.`,
              ),
            ],
          },
        ],
      };
    case 2:
      return {
        user: ask,
        assistant: [
          {
            role: "assistant",
            content: [
              textPart(
                `${prose(index, 300)}\n\n${typescriptFence(index, 2900)}\n\nAnd the client half.`,
              ),
            ],
          },
        ],
      };
    case 3:
      // python is never folded into a tool group, so its card stays top-level.
      return {
        user: ask,
        assistant: [
          {
            role: "assistant",
            content: [
              textPart(`Running it now for block ${index}.`),
              {
                type: "tool-call",
                toolCallId: `heavy-python-${index}`,
                toolName: "python",
                args: { code: pythonScript(index, 900) },
                // All three keys or the card stringifies the object; non-empty `images` would hit the backend.
                result: { text: toolOutput(index, 1400), images: [], sessionId: `heavy-${index}` },
              },
            ],
          },
        ],
      };
    case 4:
      // No text part: CodeExecutionToolUI force-closes once its message carries text.
      return {
        user: ask,
        assistant: [
          {
            role: "assistant",
            content: [
              {
                type: "tool-call",
                toolCallId: `heavy-bash-${index}`,
                toolName: "code_execution",
                args: { kind: "bash", command: `ls -la /workspace/shards/block-${index}` },
                result: bashOutput(index, 1700),
              },
            ],
          },
        ],
      };
    case 5:
      return {
        user: ask,
        assistant: [
          {
            role: "assistant",
            content: [
              textPart(
                `Here is the preview for block ${index}.\n\n${htmlArtifact(index, 2400)}\n\nOpen it to see the plot.`,
              ),
            ],
          },
        ],
      };
    case 6:
      return {
        user: ask,
        assistant: [
          {
            role: "assistant",
            content: [
              textPart(`Rendering the canvas for block ${index}.`),
              {
                type: "tool-call",
                toolCallId: `heavy-canvas-${index}`,
                toolName: "render_html",
                args: {
                  title: `Block ${index} canvas`,
                  code: htmlArtifact(index, 2300).replace(/^```html\n/, "").replace(/\n```$/, ""),
                },
                result: `Rendered HTML canvas for block ${index}`,
              },
            ],
          },
        ],
      };
    case 7:
      return {
        user: ask,
        assistant: [
          {
            role: "assistant",
            content: [
              textPart(`A diagram for block ${index}.\n\n${svgFence(index, 1500)}`),
            ],
          },
        ],
      };
    case 8:
      return {
        user: ask,
        assistant: [
          {
            role: "assistant",
            content: [
              textPart(`Two attachments for block ${index}: a rendered plot and a chart.`),
              { type: "image", image: PNG_DATA_URL },
              { type: "image", image: pngDataUrl(index) },
              textPart(prose(index, 400)),
            ],
          },
        ],
      };
    default:
      return {
        user: ask,
        assistant: [
          {
            role: "assistant",
            content: [
              textPart(`The raw rows for block ${index}.\n\n${jsonFence(index, 2700)}`),
            ],
          },
        ],
      };
  }
}

function partChars(part: Part): number {
  const anyPart = part as Record<string, unknown>;
  if (anyPart.type === "text") return String(anyPart.text ?? "").length;
  if (anyPart.type === "image") return String(anyPart.image ?? "").length;
  if (anyPart.type === "tool-call") {
    return (
      JSON.stringify(anyPart.args ?? {}).length +
      (typeof anyPart.result === "string"
        ? anyPart.result.length
        : JSON.stringify(anyPart.result ?? "").length)
    );
  }
  return 0;
}

function messageChars(message: ThreadMessageLike): number {
  if (typeof message.content === "string") return message.content.length;
  return message.content.reduce((total, part) => total + partChars(part as Part), 0);
}

type Plan = {
  chars: number;
  messages: number;
  blocks: number;
  cycles: number;
  kinds: number;
  cycleChars: number;
  expectedPerCycle: Record<string, number>;
};

/** Per-cycle DOM expectations; assistant-ui silently drops bad image parts. */
const EXPECTED_PER_CYCLE: Record<string, number> = {
  images: 3,
  toolParts: 2,
  collapsibleOutputs: 2,
  codeExecutionPanes: 2,
  artifactCards: 2,
  codeBlocks: 7,
  /* Code characters, not tokens: deferred fences render as plain shells. 12,660 per cycle. */
  codeChars: 12000,
};

function buildThread(targetChars: number): { messages: ThreadMessageLike[]; plan: Plan } {
  const messages: ThreadMessageLike[] = [];
  let chars = 0;
  let cycles = 0;
  let blocks = 0;
  let cycleChars = 0;
  do {
    const cycleStart = chars;
    for (let k = 0; k < KIND_COUNT; k += 1) {
      const block = buildBlock(blocks);
      const user: ThreadMessageLike = { role: "user", content: [textPart(block.user)] };
      messages.push(user, ...block.assistant);
      chars += messageChars(user);
      for (const message of block.assistant) chars += messageChars(message);
      blocks += 1;
    }
    cycles += 1;
    if (cycleChars === 0) cycleChars = chars - cycleStart;
  } while (chars < targetChars);
  return {
    messages,
    plan: {
      chars,
      messages: messages.length,
      blocks,
      cycles,
      kinds: KIND_COUNT,
      cycleChars,
      expectedPerCycle: EXPECTED_PER_CYCLE,
    },
  };
}

const NEVER_RUNS: ChatModelAdapter = {
  run: () => {
    throw new Error("smoke-heavy-thread does not run the model");
  },
};

function HeavyThreadApi({
  mounted,
  setMounted,
}: {
  mounted: boolean;
  setMounted: (value: boolean) => void;
}): null {
  const aui = useAui();
  // Delete is destructive to the repository, so keep the seed to restore it.
  const seeded = useRef<ThreadMessageLike[]>([]);

  useEffect(() => {
    const api = {
      seed(targetChars: number): Plan {
        const built = buildThread(targetChars);
        seeded.current = built.messages;
        aui.thread().import(ExportedMessageRepository.fromArray(built.messages));
        return built.plan;
      },
      seedCompactTail(targetChars: number, tailMessages: number): Plan {
        const built = buildThread(targetChars);
        const messages = built.messages.slice();
        const SHORT = ["ok", "thanks", "yes", "got it", "sure", "nice"];
        for (let i = 0; i < tailMessages; i += 1) {
          messages.push({
            role: i % 2 === 0 ? "user" : "assistant",
            content: [textPart(SHORT[i % SHORT.length])],
          });
        }
        seeded.current = messages;
        aui.thread().import(ExportedMessageRepository.fromArray(messages));
        return { ...built.plan, messages: messages.length };
      },
      /** Gap below the last mounted row, measured to the viewport bottom edge. */
      gapMetrics(): Record<string, number> {
        const element = api.viewport();
        if (!element) return { ok: 0 };
        const clientHeight = element.clientHeight;
        const rows = Array.from(element.querySelectorAll<HTMLElement>("[data-role]"));
        if (rows.length === 0) return { ok: 0, mountedRows: 0, clientHeight };
        const box = element.getBoundingClientRect();
        const first = rows[0].getBoundingClientRect();
        const last = rows[rows.length - 1].getBoundingClientRect();
        const spacer = element.querySelector<HTMLElement>(
          ':scope > [aria-hidden="true"].shrink-0',
        );
        const scrollHeight = element.scrollHeight;
        return {
          ok: 1,
          mountedRows: rows.length,
          clientHeight,
          scrollHeight,
          scrollTop: Math.round(element.scrollTop),
          maxScrollTop: Math.round(scrollHeight - clientHeight),
          mountedHeight: Math.round(last.bottom - first.top),
          gapTop: Math.round(first.top - box.top),
          gapBottom: Math.round(box.bottom - last.bottom),
          spacerHeight: spacer ? Math.round(spacer.getBoundingClientRect().height) : 0,
        };
      },
      restore(): number {
        aui.thread().import(ExportedMessageRepository.fromArray(seeded.current));
        return seeded.current.length;
      },
      /** Radix unmounts collapsed content, so open the cards to measure result panes. */
      expandTools(): number {
        const triggers = Array.from(
          document.querySelectorAll<HTMLElement>('[data-slot="tool-fallback-trigger"]'),
        );
        for (const trigger of triggers) {
          if (trigger.getAttribute("data-state") !== "open") trigger.click();
        }
        return triggers.length;
      },
      closeThread(): void {
        setMounted(false);
      },
      openThread(): void {
        setMounted(true);
      },
      isOpen(): boolean {
        return mounted;
      },
      /** Cheap single query for polling; counts() is too expensive at large sizes. */
      messageCount(): number {
        return document.querySelectorAll("[data-role]").length;
      },
      highlightedTokenCount(): number {
        return document.querySelectorAll("pre code span").length;
      },
      counts(): Record<string, number> {
        return {
          messages: document.querySelectorAll("[data-role]").length,
          assistantMessages: document.querySelectorAll('[data-role="assistant"]').length,
          userMessages: document.querySelectorAll('[data-role="user"]').length,
          domNodes: document.getElementsByTagName("*").length,
          codeBlocks: document.querySelectorAll("pre").length,
          highlightedTokens: document.querySelectorAll("pre code span").length,
          /* Characters, not tokens: deferred shells hold the same text node. */
          codeChars: Array.from(document.querySelectorAll("pre code")).reduce(
            (total, node) => total + (node.textContent?.length ?? 0),
            0,
          ),
          fenceBlocks: document.querySelectorAll('[data-streamdown="code-block"]').length,
          deferredFences: document.querySelectorAll("[data-unsloth-fence-deferred]").length,
          /* Neither deferred nor highlighted at rest; a settled thread must hold none. */
          unhighlightedMountedFences: Array.from(
            document.querySelectorAll('[data-streamdown="code-block"]'),
          ).filter(
            (block) =>
              !block.hasAttribute("data-unsloth-fence-deferred") &&
              block.querySelector("pre code span") === null,
          ).length,
          toolParts: document.querySelectorAll(".aui-tool-fallback-root").length,
          // Present whether open or shut, so never gate a wait on it; use codeExecutionPanes.
          collapsibleOutputs: document.querySelectorAll(
            '[data-slot="tool-fallback-content"]',
          ).length,
          codeExecutionPanes: document.querySelectorAll(
            '[data-slot="tool-fallback-content"] pre',
          ).length,
          // ArtifactCard has no class; the accessible name is its stable handle.
          artifactCards: document.querySelectorAll('button[aria-label^="Open "]').length,
          images: document.querySelectorAll("img").length,
          katexNodes: document.querySelectorAll(".katex").length,
          actionBars: document.querySelectorAll(".aui-assistant-action-bar-root").length,
          tooltipTriggers: document.querySelectorAll('[data-slot="tooltip-trigger"]').length,
        };
      },
      viewportMetrics(): { scrollHeight: number; scrollTop: number; clientHeight: number } {
        const element = api.viewport();
        if (!element) return { scrollHeight: -1, scrollTop: -1, clientHeight: -1 };
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
        return document.querySelector<HTMLTextAreaElement>(".aui-composer-input");
      },
      /** Read from the runtime: the textarea would just echo what the caller wrote. */
      composerText(): string {
        return aui.composer().getState().text;
      },
      openMenuItemCount(): number {
        return document.querySelectorAll(".aui-action-bar-more-item").length;
      },
      lastAssistantMessage(): HTMLElement | null {
        const messages = document.querySelectorAll<HTMLElement>('[data-role="assistant"]');
        return messages[messages.length - 1] ?? null;
      },
      /** TooltipIconButton puts the name in an sr-only span, so match on text. */
      actionButton(label: string): HTMLButtonElement | null {
        const last = api.lastAssistantMessage();
        if (!last) return null;
        const buttons = Array.from(last.querySelectorAll("button"));
        return buttons.find((button) => (button.textContent ?? "").trim() === label) ?? null;
      },
    };
    (window as unknown as { __heavyThread: typeof api }).__heavyThread = api;
  }, [aui, mounted, setMounted]);

  return null;
}

function Harness(): ReactElement {
  const runtime = useLocalRuntime(NEVER_RUNS);
  const [mounted, setMounted] = useState(true);
  return (
    <TooltipProvider>
      <AssistantRuntimeProvider runtime={runtime}>
        <HeavyThreadApi mounted={mounted} setMounted={setMounted} />
        <div
          data-smoke="heavy-thread"
          style={{ display: "flex", flexDirection: "column", height: "100vh" }}
        >
          {mounted ? <Thread hideWelcome={true} /> : null}
        </div>
      </AssistantRuntimeProvider>
    </TooltipProvider>
  );
}

// A memory router avoids per-render useRouter console warnings without the app shell.
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
