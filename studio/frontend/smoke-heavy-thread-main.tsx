// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Harness page for tests/studio/playwright_heavy_thread.py: the real Thread holding a HEAVY
// thread, so the measured cost of typing, scrolling, opening a menu, deleting and re-opening is
// the app's own and grows the way a long session's does.
//
// The axis is CHARACTERS OF THREAD CONTENT, not messages. Users report the slowdown after "long
// generations with any code cells and/or text", which is a statement about volume, so a fixture
// of N short paragraphs would measure the wrong variable.
//
// The content mix is the one the report names, and every kind of it is present at every size
// because the fixture is built in whole CYCLES of one message per kind (~26K characters a
// cycle). Building "until the budget runs out" instead would give the smallest size only the
// first few kinds, and the curve would then be reporting a change of fixture, not of size.
//
//   prose                long multi-paragraph answers
//   python fence         a large highlighted code fence
//   typescript fence     a second language, so one grammar is not the whole Shiki story
//   python tool call     collapsible card with a code-execution result pane
//   bash tool call       code_execution card, the other result-pane shape
//   html artifact        a full ```html document, which Studio collapses into an artifact card
//   render_html tool     the HTML canvas artifact, as a tool part rather than a fence
//   svg fence            a highlighted fence that also renders an inline <img> preview
//   image parts          a raster PNG data URL and a unique SVG data URL, as image content parts
//   json fence           one very long line, the shape that behaves worst in a highlighter
//
// Two honest limits, both of which the harness's own docstring repeats:
//
//   * The HTML artifact renders as an artifact CARD here, not as a live preview. The <iframe>
//     lives in ArtifactSurface on the chat page, and it loads its content from the backend
//     (/api/inference/artifact-preview-frame), so it cannot exist on a backend-free smoke page.
//     What is measured is the in-thread cost of an artifact, which is what a scroll pays.
//   * `python` results carry `images: []`. A non-empty list makes the card fetch each image from
//     the backend, which would put a network round trip inside a timed region.
//
// Same shape as smoke-autoscroll.html and smoke-thread-weight.html: a vite entry, no backend, no
// auth. Thread itself and the message bodies are real on purpose; mocking either deletes the
// measurement. The runtime is synthetic -- a local runtime whose model adapter never runs,
// seeded through `thread.import`.
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
import { type ReactElement, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./src/index.css";

// The local runtime's thread list item reports a synthetic `__LOCALID_...` remoteId, which is
// truthy, so the fork-count badges really do ask the backend. Answering them here, before anything
// mounts, keeps that off the wire entirely; answering them from the Playwright side instead would
// put a round trip to another process inside a timed region.
// An explicit allowlist, never a blanket `/api/` match. A blanket match resolves EVERY request the
// measured interactions make before Playwright emits it, so `stray_api_requests` stays at zero and
// the fan-out this harness exists to detect becomes invisible to it. Narrowing it is what made the
// two entries below visible in the first place.
//
// Each entry answers a request the harness itself provokes, with the body that endpoint really
// returns, so no round trip lands inside a timed region. Anything NOT listed here goes to the
// network and trips the stray counter, which is the point.
const STUBBED_API: ReadonlyArray<readonly [RegExp, string]> = [
  // The fork-count badges. One GET per THREAD, not per message: fork-count-store subscribes once
  // per rendered thread and refreshes on CHAT_HISTORY_UPDATED_EVENT, which the delete action
  // fires, so the harness provokes it during seeding and again inside the measured actions.
  //
  // The endpoint used to be per message, `/threads/{id}/messages/{id}/forks` answering
  // `{"count":n}`, and that is the shape this allowlist was written against. The per-thread
  // endpoint replaced it and this entry was not moved with it, so every one of these went to the
  // dev server and the run failed its own stray-request check. Match the shape the app actually
  // requests, and answer with the body it actually returns: `getThreadForkCounts` reads
  // `data.counts` and builds a Map from it, so `{"counts":{}}` is "no message has forks" and
  // renders no badge on any message. An empty `{}` body would leave the same empty Map today, but
  // it is not what the endpoint returns, and a stub that answers a shape the endpoint never sends
  // is how this drifted in the first place.
  [/\/api\/chat\/threads\/[^/]+\/forks$/, '{"counts":{}}'],
  // The delete action's own persistence. deleteThreadMessage syncs the exported repository
  // whenever remoteId is truthy, which the synthetic id always is, so this is the fixture
  // maintaining itself rather than app fan-out. Left on the wire it is 3 round trips inside the
  // delete measurement, and it fails the run's own stray check.
  [/\/api\/chat\/threads\/[^/]+\/messages$/, '{"messages":[]}'],
  [/\/api\/chat\/threads\/[^/]+$/, "{}"],
  // App fan-out, NOT fixture upkeep: re-opening a thread asks for the project list and the
  // knowledge bases. Stubbed so a dev-server round trip does not land inside the reopen window,
  // which would be measuring the network rather than the render. They are recorded in
  // `__stubbedApi` and printed as "stubbed api requests" rather than being silently swallowed,
  // because two whole-endpoint GETs per reopen is a real cost and should stay visible.
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
      // Recorded, not silently swallowed, so a run can still show what was answered locally.
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

// ── content generators ──────────────────────────────────────────────
//
// Every generator takes the block index and produces content that is UNIQUE to it. That is not
// decoration: code-plugin.ts caches highlighted tokens keyed on the exact source string, so a
// fixture that repeats one fence would highlight it once and hand back the cached result for
// every other copy, and the Shiki cost would vanish from the curve.

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
  // One very long line on purpose. A single-line document is the shape that behaves worst in a
  // highlighter and in any code that scans for line boundaries.
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
  // A full document, which is what Studio treats as an artifact rather than as a code block,
  // and it draws on a <canvas> so the fixture carries the shape users call a canvas artifact.
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
  // Studio renders a highlighted fence AND an inline <img> preview for an svg fence, so this one
  // block buys both a Shiki pass and an image decode. No <script>, no on*= handlers, no
  // <foreignObject>: any of those make the preview refuse to render and the block silently
  // becomes an ordinary code fence.
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

// 64x64 RGB PNG, 288 bytes. A real raster decode rather than a 1x1 placeholder, and small enough
// that its data URL does not dominate the character budget it is counted against.
const PNG_DATA_URL =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAA50lEQVR42u3PkVIlAABA0VcTBEEQBEGwECwEQRAEwUIQBAtBEARBEFwIgiAIgiAIFoIgCIIgWFi4EARBEARBEAQLQdBXBM3cmfMDZzDAERzFMRzHCZzEKZzGHziDP3EW53AeF3ARl/AXLuMKruJvXMN13MBN3MJt3EFwF/dwHw/wEI/wGE/wFP/gGZ7jBV7iFV7jDf7Ffyje4h3e4wM+4hM+4wu+4hv+x3f8wOHB8NC3VqBAgQIFChQoUKBAgQIFChQoUKBAgQIFChQoUKBAgQIFChQoUKBAgQIFChQoUKBAgQIFCny9T1mrnz8ZmAtoAAAAAElFTkSuQmCC";

const pngCache = new Map<number, string>();

/**
 * A PNG that is unique to this block, so each one is a separate decode rather than a cache hit
 * on the last.
 *
 * Drawn and exported here rather than embedded as a literal because assistant-ui only accepts
 * `data:image/(png|jpeg|jpg|gif|webp);base64,` for an image part and drops anything else with a
 * console warning -- an SVG data URL, which would have been the easy way to vary the bytes, is
 * silently discarded.
 */
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

// ── the cycle ───────────────────────────────────────────────────────

type Part = NonNullable<Exclude<ThreadMessageLike["content"], string>>[number];

/** One block: the user's prompt and the assistant messages it produced. */
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
      // python is one of the two tools the renderer never folds into a tool GROUP, so its card
      // stays a top-level collapsible even when it sits next to other calls.
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
                // All three keys, or the card falls back to stringifying the whole object.
                // `images` stays empty: a non-empty list makes the card fetch each one from a
                // backend this page does not have.
                result: { text: toolOutput(index, 1400), images: [], sessionId: `heavy-${index}` },
              },
            ],
          },
        ],
      };
    case 4:
      // No text part in this message on purpose. CodeExecutionToolUI is a CONTROLLED collapsible
      // that force-closes itself as soon as its message carries text, so a card with prose beside
      // it cannot be held open for the measurement.
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
              // One repeated attachment and one that is unique to this block: a thread has both,
              // and only the second is a fresh decode every time.
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

/**
 * What one cycle must put on screen. The Python side multiplies these by the cycle count and
 * fails the run if the DOM holds fewer.
 *
 * This is not belt and braces. assistant-ui drops an image part whose data URL is not
 * `data:image/(png|jpeg|jpg|gif|webp);base64,` with nothing but a console.warn, and a fixture
 * that quietly loses its images still renders 300K characters of prose and code and still
 * produces a rising curve -- of the wrong thing.
 */
const EXPECTED_PER_CYCLE: Record<string, number> = {
  // Two image parts in the image block, plus the inline preview the svg fence renders.
  images: 3,
  // The python card and the code_execution card. render_html renders an artifact card instead.
  toolParts: 2,
  collapsibleOutputs: 2,
  codeExecutionPanes: 2,
  // The full ```html document and the render_html tool call.
  artifactCards: 2,
  // python, typescript, json, svg, the html document, and the two tool result panes.
  codeBlocks: 7,
  /*
   * THE CODE ITSELF, in characters: the one size measure deferral cannot move.
   *
   * This was a floor of 2,500 highlighted tokens, doing two jobs at once. Off-screen fences now
   * render as a plain shell, so tokens partly measure where the viewport is: the same fixture
   * renders 1,322 per cycle where it rendered 3,216. Characters do not move, because the shell
   * carries the same text node: 12,660 for one cycle and 51,081 for four (2 of 5 and 15 of 20
   * fences deferred), and Chromium, Firefox and WebKit each report 12,660 to the character.
   *
   * Floor 12,000 against 12,660; a cycle is seven blocks averaging ~1,800 chars, so losing any one
   * block still fails. The other job, telling a settled thread from one still building itself, is
   * now `unhighlightedMountedFences`, asked per block.
   */
  codeChars: 12000,
};

/**
 * Whole cycles of one block per kind until `targetChars` is reached, never fewer than one.
 *
 * Whole cycles, not "blocks until the budget runs out": a partial cycle would give the smallest
 * size only the first few kinds of content, so the curve across sizes would be reporting a change
 * of fixture as well as a change of size.
 */
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

// A run would need a backend. Seeding goes through `thread.import`, which does not use this.
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
  // What seed() last built. The delete action removes a message from the REPOSITORY, so without
  // a way to put it back every repetition after the first measures a shorter thread than the one
  // the harness took its census of.
  const seeded = useRef<ThreadMessageLike[]>([]);

  useEffect(() => {
    const api = {
      /** Replace the thread with whole cycles of content until `targetChars` is reached. */
      seed(targetChars: number): Plan {
        const built = buildThread(targetChars);
        seeded.current = built.messages;
        aui.thread().import(ExportedMessageRepository.fromArray(built.messages));
        return built.plan;
      },
      /**
       * Like seed(), then N one-word messages after it. Used by the viewport-gap measurement in
       * #9058: the tail makes the first mount commit land entirely on compact rows, which is the
       * worst case for a fixed-size initial window.
       *
       * It REPLACES rather than appends, and the heavy part is the same buildThread() call seed()
       * makes, so every census count for `targetChars` is identical to seed(targetChars). The only
       * difference is the tail, which is text parts only: no fences, images or tool calls.
       */
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
      /**
       * The empty band below the last mounted row, in px.
       *
       * gapBottom is measured against the viewport's BOTTOM EDGE, not against scrollHeight, so the
       * viewport's own bottom spacer counts as the gap it always was and the caller subtracts
       * spacerHeight to get the part the mount window is responsible for. Computed any other way
       * the numbers stop being comparable across sizes.
       */
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
      /**
       * Put the seeded thread back, and answer with how many messages that is.
       *
       * Deleting a message is destructive to the repository, not to the view, so re-opening does
       * not undo it. Restoring is the same import seed() does: cheap on the harness side, and it
       * is untimed, but every timed repetition then runs on the same fixture rather than on a
       * thread that is one message shorter each time round.
       */
      restore(): number {
        aui.thread().import(ExportedMessageRepository.fromArray(seeded.current));
        return seeded.current.length;
      },
      /**
       * Open every tool card. Radix unmounts collapsed content, so a thread of closed cards
       * carries no result panes at all and the "tool calls with collapsible output" half of the
       * fixture would be a row of buttons. A user who has just watched those tools run is
       * looking at them open, which is the state worth measuring.
       */
      expandTools(): number {
        const triggers = Array.from(
          document.querySelectorAll<HTMLElement>('[data-slot="tool-fallback-trigger"]'),
        );
        for (const trigger of triggers) {
          if (trigger.getAttribute("data-state") !== "open") trigger.click();
        }
        return triggers.length;
      },
      /** Leave the thread. The runtime keeps the messages; the view is torn down. */
      closeThread(): void {
        setMounted(false);
      },
      /** Come back to it, which rebuilds every message from nothing. */
      openThread(): void {
        setMounted(true);
      },
      isOpen(): boolean {
        return mounted;
      },
      /**
       * One selector pass. Polling for a deletion has to read this, not counts(): counts() is a
       * dozen document-wide queries including a walk of every element, so at 300K characters a
       * poll loop built on it would spend more time measuring than the delete itself takes.
       */
      messageCount(): number {
        return document.querySelectorAll("[data-role]").length;
      },
      /** Highlighted tokens. Shiki runs after the <pre> exists, so counting <pre> gates nothing. */
      highlightedTokenCount(): number {
        return document.querySelectorAll("pre code span").length;
      },
      /**
       * Everything a caller might use to prove the fixture landed. A harness that asks for 300K
       * characters of mixed content and silently renders 200K of prose measures the wrong thing,
       * so the Python side prints every one of these and fails on any that is zero.
       */
      counts(): Record<string, number> {
        return {
          messages: document.querySelectorAll("[data-role]").length,
          assistantMessages: document.querySelectorAll('[data-role="assistant"]').length,
          userMessages: document.querySelectorAll('[data-role="user"]').length,
          domNodes: document.getElementsByTagName("*").length,
          codeBlocks: document.querySelectorAll("pre").length,
          highlightedTokens: document.querySelectorAll("pre code span").length,
          /*
           * HOW MUCH CODE IS ON THE PAGE. `highlightedTokens` cannot answer that any more: a
           * deferred fence renders as a plain shell, so tokens measure where the reader is
           * looking. The shell holds the same text node the highlighted block holds (which is
           * also why selection, clipboard and find-in-page are identical across the two states),
           * so characters read the same either way and still drop if the fixture loses code.
           */
          codeChars: Array.from(document.querySelectorAll("pre code")).reduce(
            (total, node) => total + (node.textContent?.length ?? 0),
            0,
          ),
          fenceBlocks: document.querySelectorAll('[data-streamdown="code-block"]').length,
          deferredFences: document.querySelectorAll("[data-unsloth-fence-deferred]").length,
          /*
           * A fence that is NEITHER deferred NOR highlighted, at rest: the settlement half of the
           * old token floor, asked per block. Streamdown mounts a code block on its own
           * unhighlighted fallback and colours it from a passive effect, so this state exists for
           * a frame on any build and a settled thread must hold none. Stronger than a floor on
           * the total, which one stuck block passes as long as the others make up the count.
           */
          unhighlightedMountedFences: Array.from(
            document.querySelectorAll('[data-streamdown="code-block"]'),
          ).filter(
            (block) =>
              !block.hasAttribute("data-unsloth-fence-deferred") &&
              block.querySelector("pre code span") === null,
          ).length,
          toolParts: document.querySelectorAll(".aui-tool-fallback-root").length,
          // The collapsible CONTENT ELEMENT, which Radix keeps in the tree for its collapse
          // animation. It is present whether the card is open or shut, so it counts cards, not
          // visible panes: measured at 25000 chars it reads 2 with every card closed and 2 again
          // after expandTools(). Do NOT gate a wait on this; such a gate is satisfied by a thread
          // of closed cards and cannot fail. Use codeExecutionPanes, which is 0 then 2.
          collapsibleOutputs: document.querySelectorAll(
            '[data-slot="tool-fallback-content"]',
          ).length,
          codeExecutionPanes: document.querySelectorAll(
            '[data-slot="tool-fallback-content"] pre',
          ).length,
          // ArtifactCard carries no class of its own; the accessible name is its stable handle.
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
      /**
       * What the RUNTIME thinks the composer holds. Reading the textarea back instead would only
       * echo the value the caller just wrote, so a keystroke that never reached React would still
       * look like it landed.
       */
      composerText(): string {
        return aui.composer().getState().text;
      },
      /** Items in the open action menu. An empty popover satisfies "the menu opened". */
      openMenuItemCount(): number {
        return document.querySelectorAll(".aui-action-bar-more-item").length;
      },
      lastAssistantMessage(): HTMLElement | null {
        const messages = document.querySelectorAll<HTMLElement>('[data-role="assistant"]');
        return messages[messages.length - 1] ?? null;
      },
      /**
       * The last assistant message's action-bar button with accessible name `label`.
       * TooltipIconButton puts that name in an `sr-only` span rather than an aria-label, so this
       * matches on text and stays correct if the styling classes are renamed.
       */
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
  // The runtime lives outside this flag, so unmounting Thread leaves the messages intact. That is
  // what re-opening a thread costs: the view is rebuilt, the data is not re-fetched.
  const [mounted, setMounted] = useState(true);
  return (
    <TooltipProvider>
      <AssistantRuntimeProvider runtime={runtime}>
        <HeavyThreadApi mounted={mounted} setMounted={setMounted} />
        {/* Thread is flex-1 basis-0 min-h-0, so it needs a bounded flex parent to scroll. */}
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

// Thread reaches useNavigate (the fork action, the composer tools menu). Without a router in
// context tanstack's useRouter still works, but console.warns on every render of every action
// bar, which scales with the thread and is serialised over the debugging channel. A memory router
// with one route removes that without pulling in the app shell.
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
