// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Fixture page for tests/studio/playwright_shiki_retention.py.
//
// It streams one assistant reply at a time through the SAME markdown pipeline the chat uses --
// the production `createCodePlugin` from code-plugin.ts, the production Shiki themes, the
// production `stabilizeStreamingMarkdown` + `IncrementalMarkdownCache`, and
// `<Streamdown mode="streaming">` -- and then UNMOUNTS it. Whatever heap survives the unmount
// plus a forced GC is retained by a module-level cache, not by the DOM, which is the thing under
// test.
//
// The page exposes an imperative API on `window` rather than reacting to DOM events, because the
// driver has to interleave forced-GC heap samples with stream ticks and needs a promise per step.

import { createCodePlugin as createShikiCodePlugin } from "@streamdown/code";
import { createMathPlugin } from "@streamdown/math";
import { useEffect, useState } from "react";
import { createRoot, type Root } from "react-dom/client";
import { Streamdown } from "streamdown";
import { createCodePlugin } from "@/components/assistant-ui/code-plugin";
import {
  unslothDarkTheme,
  unslothLightTheme,
} from "@/components/assistant-ui/code-themes";
import { stabilizeStreamingMarkdown } from "@/components/assistant-ui/streaming-markdown";
import { IncrementalMarkdownCache } from "@/components/assistant-ui/streaming-render-schedule";
import { preprocessLaTeX } from "@/lib/latex";
import "katex/dist/katex.min.css";

type HighlightPlugin = ReturnType<typeof createCodePlugin>;
type HighlightArgs = Parameters<HighlightPlugin["highlight"]>;

// Counters for the settle predicate. `pending` is dispatches that returned null (asynchronous
// Shiki work) minus callbacks that have landed; a synchronous cache hit never calls back, so
// counting it as pending would leave the page permanently unsettled.
const counters = { renderCalls: 0, pending: 0 };

const THEMES = [unslothLightTheme, unslothDarkTheme];

const productionCode = createCodePlugin({ themes: THEMES });
// Counting wrapper. It forwards verbatim: nothing here changes what is highlighted, only what is
// observable about when the highlighting finished.
const code: HighlightPlugin = {
  ...productionCode,
  highlight: (opts: HighlightArgs[0], callback?: HighlightArgs[1]) => {
    counters.renderCalls += 1;
    // Per-call bookkeeping, and asymmetric on purpose. Studio's throttle answers a call
    // synchronously with reused tokens AND ALSO stores its callback for a later trailing
    // dispatch, so a callback can land for a call that never went asynchronous. Decrementing on
    // every callback drives the counter negative and the page never reads as settled.
    const state = { counted: false, done: false };
    const result = productionCode.highlight(opts, (value) => {
      if (state.counted && !state.done) {
        state.done = true;
        counters.pending -= 1;
      }
      callback?.(value);
    });
    if (result === null && !state.done) {
      state.counted = true;
      counters.pending += 1;
    }
    return result;
  },
};
const math = createMathPlugin({ singleDollarTextMath: true });
const PLUGINS = { code, math };
const IMMEDIATE_UPDATES = { duration: 0, stagger: 0 };

// Deterministic fixture text. A seeded LCG, so the two arms of a before/after comparison generate
// byte-identical fixtures without shipping a 32 KB blob; the driver hashes what it got and
// refuses to compare arms whose hashes differ.
function lcg(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

const IDENTIFIERS = [
  "compute",
  "buffer",
  "offset",
  "tensor",
  "weights",
  "stride",
  "handle",
  "result",
  "payload",
  "session",
  "adapter",
  "checkpoint",
  "gradient",
  "scheduler",
  "vocab",
];
const KEYWORDS = ["if", "for", "while", "return", "def", "class", "with", "try"];

// Python-shaped source, so Shiki actually has grammar work to do. A blob of prose inside a fence
// would understate the token count and therefore the retention.
function pythonSource(chars: number, seed: number): string {
  const rand = lcg(seed);
  const pick = (xs: string[]): string => xs[Math.floor(rand() * xs.length)];
  const lines: string[] = [];
  let total = 0;
  let n = 0;
  while (total < chars) {
    n += 1;
    const indent = "    ".repeat(1 + (n % 3));
    const kind = Math.floor(rand() * 5);
    let line: string;
    if (kind === 0) {
      line = `def ${pick(IDENTIFIERS)}_${n}(${pick(IDENTIFIERS)}, ${pick(IDENTIFIERS)} = ${Math.floor(rand() * 4096)}):`;
    } else if (kind === 1) {
      line = `${indent}${pick(IDENTIFIERS)} = ${pick(IDENTIFIERS)}[${Math.floor(rand() * 64)}] * ${(rand() * 10).toFixed(4)}`;
    } else if (kind === 2) {
      line = `${indent}# ${pick(IDENTIFIERS)} ${pick(IDENTIFIERS)} ${pick(IDENTIFIERS)} ${pick(IDENTIFIERS)}`;
    } else if (kind === 3) {
      line = `${indent}${pick(KEYWORDS)} ${pick(IDENTIFIERS)} in ${pick(IDENTIFIERS)}_${n}: ${pick(IDENTIFIERS)}("${pick(IDENTIFIERS)}")`;
    } else {
      line = `${indent}${pick(IDENTIFIERS)} = {"${pick(IDENTIFIERS)}": ${Math.floor(rand() * 999)}, "${pick(IDENTIFIERS)}": ${pick(IDENTIFIERS)}}`;
    }
    lines.push(line);
    total += line.length + 1;
  }
  return lines.join("\n").slice(0, chars);
}

// The control fixture: the same character budget as prose, so the markdown parser and React do
// comparable work while the code-highlighter cache is never touched.
function proseSource(chars: number, seed: number): string {
  const rand = lcg(seed);
  const words = [
    "quantisation",
    "adapter",
    "throughput",
    "gradient",
    "attention",
    "sequence",
    "kernel",
    "residual",
    "embedding",
    "checkpoint",
    "scheduler",
    "tokeniser",
  ];
  const parts: string[] = [];
  let total = 0;
  let n = 0;
  while (total < chars) {
    n += 1;
    const sentence = Array.from(
      { length: 9 + Math.floor(rand() * 8) },
      () => words[Math.floor(rand() * words.length)],
    ).join(" ");
    const line = n % 6 === 0 ? `\n${sentence}.` : `${sentence}.`;
    parts.push(line);
    total += line.length + 1;
  }
  return parts.join(" ").slice(0, chars);
}

type Kind = "stream" | "whole" | "prose";

function bodyFor(kind: Kind, chars: number, seed: number): string {
  if (kind === "prose") return proseSource(chars, seed);
  return `\`\`\`python\n${pythonSource(chars, seed)}\n\`\`\``;
}

// Mirrors MarkdownTextImpl: same preprocessing, same incremental cache, same Streamdown props on
// the highlighting path.
function Reply({
  messageId,
  text,
  streaming,
}: {
  messageId: string;
  text: string;
  streaming: boolean;
}) {
  // One cache per mount rather than the ref-and-compare MarkdownTextImpl uses, because the driver
  // mounts a fresh host for every reply, so the message can never change under this component.
  const [cache] = useState(() => new IncrementalMarkdownCache());
  const processed = stabilizeStreamingMarkdown(preprocessLaTeX(text), streaming);
  const incremental = streaming ? cache.update(processed) : null;
  return (
    <div data-smoke="reply" data-status={streaming ? "running" : "complete"}>
      <Streamdown
        key={`${messageId}:${cache.renderGeneration}`}
        mode="streaming"
        parseIncompleteMarkdown={!incremental}
        parseMarkdownIntoBlocksFn={incremental?.parseMarkdownIntoBlocks}
        isAnimating={streaming}
        animated={IMMEDIATE_UPDATES}
        plugins={PLUGINS}
        shikiTheme={THEMES}
      >
        {incremental?.markdown ?? processed}
      </Streamdown>
    </div>
  );
}

type ReplyState = { messageId: string; text: string; streaming: boolean };
type Push = (next: ReplyState) => void;

function Host({ register }: { register: (push: Push) => void }) {
  const [state, setState] = useState<ReplyState>({
    messageId: "seed",
    text: "",
    streaming: false,
  });
  useEffect(() => {
    register(setState);
  }, [register]);
  if (!state.text) return null;
  return (
    <Reply
      messageId={state.messageId}
      text={state.text}
      streaming={state.streaming}
    />
  );
}

const container = document.getElementById("root");
if (!container) throw new Error("missing #root");

let root: Root | null = null;
let push: Push | null = null;
let mountPoint: HTMLElement | null = null;

const nextFrame = () =>
  new Promise<void>((resolve) => {
    requestAnimationFrame(() => resolve());
  });
const sleep = (ms: number) =>
  new Promise<void>((resolve) => {
    setTimeout(resolve, ms);
  });

async function mount(): Promise<void> {
  mountPoint = document.createElement("div");
  container.appendChild(mountPoint);
  // No StrictMode: its double-invoke would double every dispatch count and make the settle
  // predicate read a state the app never reaches.
  root = createRoot(mountPoint);
  await new Promise<void>((resolve) => {
    root?.render(
      <Host
        register={(fn) => {
          push = fn;
          resolve();
        }}
      />,
    );
  });
}

async function unmount(): Promise<void> {
  root?.unmount();
  root = null;
  push = null;
  if (mountPoint) {
    mountPoint.remove();
    mountPoint = null;
  }
  container.replaceChildren();
  await nextFrame();
  await nextFrame();
}

// FNV-1a over the fixture. Cheap, and only ever compared for equality between arms.
function hash(text: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return `${h.toString(16)}:${text.length}`;
}

// A second, bare instance of the upstream plugin, driven directly. It shares the module-level
// caches inside @streamdown/code with the app path, which is the point: this measures the cost of
// ONE cache entry without React, layout or the DOM anywhere in the picture. The driver gives it
// its own seeds so its keys never collide with the app path's.
const bareCode = createShikiCodePlugin({ themes: THEMES });

async function rawEntries(
  chars: number,
  seed: number,
  entries: number,
): Promise<number> {
  const source = pythonSource(chars, seed);
  let landed = 0;
  await Promise.all(
    Array.from({ length: entries }, (_unused, i) => {
      const cut = Math.round((source.length * (i + 1)) / entries);
      return new Promise<void>((resolve) => {
        const immediate = bareCode.highlight(
          {
            code: source.slice(0, cut),
            language: "python",
            themes: THEMES as unknown as HighlightArgs[0]["themes"],
          },
          () => {
            landed += 1;
            resolve();
          },
        );
        if (immediate) {
          landed += 1;
          resolve();
        }
      });
    }),
  );
  return landed;
}

declare global {
  interface Window {
    __sd: {
      counters: () => { renderCalls: number; pending: number };
      fixtureHash: (kind: Kind, chars: number, seed: number) => string;
      runOne: (spec: {
        kind: Kind;
        chars: number;
        seed: number;
        ticks: number;
        tickMs: number;
      }) => Promise<{ renderCalls: number; domNodes: number; textLength: number }>;
      rawEntries: (
        chars: number,
        seed: number,
        entries: number,
      ) => Promise<number>;
      teardown: () => Promise<void>;
      ready: boolean;
    };
  }
}

let messageCounter = 0;

window.__sd = {
  counters: () => ({ ...counters }),
  fixtureHash: (kind, chars, seed) => hash(bodyFor(kind, chars, seed)),
  runOne: async ({ kind, chars, seed, ticks, tickMs }) => {
    const body = bodyFor(kind, chars, seed);
    messageCounter += 1;
    const messageId = `msg-${messageCounter}`;
    const renderCallsBefore = counters.renderCalls;
    await unmount();
    await mount();
    if (!push) throw new Error("host did not register");
    const steps = kind === "whole" ? 1 : ticks;
    for (let i = 1; i <= steps; i += 1) {
      const cut = Math.round((body.length * i) / steps);
      push({ messageId, text: body.slice(0, cut), streaming: i < steps });
      await nextFrame();
      if (tickMs > 0 && i < steps) await sleep(tickMs);
    }
    // Final non-streaming commit, the way a finished reply lands.
    push({ messageId, text: body, streaming: false });
    await nextFrame();
    return {
      renderCalls: counters.renderCalls - renderCallsBefore,
      domNodes: container.querySelectorAll("*").length,
      textLength: body.length,
    };
  },
  rawEntries,
  teardown: async () => {
    await unmount();
  },
  ready: true,
};
