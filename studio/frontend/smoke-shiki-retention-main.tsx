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

type Kind = "stream" | "whole" | "prose" | "pair";

function bodyFor(kind: Kind, chars: number, seed: number): string {
  if (kind === "prose") return proseSource(chars, seed);
  return `\`\`\`python\n${pythonSource(chars, seed)}\n\`\`\``;
}

// TWO fences alive at once, where one is a strict prefix of the other. This shape is here because
// a cache bug that only appears under it hung a CI job for thirty minutes: bidirectional prefix
// eviction made the two fences evict each other forever, and since a miss schedules asynchronous
// work whose callback re-renders, the page never went idle. No single-fence arm can reach it. The
// retention question it answers is separate and also worth having: under one-directional
// eviction two live prefix-related fences legitimately occupy TWO entries, not one.
const PAIR_LEAD = 0.5;

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

type ReplyState = {
  messageId: string;
  text: string;
  streaming: boolean;
  /** Second, shorter reply rendered alongside the first. Only the `pair` kind sets it. */
  companion?: string;
};
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
    <>
      <Reply
        messageId={state.messageId}
        text={state.text}
        streaming={state.streaming}
      />
      {state.companion !== undefined && (
        <Reply
          messageId={`${state.messageId}-companion`}
          text={state.companion}
          streaming={state.streaming}
        />
      )}
    </>
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


// Frame recorder. Retained heap is not a user-facing number on its own, so the driver also asks
// whether a big retained heap actually costs frames.
//
// Two independent clocks, because either alone can lie. requestAnimationFrame stops being
// scheduled at all when the compositor decides nothing is on screen, which would read as "no
// dropped frames" rather than "no measurement"; a 1 ms setTimeout keeps ticking regardless, and
// the gap between its ticks IS the block, because the main thread cannot answer the timer while
// it is busy. The setTimeout clamp is about 4 ms, far below anything a user notices.
type FrameReport = {
  frames: number;
  durationMs: number;
  fps: number;
  medianFrameMs: number;
  p95FrameMs: number;
  worstFrameMs: number;
  framesOver33ms: number;
  longestStallMs: number;
};

const frameState = {
  running: false,
  rafTimes: [] as number[],
  timerGaps: [] as number[],
  startedAt: 0,
  rafHandle: 0,
  timerHandle: 0 as ReturnType<typeof setTimeout> | 0,
  lastTimerTick: 0,
};

const percentile = (sorted: number[], q: number): number => {
  if (sorted.length === 0) return 0;
  const index = Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1)));
  return sorted[index];
};

const framesStart = () => {
  frameState.running = true;
  frameState.rafTimes = [];
  frameState.timerGaps = [];
  frameState.startedAt = performance.now();
  frameState.lastTimerTick = frameState.startedAt;
  const onFrame = (now: number) => {
    if (!frameState.running) return;
    frameState.rafTimes.push(now);
    frameState.rafHandle = requestAnimationFrame(onFrame);
  };
  frameState.rafHandle = requestAnimationFrame(onFrame);
  const onTick = () => {
    if (!frameState.running) return;
    const now = performance.now();
    frameState.timerGaps.push(now - frameState.lastTimerTick);
    frameState.lastTimerTick = now;
    frameState.timerHandle = setTimeout(onTick, 1);
  };
  frameState.timerHandle = setTimeout(onTick, 1);
};

const framesStop = (): FrameReport => {
  frameState.running = false;
  cancelAnimationFrame(frameState.rafHandle);
  if (frameState.timerHandle) clearTimeout(frameState.timerHandle);
  const durationMs = performance.now() - frameState.startedAt;
  const intervals: number[] = [];
  for (let i = 1; i < frameState.rafTimes.length; i += 1) {
    intervals.push(frameState.rafTimes[i] - frameState.rafTimes[i - 1]);
  }
  const sorted = [...intervals].sort((a, b) => a - b);
  return {
    frames: frameState.rafTimes.length,
    durationMs,
    fps: durationMs > 0 ? (frameState.rafTimes.length * 1000) / durationMs : 0,
    medianFrameMs: percentile(sorted, 0.5),
    p95FrameMs: percentile(sorted, 0.95),
    worstFrameMs: sorted.length > 0 ? sorted[sorted.length - 1] : 0,
    framesOver33ms: intervals.filter((ms) => ms > 33).length,
    longestStallMs: frameState.timerGaps.reduce((a, b) => Math.max(a, b), 0),
  };
};

// Calibration for the recorder itself. A frame recorder that cannot see a block it was told about
// is not a frame recorder, so the driver blocks the main thread for a known number of milliseconds
// inside a recording window and refuses to report anything if that block does not show up.
const blockFor = (ms: number): void => {
  const until = performance.now() + ms;
  while (performance.now() < until) {
    // Busy on purpose: a sleep would yield the main thread, which is the opposite of the point.
  }
};

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
      framesStart: () => void;
      framesStop: () => FrameReport;
      blockFor: (ms: number) => void;
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
      push({
        messageId,
        text: body.slice(0, cut),
        streaming: i < steps,
        companion:
          kind === "pair"
            ? body.slice(0, Math.round(cut * PAIR_LEAD))
            : undefined,
      });
      await nextFrame();
      if (tickMs > 0 && i < steps) await sleep(tickMs);
    }
    // Final non-streaming commit, the way a finished reply lands.
    push({
      messageId,
      text: body,
      streaming: false,
      companion:
        kind === "pair"
          ? body.slice(0, Math.round(body.length * PAIR_LEAD))
          : undefined,
    });
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
  framesStart,
  framesStop,
  blockFor,
  ready: true,
};
