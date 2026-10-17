// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import type { HighlightResult } from "@streamdown/code";
import {
  memo,
  type CSSProperties,
  type RefObject,
  startTransition,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { flushSync } from "react-dom";

import { MAX_HIGHLIGHT_CHARS } from "@/lib/markdown-plugins";
import { FENCE_HEIGHT_PROPERTY } from "./code-block-containment-mode";
import { type FenceMode, resolveFenceMode } from "./code-fence-mode";
import { fenceLineForeground } from "./code-fence-foreground";
import {
  FENCE_CHUNK_LINES,
  type FenceLineChunks,
  sameFenceBodyProps,
  updateFenceChunks,
} from "./code-fence-chunks";
import {
  type PlainFenceChunk,
  type PlainFenceChunks,
  updatePlainFenceChunks,
} from "./plain-fence-chunks";
import {
  createFenceHighlightQueue,
} from "./fence-highlight-queue";
import {
  initialHighlightProgress,
  advanceHighlightProgress,
  EMPTY_HIGHLIGHT_PROGRESS,
  EMPTY_HIGHLIGHT_COUNTS,
  HIGHLIGHT_BATCH_TOKENS,
  type HighlightProgress,
  needsHighlightBatches,
  retainHighlightProgress,
} from "./fence-highlight-progress";
import {
  EMPTY_LINE_WINDOW,
  HYSTERESIS_VIEWPORTS,
  isBlankLine,
  type LineWindow,
  type LinePins,
  lineRendered,
  pinBoundaryLines,
  samePins,
  OVERSCAN_VIEWPORTS,
  plainLineText,
  selectLineWindow,
  WINDOW_CAP_LINES,
} from "./code-fence-window";
import { planPrintTokenization } from "./code-fence-print";
import { createFenceSpeculator } from "./fence-speculation";
import { highlightWorkerState, streamActive } from "./use-reasoning-highlight";
import { inputQuietIn, scheduleQuietIdleTask } from "@/lib/schedule-idle-task";
import { inputPointerHeld } from "@/lib/input-pointer-state";
import {
  MAX_CACHED_CHARACTERS,
  MAX_FENCES,
  normalizeLanguage,
} from "./code-plugin";

/*
 * MONOTONIC fence highlighting: a fence renders as a plain shell until the first time it comes near
 * the viewport, and is highlighted for the rest of its mount. There is no reverse edge.
 * An earlier attempt gated on viewport entry AND exit, so scrolling away tore the highlighted
 * subtree down and scrolling back rebuilt it: a gesture that ought to cost nothing scheduled a full
 * re-tokenize, and it measured slower. Here the only transition is cheap -> expensive, at most once
 * per mount, so the worst case is exactly today's cost and every fence never reached is saved.
 * The shell is the same text in the same elements streamdown's own unhighlighted fallback would
 * use, carrying the same `data-streamdown` attributes so index.css sizes it unchanged. The text is
 * in the DOM and selectable, which `content-visibility: auto` is not (skipped content has to be
 * rendered before it can be selected, +28.8% on select_all_ms).
 * A STREAMING fence never defers: the block being written is the one the reader is looking at.
 */

/*
 * How far outside the viewport a fence counts as "reached": one viewport of slack each way, so the
 * upgrade lands a frame or two before the block is on screen.
 * THE PERCENTAGE RESOLVES AGAINST THE ROOT'S HEIGHT, and the spec says otherwise: Intersection
 * Observer 2.2 resolves percentages against the undilated rectangle's WIDTH for all four sides, no
 * engine does that for top and bottom, and w3c/IntersectionObserver#391 is open on it. Measured as
 * the height on Chromium, Firefox and WebKit alike, which `inBand` and the jump test both assume.
 * Guarded rather than trusted: `pf9462_parity.py` re-measures two geometries every run and fails if
 * the observer's lookahead and the pre-paint band stop agreeing.
 */
const REACH_MARGIN = "100% 0px";

/*
 * The mode decision lives in `code-fence-mode.ts`, a JSX-free `.ts` so a test can RUN the table
 * rather than regex this file. Re-exported here because consumers already import this module.
 */
export { type FenceMode, resolveFenceMode, SHIP_DEFAULT } from "./code-fence-mode";

export const fenceMode = (): FenceMode =>
  resolveFenceMode(
    (globalThis as Record<string, unknown>).__UNSLOTH_DEFER_FENCE_HIGHLIGHT__,
    readBuildFlag(),
  );

const readBuildFlag = (): string => {
  try {
    return import.meta.env.VITE_UNSLOTH_DEFER_FENCE_HIGHLIGHT ?? "";
  } catch {
    return "";
  }
};

// Streamdown trims trailing newlines off a fence body before rendering it, so the shell has to as
// well or the two differ by a blank line of height. The length is split out because `warmGrammars`
// needs the size of what a warm WOULD tokenize, and slicing a 20,000 character fence to measure it
// is a copy per render.
export const trimmedLength = (text: string): number => {
  let end = text.length;
  while (end > 0 && text[end - 1] === "\n") end -= 1;
  return end;
};

export const trimTrailingNewlines = (text: string): string =>
  text.slice(0, trimmedLength(text));

/*
 * AN EMPTY FENCE IS ONE LINE TALL, not nothing. Streamdown renders one span per token line and
 * special-cases the empty line to a single newline, having trimmed trailing newlines first exactly as
 * `trimTrailingNewlines` does here, so an empty body renders as one line box of height. A `<code>`
 * holding an empty text node has no line box at all, so without this the shell is one line shorter
 * than the block it stands in for and everything below it moves when the fence upgrades.
 */
const shellBody = (source: string): string => {
  const trimmed = trimTrailingNewlines(source);
  return trimmed === "" ? "\n" : trimmed;
};

const PlainFenceTextChunk = memo(function PlainFenceTextChunk({
  nodes,
}: {
  nodes: PlainFenceChunk;
}) {
  return <>{nodes}</>;
});

function PlainFenceText({ source }: { source: string }) {
  const [previous, setPrevious] = useState<PlainFenceChunks>(() => updatePlainFenceChunks(null, source));
  const current = previous.source === source ? previous : updatePlainFenceChunks(previous, source);
  if (current !== previous) setPrevious(current);
  return <>{current.chunks.map((nodes, index) => (
    <PlainFenceTextChunk key={index} nodes={nodes} />
  ))}</>;
}

function FenceShell({
  language,
  source,
  plain = false,
}: {
  language: string | null;
  source: string;
  plain?: boolean;
}) {
  return (
    <div
      className="my-4 flex w-full flex-col gap-2 rounded-xl border border-border bg-sidebar p-2"
      data-language={language ?? undefined}
      data-streamdown="code-block"
      data-unsloth-fence-deferred={plain ? undefined : "true"}
    >
      <div
        className="flex h-8 items-center text-muted-foreground text-xs"
        data-language={language ?? undefined}
        data-streamdown="code-block-header"
      >
        <span className="ml-1 font-mono lowercase">{language}</span>
      </div>
      <div
        className="overflow-x-auto rounded-md border border-border bg-background p-4 text-sm"
        data-language={language ?? undefined}
        data-streamdown="code-block-body"
      >
        <pre>
          <code>{plain ? <PlainFenceText source={source} /> : shellBody(source)}</code>
        </pre>
      </div>
    </div>
  );
}

/**
 * Has this fence been reached yet? Latches true and never returns false again.
 *
 * Takes a ref to an element the CALLER already renders rather than mounting a wrapper: an extra div
 * would sit between a list item and its code block, breaking
 * `[data-streamdown="list-item"] > [data-streamdown="code-block"]`, and would push the block one
 * level deeper than index.css's `:last-child` margin chain walks. That is a layout change smuggled
 * in by a performance change, which is the one thing an A/B must not carry.
 */
// No IntersectionObserver, no gate. Read once at module scope so the decision is part of the
// rendered value rather than a state write from inside an effect, which would cost a cascading
// render on every fence in the thread.
const CAN_OBSERVE =
  typeof IntersectionObserver !== "undefined" &&
  typeof globalThis !== "undefined";

/*
 * THE REGISTER OF FENCES NOT YET REACHED, for the two things that must reach across all of them at
 * once without waiting for a React render: a discontinuous scroll, and a print, which puts the
 * WHOLE document on the page. Both go through `latchNow`. A gate carries the two elements its
 * observers were built against, resolved at the same moment and rebuilt with them, so nothing here
 * re-walks the ancestor chain or re-reads a computed style. `warm` comes from the caller because
 * the highlighter instance lives with the block's component.
 */
type FenceGate = {
  node: HTMLElement;
  near: HTMLElement | null;
  outer: HTMLElement | null;
  language: string | null;
  /** Upper bound on what `warm(true)` would tokenize; trimming only removes trailing newlines. */
  chars: number;
  /** `true` tokenizes this fence's source now; `false` only loads its grammar. */
  warm: (tokens: boolean) => void;
  speculate: (settle: (seeded: boolean) => void) => (() => void) | null;
  latch: () => void;
  /** A state write that changes nothing, whose only job is to give React sync work to do. */
  poke: () => void;
};

const unreached = new Set<FenceGate>();
const pendingReaches = new Map<FenceGate, () => void>();

const deferReaches = (): boolean =>
  fenceMode() === "defer" && !printing && !findBarOpen();

const queueReach = (gate: FenceGate): void => {
  if (pendingReaches.has(gate)) return;
  const cancel = queueFenceWork(() => {
    pendingReaches.delete(gate);
    if (unreached.has(gate) && gateOpen(gate)) latchNow([gate]);
  });
  pendingReaches.set(gate, cancel);
};

/** Is this gate's fence where the observers would call it reached? */
const gateOpen = (gate: FenceGate): boolean =>
  inBand(gate.node, gate.near)
  && (gate.near === gate.outer || inBand(gate.near as HTMLElement, gate.outer));

/*
 * UPGRADE THESE FENCES INSIDE THIS TASK, so the browser paints them highlighted rather than
 * painting the plain shell and correcting it frames later. Dropping any of the three steps was
 * measured to put a painted plain frame back:
 *   1. `warm(true)`. Streamdown's highlighted body falls back to plain whenever the plugin answers
 *      `null`, which it does only while a grammar loads; with the grammar in hand it tokenizes in
 *      the same call, so the render below is a cache hit.
 *   2. The inner `flushSync(latch)`, since a normally scheduled update lands after the next paint.
 *   3. The inner `flushSync(poke)`, INSIDE an outer `flushSync`. Step 2 does not produce the
 *      COLOURED commit: `HighlightedCodeBlockBody` starts at `useState(raw)` and asks for tokens
 *      from a PASSIVE effect, so every newly mounted block renders unhighlighted once.
 * Why the poke and the nesting, since an empty `flushSync(() => {})` does neither: React runs
 * pending passive effects from `performSyncWorkOnRoot`, reached only when sync work is waiting, and
 * `poke` is that work. And `flushSync` restores update priority BEFORE performing the flush, so the
 * passive effect's update would resolve against the ambient priority (a scroll is continuous, so
 * not flushed); the outer call holds the priority discrete across its whole body.
 * One way only. Nothing here can clear a latch.
 */
const latchNow = (arrived: readonly FenceGate[]): void => {
  if (arrived.length === 0) return;
  for (const gate of arrived) {
    pendingReaches.get(gate)?.();
    pendingReaches.delete(gate);
    unreached.delete(gate);
    fenceSpeculator.remove(gate);
    gate.warm(true);
  }
  flushSync(() => {
    flushSync(() => {
      for (const gate of arrived) gate.latch();
    });
    flushSync(() => {
      for (const gate of arrived) gate.poke();
    });
  });
};

/*
 * A DISCONTINUOUS SCROLL: a scrollbar drag, Ctrl+End, an anchor jump, a restored position.
 * Neither the observers' one root height of lookahead (`REACH_MARGIN`) nor the render-time
 * pre-paint gate covers a jump: the viewport can move further than the lookahead in one step with
 * no React render in between, and an IntersectionObserver record is delivered one or more frames
 * AFTER the paint (measured: 8 of 16 seeded jumps painted 3 to 4 frames of plain code).
 * A scroll listener is enough and is in time: scroll events are dispatched in the "run the scroll
 * steps" of the same update-the-rendering pass that will paint the new position, before
 * animation-frame callbacks and before style, layout and paint, so a `flushSync` from here is part
 * of the frame the reader is about to see.
 * The threshold is derived, not tuned: the band is one root height `h` bigger each way, so after a
 * scroll of `d` the newly visible strip `[d, d + h]` is inside the old band exactly when `d <= h`.
 * One listener, not one per fence: scroll does not bubble but does capture, so one capturing
 * document listener sees every element including nested reasoning panes. Attached when the first
 * fence registers and removed when the last latches.
 */
const lastScrollTop = new WeakMap<EventTarget, number>();
let scrollWatched = false;

const onScroll = (event: Event): void => {
  if (unreached.size === 0) return;
  if (deferReaches()) {
    fenceSpeculator.rerank();
    return;
  }
  const target = event.target;
  if (target === null) return;
  const element = target === document ? null : (target as HTMLElement);
  const top = element ? element.scrollTop : window.scrollY;
  const height = element ? element.clientHeight : window.innerHeight;
  const before = lastScrollTop.get(target);
  lastScrollTop.set(target, top);
  // An unseen scroller has no previous position, so its first event counts as a jump: one pass per
  // scroller, and never an assumption that the movement was small.
  if (before !== undefined && Math.abs(top - before) <= height) return;
  const arrived: FenceGate[] = [];
  for (const gate of unreached) {
    if (gateOpen(gate)) arrived.push(gate);
  }
  latchNow(arrived);
  fenceSpeculator.rerank();
};

const watchScrolling = (): void => {
  if (scrollWatched || typeof document === "undefined") return;
  scrollWatched = true;
  document.addEventListener("scroll", onScroll, { capture: true, passive: true });
};

const unwatchScrolling = (): void => {
  if (!scrollWatched || unreached.size > 0 || typeof document === "undefined") return;
  scrollWatched = false;
  document.removeEventListener("scroll", onScroll, { capture: true });
};

/*
 * PRINT, the one gesture that puts every deferred fence on the page at once.
 * Colour is all deferral costs a printed page, since the shell holds a live text node, but a page
 * that lost the colour on fences the reader never scrolled past is a defect they keep. Nor does the
 * printed window match the reader's: a print lays out at PAPER width while the scroll offset
 * carries across as raw pixels, so the page lands several fences away (matching paper to window
 * removes the difference, which is why chasing the window is the wrong fix). The whole document is
 * on the page, so every fence is revealed (highlighting capped by PRINT_TOKENIZE_CHARS).
 * An earlier attempt latched every fence from `beforeprint` with `flushSync` and 53 of 56 still
 * printed on streamdown's raw fallback, because the swap alone renders UNHIGHLIGHTED while the
 * passive effect waits on a grammar. `latchNow` closes both halves and `warmGrammars` keeps a
 * loading grammar from being what is missing at snapshot.
 * BOTH DOORS: `beforeprint` covers Ctrl+P and the print menu; headless `page.pdf()` and DevTools
 * print emulation change the media query without firing it.
 * A PRINT UPGRADES THE DOCUMENT THAT WAS PRINTED, AND NOTHING ELSE. This was a module-global
 * `printed` folded into every future fence's `reached`, so one Ctrl+P turned the default off for
 * the tab's life, including threads never on the printed page. So a print latches what is on the
 * page WHEN IT HAPPENS and a fence mounted afterwards defers again. Still one way only: reverting
 * on `afterprint` would be the bidirectional edge this design avoids.
 */
const upgradeEverythingForPrint = (): void => {
  latchNow([...unreached]);
};

/*
 * GRAMMARS, WARMED AT IDLE, ON REAL TEXT, ONE TOKENIZATION PER TASK.
 * One fence per language, so `latchNow`'s synchronous path cannot be defeated by a still-loading
 * grammar: on a jump into a language the reader has not met, and on a print, where there is no
 * later frame to correct in. Nothing runs when nothing is deferred. "Per language" means per
 * GRAMMAR, `normalizeLanguage`, not per fence tag: ```py and ```python are one grammar, and two
 * keys here would warm it twice on two different fences.
 * IT USED TO WARM ON AN EMPTY STRING. Loading a grammar is the cheap half; running it over text the
 * first time is not, and `""` never does the second, so the first REAL tokenization still paid the
 * whole one-off cost, landing in one frame during a scroll (worst scroll frame 1200 ms against
 * 185 ms warming on real text, three arms out of one build). Moved, not skipped: total tokenize
 * time RISES, because a warmed fence is tokenized once and read from cache later.
 * WHY THE TOKENIZATIONS YIELD AND THE LOADS DO NOT. An idle callback only chooses when it STARTS:
 * nothing yields once it runs, its 2,000 ms timeout can start it on a busy thread, and WebKitGTK
 * has no `requestIdleCallback` at all. With a grammar already loaded `code.highlight` tokenizes
 * INLINE and N languages concatenate. But `highlight` answers `null` WHILE a grammar loads, so
 * yielding the loads too would put the fifth grammar 500 ms x N away and a jump or print inside
 * that window would get the plain fallback out of `latchNow`'s flush, the defect this pre-warm
 * exists to prevent. So every load starts in the first pass and only the tokenizing is spread out.
 * NOTHING BOUNDED THE SIZE OF A WARM, and this comment used to claim `MAX_HIGHLIGHT_CHARS` did. It
 * does not reach here: `markdown-text.tsx` supplies the code plugin unconditionally, `FenceBlock`
 * warms the whole body, and `code-plugin.ts`'s `evict` keeps the last fence whatever its size. A
 * LATCH is demanded work and stays uncapped; a warm is SPECULATIVE, so it is capped at the same
 * 20,000 characters. Over the cap, and for a fence that is empty or nothing but newlines, the
 * grammar loads and nothing is tokenized; neither marks the language warmed, so a later fence that
 * can warm it still does.
 */
const grammarsWarmed = new Set<string>();
const grammarsLoaded = new Set<string>();
let warmScheduled = false;

export const grammarWarmed = (language: string | null): boolean =>
  grammarsWarmed.has(normalizeLanguage(language ?? "text"));

// Keyed the way `highlight` keys it, or `py` and `Python` are two keys for one grammar.
const grammarOf = (gate: FenceGate): string =>
  normalizeLanguage(gate.language ?? "text");

const warmMustWait = (): boolean =>
  inputPointerHeld() || inputQuietIn() > 0
  || (typeof (globalThis as Record<string, unknown>).requestIdleCallback !== "function"
    && streamActive());

const warmGrammars = (): void => {
  warmScheduled = false;
  if (unreached.size === 0) return;
  if (warmMustWait()) {
    scheduleGrammarWarm();
    return;
  }
  // EVERY GRAMMAR STARTS LOADING IN THE FIRST TASK. A load is cheap and asynchronous, and it is
  // what `latchNow` needs already present; only the tokenizations below are worth yielding for.
  for (const gate of unreached) {
    const language = grammarOf(gate);
    if (grammarsLoaded.has(language)) continue;
    grammarsLoaded.add(language);
    gate.warm(false);
  }
  for (const gate of unreached) {
    const language = grammarOf(gate);
    if (grammarsWarmed.has(language)) continue;
    // An EMPTY fence would tokenize `""` and teach this loop nothing, and one over the cap is not
    // ours to tokenize speculatively. Neither marks the grammar warmed, so a later fence in the
    // same language still gets its real warm.
    if (gate.chars === 0 || gate.chars > MAX_HIGHLIGHT_CHARS) continue;
    if (warmMustWait()) {
      scheduleGrammarWarm();
      return;
    }
    grammarsWarmed.add(language);
    // TRUE, not false: real text is what takes the one-off tokenizer cost off the scroll.
    gate.warm(true);
    // Yield. `grammarsWarmed` only grows, so the chain drains a language per task; a pass that
    // warms nothing falls out of the loop and schedules nothing.
    scheduleGrammarWarm();
    return;
  }
};

const scheduleGrammarWarm = (): void => {
  if (warmScheduled || typeof globalThis === "undefined") return;
  warmScheduled = true;
  const quiet = inputQuietIn();
  if (inputPointerHeld() || quiet > 0) {
    setTimeout(warmGrammars, Math.max(80, quiet));
    return;
  }
  const idle = (globalThis as Record<string, unknown>).requestIdleCallback as
    | ((cb: () => void, options?: { timeout: number }) => number)
    | undefined;
  if (typeof idle === "function") idle(warmGrammars, { timeout: 2000 });
  else setTimeout(warmGrammars, 500);
};

const rankGates = (gates: readonly FenceGate[]): number[] => {
  const boxes = new Map<HTMLElement | null, DOMRect>();
  return gates.map((gate) => {
    let box = boxes.get(gate.outer);
    if (box === undefined) {
      box = gate.outer
        ? gate.outer.getBoundingClientRect()
        : new DOMRect(0, 0, window.innerWidth, window.innerHeight);
      boxes.set(gate.outer, box);
    }
    const rect = gate.node.getBoundingClientRect();
    if (rect.bottom < box.top) return box.top - rect.bottom;
    if (rect.top > box.bottom) return rect.top - box.bottom;
    return 0;
  });
};

const SPECULATION_IN_FLIGHT = 3;

const fenceSpeculator = createFenceSpeculator<FenceGate>({
  idle: (callback, timeout) => scheduleQuietIdleTask(callback, timeout),
  wait: (callback, ms) => {
    const timer = setTimeout(callback, ms);
    return () => clearTimeout(timer);
  },
  enabled: () =>
    fenceMode() !== "off" && highlightWorkerState() !== "unavailable",
  busy: () => streamActive() || highlightWorkerState() === "stalled"
    || inputPointerHeld() || inputQuietIn() > 0,
  eligible: (gate) => grammarsWarmed.has(grammarOf(gate)),
  rank: rankGates,
  maxChars: MAX_HIGHLIGHT_CHARS,
  budgetChars: MAX_CACHED_CHARACTERS / 2,
  budgetFences: MAX_FENCES / 2,
  maxInFlight: SPECULATION_IN_FLIGHT,
});

if (typeof window !== "undefined" && typeof window.addEventListener === "function") {
  window.addEventListener("beforeprint", () => {
    upgradeEverythingForPrint();
    setPrinting(true);
  });
  window.addEventListener("afterprint", () => setPrinting(false));
  window.matchMedia?.("print")?.addEventListener?.("change", (event) => {
    if (event.matches) {
      upgradeEverythingForPrint();
      setPrinting(true);
    } else {
      setPrinting(false);
    }
  });
}

/*
 * THE NEAREST SCROLLING ANCESTOR, found rather than named.
 * This used to match two known selectors and `closest()` walks straight past anything matching
 * neither. The one that matters is the reasoning pane: while a reply streams, `reasoning.tsx` gives
 * its trace `overflow-y-auto` and `max-h-64` and pins it to the bottom, so the reader looks at an
 * arbitrarily long trace through a 256 px window nested inside the thread scroller.
 * Not a correctness fix: intermediate scrollers clip, so intersection was always computed
 * correctly. It is a LOOKAHEAD fix. `rootMargin` expands the ROOT's rectangle, so rooting at the
 * thread viewport expanded a rectangle the reader is not looking through and the one-viewport
 * warning was worth nothing inside the pane (3 of 10 with and without the margin, against 5 of 10
 * rooted at the inner scroller). Getting it wrong shows the plain shell for the frames the upgrade
 * takes.
 * `null` is deliberately still possible and deliberately NOT the default: a fence with no scrolling
 * ancestor really is clipped by the document viewport, but assuming that when a scroller exists is
 * the bug the review caught.
 */
let scrollableNow: WeakMap<Element, boolean> | null = null;

const forgetScrollable = (): void => {
  scrollableNow = null;
};

const isScrollable = (el: HTMLElement): boolean => {
  if (scrollableNow === null) {
    scrollableNow = new WeakMap();
    queueMicrotask(forgetScrollable);
  }
  const known = scrollableNow.get(el);
  if (known !== undefined) return known;
  const overflowY = getComputedStyle(el).overflowY;
  const scrollable =
    (overflowY === "auto" || overflowY === "scroll" || overflowY === "overlay")
    && el.scrollHeight > el.clientHeight;
  scrollableNow.set(el, scrollable);
  return scrollable;
};

const scrollerOf = (node: HTMLElement): HTMLElement | null => {
  for (let el = node.parentElement; el !== null; el = el.parentElement) {
    if (isScrollable(el)) return el;
  }
  return null;
};

/*
 * THE OUTERMOST ONE TOO, and why the nearest is not enough on its own.
 * An explicit root is clipped by the ancestors BETWEEN the target and the root, and by NOTHING
 * above it, so rooting at the reasoning pane asks "is this fence inside the pane's window" and
 * never "is the pane anywhere near the reader". A pane scrolled far out of the thread still reports
 * the fences inside its 256 px window as intersecting, and `reasoning.tsx` drops `max-h-64` when
 * the stream ends but KEEPS `overflow-y-auto`, so the pane stops being scrollable, its box becomes
 * the whole trace, and an observer still rooted at it reports EVERY fence at once.
 * Rooting at the outermost scroller instead would fix both and cost the lookahead the nearest one
 * bought (5 of 10 inner against 3 of 10 outer, measured). So when the two differ there are two
 * gates and the latch needs both: the FENCE against the nearest scroller answers "the reader is
 * about to reach it inside the pane", and the PANE against the outermost answers "the pane is
 * somewhere the reader can see". Watching the fence through the outer root instead would clip it at
 * the pane on the way and go false exactly where the inner lookahead works. When the two scrollers
 * are the same element there is one observer and this costs nothing.
 * THE CONJUNCTION IS NOT ENOUGH ON ITS OWN. A stale inner root is too permissive and the outer gate
 * only covers that while the pane is out of view: scroll the expanded pane partly on screen and the
 * stale inner root decides alone and reports the WHOLE trace (10 of 10 in both engines where the
 * right answer is about 3). So the inner root is re-resolved when the pane stops scrolling, which
 * collapses the fence back to the single-gate case.
 * Watched with a ResizeObserver on the pane rather than by re-resolving every frame, and only for
 * fences that have a nested scroller: the callback reads one `overflow-y` on one element.
 * Still one-way. The extra gate and the rebind can only withhold a latch.
 */
const outermostScrollerOf = (node: HTMLElement): HTMLElement | null => {
  let found: HTMLElement | null = null;
  for (let el = node.parentElement; el !== null; el = el.parentElement) {
    if (isScrollable(el)) found = el;
  }
  return found;
};

/** Is `node` inside `scroller`'s box grown by one of its own heights, the observer's margin? */
const inBand = (node: HTMLElement, scroller: HTMLElement | null): boolean => {
  const bounds = scroller?.getBoundingClientRect();
  const top = bounds ? bounds.top : 0;
  const height = bounds ? bounds.height : window.innerHeight;
  const rect = node.getBoundingClientRect();
  return rect.bottom > top - height && rect.top < top + height * 2;
};

/**
 * @param enabled  false on the shipped default, where this hook must cost nothing at all: no state
 *                 is written, no observer is built and no layout is read.
 * @param streaming  the fence is still being written. It is highlighted while it streams AND it
 *                 latches, so that finishing cannot take the highlighting back.
 * @param language  used only to load one grammar per language rather than one per fence; `null`
 *                 warms plain text.
 * @param chars  this fence's source length, read only by `warmGrammars` to keep a SPECULATIVE warm
 *                 inside `MAX_HIGHLIGHT_CHARS`. A latch is demanded work and is not capped.
 * @param warm  drive the highlighter over this fence: `true` for tokens, `false` for the grammar
 *                 alone. Held in a ref, not an effect dependency, so an unmemoized caller cannot
 *                 rebuild every observer in the thread on every render.
 */
const FENCE_HEAD_CHARS = 64;
const JUST_STREAMED_MS = 2000;

let lastStreamingFence: {
  language: string | null;
  head: string;
  chars: number;
  at: number;
} | null = null;

export const noteStreamingFence = (
  language: string | null,
  source: string,
): void => {
  lastStreamingFence = {
    language,
    head: source.slice(0, FENCE_HEAD_CHARS),
    chars: trimmedLength(source),
    at: performance.now(),
  };
};

const justStreamed = (
  language: string | null,
  head: string,
  chars: number,
): boolean =>
  lastStreamingFence !== null &&
  lastStreamingFence.language === language &&
  lastStreamingFence.head === head &&
  chars >= lastStreamingFence.chars &&
  chars - lastStreamingFence.chars < 512 &&
  performance.now() - lastStreamingFence.at < JUST_STREAMED_MS;

export function useFenceReached(
  host: RefObject<HTMLElement | null>,
  enabled: boolean,
  streaming: boolean,
  language: string | null,
  chars: number,
  source: string,
  warm: (tokens: boolean) => void,
  speculate: (settle: (seeded: boolean) => void) => (() => void) | null,
): boolean {
  const speculateRef = useRef(speculate);
  useEffect(() => {
    speculateRef.current = speculate;
  }, [speculate]);
  const [latched, setLatched] = useState(
    () =>
      streaming ||
      justStreamed(language, source.slice(0, FENCE_HEAD_CHARS), chars),
  );
  // Bumped when the resolved scrolling ancestor stops being one, which rebuilds the gates below
  // against the element that clips this fence now. Never read for anything else.
  const [generation, setGeneration] = useState(0);
  const reached = !enabled || !CAN_OBSERVE || streaming || latched;
  const warmRef = useRef(warm);
  useEffect(() => {
    warmRef.current = warm;
  }, [warm]);

  /*
     * A COMPLETING STREAM MUST NOT DOWNGRADE. `streaming` goes true -> FALSE when streamdown
     * recognises the closing delimiter, so deriving `reached` from it alone hands a fence that was
     * highlighted all through its stream back the plain shell the moment it finishes: the reverse
     * edge this design exists to remove. In a layout effect so it is never painted.
     */
  useLayoutEffect(() => {
    if (!enabled || latched || !streaming) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLatched(true);
  }, [enabled, latched, streaming]);

  /*
     * THE FIRST FRAME, which the observer cannot cover.
     * An IntersectionObserver delivers its first record asynchronously, one or more frames after
     * `observe()`, so a fence ALREADY on screen at mount shows plain code inside the viewport for 2
     * to 3 frames. `useLayoutEffect` runs after mutation and BEFORE paint, so it latches and
     * re-renders within the same frame. One `getBoundingClientRect` per unreached fence, all inside a
     * single commit with no DOM mutation between, so layout is forced once for the whole thread.
     * RE-RUN ON A REBIND. `generation` is a dependency because the ResizeObserver below bumps it when
     * the reasoning pane stops scrolling, and expanding that pane can bring a fence inside the outer
     * viewport for the first time; without it the replacement observer is built in a passive effect
     * and delivers asynchronously, so the shell is PAINTED.
     * Still one-way: this can only ever latch true.
     */
  useLayoutEffect(() => {
    if (reached) return;
    if (deferReaches()) return;
    const node = host.current;
    if (!node) return;
    // The same two questions the observers below ask, of the same two elements: is the FENCE in
    // the window the reader is looking through, and is that WINDOW itself on screen.
    const near = scrollerOf(node);
    const outer = outermostScrollerOf(node);
    if (inBand(node, near) && (near === outer || inBand(near as HTMLElement, outer))) {
      // The cascading render this warns about is the POINT: it keeps the plain shell off the screen. It
      // happens at most once per fence, only for the one or two already on screen at mount, and the
      // alternative is 2 to 3 painted frames of unhighlighted code.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setLatched(true);
    }
  }, [reached, host, generation]);

  // The one-way edge. Once `reached` is true this effect re-runs, takes the early return, and never
  // observes anything again, so a fence that has been read carries no residual per-scroll cost.
  useEffect(() => {
    if (reached) return;
    const node = host.current;
    if (!node) return;
    // ROOTED AT THE THREAD'S SCROLLER, not at the document. The chat scrolls inside a nested overflow
    // container; with `root` unset the root is the document viewport and `rootMargin` expands THAT
    // rectangle, while the intersection is still clipped by the scroller's edges, which no margin can
    // widen. See `outermostScrollerOf`: when a nested scroller is in the way, the outermost one is
    // observed as well and the latch needs both. One observer, not two, whenever they agree.
    const near = scrollerOf(node);
    const outer = outermostScrollerOf(node);
    // WHAT EACH GATE WATCHES, which is not the same element. Observing the FENCE against the outer root
    // clips it at the pane on the way, so the outer gate goes false exactly where the inner lookahead
    // should be working (2 of 10 against 4). Observing the PANE against the outer root asks the
    // question the outer gate is for, and the inner gate keeps its lookahead untouched.
    const gates: [Element, Element | null][] = near === outer
      ? [[node, near]]
      : [[node, near], [near as HTMLElement, outer]];
    const seen = gates.map(() => false);
    const observers = gates.map(([, root], i) => new IntersectionObserver(
      (entries) => {
        seen[i] = entries.some((entry) => entry.isIntersecting);
        if (!seen.every(Boolean)) return;
        if (deferReaches()) {
          queueReach(registered);
          return;
        }
        for (const each of observers) each.disconnect();
        setLatched(true);
      },
      { root, rootMargin: REACH_MARGIN },
    ));
    gates.forEach(([target], i) => observers[i].observe(target));

    // The same two elements, so a jump and a print can ask the same questions without a render or
    // another ancestor walk. Rebuilt with the observers, so a rebind cannot leave a stale root.
    const registered: FenceGate = {
      node,
      near,
      outer,
      language,
      chars,
      warm: (tokens) => warmRef.current(tokens),
      speculate: (settle) => speculateRef.current(settle),
      latch: () => setLatched(true),
      // Reuses `generation`: this fence has just latched, so every effect keyed on it
      // early-returns and the bump costs one render. See `latchNow` for why React needs the work.
      poke: () => setGeneration((n) => n + 1),
    };
    unreached.add(registered);
    watchScrolling();
    scheduleGrammarWarm();
    fenceSpeculator.add(registered);

    // `reasoning.tsx` drops `max-h-64` when a stream ends and keeps `overflow-y-auto`, so the pane
    // stops being a scroller and its box becomes the whole trace. Watch for that and rebuild.
    let resize: ResizeObserver | undefined;
    if (near !== null && near !== outer && typeof ResizeObserver !== "undefined") {
      resize = new ResizeObserver(() => {
        if (!isScrollable(near)) setGeneration((n) => n + 1);
      });
      resize.observe(near);
    }

    return () => {
      for (const observer of observers) observer.disconnect();
      resize?.disconnect();
      pendingReaches.get(registered)?.();
      pendingReaches.delete(registered);
      fenceSpeculator.remove(registered);
      unreached.delete(registered);
      unwatchScrolling();
    };
  }, [reached, host, generation]);

  return reached;
}

export const DeferredFenceShell = memo(FenceShell);

export const PlainFenceShell = memo(function PlainFenceShell({
  language,
  source,
}: {
  language: string | null;
  source: string;
}) {
  return <FenceShell language={language} plain source={source} />;
});

/*
 * The highlighted body, rendered here rather than by streamdown.
 *
 * Streamdown's `CodeBlockBody` maps the whole token array every render and memoizes on
 * `prev.result === next.result`, reference equality, while `code-plugin.ts` returns a fresh object
 * every call, so the memo never hits and every line and token rebuilds each frame. The spans are
 * the cost, not the tokenizer: #10779 kept calling the highlighter, stopped rendering the tokens
 * and still doubled the frame rate, and `scripts/coal-span-census.mjs` shows Shiki's tokens are
 * already maximally coalesced, so merging cannot help.
 *
 * So: one memoized component per line (a committed line's identity is stable, so `memo` bails),
 * plus a line window past `WINDOW_CAP_LINES` (see `code-fence-window.ts`).
 *
 * The DOM is streamdown's element for element and class for class, because
 * `playwright_code_block_flicker.py` reads computed styles off this subtree and a perf change that
 * also moved the rendering could not be attributed.
 */

export type FenceTokens = HighlightResult;
type TokenLine = HighlightResult["tokens"][number];
type FenceToken = TokenLine[number];

/* Streamdown's own class lists, copied verbatim. The `bg-[var(--sdm-bg,inherit]` spellings are
 * unbalanced in streamdown 2.5's build and therefore generate no rule at all; they are reproduced
 * as they are because the goal is the same DOM, not a tidier one. */
/*
 * The one deliberate difference from streamdown's line class: it writes a raw pixel text utility,
 * which `tests/studio/test_ui_font_scale_contract.py` forbids anywhere in frontend source, comments
 * included, because a fixed size ignores the UI font preference. `text-ui-13` scales with it.
 * Moot in the thread either way: `index.css` gives that pseudo-element `display: none`.
 */
const LINE_CLASS =
  "block before:content-[counter(line)] before:inline-block before:[counter-increment:line] before:w-6 before:mr-4 before:text-ui-13 before:text-right before:text-muted-foreground/50 before:font-mono before:select-none";
const CODE_CLASS = "[counter-increment:line_0] [counter-reset:line]";
const PRE_CLASS =
  "bg-[var(--sdm-bg,inherit] dark:bg-[var(--shiki-dark-bg,var(--sdm-bg,inherit)]";
const TOKEN_CLASS =
  "text-[var(--sdm-c,inherit)] dark:text-[var(--shiki-dark,var(--sdm-c,inherit))]";
const TOKEN_BG_CLASS =
  "bg-[var(--sdm-tbg)] dark:bg-[var(--shiki-dark-bg,var(--sdm-tbg))]";

/*
 * THE `language-x` CLASS, WHICH IS NOT DECORATION. `remark-rehype` puts `language-<info>` on the
 * `<code>` of a fenced block, streamdown passes that `className` straight through to BOTH the body
 * div and the `<pre>`, and dropping it was measured as the only DOM difference between this body
 * and the one it replaces. Nothing in the tree selects on it today, which is exactly why it would
 * have gone unnoticed: it is a published rendering contract, user stylesheets and future probes
 * reach for it, and `math-block-marker.ts` already relies on the same `language-` convention one
 * layer up. The token is the FIRST word of the info string, so ```python startLine=10 is
 * `language-python`, which is what `languageToken` already holds.
 */
const joinClasses = (...parts: (string | null)[]): string =>
  parts.filter((part): part is string => Boolean(part)).join(" ");

/* `rootStyle` arrives as a CSS declaration string. Parsed the way streamdown parses it, splitting
 * on the FIRST colon only, so a `url(data:...)` value survives. */
const parseDeclarations = (text: string): Record<string, string> => {
  const style: Record<string, string> = {};
  for (const declaration of text.split(";")) {
    const colon = declaration.indexOf(":");
    if (colon <= 0) continue;
    const property = declaration.slice(0, colon).trim();
    const value = declaration.slice(colon + 1).trim();
    if (property && value) style[property] = value;
  }
  return style;
};

const tokenStyle = (
  token: FenceToken,
): { style: Record<string, string>; hasBackground: boolean } => {
  const style: Record<string, string> = {};
  let hasBackground = Boolean(token.bgColor);
  if (token.color) style["--sdm-c"] = token.color;
  if (token.bgColor) style["--sdm-tbg"] = token.bgColor;
  if (token.htmlStyle) {
    for (const [property, value] of Object.entries(token.htmlStyle)) {
      if (property === "color") style["--sdm-c"] = value;
      else if (property === "background-color") {
        style["--sdm-tbg"] = value;
        hasBackground = true;
      } else style[property] = value;
    }
  }
  return { style, hasBackground };
};

/**
 * One line of a fence.
 *
 * Memoized on the default shallow comparison, which is all it needs: `line` is the array
 * `code-plugin.ts` committed and never touches again, and `windowed` only changes when the reader
 * moves far enough for the window to move. A fence growing by a character re-renders its last line
 * and nothing else.
 */
const FenceLine = memo(function FenceLine({
  line,
  windowed,
  highlighted = Number.POSITIVE_INFINITY,
}: {
  line: TokenLine;
  windowed: boolean;
  highlighted?: number;
}) {
  if (isBlankLine(line)) {
    return <span className={LINE_CLASS}>{"\n"}</span>;
  }
  if (!windowed) {
    // One text node for the whole line. Same characters, same block box, same height: the only
    // thing this line has given up is its colour, and it is off screen.
    return (
      <span className={LINE_CLASS}>
        {plainLineText(line)}
      </span>
    );
  }
  const foreground = fenceLineForeground(line);
  return (
    <span
      className={foreground ? `${LINE_CLASS} ${TOKEN_CLASS}` : LINE_CLASS}
      style={
        foreground
          ? { "--sdm-c": foreground.light, "--shiki-dark": foreground.dark } as CSSProperties
          : undefined
      }
    >
      {(highlighted >= line.length ? line : line.slice(0, highlighted)).map((token, index) => {
        if (foreground?.inherited[index]) return token.content;
        const { style, hasBackground } = tokenStyle(token);
        return (
          <span
            className={
              hasBackground ? `${TOKEN_CLASS} ${TOKEN_BG_CLASS}` : TOKEN_CLASS
            }
            key={index}
            style={style}
            {...token.htmlAttrs}
          >
            {token.content}
          </span>
        );
      })}{highlighted < line.length ? line.slice(highlighted).map((token) => token.content).join("") : null}
    </span>
  );
});

const FenceTokenChunk = memo(function FenceTokenChunk({
  tokens,
  foreground,
  start,
  highlighted,
}: {
  tokens: readonly FenceToken[];
  foreground: ReturnType<typeof fenceLineForeground>;
  start: number;
  highlighted: number;
}) {
  return <>{(highlighted >= tokens.length ? tokens : tokens.slice(0, highlighted)).map((token, index) => {
    if (foreground?.inherited[start + index]) return token.content;
    const { style, hasBackground } = tokenStyle(token);
    return <span
      className={hasBackground ? `${TOKEN_CLASS} ${TOKEN_BG_CLASS}` : TOKEN_CLASS}
      key={index}
      style={style}
      {...token.htmlAttrs}
    >{token.content}</span>;
  })}{highlighted < tokens.length ? tokens.slice(highlighted).map((token) => token.content).join("") : null}</>;
});

const DenseFenceLine = memo(function DenseFenceLine({
  line,
  highlighted,
}: {
  line: TokenLine;
  highlighted: number;
}) {
  const chunks = useMemo(() => updateFenceChunks(null, line), [line]);
  const foreground = useMemo(() => fenceLineForeground(line), [line]);
  return <span
    className={foreground ? `${LINE_CLASS} ${TOKEN_CLASS}` : LINE_CLASS}
    style={foreground
      ? { "--sdm-c": foreground.light, "--shiki-dark": foreground.dark } as CSSProperties
      : undefined}
  >{chunks.map((tokens, index) => <FenceTokenChunk
    key={index}
    tokens={tokens}
    foreground={foreground}
    start={index * FENCE_CHUNK_LINES}
    highlighted={Math.max(0, Math.min(tokens.length, highlighted - index * FENCE_CHUNK_LINES))}
  />)}</span>;
});

const FenceLineChunk = memo(function FenceLineChunk({
  lines,
  limits,
  dense,
}: {
  lines: readonly TokenLine[];
  limits: readonly number[] | null;
  dense: boolean;
}) {
  return <>{lines.map((line, index) => {
    const count = limits === null ? Number.POSITIVE_INFINITY : limits[index] ?? 0;
    return dense && line.length > HIGHLIGHT_BATCH_TOKENS
      ? <DenseFenceLine key={index} line={line} highlighted={count} />
      : <FenceLine key={index} line={line} windowed={count > 0}
          highlighted={count >= line.length ? Number.POSITIVE_INFINITY : count} />;
  })}</>;
});

const FenceLines = memo(function FenceLines({
  tokens,
  highlighted,
  staged = false,
}: {
  tokens: TokenLine[];
  highlighted: HighlightProgress | null;
  staged?: boolean;
}) {
  const [dense] = useState(staged);
  const [previous, setPrevious] = useState<{
    tokens: TokenLine[];
    chunks: FenceLineChunks<TokenLine>;
  }>(() => ({ tokens, chunks: updateFenceChunks(null, tokens) }));
  const current = previous.tokens === tokens
    ? previous
    : { tokens, chunks: updateFenceChunks(previous.chunks, tokens) };
  if (current !== previous) setPrevious(current);
  return <>{current.chunks.map((lines, index) => <FenceLineChunk
    key={index}
    lines={lines}
    limits={highlighted === null ? null : highlighted.counts[index] ?? EMPTY_HIGHLIGHT_COUNTS}
    dense={dense}
  />)}</>;
});

// One scroll listener and one frame for every windowed fence: each measurement reads layout, so a
// listener per fence makes scrolling cost more than the rendering the window avoids. Same shape
// `watchScrolling` uses for the reach latch, coalesced into a frame.
const windowedFences = new Set<() => void>();
let windowFrame = 0;
let windowWatched = false;
let findBarObserver: MutationObserver | null = null;
let frameRects: Map<Element, DOMRect> | null = null;
let frameFlying: Map<Element | null, boolean> | null = null;
let frameSettle = false;
let pointerHeld: Node | null = null;
let heldLastFrame = false;
let frameSelection: { read: boolean; value: Selection | null } | null = null;
const scrollerMotion = new WeakMap<Element, { top: number; at: number }>();
const windowMotion = { top: 0, at: 0 };
const FLYING_VIEWPORTS_PER_SECOND = 10;

const rectDuringFrame = (element: Element): DOMRect => {
  if (frameRects === null) return element.getBoundingClientRect();
  const known = frameRects.get(element);
  if (known !== undefined) return known;
  const rect = element.getBoundingClientRect();
  frameRects.set(element, rect);
  return rect;
};

const liveSelection = (): Selection | null => {
  const selection = document.getSelection();
  return selection !== null && selection.rangeCount > 0 && !selection.isCollapsed
    ? selection
    : null;
};

const selectionDuringFrame = (): Selection | null => {
  if (frameSelection === null) return liveSelection();
  if (!frameSelection.read) {
    frameSelection.read = true;
    frameSelection.value = liveSelection();
  }
  return frameSelection.value;
};

const draggingDuringFrame = (body: HTMLElement): boolean => {
  const held = pointerHeld !== null && body.contains(pointerHeld);
  if (held) heldLastFrame = true;
  return held;
};

const selectionBoundaryLines = (codeNode: HTMLElement, body: HTMLElement): number[] => {
  const selection = selectionDuringFrame();
  const lines: number[] = [];
  for (let index = 0; selection !== null && index < selection.rangeCount; index += 1) {
    const range = selection.getRangeAt(index);
    for (const boundary of [range.startContainer, range.endContainer]) {
      if (!body.contains(boundary)) continue;
      let line: Node | null = boundary;
      while (line !== null && line.parentNode !== codeNode) line = line.parentNode;
      if (line === null) continue;
      const at = Array.prototype.indexOf.call(codeNode.children, line);
      if (at >= 0) lines.push(at);
    }
  }
  if (lines.length > 0) heldLastFrame = true;
  return lines;
};

const flyingDuringFrame = (
  scroller: HTMLElement | null,
  viewportHeight: number,
): boolean => {
  if (frameFlying === null) return false;
  const known = frameFlying.get(scroller);
  if (known !== undefined) return known;
  const top = scroller === null ? window.scrollY : scroller.scrollTop;
  const at = performance.now();
  const before = scroller === null ? windowMotion : scrollerMotion.get(scroller);
  const elapsed = before === undefined ? 0 : at - before.at;
  const flying =
    before !== undefined
    && elapsed > 0
    && viewportHeight > 0
    && (Math.abs(top - before.top) / elapsed) * 1000
      > viewportHeight * FLYING_VIEWPORTS_PER_SECOND;
  if (scroller === null) {
    windowMotion.top = top;
    windowMotion.at = at;
  } else {
    scrollerMotion.set(scroller, { top, at });
  }
  frameFlying.set(scroller, flying);
  if (flying) frameSettle = true;
  return flying;
};

/*
 * A print puts every line on the page.
 * `upgradeEverythingForPrint` makes the same argument for a deferred fence.
 *
 * This one REVERTS where the latch does not: the tokens are already in `fence.lines`, so
 * re-windowing costs element creation only, and not reverting would let one Ctrl+P un-window every
 * huge fence for the life of the tab. Both doors, as above: `beforeprint` for Ctrl+P, the media
 * query for `page.pdf()` and devtools emulation, which do not fire it.
 */
let printing = false;

type PrintTokenizer = { chars: number; tokenizeNow: () => boolean };
const awaitingWorker = new Set<PrintTokenizer>();
const PRINT_TOKENIZE_CHARS = 10 * MAX_HIGHLIGHT_CHARS;

export const awaitWorker = (
  chars: number,
  tokenizeNow: () => boolean,
): (() => void) => {
  const entry: PrintTokenizer = { chars, tokenizeNow };
  awaitingWorker.add(entry);
  return () => {
    awaitingWorker.delete(entry);
  };
};

const tokenizeForPrint = (): void => {
  if (awaitingWorker.size === 0) return;
  const pending = [...awaitingWorker];
  const order = planPrintTokenization(
    pending.map((entry) => entry.chars),
    PRINT_TOKENIZE_CHARS,
  );
  flushSync(() => {
    for (const index of order) pending[index].tokenizeNow();
  });
};

const remeasureWindows = (): void => {
  windowFrame = 0;
  forgetScrollable();
  frameRects = new Map();
  frameFlying = new Map();
  frameSelection = { read: false, value: null };
  frameSettle = false;
  heldLastFrame = false;
  try {
    for (const measure of windowedFences) measure();
  } finally {
    frameRects = null;
    frameFlying = null;
    frameSelection = null;
  }
  if (frameSettle && windowedFences.size > 0 && !printing) {
    windowFrame = requestAnimationFrame(remeasureWindows);
  }
};

export const fencePrinting = (): boolean => printing;

const FIND_BAR_FLAG = "data-find-bar-open";

const findBarOpen = (): boolean =>
  typeof document !== "undefined"
  && document.documentElement.hasAttribute(FIND_BAR_FLAG);

export const fenceNeedsQuietWork = (
  language: string | null,
  source: string,
  streaming = false,
): boolean =>
  !streaming && typeof window !== "undefined" && deferReaches()
  && !justStreamed(language, source.slice(0, FENCE_HEAD_CHARS), trimmedLength(source));

let highlightWatching = false;
let highlightFindObserver: MutationObserver | null = null;
let highlightResizedAt = Number.NEGATIVE_INFINITY;

const highlightResize = (): void => {
  highlightResizedAt = performance.now();
};

const highlightQuietIn = (): number =>
  Math.max(inputQuietIn(), 300 - (performance.now() - highlightResizedAt));

const highlightQueue = createFenceHighlightQueue({
  schedule: (run) => {
    let frame = 0;
    let cancelIdle: (() => void) | undefined;
    const timer = setTimeout(() => {
      frame = requestAnimationFrame(() => {
        cancelIdle = scheduleQuietIdleTask(run, 100);
      });
    }, Math.max(0, highlightQuietIn(), inputPointerHeld() || liveSelection() ? 80 : 0));
    return () => {
      clearTimeout(timer);
      cancelAnimationFrame(frame);
      cancelIdle?.();
    };
  },
  blocked: () => inputPointerHeld() || highlightQuietIn() > 0 || liveSelection() !== null,
  commit: (run) => flushSync(run),
});

const watchHighlighting = (): void => {
  if (highlightWatching) return;
  highlightWatching = true;
  window.addEventListener("resize", highlightResize, { passive: true });
  highlightFindObserver = new MutationObserver(() => {
    if (findBarOpen()) highlightQueue.flush();
  });
  highlightFindObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: [FIND_BAR_FLAG],
  });
};

let highlightTeardownPending = false;
const unwatchHighlighting = (): void => {
  if (!highlightWatching || highlightTeardownPending) return;
  highlightTeardownPending = true;
  queueMicrotask(() => {
    highlightTeardownPending = false;
    if (!highlightWatching || highlightQueue.pending() > 0) return;
    highlightWatching = false;
    window.removeEventListener("resize", highlightResize);
    highlightFindObserver?.disconnect();
    highlightFindObserver = null;
  });
};

export function queueFenceWork(work: () => void): () => void {
  if (typeof window === "undefined" || printing || findBarOpen()) {
    work();
    return () => {};
  }
  watchHighlighting();
  let cancel = () => {};
  const finish = (): void => {
    cancel();
    work();
  };
  const remove = highlightQueue.add(() => {
    finish();
    return true;
  }, finish);
  cancel = () => {
    remove();
    unwatchHighlighting();
  };
  return cancel;
}

const visibleHighlightLine = (code: HTMLElement | null, frame: HTMLElement | null): number => {
  const first = code?.firstElementChild;
  if (!first || !frame) return 0;
  const line = first.getBoundingClientRect();
  if (line.height <= 0) return 0;
  let top = 0;
  let bottom = window.innerHeight;
  for (let ancestor = frame.parentElement; ancestor !== null; ancestor = ancestor.parentElement) {
    if (!isScrollable(ancestor)) continue;
    const rect = ancestor.getBoundingClientRect();
    top = Math.max(top, rect.top + ancestor.clientTop);
    bottom = Math.min(bottom, rect.top + ancestor.clientTop + ancestor.clientHeight);
  }
  if (top >= bottom || line.top >= bottom) return 0;
  return Math.max(0, Math.floor((top - line.top) / line.height));
};

function useFenceHighlightLimit(
  tokens: FenceTokens["tokens"] | null,
  isIncomplete: boolean | undefined,
  windowing: boolean,
  language: string | null,
  source: string,
  code: RefObject<HTMLElement | null>,
  frame: RefObject<HTMLElement | null>,
): { progress: HighlightProgress | null; staged: boolean } {
  const eligible = !(typeof window === "undefined" || windowing || isIncomplete
    || fenceMode() !== "defer"
    || justStreamed(language, source.slice(0, FENCE_HEAD_CHARS), trimmedLength(source)));
  const [eager, setEager] = useState(!eligible);
  if (!eligible && !eager) setEager(true);
  const staged = eligible && !eager;
  const large = useMemo(
    () => staged && tokens !== null && needsHighlightBatches(tokens, WINDOW_CAP_LINES),
    [staged, tokens],
  );
  const immediate = !staged || printing || findBarOpen() || (tokens !== null && !large);
  const [previous, setPrevious] = useState<{
    tokens: typeof tokens;
    progress: HighlightProgress | null;
  }>(() => ({ tokens, progress: immediate ? null : tokens === null
    ? EMPTY_HIGHLIGHT_PROGRESS : initialHighlightProgress(tokens) }));
  let current = previous;
  if (staged && previous.tokens !== tokens) {
    current = { tokens, progress: immediate || tokens === null
      ? null : retainHighlightProgress(previous.tokens, tokens, previous.progress) };
  } else if (immediate && previous.progress !== null) {
    current = { tokens, progress: null };
  }
  if (current !== previous) setPrevious(current);
  const progress = immediate ? null : current.progress;
  useLayoutEffect(() => {
    if (progress === null || tokens === null) return;
    watchHighlighting();
    const cancel = highlightQueue.add(() => {
      const visible = progress.located ? 0 : visibleHighlightLine(code.current, frame.current);
      setPrevious({ tokens, progress: advanceHighlightProgress(tokens, progress, visible) });
      return true;
    }, () => setPrevious({ tokens, progress: null }));
    return () => {
      cancel();
      unwatchHighlighting();
    };
  }, [tokens, progress, code, frame]);
  return { progress, staged };
}

/*
 * Synchronous, and inside `flushSync`, for the same reason `latchNow` is: a normally scheduled
 * update lands after the next paint, and there is no next paint before the print snapshot.
 */
const setPrinting = (value: boolean): void => {
  // BEFORE the window check: a print that starts while a fence still shows its shell has no window
  // to remeasure, but the state must still be recorded or the tokens arrive mid-preview and window
  // it. Only the flush is skipped when there is nothing to remeasure.
  if (printing === value) return;
  printing = value;
  if (windowFrame !== 0) {
    cancelAnimationFrame(windowFrame);
    windowFrame = 0;
  }
  if (value) highlightQueue.flush();
  if (value) tokenizeForPrint();
  if (value) highlightQueue.flush();
  if (windowedFences.size === 0) return;
  flushSync(remeasureWindows);
};

const scheduleRemeasure = (): void => {
  if (windowFrame !== 0 || windowedFences.size === 0) return;
  windowFrame = requestAnimationFrame(remeasureWindows);
};

const onPointerDown = (event: PointerEvent): void => {
  if (event.pointerType === "touch" || event.button !== 0) return;
  pointerHeld = event.target instanceof Node ? event.target : null;
};

const releasePointer = (): void => {
  if (pointerHeld === null) return;
  pointerHeld = null;
  if (heldLastFrame) scheduleRemeasure();
};

const onPointerMove = (event: PointerEvent): void => {
  if (pointerHeld !== null && event.buttons === 0) releasePointer();
};

const onSelectionChange = (): void => {
  if (heldLastFrame) scheduleRemeasure();
};

const watchWindows = (): void => {
  if (windowWatched || typeof document === "undefined") return;
  windowWatched = true;
  // Capturing, because scroll does not bubble but does capture, so this one listener sees the
  // thread scroller AND the nested reasoning pane.
  document.addEventListener("scroll", scheduleRemeasure, {
    capture: true,
    passive: true,
  });
  window.addEventListener("resize", scheduleRemeasure, { passive: true });
  document.addEventListener("pointerdown", onPointerDown, {
    capture: true,
    passive: true,
  });
  window.addEventListener("pointerup", releasePointer, {
    capture: true,
    passive: true,
  });
  window.addEventListener("pointercancel", releasePointer, {
    capture: true,
    passive: true,
  });
  window.addEventListener("dragend", releasePointer, {
    capture: true,
    passive: true,
  });
  window.addEventListener("drop", releasePointer, {
    capture: true,
    passive: true,
  });
  window.addEventListener("pointermove", onPointerMove, {
    capture: true,
    passive: true,
  });
  window.addEventListener("blur", releasePointer);
  document.addEventListener("selectionchange", onSelectionChange, {
    passive: true,
  });
  findBarObserver = new MutationObserver(scheduleRemeasure);
  findBarObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: [FIND_BAR_FLAG],
  });
};

const unwatchWindows = (): void => {
  if (!windowWatched || windowedFences.size > 0 || typeof document === "undefined") {
    return;
  }
  windowWatched = false;
  document.removeEventListener("scroll", scheduleRemeasure, { capture: true });
  window.removeEventListener("resize", scheduleRemeasure);
  document.removeEventListener("pointerdown", onPointerDown, { capture: true });
  window.removeEventListener("pointerup", releasePointer, { capture: true });
  window.removeEventListener("pointercancel", releasePointer, { capture: true });
  window.removeEventListener("dragend", releasePointer, { capture: true });
  window.removeEventListener("drop", releasePointer, { capture: true });
  window.removeEventListener("pointermove", onPointerMove, { capture: true });
  window.removeEventListener("blur", releasePointer);
  document.removeEventListener("selectionchange", onSelectionChange);
  findBarObserver?.disconnect();
  findBarObserver = null;
  pointerHeld = null;
  heldLastFrame = false;
  if (windowFrame !== 0) {
    cancelAnimationFrame(windowFrame);
    windowFrame = 0;
  }
};

/**
 * Which lines of this fence carry token spans, or `null` for all of them.
 *
 * The geometry is read here and the decision is made in `code-fence-window.ts`, which is a
 * JSX-free module so that a test can RUN the arithmetic rather than regex this file.
 */
type LineWindowState =
  | { measured: false }
  | { measured: true; window: LineWindow | null; pins: LinePins | null };

const UNMEASURED: LineWindowState = { measured: false };

type FenceMetrics = {
  lineHeight: number;
  contentInset: number;
  insets: number;
};

type FenceGeometry = FenceMetrics & {
  scroller: HTMLElement | null;
};

const linePitches = new Map<string, number>();
const PITCH_PROBE_LINES = 64;
let intrinsicScale = 0;
if (typeof window !== "undefined") {
  window.addEventListener("resize", () => {
    intrinsicScale = 0;
  });
}
const intrinsicSizeSupported =
  typeof CSS !== "undefined"
  && typeof CSS.supports === "function"
  && CSS.supports("contain-intrinsic-size", "0 1000px");

const measureIntrinsicScale = (beside: HTMLElement): number => {
  if (!intrinsicSizeSupported) return 1;
  if (intrinsicScale > 0) return intrinsicScale;
  const probe = document.createElement("div");
  probe.setAttribute("aria-hidden", "true");
  probe.style.cssText =
    "position:absolute;visibility:hidden;pointer-events:none;top:0;left:0;width:0;contain:size;contain-intrinsic-size:0 1000px";
  beside.before(probe);
  const height = probe.getBoundingClientRect().height;
  probe.remove();
  if (!(height > 0)) return 1;
  intrinsicScale = Math.round(1000000 / height) / 1000;
  return intrinsicScale;
};

const measureLinePitch = (beside: HTMLElement): number => {
  const parent = beside.parentElement;
  if (!parent) return 0;
  const context = getComputedStyle(parent);
  const key = [
    context.fontSize,
    context.lineHeight,
    getComputedStyle(document.documentElement).getPropertyValue("--custom-code-font-size"),
    window.devicePixelRatio,
    measureIntrinsicScale(beside),
  ].join("|");
  const known = linePitches.get(key);
  if (known !== undefined) return known;
  const probe = document.createElement("pre");
  probe.setAttribute("aria-hidden", "true");
  probe.style.cssText = "position:absolute;visibility:hidden;pointer-events:none;top:0;left:0";
  const code = document.createElement("code");
  const lines: HTMLElement[] = [];
  for (let index = 0; index < PITCH_PROBE_LINES; index += 1) {
    const line = document.createElement("span");
    line.className = LINE_CLASS;
    line.textContent = "x";
    lines.push(line);
  }
  code.append(...lines);
  probe.append(code);
  beside.before(probe);
  const pitch =
    (lines[PITCH_PROBE_LINES - 1].getBoundingClientRect().top - lines[0].getBoundingClientRect().top)
    / (PITCH_PROBE_LINES - 1);
  probe.remove();
  if (pitch > 0) linePitches.set(key, pitch);
  return pitch;
};

const readFenceMetrics = (
  node: HTMLElement,
  surface: HTMLElement,
  region: HTMLElement,
  lineCount: number,
  previous: FenceMetrics | null,
): FenceMetrics => {
  const style = getComputedStyle(surface);
  const pitch = measureLinePitch(region);
  const declared = Number.parseFloat(style.lineHeight);
  const lineHeight =
    pitch > 0
      ? pitch
      : declared > 0
        ? style.lineHeight.endsWith("px")
          ? declared
          : declared * (Number.parseFloat(style.fontSize) || 0)
        : previous?.lineHeight
          ?? (lineCount > 0 ? node.getBoundingClientRect().height / lineCount : 0);
  return {
    lineHeight,
    contentInset:
      (Number.parseFloat(style.borderTopWidth) || 0)
      + (Number.parseFloat(style.paddingTop) || 0),
    insets: Math.max(0, surface.getBoundingClientRect().height - lineCount * lineHeight),
  };
};

const readFenceGeometry = (
  node: HTMLElement,
  surface: HTMLElement,
  region: HTMLElement,
  outer: HTMLElement,
  lineCount: number,
): FenceGeometry => {
  const scroller = scrollerOf(outer);
  return { scroller, ...readFenceMetrics(node, surface, region, lineCount, null) };
};

const skippedByContainment = (element: HTMLElement): boolean =>
  typeof element.checkVisibility === "function"
  && !element.checkVisibility({ contentVisibilityAuto: true });

const FAR_VIEWPORTS = OVERSCAN_VIEWPORTS + HYSTERESIS_VIEWPORTS + 1;
const RESIZE_SETTLE_MS = 300;

function useLineWindow(
  code: RefObject<HTMLElement | null>,
  surface: RefObject<HTMLElement | null>,
  wrapper: RefObject<HTMLElement | null>,
  frame: RefObject<HTMLElement | null>,
  lineCount: number,
  enabled: boolean,
  deferShifts: boolean,
): { window: LineWindow | null; pins: LinePins | null; measured: boolean } {
  const [state, setState] = useState<LineWindowState>(UNMEASURED);
  const viewportSize = useRef("");
  const resizedAt = useRef(Number.NEGATIVE_INFINITY);
  const current = useRef<LineWindow | null>(null);
  const pinned = useRef<LinePins | null>(null);
  const geometry = useRef<FenceGeometry | null>(null);
  const metricsStale = useRef(false);
  const lines = useRef(lineCount);
  lines.current = lineCount;
  const hasBody = lineCount > 0;
  const overCap = lineCount > WINDOW_CAP_LINES;
  const measure = useRef<() => void>(() => {});

  measure.current = () => {
    const node = code.current;
    const outer = frame.current;
    const body = surface.current;
    const region = wrapper.current;
    if (!node || !outer || !body || !region) return;
    if (lines.current <= WINDOW_CAP_LINES && current.current === null) return;
    if (printing || findBarOpen()) {
      if (current.current === null && pinned.current === null) return;
      current.current = null;
      pinned.current = null;
      setState({ measured: true, window: null, pins: null });
      return;
    }
    let known = geometry.current;
    if (
      known === null
      || (known.scroller !== null && !isScrollable(known.scroller))
    ) {
      if (skippedByContainment(body)) return;
      known = readFenceGeometry(node, body, region, outer, lines.current);
      metricsStale.current = false;
    } else if (metricsStale.current && !skippedByContainment(body)) {
      known = {
        scroller: known.scroller,
        ...readFenceMetrics(node, body, region, lines.current, known),
      };
      metricsStale.current = false;
    }
    geometry.current = known;
    const height =
      Math.round(
        (lines.current * known.lineHeight + known.insets) * measureIntrinsicScale(region) * 1000,
      ) / 1000;
    const declared = `${height}px`;
    if (region.style.getPropertyValue(FENCE_HEIGHT_PROPERTY) !== declared) {
      region.style.setProperty(FENCE_HEIGHT_PROPERTY, declared);
      forgetScrollable();
    }
    const bounds = known.scroller === null ? null : rectDuringFrame(known.scroller);
    const viewportTop = bounds ? bounds.top : 0;
    const viewportHeight = bounds ? bounds.height : window.innerHeight;
    const size = `${bounds ? bounds.width : window.innerWidth}x${viewportHeight}`;
    if (size !== viewportSize.current) resizedAt.current = performance.now();
    viewportSize.current = size;
    const resized = performance.now() - resizedAt.current < RESIZE_SETTLE_MS;
    const box = rectDuringFrame(region);
    const reach = viewportHeight * FAR_VIEWPORTS;
    if (
      current.current !== null
      && (box.bottom < viewportTop - reach || box.top > viewportTop + viewportHeight + reach)
    ) {
      return;
    }
    if (current.current !== null && flyingDuringFrame(known.scroller, viewportHeight)) return;
    if (current.current !== null && draggingDuringFrame(body)) return;
    const next = selectLineWindow({
      lineCount: lines.current,
      lineHeight: known.lineHeight,
      contentTop: box.top + known.contentInset,
      viewportTop,
      viewportHeight,
      previous: current.current,
    });
    const pins = pinBoundaryLines(
      selectionBoundaryLines(node, body),
      current.current,
      pinned.current,
    );
    if (next === current.current && samePins(pins, pinned.current)) return;
    const shifting =
      deferShifts && current.current !== null && frameRects !== null && !resized;
    current.current = next;
    pinned.current = pins;
    const update = { measured: true, window: next, pins };
    if (shifting) startTransition(() => setState(update));
    else setState(update);
  };

  useLayoutEffect(() => {
    if (!enabled) {
      current.current = null;
      geometry.current = null;
      return;
    }
    const run = () => measure.current();
    windowedFences.add(run);
    watchWindows();
    run();
    let resize: ResizeObserver | undefined;
    const box = frame.current;
    if (box && typeof ResizeObserver !== "undefined") {
      resize = new ResizeObserver(() => {
        metricsStale.current = true;
        scheduleRemeasure();
      });
      resize.observe(box);
    }
    return () => {
      resize?.disconnect();
      windowedFences.delete(run);
      unwatchWindows();
    };
  }, [enabled, hasBody, code, surface, wrapper, frame]);

  useLayoutEffect(() => {
    if (!enabled || !overCap || state.measured) return;
    measure.current();
  }, [enabled, overCap, state.measured]);

  if (!enabled) return NO_WINDOW;
  if (state.measured) return state;
  return overCap && !printing && !findBarOpen() ? EMPTY_WINDOW : NO_WINDOW;
}

const NO_WINDOW = { window: null, pins: null, measured: false } as const;
const EMPTY_WINDOW = { window: EMPTY_LINE_WINDOW, pins: null, measured: false } as const;

/**
 * A fence's body, highlighted, with the spans bounded to what is on screen.
 *
 * Falls back to the plain shell whenever there are no tokens to render, which is the window
 * between a fence appearing and its grammar chunk arriving. That is the same markup the deferred
 * shell uses and the same markup streamdown's own unhighlighted fallback uses, so the fence does
 * not change shape when the colours land.
 */
export const FenceBody = memo(function FenceBody({
  isIncomplete,
  language,
  result,
  source,
  windowing,
}: {
  /** Streamdown's unclosed-fence flag, reproduced as `data-incomplete` on the wrapper. */
  isIncomplete: boolean | undefined;
  language: string | null;
  result: FenceTokens | null;
  source: string;
  /** False keeps every line highlighted however long the fence is, which is what main does. */
  windowing: boolean;
}) {
  const code = useRef<HTMLElement | null>(null);
  const surface = useRef<HTMLDivElement | null>(null);
  const wrapper = useRef<HTMLDivElement | null>(null);
  const frame = useRef<HTMLDivElement | null>(null);
  const tokens = result?.tokens ?? null;
  const highlight = useFenceHighlightLimit(tokens, isIncomplete, windowing, language, source, code, frame);
  const { window: lineWindow, pins, measured } = useLineWindow(
    code,
    surface,
    wrapper,
    frame,
    tokens?.length ?? 0,
    windowing,
    !isIncomplete,
  );
  const languageClass = language === null ? null : `language-${language}`;

  const rootStyle = useMemo(() => {
    const style: Record<string, string> = {};
    if (!result) return style;
    if (result.bg) style["--sdm-bg"] = result.bg;
    if (result.fg) style["--sdm-fg"] = result.fg;
    if (result.rootStyle) Object.assign(style, parseDeclarations(result.rootStyle));
    return style;
  }, [result]);

  // No tokens yet, or a result carrying no lines at all. Both fall back to the plain shell, which
  // is one line tall for an empty body where a `<code>` with no children is nothing at all.
  if (!tokens || tokens.length === 0) {
    return <FenceShell language={language} source={source} />;
  }

  return (
    <div
      className="my-4 flex w-full flex-col gap-2 rounded-xl border border-border bg-sidebar p-2"
      data-incomplete={isIncomplete || undefined}
      data-language={language ?? undefined}
      data-streamdown="code-block"
      ref={frame}
      // Streamdown declares both inline. `index.css` then forces `content-visibility: visible`
      // back on for code blocks, because WebKit before Safari 26 cannot find-in-page skipped
      // content, but the declaration is reproduced so the computed cascade is identical to the one
      // `playwright_code_block_flicker.py` reads.
      style={{ containIntrinsicSize: "auto 200px", contentVisibility: "auto" }}
    >
      <div
        className="flex h-8 items-center text-muted-foreground text-xs"
        data-language={language ?? undefined}
        data-streamdown="code-block-header"
      >
        <span className="ml-1 font-mono lowercase">{language}</span>
      </div>
      <div
        data-unsloth-fence-paint={
          !isIncomplete && !windowing && tokens.length > WINDOW_CAP_LINES
            ? "true"
            : undefined
        }
        data-unsloth-fence-windowed={lineWindow === null || !measured ? undefined : "true"}
        ref={wrapper}
      >
        <div
          className={joinClasses(
            languageClass,
            "overflow-x-auto rounded-md border border-border bg-background p-4 text-sm",
          )}
          data-language={language ?? undefined}
          data-streamdown="code-block-body"
          ref={surface}
        >
          <pre className={joinClasses(languageClass, PRE_CLASS)} style={rootStyle}>
            <code className={CODE_CLASS} ref={code}>
              {windowing ? tokens.map((line, index) => (
                <FenceLine
                  key={index}
                  line={line}
                  windowed={lineRendered(lineWindow, pins, index)}
                />
              )) : <FenceLines tokens={tokens} highlighted={highlight.progress} staged={highlight.staged} />}
            </code>
          </pre>
        </div>
      </div>
    </div>
  );
}, sameFenceBodyProps);
