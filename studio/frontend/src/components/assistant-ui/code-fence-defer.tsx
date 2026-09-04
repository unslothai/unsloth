// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import {
  memo,
  type RefObject,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { flushSync } from "react-dom";

import { MAX_HIGHLIGHT_CHARS } from "@/lib/markdown-plugins";
import { type FenceMode, resolveFenceMode } from "./code-fence-mode";
import {
  type EvictCandidate,
  type FenceEvictMode,
  PASS_INTERVAL_MS,
  REACH_BAND,
  type ScrollerBand,
  nextPassDelayMs,
  passIsDue,
  planEviction,
  resolveFenceEvictMode,
  withinBand,
} from "./code-fence-evict";
import { normalizeLanguage } from "./code-plugin";

/*
 * MONOTONIC fence highlighting: a fence is rendered as a plain shell until the
 * first time it comes near the viewport, and from that moment it is highlighted
 * for the rest of its mount. There is no reverse edge.
 *
 * WHY THE DIRECTION MATTERS. The previous attempt at this gated on viewport
 * entry AND on viewport EXIT, so scrolling away from a fence, or collapsing the
 * pane holding it, tore the highlighted subtree down and scrolling back built
 * it again. That makes the gate BIDIRECTIONAL: a user gesture that ought to
 * cost nothing instead schedules a full re-tokenize and a re-mount of every
 * span in the block. It was predicted to save 55% and measured slower.
 *
 * Here the only transition is cheap -> expensive, taken at most once per mount.
 * Scrolling away costs nothing, because there is nothing to undo; the observer
 * has already been disconnected and the component has no effect left running.
 * The worst case is therefore exactly today's cost (every fence looked at), and
 * every fence the reader never reaches is saved outright.
 *
 * WHAT THE SHELL IS. The same text, in the same elements streamdown's own
 * unhighlighted fallback would use, carrying the same `data-streamdown`
 * attributes so every rule in index.css that sizes a code block applies to it
 * unchanged. The text is present in the DOM and selectable, which is the
 * property `content-visibility: auto` does not have: skipped content has to be
 * rendered before it can be selected, and that was measured at +28.8% on
 * `select_all_copy.select_all_ms`. Selection, clipboard and native find-in-page
 * see the same characters here whether the fence has upgraded or not.
 *
 * A STREAMING fence never defers. The block being written is by definition the
 * one the reader is looking at, and deferring it would change what streaming
 * renders rather than what a settled thread costs.
 */

/*
 * How far outside the viewport a fence counts as "reached": one viewport of slack each way, so the
 * upgrade lands a frame or two before the block is on screen and the reader never sees the swap.
 *
 * THE PERCENTAGE RESOLVES AGAINST THE ROOT'S HEIGHT, and the spec says otherwise. Intersection
 * Observer 2.2 says percentages "are resolved relative to the width of the undilated rectangle"
 * for all four sides; no engine does that for top and bottom, and w3c/IntersectionObserver#391 is
 * open on exactly that. Measured on a synthetic scroller, with `0px 0px` and `300px 0px` alongside
 * as controls that must come back 0 and 300:
 *
 *   root 300x800 -> 800 px    root 1600x600 -> 600 px    root 600x1400 -> 1400 px
 *
 * on Chromium, Firefox and WebKit alike, nine rows, every control reproduced. So it is the HEIGHT,
 * which `inBand` and the jump test below both assume. That is today's engines against a spec that
 * reads otherwise, so it is guarded rather than trusted: `pf9462_parity.py` re-measures two
 * geometries every run and fails if the observer's lookahead and the pre-paint band stop agreeing,
 * which is what an engine moving to the spec's reading would look like.
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

// The eviction decision, same shape: a JSX-free `.ts` RUN by `tests/code-fence-evict.test.ts`.
// It ships OFF, so nothing below changes for an install that has not asked for it.
export { type FenceEvictMode, resolveFenceEvictMode } from "./code-fence-evict";

export const fenceEvictMode = (): FenceEvictMode =>
  resolveFenceEvictMode(
    (globalThis as Record<string, unknown>).__UNSLOTH_EVICT_FENCE_HIGHLIGHT__,
    readBuildEvictFlag(),
  );

const readBuildEvictFlag = (): string => {
  try {
    return import.meta.env.VITE_UNSLOTH_EVICT_FENCE_HIGHLIGHT ?? "";
  } catch {
    return "";
  }
};

// Streamdown trims trailing newlines off a fence body before rendering it, so
// the shell has to as well or the two differ by a blank line of height.
//
// The length is split out because `warmGrammars` needs the size of what a warm WOULD tokenize on
// every render, and slicing a 20,000 character fence to measure it is a copy per render.
export const trimmedLength = (text: string): number => {
  let end = text.length;
  while (end > 0 && text[end - 1] === "\n") end -= 1;
  return end;
};

export const trimTrailingNewlines = (text: string): string =>
  text.slice(0, trimmedLength(text));

/*
 * AN EMPTY FENCE IS ONE LINE TALL, not nothing.
 *
 * Streamdown renders a code block as one span per token line, and it special-cases the empty line:
 *
 *   children: c.length === 0 || (c.length === 1 && c[0].content === "") ? `\n` : c.map(...)
 *
 * It also trims trailing newlines off the source first, exactly as `trimTrailingNewlines` does
 * here, so a fence whose body is empty or is nothing but newlines tokenizes to a single empty line
 * and renders as one newline: one line box of height. A `<code>` holding an empty text node has no
 * line box at all, so without this the shell is one line shorter than the block it stands in for
 * and the fence grows by a line when it upgrades, moving everything below it.
 */
const shellBody = (source: string): string => {
  const trimmed = trimTrailingNewlines(source);
  return trimmed === "" ? "\n" : trimmed;
};

function FenceShell({
  language,
  source,
}: {
  language: string | null;
  source: string;
}) {
  return (
    <div
      className="my-4 flex w-full flex-col gap-2 rounded-xl border border-border bg-sidebar p-2"
      data-language={language ?? undefined}
      data-streamdown="code-block"
      data-unsloth-fence-deferred="true"
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
          <code>{shellBody(source)}</code>
        </pre>
      </div>
    </div>
  );
}

/**
 * Has this fence been reached yet? Latches true and never returns false again.
 *
 * Takes a ref to an element the CALLER already renders rather than mounting a
 * wrapper of its own. An extra div here would sit between a list item and its
 * code block and break `[data-streamdown="list-item"] > [data-streamdown=
 * "code-block"]`, and would push the block one level deeper than the
 * `:last-child` margin chain in index.css walks -- a layout change smuggled in
 * by a performance change, which is the one thing an A/B must not carry.
 */
// No IntersectionObserver, no gate. Read once at module scope so the decision
// is part of the rendered value rather than a state write from inside an
// effect, which would cost a cascading render on every fence in the thread.
const CAN_OBSERVE =
  typeof IntersectionObserver !== "undefined" &&
  typeof globalThis !== "undefined";

/*
 * THE REGISTER OF FENCES THAT HAVE NOT BEEN REACHED YET, for the two things that must reach across
 * all of them at once without waiting for a React render: a discontinuous scroll, and a print,
 * which puts the WHOLE document on the page. Both go through `latchNow` below.
 *
 * A gate carries the two elements its observers were built against, resolved at the same moment
 * and rebuilt with them, so nothing here re-walks the ancestor chain or re-reads a computed style.
 * `warm` comes from the caller because the highlighter instance lives with the block's component.
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
  latch: () => void;
  /** A state write that changes nothing, whose only job is to give React sync work to do. */
  poke: () => void;
};

const unreached = new Set<FenceGate>();

/** Is this gate's fence where the observers would call it reached? */
const gateOpen = (gate: FenceGate): boolean =>
  inBand(gate.node, gate.near)
  && (gate.near === gate.outer || inBand(gate.near as HTMLElement, gate.outer));

/*
 * UPGRADE THESE FENCES INSIDE THIS TASK, so the browser paints them highlighted instead of
 * painting the plain shell and correcting it a few frames later. Dropping any of the three steps
 * was measured to put a painted plain frame back:
 *
 *   1. `warm(true)`. Streamdown's highlighted body renders its own plain fallback whenever the
 *      plugin answers `null`, which it does only while a grammar is loading; with the grammar in
 *      hand it tokenizes in the same call. Warming makes the render below a cache hit.
 *   2. The inner `flushSync(latch)`. A normally scheduled update lands after the next paint, which
 *      is the whole defect. This commits the shell-to-block swap in this task.
 *   3. The inner `flushSync(poke)`, INSIDE an outer `flushSync`. Step 2 does not produce the
 *      COLOURED commit: `HighlightedCodeBlockBody` starts at `useState(raw)` and asks for tokens
 *      from a PASSIVE effect, so every newly mounted block renders unhighlighted once. Off screen
 *      at thread mount that is invisible; deferred, it is a plain frame on the reader's screen.
 *
 * WHY THE POKE AND THE NESTING, since an empty `flushSync(() => {})` does neither:
 *
 *   - React runs pending passive effects from `performSyncWorkOnRoot`, reached only when sync work
 *     is waiting, so a flush with nothing to do never runs the effect that colours the block.
 *     `poke` is that work: a state write on an already-latched fence whose own effects all
 *     early-return.
 *   - `flushSync` restores React's update priority BEFORE performing the flush, so the passive
 *     effect's update resolves against the ambient priority, and a scroll is continuous rather
 *     than discrete, so it is not flushed and lands next frame after all. The outer call holds the
 *     priority discrete across its whole body, so the inner flushes carry the effect's update too.
 *
 * One way only. Nothing here can clear a latch.
 */
const latchNow = (arrived: readonly FenceGate[]): void => {
  if (arrived.length === 0) return;
  for (const gate of arrived) {
    unreached.delete(gate);
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
 *
 * Neither the observers' one root height of lookahead (`REACH_MARGIN`) nor the render-time
 * pre-paint gate covers a jump: the viewport can move further than the lookahead in one step with
 * no React render in between, and an IntersectionObserver record is delivered one or more frames
 * AFTER the paint. Chromium at the 100K rung, 16 seeded jumps: 8 painted 3 to 4 frames of plain
 * code, 50 to 190 ms.
 *
 * A SCROLL LISTENER IS ENOUGH AND IS IN TIME: scroll events are dispatched in the "run the scroll
 * steps" of the same update-the-rendering pass that will paint the new position, before
 * animation-frame callbacks and before style, layout and paint, so a `flushSync` from here is part
 * of the frame the reader is about to see. `scrollTo`, wheel, drag and keyboard all arrive here.
 *
 * THE THRESHOLD IS DERIVED, NOT TUNED. The band is one root height `h` bigger each way, so after a
 * scroll of `d` the newly visible strip `[d, d + h]` is inside the old band exactly when `d <= h`.
 * The pass therefore runs only when the lookahead provably did not cover the movement, and a
 * continuous scroll pays one subtraction per event.
 *
 * ONE LISTENER, NOT ONE PER FENCE. Scroll does not bubble but does capture, so one capturing
 * document listener sees every element including nested reasoning panes. Attached when the first
 * fence registers and removed when the last latches, so nothing left to defer costs nothing.
 */
const lastScrollTop = new WeakMap<EventTarget, number>();
let scrollWatched = false;

const onScroll = (event: Event): void => {
  // A scroll only says positions moved; when a pass is already scheduled this is one boolean.
  if (latchedFences.size > 0) scheduleEvictionPass();
  if (unreached.size === 0) return;
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
};

const watchScrolling = (): void => {
  if (scrollWatched || typeof document === "undefined") return;
  scrollWatched = true;
  document.addEventListener("scroll", onScroll, { capture: true, passive: true });
};

const unwatchScrolling = (): void => {
  if (
    !scrollWatched
    || unreached.size > 0
    || latchedFences.size > 0
    || typeof document === "undefined"
  ) {
    return;
  }
  scrollWatched = false;
  document.removeEventListener("scroll", onScroll, { capture: true });
};

/*
 * THE EVICTION REGISTER, and the one edge in this file that is not one-way.
 *
 * Everything above bounds the span count at MOUNT, not over a SESSION: every fence the reader
 * scrolls past keeps its spans for the rest of the mount, and the STANDING count is the cost.
 * 63 mutations over 5 s at 16,958 elements read 18.9 fps against 49 mutations at 22,789 elements
 * reading 14.6 fps, fps against mutation count at r = -0.88. That sign says having spans costs,
 * not making them. So a fence FAR outside the viewport gives its highlighting back and returns to
 * the plain shell it started as. OFF BY DEFAULT (`code-fence-evict.ts`, `SHIP_DEFAULT = "off"`).
 *
 * WHY THIS IS NOT THE BIDIRECTIONAL GATE THIS FILE'S HEADER REJECTS. That one evicted on the
 * complement of the latch predicate, at a single boundary, from the scroll handler. Three things
 * differ, all of them in `code-fence-evict.ts` where they can be run as tests:
 *
 *   1. A WIDER BAND. Latch is one root height, eviction three, so a fence given back is two root
 *      heights outside the band that would take it again. No boundary to oscillate across.
 *   2. NO WORK IN THE SCROLL EVENT. A scroll schedules one idle callback and nothing else; the
 *      pass runs when the main thread is free, never inside frames that are already dropping.
 *   3. A BUDGET AND A DWELL. At most `EVICT_BUDGET` fences per pass, furthest first, and only
 *      fences highlighted for `DWELL_MS`, so a fast scroll cannot unmount what it just built.
 *
 * `planEviction` also refuses outright while a selection is live or a print is in force.
 */
type LatchedFence = {
  node: HTMLElement;
  /*
   * Resolved ONCE at latch, not re-walked every pass. It can go stale when `reasoning.tsx` drops
   * `max-h-64` and its pane stops being a scroller, but stale is safe here: the pane's box becomes
   * the whole trace, so the band is enormous. A stale root withholds evictions; it cannot cause one.
   */
  near: HTMLElement | null;
  latchedAt: number;
  unlatch: () => void;
};

const latchedFences = new Set<LatchedFence>();

let lastEvictionPass: number | null = null;
let evictionScheduled = false;

const nowMs = (): number =>
  typeof performance !== "undefined" && typeof performance.now === "function"
    ? performance.now()
    : Date.now();

/** The same two numbers `inBand` reads, so both bands are measured against one rectangle. */
const bandOf = (scroller: HTMLElement | null): ScrollerBand => {
  const bounds = scroller?.getBoundingClientRect();
  return bounds
    ? { top: bounds.top, height: bounds.height }
    : { top: 0, height: window.innerHeight };
};

/*
 * A LIVE SELECTION STOPS THE PASS: giving a fence back unmounts its subtree and a range anchored
 * inside it dies with it, so a reader who selects a long thread and copies would get a document
 * that changed under the selection. The shell's character-for-character equality cannot cover that.
 */
const selectionIsLive = (): boolean => {
  if (typeof document === "undefined" || typeof document.getSelection !== "function") {
    return false;
  }
  const selection = document.getSelection();
  return selection !== null && selection.rangeCount > 0 && !selection.isCollapsed;
};

/** `upgradeEverythingForPrint` latches the whole document on purpose. Do not race it. */
const printIsInForce = (): boolean =>
  typeof window !== "undefined"
  && Boolean(window.matchMedia?.("print")?.matches);

const runEvictionPass = (): void => {
  evictionScheduled = false;
  if (latchedFences.size === 0) return;
  const at = nowMs();
  if (!passIsDue(at, lastEvictionPass)) {
    // Wait out the remainder rather than drop the pass. `passIsDue` is true once it fires, so no loop.
    scheduleEvictionPass(PASS_INTERVAL_MS - (at - (lastEvictionPass ?? at)));
    return;
  }
  lastEvictionPass = at;
  const fences = [...latchedFences];
  const candidates: EvictCandidate[] = fences.map((fence, id) => {
    const rect = fence.node.getBoundingClientRect();
    return {
      id,
      rect: { top: rect.top, bottom: rect.bottom },
      band: bandOf(fence.near),
      latchedAt: fence.latchedAt,
      // Never registered while streaming; restated so `code-fence-evict.ts` still runs the row.
      streaming: false,
    };
  });
  const plan = planEviction(candidates, at, {
    selectionLive: selectionIsLive(),
    printing: printIsInForce(),
  });
  for (const id of plan) {
    const fence = fences[id];
    latchedFences.delete(fence);
    fence.unlatch();
  }
  // A pass is scheduled BY a scroll, so without this the budget would strand a scrolled-through
  // thread part way until the next gesture. `null` is the resting state: no timer at all.
  const next = nextPassDelayMs(candidates, at, plan.length);
  if (next !== null) scheduleEvictionPass(next);
};

const scheduleEvictionPass = (delayMs?: number): void => {
  if (evictionScheduled || typeof globalThis === "undefined") return;
  evictionScheduled = true;
  if (delayMs !== undefined) {
    setTimeout(runEvictionPass, Math.max(0, delayMs));
    return;
  }
  const idle = (globalThis as Record<string, unknown>).requestIdleCallback as
    | ((cb: () => void, options?: { timeout: number }) => number)
    | undefined;
  if (typeof idle === "function") idle(runEvictionPass, { timeout: PASS_INTERVAL_MS });
  else setTimeout(runEvictionPass, PASS_INTERVAL_MS);
};

/*
 * PRINT, the one gesture that puts every deferred fence on the page at once.
 *
 * Colour is all deferral costs a printed page -- the shell holds a live text node, so it prints
 * complete, in order and in the right font -- but a page that lost the colour on fences the reader
 * never scrolled past is a defect they can see and keep. Nor does the printed window match the
 * reader's: a print lays out at PAPER width while the scroll offset carries across as raw pixels,
 * so the page lands several fences away. Letter against a 1280 px window: 1 to 2 deferred fences
 * on every sampled page, 6.6% pixel difference against the same page with the flag off; matching
 * paper to window removes it, which identifies the cause and is why chasing the window is the
 * wrong fix. The whole document is on the page, so the whole document has to be highlighted.
 *
 * WHY AN EARLIER ATTEMPT FAILED and was reported as impossible: it latched every fence from
 * `beforeprint` with `flushSync`, the blocks swapped, and 53 of 56 still printed on streamdown's
 * raw fallback, because the swap alone renders UNHIGHLIGHTED while the passive effect waits on a
 * grammar. `latchNow` closes both halves (warm, then flush the colouring render as well as the
 * swap) and `warmGrammars` below keeps a loading grammar from being what is missing at snapshot.
 *
 * BOTH DOORS: `beforeprint` covers Ctrl+P and the print menu; headless `page.pdf()` and DevTools
 * print emulation change the media query without firing it, and a PDF export is how a reader keeps
 * a copy.
 *
 * A PRINT UPGRADES THE DOCUMENT THAT WAS PRINTED, AND NOTHING ELSE. This was a module-global
 * `printed` folded into every future fence's `reached`, a session-wide latch rather than a
 * monotonic one: at the 100K rung a thread with 53 of 56 fences deferred, 2,458 spans and 22,794
 * elements printed once, then after navigating away inside the app and back remounted with 0
 * deferred, 41,410 spans and 61,747 elements. One Ctrl+P turned the default off for the tab's
 * life, including threads never on the printed page. So a print latches what is on the page WHEN
 * IT HAPPENS and a fence mounted afterwards defers again; every print does it again, so print,
 * scroll, print gives a second page as complete as the first. Still one way only: `latchNow` can
 * only latch, and reverting on `afterprint` would be the bidirectional edge this design avoids.
 */
const upgradeEverythingForPrint = (): void => {
  latchNow([...unreached]);
};

/*
 * GRAMMARS, WARMED AT IDLE, ON REAL TEXT, ONE TOKENIZATION PER TASK.
 *
 * One fence per language, so `latchNow`'s synchronous path cannot be defeated by a still-loading
 * grammar: on a jump into a language the reader has not met, and on a print, where there is no
 * later frame to correct in. Nothing runs when nothing is deferred. "Per language" means per
 * GRAMMAR, `normalizeLanguage`, not per fence tag: ```py and ```python are one grammar to the
 * highlighter, and two keys here would warm it twice, on two different fences.
 *
 * IT USED TO WARM ON AN EMPTY STRING. Loading a grammar is the cheap half; running it over text
 * the first time is not, and `""` never does the second. So the first REAL tokenization still paid
 * the whole one-off cost, and with deferral on that landed in one frame, during a scroll, on the
 * fence the reader had just reached. Three arms out of ONE build at the 100K rung on WebKitGTK
 * 2.50.4, two reps each, so a difference between arms cannot be a build difference:
 *
 *                          worst scroll frame   warm callbacks, ms
 *   warm on ""                 1200, 1085 ms     1                and 1
 *   warm on real text           183,  190 ms     1                and 2
 *   this: split, eager loads    187,  183 ms     1/64/21/32/1016  and 2/68/36/52/1104
 *
 * The 1,016 ms is ONE tokenize call, typescript's first, which nothing here can subdivide; the
 * split only keeps the other four off it. Every arm pays that call: unsplit at t=6.5 s inside a
 * grammar-load callback, which is why its warm reads 1 ms, and on the `""` arm at t=36.5 s, inside
 * the scroll, where it IS the worst frame. Moved, not skipped, and the totals say so: tokenize time
 * RISES, 1607-1746 ms to 1917-2039 ms, because a warmed fence is tokenized once and read from cache
 * later. Mount is unaffected, worst idle frame is best here (25 and 32 ms against 46/37 unsplit and
 * 38/43 on `""`), idle is 62.5 fps and 7.1 to 7.4% busy on every arm, the three arms render the
 * same document (7,259 spans, 27,045 elements, 56 code blocks), and the rig is jam-controlled: a
 * 200 ms/250 ms hog reads 81% busy.
 *
 * WHY THE TOKENIZATIONS YIELD AND THE LOADS DO NOT. An idle callback only chooses when it STARTS:
 * nothing yields once it runs, its 2,000 ms timeout can start it on a busy thread, and WebKitGTK
 * has no `requestIdleCallback` at all, so this venue takes the `setTimeout` fallback and cannot
 * even pick a quiet moment. With a grammar already loaded `code.highlight` tokenizes INLINE and N
 * languages concatenate -- driving shiki with this component's configuration, the five languages
 * here (c, python, go, typescript, rust) cost 746 ms back to back. But `highlight` answers `null`
 * WHILE a grammar loads, so yielding the loads too would put the fifth grammar 500 ms x N away and
 * a jump or a print inside that window would get streamdown's plain fallback out of `latchNow`'s
 * flush -- the defect this whole pre-warm exists to prevent. So every load starts in the first
 * pass and only the tokenizing is spread out.
 *
 * NOTHING BOUNDED THE SIZE OF A WARM, and this comment used to claim `MAX_HIGHLIGHT_CHARS` did.
 * It does not reach here: `markdownPluginNeeds` applies it in `markdown-preview.tsx` and
 * `model-readme.tsx`, `tool-code-cell.tsx` and `attachment-preview.tsx` apply it to their own
 * source, but `markdown-text.tsx` supplies the code plugin unconditionally and `FenceBlock` warms
 * the whole body -- and `code-plugin.ts`'s `evict` keeps the last fence whatever its size. A LATCH
 * is demanded work and stays uncapped; a warm is SPECULATIVE, on a fence the reader may never
 * reach, so it is capped at the same 20,000 characters. Over the cap, and for a fence that is
 * EMPTY or nothing but newlines, the grammar loads and nothing is tokenized; neither marks the
 * language warmed, so a later fence that can warm it still does. The cap costs the measurement
 * nothing: the largest fence at this rung is 2,817 characters.
 */
const grammarsWarmed = new Set<string>();
const grammarsLoaded = new Set<string>();
let warmScheduled = false;

// Keyed the way `highlight` keys it, or `py` and `Python` are two keys for one grammar.
const grammarOf = (gate: FenceGate): string =>
  normalizeLanguage(gate.language ?? "text");

const warmGrammars = (): void => {
  warmScheduled = false;
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
  const idle = (globalThis as Record<string, unknown>).requestIdleCallback as
    | ((cb: () => void, options?: { timeout: number }) => number)
    | undefined;
  if (typeof idle === "function") idle(warmGrammars, { timeout: 2000 });
  else setTimeout(warmGrammars, 500);
};

if (typeof window !== "undefined" && typeof window.addEventListener === "function") {
  window.addEventListener("beforeprint", upgradeEverythingForPrint);
  window.matchMedia?.("print")?.addEventListener?.("change", (event) => {
    if (event.matches) upgradeEverythingForPrint();
  });
}

/*
 * THE NEAREST SCROLLING ANCESTOR, found rather than named.
 *
 * This used to match two known selectors, `[data-slot='thread-viewport']` and
 * `.aui-thread-viewport`, and `closest()` walks straight past anything that matches neither. The
 * one that matters is the reasoning pane: while a reply streams, `reasoning.tsx` gives its trace
 * `overflow-y-auto` and `max-h-64` and pins it to the bottom, so the reader is looking at an
 * arbitrarily long trace through a 256 px window nested inside the thread scroller.
 *
 * What this is NOT. It is not a correctness fix. Intermediate scrollers clip, so intersection was
 * always computed correctly and fences inside that pane were always deferred properly; measured in
 * Chromium with ten fences in a 256 px inner scroller inside a 400 px outer one, 3 of 10 reported
 * intersecting either way. It is a LOOKAHEAD fix. `rootMargin` expands the ROOT's rectangle, so
 * rooting at the thread viewport expanded a rectangle the reader is not looking through and the
 * one-viewport warning was worth nothing inside the pane: the "100% 0px" column and the no-margin
 * column were identical at 3 of 10, while rooting at the inner scroller gave 5 of 10. The
 * user-visible consequence of getting it wrong is narrow and real: a fence scrolling into the
 * 256 px window gets no pre-warm and shows the plain shell for the frames the upgrade takes.
 *
 * `null` is deliberately still possible and is deliberately NOT the default: a fence with no
 * scrolling ancestor really is clipped by the document viewport, but assuming that when a scroller
 * exists is the bug the review caught.
 */
const isScrollable = (el: HTMLElement): boolean => {
  const overflowY = getComputedStyle(el).overflowY;
  return (
    (overflowY === "auto" || overflowY === "scroll" || overflowY === "overlay")
    && el.scrollHeight > el.clientHeight
  );
};

const scrollerOf = (node: HTMLElement): HTMLElement | null => {
  for (let el = node.parentElement; el !== null; el = el.parentElement) {
    if (isScrollable(el)) return el;
  }
  return null;
};

/*
 * THE OUTERMOST ONE TOO, and why the nearest is not enough on its own.
 *
 * An explicit root is clipped by the ancestors BETWEEN the target and the root, and by NOTHING
 * above it. Rooting at the reasoning pane therefore asks "is this fence inside the pane's own
 * window" and never "is the pane itself anywhere near the reader". Two ways that goes wrong, and
 * the second one undoes the whole change on the pane most likely to hold the spans:
 *
 *   1. A pane scrolled far out of the thread still reports the two or three fences inside its
 *      256 px window as intersecting, so they upgrade where nobody can see them.
 *   2. `reasoning.tsx` drops `max-h-64` when the stream ends but KEEPS `overflow-y-auto`, so the
 *      pane stops being scrollable and its box becomes the whole trace. An observer still rooted
 *      at it from the streaming frames then reports EVERY fence in that trace as intersecting at
 *      once.
 *
 * Rooting at the outermost scroller instead would fix both and cost the lookahead that rooting at
 * the nearest one bought: measured in Chromium with ten fences in a 256 px inner scroller inside a
 * 400 px outer one, the inner root reports 5 of 10 with a one-window margin against 3 of 10 for
 * the outer root, and the difference is exactly the pre-warm that keeps a fence from showing its
 * plain shell as it scrolls into the pane.
 *
 * So when the two differ there are two gates, watching two different elements, and the latch needs
 * both: the FENCE against the nearest scroller answers "the reader is about to reach it inside the
 * pane", and the PANE against the outermost scroller answers "the pane is somewhere the reader can
 * see". Watching the fence through the outer root instead would clip it at the pane on the way and
 * go false exactly where the inner lookahead was doing its work -- 2 of 10 against 4, measured with
 * the pane fully in view. When the two scrollers are the same element, which is every fence outside
 * a nested scroller, there is exactly one observer and this costs nothing.
 *
 * THE CONJUNCTION IS NOT ENOUGH ON ITS OWN, which took a measurement to see. A stale inner root
 * is too permissive, and the outer gate only covers that while the pane is out of view. Scroll the
 * expanded pane so much as partly on screen and the outer gate is true, so the stale inner root
 * decides alone and reports the WHOLE trace: 10 of 10 in both engines, on a 4,080 px trace against
 * a 900 px viewport where the right answer is about 3. So the inner root is re-resolved when the
 * pane stops scrolling, which collapses the fence back to the single-gate case where the outermost
 * scroller is the root and nothing can be stale.
 *
 * Watched with a ResizeObserver on the pane rather than by re-resolving on every frame, and only
 * for fences that actually have a nested scroller. The callback reads one `overflow-y` on one
 * element, so a streaming pane resizing every frame costs one computed-style read per fence per
 * frame and rebinds nothing until the cap actually comes off.
 *
 * Still one-way. Nothing here can clear a latch; the extra gate and the rebind can only withhold
 * one.
 */
const outermostScrollerOf = (node: HTMLElement): HTMLElement | null => {
  let found: HTMLElement | null = null;
  for (let el = node.parentElement; el !== null; el = el.parentElement) {
    if (isScrollable(el)) found = el;
  }
  return found;
};

/**
 * Is `node` inside `scroller`'s box grown by one of its own heights, the observer's margin?
 *
 * The arithmetic lives in `code-fence-evict.ts` so the LATCH and EVICT bands are one function at
 * two widths, not two copies that drift; its tests run `withinBand(rect, band, REACH_BAND)` over a
 * grid against a literal transcription of what this used to compute.
 */
const inBand = (node: HTMLElement, scroller: HTMLElement | null): boolean => {
  const rect = node.getBoundingClientRect();
  return withinBand(
    { top: rect.top, bottom: rect.bottom },
    bandOf(scroller),
    REACH_BAND,
  );
};

/**
 * @param enabled  false on the shipped default, where this hook must cost nothing at all: no
 *                 state is ever written, no observer is built and no layout is read.
 * @param streaming  the fence is still being written. It is highlighted while it streams AND it
 *                 latches, so that finishing cannot take the highlighting back.
 * @param language  used only to load one grammar per language rather than one per fence; `null`
 *                 warms plain text.
 * @param chars  this fence's source length, read only by `warmGrammars` to keep a SPECULATIVE
 *                 warm inside `MAX_HIGHLIGHT_CHARS`. A latch is demanded work and is not capped.
 * @param warm  drive the highlighter over this fence: `true` for tokens, `false` for the grammar
 *                 alone. See `latchNow`. Held in a ref, not an effect dependency, so an
 *                 unmemoized caller cannot rebuild every observer in the thread on every render.
 */
export function useFenceReached(
  host: RefObject<HTMLElement | null>,
  enabled: boolean,
  streaming: boolean,
  language: string | null,
  chars: number,
  warm: (tokens: boolean) => void,
): boolean {
  const [latched, setLatched] = useState(false);
  // Bumped when the resolved scrolling ancestor stops being one, which rebuilds the gates below
  // against the element that clips this fence now. Never read for anything else.
  const [generation, setGeneration] = useState(0);
  const reached = !enabled || !CAN_OBSERVE || streaming || latched;
  // Read here, not threaded through the caller, so the flag off gives an identical component tree.
  const evicting = enabled && CAN_OBSERVE && fenceEvictMode() === "evict";
  const warmRef = useRef(warm);
  useEffect(() => {
    warmRef.current = warm;
  }, [warm]);

  /*
   * A COMPLETING STREAM MUST NOT DOWNGRADE.
   *
   * `streaming` goes true -> FALSE when streamdown recognises the closing delimiter. Deriving
   * `reached` from it alone therefore takes a fence that was highlighted all through its stream
   * and hands it back the plain shell the moment it finishes: highlighted -> plain, on a block the
   * reader is watching, which is precisely the reverse edge this design exists to remove. The
   * fence was reached; that has to be recorded, not recomputed.
   *
   * In a layout effect so the downgrade is never painted even for one frame.
   */
  useLayoutEffect(() => {
    if (!enabled || latched || !streaming) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLatched(true);
  }, [enabled, latched, streaming]);

  /*
   * THE FIRST FRAME, which the observer cannot cover.
   *
   * An IntersectionObserver delivers its first record asynchronously, one or more frames after
   * `observe()`. For a fence that is ALREADY on screen when the thread mounts, those frames are
   * plain code inside the viewport: measured at 2 to 3 frames, 30 to 190 ms. That is a visible
   * difference, not an off-screen one, so it is closed here rather than argued about.
   *
   * `useLayoutEffect` runs after mutation and BEFORE paint, so a fence that is on screen latches
   * and re-renders within the same frame and the plain shell is never painted. One
   * `getBoundingClientRect` per unreached fence, all of them inside a single commit with no DOM
   * mutation in between, so the layout is forced once for the whole thread rather than once per
   * fence.
   *
   * RE-RUN ON A REBIND. `generation` is a dependency because the ResizeObserver below bumps it
   * when the reasoning pane stops scrolling, and expanding that pane can bring a fence inside the
   * outer viewport for the first time. Without it the rebind's own render leaves the plain shell
   * in place, and the replacement observer is built in a passive effect and delivers
   * asynchronously, so the shell is PAINTED. That is an on-screen difference, which is the one
   * kind this change is not allowed to have.
   *
   * Still one-way: this can only ever latch true.
   */
  useLayoutEffect(() => {
    if (reached) return;
    const node = host.current;
    if (!node) return;
    // The same two questions the observers below ask, of the same two elements: is the FENCE in
    // the window the reader is looking through, and is that WINDOW itself on screen.
    const near = scrollerOf(node);
    const outer = outermostScrollerOf(node);
    if (inBand(node, near) && (near === outer || inBand(near as HTMLElement, outer))) {
      // The cascading render this warns about is the POINT: it is what keeps the plain shell off
      // the screen. It happens at most once per fence, only for the one or two fences that are
      // already on screen at mount, and the alternative is the 2 to 3 painted frames of
      // unhighlighted code this replaced.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setLatched(true);
    }
  }, [reached, host, generation]);

  // The one-way edge. Once `reached` is true this effect re-runs, takes the
  // early return, and never observes anything again, so a fence that has been
  // read carries no residual per-scroll cost at all.
  useEffect(() => {
    if (reached) return;
    const node = host.current;
    if (!node) return;
    // ROOTED AT THE THREAD'S SCROLLER, not at the document.
    //
    // The chat scrolls inside a nested overflow container. With `root` unset the root is the
    // document viewport, and `rootMargin` expands THAT rectangle -- while the intersection is
    // still clipped by the scroller's own edges, which no margin can widen. The lookahead would
    // then be worth nothing: a fence would report intersecting only once it was already inside
    // the visible band, and the reader would get the plain shell for the frames it takes the
    // observer to deliver and the upgrade to render. Rooting at the scroller is what makes the
    // margin mean one viewport of warning.
    //
    // See `outermostScrollerOf`: when a nested scroller is in the way, the outermost one is
    // observed as well and the latch needs both. One observer, not two, whenever they agree.
    const near = scrollerOf(node);
    const outer = outermostScrollerOf(node);
    // WHAT EACH GATE WATCHES, which is not the same element.
    //
    // Observing the FENCE against the outer root clips it at the pane on the way, so the outer
    // gate goes false exactly where the inner lookahead was supposed to be doing its work: 2 of 10
    // against the inner root's 4, measured, with the pane fully in view. Observing the PANE
    // against the outer root asks the question the outer gate is for -- is the pane anywhere the
    // reader can see -- and the inner gate keeps its lookahead untouched at 4 of 10.
    const gates: [Element, Element | null][] = near === outer
      ? [[node, near]]
      : [[node, near], [near as HTMLElement, outer]];
    const seen = gates.map(() => false);
    const observers = gates.map(([, root], i) => new IntersectionObserver(
      (entries) => {
        seen[i] = entries.some((entry) => entry.isIntersecting);
        if (!seen.every(Boolean)) return;
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
      latch: () => setLatched(true),
      // Reuses `generation`: this fence has just latched, so every effect keyed on it
      // early-returns and the bump costs one render. See `latchNow` for why React needs the work.
      poke: () => setGeneration((n) => n + 1),
    };
    unreached.add(registered);
    watchScrolling();
    scheduleGrammarWarm();

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
      unreached.delete(registered);
      unwatchScrolling();
    };
  }, [reached, host, generation]);

  /*
   * THE EVICTION REGISTER. See the block comment on `latchedFences` for why this edge exists and
   * why it is not the bidirectional gate this file's header rejects.
   *
   * The ONLY place in this file that clears a latch, and unreachable with the flag off: nothing is
   * registered, no pass is scheduled, and `reached` behaves exactly as it did before this existed.
   * A STREAMING FENCE IS NOT REGISTERED; the layout effect above latches it precisely so finishing
   * the stream cannot take its colours back. The cleanup deregisters, so an unmount, or a session
   * that turns the flag off, leaves nothing behind for a pass to walk.
   */
  useEffect(() => {
    if (!enabled || !evicting || !latched || streaming) return;
    const node = host.current;
    if (!node) return;
    const registered: LatchedFence = {
      node,
      near: scrollerOf(node),
      latchedAt: nowMs(),
      unlatch: () => setLatched(false),
    };
    latchedFences.add(registered);
    // The pass reads positions, so it must hear about scrolling; scheduled once here so a fence
    // latched by a jump or a print is considered even if that scroller never moves again.
    watchScrolling();
    scheduleEvictionPass();
    return () => {
      latchedFences.delete(registered);
      unwatchScrolling();
    };
  }, [enabled, evicting, latched, streaming, host]);

  return reached;
}

export const DeferredFenceShell = memo(FenceShell);
