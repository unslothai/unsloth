// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import type { HighlightResult } from "@streamdown/code";
import {
  memo,
  type RefObject,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { flushSync } from "react-dom";

import { MAX_HIGHLIGHT_CHARS } from "@/lib/markdown-plugins";
import { type FenceMode, resolveFenceMode } from "./code-fence-mode";
import {
  isBlankLine,
  type LineWindow,
  lineIsWindowed,
  plainLineText,
  selectLineWindow,
} from "./code-fence-window";
import { normalizeLanguage } from "./code-plugin";

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
 * on the page, so the whole document has to be highlighted.
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

  return reached;
}

export const DeferredFenceShell = memo(FenceShell);

/* ------------------------------------------------------------------------------------------- *
 * THE HIGHLIGHTED BODY, RENDERED HERE RATHER THAN BY STREAMDOWN.
 *
 * Streamdown's `CodeBlockBody` maps the WHOLE token array to elements on every render and is
 * memoized on `prev.result === next.result`, reference equality. `code-plugin.ts` hands back a
 * fresh result object on every call -- including `approximateResult`, which runs on every streamed
 * frame between the 250 ms refreshes -- so that memo never hits and React rebuilds every line and
 * every token in the fence about sixty times a second. At the 140K characters of #10769 that is
 * some fifty to seventy thousand elements per frame, and it is what made the UI unusable.
 *
 * It is also, separately, what PR #10779 measured. That change kept calling the highlighter and
 * only stopped RENDERING the tokens, and still took a 140K stream from 29 fps to 59 fps. The
 * tokenizer was never the problem; the spans were. `scripts/coal-span-census.mjs` had already
 * closed the other door: Shiki emits maximally coalesced tokens (537013 -> 537013 over the whole
 * corpus), so the span count cannot be reduced by merging. It can only be reduced by not mounting
 * spans for code nobody is looking at.
 *
 * So this body does two things Streamdown's cannot:
 *   1. ONE MEMOIZED COMPONENT PER LINE. `code-plugin.ts` pushes each completed line into
 *      `fence.lines` exactly once and never rebuilds it, so a committed line's array identity is
 *      already stable across refreshes and `memo` bails out on it. A growing fence then reconciles
 *      the live tail line and whatever was just committed, instead of all of it.
 *   2. A LINE WINDOW. Past `WINDOW_CAP_LINES` a line outside the window renders its text as one
 *      node instead of its token spans. See `code-fence-window.ts` for why that is safe and for
 *      why it is not virtualization: the characters never leave the document, so selection, copy,
 *      find-in-page and print are untouched and only off-screen COLOUR is given up.
 *
 * The DOM is Streamdown's, element for element and class for class, including the line-number
 * pseudo-element `index.css` then nulls out. That is not tidiness: `playwright_code_block_flicker.py`
 * reads computed styles off this subtree, and a performance change that also moves the rendering
 * is a change no A/B can attribute.
 * ------------------------------------------------------------------------------------------- */

export type FenceTokens = HighlightResult;
type TokenLine = HighlightResult["tokens"][number];
type FenceToken = TokenLine[number];

/* Streamdown's own class lists, copied verbatim. The `bg-[var(--sdm-bg,inherit]` spellings are
 * unbalanced in streamdown 2.5's build and therefore generate no rule at all; they are reproduced
 * as they are because the goal is the same DOM, not a tidier one. */
const LINE_CLASS =
  "block before:content-[counter(line)] before:inline-block before:[counter-increment:line] before:w-6 before:mr-4 before:text-[13px] before:text-right before:text-muted-foreground/50 before:font-mono before:select-none";
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
}: {
  line: TokenLine;
  windowed: boolean;
}) {
  if (isBlankLine(line)) {
    return <span className={LINE_CLASS}>{"\n"}</span>;
  }
  if (!windowed) {
    // One text node for the whole line. Same characters, same block box, same height: the only
    // thing this line has given up is its colour, and it is off screen.
    return <span className={LINE_CLASS}>{plainLineText(line)}</span>;
  }
  return (
    <span className={LINE_CLASS}>
      {line.map((token, index) => {
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
      })}
    </span>
  );
});

/*
 * ONE SCROLL LISTENER AND ONE FRAME FOR EVERY WINDOWED FENCE ON THE PAGE.
 * A listener per fence would be a listener per fence, and the measurement each one performs reads
 * layout; doing that inside the scroll handler for every fence separately is how a scroll gets
 * slower than the rendering the window exists to avoid. Registered on the first windowed fence and
 * removed with the last, exactly as `watchScrolling` does for the reach latch.
 * Coalesced into an animation frame: scroll fires faster than the screen updates, and the window
 * only has to be right for the frame that is about to be painted.
 */
const windowedFences = new Set<() => void>();
let windowFrame = 0;
let windowWatched = false;

/*
 * A PRINT PUTS THE WHOLE DOCUMENT ON THE PAGE, SO THE WHOLE FENCE HAS TO BE COLOURED.
 * `upgradeEverythingForPrint` above makes exactly this argument for a DEFERRED fence, and the line
 * window reintroduces the same defect one level down: measured, a 3,000 line fence printed with
 * 342 spans against the 23,139 the merge base printed, so all but a screenful of a printed listing
 * came out uncoloured. Colour is the only thing a window costs, and a printed page is the one
 * place the reader keeps it.
 * WHY THIS ONE REVERTS AND THE LATCH DOES NOT. The latch refuses `afterprint` because giving a
 * fence back its plain shell is the bidirectional edge that design exists to remove, and because
 * re-latching would mean re-tokenizing. Neither applies here: the tokens are already in
 * `fence.lines`, so re-windowing after the print costs element creation and nothing else, and NOT
 * reverting would mean one Ctrl+P un-windows every huge fence for the life of the tab, which is
 * precisely the cost the window exists to avoid.
 * BOTH DOORS, as above: `beforeprint` covers Ctrl+P and the print menu; headless `page.pdf()` and
 * devtools print emulation change the media query without firing it.
 */
let printing = false;

const remeasureWindows = (): void => {
  windowFrame = 0;
  for (const measure of windowedFences) measure();
};

/** Is a print in progress? While it is, every fence renders every line highlighted. */
export const fencePrinting = (): boolean => printing;

/*
 * Synchronous, and inside `flushSync`, for the same reason `latchNow` is: a normally scheduled
 * update lands after the next paint, and there is no next paint before the print snapshot.
 */
const setPrinting = (value: boolean): void => {
  if (printing === value || windowedFences.size === 0) return;
  printing = value;
  if (windowFrame !== 0) {
    cancelAnimationFrame(windowFrame);
    windowFrame = 0;
  }
  flushSync(remeasureWindows);
};

const scheduleRemeasure = (): void => {
  if (windowFrame !== 0 || windowedFences.size === 0) return;
  windowFrame = requestAnimationFrame(remeasureWindows);
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
};

const unwatchWindows = (): void => {
  if (!windowWatched || windowedFences.size > 0 || typeof document === "undefined") {
    return;
  }
  windowWatched = false;
  document.removeEventListener("scroll", scheduleRemeasure, { capture: true });
  window.removeEventListener("resize", scheduleRemeasure);
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
function useLineWindow(
  code: RefObject<HTMLElement | null>,
  /*
     * The fence's outermost element, and the ONLY thing it is used for is finding the scrolling
     * ancestor. `scrollerOf` starts at `parentElement` and tests `overflow-y`, and a code block
     * carries `overflow-x: auto`, which makes `overflow-y` compute to `auto` as well
     * (css-overflow-3: a non-visible value on one axis forces the other off `visible`). Walking up
     * from the `<code>` would therefore be one `scrollHeight` away from rooting the whole window
     * calculation inside the fence's own horizontal scroller. Starting outside the block skips
     * both of them, and it is the element the reach latch already measures against.
     */
  frame: RefObject<HTMLElement | null>,
  lineCount: number,
  enabled: boolean,
): LineWindow | null {
  const [lineWindow, setLineWindow] = useState<LineWindow | null>(null);
  // The rendered window, read by `measure` without making it an effect dependency: the effect
  // registers a listener, and rebuilding that on every window move would defeat the coalescing.
  const current = useRef<LineWindow | null>(null);
  const lines = useRef(lineCount);
  lines.current = lineCount;
  const hasBody = lineCount > 0;
  const measure = useRef<() => void>(() => {});

  measure.current = () => {
    const node = code.current;
    const outer = frame.current;
    if (!node || !outer) return;
    // See `setPrinting`: the whole document is on the page, so the whole fence is coloured.
    if (printing) {
      if (current.current === null) return;
      current.current = null;
      setLineWindow(null);
      return;
    }
    const scroller = scrollerOf(outer);
    const bounds = scroller?.getBoundingClientRect();
    const rect = node.getBoundingClientRect();
    const count = lines.current;
    const next = selectLineWindow({
      lineCount: count,
      // MEASURED, never assumed. `index.css` pins `line-height: 1.55` on the code block, but the
      // font size is a `--ui-font-scale` multiple inside a container query, so the pixel height is
      // not knowable from here. Every line is rendered, so the mean IS the line height.
      lineHeight: count > 0 ? rect.height / count : 0,
      contentTop: rect.top,
      viewportTop: bounds ? bounds.top : 0,
      viewportHeight: bounds ? bounds.height : window.innerHeight,
      previous: current.current,
    });
    // `selectLineWindow` hands the previous object straight back when nothing moved, so this is an
    // identity check and an ordinary scroll costs no render at all.
    if (next === current.current) return;
    current.current = next;
    setLineWindow(next);
  };

  /*
     * IN A LAYOUT EFFECT, AND KEYED ON THE BODY EXISTING.
     * Two things go wrong with a passive effect keyed on `enabled` alone, and the browser probe
     * caught both. A fence renders the plain shell until its grammar chunk lands, so at mount there
     * is no `<code>` to measure and no element for the ResizeObserver to watch; keyed only on
     * `enabled` the effect never runs again once the tokens arrive, and the window stayed off until
     * the reader happened to scroll (measured: 30,861 spans still mounted). And a passive effect
     * lands after the paint, so the frame that introduces a 20,000 line fence paints every span in
     * it before the window takes them away. Layout effects run after mutation and before paint, so
     * the first painted frame is already windowed.
     */
  useLayoutEffect(() => {
    if (!enabled) {
      current.current = null;
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setLineWindow(null);
      return;
    }
    const run = () => measure.current();
    windowedFences.add(run);
    watchWindows();
    run();
    /*
       * The fence's own growth moves nothing the scroll listener would notice: a streamed line
       * lands below the viewport and the reader has not moved. A ResizeObserver on the code
       * element is what sees it, and it also covers the font finishing loading, the thread column
       * changing width, and the reasoning pane expanding.
       */
    /*
       * OBSERVED ON THE WRAPPER, NOT ON THE `<code>`.
       * A `<code>` is `display: inline`, and ResizeObserver does not observe an element with no
       * principal box: the callback simply never fires. Measured, not read off the spec -- with the
       * observer on the `<code>` a fence streamed past 3,000 lines and 23,139 spans and the window
       * never once engaged, because the only thing that ever called `measure` was the one call this
       * effect makes, back when the fence was one line long. The wrapper is a flex container, it
       * grows by exactly what the code grows by, and it is already resolved here.
       */
    let resize: ResizeObserver | undefined;
    const box = frame.current;
    if (box && typeof ResizeObserver !== "undefined") {
      resize = new ResizeObserver(scheduleRemeasure);
      resize.observe(box);
    }
    return () => {
      resize?.disconnect();
      windowedFences.delete(run);
      unwatchWindows();
    };
    // `hasBody` and not `lineCount`: the count changes on every streamed line and re-registering
    // per line would throw away the coalescing this exists for. The transition that matters is the
    // shell becoming a real body, and it happens once.
  }, [enabled, hasBody, code, frame]);

  return enabled ? lineWindow : null;
}

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
  const frame = useRef<HTMLDivElement | null>(null);
  const tokens = result?.tokens ?? null;
  const lineWindow = useLineWindow(code, frame, tokens?.length ?? 0, windowing);
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
        className={joinClasses(
          languageClass,
          "overflow-x-auto rounded-md border border-border bg-background p-4 text-sm",
        )}
        data-language={language ?? undefined}
        data-streamdown="code-block-body"
        data-unsloth-fence-windowed={lineWindow === null ? undefined : "true"}
      >
        <pre className={joinClasses(languageClass, PRE_CLASS)} style={rootStyle}>
          <code className={CODE_CLASS} ref={code}>
            {tokens.map((line, index) => (
              <FenceLine
                key={index}
                line={line}
                windowed={lineIsWindowed(lineWindow, index)}
              />
            ))}
          </code>
        </pre>
      </div>
    </div>
  );
});
