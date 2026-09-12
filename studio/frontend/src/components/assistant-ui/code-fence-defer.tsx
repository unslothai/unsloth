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
  window.addEventListener("beforeprint", upgradeEverythingForPrint);
  window.matchMedia?.("print")?.addEventListener?.("change", (event) => {
    if (event.matches) upgradeEverythingForPrint();
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
