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

import { type FenceMode, resolveFenceMode } from "./code-fence-mode";

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

// How far outside the viewport a fence counts as "reached". One viewport of
// slack in each direction, so the upgrade has a frame or two to land before the
// block is actually on screen and the reader never sees the swap.
const REACH_MARGIN = "100% 0px";

/*
 * WHICH MODE IS IN FORCE lives in `code-fence-mode.ts`, a plain `.ts` module with no JSX in it, so
 * the decision table is exercised by a test that RUNS it rather than by regexes over this file.
 * Re-exported here because this is the module every consumer already imports.
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

// Streamdown trims trailing newlines off a fence body before rendering it, so
// the shell has to as well or the two differ by a blank line of height.
export const trimTrailingNewlines = (text: string): string => {
  let end = text.length;
  while (end > 0 && text[end - 1] === "\n") end -= 1;
  return text.slice(0, end);
};

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
 * THE REGISTER OF FENCES THAT HAVE NOT BEEN REACHED YET.
 *
 * Two things need to reach across all of them at once, and neither can wait for a React render:
 * a discontinuous scroll, which lands the viewport somewhere no observer has warned about, and a
 * print, which puts the WHOLE document on the page at once. Both are handled below, and both go
 * through `latchNow`.
 *
 * A gate carries the two elements its observers were built against, resolved at the same moment
 * and rebuilt with them, so nothing here walks the ancestor chain or reads a computed style
 * again. `warm` is supplied by the caller because the highlighter instance lives with the
 * component that renders the block, not here.
 */
type FenceGate = {
  node: HTMLElement;
  near: HTMLElement | null;
  outer: HTMLElement | null;
  language: string | null;
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
 * painting the plain shell first and correcting it a few frames later.
 *
 * THREE STEPS, and all three are needed. Each one was put here because dropping it was measured
 * to put a painted plain frame back.
 *
 *   1. `warm(true)`. Streamdown's highlighted body asks the code plugin for tokens and renders
 *      its own plain fallback whenever the plugin answers `null`. The plugin answers `null` only
 *      while a grammar is still loading; once the grammar is in hand it tokenizes and returns in
 *      the same call. Warming first is what turns the render below into a cache hit.
 *   2. The inner `flushSync(latch)`. A state update scheduled normally lands after the next paint,
 *      which is the whole defect. This commits the swap from shell to block inside this task.
 *   3. The inner `flushSync(poke)`, INSIDE an outer `flushSync`. Step 2 does not produce the
 *      COLOURED commit: `HighlightedCodeBlockBody` starts from `useState(raw)` and asks for its
 *      tokens from a PASSIVE effect, so every newly mounted code block renders unhighlighted once.
 *      On the shipped default that render happens at thread mount, off screen; deferred, it
 *      happens wherever the latch happens, and it is a real plain frame on the reader's screen.
 *
 * WHY THE POKE, AND WHY THE NESTING. Both follow from how react-dom flushes, and an empty
 * `flushSync(() => {})` -- the obvious thing to write -- does neither:
 *
 *   - React runs pending passive effects from `performSyncWorkOnRoot`, which it only reaches when
 *     there is sync work waiting. A flush with nothing to do never enters the work loop, so it
 *     never runs the effect that colours the block. `poke` is that work: a state write on a fence
 *     that has already latched, whose own effects all early-return, so it re-renders the one or
 *     two blocks just swapped in and changes nothing else.
 *   - `flushSync` restores React's update priority BEFORE it performs the flush. So the update the
 *     passive effect schedules is resolved against the ambient priority, and a scroll is
 *     continuous rather than discrete, which means it is NOT flushed synchronously and lands on
 *     the next frame after all. Nesting fixes exactly that: the outer call holds the priority at
 *     discrete for the whole of its body, and the inner flushes perform their work inside that
 *     body, so the effect's update is discrete too and goes out with them.
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
 * The observers below carry one root height of lookahead (`REACH_MARGIN`), and the pre-paint gate
 * runs only when this component renders. Neither covers a jump: the viewport can move further
 * than the lookahead in one step, with no React render in between, and an IntersectionObserver
 * record is delivered one or more frames AFTER the browser has painted. Measured in Chromium at
 * the 100K rung, 16 seeded jumps: 8 of them painted 3 to 4 frames of plain code, 50 to 190 ms.
 *
 * WHY A SCROLL LISTENER IS ENOUGH, AND WHEN IT PAINTS. Scroll events are dispatched in the "run
 * the scroll steps" of the same update-the-rendering pass that will paint the new position, and
 * that step runs BEFORE animation-frame callbacks and before style, layout and paint. A
 * `flushSync` from here is therefore part of the frame the reader is about to see, not the one
 * after it. A programmatic `scrollTo`, a wheel, a drag and a keyboard jump all arrive down this
 * one path.
 *
 * THE THRESHOLD IS DERIVED, NOT TUNED. `REACH_MARGIN` grows the root's band by one root height in
 * each direction, so after a scroll of `d` the newly visible strip sits at `[d, d + h]` measured
 * from the old top, and that is inside the old band exactly when `d <= h`. A scroll of at most one
 * root height can therefore only reveal fences the observers had already reached; anything further
 * can reveal one they had not. So the pass below runs when, and only when, the lookahead provably
 * did not cover the movement, and a continuous scroll pays one subtraction per event.
 *
 * ONE LISTENER, NOT ONE PER FENCE. Scroll events do not bubble, but they do capture, so a single
 * capturing listener on the document sees scrolling on every element including nested reasoning
 * panes. It is attached when the first fence registers and removed when the last one latches, so
 * a thread with nothing left to defer carries no scroll cost at all.
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
  // An unseen scroller has no previous position to difference, so its first event is treated as a
  // jump. That costs one pass, once per scroller, and never assumes a movement was small.
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
 * PRINT, which is the one gesture that puts every deferred fence on the page at once.
 *
 * Colour is the only thing deferral costs a printed page: the shell holds a live text node, so it
 * prints complete, in order, correctly laid out and in the right font. But a printed page that
 * loses the colouring on the fences the reader never scrolled past is a defect the reader can see
 * and keep, and the printed window does not even coincide with the reader's -- a print lays the
 * document out at PAPER width while the scroll offset carries across as a raw pixel count, so the
 * page lands several fences away from where the reader was. Measured at Letter against a 1280 px
 * window: 1 to 2 deferred fences on every sampled page, and a 6.6% pixel difference against the
 * same page printed with the flag off. Matching the paper to the window removes it, which is what
 * identifies the cause -- and is exactly why chasing the window is the wrong fix. The whole
 * document is on the page, so the whole document is what has to be highlighted.
 *
 * WHY AN EARLIER ATTEMPT AT THIS FAILED, since it was measured and reported as impossible. It
 * latched every fence from `beforeprint` with `flushSync` and the blocks did swap, but 53 of 56
 * of them printed on streamdown's raw fallback: the swap alone renders the block UNHIGHLIGHTED,
 * because the highlighted body asks the plugin for tokens from a passive effect and the plugin
 * answers `null` until the grammar is loaded. `latchNow` closes both halves -- it warms the tokens
 * first, and it flushes the colouring render as well as the swap -- and `warmGrammars` below
 * removes the remaining asynchrony by making sure a grammar is never what is missing at the moment
 * the snapshot is taken.
 *
 * BOTH DOORS. `beforeprint` covers Ctrl+P and the print menu. A headless `page.pdf()` and
 * DevTools' print emulation change the media query without ever firing it, and a PDF export is
 * exactly the path a reader uses to keep a copy.
 *
 * Still one way only: `printed` never goes back to false, so a print permanently upgrades the
 * thread rather than putting it back afterwards. Reverting on `afterprint` would be the
 * bidirectional edge this whole design exists to avoid.
 */
let printed = false;

const upgradeEverythingForPrint = (): void => {
  if (printed) return;
  printed = true;
  latchNow([...unreached]);
};

/*
 * GRAMMARS, WARMED AT IDLE. NOT TOKENS.
 *
 * `warm(false)` asks the plugin to highlight an EMPTY string in this fence's language, which
 * loads the grammar and tokenizes nothing. That is the whole of it: no fence is tokenized, no
 * span is mounted, and the document stays exactly the size deferral made it. What it buys is that
 * the synchronous path in `latchNow` can never be defeated by a grammar that happens to still be
 * loading -- on a jump into a language the reader has not met yet, and on a print, where there is
 * no later frame to correct in.
 *
 * Deduplicated by language and scheduled at idle, so a thread whose fences are all one language
 * pays one grammar load and a thread with nothing deferred pays nothing.
 */
const grammarsWarmed = new Set<string>();
let warmScheduled = false;

const warmGrammars = (): void => {
  warmScheduled = false;
  for (const gate of unreached) {
    const language = gate.language ?? "text";
    if (grammarsWarmed.has(language)) continue;
    grammarsWarmed.add(language);
    gate.warm(false);
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

/** Is `node` inside `scroller`'s box grown by one of its own heights, the observer's margin? */
const inBand = (node: HTMLElement, scroller: HTMLElement | null): boolean => {
  const bounds = scroller?.getBoundingClientRect();
  const top = bounds ? bounds.top : 0;
  const height = bounds ? bounds.height : window.innerHeight;
  const rect = node.getBoundingClientRect();
  return rect.bottom > top - height && rect.top < top + height * 2;
};

/**
 * @param enabled  false on the shipped default, where this hook must cost nothing at all: no
 *                 state is ever written, no observer is built and no layout is read.
 * @param streaming  the fence is still being written. It is highlighted while it streams AND it
 *                 latches, so that finishing cannot take the highlighting back.
 * @param language  the fence's language token, used only to load one grammar per language rather
 *                 than one per fence. `null` is the unknown-language case and warms plain text.
 * @param warm  drive the highlighter over this fence: `true` for its tokens, `false` for its
 *                 grammar alone. Called from a jump and from a print, where the swap has to be
 *                 coloured inside the same task; see `latchNow`. Held in a ref rather than taken
 *                 as an effect dependency so that a caller who does not memoize it cannot rebuild
 *                 every observer in the thread on every render.
 */
export function useFenceReached(
  host: RefObject<HTMLElement | null>,
  enabled: boolean,
  streaming: boolean,
  language: string | null,
  warm: (tokens: boolean) => void,
): boolean {
  const [latched, setLatched] = useState(false);
  // Bumped when the resolved scrolling ancestor stops being one, which rebuilds the gates below
  // against the element that clips this fence now. Never read for anything else.
  const [generation, setGeneration] = useState(0);
  const reached = !enabled || !CAN_OBSERVE || streaming || latched || printed;
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

    // The same two elements, handed to the register so a jump and a print can ask the same two
    // questions of this fence without a render and without walking the ancestor chain again. The
    // register is rebuilt with the observers, so a rebind can never leave a stale root behind in
    // one gate and not the other.
    const registered: FenceGate = {
      node,
      near,
      outer,
      language,
      warm: (tokens) => warmRef.current(tokens),
      latch: () => setLatched(true),
      // `generation` is bumped rather than a state of its own being added: this fence has just
      // latched, so every effect keyed on it early-returns and the bump costs one render of one
      // already-upgraded block. See `latchNow` for why React needs the work.
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
