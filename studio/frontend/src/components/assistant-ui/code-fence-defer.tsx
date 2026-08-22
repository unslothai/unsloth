// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import {
  memo,
  type RefObject,
  useEffect,
  useLayoutEffect,
  useState,
} from "react";

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
 * Three states, not two.
 *
 *   "off"        ship default. Every fence is highlighted at mount, as today.
 *   "defer"      the change: an unreached fence is a plain shell and is never tokenized.
 *   "tokenize"   MEASUREMENT ONLY. An unreached fence is the same plain shell, but the
 *                highlighter is still driven over its source and the result thrown away.
 *
 * `tokenize` exists to answer one question and is not a shipping mode. `defer` removes two
 * things at once: the spans from the document, and the tokenizer work that produces them. An
 * improvement seen under `defer` alone cannot say which of the two paid for it. `tokenize`
 * holds the DOM at `defer`'s size and puts only the tokenizer work back, so the gap between
 * `tokenize` and `defer` is the tokenizer's contribution and the gap between `off` and
 * `tokenize` is the document's.
 */
export type FenceMode = "off" | "defer" | "tokenize";

export const fenceMode = (): FenceMode => {
  const runtime = (globalThis as Record<string, unknown>)
    .__UNSLOTH_DEFER_FENCE_HIGHLIGHT__;
  const raw =
    typeof runtime === "string"
      ? runtime
      : runtime === true
        ? "defer"
        : runtime === false
          ? "off"
          : readBuildFlag();
  return raw === "1" || raw === "defer"
    ? "defer"
    : raw === "tokenize"
      ? "tokenize"
      : "off";
};

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
 * PRINT: NOT HANDLED, and measured rather than assumed.
 *
 * A print renders the whole document, so every deferred fence is on the page whether or not the
 * reader reached it. Colour is the only thing deferral costs there: the text is a live text node,
 * so it prints complete, in order, correctly laid out and in the right font.
 *
 * There WAS a `beforeprint` path here that latched every mounted fence, warmed each language's
 * grammar at idle and flushed twice. It has been removed because it did not work, and the
 * measurement is the reason. Synchronous dispatch and read in one task, 100K rung, cold open:
 *
 *   delay after open   flag off: blocks on the raw fallback   flag on
 *   1.5 s              56 of 56                               56 of 56
 *   3 s and beyond     0                                      53 of 56
 *
 * The shipped build has a real window of about three seconds and is fully highlighted after it.
 * With the flag on, 53 of 56 blocks stayed on streamdown's raw fallback at every delay out to
 * twenty seconds. Neither a warm grammar nor a second `flushSync` makes a lazily imported,
 * effect-driven highlighter produce tokens inside the synchronous print task, because
 * `beforeprint` returns and the browser snapshots before any of that can land.
 *
 * So the honest position is that this is a stated limitation, not a solved problem: with the flag
 * on, printing a long thread prints unreached fences without syntax colour. Keeping the machinery
 * would have meant maintaining a module-global claim, two bookkeeping sets and an idle scheduler
 * for a benefit that was measured at zero.
 */

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
 */
export function useFenceReached(
  host: RefObject<HTMLElement | null>,
  enabled: boolean,
  streaming: boolean,
): boolean {
  const [latched, setLatched] = useState(false);
  // Bumped when the resolved scrolling ancestor stops being one, which rebuilds the gates below
  // against the element that clips this fence now. Never read for anything else.
  const [generation, setGeneration] = useState(0);
  const reached = !enabled || !CAN_OBSERVE || streaming || latched;

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
    };
  }, [reached, host, generation]);

  return reached;
}

export const DeferredFenceShell = memo(FenceShell);
