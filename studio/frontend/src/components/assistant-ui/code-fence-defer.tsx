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
import { flushSync } from "react-dom";

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
          <code>{trimTrailingNewlines(source)}</code>
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
 * PRINT. A print or a PDF export renders the WHOLE document at once, so every fence is on the
 * page whether or not the reader ever scrolled to it, and a fence that prints without its
 * colours is a defect the reader can see and keep. Selection, clipboard and find-in-page all
 * survive deferral on their own because the text is a live text node; colour does not.
 *
 * `beforeprint` fires before the print snapshot is taken, but a React state update scheduled
 * from it lands after. `flushSync` is what makes the upgrade part of the same task, so the
 * document the printer serialises is the upgraded one.
 *
 * LATCHING IS NOT ENOUGH ON ITS OWN, and this is the part that was wrong before. The swap commits
 * `<Block>`, but `<Block>` renders `<Suspense>` around a `React.lazy` import of streamdown's
 * highlighted body, and that body asks the plugin for tokens from an EFFECT. So a snapshot taken
 * in the same task can still catch the plain Suspense fallback whenever the highlighter chunk or
 * the fence's grammar was never loaded. Three things together close it:
 *
 *   1. one fence per document renders eagerly, which loads the lazy chunk (see `claimChunkWarm`);
 *   2. every language present in the thread has its grammar warmed at idle, with a one-character
 *      source, so no real fence is tokenized for it (see `warmGrammar` in markdown-text);
 *   3. a SECOND, empty `flushSync` below, because `flushSync` flushes pending passive effects
 *      before it renders. The first flush swaps the shells for blocks and schedules streamdown's
 *      own highlight effect; the second is what lets that effect's state update land.
 *
 * With the grammar warm the plugin returns tokens synchronously, so step 3 has something real to
 * commit rather than another pending promise.
 *
 * SCOPED TO WHAT IS MOUNTED. Each mounted fence latches itself; nothing is recorded at module
 * scope. A module-global "we have printed" flag would be read by every fence mounted afterwards,
 * so one Ctrl+P would silently switch deferral off for the rest of the session, including in
 * conversations opened later. Upgrading a thread the reader printed is the intent; disabling the
 * feature for the process is not.
 *
 * Still one-way: a latched fence stays latched. Nothing listens for `afterprint`, because putting
 * a printed thread back to shells is exactly the bidirectional edge this design exists to avoid.
 */
const printListeners = new Set<() => void>();

const upgradeEverythingForPrint = (): void => {
  if (printListeners.size === 0) return;
  flushSync(() => {
    for (const notify of printListeners) notify();
  });
  // The second flush. See the note above: this one renders nothing and exists only so that the
  // passive effect the first flush scheduled -- streamdown asking the plugin for tokens -- is
  // flushed before the printer serialises the page.
  flushSync(() => {});
};

/*
 * THE LAZY CHUNK, warmed by letting exactly one fence per document render eagerly.
 *
 * streamdown's highlighted body arrives through `React.lazy`, so nothing can render highlighted
 * until that chunk has been fetched, and the only thing that fetches it is a `<Block>` mounting.
 * On a thread where no fence is anywhere near the viewport at mount, deferral would mean the
 * chunk is never requested at all, and the first thing that needs it would be a print.
 *
 * So the first fence to mount pays for it. ONE fence, once per document, and it is monotonic: the
 * claim is never released, so this can only ever make one fence expensive and never a second.
 * That is a different thing from the module-global print flag that was removed here, which fed
 * into every later fence's `reached` decision and switched the feature off for the session.
 */
let chunkWarmClaimed = false;

/** True for the first caller only. The caller that gets it renders eagerly. */
export const claimChunkWarm = (): boolean => {
  if (chunkWarmClaimed) return false;
  chunkWarmClaimed = true;
  return true;
};

if (typeof window !== "undefined" && typeof window.addEventListener === "function") {
  window.addEventListener("beforeprint", upgradeEverythingForPrint);
  // Headless Chromium's `page.pdf()` and DevTools' print emulation change the media query
  // without ever firing `beforeprint`, and a PDF export is exactly the path a reader uses to
  // keep a copy. Both doors are covered.
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

  useEffect(() => {
    if (reached) return;
    const notify = () => setLatched(true);
    printListeners.add(notify);
    return () => {
      printListeners.delete(notify);
    };
  }, [reached]);

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
   * Still one-way: this can only ever latch true.
   */
  useLayoutEffect(() => {
    if (reached) return;
    const node = host.current;
    if (!node) return;
    const bounds = scrollerOf(node)?.getBoundingClientRect();
    const top = bounds ? bounds.top : 0;
    const height = bounds ? bounds.height : window.innerHeight;
    // The same one-viewport slack the observer's rootMargin uses, so the two doors agree on what
    // "reached" means and a fence cannot latch through one and not the other.
    const rect = node.getBoundingClientRect();
    if (rect.bottom > top - height && rect.top < top + height * 2) {
      // The cascading render this warns about is the POINT: it is what keeps the plain shell off
      // the screen. It happens at most once per fence, only for the one or two fences that are
      // already on screen at mount, and the alternative is the 2 to 3 painted frames of
      // unhighlighted code this replaced.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setLatched(true);
    }
  }, [reached, host]);

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
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          observer.disconnect();
          setLatched(true);
        }
      },
      { root: scrollerOf(node), rootMargin: REACH_MARGIN },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [reached, host]);

  return reached;
}

export const DeferredFenceShell = memo(FenceShell);
