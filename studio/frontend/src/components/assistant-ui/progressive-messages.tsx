// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// React glue for the widen-only mount window. The state machine, and why this is not a virtualizer,
// are in progressive-mount-controller.ts.
//
// This replaces `<ThreadPrimitive.Messages>{renderThreadMessage}</ThreadPrimitive.Messages>` with
// an equivalent map over a bounded index range: upstream renders `MessageByIndexProvider ->
// RenderChildrenWithAccessor -> children({ message })`, this renders `MessageByIndexProvider ->
// children`, dropping an accessor wrapper the thread's slot does not use and that emits no DOM.
//
// The replacement is permanent, not a mode the thread leaves once the window closes: swapping trees
// would change the element type at that position, so React would unmount and rebuild every message
// on the convergence commit, which is the cost being avoided.
//
// It takes the SAME propless slot #9042 gave `ThreadPrimitive.Messages`, for the same reason.
// `renderMessage()` returns one shared element object, so a commit that changes the row count hands
// React an identical element for every pre-existing row and React skips those subtrees. A
// `components` object would allocate fresh props per row per commit, so every delete would
// re-render every body, action bar and tooltip: the regression #9042 removed.

import {
  MessageByIndexProvider,
  useAui,
  useAuiState,
} from "@assistant-ui/react";
import {
  type FC,
  type ReactElement,
  type RefObject,
  memo,
  startTransition,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  type AnchorSample,
  type MountWindow,
  anchorCorrection,
  initialWindow,
  widen,
} from "@/components/assistant-ui/progressive-mount-controller";
import {
  type ThreadMessageRole,
  rendersAsRow,
} from "@/components/assistant-ui/thread-message-slot";
import { useAdjustForContentInsertedAbove } from "@/components/assistant-ui/use-intent-aware-autoscroll";

/**
 * Every mount window currently withholding rows. Module-level rather than context: the consumers are
 * imperative DOM readers (screenshot capture, the #9016 harness census) not necessarily in the React
 * tree, and a thread can be mounted twice at once (the Compare panes).
 */
interface ActiveCompleter {
  complete: () => Promise<void>;
  /** The scroll viewport these withheld rows belong to, so a caller can ask for only its own. */
  viewportRef: RefObject<HTMLElement | null>;
}

const activeCompleters = new Set<ActiveCompleter>();

/**
 * Force every in-flight progressive mount to completion, resolving once the commit has painted.
 *
 * Anything reading the thread out of the DOM rather than the runtime store must await this, or a
 * read during the few frames a long thread takes to converge silently sees a short conversation.
 * Store readers need nothing: the window gated rendering, never the store.
 */
export const PROGRESSIVE_MOUNT_SEARCH_MS = 400;

export async function completeProgressiveMounts(
  /**
   * Force only the threads whose viewport this accepts. Every one of them by default. Find-in-page
   * passes one: the shell keeps each workspace mounted and marks the off-route ones `inert`, and
   * mounting a retained conversation's withheld rows serves nobody looking at another route.
   */
  wants?: (viewport: HTMLElement | null) => boolean,
): Promise<void> {
  const wanted = () =>
    wants === undefined
      ? [...activeCompleters]
      : [...activeCompleters].filter((entry) => wants(entry.viewportRef.current));
  // Two waits, failing in opposite directions.
  //
  // Draining is easy: ask every completer that exists to finish. Reading the set ONCE is not enough,
  // since completers register from a layout effect and a caller in the same task as the thread
  // opening finds it empty (measured: returned in 0.1ms, 16 of 220 rows two frames later).
  //
  // The hard half is that "empty" is ambiguous: a settled thread and a thread still loading history
  // (about 160ms on a cold open) look alike, so returning at the first empty reading resolves before
  // any row exists. An empty set is therefore only believed after SEARCH_MS of looking; a set that
  // was non-empty and has drained is believed immediately.
  //
  // So a caller on a settled thread pays SEARCH_MS, the deliberate direction to be wrong in: the
  // alternative is a screenshot or export of a conversation that is not there yet.
  const deadline = Date.now() + PROGRESSIVE_MOUNT_SEARCH_MS;
  let observed = false;
  for (;;) {
    const pending = wanted();
    if (pending.length > 0) {
      observed = true;
      await Promise.all(pending.map((entry) => entry.complete()));
    }
    await new Promise<void>((resolve) =>
      requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
    );
    // The filtered set, not the whole one: a completer this caller does not want would otherwise
    // hold the loop open forever.
    if (wanted().length === 0 && (observed || Date.now() >= deadline)) {
      return;
    }
  }
}

/** True while any thread is still widening. Exposed for tests and diagnostics. */
export function hasPendingProgressiveMounts(): boolean {
  return activeCompleters.size > 0;
}

/**
 * Hand native scroll anchoring back, leaving no trace of having taken it.
 *
 * `removeProperty` empties the inline declaration but keeps the attribute, so the settled viewport
 * carried `style=""` where the shipping one carries no `style` at all. That is inert, but it was the
 * ONLY difference a whole-document structural digest could find between this branch and its merge
 * base at 100K and 300K (45,014 lines, one changed line, on a null scoring exactly zero), and a
 * byte-identical converged document is a claim worth being able to make rather than footnote.
 *
 * Only an empty declaration is cleared, so any other inline style a caller has set survives.
 */
function restoreScrollAnchoring(viewport: HTMLElement | null): void {
  if (!viewport) return;
  viewport.style.removeProperty("overflow-anchor");
  if (viewport.getAttribute("style") === "") viewport.removeAttribute("style");
}

/**
 * The row to hold still: the first one the reader can actually SEE, not the first in the list.
 *
 * Widening prepends above everything, so for a widening any row will do. A RELAYOUT will not: a row
 * growing between the topmost row and the fold moves the reader while leaving the topmost row where
 * it was. Measured with a 600px height change injected into one row above a detached reader with the
 * window open: 600px of movement, reported as 0 by a topmost-row anchor.
 */
function pickAnchorRow(viewport: HTMLElement): Element | null {
  const fold = viewport.getBoundingClientRect().top;
  const rows = viewport.querySelectorAll("[data-role]");
  let anchor: Element | null = rows.item(rows.length - 1);
  for (const row of rows) {
    if (row.getBoundingClientRect().bottom > fold) {
      anchor = row;
      break;
    }
  }
  if (!anchor) return null;
  // Then descend to the fold. A row can be taller than the viewport (a long answer with images and
  // code usually is), so a reader partway through one has the row's top ABOVE them: a block earlier
  // in that message growing moves everything they see while the row's top stays put, and a whole-row
  // anchor reports zero. The first descendant reaching the fold does not. Bounded because that block
  // is a few levels down, and this runs once per re-pick, not per frame.
  //
  // Known residual: if the fold lands inside a tall LEAF block (a paragraph with no element
  // children) the descent stops with its top above the reader, so lines reflowing inside it move
  // what they see without moving the block. Closing that needs a caret point at the fold, spelled
  // differently on all three engines, costing a Range and a rect every frame, and not itself stable
  // across the reflow it would measure.
  for (let depth = 0; depth < 8; depth += 1) {
    if (anchor.getBoundingClientRect().top >= fold) break;
    let next: Element | null = null;
    for (const child of anchor.children) {
      if (child.getBoundingClientRect().bottom > fold) {
        next = child;
        break;
      }
    }
    if (!next) break;
    anchor = next;
  }
  return anchor;
}

function sampleAnchor(viewport: HTMLElement, element: Element): AnchorSample {
  return {
    viewportOffset:
      element.getBoundingClientRect().top -
      viewport.getBoundingClientRect().top,
    scrollTop: viewport.scrollTop,
    maxScrollTop: viewport.scrollHeight - viewport.clientHeight,
  };
}

/**
 * An anchor plus the row it sits in, sampled together. The row is the fallback: pickAnchorRow
 * descends to the fold, so the anchor is often the `<pre>` Streamdown replaces when Shiki finishes,
 * and a replaced node takes its baseline with it. A `[data-role]` row is never replaced in place.
 *
 * Being an ANCESTOR, the row does not move when the replacement changes height inside it, so that
 * height change goes uncorrected. That is the better of the two available errors: an anchor BELOW
 * the replaced block would also see it grow DOWNWARD past the fold, which moves nothing the reader
 * sees, trading an error in a rare case for one in the common case. Either way the residual is one
 * code block's height change, for a detached reader, while a window is open.
 */
function holdAnchor(viewport: HTMLElement, element: Element): HeldAnchor {
  const row = element.closest("[data-role]");
  return {
    element,
    sample: sampleAnchor(viewport, element),
    row,
    rowSample: row ? sampleAnchor(viewport, row) : null,
  };
}

type HeldAnchor = {
  element: Element;
  sample: AnchorSample;
  row: Element | null;
  rowSample: AnchorSample | null;
};

/** True while `element` still covers some of what the reader can see. */
function isAnchorVisible(viewport: HTMLElement, element: Element): boolean {
  const box = element.getBoundingClientRect();
  const fold = viewport.getBoundingClientRect().top;
  return box.bottom > fold && box.top < fold + viewport.clientHeight;
}

function useProgressiveMountWindow(
  count: number,
  resetKey: string | undefined,
  viewportRef: RefObject<HTMLElement | null>,
): MountWindow {
  const aui = useAui();
  const adjustForContentInsertedAbove = useAdjustForContentInsertedAbove();

  // Imperative rather than useAuiState: a reactive subscription would re-render this component,
  // and so rebuild the whole row array, on every run start and stop.
  const isRunningNow = useCallback(
    () => aui.thread().getState().isRunning === true,
    [aui],
  );

  // Which indices actually paint, read IMPERATIVELY off the store for the same reason isRunning is:
  // a `useAuiState` selector touching the messages array runs on every store write, and the
  // composer is controlled off store state, so that would walk the thread once per keystroke and
  // once per streamed token. This is called only where a window is armed -- mount, thread switch,
  // and the commit that first brings messages in -- so it costs one snapshot each time, not one
  // per render. See thread-research-presence.ts for the same hazard.
  const renderableAt = useCallback(() => {
    const messages = aui.thread().getState().messages as ReadonlyArray<{
      role?: ThreadMessageRole;
      composer?: { isEditing?: boolean };
    }>;
    return (index: number) => {
      const message = messages[index];
      // A shape this cannot read counts as renderable, which degrades to the count-based window
      // that shipped before. The opposite default would silently stop windowing altogether.
      if (message?.role == null) return true;
      return rendersAsRow(message.role, message.composer?.isEditing === true);
    };
  }, [aui]);

  const [mountWindow, setMountWindow] = useState<MountWindow>(() =>
    initialWindow(count, isRunningNow(), renderableAt()),
  );

  // Re-arm per thread, adjusting state during render as React documents, so the previous thread's
  // window can never gate the new one for a frame.
  //
  // UNEXERCISED today: `<GeneratedImageOverlayProvider key={runtimeThreadId}>` in thread.tsx wraps
  // this subtree, so a thread switch rebuilds this component and the useState initialiser above
  // re-arms instead (measured across in-app switches: zero arm records from this branch). Kept
  // because it is what makes this component correct on its own terms; remove that ancestor key and
  // it is the only thing between a thread switch and the previous window gating the next thread.
  const [previousKey, setPreviousKey] = useState(resetKey);
  if (previousKey !== resetKey) {
    setPreviousKey(resetKey);
    setMountWindow(initialWindow(count, isRunningNow(), renderableAt()));
  }

  // The commit that puts the FIRST rows into an empty tree.
  //
  // Mounting is not when the app learns the thread length. `threadListItem.id` flips as soon as
  // switchToThread resolves and that remounts this subtree, but the messages arrive only when the
  // history adapter settles, a Dexie read plus two HTTP calls later. So on a cold open the useState
  // above runs at `count === 0` and declines, and without this the whole thread lands in one
  // unbounded commit: the stall this file exists to remove. Measured opening a 220-message thread:
  // mount at count 0, count 220 about 160ms later, same resetKey, first paint carrying all 220.
  //
  // Gated on the PREVIOUS count being zero rather than on crossing MIN_PROGRESSIVE_MESSAGES: a tree
  // that never rendered a row has no scroll position to lose, so arming here cannot move a reader.
  // Appending a 40th message to a 39-message thread crosses the threshold too, and would window a
  // conversation the reader is in the middle of.
  const [previousCount, setPreviousCount] = useState(count);
  if (previousCount !== count) {
    setPreviousCount(count);
    // Same call as the mount path, so a run in flight suppresses the window here too.
    if (previousCount === 0 && count > 0) {
      setMountWindow(initialWindow(count, isRunningNow(), renderableAt()));
    }
  }

  // A thread that shrank at or past the live window's start is reconciled HERE, during render, not
  // at the next widening. Rendering every row (below) is only half: the STATE still says start 204,
  // and a widen against a 100-message thread would produce start 68 and unmount the 68 rows this
  // render just mounted. Before this existed the stale state painted an empty column for 2 to 8
  // frames.
  if (mountWindow != null && mountWindow.start >= count) {
    setMountWindow(null);
  }

  // A run starting mid-widening drops the window immediately: streaming writes the same scroll
  // position widening does, and a reply must never commit into a tree that has not reached it. The
  // remaining rows land in one commit, which is what happened before this change anyway.
  const threadIsRunning = useAuiState(({ thread }) => thread.isRunning);
  useEffect(() => {
    if (threadIsRunning && mountWindow != null) setMountWindow(null);
  }, [threadIsRunning, mountWindow]);

  // The row held still across a widening commit, plus the scroll state it was held against, both
  // captured against the SCROLL CONTAINER rather than the window (see AnchorSample).
  // anchorCorrection turns the pair into a scrollTop delta in document space, assuming nothing else
  // moved scrollTop, which is what disarming native anchoring below is for.
  const anchorRef = useRef<
    | ({
        element: Element;
        row: Element | null;
        rowSample: AnchorSample | null;
      } & AnchorSample)
    | null
  >(null);
  /** The row the idle sampler below holds still, and where it last saw it. */
  const idleRef = useRef<HeldAnchor | null>(null);

  const captureAnchor = useCallback(() => {
    const viewport = viewportRef.current;
    // Disarm native scroll anchoring HERE, immediately before the capture it would invalidate. The
    // correction is document space and only holds while nothing else moves scrollTop, so the
    // browser must be out of the loop before the first sample, not merely soon.
    //
    // A layout effect is too early: this component is a DESCENDANT of the viewport, so on the
    // mounting commit React runs this subtree's layout effects before the viewport's ref callback,
    // `viewportRef.current` is null and the style is silently skipped. Measured on that version,
    // computed `overflow-anchor` stayed `auto` until the FIRST widening at +803ms, so that widening
    // applied the document-space correction on top of the browser's: a reader who scrolled 4000px
    // was left 776px short. This callback runs from a requestAnimationFrame, after a paint, so the
    // ref is populated.
    if (viewport) viewport.style.setProperty("overflow-anchor", "none");
    const anchor = viewport ? pickAnchorRow(viewport) : null;
    // The enclosing row is a fallback: the anchor is often the `<pre>` Streamdown replaces when
    // Shiki finishes. Without a fallback the correction is dropped, and with anchoring off a whole
    // chunk of prepended rows moves the reader permanently. A row is never replaced in place.
    const row = anchor?.closest("[data-role]") ?? null;
    anchorRef.current =
      viewport && anchor
        ? {
            element: anchor,
            ...sampleAnchor(viewport, anchor),
            row,
            rowSample: row ? sampleAnchor(viewport, row) : null,
          }
        : null;
  }, [viewportRef]);

  useEffect(() => {
    if (mountWindow == null) return;
    const frame = requestAnimationFrame(() => {
      // Re-check: a run can start between the commit and this frame.
      if (isRunningNow()) {
        setMountWindow(null);
        return;
      }
      captureAnchor();
      startTransition(() => {
        setMountWindow((current) => widen(current, count));
      });
    });
    return () => cancelAnimationFrame(frame);
  }, [mountWindow, count, captureAnchor, isRunningNow]);

  // mountWindow is the TRIGGER, not a value the body reads, which is why the rule cannot see it:
  // this effect must run in the commit that widened the window, before it paints.
  // biome-ignore lint/correctness/useExhaustiveDependencies: mountWindow is a trigger, see above
  useLayoutEffect(() => {
    const captured = anchorRef.current;
    anchorRef.current = null;
    const viewport = viewportRef.current;
    if (!captured || !viewport) return;
    // The anchor, or its row if the anchor has been replaced since the capture.
    const [element, baseline] = captured.element.isConnected
      ? [captured.element, captured as AnchorSample]
      : captured.row?.isConnected && captured.rowSample
        ? [captured.row, captured.rowSample]
        : [null, null];
    if (!element || !baseline) return;
    const shift = anchorCorrection(baseline, sampleAnchor(viewport, element));
    // Called on EVERY widening commit, including no-op ones: a zero correction still resyncs the
    // hook's scroll bookkeeping, or the scroll event native anchoring fires arrives as a downward
    // scroll the reader did not make. See adjustForContentInsertedAbove.
    adjustForContentInsertedAbove(shift ?? 0);
    // Hand the idle sampler a POST-correction baseline rather than nulling it. Nulling made the next
    // frame re-pick and re-base, so a block landing in between was folded into the new baseline and
    // never corrected: with anchoring off the reader kept the displacement.
    idleRef.current = holdAnchor(viewport, element);
  }, [mountWindow, adjustForContentInsertedAbove, viewportRef]);

  // Content above the reader also moves for reasons that are not a widening, and nothing else
  // watches for it while the window is open: Streamdown replaces a `<pre>` when Shiki finishes,
  // KaTeX resizes a formula, an image lands. Native anchoring absorbs those and is disabled for the
  // duration, so compensation has to cover the whole interval, not just widening commits. The
  // autoscroll hook's mutation path pins a FOLLOWING reader and leaves a detached one alone.
  //
  // Measured when written, injecting a 600px height change into one row above a detached reader with
  // the window open: without this the reader moved the full 600px, against 0px on the merge base.
  //
  // RE-MEASURED SINCE, AND IT NO LONGER REPRODUCES ITS OWN MEASUREMENT: ablating this whole effect
  // leaves grow 600, grow 3600 and shrink 1200 identical to head within noise. The cause is
  // declaration ORDER, not the logic: the widening capture effect above registers its rAF first, so
  // `anchorRef.current` is non-null on every frame the window is open and this body returns at its
  // first line. Instrumented, 0 of 1110 frames across chromium, firefox and webkit got past it, and
  // the widening layout effect absorbs the reflow instead. Confirmed causally by swapping only the
  // two effect declarations, after which this body runs about 5 times per repetition. Anyone
  // reordering these two effects is switching this on.
  //
  // Kept because the widening path it defers to has no visibility test, so this is the only thing
  // that would catch a reflow on a frame with no widening pending. The anchor is the first row the
  // reader can SEE, re-picked only when gone or scrolled out of view, so per frame this costs one
  // rect and one scrollTop read.
  useEffect(() => {
    if (mountWindow == null) {
      idleRef.current = null;
      return;
    }
    let frame = 0;
    const tick = () => {
      frame = requestAnimationFrame(tick);
      const viewport = viewportRef.current;
      // A pending widening capture owns the correction until its layout effect runs; stepping in
      // between would correct the same movement twice.
      if (!viewport || anchorRef.current) return;
      const held = idleRef.current;
      if (!held) {
        const element = pickAnchorRow(viewport);
        idleRef.current = element ? holdAnchor(viewport, element) : null;
        return;
      }
      // The anchor, or its row if replaced since the last frame. Re-picking instead would discard
      // the only pre-replacement sample and re-base AFTER the replacement's own reflow.
      const [element, baseline] = held.element.isConnected
        ? [held.element, held.sample]
        : held.row?.isConnected && held.rowSample
          ? [held.row, held.rowSample]
          : [null, null];
      if (!element || !baseline) {
        const replacement = pickAnchorRow(viewport);
        idleRef.current = replacement
          ? holdAnchor(viewport, replacement)
          : null;
        return;
      }
      // Scrolled past: hold something the reader can see instead. Tested by the row's box, not its
      // top offset, since a tall row just off screen can still have its top within a viewport of the
      // fold, and a reflow between it and the first visible row moves the reader while it stays put.
      if (!isAnchorVisible(viewport, element)) {
        // Re-pick in THIS frame rather than nulling and waiting: a baseline-free frame folds a
        // reflow above the newly visible content into the next baseline.
        const replacement = pickAnchorRow(viewport);
        idleRef.current = replacement
          ? holdAnchor(viewport, replacement)
          : null;
        return;
      }
      const shift = anchorCorrection(baseline, sampleAnchor(viewport, element));
      if (shift !== null) adjustForContentInsertedAbove(shift);
      // Re-based EVERY frame, no-ops included. Returning early on a no-op left the baseline's
      // scrollTop stale, and anchorCorrection's clamp term reads it, so a later shrink near the
      // bottom would subtract the wrong amount.
      idleRef.current = holdAnchor(viewport, element);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [mountWindow, adjustForContentInsertedAbove, viewportRef]);

  // Native scroll anchoring goes back on the moment the window closes, and on unmount, so a settled
  // thread is exactly the thread that shipped before this change. Turning it OFF happens at capture
  // time instead, for the ordering reason in captureAnchor.
  //
  // Why it must be off: left on, the browser moves scrollTop by the inserted height on some frames
  // and not others (never on any shipping Safari, and suppressed per frame elsewhere after a
  // programmatic scroll), and no in-frame measurement can tell which kind of frame it is in. Off,
  // scrollTop moves only when the reader or this code moves it, which anchorCorrection assumes.
  useLayoutEffect(() => {
    if (mountWindow != null) return;
    restoreScrollAnchoring(viewportRef.current);
  }, [mountWindow, viewportRef]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: unmount-only cleanup
  useLayoutEffect(() => {
    const viewport = viewportRef.current;
    return () => {
      restoreScrollAnchoring(viewport);
    };
  }, [viewportRef]);

  // Registered only while rows are actually withheld, so completeProgressiveMounts is free on the
  // settled thread that is the common case. A layout effect runs one phase earlier, narrowing (not
  // closing) the window in which a caller sees an empty completer set while rows are withheld.
  const isWithholding = mountWindow != null;
  const completionWaiters = useRef<Array<() => void>>([]);
  useLayoutEffect(() => {
    if (!isWithholding) return;
    const complete = () =>
      new Promise<void>((resolve) => {
        completionWaiters.current.push(resolve);
        setMountWindow(null);
      });
    const entry: ActiveCompleter = { complete, viewportRef };
    activeCompleters.add(entry);
    return () => {
      activeCompleters.delete(entry);
    };
  }, [isWithholding, viewportRef]);

  // Resolve on the commit that actually dropped the window, then after a paint. A bare two-frame
  // timer started at the call raced the commit it had just asked for: measured, it resolved with 16
  // of 220 rows still in the document on 4 of 5 WebKit and 2 of 5 Firefox runs.
  const flushCompletionWaiters = useCallback(() => {
    const waiters = completionWaiters.current;
    completionWaiters.current = [];
    for (const resolve of waiters) resolve();
  }, []);

  useEffect(() => {
    if (mountWindow != null || completionWaiters.current.length === 0) return;
    // The ref is emptied only when the frame fires, so a cancelled frame leaves the waiters for the
    // next commit or the unmount below rather than dropping them.
    const frame = requestAnimationFrame(() =>
      requestAnimationFrame(flushCompletionWaiters),
    );
    return () => cancelAnimationFrame(frame);
  }, [mountWindow, flushCompletionWaiters]);

  // A thread that goes away settles anything still waiting on it. Without this a DOM capture racing
  // a thread switch waited forever: the cleanup above removes the completer, but the promise it
  // already handed out is only resolved by the effect above, which no longer runs.
  useEffect(() => flushCompletionWaiters, [flushCompletionWaiters]);

  return mountWindow;
}

/**
 * The thread's message list. Drop-in for `ThreadPrimitive.Messages`, with a long thread's first
 * commit bounded to the tail and the rest arriving over the following frames. Memoized on component
 * identity exactly as upstream is, so a `Thread` re-render does not rebuild the row array.
 */
export const ProgressiveMessages: FC<{
  /** The thread's propless message slot. See thread-message-slot.ts. */
  renderMessage: () => ReactElement;
  /** Changing this re-arms the window. The runtime thread id. */
  resetKey: string | undefined;
  /** The scroll viewport these rows live in. Identity is stable, so it is not compared below. */
  viewportRef: RefObject<HTMLElement | null>;
}> = memo(
  ({ renderMessage, resetKey, viewportRef }) => {
    const count = useAuiState(({ thread }) => thread.messages.length);
    const mountWindow = useProgressiveMountWindow(count, resetKey, viewportRef);

    return useMemo(() => {
      if (count === 0) return null;
      // `start >= count` means the thread shrank under a live window (a bulk delete, or a switch to
      // a shorter thread), and clamping `start` to `count` would make this loop emit NOTHING.
      // Measured before this line: dropping a 220-message thread to 10 with the window at start=204
      // painted an empty column for 2 to 8 frames (37 to 146ms) on all three engines. `widen` heals
      // it a frame later but runs inside startTransition, so the blank persists. Drop the
      // restriction in the same commit: it can withhold nothing now.
      const first =
        mountWindow == null || mountWindow.start >= count
          ? 0
          : Math.max(mountWindow.start, 0);
      // One shared element for every row, which is what lets React skip unchanged rows. See header.
      const message = renderMessage();
      const rows: ReactElement[] = [];
      for (let index = first; index < count; index += 1) {
        rows.push(
          <MessageByIndexProvider key={index} index={index}>
            {message}
          </MessageByIndexProvider>,
        );
      }
      return <>{rows}</>;
    }, [count, mountWindow, renderMessage]);
  },
  (prev, next) =>
    prev.resetKey === next.resetKey &&
    prev.renderMessage === next.renderMessage,
);

ProgressiveMessages.displayName = "ProgressiveMessages";
