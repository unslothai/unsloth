// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// React glue for the widen-only mount window. The state machine it drives, and the reasoning for
// why this is not a virtualizer, are in progressive-mount-controller.ts.
//
// This replaces `<ThreadPrimitive.Messages>{renderThreadMessage}</ThreadPrimitive.Messages>` with
// an equivalent map over a bounded index range. `ThreadPrimitive.Messages` renders
// `MessageByIndexProvider -> RenderChildrenWithAccessor -> children({ message })`; this renders
// `MessageByIndexProvider -> children`, dropping the accessor wrapper that the thread's slot does
// not use. RenderChildrenWithAccessor emits no DOM, so the document is the same either way.
//
// That replacement is permanent rather than a mode the thread leaves once the window closes, and
// it has to be: rendering upstream's tree while settled and this one while windowed would change
// the element type at that position and React would unmount and rebuild every message on the
// convergence commit, which is precisely the cost being avoided. Rendering one of them throughout
// means the only thing that ever changes is how many rows the map emits.
//
// It takes the SAME propless slot #9042 gave `ThreadPrimitive.Messages`, and depends on it for the
// same reason. `renderMessage()` returns one shared element object, so a commit that changes the
// row count hands React an identical element for every row that already existed and React skips
// those subtrees. Passing a `components` object instead would allocate fresh props per row per
// commit and every delete would re-render every body, action bar and tooltip in the thread, which
// is the regression #9042 removed.

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
import { useAdjustForContentInsertedAbove } from "@/components/assistant-ui/use-intent-aware-autoscroll";

/**
 * Every mount window currently withholding rows. Module-level rather than context because the
 * consumers are imperative DOM readers (screenshot capture, the #9016 harness census) that are not
 * necessarily inside the React tree, and because a thread can be mounted more than once at a time:
 * the Compare panes each run their own Thread.
 */
const activeCompleters = new Set<() => Promise<void>>();

/**
 * Force every in-flight progressive mount to completion, resolving once the resulting commit has
 * painted.
 *
 * Anything that reads the thread out of the DOM rather than out of the runtime store must await
 * this first, or a read taken during the few frames a long thread takes to converge silently sees
 * a short conversation. Reading the store needs nothing: the window never gated the store, only
 * how much of it was rendered.
 */
export const PROGRESSIVE_MOUNT_SEARCH_MS = 400;

export async function completeProgressiveMounts(): Promise<void> {
  // Two things have to be waited for, and they fail in opposite directions.
  //
  // Draining is the easy half: a completer that exists is asked to finish and awaited. Reading
  // the set ONCE was not enough, because the completers register from a layout effect, so a
  // caller in the same task as the thread opening finds it empty. Measured on that version: the
  // call returned in 0.1ms and the document held 16 of 220 rows two frames later.
  //
  // The hard half is that "empty" is ambiguous. A settled thread and a thread whose history is
  // still loading both present an empty set, and on a cold open the load takes around 160ms, so
  // returning at the first empty reading resolves before a single row exists. So an empty set is
  // only believed after SEARCH_MS of looking, while a set that has been non-empty and has since
  // drained is believed immediately.
  //
  // That means a caller on a settled thread pays SEARCH_MS. That is the deliberate direction to
  // be wrong in: the alternative is a screenshot or an export of a conversation that is not
  // there yet, and anything reading the whole thread out of the DOM is already doing something
  // far more expensive than this.
  const deadline = Date.now() + PROGRESSIVE_MOUNT_SEARCH_MS;
  let observed = false;
  for (;;) {
    if (activeCompleters.size > 0) {
      observed = true;
      await Promise.all([...activeCompleters].map((complete) => complete()));
    }
    await new Promise<void>((resolve) =>
      requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
    );
    if (activeCompleters.size === 0 && (observed || Date.now() >= deadline)) {
      return;
    }
  }
}

/** True while any thread is still widening. Exposed for tests and diagnostics. */
export function hasPendingProgressiveMounts(): boolean {
  return activeCompleters.size > 0;
}

/**
 * The anchor row's position, measured against its scroll container rather than against the window,
 * so that the container moving does not read as content inserted above it. See AnchorSample.
 */
/**
 * The row to hold still: the first one the reader can actually SEE, not the first one in the list.
 *
 * That distinction is load-bearing and it cost two measurement rounds to find. Widening prepends
 * above everything, so for a widening any row will do and the topmost one is cheapest. Content
 * that RELAYOUTS does not: a row growing between the topmost row and the fold moves the reader
 * while leaving the topmost row exactly where it was, and an anchor taken there reports zero.
 * Measured with a 600px height change injected into one row above a detached reader while the
 * window was open: 600px of movement, reported as 0 by a topmost-row anchor.
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
  // Then descend to the fold itself. A row can be taller than the viewport -- a long answer with
  // images and code usually is -- and a reader partway through one has the row's top ABOVE them.
  // An image or a Shiki block earlier in that same message growing moves everything the reader can
  // see and leaves the row's top exactly where it was, so a whole-row anchor reports zero. The
  // first descendant that reaches the fold does not.
  //
  // Bounded: markdown nests, but the block that straddles the fold is a handful of levels down,
  // and this runs once per re-pick rather than per frame.
  // What this deliberately does NOT cover, stated so the bound is known: if the fold lands inside
  // a tall LEAF block, a plain paragraph with no element children, the descent stops there with
  // its top still above the reader. Lines reflowing inside that one block, above the reader's eye
  // line, then move what they see without moving the block. Everything else is covered, since a
  // change in an earlier block moves this one's top and a change in a later block does not move
  // what the reader is looking at, so the residual is one paragraph's height change, for a
  // detached reader, during the few hundred milliseconds a window is open. Closing it means a
  // caret point at the fold, which is spelled differently on all three engines, costs a Range and
  // a rect on every frame rather than once per re-pick, and is not itself stable across the
  // reflow it would be measuring.
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
 * An anchor plus the row it sits in, sampled together.
 *
 * The row is the fallback. pickAnchorRow descends to the fold, so the anchor is very often the
 * `<pre>` Streamdown replaces when Shiki finishes highlighting it, and a replaced node takes its
 * baseline with it. A `[data-role]` row is not replaced in place by anything, so it survives to
 * carry the measurement across.
 *
 * Being an ANCESTOR, the row does not move when the replacement changes height inside it, so that
 * one height change is not corrected. That is deliberate and it is the better of the two available
 * errors. An anchor BELOW the replaced block would see it, but it would also see the block growing
 * DOWNWARD, past the fold, which does not move anything the reader is looking at, and correcting
 * that would introduce an error in the common case to remove one in a rarer case. Measuring what
 * the reader actually sees when the block AT the fold is itself replaced needs a caret point at
 * the fold, and the note on pickAnchorRow says why that is not worth it here. The residual either
 * way is one code block's height change, for a detached reader, during the few hundred
 * milliseconds a window is open.
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

  // `isRunning` is read imperatively rather than through useAuiState on purpose: a reactive
  // subscription would re-render this component, and therefore rebuild the whole row array, on
  // every run start and stop.
  const isRunningNow = useCallback(
    () => aui.thread().getState().isRunning === true,
    [aui],
  );

  const [mountWindow, setMountWindow] = useState<MountWindow>(() =>
    initialWindow(count, isRunningNow()),
  );

  // Re-arm per thread, adjusting state during render as React documents, so the previous
  // thread's window can never gate the new one for a frame.
  //
  // This branch does not run in the app today, and saying so is better than letting it read as a
  // live safeguard. `<GeneratedImageOverlayProvider key={runtimeThreadId}>` in thread.tsx wraps
  // this subtree, so a thread switch destroys and rebuilds this component and the window re-arms
  // through the useState initialiser above instead. Measured across in-app switches: zero
  // arm records from this branch, a fresh instance at every switch, and the viewport DOM node
  // recreated each time. It is kept rather than deleted because it is what makes the component
  // correct on its own terms -- remove that ancestor key and this is the only thing standing
  // between a thread switch and the previous thread's window gating the next one -- and it is
  // exercised by the unkeyed control in tests/studio/probe_pm_edge.py rather than left untested.
  const [previousKey, setPreviousKey] = useState(resetKey);
  if (previousKey !== resetKey) {
    setPreviousKey(resetKey);
    setMountWindow(initialWindow(count, isRunningNow()));
  }

  // The commit that puts the FIRST rows into an empty tree.
  //
  // Mounting is not when the app learns how long the thread is. `threadListItem.id` flips as soon
  // as switchToThread resolves, and that is what remounts this subtree, but the messages only
  // arrive when the history adapter's load settles -- a Dexie read plus two HTTP calls later. So
  // on a cold open the `useState` above runs at `count === 0`, declines because 0 is below
  // MIN_PROGRESSIVE_MESSAGES, and without this the whole thread then lands in one unbounded
  // commit: exactly the stall this file exists to remove. Measured in the app, opening a
  // 220-message thread: mount at count 0, count 220 about 160ms later, same resetKey, and with
  // no second look the first paint carried all 220 rows.
  //
  // Gated on the PREVIOUS count being zero rather than on crossing MIN_PROGRESSIVE_MESSAGES, and
  // that is the whole safety argument. A tree that has never rendered a row has nothing on screen
  // and no scroll position to lose, so arming here cannot move anything a reader was looking at.
  // A threshold-crossing rule could not promise that: appending a 40th message to a 39-message
  // thread crosses it too, and would window a conversation the reader is in the middle of.
  const [previousCount, setPreviousCount] = useState(count);
  if (previousCount !== count) {
    setPreviousCount(count);
    // Same call as the mount path, so a run in flight suppresses the window here too.
    if (previousCount === 0 && count > 0) {
      setMountWindow(initialWindow(count, isRunningNow()));
    }
  }

  // A thread that shrank at or past the live window's start is reconciled HERE, during render,
  // rather than left for the next widening. Rendering every row (below) is only half of it: the
  // STATE still says start 204, and a widen against a 100-message thread would then produce
  // start 68 and unmount the 68 rows this render just mounted. Measured before the render-time
  // fallback existed, the same stale state painted an empty column for 2 to 8 frames.
  if (mountWindow != null && mountWindow.start >= count) {
    setMountWindow(null);
  }

  // A run that starts mid-widening drops the window immediately. Streaming writes to the same
  // scroll position the widening does, and a reply must never commit into a tree that has not
  // reached it. The remaining rows land in one commit, which is what would have happened without
  // this change at all.
  const threadIsRunning = useAuiState(({ thread }) => thread.isRunning);
  useEffect(() => {
    if (threadIsRunning && mountWindow != null) setMountWindow(null);
  }, [threadIsRunning, mountWindow]);

  // The row whose position is held still across a widening commit, and the scroll state it was
  // held against. Both are captured against the SCROLL CONTAINER rather than the window; see
  // AnchorSample. The arithmetic that turns the two samples into a scrollTop delta is
  // anchorCorrection, which is document space and assumes nothing else moved scrollTop -- which is
  // what disarming native anchoring, below, is for.
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
    // Disarm native scroll anchoring HERE, immediately before the capture it invalidates, rather
    // than from a layout effect. The correction is document space and only holds while nothing
    // else moves scrollTop, so the browser has to be out of the loop before the first sample is
    // taken, not merely soon.
    //
    // A layout effect is too early to do it. This component is a DESCENDANT of the viewport
    // element, so on the commit that mounts them both React runs this subtree's layout effects
    // before the viewport's own ref callback, `viewportRef.current` is still null, and the style
    // is silently skipped. Measured on the version that set it from a layout effect: computed
    // `overflow-anchor` stayed `auto` from the first painted row at +305ms until the FIRST
    // widening at +803ms, so that widening ran with anchoring live and the document-space
    // correction applied on top of the browser's own. A reader who scrolled 4000px in that window
    // was left 776px short, and one whose whole gesture landed inside it was carried back to the
    // bottom of a 118,004px thread, 24px from it instead of 4000. This callback runs from a
    // requestAnimationFrame, after a paint, so the ref is populated by definition.
    if (viewport) viewport.style.setProperty("overflow-anchor", "none");
    const anchor = viewport ? pickAnchorRow(viewport) : null;
    // The row the anchor sits in is captured alongside it, as a fallback. pickAnchorRow descends
    // to the fold, so the anchor is often the very `<pre>` that Streamdown replaces when Shiki
    // finishes highlighting it, and a transition-deferred widening leaves plenty of room for that
    // to happen. Without a fallback the correction is dropped when it does, and with anchoring off
    // a whole chunk of prepended rows moves the reader permanently. A `[data-role]` row is not
    // replaced in place by anything.
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
      // Re-check rather than trust the render that scheduled this: a run can start between
      // the commit and this frame.
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

  // mountWindow is the TRIGGER here, not a value the body reads, which is why the rule below
  // cannot see it. The whole purpose of this effect is to run in the commit that widened the
  // window, before that commit paints. Dropping the dependency would run it once on mount and
  // never correct a shift again.
  // biome-ignore lint/correctness/useExhaustiveDependencies: mountWindow is a trigger, see above
  useLayoutEffect(() => {
    const captured = anchorRef.current;
    anchorRef.current = null;
    const viewport = viewportRef.current;
    if (!captured || !viewport) return;
    // The anchor first, the row it sat in if the anchor has been replaced since the capture.
    const [element, baseline] = captured.element.isConnected
      ? [captured.element, captured as AnchorSample]
      : captured.row?.isConnected && captured.rowSample
        ? [captured.row, captured.rowSample]
        : [null, null];
    if (!element || !baseline) return;
    const shift = anchorCorrection(baseline, sampleAnchor(viewport, element));
    // Called on EVERY widening commit, including the ones with nothing to apply: a zero
    // correction still has to resync the hook's scroll bookkeeping past the offset native scroll
    // anchoring moved, or the scroll event that anchoring fires arrives as a downward scroll the
    // reader did not make. The hook decides whether anything is written; see
    // adjustForContentInsertedAbove.
    adjustForContentInsertedAbove(shift ?? 0);
    // Hand the idle sampler below a POST-correction baseline rather than nulling it. Nulling it
    // meant the next frame merely re-picked and re-based, so an image or a Shiki block landing
    // between this layout effect and that frame was folded into the new baseline and never
    // corrected: with anchoring off, the reader kept the whole displacement.
    idleRef.current = holdAnchor(viewport, element);
  }, [mountWindow, adjustForContentInsertedAbove, viewportRef]);

  // Content above the reader also moves for reasons that are not a widening, and while the window
  // is open nothing else is watching for it. Streamdown replaces a `<pre>` when Shiki finishes
  // highlighting it, KaTeX resizes a formula, an image lands: each of those relayouts a row that
  // may be above the fold. Native anchoring is exactly what absorbs those, and it is disabled for
  // the duration, so the compensation has to cover the whole interval rather than only the
  // widening commits. The autoscroll hook's own mutation path does not help: it pins a FOLLOWING
  // reader and deliberately leaves a detached one alone.
  //
  // Measured, injecting a 600px height change into one row above a detached reader while the
  // window was open: without this the reader moved the full 600px, against 0px on the merge base
  // where native anchoring absorbed it.
  //
  // The anchor here is the first row the reader can actually SEE, not the first row in the list,
  // and that distinction is the whole reason the first version of this measured nothing. Widening
  // prepends above everything, so the topmost row moves and any row will do. A reflow does not: a
  // row growing between the topmost row and the fold moves the reader while leaving the topmost
  // row exactly where it was. The visible row is re-picked only when it is gone or has scrolled
  // out of view, so the per-frame cost is one getBoundingClientRect and one scrollTop read.
  useEffect(() => {
    if (mountWindow == null) {
      idleRef.current = null;
      return;
    }
    let frame = 0;
    const tick = () => {
      frame = requestAnimationFrame(tick);
      const viewport = viewportRef.current;
      // A pending widening capture owns the correction until its layout effect has run; stepping
      // in between would correct the same movement twice.
      if (!viewport || anchorRef.current) return;
      const held = idleRef.current;
      if (!held) {
        const element = pickAnchorRow(viewport);
        idleRef.current = element ? holdAnchor(viewport, element) : null;
        return;
      }
      // The anchor first, its row if the anchor has been replaced since the last frame. Re-picking
      // instead would discard the only pre-replacement sample and re-base AFTER the replacement's
      // own reflow, so with anchoring off the reader would keep that height change.
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
      // Out of view: the reader has scrolled past it, so hold something they can see instead.
      // Tested by the row's own box rather than by its top offset: a tall row scrolled just past
      // can have its top within a viewport of the fold while none of it is on screen, and a
      // reflow between it and the first visible row then moves the reader while leaving it still.
      if (!isAnchorVisible(viewport, element)) {
        // Re-pick in THIS frame rather than nulling and waiting for the next one. A baseline-free
        // frame is a frame in which a reflow above the newly visible content is folded into the
        // next baseline and kept by the reader, which is the same gap that made nulling after a
        // widening wrong.
        const replacement = pickAnchorRow(viewport);
        idleRef.current = replacement
          ? holdAnchor(viewport, replacement)
          : null;
        return;
      }
      const shift = anchorCorrection(baseline, sampleAnchor(viewport, element));
      if (shift !== null) adjustForContentInsertedAbove(shift);
      // Re-based EVERY frame, including the ones with nothing to correct. Returning early on a
      // no-op frame left the baseline's scrollTop at wherever the reader was several frames ago,
      // and anchorCorrection's clamp term reads that scrollTop, so a later shrink near the bottom
      // would have had the wrong amount subtracted from it.
      idleRef.current = holdAnchor(viewport, element);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [mountWindow, adjustForContentInsertedAbove, viewportRef]);

  // Native scroll anchoring goes back on the moment the window closes, and on unmount, so a
  // settled thread is exactly the thread that shipped before this change. Turning it OFF is done
  // at capture time instead, for the ordering reason in captureAnchor.
  //
  // Why it has to be off at all: left on, the browser moves scrollTop by the inserted height on
  // some frames and not others -- not at all on any shipping Safari, and suppressed per frame
  // everywhere else after a programmatic scroll, which is what a scrollbar drag, PageUp and
  // middle-click autoscroll are -- and no measurement taken inside the frame can tell which kind
  // of frame it is in. Off, scrollTop moves only when the reader or this code moves it, which is
  // the assumption anchorCorrection is built on.
  useLayoutEffect(() => {
    if (mountWindow != null) return;
    viewportRef.current?.style.removeProperty("overflow-anchor");
  }, [mountWindow, viewportRef]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: unmount-only cleanup
  useLayoutEffect(() => {
    const viewport = viewportRef.current;
    return () => {
      viewport?.style.removeProperty("overflow-anchor");
    };
  }, [viewportRef]);

  // Registered only while rows are actually being withheld, so completeProgressiveMounts is free
  // for the settled thread that is the overwhelmingly common case.
  //
  // A layout effect, not an effect: it runs one phase earlier, which narrows (it cannot close)
  // the window in which a caller sees an empty completer set while rows are already withheld.
  const isWithholding = mountWindow != null;
  const completionWaiters = useRef<Array<() => void>>([]);
  useLayoutEffect(() => {
    if (!isWithholding) return;
    const complete = () =>
      new Promise<void>((resolve) => {
        completionWaiters.current.push(resolve);
        setMountWindow(null);
      });
    activeCompleters.add(complete);
    return () => {
      activeCompleters.delete(complete);
    };
  }, [isWithholding]);

  // Resolve on the commit that actually dropped the window, then after a paint. The previous
  // version resolved on a bare two-frame timer started at the call, which raced the commit it had
  // just asked for: measured, it resolved with 16 of 220 rows still in the document on 4 of 5
  // WebKit runs and 2 of 5 Firefox runs, and held on Chromium only because the widening commit
  // happened to block the frame loop.
  const flushCompletionWaiters = useCallback(() => {
    const waiters = completionWaiters.current;
    completionWaiters.current = [];
    for (const resolve of waiters) resolve();
  }, []);

  useEffect(() => {
    if (mountWindow != null || completionWaiters.current.length === 0) return;
    // The ref is not emptied until the frame actually fires, so a cancelled frame leaves the
    // waiters to be settled by the next commit or by the unmount below rather than dropping them.
    const frame = requestAnimationFrame(() =>
      requestAnimationFrame(flushCompletionWaiters),
    );
    return () => cancelAnimationFrame(frame);
  }, [mountWindow, flushCompletionWaiters]);

  // A thread that goes away settles anything still waiting on it. Without this, a DOM capture that
  // raced a thread switch or a navigation waited forever: the layout effect's cleanup removes the
  // completer from activeCompleters but the promise it already handed out was only ever resolved
  // by the effect above, which does not run again on an unmounted component.
  useEffect(() => flushCompletionWaiters, [flushCompletionWaiters]);

  return mountWindow;
}

/**
 * The thread's message list. Drop-in for `ThreadPrimitive.Messages`, with the first commit on a
 * long thread bounded to the tail and the rest arriving over the following frames.
 *
 * Memoized on the component identities exactly as upstream's `ThreadPrimitive.Messages` is, so a
 * re-render of `Thread` does not rebuild the row array.
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
      // `start >= count` means the thread shrank underneath a live window -- a bulk delete, or a
      // switch to a shorter thread -- and clamping `start` to `count` would make this loop emit
      // NOTHING. Measured on the PR before this line: dropping a 220-message thread to 10 while
      // the window sat at start=204 painted an empty column for 2 to 8 frames (37 to 146ms) on
      // all three engines, with 10 messages sitting in the store. `widen` heals it a frame later,
      // but the widen is inside startTransition, so the blank persists. Drop the restriction in
      // the same commit instead: the window can no longer withhold anything meaningful anyway.
      const first =
        mountWindow == null || mountWindow.start >= count
          ? 0
          : Math.max(mountWindow.start, 0);
      // One shared element for every row, which is what makes React skip the rows that did not
      // change when the count does. See the header.
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
