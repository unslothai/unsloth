// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// React glue for the widen-only mount window. The state machine it drives, and the reasoning for
// why this is not a virtualizer, are in progressive-mount-controller.ts.
//
// This replaces `<ThreadPrimitive.Messages components={...} />` with an equivalent map over
// `ThreadPrimitive.MessageByIndex`. That replacement is permanent rather than a mode the thread
// leaves once the window closes, and it has to be: `ThreadPrimitive.Messages` renders
// MessageByIndexProvider -> RenderChildrenWithAccessor -> ThreadMessageComponent, while
// MessageByIndex renders MessageByIndexProvider -> ThreadMessageComponent. Switching between the
// two after convergence would change the element type at that position and React would unmount and
// rebuild every message, which is precisely the cost being avoided. Rendering one of them
// throughout means the only thing that ever changes is how many rows the map emits.
//
// The output is identical either way: RenderChildrenWithAccessor emits no DOM (it returns
// `children(getItem)` through a propless-element memo), and in the `components` form upstream
// passes a child that ignores the accessor, so it is a pure pass-through here.

import { ThreadPrimitive, useAui, useAuiState } from "@assistant-ui/react";
import {
  type ComponentType,
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
  type MountWindow,
  anchorCorrection,
  initialWindow,
  widen,
} from "@/components/assistant-ui/progressive-mount-controller";
import { useAdjustForContentInsertedAbove } from "@/components/assistant-ui/use-intent-aware-autoscroll";

/** The subset of upstream's MessagesComponentConfig the thread actually passes. */
export type ThreadMessageComponents = {
  UserMessage: ComponentType;
  EditComposer: ComponentType;
  AssistantMessage: ComponentType;
};

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

  // A run that starts mid-widening drops the window immediately. Streaming writes to the same
  // scroll position the widening does, and a reply must never commit into a tree that has not
  // reached it. The remaining rows land in one commit, which is what would have happened without
  // this change at all.
  const threadIsRunning = useAuiState(({ thread }) => thread.isRunning);
  useEffect(() => {
    if (threadIsRunning && mountWindow != null) setMountWindow(null);
  }, [threadIsRunning, mountWindow]);

  // The row whose position is held still across a widening commit. Both ends of the measurement
  // are captured here -- its offset from the top of the scroll container, the container's
  // scrollTop, and the user-gesture counter -- because which of them the correction uses depends
  // on whether the engine compensates for content inserted above. anchorCorrection has the
  // reasoning and the numbers for both branches; the short version is that on an engine with CSS
  // scroll anchoring the browser has already done the work and only a 3 to 5 pixel residual is
  // left, while on one without it (every shipping Safari today) nothing has been done and the
  // full inserted height has to be applied.
  const anchorRef = useRef<{
    element: Element;
    viewportOffset: number;
    scrollTop: number;
  } | null>(null);

  const captureAnchor = useCallback(() => {
    const viewport = viewportRef.current;
    const first = viewport?.querySelector("[data-role]") ?? null;
    anchorRef.current =
      viewport && first
        ? {
            element: first,
            viewportOffset: first.getBoundingClientRect().top,
            scrollTop: viewport.scrollTop,
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
    if (!captured || !viewport || !captured.element.isConnected) return;
    // Which space this is measured in, and whether a frame the reader scrolled through can be
    // dropped, both depend on whether the engine already moved scrollTop. See anchorCorrection.
    const shift = anchorCorrection(captured, {
      viewportOffset: captured.element.getBoundingClientRect().top,
      scrollTop: viewport.scrollTop,
    });
    // Called on EVERY widening commit, including the ones with nothing to apply: a zero
    // correction still has to resync the hook's scroll bookkeeping past the offset native scroll
    // anchoring moved, or the scroll event that anchoring fires arrives as a downward scroll the
    // reader did not make. The hook decides whether anything is written; see
    // adjustForContentInsertedAbove.
    adjustForContentInsertedAbove(shift ?? 0);
  }, [mountWindow, adjustForContentInsertedAbove, viewportRef]);

  // CSS scroll anchoring is turned OFF for as long as the window is open, and turned back on the
  // moment it closes, so a settled thread is exactly as it was before this change.
  //
  // This is the one thing that makes the correction below deterministic. Left on, the browser
  // moves scrollTop by the inserted height on some frames and not others -- not at all on any
  // shipping Safari, and suppressed per frame everywhere else after a programmatic scroll, which
  // is what a scrollbar drag, PageUp and middle-click autoscroll are -- and no measurement taken
  // inside the frame can tell which kind of frame it is in. Off, scrollTop moves only when the
  // reader or this code moves it, which is the assumption anchorCorrection is built on.
  //
  // A layout effect keyed on the same trigger as the correction, so it is in place before the
  // first widening rather than a frame after it.
  // biome-ignore lint/correctness/useExhaustiveDependencies: isWithholding is the trigger
  useLayoutEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    if (mountWindow == null) {
      viewport.style.removeProperty("overflow-anchor");
      return;
    }
    viewport.style.setProperty("overflow-anchor", "none");
  }, [mountWindow, viewportRef]);

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
  useEffect(() => {
    if (mountWindow != null || completionWaiters.current.length === 0) return;
    const waiters = completionWaiters.current;
    completionWaiters.current = [];
    const frame = requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        for (const resolve of waiters) resolve();
      }),
    );
    return () => cancelAnimationFrame(frame);
  }, [mountWindow]);

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
  components: ThreadMessageComponents;
  /** Changing this re-arms the window. The runtime thread id. */
  resetKey: string | undefined;
  /** The scroll viewport these rows live in. Identity is stable, so it is not compared below. */
  viewportRef: RefObject<HTMLElement | null>;
}> = memo(
  ({ components, resetKey, viewportRef }) => {
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
      const rows: ReactElement[] = [];
      for (let index = first; index < count; index += 1) {
        rows.push(
          <ThreadPrimitive.MessageByIndex
            key={index}
            index={index}
            components={components}
          />,
        );
      }
      return <>{rows}</>;
    }, [count, mountWindow, components]);
  },
  (prev, next) =>
    prev.resetKey === next.resetKey &&
    prev.components.UserMessage === next.components.UserMessage &&
    prev.components.EditComposer === next.components.EditComposer &&
    prev.components.AssistantMessage === next.components.AssistantMessage,
);

ProgressiveMessages.displayName = "ProgressiveMessages";
