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
  initialWindow,
  widen,
} from "@/components/assistant-ui/progressive-mount-controller";
import {
  useAdjustForContentInsertedAbove,
  useUserGestureSeq,
} from "@/components/assistant-ui/use-intent-aware-autoscroll";

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
export async function completeProgressiveMounts(): Promise<void> {
  if (activeCompleters.size === 0) return;
  await Promise.all([...activeCompleters].map((complete) => complete()));
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
  const getUserGestureSeq = useUserGestureSeq();

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
  const [previousKey, setPreviousKey] = useState(resetKey);
  if (previousKey !== resetKey) {
    setPreviousKey(resetKey);
    setMountWindow(initialWindow(count, isRunningNow()));
  }

  // A run that starts mid-widening drops the window immediately. Streaming writes to the same
  // scroll position the widening does, and a reply must never commit into a tree that has not
  // reached it. The remaining rows land in one commit, which is what would have happened without
  // this change at all.
  const threadIsRunning = useAuiState(({ thread }) => thread.isRunning);
  useEffect(() => {
    if (threadIsRunning && mountWindow != null) setMountWindow(null);
  }, [threadIsRunning, mountWindow]);

  // The row whose position is held still across a widening commit, captured in VIEWPORT space
  // (its offset from the top of the scroll container).
  //
  // Viewport space, not document space, and this is the whole correctness argument. Chromium
  // implements CSS scroll anchoring, this viewport does not set `overflow-anchor: none`, and
  // inserting rows above the scroll position is exactly what scroll anchoring exists to
  // compensate. So by the time this is read the browser has usually ALREADY moved scrollTop by
  // the inserted height. What has to be corrected is only the residual the browser did not
  // absorb, and viewport space is what measures a residual: it is zero precisely when the anchor
  // is still where the reader left it.
  //
  // Document space measures the wrong thing here. Rows inserted above move an element down the
  // document by their height whether or not the viewport was compensated, so a document-space
  // delta reports the full insertion even when nothing needs doing, and applying it on top of
  // the browser's own compensation doubles it. Measured, on a reader parked 4000px above the
  // bottom of a 300K thread: the document-space version walked scrollTop 22,897 -> 117,104
  // across seven widenings and dumped them at the bottom of the thread, distance 4000px -> 0px.
  // With this version the distance holds.
  const anchorRef = useRef<{
    element: Element;
    viewportOffset: number;
    gestureSeq: number;
  } | null>(null);

  const captureAnchor = useCallback(() => {
    const viewport = viewportRef.current;
    const first = viewport?.querySelector("[data-role]") ?? null;
    anchorRef.current =
      viewport && first
        ? {
            element: first,
            viewportOffset: first.getBoundingClientRect().top,
            gestureSeq: getUserGestureSeq(),
          }
        : null;
  }, [viewportRef, getUserGestureSeq]);

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
    // The reader scrolled between the capture and this commit. In viewport space their gesture is
    // indistinguishable from a layout shift, so correcting here would cancel their own scroll:
    // measured, a 4000px wheel during a widening was undone within the frame and the reader was
    // put back at the bottom of the thread. Skip this one. The residual a widening actually
    // leaves is single-digit pixels, so a skipped frame is invisible, and the next widening
    // corrects normally.
    if (getUserGestureSeq() !== captured.gestureSeq) return;
    const shift =
      captured.element.getBoundingClientRect().top - captured.viewportOffset;
    // The hook decides whether this is acted on; see adjustForContentInsertedAbove. Rounding
    // down to whole pixels keeps a subpixel reflow from issuing a scroll write every frame.
    if (Math.abs(shift) >= 1) adjustForContentInsertedAbove(shift);
  }, [
    mountWindow,
    adjustForContentInsertedAbove,
    viewportRef,
    getUserGestureSeq,
  ]);

  // Registered only while rows are actually being withheld, so completeProgressiveMounts is free
  // for the settled thread that is the overwhelmingly common case.
  const isWithholding = mountWindow != null;
  useEffect(() => {
    if (!isWithholding) return;
    const complete = () =>
      new Promise<void>((resolve) => {
        setMountWindow(null);
        requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
      });
    activeCompleters.add(complete);
    return () => {
      activeCompleters.delete(complete);
    };
  }, [isWithholding]);

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
      const first =
        mountWindow == null
          ? 0
          : Math.min(Math.max(mountWindow.start, 0), count);
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
