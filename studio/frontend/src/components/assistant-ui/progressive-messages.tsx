// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Widen-only mount window glue; see progressive-mount-controller.ts. One shared propless slot
// element lets React skip unchanged rows; a components object would re-render every row.

import {
  AuiProvider,
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

import { createRowNotificationGate } from "@/components/assistant-ui/row-notification-gate";
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

/** Module-level: consumers are imperative DOM readers, and a thread can mount twice (Compare). */
interface ActiveCompleter {
  complete: () => Promise<void>;
  viewportRef: RefObject<HTMLElement | null>;
}

const activeCompleters = new Set<ActiveCompleter>();

/** DOM readers (not store readers) must await this, or they see a short conversation. */
export const PROGRESSIVE_MOUNT_SEARCH_MS = 400;

export async function completeProgressiveMounts(
  /** Viewports to force; find-in-page passes one to skip inert off-route workspaces. */
  wants?: (viewport: HTMLElement | null) => boolean,
): Promise<void> {
  const wanted = () =>
    wants === undefined
      ? [...activeCompleters]
      : [...activeCompleters].filter((entry) => wants(entry.viewportRef.current));
  // An empty completer set is only believed after SEARCH_MS: a thread still loading history also
  // looks empty. A set that was non-empty and drained is believed immediately.
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
    // Filtered set: an unwanted completer would otherwise hold the loop open forever.
    if (wanted().length === 0 && (observed || Date.now() >= deadline)) {
      return;
    }
  }
}

export function hasPendingProgressiveMounts(): boolean {
  return activeCompleters.size > 0;
}

/** Clears only an empty inline declaration so the viewport carries no stray `style=""`. */
function restoreScrollAnchoring(viewport: HTMLElement | null): void {
  if (!viewport) return;
  viewport.style.removeProperty("overflow-anchor");
  if (viewport.getAttribute("style") === "") viewport.removeAttribute("style");
}

/** The first row the reader can SEE: a relayout above the fold moves them, not the top row. */
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
  // Descend to the fold: rows can be taller than the viewport. A tall leaf block is a known residual.
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

/** The row is the fallback when the anchor (often a Shiki-replaced `<pre>`) is replaced. */
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

  // Imperative, so run start/stop does not re-render and rebuild the row array.
  const isRunningNow = useCallback(
    () => aui.thread().getState().isRunning === true,
    [aui],
  );

  // Read imperatively: a useAuiState selector on messages would walk the thread every keystroke.
  const renderableAt = useCallback(() => {
    const messages = aui.thread().getState().messages as ReadonlyArray<{
      role?: ThreadMessageRole;
      composer?: { isEditing?: boolean };
    }>;
    return (index: number) => {
      const message = messages[index];
      // Unknown shapes count as renderable, degrading to the count-based window rather than none.
      if (message?.role == null) return true;
      return rendersAsRow(message.role, message.composer?.isEditing === true);
    };
  }, [aui]);

  const [mountWindow, setMountWindow] = useState<MountWindow>(() =>
    initialWindow(count, isRunningNow(), renderableAt()),
  );

  // Re-arm per thread during render. Unexercised while an ancestor key remounts on switch, but
  // needed if that key is removed.
  const [previousKey, setPreviousKey] = useState(resetKey);
  if (previousKey !== resetKey) {
    setPreviousKey(resetKey);
    setMountWindow(initialWindow(count, isRunningNow(), renderableAt()));
  }

  // Arm on the commit that first brings rows into an empty tree: history arrives after mount, and
  // gating on the threshold instead would window a thread the reader is in the middle of.
  const [previousCount, setPreviousCount] = useState(count);
  if (previousCount !== count) {
    setPreviousCount(count);
    if (previousCount === 0 && count > 0) {
      setMountWindow(initialWindow(count, isRunningNow(), renderableAt()));
    }
  }

  // Reconcile a thread that shrank past the window here, or a later widen unmounts rows.
  if (mountWindow != null && mountWindow.start >= count) {
    setMountWindow(null);
  }

  // A run starting mid-widening drops the window: streaming and widening write the same scrollTop.
  const threadIsRunning = useAuiState(({ thread }) => thread.isRunning);
  useEffect(() => {
    if (threadIsRunning && mountWindow != null) setMountWindow(null);
  }, [threadIsRunning, mountWindow]);

  // Captured against the scroll container; the correction assumes native anchoring is off.
  const anchorRef = useRef<
    | ({
        element: Element;
        row: Element | null;
        rowSample: AnchorSample | null;
      } & AnchorSample)
    | null
  >(null);
  const idleRef = useRef<HeldAnchor | null>(null);

  const captureAnchor = useCallback(() => {
    const viewport = viewportRef.current;
    // Disarm native anchoring right before the capture. A layout effect is too early: as a descendant,
    // this runs before the viewport ref callback on the mounting commit.
    if (viewport) viewport.style.setProperty("overflow-anchor", "none");
    const anchor = viewport ? pickAnchorRow(viewport) : null;
    // Fall back to the row: the anchor is often a `<pre>` replaced when Shiki finishes.
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

  // mountWindow is a trigger: this must run in the commit that widened the window, before paint.
  // biome-ignore lint/correctness/useExhaustiveDependencies: mountWindow is a trigger, see above
  useLayoutEffect(() => {
    const captured = anchorRef.current;
    anchorRef.current = null;
    const viewport = viewportRef.current;
    if (!captured || !viewport) return;
    const [element, baseline] = captured.element.isConnected
      ? [captured.element, captured as AnchorSample]
      : captured.row?.isConnected && captured.rowSample
        ? [captured.row, captured.rowSample]
        : [null, null];
    if (!element || !baseline) return;
    const shift = anchorCorrection(baseline, sampleAnchor(viewport, element));
    // Call even with zero shift: it resyncs the hook's scroll bookkeeping.
    adjustForContentInsertedAbove(shift ?? 0);
    // Keep a post-correction baseline; nulling it folds an in-between reflow into the next one.
    idleRef.current = holdAnchor(viewport, element);
  }, [mountWindow, adjustForContentInsertedAbove, viewportRef]);

  // Compensates reflows above a detached reader while native anchoring is off. Currently dormant
  // because the widening effect above registers first: reordering these two effects turns it on.
  useEffect(() => {
    if (mountWindow == null) {
      idleRef.current = null;
      return;
    }
    let frame = 0;
    const tick = () => {
      frame = requestAnimationFrame(tick);
      const viewport = viewportRef.current;
      // A pending widening capture owns the correction; stepping in would correct twice.
      if (!viewport || anchorRef.current) return;
      const held = idleRef.current;
      if (!held) {
        const element = pickAnchorRow(viewport);
        idleRef.current = element ? holdAnchor(viewport, element) : null;
        return;
      }
      // Fall back to the row if replaced; re-picking would re-base after the replacement's reflow.
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
      // Tested by the row's box, not its top: a tall row just off screen can keep its top near the fold.
      if (!isAnchorVisible(viewport, element)) {
        // Re-pick in this frame; a baseline-free frame folds a reflow into the next baseline.
        const replacement = pickAnchorRow(viewport);
        idleRef.current = replacement
          ? holdAnchor(viewport, replacement)
          : null;
        return;
      }
      const shift = anchorCorrection(baseline, sampleAnchor(viewport, element));
      if (shift !== null) adjustForContentInsertedAbove(shift);
      // Re-base every frame: anchorCorrection's clamp term reads the baseline scrollTop.
      idleRef.current = holdAnchor(viewport, element);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [mountWindow, adjustForContentInsertedAbove, viewportRef]);

  // Restore native anchoring when the window closes or on unmount; it is turned off at capture.
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

  // Registered only while rows are withheld, so completeProgressiveMounts is free when settled.
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

  // Resolve after the commit that dropped the window, then a paint; a bare timer raced the commit.
  const flushCompletionWaiters = useCallback(() => {
    const waiters = completionWaiters.current;
    completionWaiters.current = [];
    for (const resolve of waiters) resolve();
  }, []);

  useEffect(() => {
    if (mountWindow != null || completionWaiters.current.length === 0) return;
    // Waiters are emptied only when the frame fires, so a cancelled frame keeps them.
    const frame = requestAnimationFrame(() =>
      requestAnimationFrame(flushCompletionWaiters),
    );
    return () => cancelAnimationFrame(frame);
  }, [mountWindow, flushCompletionWaiters]);

  // Settle waiters on unmount, or a DOM capture racing a thread switch waits forever.
  useEffect(() => flushCompletionWaiters, [flushCompletionWaiters]);

  return mountWindow;
}

/** Drop-in for `ThreadPrimitive.Messages` that bounds a long thread's first commit to the tail. */
export const ProgressiveMessages: FC<{
  renderMessage: () => ReactElement;
  resetKey: string | undefined;
  viewportRef: RefObject<HTMLElement | null>;
}> = memo(
  ({ renderMessage, resetKey, viewportRef }) => {
    const count = useAuiState(({ thread }) => thread.messages.length);
    const mountWindow = useProgressiveMountWindow(count, resetKey, viewportRef);
    const aui = useAui();
    const gate = useMemo(() => createRowNotificationGate(aui), [aui]);

    return useMemo(() => {
      if (count === 0) return null;
      // A thread that shrank under a live window: drop the restriction now; clamping would emit nothing.
      const first =
        mountWindow == null || mountWindow.start >= count
          ? 0
          : Math.max(mountWindow.start, 0);
      const message = renderMessage();
      const rows: ReactElement[] = [];
      for (let index = first; index < count; index += 1) {
        rows.push(
          <AuiProvider key={index} value={gate.row(index)}>
            <MessageByIndexProvider index={index}>{message}</MessageByIndexProvider>
          </AuiProvider>,
        );
      }
      return <>{rows}</>;
    }, [count, mountWindow, renderMessage, gate]);
  },
  (prev, next) =>
    prev.resetKey === next.resetKey &&
    prev.renderMessage === next.renderMessage,
);

ProgressiveMessages.displayName = "ProgressiveMessages";
