// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Sliding mount window glue; the window maths is in thread-window.ts. Only the rows near the
// reader are mounted and spacers stand in for the rest, so a thread's cost follows what is on
// screen, not its length. One shared propless slot element lets React skip unchanged rows; a
// components object would re-render every row.

import { AuiProvider, useAui, useAuiState } from "@assistant-ui/react";
import {
  type FC,
  type ReactElement,
  type RefObject,
  memo,
  startTransition,
  useCallback,
  useEffect,
  useInsertionEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { flushSync } from "react-dom";

import { createRowNotificationGate } from "@/components/assistant-ui/row-notification-gate";
import { MessageByIndexProvider } from "@/components/assistant-ui/stale-safe-scopes";
import {
  type AnchorSample,
  CHUNK_MESSAGES,
  anchorCorrection,
  nextMountChunk,
  watchProgressiveScroll,
} from "@/components/assistant-ui/progressive-mount-controller";
import {
  type MountedRow,
  type RowMeta,
  type RowWindow,
  RowHeights,
  WINDOW_REACH_PX,
  absorbShift,
  followCount,
  holdTailTurn,
  initialRowWindow,
  messageWeight,
  planEngage,
  planFill,
  planSlide,
  sameWindow,
  spacerHeights,
  spacerRows,
  threadSlots,
  withPins,
} from "@/components/assistant-ui/thread-window";
import {
  type ThreadMessageRole,
  rendersAsRow,
} from "@/components/assistant-ui/thread-message-slot";
import { useAdjustForContentInsertedAbove } from "@/components/assistant-ui/use-intent-aware-autoscroll";
import { inputPointerHeld } from "@/lib/input-pointer-state";
import { isWebKitGTK } from "@/components/ui/theme-view-transition";

const WEBKIT_GTK = typeof navigator !== "undefined"
  && isWebKitGTK(navigator.platform, navigator.userAgent);
/** Rows a scroll step mounts. Each is a main-thread commit, so WebKitGTK (which commits inside
 *  the frame) takes fewer. */
const SLIDE_STEP = WEBKIT_GTK ? 2 : 8;
const FILL_CHUNK = WEBKIT_GTK ? 2 : CHUNK_MESSAGES;
/** Sub-pixel shifts are real here: the spacer takes them, where scrollTop would round them. */
const ANCHOR_PRECISION = 1 / 64;
/** A full mount asked for by a DOM reader is kept this long after its last request. */
const FILL_RELEASE_MS = 3000;
/** How soon a settle that found the reader still scrolling looks again. */
const SETTLE_RETRY_MS = 350;
/** How often a full mount nothing announces the end of is checked for release. */
const HOLD_RECHECK_MS = 1000;
const FIND_FLAG = "data-find-bar-open";
// Not `[data-role]`: a prompt being edited renders as an EditComposer, which has no role.
const ROW_SELECTOR = "[data-message-id]";
const SPACER_ATTRIBUTE = "data-thread-spacer";
const PINNED_CONTENT = "textarea, [contenteditable=true]";

/** Module-level: consumers are imperative DOM readers, and a thread can mount twice (Compare). */
interface ActiveWindow {
  /** Mounts every row; resolves once they have painted. */
  complete: () => Promise<void>;
  /** Mounts the row for a message if the window left it out. */
  reveal: (messageId: string) => boolean;
  withholding: () => boolean;
  filling: () => boolean;
  /** Looks again on the next frame; a hold was taken or released. */
  wake: () => void;
  viewportRef: RefObject<HTMLElement | null>;
}

const activeWindows = new Set<ActiveWindow>();
const rowListeners = new Set<() => void>();
const fullMountHolds = new Set<symbol>();

/**
 * Keeps every row of every thread mounted until the returned release is called, for a reader that
 * needs the whole conversation in the DOM for as long as it runs rather than for one read (which
 * `completeProgressiveMounts` serves). The window reopens once the last hold is gone.
 */
export function holdProgressiveMounts(): () => void {
  const hold = Symbol("full mount");
  fullMountHolds.add(hold);
  for (const entry of activeWindows) entry.wake();
  return () => {
    if (fullMountHolds.delete(hold)) for (const entry of activeWindows) entry.wake();
  };
}

/** Runs after a commit that mounted or unmounted rows, for readers that hold row elements. */
export function subscribeMountedRows(listener: () => void): () => void {
  rowListeners.add(listener);
  return () => {
    rowListeners.delete(listener);
  };
}

/** DOM readers (not store readers) must await this, or they see a short conversation. */
export const PROGRESSIVE_MOUNT_SEARCH_MS = 400;

export async function completeProgressiveMounts(
  /** Viewports to force; find-in-page passes one to skip inert off-route workspaces. */
  wants?: (viewport: HTMLElement | null) => boolean,
): Promise<void> {
  const wanted = () =>
    [...activeWindows].filter((entry) =>
      wants === undefined || wants(entry.viewportRef.current));
  // An empty set is only believed after SEARCH_MS: a thread still loading history also looks
  // empty. A set that was non-empty and drained is believed immediately.
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
    // Filtered set: an unwanted window would otherwise hold the loop open forever.
    if (
      !wanted().some((entry) => entry.withholding()) &&
      (observed || Date.now() >= deadline)
    ) {
      return;
    }
  }
}

/** True only while a full mount is in progress; a settled window withholds rows without being
 *  pending, and so does a hidden thread, whose full mount waits until it is shown. */
export function hasPendingProgressiveMounts(): boolean {
  for (const entry of activeWindows) if (entry.filling()) return true;
  return false;
}

/** Mounts the row for a message the window left out, so a jump can scroll to it. Resolves false
 *  when no mounted thread under `viewport` holds the message. */
export function revealProgressiveRow(viewport: HTMLElement, messageId: string): boolean {
  for (const entry of activeWindows) {
    if (entry.viewportRef.current === viewport) return entry.reveal(messageId);
  }
  return false;
}

/** Clears only an empty inline declaration so the viewport carries no stray `style=""`. */
function restoreScrollAnchoring(viewport: HTMLElement | null): void {
  if (!viewport) return;
  viewport.style.removeProperty("overflow-anchor");
  if (viewport.getAttribute("style") === "") viewport.removeAttribute("style");
}

function* layoutChildren(element: Element): Generator<Element> {
  for (const child of element.children) {
    if (getComputedStyle(child).display === "contents") yield* layoutChildren(child);
    else yield child;
  }
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
  if (anchor.hasAttribute("data-message-parked")) return anchor;
  if (
    typeof anchor.checkVisibility === "function"
    && !anchor.checkVisibility({ contentVisibilityAuto: true })
  ) {
    return anchor;
  }
  // Descends to the first descendant reaching the fold, since a whole-row anchor reports zero for a
  // row taller than the viewport; capped at 8 levels, run once per re-pick. Residual: a tall leaf
  // block can still reflow above the reader; a caret fix would cost a Range per frame.
  for (let depth = 0; depth < 8; depth += 1) {
    if (anchor.getBoundingClientRect().top >= fold) break;
    let next: Element | null = null;
    for (const child of layoutChildren(anchor)) {
      if (
        typeof child.checkVisibility === "function"
        && !child.checkVisibility({ contentVisibilityAuto: true })
      ) {
        if (child.checkVisibility()) return anchor;
        continue;
      }
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

type ThreadMessage = {
  id?: string;
  role?: ThreadMessageRole;
  content?: readonly { type?: string; text?: string }[];
  composer?: { isEditing?: boolean };
};

type Metas = {
  metas: RowMeta[];
  index: Map<string, number>;
  /** The last user prompt that renders, or -1. */
  tailTurn: number;
};

/** Message objects are immutable, so a meta lives as long as its message does. */
const metaCache = new WeakMap<object, RowMeta>();

function metaOf(message: ThreadMessage, index: number): RowMeta {
  const cached = metaCache.get(message);
  if (cached) return cached;
  const meta: RowMeta = {
    id: message.id ?? `#${index}`,
    role: message.role ?? "assistant",
    weight: messageWeight(message.content),
    // Unknown shapes count as rendering: room for a row that turns out empty costs nothing.
    renders:
      message.role == null ||
      rendersAsRow(message.role, message.composer?.isEditing === true),
  };
  metaCache.set(message, meta);
  return meta;
}

/** The messages array is replaced, never edited, so it keys what was derived from it. */
const metasCache = new WeakMap<readonly ThreadMessage[], Metas>();

function metasOf(messages: readonly ThreadMessage[]): Metas {
  const cached = metasCache.get(messages);
  if (cached) return cached;
  const metas = messages.map(metaOf);
  const index = new Map<string, number>();
  let tailTurn = -1;
  metas.forEach((meta, at) => {
    index.set(meta.id, at);
    if (meta.role === "user" && meta.renders) tailTurn = at;
  });
  const derived = { metas, index, tailTurn };
  metasCache.set(messages, derived);
  return derived;
}

/** Heights outlive a thread switch, so coming back to a thread opens on measured spacers. */
const threadHeights = new Map<string, RowHeights>();
const REMEMBERED_THREADS = 8;

function heightsFor(key: string | undefined): RowHeights {
  if (key === undefined) return new RowHeights();
  let heights = threadHeights.get(key);
  if (heights) threadHeights.delete(key);
  else heights = new RowHeights();
  threadHeights.set(key, heights);
  if (threadHeights.size > REMEMBERED_THREADS) {
    const oldest = threadHeights.keys().next().value;
    if (oldest !== undefined) threadHeights.delete(oldest);
  }
  return heights;
}

type Measured = {
  /** Every mounted row in document order, the window's and the pinned ones outside it. */
  rows: MountedRow[];
  /** Top of the thread's first spacer, in px from the viewport's top edge. */
  threadTop: number;
};

function rowIndex(child: Element, lookup: Metas): number | undefined {
  if (!child.matches(ROW_SELECTOR)) return undefined;
  return lookup.index.get(child.getAttribute("data-message-id") ?? "");
}

/** One pass of layout reads over the thread, which also records each row's height: its top to
 *  the top of what follows it (a row or a spacer), so margins between rows are counted. */
function measureRows(viewport: HTMLElement, lookup: Metas, heights: RowHeights): Measured {
  const origin = viewport.getBoundingClientRect().top;
  const rows: MountedRow[] = [];
  let threadTop: number | null = null;
  let open: MountedRow | null = null;
  for (const child of viewport.children) {
    const index = rowIndex(child, lookup);
    if (index === undefined && !child.hasAttribute(SPACER_ATTRIBUTE)) continue;
    const box = child.getBoundingClientRect();
    const top = box.top - origin;
    threadTop ??= top;
    if (open) heights.measure(lookup.metas[open.index], Math.max(0, top - open.top));
    open = null;
    if (index === undefined) continue;
    open = { index, top, bottom: box.bottom - origin };
    rows.push(open);
  }
  if (open) heights.measure(lookup.metas[open.index], Math.max(0, open.bottom - open.top));
  return { rows, threadTop: threadTop ?? 0 };
}

/** A selection reaching into the thread keeps its rows: unmounting one would cut the range. */
function heldSelection(viewport: HTMLElement): Range | null {
  const selection = document.getSelection();
  if (selection === null || selection.isCollapsed || selection.rangeCount === 0) return null;
  const range = selection.getRangeAt(0);
  return range.intersectsNode(viewport) ? range : null;
}

/** A row the reader is working in keeps its state: focus, an open edit, media that is playing,
 *  the text they selected. */
function isPinned(row: Element, selection: Range | null): boolean {
  if (row.contains(document.activeElement)) return true;
  if (row.querySelector(PINNED_CONTENT) !== null) return true;
  if (selection?.intersectsNode(row)) return true;
  for (const media of row.querySelectorAll<HTMLMediaElement>("video, audio")) {
    if (!media.paused) return true;
  }
  return false;
}

/**
 * `next` holding on to the mounted rows it would leave out while the reader is working in them.
 * They stay mounted where they are, apart from the window, so a scrollbar drag to the other end
 * of the thread does not take an unsaved edit with it and does not mount what lies between.
 * `tail` is the row of a reply being written, whose effects run the stream.
 */
function keepPinned(
  viewport: HTMLElement,
  lookup: Metas,
  next: RowWindow,
  tail: number | null,
): RowWindow {
  const selection = heldSelection(viewport);
  const pins: number[] = [];
  for (const child of viewport.children) {
    const index = rowIndex(child, lookup);
    if (index === undefined || (index >= next.start && index < next.end)) continue;
    if (index === tail || isPinned(child, selection)) pins.push(index);
  }
  return withPins(next, pins, lookup.metas.length);
}

/**
 * A list that leaves rows out of the DOM has to say how long it is and where each row sits
 * (WAI-ARIA), or assistive technology counts only what is mounted. Written here and not by the
 * rows: the total changes on every append, and a row must not re-render for it. A full mount
 * publishes nothing.
 */
function publishOrdinals(viewport: HTMLElement, lookup: Metas, windowed: boolean): void {
  const total = String(lookup.metas.length);
  for (const child of viewport.children) {
    if (!child.matches(ROW_SELECTOR)) continue;
    const index = windowed
      ? lookup.index.get(child.getAttribute("data-message-id") ?? "")
      : undefined;
    if (index === undefined) {
      child.removeAttribute("aria-posinset");
      child.removeAttribute("aria-setsize");
      if (child.getAttribute("role") === "article") child.removeAttribute("role");
      continue;
    }
    // Ordinals need a role that supports them: a focusable message in a feed is an article.
    if (!child.hasAttribute("role")) child.setAttribute("role", "article");
    const position = String(index + 1);
    if (child.getAttribute("aria-posinset") !== position) {
      child.setAttribute("aria-posinset", position);
    }
    if (child.getAttribute("aria-setsize") !== total) child.setAttribute("aria-setsize", total);
  }
}

function useThreadWindow(
  count: number,
  resetKey: string | undefined,
  viewportRef: RefObject<HTMLElement | null>,
  topRef: RefObject<HTMLDivElement | null>,
  bottomRef: RefObject<HTMLDivElement | null>,
): { win: RowWindow | null; gapHeight: (from: number, to: number) => number } {
  const aui = useAui();
  const adjustScroll = useAdjustForContentInsertedAbove();

  // Read imperatively: a useAuiState selector on messages would walk the thread every keystroke.
  const readMetas = useCallback(
    (): Metas =>
      metasOf(aui.thread().getState().messages as unknown as readonly ThreadMessage[]),
    [aui],
  );

  const heights = useMemo(() => heightsFor(resetKey), [resetKey]);

  const open = useCallback((): RowWindow | null => {
    const { metas, tailTurn } = readMetas();
    const view = typeof window === "undefined" ? 900 : window.innerHeight;
    const opened = initialRowWindow(
      metas.length,
      (index) => heights.heightOf(metas[index]),
      view + WINDOW_REACH_PX,
    );
    return holdTailTurn(opened, metas.length, tailTurn);
  }, [readMetas, heights]);

  const [win, setWin] = useState<RowWindow | null>(open);
  const winRef = useRef(win);
  /** "fill" mounts every row for a DOM reader (find, print, select all); "slide" follows the reader. */
  const modeRef = useRef<"slide" | "fill">("slide");
  /** Px the leading spacer carries beyond its rows' heights; see absorbShift. */
  const slackRef = useRef(0);
  const releaseAtRef = useRef(0);
  const printingRef = useRef(false);
  const anchorRef = useRef<HeldAnchor | null>(null);
  /** A commit is in flight; its layout effect owns the correction. */
  const pendingRef = useRef(false);
  const frameRef = useRef(0);
  const chunkRef = useRef(0);
  const waitersRef = useRef<Array<() => void>>([]);
  const scrollActivity = useRef<ReturnType<typeof watchProgressiveScroll> | null>(null);
  const stepRef = useRef<() => void>(() => undefined);
  const settleTimerRef = useRef(0);
  /** Rows carry ordinals, so the next full mount has them to clear. */
  const ordinalsRef = useRef(false);

  // Re-arm per thread during render. Unexercised while an ancestor key remounts on switch, but
  // needed if that key is removed.
  const [previousKey, setPreviousKey] = useState(resetKey);
  if (previousKey !== resetKey) {
    setPreviousKey(resetKey);
    setWin(open());
  }

  const [previousCount, setPreviousCount] = useState(count);
  if (previousCount !== count) {
    setPreviousCount(count);
    if (previousCount === 0 && count > 0) {
      // History arrives after mount: open on the commit that first brings rows in.
      setWin(open());
    } else if (win != null) {
      const followed = followCount(win, previousCount, count);
      if (followed !== win) setWin(followed);
    }
  }

  // Read on render: the component renders when the count changes, which is when rows are added.
  const { metas } = readMetas();
  const spacers = useMemo(() => spacerHeights(heights, metas, win), [heights, metas, win]);

  const schedule = useCallback(() => {
    if (frameRef.current) return;
    frameRef.current = requestAnimationFrame(() => {
      frameRef.current = 0;
      stepRef.current();
    });
  }, []);

  const commit = useCallback(
    (next: RowWindow | null, sync: boolean) => {
      const viewport = viewportRef.current;
      if (viewport) {
        // Off before the capture: native anchoring would correct the same shift a second time.
        viewport.style.setProperty("overflow-anchor", "none");
        const element = pickAnchorRow(viewport);
        anchorRef.current = element ? holdAnchor(viewport, element) : null;
      }
      pendingRef.current = true;
      const apply = () => setWin(next);
      // Keep the commit and its correction inside the same WebKit rendering frame.
      if (sync) flushSync(apply);
      else startTransition(apply);
    },
    [viewportRef],
  );

  const flushWaiters = useCallback(() => {
    const waiters = waitersRef.current;
    waitersRef.current = [];
    for (const resolve of waiters) resolve();
  }, []);

  const ordinals = useCallback(
    (viewport: HTMLElement, lookup: Metas) => {
      const windowed = winRef.current != null;
      if (!windowed && !ordinalsRef.current) return;
      publishOrdinals(viewport, lookup, windowed);
      ordinalsRef.current = windowed;
    },
    [],
  );

  /**
   * The window is where the reader needs it. Once input has gone quiet, fold what the leading
   * spacer took on (see absorbShift) into scrollTop, so the spacer is its rows' heights again and
   * the scroll range tracks the estimates as they improve. A write no one can see: the anchor
   * stays put, and nothing is scrolling for it to interrupt.
   */
  const settle = useCallback(
    (viewport: HTMLElement, current: RowWindow, lookup: Metas) => {
      const top = topRef.current;
      window.clearTimeout(settleTimerRef.current);
      if (!top || current.start <= spacerRows(current, lookup.metas.length).topFrom) {
        restoreScrollAnchoring(viewport);
        return;
      }
      const wanted = spacerHeights(heights, lookup.metas, current).top;
      const now = Number.parseFloat(top.style.height) || 0;
      if (Math.abs(now - wanted) < 1) {
        slackRef.current = now - wanted;
        restoreScrollAnchoring(viewport);
        return;
      }
      if (scrollActivity.current?.blocked(inputPointerHeld())) {
        settleTimerRef.current = window.setTimeout(schedule, SETTLE_RETRY_MS);
        return;
      }
      viewport.style.setProperty("overflow-anchor", "none");
      const element = pickAnchorRow(viewport);
      const held = element ? holdAnchor(viewport, element) : null;
      top.style.height = `${wanted}px`;
      slackRef.current = 0;
      const shift = held
        ? anchorCorrection(held.sample, sampleAnchor(viewport, held.element), ANCHOR_PRECISION)
        : null;
      adjustScroll(shift ?? 0);
      restoreScrollAnchoring(viewport);
    },
    [adjustScroll, heights, schedule, topRef],
  );

  const step = useCallback(() => {
    const viewport = viewportRef.current;
    if (!viewport || pendingRef.current) return;
    // Another route keeps the chat page alive but hidden. Nothing has a box there: every row
    // would measure zero and the window would walk the whole thread looking for its reach. The
    // resize observer steps again when the viewport is shown. A reader waiting for every row is
    // let go: it cannot read rows that have no boxes, and it would wait for as long as this hides.
    if (viewport.clientHeight === 0) {
      flushWaiters();
      return;
    }
    const lookup = readMetas();
    const total = lookup.metas.length;
    const current = winRef.current;
    heights.resize(viewport.clientWidth);
    const measured = measureRows(viewport, lookup, heights);
    // Also here: a row can replace its root (an edit opening) without the window committing.
    ordinals(viewport, lookup);

    const readerHolds =
      printingRef.current ||
      fullMountHolds.size > 0 ||
      document.documentElement.hasAttribute(FIND_FLAG);
    if (readerHolds) modeRef.current = "fill";
    else if (
      modeRef.current === "fill" &&
      // A full mount still under way is dropped too once nobody is waiting for it (find closed).
      (current == null || waitersRef.current.length === 0) &&
      performance.now() >= releaseAtRef.current
    ) {
      modeRef.current = "slide";
    }
    // The reply being written keeps its row: its effects run the stream.
    const tail = aui.thread().getState().isRunning === true ? total - 1 : null;

    if (current == null) {
      if (waitersRef.current.length > 0) {
        // Resolve after a paint; the request keeps the full mount a little longer.
        releaseAtRef.current = performance.now() + FILL_RELEASE_MS;
        requestAnimationFrame(flushWaiters);
      }
      const next = holdTailTurn(
        planEngage(measured.rows, total, viewport.clientHeight),
        total,
        lookup.tailTurn,
      );
      const held =
        modeRef.current === "fill" || heldSelection(viewport) !== null || inputPointerHeld();
      if (next == null || held) {
        restoreScrollAnchoring(viewport);
        // A reader's hold wakes the window when it ends. The others end silently (the release
        // time passing, a selection cleared, a pointer lifted), so look again.
        if (next != null && !readerHolds) {
          window.clearTimeout(settleTimerRef.current);
          settleTimerRef.current = window.setTimeout(schedule, HOLD_RECHECK_MS);
        }
        return;
      }
      commit(keepPinned(viewport, lookup, next, tail), true);
      return;
    }

    const inWindow = measured.rows.filter(
      (row) => row.index >= current.start && row.index < current.end,
    );
    const heightOf = (index: number) => heights.heightOf(lookup.metas[index]);
    let next: RowWindow | null;
    let trim = true;
    if (modeRef.current === "fill") {
      // A commit also interrupts native wheel momentum, so a full mount waits for a quiet moment.
      if (scrollActivity.current?.blocked(inputPointerHeld())) {
        schedule();
        return;
      }
      // Px from the viewport top up to the window's first row; prepends land above it.
      const frontier = inWindow.length > 0 ? -inWindow[0].top : null;
      const chunk = WEBKIT_GTK
        ? (chunkRef.current = nextMountChunk(
            chunkRef.current, frontier, FILL_CHUNK, WINDOW_REACH_PX))
        : FILL_CHUNK;
      next = planFill(current, total, chunk, heightOf);
    } else {
      chunkRef.current = 0;
      next = planSlide({
        win: current,
        count: total,
        rows: inWindow,
        viewHeight: viewport.clientHeight,
        threadTop: measured.threadTop,
        slack: slackRef.current,
        heightOf,
        maxStep: SLIDE_STEP,
        trim: (trim = heldSelection(viewport) === null && !inputPointerHeld()),
      });
      next = holdTailTurn(next, total, lookup.tailTurn);
    }
    if (next != null) next = keepPinned(viewport, lookup, next, tail);
    if (sameWindow(next, current)) {
      settle(viewport, current, lookup);
      // A held selection or pointer kept rows; nothing announces its end, so look again.
      if (!trim) {
        window.clearTimeout(settleTimerRef.current);
        settleTimerRef.current = window.setTimeout(schedule, HOLD_RECHECK_MS);
      }
      return;
    }
    const trimOnly = next != null && next.start >= current.start && next.end <= current.end;
    // A spacer in view is blank screen: not behind an interruptible transition.
    const first = inWindow[0];
    const last = inWindow[inWindow.length - 1];
    const exposed =
      !first ||
      (current.start > 0 && first.top > 0) ||
      (current.end < total && last.bottom < viewport.clientHeight);
    commit(next, WEBKIT_GTK || trimOnly || exposed);
  }, [aui, commit, flushWaiters, heights, ordinals, readMetas, schedule, settle, viewportRef]);
  useLayoutEffect(() => {
    stepRef.current = step;
  }, [step]);

  // Declared before the commit effect below, which reads what this resets.
  // biome-ignore lint/correctness/useExhaustiveDependencies: resetKey is the thread these belong to
  useLayoutEffect(() => {
    modeRef.current = "slide";
    slackRef.current = 0;
  }, [resetKey]);

  // The leading spacer stands for no rows: the window starts at the first row, or at a pin.
  const atFirstRow = win == null || win.start <= spacerRows(win, count).topFrom;

  // The spacers get their heights before any layout effect runs. A commit that moves the window
  // across a pinned row creates a spacer node (their keys follow the pins), which has no height
  // yet; a row whose own layout effect reads layout would see a thread a few rows long, and the
  // browser would clamp scrollTop to it (seen: 191,305 to 3,214). The refs are not attached yet.
  useInsertionEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    const spacer = (name: string) =>
      viewport.querySelector<HTMLElement>(`:scope > [${SPACER_ATTRIBUTE}="${name}"]`);
    const leading = atFirstRow ? 0 : Math.max(0, spacers.top + slackRef.current);
    spacer("top")?.style.setProperty("height", `${leading}px`);
    spacer("bottom")?.style.setProperty("height", `${spacers.bottom}px`);
  }, [atFirstRow, spacers, viewportRef]);

  // Runs in the commit that changed the window, before paint: size the spacers, then put the
  // reader's anchor back where it was.
  useLayoutEffect(() => {
    winRef.current = win;
    const captured = anchorRef.current;
    anchorRef.current = null;
    const committed = pendingRef.current;
    pendingRef.current = false;
    const viewport = viewportRef.current;
    const top = topRef.current;
    const bottom = bottomRef.current;
    if (!viewport || !top || !bottom) return;
    let spacer = atFirstRow ? 0 : Math.max(0, spacers.top + slackRef.current);
    top.style.height = `${spacer}px`;
    bottom.style.height = `${spacers.bottom}px`;
    let scroll = 0;
    if (captured) {
      const [element, baseline] = captured.element.isConnected
        ? [captured.element, captured.sample]
        : captured.row?.isConnected && captured.rowSample
          ? [captured.row, captured.rowSample]
          : [null, null];
      const shift =
        element && baseline
          ? anchorCorrection(baseline, sampleAnchor(viewport, element), ANCHOR_PRECISION)
          : null;
      if (shift !== null) {
        // The spacer takes the shift, so scrollTop (and a wheel's momentum, or a held scrollbar
        // thumb) is left alone unless the spacer has nothing left to give.
        const taken = absorbShift(spacer, shift, atFirstRow);
        spacer = taken.spacer;
        scroll = taken.scroll;
        top.style.height = `${spacer}px`;
      }
    }
    slackRef.current = atFirstRow ? 0 : spacer - spacers.top;
    // Called with zero too: the hook owns scrollTop and resyncs its bookkeeping on every commit.
    adjustScroll(scroll);
    ordinals(viewport, readMetas());
    if (committed) for (const listener of rowListeners) listener();
    schedule();
  }, [win, atFirstRow, spacers, adjustScroll, ordinals, readMetas, schedule, viewportRef, topRef, bottomRef]);

  /** Behind another route: no boxes, so no step runs and a full mount cannot make progress. */
  const hidden = useCallback(() => viewportRef.current?.clientHeight === 0, [viewportRef]);

  const complete = useCallback(
    () =>
      new Promise<void>((resolve) => {
        // Nothing can read a hidden thread's rows, and a waiter here would never be resolved.
        if (hidden()) {
          resolve();
          return;
        }
        modeRef.current = "fill";
        releaseAtRef.current = performance.now() + FILL_RELEASE_MS;
        if (winRef.current == null) {
          resolve();
          return;
        }
        waitersRef.current.push(resolve);
        schedule();
      }),
    [hidden, schedule],
  );

  const reveal = useCallback(
    (messageId: string): boolean => {
      const lookup = readMetas();
      const index = lookup.index.get(messageId);
      if (index === undefined) return false;
      const current = winRef.current;
      const viewport = viewportRef.current;
      if (
        current == null ||
        !viewport ||
        (index >= current.start && index < current.end) ||
        current.pins?.includes(index)
      ) {
        return true;
      }
      const total = lookup.metas.length;
      const tail = aui.thread().getState().isRunning === true ? total - 1 : null;
      const next = keepPinned(
        viewport,
        lookup,
        { start: index, end: Math.min(total, index + SLIDE_STEP) },
        tail,
      );
      // No anchor: the caller scrolls to the row, so there is no reading position to keep.
      viewport.style.setProperty("overflow-anchor", "none");
      anchorRef.current = null;
      pendingRef.current = true;
      flushSync(() => {
        setWin(next);
      });
      return true;
    },
    [aui, readMetas, viewportRef],
  );

  useLayoutEffect(() => {
    const entry: ActiveWindow = {
      complete,
      reveal,
      // A hidden thread is neither: it mounts nothing until it is shown, and a reader waiting on
      // it (the turn rail of a Compare pane, find) would wait for as long as it stays hidden.
      withholding: () => winRef.current != null && !hidden(),
      filling: () => modeRef.current === "fill" && winRef.current != null && !hidden(),
      wake: schedule,
      viewportRef,
    };
    activeWindows.add(entry);
    return () => {
      activeWindows.delete(entry);
    };
  }, [complete, hidden, reveal, schedule, viewportRef]);

  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    const activity = watchProgressiveScroll(viewport);
    scrollActivity.current = activity;
    const flags = new MutationObserver(schedule);
    flags.observe(document.documentElement, { attributes: true, attributeFilter: [FIND_FLAG] });
    const resize = new ResizeObserver(schedule);
    resize.observe(viewport);
    const media = window.matchMedia("print");
    const beforePrint = () => {
      printingRef.current = true;
      modeRef.current = "fill";
      // Print lays out only what is mounted, even over a commit in flight.
      if (winRef.current != null) commit(null, true);
    };
    const afterPrint = () => {
      printingRef.current = media.matches;
      releaseAtRef.current = performance.now();
      schedule();
    };
    const onKeyDown = (event: KeyboardEvent) => {
      // Select all means the whole conversation, so the rows it should cover start mounting.
      if (!(event.ctrlKey || event.metaKey) || event.key.toLowerCase() !== "a") return;
      const target = event.target as Element | null;
      if (target?.closest?.("input, textarea, [contenteditable=true]")) return;
      // A thread behind another route is not what select all selects.
      if (viewport.clientHeight === 0 || viewport.closest("[inert]")) return;
      void complete();
    };
    const onCopy = () => {
      // The clipboard takes what the selection holds in the DOM. A selection that runs across a
      // spacer (select all pressed a moment ago, or a range set by script) would copy a short
      // conversation, so the rows mount before the copy is read: one slow copy over lost text.
      if (winRef.current == null) return;
      const selection = document.getSelection();
      if (!selection || selection.isCollapsed) return;
      const top = topRef.current;
      const bottom = bottomRef.current;
      const standsIn = (spacer: HTMLElement | null) =>
        spacer !== null && spacer.offsetHeight > 0 && selection.containsNode(spacer, true);
      const fromTop = standsIn(top);
      const toBottom = standsIn(bottom);
      const acrossGap = [
        ...viewport.querySelectorAll<HTMLElement>(`:scope > [${SPACER_ATTRIBUTE}="gap"]`),
      ].some(standsIn);
      if (!fromTop && !toBottom && !acrossGap) return;
      modeRef.current = "fill";
      releaseAtRef.current = performance.now() + FILL_RELEASE_MS;
      commit(null, true);
      // WebKit serialises the selection it resolved before the rows mounted, even where the range
      // it reports has grown to hold them, so the selection is set again; and an end it had moved
      // off a spacer onto the next row goes back to the spacer, which the new rows are inside of.
      if (selection.rangeCount !== 1) return;
      const range = selection.getRangeAt(0).cloneRange();
      const edge = document.createRange();
      if (fromTop && top) {
        edge.setStartBefore(top);
        edge.collapse(true);
        if (range.compareBoundaryPoints(Range.START_TO_START, edge) > 0) range.setStartBefore(top);
      }
      if (toBottom && bottom) {
        edge.setStartAfter(bottom);
        edge.collapse(true);
        if (range.compareBoundaryPoints(Range.END_TO_END, edge) < 0) range.setEndAfter(bottom);
      }
      selection.removeAllRanges();
      selection.addRange(range);
    };
    viewport.addEventListener("scroll", schedule, { passive: true });
    // A pinned row is let go once the reader has left it.
    viewport.addEventListener("focusout", schedule);
    document.addEventListener("copy", onCopy, true);
    document.addEventListener("keydown", onKeyDown, true);
    window.addEventListener("beforeprint", beforePrint);
    window.addEventListener("afterprint", afterPrint);
    schedule();
    return () => {
      activity.dispose();
      scrollActivity.current = null;
      flags.disconnect();
      resize.disconnect();
      viewport.removeEventListener("scroll", schedule);
      viewport.removeEventListener("focusout", schedule);
      document.removeEventListener("copy", onCopy, true);
      document.removeEventListener("keydown", onKeyDown, true);
      window.removeEventListener("beforeprint", beforePrint);
      window.removeEventListener("afterprint", afterPrint);
      cancelAnimationFrame(frameRef.current);
      frameRef.current = 0;
      window.clearTimeout(settleTimerRef.current);
      restoreScrollAnchoring(viewport);
    };
  }, [viewportRef, topRef, bottomRef, schedule, commit, complete]);

  // Settle waiters on unmount, or a DOM capture racing a thread switch waits forever.
  useEffect(() => flushWaiters, [flushWaiters]);

  /** Estimated height of rows `[from, to)`, for a spacer between pinned rows. */
  const gapHeight = useCallback(
    (from: number, to: number): number => {
      const { metas: rows } = readMetas();
      let height = 0;
      for (let index = from; index < Math.min(to, rows.length); index += 1) {
        height += heights.heightOf(rows[index]);
      }
      return height;
    },
    [readMetas, heights],
  );

  return { win, gapHeight };
}

/** Drop-in for `ThreadPrimitive.Messages` that mounts only the rows near the reader. */
export const ProgressiveMessages: FC<{
  renderMessage: () => ReactElement;
  resetKey: string | undefined;
  viewportRef: RefObject<HTMLElement | null>;
}> = memo(
  ({ renderMessage, resetKey, viewportRef }) => {
    const count = useAuiState(({ thread }) => thread.messages.length);
    const topRef = useRef<HTMLDivElement>(null);
    const bottomRef = useRef<HTMLDivElement>(null);
    const { win, gapHeight } = useThreadWindow(count, resetKey, viewportRef, topRef, bottomRef);
    const aui = useAui();
    const gate = useMemo(() => createRowNotificationGate(aui), [aui]);

    return useMemo(() => {
      if (count === 0) return null;
      const message = renderMessage();
      // A spacer is keyed by where it sits among the rows, so no commit ever has to move a row
      // past one: moving a row's node would take its focus and stop its media.
      return threadSlots(win, count).map((slot) => {
        if (slot.kind === "row") {
          const index = slot.index;
          return (
            <AuiProvider key={index} value={gate.row(index)}>
              <MessageByIndexProvider index={index}>{message}</MessageByIndexProvider>
            </AuiProvider>
          );
        }
        // The heights of these two are written in the window's layout effect, never by React: it
        // also moves them to hold the reader's anchor, and a re-render must not put the old
        // value back.
        if (slot.name === "top") {
          const key = slot.from === 0 ? "top" : `top:${slot.from}`;
          return <div key={key} ref={topRef} data-thread-spacer="top" aria-hidden="true" style={{ flexShrink: 0 }} />;
        }
        if (slot.name === "bottom") {
          const key = slot.to === count ? "bottom" : `bottom:${slot.to}`;
          return <div key={key} ref={bottomRef} data-thread-spacer="bottom" aria-hidden="true" style={{ flexShrink: 0 }} />;
        }
        return (
          <div
            key={`gap:${slot.from}`}
            data-thread-spacer="gap"
            aria-hidden="true"
            style={{ flexShrink: 0, height: gapHeight(slot.from, slot.to) }}
          />
        );
      });
    }, [count, win, renderMessage, gate, gapHeight]);
  },
  (prev, next) =>
    prev.resetKey === next.resetKey &&
    prev.renderMessage === next.renderMessage,
);

ProgressiveMessages.displayName = "ProgressiveMessages";
