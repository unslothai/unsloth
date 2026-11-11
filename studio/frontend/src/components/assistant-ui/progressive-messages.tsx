// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Thread window glue (maths in thread-window.ts). One shared propless slot element lets React
// skip unchanged rows; a components object would re-render every row.

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
  numberedEquations,
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
/** WebKitGTK commits inside the frame, so it takes fewer rows per step. */
const SLIDE_STEP = WEBKIT_GTK ? 2 : 8;
const FILL_CHUNK = WEBKIT_GTK ? 2 : CHUNK_MESSAGES;
/** Sub-pixel shifts are real here: the spacer takes them, where scrollTop would round them. */
const ANCHOR_PRECISION = 1 / 64;
const FILL_RELEASE_MS = 3000;
const SETTLE_RETRY_MS = 350;
const HOLD_RECHECK_MS = 1000;
const FIND_FLAG = "data-find-bar-open";
// Not `[data-role]`: a prompt being edited renders as an EditComposer, which has no role.
const ROW_SELECTOR = "[data-message-id]";
const SPACER_ATTRIBUTE = "data-thread-spacer";
// An iframe (an inline MCP app) holds state and a bridge port a remount would lose.
const PINNED_CONTENT = "textarea, [contenteditable=true], iframe";

/** Module-level: consumers are imperative DOM readers, and a thread can mount twice (Compare). */
interface ActiveWindow {
  complete: (signal?: AbortSignal) => Promise<void>;
  reveal: (messageId: string) => boolean;
  withholding: () => boolean;
  filling: () => boolean;
  wake: () => void;
  viewportRef: RefObject<HTMLElement | null>;
}

const activeWindows = new Set<ActiveWindow>();
const rowListeners = new Set<() => void>();
const fullMountHolds = new Set<symbol>();

/** Keeps every row mounted until released, for a reader of the whole DOM over time. */
export function holdProgressiveMounts(): () => void {
  const hold = Symbol("full mount");
  fullMountHolds.add(hold);
  for (const entry of activeWindows) entry.wake();
  return () => {
    if (fullMountHolds.delete(hold)) for (const entry of activeWindows) entry.wake();
  };
}

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
  /** A reader that stops waiting (Find closed) lets the window slide again at once. */
  signal?: AbortSignal,
): Promise<void> {
  const wanted = () =>
    [...activeWindows].filter((entry) =>
      wants === undefined || wants(entry.viewportRef.current));
  // An empty set is only believed after SEARCH_MS: a thread still loading history also looks
  // empty. A set that was non-empty and drained is believed immediately.
  const deadline = Date.now() + PROGRESSIVE_MOUNT_SEARCH_MS;
  let observed = false;
  for (;;) {
    if (signal?.aborted) return;
    const pending = wanted();
    if (pending.length > 0) {
      observed = true;
      await Promise.all(pending.map((entry) => entry.complete(signal)));
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

/** A settled window or a hidden thread withholds rows without being pending. */
export function hasPendingProgressiveMounts(): boolean {
  for (const entry of activeWindows) if (entry.filling()) return true;
  return false;
}

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
  tailTurn: number;
};

const metaCache = new WeakMap<object, RowMeta>();

function metaOf(message: ThreadMessage, index: number): RowMeta {
  const cached = metaCache.get(message);
  if (cached) return cached;
  const meta: RowMeta = {
    id: message.id ?? `#${index}`,
    role: message.role ?? "assistant",
    weight: messageWeight(message.content),
    equations: numberedEquations(message.content),
    // Unknown shapes count as rendering: room for a row that turns out empty costs nothing.
    renders:
      message.role == null ||
      rendersAsRow(message.role, message.composer?.isEditing === true),
  };
  metaCache.set(message, meta);
  return meta;
}

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
  rows: MountedRow[];
  threadTop: number;
};

function rowIndex(child: Element, lookup: Metas): number | undefined {
  if (!child.matches(ROW_SELECTOR)) return undefined;
  return lookup.index.get(child.getAttribute("data-message-id") ?? "");
}

/** Height = top to the top of what follows, so the margins between rows are counted. */
function measureRows(viewport: HTMLElement, lookup: Metas, heights: RowHeights): Measured {
  const origin = viewport.getBoundingClientRect().top;
  const rows: MountedRow[] = [];
  let threadTop: number | null = null;
  let open: MountedRow | null = null;
  let openRow: Element | null = null;
  const close = (height: number) => {
    if (!open || !openRow) return;
    const meta = lookup.metas[open.index];
    // A re-render that changed the row's equations changed its height too.
    if (heights.measure(meta, Math.max(0, height))) {
      heights.countEquations(meta.id, {
        html: openRow.querySelectorAll(".eqn-num").length,
        mathml: openRow.querySelectorAll(".mml-eqn-num").length,
      });
    }
  };
  for (const child of viewport.children) {
    const index = rowIndex(child, lookup);
    if (index === undefined && !child.hasAttribute(SPACER_ATTRIBUTE)) continue;
    const box = child.getBoundingClientRect();
    const top = box.top - origin;
    threadTop ??= top;
    if (open) close(top - open.top);
    open = null;
    if (index === undefined) continue;
    open = { index, top, bottom: box.bottom - origin };
    openRow = child;
    rows.push(open);
  }
  if (open) close(open.bottom - open.top);
  return { rows, threadTop: threadTop ?? 0 };
}

// A dialog opened from a row (an attachment viewer) is portaled out of it, so its focus and
// media are invisible to isPinned: while one is open, no mounted row is let go.
const OPEN_DIALOG = '[role="dialog"][data-state="open"], [role="alertdialog"][data-state="open"]';

function heldSelection(viewport: HTMLElement): Range | null {
  const selection = document.getSelection();
  if (selection === null || selection.isCollapsed || selection.rangeCount === 0) return null;
  const range = selection.getRangeAt(0);
  return range.intersectsNode(viewport) ? range : null;
}

function isPinned(row: Element, selection: Range | null): boolean {
  if (row.contains(document.activeElement)) return true;
  if (row.querySelector(PINNED_CONTENT) !== null) return true;
  if (selection?.intersectsNode(row)) return true;
  for (const media of row.querySelectorAll<HTMLMediaElement>("video, audio")) {
    if (!media.paused) return true;
  }
  return false;
}

/** Rows the reader is working in stay mounted where they are, apart from the window, so a jump
 *  keeps an unsaved edit without mounting what lies between. `tail` is a reply being streamed. */
function keepPinned(
  viewport: HTMLElement,
  lookup: Metas,
  next: RowWindow,
  tail: number | null,
): RowWindow {
  const selection = heldSelection(viewport);
  const dialog = document.querySelector(OPEN_DIALOG) !== null;
  const pins: number[] = [];
  for (const child of viewport.children) {
    const index = rowIndex(child, lookup);
    if (index === undefined || (index >= next.start && index < next.end)) continue;
    if (dialog || index === tail || isPinned(child, selection)) pins.push(index);
  }
  return withPins(next, pins, lookup.metas.length);
}

/** WAI-ARIA ordinals for a windowed list. Written here, not by the rows: the total changes on
 *  every append, and a row must not re-render for it. */
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
): {
  win: RowWindow | null;
  gapHeight: (from: number, to: number) => number;
  counterIncrement: (from: number, to: number) => string | undefined;
} {
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
  /** "fill" mounts every row for a DOM reader (find, print, select all). */
  const modeRef = useRef<"slide" | "fill">("slide");
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
      setWin(open());
    } else if (win != null) {
      const followed = followCount(win, previousCount, count);
      if (followed !== win) setWin(followed);
    }
  }

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

  /** Once input is quiet, folds the leading spacer's slack into scrollTop, anchor held. */
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
    // Hidden behind another route: every row measures zero (the resize observer steps on show),
    // and a waiter would wait for as long as it hides.
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
        // Only a reader's hold wakes the window when it ends; the others end silently.
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
        trim: (trim =
          heldSelection(viewport) === null &&
          !inputPointerHeld() &&
          document.querySelector(OPEN_DIALOG) === null),
      });
      next = holdTailTurn(next, total, lookup.tailTurn);
    }
    if (next != null) next = keepPinned(viewport, lookup, next, tail);
    if (sameWindow(next, current)) {
      settle(viewport, current, lookup);
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

  // Spacers are sized before any layout effect: a new spacer node has no height yet, and a row
  // reading layout in its own effect would let the browser clamp scrollTop to a short thread.
  useInsertionEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    const spacer = (name: string) =>
      viewport.querySelector<HTMLElement>(`:scope > [${SPACER_ATTRIBUTE}="${name}"]`);
    const leading = atFirstRow ? 0 : Math.max(0, spacers.top + slackRef.current);
    spacer("top")?.style.setProperty("height", `${leading}px`);
    spacer("bottom")?.style.setProperty("height", `${spacers.bottom}px`);
  }, [atFirstRow, spacers, viewportRef]);

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
        // The spacer takes the shift, so wheel momentum or a held thumb is left alone.
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
    (signal?: AbortSignal) =>
      new Promise<void>((resolve) => {
        if (signal?.aborted) {
          resolve();
          return;
        }
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
        signal?.addEventListener(
          "abort",
          () => {
            waitersRef.current = waitersRef.current.filter((waiter) => waiter !== resolve);
            // Nobody else waiting: the release deadline is all that holds the full mount now.
            if (waitersRef.current.length === 0) releaseAtRef.current = performance.now();
            resolve();
            schedule();
          },
          { once: true },
        );
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
      // A hidden thread is neither: a reader waiting on it would wait for as long as it hides.
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
    // A mounted row that shrinks (a code block collapsed) can bare a spacer with no scroll to
    // wake the window. Only a shrink: a streamed reply grows on every chunk.
    const rowHeights = new WeakMap<Element, number>();
    const rowResize = new ResizeObserver((entries) => {
      let shrank = false;
      for (const entry of entries) {
        const height = entry.contentRect.height;
        const before = rowHeights.get(entry.target);
        rowHeights.set(entry.target, height);
        if (before !== undefined && height < before - 1) shrank = true;
      }
      if (shrank) schedule();
    });
    const watchRows = (nodes: Iterable<Node>) => {
      for (const node of nodes) {
        if (node instanceof Element && node.matches(ROW_SELECTOR)) rowResize.observe(node);
      }
    };
    const rowsChanged = new MutationObserver((records) => {
      for (const record of records) {
        for (const node of record.removedNodes) {
          if (node instanceof Element) rowResize.unobserve(node);
        }
        watchRows(record.addedNodes);
      }
    });
    rowsChanged.observe(viewport, { childList: true });
    watchRows(viewport.children);
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
      if (!(event.ctrlKey || event.metaKey) || event.key.toLowerCase() !== "a") return;
      const target = event.target as Element | null;
      if (target?.closest?.("input, textarea, [contenteditable=true]")) return;
      if (viewport.clientHeight === 0 || viewport.closest("[inert]")) return;
      void complete();
    };
    const onCopy = () => {
      // A selection across a spacer would copy a short conversation: mount before it is read.
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
      const lookup = readMetas();
      const { topFrom, bottomTo } = spacerRows(winRef.current, lookup.metas.length);
      modeRef.current = "fill";
      releaseAtRef.current = performance.now() + FILL_RELEASE_MS;
      commit(null, true);
      // WebKit serialises the selection resolved before the mount, so set it again, with an end
      // it moved off a spacer put on the first (or last) row that spacer stood for. Rows, not the
      // spacers: a spacer keyed by a pin is replaced by the mount.
      if (selection.rangeCount !== 1) return;
      const rowAt = (from: number, step: 1 | -1): Element | null => {
        for (let index = from; index >= 0 && index < lookup.metas.length; index += step) {
          const id = lookup.metas[index].id;
          const row = viewport.querySelector(`:scope > [data-message-id="${CSS.escape(id)}"]`);
          if (row) return row;
        }
        return null;
      };
      const range = selection.getRangeAt(0).cloneRange();
      const edge = document.createRange();
      const first = fromTop ? rowAt(topFrom, 1) : null;
      if (first) {
        edge.setStartBefore(first);
        edge.collapse(true);
        if (range.compareBoundaryPoints(Range.START_TO_START, edge) > 0) range.setStartBefore(first);
      }
      const last = toBottom ? rowAt(bottomTo - 1, -1) : null;
      if (last) {
        edge.setStartAfter(last);
        edge.collapse(true);
        if (range.compareBoundaryPoints(Range.END_TO_END, edge) < 0) range.setEndAfter(last);
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
      rowResize.disconnect();
      rowsChanged.disconnect();
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
  }, [viewportRef, topRef, bottomRef, schedule, commit, complete, readMetas]);

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

  /** Equation counter increments of rows `[from, to)`, for the spacer standing in for them. */
  const counterIncrement = useCallback(
    (from: number, to: number): string | undefined => {
      const { html, mathml } = heights.equationsIn(readMetas().metas, from, to);
      return html || mathml ? `katexEqnNo ${html} mmlEqnNo ${mathml}` : undefined;
    },
    [readMetas, heights],
  );

  return { win, gapHeight, counterIncrement };
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
    const { win, gapHeight, counterIncrement } = useThreadWindow(
      count,
      resetKey,
      viewportRef,
      topRef,
      bottomRef,
    );
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
        // Heights are written in the layout effect, never by React: a re-render must not undo them.
        if (slot.name === "top") {
          const key = slot.from === 0 ? "top" : `top:${slot.from}`;
          return <div key={key} ref={topRef} data-thread-spacer="top" aria-hidden="true" style={{ flexShrink: 0, counterIncrement: counterIncrement(slot.from, slot.to) }} />;
        }
        if (slot.name === "bottom") {
          const key = slot.to === count ? "bottom" : `bottom:${slot.to}`;
          return <div key={key} ref={bottomRef} data-thread-spacer="bottom" aria-hidden="true" style={{ flexShrink: 0, counterIncrement: counterIncrement(slot.from, slot.to) }} />;
        }
        return (
          <div
            key={`gap:${slot.from}`}
            data-thread-spacer="gap"
            aria-hidden="true"
            style={{
              flexShrink: 0,
              height: gapHeight(slot.from, slot.to),
              counterIncrement: counterIncrement(slot.from, slot.to),
            }}
          />
        );
      });
    }, [count, win, renderMessage, gate, gapHeight, counterIncrement]);
  },
  (prev, next) =>
    prev.resetKey === next.resetKey &&
    prev.renderMessage === next.renderMessage,
);

ProgressiveMessages.displayName = "ProgressiveMessages";
