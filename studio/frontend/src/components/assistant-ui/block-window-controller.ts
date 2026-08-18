// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * One streaming reasoning pane document's block window, as a plain object graph.
 *
 * Deliberately free of React and of JSX. React reads it through a context whose value NEVER
 * changes identity while the window is live (see block-window-context.tsx): if the window state
 * were context state, every one of the thousands of block components would re-render on every
 * window move, which is the cost being removed. Instead each block subscribes to its own index
 * and only the handful whose mounted state flips is told anything.
 *
 * Being plain is also what makes it testable: the only browser API it needs is
 * `getBoundingClientRect`, three observers and four numbers off the scroll container, all of
 * which a test can supply.
 */

import { flushSync } from "react-dom";

import {
  type BlockWindowState,
  PANE_BOTTOM_THRESHOLD_PX,
  WINDOW_ROOT_MARGIN_PX,
  blockWindowFlippedRange,
  chooseBlockWindowStart,
  createBlockWindowState,
  isBlockMounted,
  recordBlockContent,
  recordBlockOffset,
  resetBlockWindow,
  setBlockWindowStart,
} from "./block-window";

/** How many mounted blocks at the top of the window are watched by the observer. */
const OBSERVED_EDGE_BLOCKS = 3;

const NO_UNSUBSCRIBE = (): void => {};

/**
 * One Streamdown document's window.
 *
 * Deliberately not a React component: the block components read it through a context whose value
 * NEVER changes identity while the window is live. If the window state were context state, every
 * one of the thousands of block components would re-render on every window move, which is the
 * cost being removed. Instead each block subscribes to its own index and only the handful whose
 * mounted state flips is told anything.
 */
export class BlockWindowController {
  private readonly state: BlockWindowState = createBlockWindowState();
  private readonly markers = new Map<number, HTMLElement>();
  private readonly markerRefs = new Map<
    number,
    (element: HTMLElement | null) => void
  >();
  private readonly blockListeners = new Map<number, Set<() => void>>();
  private readonly documentListeners = new Set<() => void>();
  private readonly observedEdge = new Set<Element>();
  private pane: HTMLElement | null = null;
  private origin: HTMLElement | null = null;
  private intersection: IntersectionObserver | null = null;
  private paneWidth = 0;
  private pendingReset = false;
  private pendingCompensation: {
    index: number;
    top: number;
    expectStart: number;
  } | null = null;
  /**
   * The last residual the compensation had to correct, in CSS pixels. The spacer is sized to the
   * exact height of the blocks it replaces, so this should be 0 and the compensation should be
   * dead code; it is measured rather than assumed, and a test asserts the 0.
   */
  lastResidualPx = 0;

  // ── what React reads ──────────────────────────────────────────────

  isMounted(index: number): boolean {
    return isBlockMounted(this.state, index);
  }

  spacerHeight(): number {
    return this.state.spacerHeight;
  }

  windowStart(): number {
    return this.state.start;
  }

  subscribeBlock(index: number, onChange: () => void): () => void {
    let listeners = this.blockListeners.get(index);
    if (!listeners) {
      listeners = new Set();
      this.blockListeners.set(index, listeners);
    }
    listeners.add(onChange);
    return () => {
      listeners.delete(onChange);
      if (listeners.size === 0) {
        this.blockListeners.delete(index);
      }
    };
  }

  subscribeDocument = (onChange: () => void): (() => void) => {
    this.documentListeners.add(onChange);
    return () => {
      this.documentListeners.delete(onChange);
    };
  };

  /** A stable ref callback per index, so a block's marker registration is not a new prop. */
  markerRef(index: number): (element: HTMLElement | null) => void {
    const existing = this.markerRefs.get(index);
    if (existing) {
      return existing;
    }
    const ref = (element: HTMLElement | null): void => {
      if (element) {
        this.markers.set(index, element);
      } else {
        this.markers.delete(index);
      }
    };
    this.markerRefs.set(index, ref);
    return ref;
  }

  /** The spacer, which is both the thing that replaces the dropped blocks and the measuring origin. */
  spacerRef = (element: HTMLElement | null): void => {
    this.origin = element;
  };

  /**
   * Record a block's content, and reset if it shows the renderer re-segmented the document behind
   * the live edge. Every block reports, mounted or not: a withheld block still renders (it returns
   * null), so the whole document is observed even though only a suffix of it is in the DOM.
   */
  reportContent(index: number, content: string): void {
    if (recordBlockContent(this.state, index, content)) {
      this.invalidate();
    }
  }

  /** Throw away every frozen height and remount the document on the next frame. */
  invalidate(): void {
    this.pendingReset = true;
    this.measure();
  }

  // ── lifecycle ─────────────────────────────────────────────────────

  attach(pane: HTMLElement | null): () => void {
    this.pane = pane;
    if (!pane) {
      return NO_UNSUBSCRIBE;
    }
    this.paneWidth = pane.clientWidth;

    // The window has to be re-decided on every commit that changed the pane, and the provider
    // does not re-render when its children stream. A MutationObserver is what sees those commits.
    // It runs in the microtask checkpoint after the commit, before paint.
    const mutations = new MutationObserver(() => {
      this.measure();
    });
    mutations.observe(pane, {
      childList: true,
      characterData: true,
      subtree: true,
    });

    // A WIDTH change reflows every block, so every frozen height is wrong at once. Answer by
    // resetting: that is what makes "a frozen height that is never revalidated" impossible here.
    // A height change is the pane's own cap animating and means nothing.
    const resizes = new ResizeObserver(() => {
      if (pane.clientWidth === this.paneWidth) {
        return;
      }
      this.paneWidth = pane.clientWidth;
      this.invalidate();
    });
    resizes.observe(pane);

    // Scrolling produces no mutations, so this is what notices a reader moving. Scroll events are
    // deliberately not used. flushSync because an observation is delivered inside the rendering
    // steps: without it React would commit the remount after the frame the reader already saw,
    // and scrolling up would flash the spacer.
    this.intersection = new IntersectionObserver(
      () => {
        flushSync(() => {
          this.measure();
        });
      },
      {
        root: pane,
        rootMargin: `${WINDOW_ROOT_MARGIN_PX}px 0px ${WINDOW_ROOT_MARGIN_PX}px 0px`,
      },
    );

    this.measure();
    return () => {
      mutations.disconnect();
      resizes.disconnect();
      this.intersection?.disconnect();
      this.intersection = null;
      this.observedEdge.clear();
      this.pane = null;
    };
  }

  // ── the pass ──────────────────────────────────────────────────────

  /**
   * Measure what is mounted, then decide where the window starts.
   *
   * Order matters and is the whole trick. Offsets are recorded from the DOM as it stands NOW,
   * before anything is dropped, so the height of the range that is about to be removed comes from
   * a frame in which those blocks were still laid out. Nothing is derived from a scrollHeight
   * delta, which streaming would poison.
   */
  private measure(): void {
    const pane = this.pane;
    const origin = this.origin;
    if (!pane || !origin) {
      return;
    }

    if (this.pendingReset) {
      this.pendingReset = false;
      const previousStart = this.state.start;
      resetBlockWindow(this.state);
      this.notifyBlocks(0, previousStart);
      this.notifyDocument();
      return;
    }

    const originTop = origin.getBoundingClientRect().top;
    const paneTop = pane.getBoundingClientRect().top;
    for (const [index, slot] of this.markers) {
      const marker = slot.firstElementChild;
      if (!marker) {
        // An empty block. Streamdown's splitter emits "\n\n" blocks that render nothing, so a
        // slot without an element child is normal and has no height to record.
        continue;
      }
      recordBlockOffset(
        this.state,
        index,
        marker.getBoundingClientRect().top - originTop,
      );
    }

    // The pane's scroll position, in the same content coordinates as the offsets above.
    const viewTop = paneTop - originTop;
    const nextStart = chooseBlockWindowStart(this.state, viewTop);
    this.refreshObservedEdge();
    if (nextStart === this.state.start) {
      return;
    }

    // The anchor has to be an element that exists on BOTH sides of the commit, because the
    // compensation subtracts its position after from its position before.
    //
    // Moving the window FORWARD, the new start is already mounted and the old one is about to
    // go, so the new start is the anchor. Moving it BACKWARD, which is what a reader scrolling
    // up does, the new start is NOT MOUNTED YET: `markerElement(nextStart)` is null and the old
    // code fell back to the spacer, whose top is the top of the whole content. It then compared
    // the spacer's position before against the newly mounted BLOCK's position after, two
    // different elements, and computed a residual the size of the spacer. Measured: a reader
    // 1,440px up the pane was moved 8,271px down in one write, which put them at the bottom,
    // which re-armed the pane's autoscroll, which pinned them there for the rest of the
    // generation. Scroll-back was not degraded, it was inverted.
    //
    // The highest of the two starts is mounted in both states in either direction, so that is
    // the anchor.
    const previousStart = this.state.start;
    const anchorIndex = Math.max(previousStart, nextStart);
    const anchor = this.markerElement(anchorIndex) ?? origin;
    const anchorTop = anchor.getBoundingClientRect().top - paneTop;
    if (!setBlockWindowStart(this.state, nextStart)) {
      return;
    }
    this.pendingCompensation = {
      index: anchorIndex,
      top: anchorTop,
      expectStart: nextStart,
    };
    const flipped = blockWindowFlippedRange(previousStart, nextStart);
    this.notifyBlocks(flipped.from, flipped.to);
    this.notifyDocument();
  }

  /**
   * Put the reader back where they were, explicitly, in the same layout pass as the mutation.
   *
   * CSS `overflow-anchor` is NOT relied on and is switched off on the pane: Playwright's WebKit
   * implements it and real Safari does not, so the sanctioned test proxy would show scroll
   * anchoring working while the WebKitGTK embed that ships in Tauri drifted.
   */
  settleAfterCommit(): void {
    // The blocks that just entered the window registered their markers in this same commit, so
    // this is the first moment the observer can be pointed at the new top of the window.
    this.refreshObservedEdge();
    const pending = this.pendingCompensation;
    const pane = this.pane;
    if (!pending || !pane) {
      return;
    }
    // The window is decided in a MutationObserver microtask and applied by a React update, which
    // does not have to land on the very next commit. A token arriving in between commits this
    // component too, and answering THAT commit would compare the anchor against a DOM the drop
    // has not reached yet and then throw the correction away. Wait until the registered markers
    // say the drop is on screen.
    // Wait until the DOM matches the window that was decided, in EITHER direction. The old
    // test was "the lowest mounted index has reached the anchor", which only describes a window
    // moving forward; a window moving backward mounts blocks BELOW the anchor, so that test
    // never came true and the correction for a scroll-back was dropped every time.
    if (this.lowestMarkerIndex() !== Math.max(pending.expectStart, 1)) {
      return;
    }
    this.pendingCompensation = null;
    const anchor = this.markerElement(pending.index) ?? this.origin;
    if (!anchor) {
      return;
    }
    const after =
      anchor.getBoundingClientRect().top - pane.getBoundingClientRect().top;
    const residual = after - pending.top;
    this.lastResidualPx = residual;
    if (residual === 0) {
      return;
    }
    // While the pane is pinned to the bottom the autoscroll observer puts it back on this same
    // frame, and a write here would only look like the reader scrolling up to the handler that
    // watches for exactly that. Nothing is visible in that state either: the bottom is the bottom.
    const distanceFromBottom =
      pane.scrollHeight - pane.scrollTop - pane.clientHeight;
    if (distanceFromBottom <= PANE_BOTTOM_THRESHOLD_PX) {
      return;
    }
    pane.scrollTop += residual;
  }

  // ── plumbing ──────────────────────────────────────────────────────

  private markerElement(index: number): Element | null {
    return this.markers.get(index)?.firstElementChild ?? null;
  }

  /** The lowest block index currently registered, i.e. what the DOM says the window start is. */
  private lowestMarkerIndex(): number {
    let lowest = Number.POSITIVE_INFINITY;
    for (const index of this.markers.keys()) {
      if (index < lowest) {
        lowest = index;
      }
    }
    return lowest;
  }

  /**
   * Watch the spacer and the first few mounted blocks, so a reader moving upward is noticed
   * before the spacer can come into view.
   */
  private refreshObservedEdge(): void {
    const intersection = this.intersection;
    const origin = this.origin;
    if (!intersection || !origin) {
      return;
    }
    const wanted = new Set<Element>([origin]);
    const edge: number[] = [];
    for (const index of this.markers.keys()) {
      if (index >= this.state.start) {
        edge.push(index);
      }
    }
    edge.sort((left, right) => left - right);
    for (const index of edge.slice(0, OBSERVED_EDGE_BLOCKS)) {
      const marker = this.markerElement(index);
      if (marker) {
        wanted.add(marker);
      }
    }
    for (const element of this.observedEdge) {
      if (!wanted.has(element)) {
        intersection.unobserve(element);
        this.observedEdge.delete(element);
      }
    }
    for (const element of wanted) {
      if (!this.observedEdge.has(element)) {
        intersection.observe(element);
        this.observedEdge.add(element);
      }
    }
  }

  private notifyBlocks(from: number, to: number): void {
    for (let index = from; index < to; index += 1) {
      const listeners = this.blockListeners.get(index);
      if (!listeners) {
        continue;
      }
      for (const listener of listeners) {
        listener();
      }
    }
  }

  private notifyDocument(): void {
    for (const listener of this.documentListeners) {
      listener();
    }
  }
}
