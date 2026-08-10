// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What the bottom-right overlay stack has to keep clear of, in viewport
// coordinates. The Live monitor is draggable and resizable and defaults to that
// same corner; the chat composer docks to the bottom of the same column once a
// thread has turns. Each publishes its box here while it is mounted.

import { useCallback, useEffect, useMemo, useState } from "react";
import { create } from "zustand";

export type MonitorFrame = {
  left: number;
  top: number;
  right: number;
  bottom: number;
};

/**
 * Whose box this is. Identity matters twice over: reopening the monitor during
 * its exit animation leaves two panels mounted at once and the one on its way
 * out unmounts last, so its cleanup must clear only its own; and the monitor
 * and the composer are published side by side.
 */
export type MonitorFramePublisher = object;

interface MonitorFrameState {
  /** Every published box, kept apart. Deliberately not merged into one: the
   *  rectangle around a left-hand monitor and a bottom-right composer spans
   *  the empty space between them, and reading that as a single obstacle
   *  lifted the stack to its cap for a monitor nowhere near its column.
   *  `stackGeometry` folds them one at a time. */
  frames: ReadonlyMap<MonitorFramePublisher, MonitorFrame>;
  setFrame: (publisher: MonitorFramePublisher, frame: MonitorFrame) => void;
  /** Drops only this publisher's box; the others still count. */
  clearFrame: (publisher: MonitorFramePublisher) => void;
}

function sameFrame(a: MonitorFrame | null, b: MonitorFrame | null): boolean {
  if (a === null || b === null) return a === b;
  return (
    a.left === b.left &&
    a.top === b.top &&
    a.right === b.right &&
    a.bottom === b.bottom
  );
}

export const useMonitorFrameStore = create<MonitorFrameState>((set) => ({
  frames: new Map(),
  // Written from a layout effect on every reconcile, so no-op writes must not
  // notify: the overlay stack re-renders on this.
  setFrame: (publisher, frame) =>
    set((state) => {
      if (sameFrame(state.frames.get(publisher) ?? null, frame)) return state;
      return { frames: new Map(state.frames).set(publisher, frame) };
    }),
  clearFrame: (publisher) =>
    set((state) => {
      if (!state.frames.has(publisher)) return state;
      const frames = new Map(state.frames);
      frames.delete(publisher);
      return { frames };
    }),
}));

// The corner stack's own inset, and the gap left between it and the monitor.
const STACK_INSET = 16;
const STACK_GAP = 8;
// Widest overlay the stack holds: the update banners, at max-w-[448px].
const STACK_WIDTH = 448;
// Never lift so far that the stack itself is pushed off the top.
const MIN_STACK_ROOM = 120;
// Fallback for "how much room does the stack need under a box" before it has been
// measured. Deliberately small: guessing high is what parked the loaded models card in
// the middle of a short window, with the corner underneath it empty and big enough to
// hold it. The real height arrives from the stack's own ResizeObserver on the next frame.
const ASSUMED_STACK_HEIGHT = 56;
// The monitor's own controls, measured from its top edge: 12px of panel
// padding, a 24px control row (the drag handle and the size-6 Close button),
// an 8px rule and an 8px margin come to 54. Rounded up so a font or a border
// cannot eat the margin.
const MONITOR_HEADER = 64;
// Its native `resize` grip, in the opposite corner. Chromium's hit area
// reaches 15px in from the bottom-right corner and Firefox's 12px, so one
// STACK_INSET of clearance takes the stack off all of it.
const MONITOR_GRIP = 16;

/**
 * How far above the bottom edge the overlay stack must sit to clear the Live
 * monitor. The monitor defaults to the same corner, so without this the
 * update banners, the download panel and the loaded models card all land
 * underneath it.
 */
/** Whether the monitor overlaps the column the stack's overlays occupy. */
function inStackColumn(frame: MonitorFrame, viewportWidth: number): boolean {
  const columnLeft = viewportWidth - STACK_INSET - STACK_WIDTH;
  return frame.right > columnLeft && frame.left < viewportWidth - STACK_INSET;
}

/**
 * Whether the box leaves the stack too little room to sit under it, in which
 * case the stack has to go over it instead. Anything higher leaves the corner
 * usable, however far down the screen it starts.
 *
 * This is the whole difference between the two composer layouts. Docked under a
 * thread it crowds the corner and has to be dodged, or the card covers Send. On
 * an empty chat the welcome layout pads it well clear of the bottom, and
 * lifting over it there is what put the banners in the middle of the page with
 * the corner underneath them empty.
 *
 * Measured against what the stack actually is, not a fixed guess. A guess is
 * what put the loaded models card in the middle of a short window: the welcome
 * composer left 83px under it, the card is 80px tall and fits there, but the
 * constant said 120 so the stack lifted clear over the composer's top edge and
 * left the corner it was dodging empty. `neededRoom` comes from the stack's own
 * ResizeObserver (see useStackGeometry), so a taller stack still gets dodged.
 *
 * Asked against the inset in force, never a fixed one: lifting over one box
 * moves the stack up into the next, and a box that had room at the corner may
 * have none there.
 */
function reachesStack(
  frame: MonitorFrame,
  viewportHeight: number,
  bottomInset: number,
  neededRoom: number,
): boolean {
  // At least a pixel, even for an empty stack. Zero would let the capped branch
  // hand back a max-height of exactly 0, which is a valid value browsers keep,
  // so the stack would be there and invisible rather than dodged.
  return roomBelow(frame, viewportHeight, bottomInset) < Math.max(1, neededRoom);
}

/** The inset that clears this box's top edge, whether or not it fits there. */
function liftOver(frame: MonitorFrame, viewportHeight: number): number {
  return viewportHeight - frame.top + STACK_GAP;
}

/**
 * Whether the stack still has its floor above the box once lifted clear of it.
 * A box reaching the top of the viewport has nothing above it to lift into.
 */
function liftFits(frame: MonitorFrame, viewportHeight: number): boolean {
  return liftOver(frame, viewportHeight) <= viewportHeight - MIN_STACK_ROOM;
}

/**
 * The inset that dodges this box: over its top edge while the stack still fits
 * there, and otherwise inside it, above its own resize grip.
 *
 * Bounding the lift to the stack's floor rather than giving it up is what the
 * second branch replaces. That parked the stack across the box's own top edge,
 * over the very controls it was dodging: a monitor resized to fill the viewport
 * had its Close button swallowed by the loaded models card. Only the Live
 * monitor is ever tall enough to get here, and its native `resize` grip is in
 * the opposite corner, so the one place left for the stack is inside it with
 * that grip kept clear. `stackMaxHeight` holds the other edge.
 */
function dodgeInset(frame: MonitorFrame, viewportHeight: number): number {
  if (liftFits(frame, viewportHeight)) {
    return Math.max(STACK_INSET, liftOver(frame, viewportHeight));
  }
  return Math.max(
    STACK_INSET,
    viewportHeight - frame.bottom + MONITOR_GRIP + STACK_GAP,
  );
}

/** Height available between the box and the stack sitting on `bottomInset`. */
function roomBelow(
  frame: MonitorFrame,
  viewportHeight: number,
  bottomInset: number,
): number {
  return viewportHeight - bottomInset - frame.bottom - STACK_GAP;
}

export function stackBottomInset(
  frame: MonitorFrame | null,
  viewportWidth: number,
  viewportHeight: number,
  neededRoom: number = ASSUMED_STACK_HEIGHT,
): number {
  if (!frame) return STACK_INSET;
  // Only dodge a box that is in the stack's column and crowds its corner; one
  // parked anywhere else, or one leaving room underneath itself, leaves that
  // corner free.
  const inTheWay =
    inStackColumn(frame, viewportWidth) &&
    reachesStack(frame, viewportHeight, STACK_INSET, neededRoom);
  return inTheWay ? dodgeInset(frame, viewportHeight) : STACK_INSET;
}

/**
 * How tall the stack may grow while sitting on `bottomInset`, keeping its own
 * margin at the top. Lifting the stack over the monitor shortens it by the same
 * amount, or a long download list plus the card runs off the top of the screen.
 *
 * A monitor parked high in the same column is not lifted over, because the free
 * space is underneath it. It still has to be dodged: the stack grows upwards
 * from the bottom, and a full download list plus the card is easily tall enough
 * to reach it. Cap the height at the gap below it instead.
 *
 * A monitor too tall to lift over needs the same treatment from the other side.
 * The stack is seated inside it, and the inset alone only holds its bottom
 * edge: an expanded download list plus the loaded models card grows from there
 * back over the monitor's header, which is what the inset was dodging.
 */
export function stackMaxHeight(
  frame: MonitorFrame | null,
  viewportWidth: number,
  viewportHeight: number,
  bottomInset: number,
  neededRoom: number = ASSUMED_STACK_HEIGHT,
): number {
  const ownMargin = viewportHeight - bottomInset - STACK_INSET;
  if (!frame || !inStackColumn(frame, viewportWidth)) return ownMargin;
  if (reachesStack(frame, viewportHeight, bottomInset, neededRoom)) {
    // Lifted over: bottomInset already cleared it.
    if (liftFits(frame, viewportHeight)) return ownMargin;
    // Seated inside it instead. Stop below its header, or the Close button goes
    // back under the stack the inset has just moved off it. Floored, because a
    // box taller than the room the header leaves would ask for a negative cap
    // and browsers drop one of those, taking the limit with it.
    const belowHeader =
      viewportHeight - bottomInset - frame.top - MONITOR_HEADER - STACK_GAP;
    return Math.max(MIN_STACK_ROOM, Math.min(ownMargin, belowHeader));
  }
  // At least MIN_STACK_ROOM, since anything tighter reaches at this inset.
  return Math.min(ownMargin, roomBelow(frame, viewportHeight, bottomInset));
}

export type StackGeometry = { bottom: number; maxHeight: number };

/**
 * Where the overlay stack sits and how tall it may be, given everything it has
 * to keep clear of.
 *
 * Folded per box, never over their union: a tall monitor and the wide docked
 * composer share almost no area, and the rectangle around the pair covers most
 * of the viewport. Reading that as one obstacle pinned the stack to the top of
 * the screen and put it back over the monitor it was dodging. Each box asks for
 * the lift it needs; the stack takes the largest, and the shortest height.
 */
export function stackGeometry(
  frames: MonitorFrame | null | readonly MonitorFrame[],
  viewportWidth: number,
  viewportHeight: number,
  neededRoom: number = ASSUMED_STACK_HEIGHT,
): StackGeometry {
  const list = frames === null ? [] : Array.isArray(frames) ? frames : [frames];
  if (list.length === 0) {
    const bottom = stackBottomInset(
      null,
      viewportWidth,
      viewportHeight,
      neededRoom,
    );
    return {
      bottom,
      maxHeight: stackMaxHeight(
        null,
        viewportWidth,
        viewportHeight,
        bottom,
        neededRoom,
      ),
    };
  }
  // Settled, not summed. Lifting over one box moves the stack up into the next,
  // which may then need a lift of its own, so keep going until nothing more
  // asks. Each pass can only promote a box once, so this bounds at their count.
  const column = list.filter((f) => inStackColumn(f, viewportWidth));
  let bottom = STACK_INSET;
  for (let pass = 0; pass <= column.length; pass += 1) {
    let next = bottom;
    for (const frame of column) {
      if (reachesStack(frame, viewportHeight, bottom, neededRoom)) {
        next = Math.max(next, dodgeInset(frame, viewportHeight));
      }
    }
    if (next === bottom) break;
    bottom = next;
  }
  return {
    bottom,
    maxHeight: Math.min(
      ...list.map((f) =>
        stackMaxHeight(f, viewportWidth, viewportHeight, bottom, neededRoom),
      ),
    ),
  };
}

export type StackPlacement = StackGeometry & {
  /** Attach to the stack container so its height feeds back into the placement. */
  ref: (node: HTMLElement | null) => void;
};

/**
 * `stackGeometry` in px, recomputed as the monitor moves or resizes, and as the
 * stack's own content grows.
 *
 * The measurement has to be taken with this hook's own `maxHeight` off. The
 * overlays are flex items with `min-h-0` and their own inner scrollers, so under
 * the cap they shrink to fit it rather than overflowing, and both `clientHeight`
 * and `scrollHeight` come back as the cap itself. Measured in a browser: a stack
 * whose content is 83px tall reports 83, and reports 40 the moment a 40px cap is
 * applied. Fed back in, that is the placement reading its own output: a stack
 * taller than the gap under an obstacle would measure as exactly the gap, never
 * ask to be lifted, and sit there clipped.
 */
export function useStackGeometry(): StackPlacement {
  const frames = useMonitorFrameStore((state) => state.frames);
  // Every published box, not their union: see stackGeometry.
  const published = useMemo(() => [...frames.values()], [frames]);
  const [viewport, setViewport] = useState(() => ({
    width: typeof window === "undefined" ? 0 : window.innerWidth,
    height: typeof window === "undefined" ? 0 : window.innerHeight,
  }));
  const [neededRoom, setNeededRoom] = useState(ASSUMED_STACK_HEIGHT);
  useEffect(() => {
    const onResize = () =>
      setViewport({ width: window.innerWidth, height: window.innerHeight });
    onResize();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
  const ref = useCallback((node: HTMLElement | null) => {
    if (node === null || typeof ResizeObserver === "undefined") return;
    const measure = () => {
      // An empty stack asks for nothing, so nothing is dodged for it.
      if (node.childElementCount === 0) {
        setNeededRoom((current) => (current === 0 ? current : 0));
        return;
      }
      // Drop the cap for the read and put it straight back, in one synchronous
      // block: reading scrollHeight in between forces the unconstrained layout,
      // and restoring before yielding means no observer, paint or other reader
      // ever sees the uncapped box. Written through style rather than state
      // because React owns this property and will rewrite the same value on the
      // next render anyway.
      const capped = node.style.maxHeight;
      node.style.maxHeight = "none";
      const natural = node.scrollHeight;
      node.style.maxHeight = capped;
      setNeededRoom((current) => (current === natural ? current : natural));
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    // Overlays mount and unmount inside the stack without resizing it while it is
    // empty (a 0-height box stays 0), so watch the child list too.
    const mutations = new MutationObserver(measure);
    mutations.observe(node, { childList: true, subtree: true });
    return () => {
      observer.disconnect();
      mutations.disconnect();
    };
  }, []);
  return {
    ...stackGeometry(published, viewport.width, viewport.height, neededRoom),
    ref,
  };
}
