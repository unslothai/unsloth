// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What the bottom-right overlay stack has to keep clear of, in viewport
// coordinates. The Live monitor is draggable and resizable and defaults to that
// same corner; the chat composer docks to the bottom of the same column once a
// thread has turns; the API monitor panel opens in that corner on its own. Each
// publishes its box here while it is mounted.
//
// Read as well as written: the API monitor panel places itself against every
// box but its own (api-monitor/panel-placement), so the three surfaces resolve
// in one direction -- the monitor is placed by the user, the API panel steps
// around the monitor, and the stack steps around both.

import { useCallback, useEffect, useMemo, useState } from "react";
import { create } from "zustand";

export type MonitorFrame = {
  left: number;
  top: number;
  right: number;
  bottom: number;
  /**
   * Whether the stack may paint over this box when there is nowhere left to
   * put itself. Off by default, because the box that started this store is the
   * Live monitor and its Close button and resize grip have to stay clickable.
   *
   * The chat composer opts in. In a short window there is no arrangement that
   * both dodges it and shows the cards whole, and the two ways to lose are not
   * equal: a clipped card reads as broken, a card over the composer reads as a
   * card over the composer and has a dismiss button on it.
   */
  coverable?: boolean;
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
    a.bottom === b.bottom &&
    Boolean(a.coverable) === Boolean(b.coverable)
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
// Room the stack asks for before it has been measured. Deliberately small: guessing
// high is what parked the loaded models card mid-window with the corner underneath it
// empty. The real height arrives from the stack's ResizeObserver on the next frame.
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
 * an empty chat it sits well clear of the bottom, and lifting over it there
 * strands the stack mid-page with the corner underneath it empty.
 *
 * So the room is measured, never guessed: `neededRoom` is the stack's own
 * ResizeObserver height (see useStackGeometry). A fixed 120 once lifted an 80px
 * card out of an 83px gap it fit in.
 *
 * Asked against the inset in force, never a fixed one: lifting over one box
 * moves the stack up into the next, and a box that had room at the corner may
 * have none there.
 *
 * A box that may be covered is asked against the floor instead. It has already
 * said the stack outranks it, so a band holding every card whole is enough, and
 * the height the cards would prefer is not worth leaving the corner for: a rail
 * taller than the welcome composer's band was being parked mid-window for room
 * it scrolls anyway.
 */
function reachesStack(
  frame: MonitorFrame,
  viewportHeight: number,
  bottomInset: number,
  neededRoom: number,
  floorRoom: number = neededRoom,
): boolean {
  const wanted = frame.coverable ? floorRoom : neededRoom;
  // At least a pixel: a needed room of 0 lets the capped branch hand back a
  // max-height of 0, which browsers honour, leaving the stack invisible.
  return roomBelow(frame, viewportHeight, bottomInset) < Math.max(1, wanted);
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
  floorRoom: number = neededRoom,
): number {
  if (!frame) return STACK_INSET;
  // Only dodge a box that is in the stack's column and crowds its corner; one
  // parked anywhere else, or one leaving room underneath itself, leaves that
  // corner free.
  const inTheWay =
    inStackColumn(frame, viewportWidth) &&
    reachesStack(frame, viewportHeight, STACK_INSET, neededRoom, floorRoom);
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
  floorRoom: number = neededRoom,
): number {
  const ownMargin = viewportHeight - bottomInset - STACK_INSET;
  if (!frame || !inStackColumn(frame, viewportWidth)) return ownMargin;
  if (reachesStack(frame, viewportHeight, bottomInset, neededRoom, floorRoom)) {
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
  // What the stack cannot give up, as opposed to what it would like. Defaults
  // to `neededRoom` so a caller that knows only one number keeps the stricter
  // reading of it.
  floorRoom: number = neededRoom,
  // Height of the run of cards at the bottom of the stack that the reader
  // cannot dismiss. Covering is refused when that run would land on a box.
  persistentTail = 0,
): StackGeometry {
  const list = frames === null ? [] : Array.isArray(frames) ? frames : [frames];
  const placed = place(
    list,
    viewportWidth,
    viewportHeight,
    neededRoom,
    floorRoom,
  );
  if (placed.maxHeight >= floorRoom || !list.some((f) => f.coverable)) {
    return placed;
  }
  // Nowhere to put the stack even at its floor while dodging everything. Drop
  // the boxes that said they may be covered and try again: the stack takes the
  // corner and paints over the composer, which is what the cards being on top
  // means. Clipping them instead is what the report was about, and a card
  // sliced off at the rail's edge looks like it has slid behind the page.
  //
  // Tested against the floor and not the natural height on purpose. The cards
  // are allowed to give up their notes, so a placement 3px short of the height
  // they would prefer is still a placement that shows all of them, and covering
  // the composer to win those 3px is a worse answer than a slightly shorter
  // notes preview.
  const uncoverable = list.filter((f) => !f.coverable);
  const covering = place(
    uncoverable,
    viewportWidth,
    viewportHeight,
    neededRoom,
    floorRoom,
  );
  // Where the run the reader cannot dismiss would actually land, measured from
  // the placement being considered rather than from the corner. Dropping the
  // composer does not always leave the stack at the bottom: an uncoverable
  // monitor can still lift it, which carries the tail up onto the very box the
  // corner-based reading had just cleared.
  const tailTop = viewportHeight - covering.bottom - persistentTail;
  const safeToCover = list.every(
    (frame) => !frame.coverable || frame.bottom <= tailTop,
  );
  // Covering has to buy the thing it is for. A placement that takes the
  // composer and STILL cannot show the cards at their floor has paid the whole
  // price for nothing: the rail scrolls either way, so the one that leaves Send
  // reachable is the better of two bad answers.
  if (!safeToCover || covering.maxHeight < floorRoom) {
    return placed;
  }
  return covering.maxHeight > placed.maxHeight ? covering : placed;
}

/** `stackGeometry` for one set of boxes, all of which must be dodged. */
function place(
  list: readonly MonitorFrame[],
  viewportWidth: number,
  viewportHeight: number,
  neededRoom: number,
  floorRoom: number = neededRoom,
): StackGeometry {
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
      if (reachesStack(frame, viewportHeight, bottom, neededRoom, floorRoom)) {
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
        stackMaxHeight(
          f,
          viewportWidth,
          viewportHeight,
          bottom,
          neededRoom,
          floorRoom,
        ),
      ),
    ),
  };
}

export type StackPlacement = StackGeometry & {
  /** Attach to the stack container so its height feeds back into the placement. */
  ref: (node: HTMLElement | null) => void;
  /**
   * The cards need more room than the cap allows, so the stack is scrolling.
   * It is click-through the rest of the time, which also costs it its
   * scrollbar, and a scroller nobody can drag hides the cards below the fold.
   */
  overflowing: boolean;
};

/**
 * `stackGeometry` in px, recomputed as the monitor moves or resizes, and as the
 * stack's own content grows.
 *
 * The measurement has to be taken with this hook's own `maxHeight` off. The
 * overlays are `min-h-0` flex items with inner scrollers, so under the cap they
 * shrink to it and `scrollHeight` reports the cap, not the content: 83px of
 * content reads back as 40 under a 40px cap. Fed back in that is the placement
 * reading its own output, so a stack taller than the gap under an obstacle would
 * measure as exactly the gap, never ask to be lifted, and sit there clipped.
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
  const [floorRoom, setFloorRoom] = useState(ASSUMED_STACK_HEIGHT);
  const [persistentTail, setPersistentTail] = useState(0);
  // Whether the rail is ACTUALLY scrolling, read off the node at its real cap.
  // The derived answer below is a prediction, and a prediction can be wrong: the
  // cards are only assumed to collapse to `floorRoom`, so whenever they stop
  // short of it the box overflows a cap that `floorRoom > maxHeight` says it
  // fits inside. That combination is a rail that scrolls while it is still
  // click-through, which costs it its scrollbar and strands every card above the
  // fold where no pointer can reach them.
  const [domOverflowing, setDomOverflowing] = useState(false);
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
        setFloorRoom((current) => (current === 0 ? current : 0));
        return;
      }
      // Drop the cap and put it back in one synchronous block, so scrollHeight
      // sees the unconstrained layout but nothing else ever sees the uncapped
      // box. Through style, not state: React rewrites the same value next render.
      // An uncapped box does not overflow, so lifting the cap clamps scrollTop
      // to 0 and putting it back does not undo that: every descendant resize
      // would throw a reader back to the first card. Restore it with the cap.
      // `transition-property: all` reaches this box, so each write below starts
      // a transition on max-height, and a transition's computed value is its
      // *start* value until the timeline advances. Reading scrollHeight flushes
      // style within the same frame, so the probe would read the cap it just
      // replaced, and the restore would leave the box computing 0px while its
      // inline style says otherwise: a rail with three whole cards laid out
      // below a zero-height box, which is what the loaded models indicator
      // turned up. Suppressed for the probe and restored with it.
      const eased = node.style.transition;
      node.style.transition = "none";
      const capped = node.style.maxHeight;
      // The rail's own scroll position is not the only one at stake. Uncapping
      // it grows every scroller inside it, which shortens their scrollable
      // range and clamps any that were scrolled past the new end: read the
      // release notes to the bottom, let a download tick, and the list the
      // reader was in jumps. The rail's cap comes back, but a clamped
      // descendant does not come back with it, so each one is noted here and
      // put back below.
      const scrollers: Array<[Element, number]> = [];
      for (const child of node.querySelectorAll("*")) {
        if (child.scrollTop > 0) {
          scrollers.push([child, child.scrollTop]);
        }
      }
      const scrolled = node.scrollTop;
      node.style.maxHeight = "none";
      const natural = node.scrollHeight;
      // Taken here, uncapped, and not after the cap goes back on. The tail
      // panels are min-h-0 with their own scrollers, so under a tight cap they
      // measure as almost nothing, the placement reads that as a tail it can
      // safely put in the corner, the corner's larger cap lets them grow, and
      // the next measurement says the opposite: the two placements would swap
      // back and forth for as long as a download and an update card share the
      // rail. One size that does not depend on the answer, as with the two
      // heights above.
      let tail = 0;
      for (let i = node.children.length - 1; i >= 0; i -= 1) {
        const child = node.children[i];
        if (child.hasAttribute("data-overlay-dismissible")) break;
        tail += child.getBoundingClientRect().height + STACK_GAP;
      }
      const persistent = Math.round(tail);
      // And the other end of the same measurement: squeezed to nothing, what is
      // left is what the cards refuse to give up. The difference between the two
      // is the height the stack can donate to a dodge, and asking a placement to
      // hold `natural` when it only has to hold `floor` is what made a 3px
      // shortfall at 1280x830 give up on dodging the composer entirely.
      node.style.maxHeight = "0px";
      const floor = node.scrollHeight;
      node.style.maxHeight = capped;
      if (node.scrollTop !== scrolled) {
        node.scrollTop = scrolled;
      }
      for (const [child, top] of scrollers) {
        if (child.scrollTop !== top) {
          child.scrollTop = top;
        }
      }
      // Flush the restore under the suppression, or putting `transition` back
      // hands the pending max-height change to a transition after all.
      void node.scrollHeight;
      node.style.transition = eased;
      // Taken here, with the real cap back on and the layout already flushed by
      // the line above, so it describes the box the reader has rather than
      // either probe. After the `transition` restore, not between it and the
      // flush: the cap change is already committed under the suppression, and
      // `transition` is not itself a transitionable property, so no reflow this
      // read forces can hand that cap to an animation.
      //
      // Read every time this runs, never latched: the observers below watch the
      // rail AND every descendant, and a placement change moves the rail's own
      // border box, so a cap that grows to fit clears this on the same pass that
      // applied it. That is what the derived value was protecting against, and
      // it is still true when the reading is refreshed rather than remembered.
      const overflows = node.scrollHeight > node.clientHeight;
      setDomOverflowing((current) =>
        current === overflows ? current : overflows,
      );
      setNeededRoom((current) => (current === natural ? current : natural));
      setFloorRoom((current) => (current === floor ? current : floor));
      // How much of the stack, measured up from the corner, the reader cannot
      // dismiss. The loaded models indicator and the download panel are last,
      // so a covering placement puts them nearest the bottom edge: over Send
      // that is #8210 again and permanently, rather than until a dismiss. The
      // indicator ships off by default (#8346), which makes this whoever turned
      // it on rather than nobody.
      //
      // A height rather than a flag, because whether it is a problem depends on
      // where the composer is. Docked under a thread it sits on the bottom edge
      // and the tail lands on it; on an empty chat it is centred and the corner
      // below it is free, so the cards that reach it are the dismissible ones
      // and there is nothing to protect.
      setPersistentTail((current) =>
        current === persistent ? current : persistent,
      );
    };
    measure();
    // Every box inside the stack, not just the stack itself. At its cap the container's
    // border box does not move when its content grows, so a root-only observer never
    // fires for a release-note image finishing its load, and childList says nothing
    // either: the stack would keep the pre-load height and stay clipped.
    // Height only. The llama.cpp update banner animates its progress bar's width on every
    // frame, and that bar is inside this stack, so an unfiltered observer would remeasure
    // at ~60Hz for a stack whose height never moved, each one forcing a synchronous layout
    // to read scrollHeight with the cap lifted.
    const heights = new WeakMap<Element, number>();
    const observer = new ResizeObserver((entries) => {
      let moved = false;
      for (const entry of entries) {
        const height =
          entry.borderBoxSize?.[0]?.blockSize ?? entry.contentRect.height;
        if (heights.get(entry.target) !== height) {
          heights.set(entry.target, height);
          moved = true;
        }
      }
      if (moved) measure();
    });
    const observed = new Set<Element>();
    const syncObserved = () => {
      const wanted = new Set<Element>([node, ...node.querySelectorAll("*")]);
      for (const element of observed) {
        if (!wanted.has(element)) {
          observer.unobserve(element);
          observed.delete(element);
        }
      }
      for (const element of wanted) {
        if (!observed.has(element)) {
          observer.observe(element);
          observed.add(element);
        }
      }
    };
    syncObserved();
    // A 0-height stack stays 0 as children come and go, so watch the child list too, and
    // observe whatever just arrived.
    const mutations = new MutationObserver(() => {
      syncObserved();
      measure();
    });
    mutations.observe(node, { childList: true, subtree: true });
    return () => {
      observer.disconnect();
      observed.clear();
      mutations.disconnect();
    };
  }, []);
  const geometry = stackGeometry(
    published,
    viewport.width,
    viewport.height,
    neededRoom,
    floorRoom,
    persistentTail,
  );
  return {
    ...geometry,
    ref,
    // Derived from the placement, not read back off the node. A DOM reading
    // latches: the stack is capped and scrolling for a frame, the placement
    // then changes to one that fits, and nothing resizes afterwards to correct
    // the flag, so a rail with nothing to scroll to keeps the pointer input it
    // took. The cards absorb everything between their floor and their natural
    // height, so a cap below the floor is exactly when the rail has to scroll.
    // A pixel of slack, since the floor is a rounded scrollHeight and the cap
    // is not.
    //
    // Or the box is simply scrolling, whatever the prediction says. The two
    // agree wherever the cards do collapse to the floor they measured, so this
    // only ever adds the case the prediction misses; it cannot take pointer
    // input away from a rail that the derived reading already claimed.
    overflowing: floorRoom > geometry.maxHeight + 1 || domOverflowing,
  };
}
