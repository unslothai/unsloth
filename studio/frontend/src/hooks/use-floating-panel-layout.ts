// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useMonitorFrameStore } from "@/features/settings";
import {
  type PointerEvent,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";

interface MonitorLayout {
  left: number;
  top: number;
  minWidth: number;
  minHeight: number;
  maxWidth: number;
  maxHeight: number;
}

interface DragSession {
  pointerId: number;
  startX: number;
  startY: number;
  left: number;
  top: number;
  maxLeft: number;
  maxTop: number;
  constraintsWidth: number;
  constraintsHeight: number;
  /** The committed left/top the drag's transform offsets from. */
  baseLeft: number;
  baseTop: number;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

// Anchored to the far edge until the user drags, then clamped where they left it.
function place(dragged: boolean, current: number, max: number): number {
  return dragged ? clamp(current, 0, max) : max;
}

function sameLayout(a: MonitorLayout, b: MonitorLayout): boolean {
  return (
    a.left === b.left &&
    a.top === b.top &&
    a.minWidth === b.minWidth &&
    a.minHeight === b.minHeight &&
    a.maxWidth === b.maxWidth &&
    a.maxHeight === b.maxHeight
  );
}

// Height the panel wants. Reading the rendered box instead hides growth once
// maxHeight caps it, so the observer never fires and the cap is never lifted.
function desiredPanelHeight(
  renderedHeight: number,
  scroll: HTMLDivElement | null,
  content: HTMLDivElement | null,
): number {
  if (!(scroll && content)) {
    return renderedHeight;
  }
  // The scroll region is the only flexible child, so the rest is fixed chrome.
  const chrome = renderedHeight - scroll.getBoundingClientRect().height;
  return chrome + content.getBoundingClientRect().height;
}

// Width the panel wants. While anchored, maxWidth equals the current width, so
// the cap is also a floor: a monitor opened in a narrow window never widens
// again. Lift the cap for one measurement to break that.
function naturalWidth(monitor: HTMLDivElement): number {
  const capped = monitor.style.maxWidth;
  // "none", not "", so the class-level max-w-full lifts too.
  monitor.style.maxWidth = "none";
  const width = monitor.getBoundingClientRect().width;
  monitor.style.maxWidth = capped;
  return width;
}

export function useFloatingPanelLayout(
  constraintsElement: HTMLDivElement | null,
  narrowed: boolean,
  hidden: boolean,
  initialPlacement?: (
    size: { width: number; height: number },
    bounds: DOMRect,
  ) => { left: number; top: number },
) {
  // This panel's claim on the shared frame. Reopening the monitor mid-exit
  // mounts the replacement while the old panel is still animating out, and the
  // old one unmounts last, so its cleanup must only clear its own frame.
  const publisher = useMemo(() => ({}), []);
  const monitorRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const dragSessionRef = useRef<DragSession | null>(null);
  const dragFrameRef = useRef(0);
  const hasDraggedLeftRef = useRef(false);
  const hasDraggedTopRef = useRef(false);
  const preferredWidthRef = useRef<number | null>(null);
  const preferredHeightRef = useRef<number | null>(null);
  const surfaceWidthRef = useRef(0);
  const narrowedRef = useRef(narrowed);
  const hiddenRef = useRef(hidden);
  // The user's own placement while the container is at full width. Cleared when
  // they drag while it is narrowed, which is a newer choice, not a clamp.
  const chosenLeftRef = useRef<number | null>(null);
  const restoreLeftRef = useRef<number | null>(null);
  const remeasureRef = useRef(0);
  const [layout, setLayout] = useState<MonitorLayout | null>(null);
  // Reconcile normally arrives from the observers. A transition that leaves the
  // constraint geometry untouched -- suppressed to undocked -- fires none.
  const reconcileRef = useRef<(() => void) | null>(null);
  const initialPlacementRef = useRef(initialPlacement);
  const initializedRef = useRef(false);

  useLayoutEffect(() => {
    const monitor = monitorRef.current;
    const constraints = constraintsElement;
    if (!(monitor && constraints)) {
      return;
    }
    // Deferred to the next frame: writing a style while ResizeObserver entries
    // are delivered makes Firefox report an observer loop. A hand-resized panel
    // keeps the user's width instead of re-measuring.
    const scheduleWidthRemeasure = (surfaceWidth: number) => {
      if (surfaceWidth === surfaceWidthRef.current) {
        return;
      }
      surfaceWidthRef.current = surfaceWidth;
      if (monitor.style.width || remeasureRef.current) {
        return;
      }
      remeasureRef.current = requestAnimationFrame(() => {
        remeasureRef.current = 0;
        preferredWidthRef.current = naturalWidth(monitor);
        reconcileGeometry();
      });
    };

    const reconcileGeometry = () => {
      const constraintsBox = constraints.getBoundingClientRect();
      const monitorBox = monitor.getBoundingClientRect();
      const desiredHeight = desiredPanelHeight(
        monitorBox.height,
        scrollRef.current,
        contentRef.current,
      );

      scheduleWidthRemeasure(constraintsBox.width);
      const desiredWidth = Math.max(
        monitorBox.width,
        preferredWidthRef.current ?? monitorBox.width,
      );

      // Content height is the floor, as a resolved number: intrinsic
      // min-content outranks max-height, a number clamped to it cannot.
      if (!monitor.style.height) {
        preferredHeightRef.current = desiredHeight;
      }

      const width = Math.min(desiredWidth, constraintsBox.width);
      // Clamp position against the height actually rendered. A hand-resized panel keeps its own
      // height and scrolls, so growing content must not drag it upwards and leave a gap below.
      const height = Math.min(
        monitor.style.height ? monitorBox.height : desiredHeight,
        constraintsBox.height,
      );
      const maxLeft = Math.max(0, constraintsBox.width - width);
      const maxTop = Math.max(0, constraintsBox.height - height);
      const currentLeft = monitorBox.left - constraintsBox.left;
      const currentTop = monitorBox.top - constraintsBox.top;
      // A restored position is a deliberate left the constraint had clamped
      // away, so it replaces `place()` for exactly one pass.
      const restoreTo = restoreLeftRef.current;
      let left =
        restoreTo === null
          ? place(hasDraggedLeftRef.current, currentLeft, maxLeft)
          : clamp(restoreTo, 0, maxLeft);
      restoreLeftRef.current = null;
      if (!narrowedRef.current && hasDraggedLeftRef.current) {
        chosenLeftRef.current = left;
      }
      let top = place(hasDraggedTopRef.current, currentTop, maxTop);
      // Optional obstacle avoidance runs only on opening. Native resize and
      // pointer dragging then use the same geometry as the resource monitor.
      if (!initializedRef.current && initialPlacementRef.current) {
        const initial = initialPlacementRef.current(
          { width, height },
          constraintsBox,
        );
        left = clamp(initial.left, 0, maxLeft);
        top = clamp(initial.top, 0, maxTop);
        hasDraggedLeftRef.current = true;
        hasDraggedTopRef.current = true;
      }
      initializedRef.current = true;

      const session = dragSessionRef.current;
      if (session) {
        session.left = left;
        session.top = top;
        session.maxLeft = maxLeft;
        session.maxTop = maxTop;
        session.constraintsWidth = constraintsBox.width;
        session.constraintsHeight = constraintsBox.height;
      }

      // Publish the real box so the overlay stack can keep clear of it.
      if (!hiddenRef.current) {
        useMonitorFrameStore.getState().setFrame(publisher, {
          left: monitorBox.left,
          top: monitorBox.top,
          right: monitorBox.right,
          bottom: monitorBox.bottom,
        });
      }

      setLayout((current) => {
        // Mid-drag the offset lives in a transform, and the measured box already includes it, so
        // committing left/top here would apply it twice. finishDrag lands the position instead.
        const held = session && current ? current : null;
        const restLeft = held?.left ?? left;
        const restTop = held?.top ?? top;
        const next = {
          left: restLeft,
          top: restTop,
          minWidth: preferredWidthRef.current ?? monitorBox.width,
          minHeight: preferredHeightRef.current ?? monitorBox.height,
          maxWidth: constraintsBox.width - restLeft,
          maxHeight: constraintsBox.height - restTop,
        };
        return current && sameLayout(current, next) ? current : next;
      });
    };

    reconcileRef.current = reconcileGeometry;
    reconcileGeometry();
    const observer = new ResizeObserver(reconcileGeometry);
    observer.observe(constraints);
    observer.observe(monitor);
    // The unclamped content wrapper is what makes late GPU rows reposition the
    // panel instead of being cut off.
    if (contentRef.current) {
      observer.observe(contentRef.current);
    }
    return () => {
      observer.disconnect();
      reconcileRef.current = null;
      useMonitorFrameStore.getState().clearFrame(publisher);
      if (remeasureRef.current) {
        cancelAnimationFrame(remeasureRef.current);
        remeasureRef.current = 0;
      }
      if (dragFrameRef.current) {
        cancelAnimationFrame(dragFrameRef.current);
        dragFrameRef.current = 0;
      }
    };
  }, [constraintsElement, publisher]);

  // Narrowing clamps the monitor left, and `place()` keeps the clamped spot.
  // The position the user did drag to is put back when the container widens.
  // Settled in a layout effect, before the next observation can reconcile.
  // The API monitor treats any published frame as a live obstacle, so an
  // invisible resource monitor must not keep publishing its box. Visibility and
  // aria-hidden fire no ResizeObserver, so `hidden` also feeds the republish
  // below: it is what restores the box once the monitor is on screen again.
  useLayoutEffect(() => {
    hiddenRef.current = hidden;
    if (hidden) {
      useMonitorFrameStore.getState().clearFrame(publisher);
    }
  }, [hidden, publisher]);

  useLayoutEffect(() => {
    if (narrowedRef.current === narrowed) {
      return;
    }
    narrowedRef.current = narrowed;
    if (!narrowed) {
      restoreLeftRef.current = chosenLeftRef.current;
      reconcileRef.current?.();
    }
  }, [narrowed]);

  // ResizeObserver never fires for a position-only change, so dragging alone would leave the
  // published frame at the monitor's old corner and the overlay stack dodging where it used to be.
  // Re-publish once each layout is committed, which after a drag is on release: the frames in
  // between are a transform, and republishing through them would re-render every overlay in the
  // stack for each one, which is most of what made dragging feel heavy.
  useLayoutEffect(() => {
    void layout;
    const monitor = monitorRef.current;
    if (!(monitor && constraintsElement) || hiddenRef.current) {
      return;
    }
    const box = monitor.getBoundingClientRect();
    useMonitorFrameStore.getState().setFrame(publisher, {
      left: box.left,
      top: box.top,
      right: box.right,
      bottom: box.bottom,
    });
  }, [layout, constraintsElement, publisher, hidden]);

  function startDrag(event: PointerEvent<HTMLDivElement>) {
    const monitor = monitorRef.current;
    if (event.button !== 0 || !(monitor && constraintsElement)) {
      return;
    }

    event.preventDefault();
    const constraintsBox = constraintsElement.getBoundingClientRect();
    const monitorBox = monitor.getBoundingClientRect();
    const left = monitorBox.left - constraintsBox.left;
    const top = monitorBox.top - constraintsBox.top;

    // Native resize records attempted inline dimensions even when max-width
    // or max-height hides them. Normalize only hidden dimensions so an
    // auto-sized monitor can still grow when system rows arrive later.
    const inlineWidth = Number.parseFloat(monitor.style.width);
    const inlineHeight = Number.parseFloat(monitor.style.height);
    if (
      Number.isFinite(inlineWidth) &&
      Math.abs(inlineWidth - monitorBox.width) > 0.5
    ) {
      monitor.style.width = `${monitorBox.width}px`;
    }
    if (
      Number.isFinite(inlineHeight) &&
      Math.abs(inlineHeight - monitorBox.height) > 0.5
    ) {
      monitor.style.height = `${monitorBox.height}px`;
    }

    dragSessionRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      left,
      top,
      maxLeft: Math.max(0, constraintsBox.width - monitorBox.width),
      maxTop: Math.max(0, constraintsBox.height - monitorBox.height),
      constraintsWidth: constraintsBox.width,
      constraintsHeight: constraintsBox.height,
      baseLeft: left,
      baseTop: top,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  // One paint per frame, and through a transform rather than left/top. The panel is
  // backdrop-blurred, so every layout-driven move re-sampled what is behind it; a trackpad also
  // reports moves faster than the display refreshes, so most of those renders were never shown.
  function paintDrag() {
    dragFrameRef.current = 0;
    const session = dragSessionRef.current;
    const monitor = monitorRef.current;
    if (!(session && monitor)) {
      return;
    }
    monitor.style.transform = `translate3d(${session.left - session.baseLeft}px, ${
      session.top - session.baseTop
    }px, 0)`;
  }

  function updateDrag(event: PointerEvent<HTMLDivElement>) {
    const session = dragSessionRef.current;
    if (!session || session.pointerId !== event.pointerId) {
      return;
    }

    // Horizontal placement is chosen on release. An intermediate move that
    // returns to its starting X must leave an anchored monitor anchored.
    const previousTop = session.top;
    const left = clamp(
      session.left + event.clientX - session.startX,
      0,
      session.maxLeft,
    );
    const top = clamp(
      session.top + event.clientY - session.startY,
      0,
      session.maxTop,
    );
    if (top !== previousTop) {
      hasDraggedTopRef.current = true;
    }
    session.startX = event.clientX;
    session.startY = event.clientY;
    session.left = left;
    session.top = top;
    if (!dragFrameRef.current) {
      dragFrameRef.current = requestAnimationFrame(paintDrag);
    }
  }

  function finishDrag(event: PointerEvent<HTMLDivElement>) {
    const session = dragSessionRef.current;
    if (session?.pointerId !== event.pointerId) {
      return;
    }
    if (dragFrameRef.current) {
      cancelAnimationFrame(dragFrameRef.current);
      dragFrameRef.current = 0;
    }
    const { left, top, baseLeft, constraintsWidth, constraintsHeight } =
      session;
    dragSessionRef.current = null;
    // Only the released horizontal position is a new choice. Returning to the
    // starting X keeps the saved full-width position even after intermediate moves.
    if (left !== baseLeft) {
      hasDraggedLeftRef.current = true;
      chosenLeftRef.current = narrowedRef.current ? null : left;
    }
    // Written to the node as well as to state, in this order, so handing the
    // offset back to left/top cannot show a frame at the spot it started from.
    const monitor = monitorRef.current;
    if (monitor) {
      monitor.style.left = `${left}px`;
      monitor.style.top = `${top}px`;
      monitor.style.transform = "";
    }
    setLayout((current) =>
      !current || (current.left === left && current.top === top)
        ? current
        : {
            ...current,
            left,
            top,
            maxWidth: constraintsWidth - left,
            maxHeight: constraintsHeight - top,
          },
    );
  }

  return {
    publisher,
    monitorRef,
    scrollRef,
    contentRef,
    layout,
    startDrag,
    updateDrag,
    finishDrag,
  };
}
