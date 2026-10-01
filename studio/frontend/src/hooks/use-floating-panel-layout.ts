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
  // Committed left/top the drag transform offsets from.
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

// Content height, not the rendered box: maxHeight would hide growth.
function desiredPanelHeight(
  renderedHeight: number,
  scroll: HTMLDivElement | null,
  content: HTMLDivElement | null,
): number {
  if (!(scroll && content)) {
    return renderedHeight;
  }
  const chrome = renderedHeight - scroll.getBoundingClientRect().height;
  return chrome + content.getBoundingClientRect().height;
}

// Lift the maxWidth cap for one measurement so an anchored panel can widen again.
function naturalWidth(monitor: HTMLDivElement): number {
  const capped = monitor.style.maxWidth;
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
  // Per-instance frame owner: a panel unmounting mid-exit clears only its own frame.
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
  // User's full-width placement; cleared by a drag while narrowed.
  const chosenLeftRef = useRef<number | null>(null);
  const restoreLeftRef = useRef<number | null>(null);
  const remeasureRef = useRef(0);
  const [layout, setLayout] = useState<MonitorLayout | null>(null);
  // For transitions that fire no observer (suppressed to undocked).
  const reconcileRef = useRef<(() => void) | null>(null);
  const initialPlacementRef = useRef(initialPlacement);
  const initializedRef = useRef(false);

  useLayoutEffect(() => {
    const monitor = monitorRef.current;
    const constraints = constraintsElement;
    if (!(monitor && constraints)) {
      return;
    }
    // Next frame: a style write during ResizeObserver delivery trips Firefox's loop check.
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

      if (!monitor.style.height) {
        preferredHeightRef.current = desiredHeight;
      }

      const width = Math.min(desiredWidth, constraintsBox.width);
      // A hand-resized panel keeps its height and scrolls instead of moving up.
      const height = Math.min(
        monitor.style.height ? monitorBox.height : desiredHeight,
        constraintsBox.height,
      );
      const maxLeft = Math.max(0, constraintsBox.width - width);
      const maxTop = Math.max(0, constraintsBox.height - height);
      const currentLeft = monitorBox.left - constraintsBox.left;
      const currentTop = monitorBox.top - constraintsBox.top;
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
      // Obstacle avoidance applies to the first placement only.
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

      if (!hiddenRef.current) {
        useMonitorFrameStore.getState().setFrame(publisher, {
          left: monitorBox.left,
          top: monitorBox.top,
          right: monitorBox.right,
          bottom: monitorBox.bottom,
        });
      }

      setLayout((current) => {
        // Mid-drag the measured box includes the transform; finishDrag commits instead.
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

  // A hidden panel must not publish a frame others dodge.
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

  // Position-only changes fire no ResizeObserver, so republish per committed layout (not per drag frame).
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

    // Sync inline size to the rendered box where max-width/height clipped a native resize.
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

  // One transform per frame: left/top moves re-sample the backdrop blur.
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
    if (left !== baseLeft) {
      hasDraggedLeftRef.current = true;
      chosenLeftRef.current = narrowedRef.current ? null : left;
    }
    // Write the node before state so the handoff never flashes the start position.
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
