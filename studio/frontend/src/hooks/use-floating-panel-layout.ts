// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useMonitorFrameStore } from "@/features/settings";
import {
  type FrameBox,
  type PanelIntent,
  clearPanelIntent,
  isDegenerateBox,
  liftAboveObstacle,
  nudgeDelta,
  placeAxis,
  readPanelIntent,
  snapsToAnchor,
  storePanelIntent,
} from "@/hooks/floating-panel-position";
import {
  type KeyboardEvent,
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
  baseLeft: number;
  baseTop: number;
}

const OBSTACLE_GAP_PX = 8;

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
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
  storageKey?: string,
  restingObstacle?: FrameBox | null,
) {
  // Per instance: a panel unmounting mid-exit clears only its own frame.
  const publisher = useMemo(() => ({}), []);
  const monitorRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const dragSessionRef = useRef<DragSession | null>(null);
  const dragFrameRef = useRef(0);
  // User's spot, never rewritten by a clamp.
  const [storedIntent] = useState(() => readPanelIntent(storageKey));
  const intentRef = useRef<PanelIntent>(storedIntent ?? { left: null, top: null });
  const preferredWidthRef = useRef<number | null>(null);
  const preferredHeightRef = useRef<number | null>(null);
  const surfaceWidthRef = useRef(0);
  const narrowedRef = useRef(narrowed);
  const hiddenRef = useRef(hidden);
  const remeasureRef = useRef(0);
  const [layout, setLayout] = useState<MonitorLayout | null>(null);
  const reconcileRef = useRef<(() => void) | null>(null);
  const initialPlacementRef = useRef(initialPlacement);
  const initializedRef = useRef(false);
  const obstacleRef = useRef(restingObstacle ?? null);
  const restRef = useRef({ left: 0, top: 0, maxLeft: 0, maxTop: 0 });

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
      if (isDegenerateBox(constraintsBox)) {
        return;
      }
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
      const height = Math.min(
        monitor.style.height ? monitorBox.height : desiredHeight,
        constraintsBox.height,
      );
      const maxLeft = Math.max(0, constraintsBox.width - width);
      const maxTop = Math.max(0, constraintsBox.height - height);
      const session = dragSessionRef.current;
      // Mid-drag the measured box (transform included) is where the pointer has it.
      let left = session
        ? clamp(monitorBox.left - constraintsBox.left, 0, maxLeft)
        : placeAxis(intentRef.current.left, maxLeft);
      let top = session
        ? clamp(monitorBox.top - constraintsBox.top, 0, maxTop)
        : placeAxis(intentRef.current.top, maxTop);
      // Obstacle avoidance applies to the first placement only, and a remembered spot wins.
      if (
        !initializedRef.current &&
        initialPlacementRef.current &&
        storedIntent === null
      ) {
        const initial = initialPlacementRef.current(
          { width, height },
          constraintsBox,
        );
        left = clamp(initial.left, 0, maxLeft);
        top = clamp(initial.top, 0, maxTop);
        intentRef.current = { left, top };
      }
      initializedRef.current = true;

      const bottomTop = (atLeft: number) => {
        const lifted = liftAboveObstacle(
          {
            left: constraintsBox.left + atLeft,
            top: constraintsBox.top + maxTop,
            width,
            height,
          },
          obstacleRef.current,
          OBSTACLE_GAP_PX,
        );
        return lifted === null
          ? maxTop
          : clamp(lifted - constraintsBox.top, 0, maxTop);
      };
      restRef.current = {
        left: maxLeft,
        top: bottomTop(maxLeft),
        maxLeft,
        maxTop,
      };
      if (!session && intentRef.current.top === null) {
        top = bottomTop(left);
      }

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
  }, [constraintsElement, publisher, storedIntent]);

  // A hidden panel must not publish a frame others dodge.
  useLayoutEffect(() => {
    hiddenRef.current = hidden;
    if (hidden) {
      useMonitorFrameStore.getState().clearFrame(publisher);
    }
  }, [hidden, publisher]);

  useLayoutEffect(() => {
    obstacleRef.current = restingObstacle ?? null;
    if (!dragSessionRef.current) {
      reconcileRef.current?.();
    }
  }, [restingObstacle]);

  useLayoutEffect(() => {
    if (narrowedRef.current === narrowed) {
      return;
    }
    narrowedRef.current = narrowed;
    if (!narrowed) {
      reconcileRef.current?.();
    }
  }, [narrowed]);

  // Position-only changes fire no ResizeObserver, so republish per committed layout.
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

    // Sync inline size where max-width/height clipped a native resize.
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
    const {
      left,
      top,
      baseLeft,
      baseTop,
      constraintsWidth,
      constraintsHeight,
    } = session;
    dragSessionRef.current = null;
    const rest = restRef.current;
    if (
      (left !== baseLeft || top !== baseTop) &&
      snapsToAnchor(
        { left, top },
        rest,
        { left: rest.maxLeft, top: rest.maxTop },
      )
    ) {
      const monitor = monitorRef.current;
      if (monitor) monitor.style.transform = "";
      resetPosition();
      return;
    }
    if (left !== baseLeft || top !== baseTop) {
      const intent = intentRef.current;
      intentRef.current = {
        left: left !== baseLeft ? left : intent.left,
        top: top !== baseTop ? top : intent.top,
      };
      storePanelIntent(storageKey, intentRef.current);
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

  function commitPosition(left: number, top: number, constraints: DOMRect) {
    const monitor = monitorRef.current;
    if (monitor) {
      monitor.style.left = `${left}px`;
      monitor.style.top = `${top}px`;
    }
    setLayout((current) =>
      !current || (current.left === left && current.top === top)
        ? current
        : {
            ...current,
            left,
            top,
            maxWidth: constraints.width - left,
            maxHeight: constraints.height - top,
          },
    );
  }

  function resetPosition() {
    intentRef.current = { left: null, top: null };
    clearPanelIntent(storageKey);
    reconcileRef.current?.();
  }

  function nudge(event: KeyboardEvent<HTMLElement>): boolean {
    const delta = nudgeDelta(event.key, event.shiftKey);
    const monitor = monitorRef.current;
    if (!(delta && monitor && constraintsElement) || dragSessionRef.current) {
      return false;
    }
    event.preventDefault();
    const constraintsBox = constraintsElement.getBoundingClientRect();
    const monitorBox = monitor.getBoundingClientRect();
    const left = clamp(
      monitorBox.left - constraintsBox.left + delta.dx,
      0,
      Math.max(0, constraintsBox.width - monitorBox.width),
    );
    const top = clamp(
      monitorBox.top - constraintsBox.top + delta.dy,
      0,
      Math.max(0, constraintsBox.height - monitorBox.height),
    );
    // Only the moved axis becomes absolute: a null axis stays anchored to the far edge, as after a drag.
    const previous = intentRef.current;
    intentRef.current = {
      left: delta.dx !== 0 ? left : previous.left,
      top: delta.dy !== 0 ? top : previous.top,
    };
    storePanelIntent(storageKey, intentRef.current);
    commitPosition(left, top, constraintsBox);
    return true;
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
    resetPosition,
    nudge,
  };
}
