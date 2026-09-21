// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The React side of sidebar drag-and-drop: the lifted row, the plan under the pointer, and the
// native drag events that feed both. What a drop does is decided in lib/sidebar-drag.ts.

import { useCallback, useEffect, useRef, useState } from "react";

import {
  dropEdgeAt,
  planKey,
  planSidebarDrop,
  rowKey,
  STAY,
  type DropEdge,
  type SidebarDragItem,
  type SidebarDropContext,
  type SidebarDropPlan,
  type SidebarDropZone,
} from "../lib/sidebar-drag.ts";
import {
  setSidebarDragSource,
  sidebarDragSource,
} from "../stores/sidebar-drag-source.ts";

/** How long the pointer rests on a closed folder or section before it opens. */
export const SPRING_OPEN_DELAY_MS = 450;

// Zones nest (row in folder block in section). The first zone to answer a dragover marks the
// event here; outer zones stand down, and the document listener reads the same mark to know
// whether anything answered. Marking instead of stopping propagation keeps the document
// listener, which moves the hint, in the loop.
let lastHandledEvent: Event | null = null;

export interface UseSidebarDragOptions {
  /** Read at event time, so a plan sees the lists as they stand. */
  context: () => SidebarDropContext;
  /** Commits the plan the row was dropped on. */
  onDrop: (plan: SidebarDropPlan, drag: SidebarDragItem) => void;
  /** Opens the closed folder or section the pointer rested on. */
  onSpringOpen?: (zone: SidebarDropZone) => void;
  /** Whether resting on a closed folder or section opens it. */
  springOpen: boolean;
  /** The hint element. Positioned directly, so pointer moves do not re-render the sidebar. */
  hintRef?: React.RefObject<HTMLDivElement | null>;
}

export interface SidebarDragApi {
  /** The lifted row, for painting it faded. Null between drags. */
  drag: SidebarDragItem | null;
  /** What the drop under the pointer would do. */
  plan: SidebarDropPlan | null;
  /** Props that let a row be picked up. */
  dragHandleProps: (item: SidebarDragItem) => {
    draggable: true;
    onDragStart: (event: React.DragEvent) => void;
    onDragEnd: () => void;
  };
  /** Props that let a spot take a drop. `closed` marks a collapsed folder or section. */
  dropZoneProps: (
    zone: SidebarDropZone,
    options?: { closed?: boolean },
  ) => {
    onDragOver: (event: React.DragEvent) => void;
    onDrop: (event: React.DragEvent) => void;
  };
  /** The edge the insertion line is drawn on for this row in this list, if any. */
  lineEdge: (scope: string, id: string) => DropEdge | undefined;
  /** Whether the whole target under this key is lit. */
  ringLit: (key: string) => boolean;
}

export function useSidebarDrag(options: UseSidebarDragOptions): SidebarDragApi {
  const [drag, setDrag] = useState<SidebarDragItem | null>(null);
  const [plan, setPlan] = useState<SidebarDropPlan | null>(null);
  const planRef = useRef<{ key: string; plan: SidebarDropPlan | null }>({
    key: "",
    plan: null,
  });
  // Handlers are made once and read the latest options at event time.
  const optionsRef = useRef(options);
  useEffect(() => {
    optionsRef.current = options;
  });
  const spring = useRef<{ key: string; timer: number } | null>(null);

  const cancelSpring = useCallback(() => {
    if (spring.current) {
      window.clearTimeout(spring.current.timer);
      spring.current = null;
    }
  }, []);

  const showPlan = useCallback((next: SidebarDropPlan | null) => {
    const key = planKey(next);
    if (planRef.current.key === key) return;
    planRef.current = { key, plan: next };
    setPlan(next);
  }, []);

  const clear = useCallback(() => {
    setSidebarDragSource(null);
    cancelSpring();
    setDrag(null);
    showPlan(null);
  }, [cancelSpring, showPlan]);

  // One listener per drag: moves the hint, and clears the cue when nothing answered the
  // dragover. `dragleave` fires for every child crossed, so it cannot be trusted for that.
  useEffect(() => {
    if (!drag) return;
    const onDragOver = (event: DragEvent) => {
      const hint = optionsRef.current.hintRef?.current;
      if (hint) {
        hint.style.transform = `translate(${event.clientX + 14}px, ${event.clientY + 18}px)`;
      }
      if (lastHandledEvent !== event) {
        cancelSpring();
        showPlan(null);
      }
    };
    // A drop that unmounts the lifted row loses its dragend, so drop ends the gesture too.
    const onEnd = () => clear();
    document.addEventListener("dragover", onDragOver);
    document.addEventListener("dragend", onEnd);
    document.addEventListener("drop", onEnd);
    return () => {
      document.removeEventListener("dragover", onDragOver);
      document.removeEventListener("dragend", onEnd);
      document.removeEventListener("drop", onEnd);
    };
  }, [drag, cancelSpring, clear, showPlan]);

  useEffect(() => cancelSpring, [cancelSpring]);

  const dragHandleProps = useCallback(
    (item: SidebarDragItem) => ({
      draggable: true as const,
      onDragStart: (event: React.DragEvent) => {
        // Firefox needs a payload to drag at all.
        event.dataTransfer.effectAllowed = "move";
        event.dataTransfer.setData("text/plain", item.id);
        event.stopPropagation();
        setSidebarDragSource(item);
        setDrag(item);
      },
      onDragEnd: clear,
    }),
    [clear],
  );

  const dropZoneProps = useCallback(
    (zone: SidebarDropZone, zoneOptions?: { closed?: boolean }) => {
      const springKey = zone.folderId
        ? `folder:${zone.folderId}`
        : `section:${zone.section}`;
      return {
        onDragOver: (event: React.DragEvent) => {
          const dragged = sidebarDragSource();
          if (!dragged || lastHandledEvent === event.nativeEvent) return;
          const next = planSidebarDrop(
            dragged,
            zone,
            dropEdgeAt(event.currentTarget.getBoundingClientRect(), event.clientY),
            optionsRef.current.context(),
          );
          // No answer here lets the zone around it answer instead.
          if (!next) return;
          event.preventDefault();
          lastHandledEvent = event.nativeEvent;
          event.dataTransfer.dropEffect = "move";
          if (next === STAY) {
            // Already here: nothing to paint, and nothing for the outer zones to add.
            cancelSpring();
            showPlan(null);
            return;
          }
          showPlan(next);
          if (zoneOptions?.closed && optionsRef.current.springOpen) {
            if (spring.current?.key !== springKey) {
              cancelSpring();
              spring.current = {
                key: springKey,
                timer: window.setTimeout(() => {
                  spring.current = null;
                  optionsRef.current.onSpringOpen?.(zone);
                }, SPRING_OPEN_DELAY_MS),
              };
            }
          } else if (spring.current?.key !== springKey) {
            cancelSpring();
          }
        },
        onDrop: (event: React.DragEvent) => {
          const dragged = sidebarDragSource();
          if (!dragged || lastHandledEvent === event.nativeEvent) return;
          const next = planSidebarDrop(
            dragged,
            zone,
            dropEdgeAt(event.currentTarget.getBoundingClientRect(), event.clientY),
            optionsRef.current.context(),
          );
          if (!next) return;
          event.preventDefault();
          lastHandledEvent = event.nativeEvent;
          clear();
          if (next !== STAY) optionsRef.current.onDrop(next, dragged);
        },
      };
    },
    [cancelSpring, clear, showPlan],
  );

  const lineEdge = useCallback(
    (scope: string, id: string): DropEdge | undefined => {
      if (!plan || !("line" in plan.cue)) return undefined;
      return plan.cue.line.rowKey === rowKey(scope, id)
        ? plan.cue.line.edge
        : undefined;
    },
    [plan],
  );

  const ringLit = useCallback(
    (key: string): boolean =>
      Boolean(plan && "ring" in plan.cue && plan.cue.ring === key),
    [plan],
  );

  return {
    drag,
    plan,
    dragHandleProps,
    dropZoneProps,
    lineEdge,
    ringLit,
  };
}
