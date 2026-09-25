// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The React side of sidebar drag-and-drop: the lifted row, the plan under the pointer, and the
// pointer events that feed both. What a drop does is decided in lib/sidebar-drag.ts.
//
// Pointer events, not the HTML5 drag API. The desktop webview answers every OS drag itself and
// never forwards it to the page, so dragover and drop never fire there and a row wired to them
// is dead. Same reason shared-composer.tsx skips its drop handlers under isTauri.

import { useCallback, useEffect, useRef, useState } from "react";

import {
  dropEdgeAt,
  equivalentDrop,
  planKey,
  planSidebarDrop,
  rowKey,
  SIDEBAR_TAIL_SCOPE,
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

/** Pointer travel before a press on a row becomes a drag, so a click stays a click and a
 *  double-click still renames. */
export const DRAG_THRESHOLD_PX = 5;

/** Marks a drop zone in the DOM, carrying the zone itself. A zone needs no ref and no
 *  registration: the hit test reads it back off whatever sits under the pointer. */
export const DROP_ZONE_ATTR = "data-sidebar-drop";

/** Set on the body for the length of a drag: no text selection, and the grabbing cursor
 *  follows the row off its own list. */
export const DRAGGING_BODY_CLASS = "sidebar-row-dragging";

/** Controls that own their own press. A drag never starts from the pin or the kebab. */
const NO_DRAG_SELECTOR = ".sidebar-row-action";

/** How near an edge of the scroller the pointer scrolls the list, and by how much per move. */
const EDGE_PX = 48;
const EDGE_STEP_PX = 12;

interface ZoneHit {
  zone: SidebarDropZone;
  closed: boolean;
  rect: DOMRect;
}

/** The drop zones under the pointer, innermost first: a row paints over its folder block,
 *  which paints over its section. */
function zonesUnder(x: number, y: number): ZoneHit[] {
  if (typeof document === "undefined") return [];
  const hits: ZoneHit[] = [];
  for (const element of document.elementsFromPoint(x, y)) {
    const raw = element.getAttribute(DROP_ZONE_ATTR);
    if (!raw) continue;
    try {
      const parsed = JSON.parse(raw) as {
        zone: SidebarDropZone;
        closed?: boolean;
      };
      hits.push({
        zone: parsed.zone,
        closed: Boolean(parsed.closed),
        rect: element.getBoundingClientRect(),
      });
    } catch {
      // A zone that cannot be read is one the pointer passes through.
    }
  }
  return hits;
}

/** Max distance between a row's bottom and the next row's top for them to share a gap. */
const ADJACENT_PX = 3;

/** Marks the row, or section tail, a line can be drawn on with its row key, so a neighbour
 *  lookup finds it directly instead of scanning every zone. */
const ROW_KEY_ATTR = "data-sidebar-row-key";

/** The row drawn directly below or above the row, or section tail, that `key` names. Probes
 *  just past its edge, so the cost does not grow with the list. */
function rowNextTo(key: string, side: "below" | "above"): ZoneHit | null {
  if (typeof document === "undefined") return null;
  const self = document.querySelector(`[${ROW_KEY_ATTR}="${CSS.escape(key)}"]`);
  if (!self) return null;
  const from = self.getBoundingClientRect();
  // Rows overlap by a pixel (DROP_ROW_HIT), so step two in to land inside the neighbour.
  const y = side === "below" ? from.bottom + 1 : from.top - 2;
  for (const element of document.elementsFromPoint(from.left + from.width / 2, y)) {
    if (element === self || !element.hasAttribute(ROW_KEY_ATTR)) continue;
    const raw = element.getAttribute(DROP_ZONE_ATTR);
    if (!raw) continue;
    try {
      const parsed = JSON.parse(raw) as { zone: SidebarDropZone; closed?: boolean };
      if (!parsed.zone.row) continue;
      const rect = element.getBoundingClientRect();
      const gap = side === "below" ? rect.top - from.bottom : from.top - rect.bottom;
      if (Math.abs(gap) > ADJACENT_PX) return null;
      return { zone: parsed.zone, closed: Boolean(parsed.closed), rect };
    } catch {
      return null;
    }
  }
  return null;
}

/** The list the row scrolls while it is carried: the nearest ancestor that actually scrolls. */
function scrollerOf(element: Element | null): HTMLElement | null {
  for (let node = element; node; node = node.parentElement) {
    if (!(node instanceof HTMLElement)) continue;
    const overflow = getComputedStyle(node).overflowY;
    if (
      (overflow === "auto" || overflow === "scroll") &&
      node.scrollHeight > node.clientHeight
    ) {
      return node;
    }
  }
  return null;
}

export interface UseSidebarDragOptions {
  /** Read at event time, so a plan sees the lists as they stand. */
  context: () => SidebarDropContext;
  /** Commits the plan the row was dropped on. */
  onDrop: (plan: SidebarDropPlan, drag: SidebarDragItem) => void;
  /** Opens the closed folder or section the pointer rested on. */
  onSpringOpen?: (zone: SidebarDropZone) => void;
}

export interface SidebarDragApi {
  /** The lifted row, for painting it faded. Null between drags. */
  drag: SidebarDragItem | null;
  /** What the drop under the pointer would do. */
  plan: SidebarDropPlan | null;
  /** Props that let a row be picked up. */
  dragHandleProps: (item: SidebarDragItem) => {
    onPointerDown: (event: React.PointerEvent) => void;
  };
  /** Props that let a spot take a drop. `closed` marks a collapsed folder or section. */
  dropZoneProps: (
    zone: SidebarDropZone,
    options?: { closed?: boolean },
  ) => Record<string, string>;
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
  const scroller = useRef<HTMLElement | null>(null);
  /** The gesture in flight, so a second pointer cannot start another over the top of it. */
  const press = useRef<{ end: () => void } | null>(null);

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
    scroller.current = null;
    document.body.classList.remove(DRAGGING_BODY_CLASS);
    setDrag(null);
    showPlan(null);
  }, [cancelSpring, showPlan]);

  useEffect(() => () => clear(), [clear]);

  /** The first zone under the pointer with an answer. No answer lets the zone around it
   *  answer instead. */
  const aim = useCallback(
    (
      x: number,
      y: number,
    ): { hit: ZoneHit; outcome: SidebarDropPlan | typeof STAY } | null => {
      const dragged = sidebarDragSource();
      if (!dragged) return null;
      const context = optionsRef.current.context();
      for (const hit of zonesUnder(x, y)) {
        const outcome = planSidebarDrop(
          dragged,
          hit.zone,
          dropEdgeAt(hit.rect, y),
          context,
        );
        if (!outcome) continue;
        // One line per gap: a bottom line defers to the row below's top, and a section tail's
        // line to the row above's bottom, when that lands the drop identically. Different drops
        // (a folder's last chat vs the next row) keep their own lines.
        if (outcome !== STAY && "line" in outcome.cue) {
          const { rowKey: key, edge } = outcome.cue.line;
          const tail = key.startsWith(`${SIDEBAR_TAIL_SCOPE}:`);
          const next = tail
            ? rowNextTo(key, "above")
            : edge === "bottom"
              ? rowNextTo(key, "below")
              : null;
          const alt = next
            ? planSidebarDrop(dragged, next.zone, tail ? "bottom" : "top", context)
            : null;
          if (
            alt &&
            alt !== STAY &&
            "line" in alt.cue &&
            equivalentDrop(alt, outcome)
          ) {
            // Only the painted line moves: the original `place` is what a slow move re-aims by.
            return { hit, outcome: { ...outcome, cue: alt.cue } };
          }
        }
        return { hit, outcome };
      }
      return null;
    },
    [],
  );

  /** One step of the edge scroll. Driven by the frame loop, not by pointermove: a pointer resting
   *  on the edge sends no moves and would stall. */
  const edgeScroll = useCallback((y: number) => {
    const list = scroller.current;
    if (!list) return;
    const rect = list.getBoundingClientRect();
    if (y < rect.top + EDGE_PX) list.scrollTop -= EDGE_STEP_PX;
    else if (y > rect.bottom - EDGE_PX) list.scrollTop += EDGE_STEP_PX;
  }, []);

  const track = useCallback(
    (x: number, y: number) => {
      const aimed = aim(x, y);
      if (!aimed || aimed.outcome === STAY) {
        // Nothing here, or the row is already in this slot: nothing to paint either way.
        cancelSpring();
        showPlan(null);
        return;
      }
      showPlan(aimed.outcome);

      const { zone, closed } = aimed.hit;
      const springKey = zone.folderId
        ? `folder:${zone.folderId}`
        : `section:${zone.section}`;
      if (!closed) {
        if (spring.current?.key !== springKey) cancelSpring();
        return;
      }
      if (spring.current?.key === springKey) return;
      cancelSpring();
      spring.current = {
        key: springKey,
        timer: window.setTimeout(() => {
          spring.current = null;
          optionsRef.current.onSpringOpen?.(zone);
        }, SPRING_OPEN_DELAY_MS),
      };
    },
    [aim, cancelSpring, showPlan],
  );

  const dragHandleProps = useCallback(
    (item: SidebarDragItem) => ({
      onPointerDown: (event: React.PointerEvent) => {
        // Touch scrolls the list and keeps Move up and Move down. The right button opens the
        // row menu, and a control with its own press keeps it.
        if (event.button !== 0 || event.pointerType === "touch") return;
        if (!event.isPrimary) return;
        if ((event.target as Element | null)?.closest?.(NO_DRAG_SELECTOR)) {
          return;
        }
        // Abandoned rather than refused: a release the window never saw would otherwise leave a
        // gesture in flight for good and block every drag after it.
        press.current?.end();

        const startX = event.clientX;
        const startY = event.clientY;
        const row = event.currentTarget as HTMLElement;
        const pointerId = event.pointerId;
        const at = { x: startX, y: startY };
        let started = false;
        let escaped = false;
        let frame = 0;

        const detach = () => {
          if (press.current === self) press.current = null;
          if (frame) cancelAnimationFrame(frame);
          frame = 0;
          window.removeEventListener("pointermove", onMove);
          window.removeEventListener("pointerup", onUp);
          window.removeEventListener("pointercancel", onCancel);
          window.removeEventListener("keydown", onKey, true);
          // Captured on the body, which no re-render can unmount mid-gesture. A row can.
          if (document.body.hasPointerCapture?.(pointerId)) {
            document.body.releasePointerCapture(pointerId);
          }
        };
        // The rows move under the pointer, not the other way about: the list edge-scrolls, a
        // folder springs open under a pointer that by definition is resting. So every frame
        // re-aims, not only the ones that scroll, or the cue would describe the layout as it was
        // when the pointer last moved while the release hit-tests the layout as it is.
        const onFrame = () => {
          // Self-terminating, so an unmount mid-drag cannot leave the loop running.
          if (!sidebarDragSource()) {
            frame = 0;
            return;
          }
          frame = requestAnimationFrame(onFrame);
          edgeScroll(at.y);
          track(at.x, at.y);
        };
        // Abandons this gesture whole, for a drop that never came: the same as a cancel.
        const self = {
          end: () => {
            detach();
            if (started) clear();
          },
        };
        press.current = self;

        // Swallow the click the release would otherwise fire on the row it landed on.
        const swallowClick = () => {
          const stop = (clicked: MouseEvent) => {
            clicked.preventDefault();
            clicked.stopPropagation();
          };
          window.addEventListener("click", stop, { capture: true, once: true });
          window.setTimeout(() => {
            window.removeEventListener("click", stop, { capture: true });
          }, 0);
        };

        // Every window handler answers only the pointer that started the gesture. A second
        // pointer, a finger on a touch screen above all, is not this drag.
        function onMove(moved: PointerEvent) {
          if (moved.pointerId !== pointerId || escaped) return;
          at.x = moved.clientX;
          at.y = moved.clientY;
          if (!started) {
            if (
              Math.abs(moved.clientX - startX) < DRAG_THRESHOLD_PX &&
              Math.abs(moved.clientY - startY) < DRAG_THRESHOLD_PX
            ) {
              return;
            }
            started = true;
            scroller.current = scrollerOf(row);
            document.body.classList.add(DRAGGING_BODY_CLASS);
            // Without capture a release outside the window never arrives and the row stays
            // stuck. Capture retargets events only, so the hit test still finds the zone.
            try {
              document.body.setPointerCapture(pointerId);
            } catch {
              // No capture: window listeners still carry the drag inside the window.
            }
            setSidebarDragSource(item);
            setDrag(item);
            frame = requestAnimationFrame(onFrame);
          }
          // Held by the window, so the row keeps following the pointer outside the sidebar.
          moved.preventDefault();
          track(moved.clientX, moved.clientY);
        }

        function onUp(released: PointerEvent) {
          if (released.pointerId !== pointerId) return;
          detach();
          if (!started) return;
          // A drop or an escape: either way this release must not click the row it landed on.
          swallowClick();
          if (escaped) return;
          const dragged = sidebarDragSource();
          const aimed = aim(released.clientX, released.clientY);
          clear();
          if (dragged && aimed && aimed.outcome !== STAY) {
            optionsRef.current.onDrop(aimed.outcome, dragged);
          }
        }

        function onCancel(aborted: PointerEvent) {
          if (aborted.pointerId !== pointerId) return;
          detach();
          if (started) clear();
        }

        function onKey(pressed: KeyboardEvent) {
          if (pressed.key !== "Escape" || escaped) return;
          if (!started) {
            // Nothing was lifted, so the press is abandoned and a click after it is the row's.
            detach();
            return;
          }
          pressed.preventDefault();
          // The button is still down. The gesture keeps its listeners so the release, whenever
          // it comes, is still ours to swallow: a guard armed now would be long gone by then.
          escaped = true;
          clear();
        }

        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp);
        window.addEventListener("pointercancel", onCancel);
        window.addEventListener("keydown", onKey, true);
      },
    }),
    [aim, clear, edgeScroll, track],
  );

  const dropZoneProps = useCallback(
    (zone: SidebarDropZone, zoneOptions?: { closed?: boolean }) => {
      const props: Record<string, string> = {
        [DROP_ZONE_ATTR]: JSON.stringify(
          zoneOptions?.closed ? { zone, closed: true } : { zone },
        ),
      };
      const key =
        zone.row && !zone.header
          ? rowKey(zone.row.scope, zone.row.id)
          : !zone.row && zone.blockEnd?.scope === SIDEBAR_TAIL_SCOPE
            ? rowKey(SIDEBAR_TAIL_SCOPE, zone.blockEnd.id)
            : null;
      if (key) props[ROW_KEY_ATTR] = key;
      return props;
    },
    [],
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
