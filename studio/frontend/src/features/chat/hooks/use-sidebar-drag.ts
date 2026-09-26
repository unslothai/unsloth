// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The React side of sidebar drag-and-drop: the lifted row, the plan under the pointer, and the
// pointer events that feed both. What a drop does is decided in lib/sidebar-drag.ts.
//
// The carried row lifts as a copy that follows the pointer, the way a section's header does
// (use-section-drag.ts), while the row it came from dims. The copy is drawn straight onto the
// DOM and moved from the frame loop, so the sidebar does not redraw as the pointer moves: React
// only hears when the plan under the pointer changes, and on the drop.
//
// Pointer events, not the HTML5 drag API. The desktop webview answers every OS drag itself and
// never forwards it to the page, so dragover and drop never fire there and a row wired to them
// is dead. Same reason shared-composer.tsx skips its drop handlers under isTauri.

import { useCallback, useEffect, useRef, useState } from "react";

import {
  dropEdgeAt,
  equivalentDrop,
  litRingKey,
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

/** The raised copy of a carried row. */
export const ROW_GHOST_CLASS = "sidebar-row-ghost";

/** How long a dropped row takes to slide from where it was let go into its slot. */
const SETTLE_MS = 180;

/** How near an edge of the scroller the pointer scrolls the list, and by how much per frame. */
export const EDGE_PX = 48;
export const EDGE_STEP_PX = 12;

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

/** A row's face: its icon and name, without the pin and menu that show on hover. */
const ROW_FACE_SELECTOR = '[data-sidebar="menu-button"]';

/** Attributes the copy of a row must not carry: it is a picture, not a control, a drop zone, a
 *  row a test or a lookup can find, or the open chat. */
const GHOST_DROPPED_ATTRS = [
  "id",
  DROP_ZONE_ATTR,
  ROW_KEY_ATTR,
  "data-active",
  "data-testid",
  "data-thread-id",
  "data-thread-type",
] as const;

interface RowGhost {
  element: HTMLElement;
  /** Where on the row the press landed, so the copy does not jump when it lifts. */
  grab: number;
  /** The box the copy stays inside: the list it scrolls in. */
  view: Element;
}

/** A copy of the row's face on a raised pill, over the row it came from. */
function liftRow(row: HTMLElement, pressY: number, view: Element): RowGhost | null {
  const face = row.querySelector<HTMLElement>(ROW_FACE_SELECTOR);
  if (!face) return null;
  const rect = face.getBoundingClientRect();
  const style = getComputedStyle(face);
  const copy = face.cloneNode(true) as HTMLElement;
  for (const node of [copy, ...copy.querySelectorAll<HTMLElement>("*")]) {
    for (const name of GHOST_DROPPED_ATTRS) node.removeAttribute(name);
  }
  copy.tabIndex = -1;
  const element = document.createElement("div");
  element.setAttribute("aria-hidden", "true");
  element.className = ROW_GHOST_CLASS;
  element.append(copy);
  Object.assign(element.style, {
    left: `${rect.left}px`,
    width: `${rect.width}px`,
    height: `${rect.height}px`,
    borderRadius: style.borderRadius,
    fontFamily: style.fontFamily,
    color: style.color,
  });
  // The icon size is the sidebar's own variable, which the copy leaves behind on the body.
  const iconSize = style.getPropertyValue("--icon-size");
  if (iconSize) element.style.setProperty("--icon-size", iconSize);
  document.body.append(element);
  return { element, grab: pressY - rect.top, view };
}

/** Keeps the copy under the pointer, inside the list it came from. */
function placeGhost(ghost: RowGhost, y: number) {
  const view = ghost.view.getBoundingClientRect();
  const height = ghost.element.offsetHeight;
  const top = Math.min(Math.max(y - ghost.grab, view.top), view.bottom - height);
  ghost.element.style.transform = `translate3d(0, ${Math.round(top)}px, 0)`;
}

/** What a drop zone says it is, or null if it cannot be read. */
function zoneOf(element: Element): SidebarDropZone | null {
  try {
    return (JSON.parse(element.getAttribute(DROP_ZONE_ATTR) ?? "") as { zone: SidebarDropZone }).zone;
  } catch {
    return null;
  }
}

/** Slides a dropped row from where its copy was let go into its new slot, and a folder's open
 *  chats with it. Read once the drop has redrawn the list, so it costs one pass per drop. */
function settleRow(item: SidebarDragItem, from: number) {
  // Two frames: the drop re-renders the sidebar, and the row is measured once it has moved.
  requestAnimationFrame(() =>
    requestAnimationFrame(() => {
      let row: HTMLElement | null = null;
      const moving: HTMLElement[] = [];
      for (const element of document.querySelectorAll<HTMLElement>(`[${DROP_ZONE_ATTR}]`)) {
        if (element.offsetHeight === 0) continue;
        const zone = zoneOf(element);
        if (!zone) continue;
        const isRow = zone.row?.id === item.id && zone.row.kind === item.kind && !zone.header;
        if (isRow && element.hasAttribute(ROW_KEY_ATTR)) {
          // The copy of the row nearest where it was let go, should it be drawn twice.
          const face = element.querySelector(ROW_FACE_SELECTOR) ?? element;
          const top = face.getBoundingClientRect().top;
          const best = row?.querySelector(ROW_FACE_SELECTOR) ?? row;
          if (!best || Math.abs(top - from) < Math.abs(best.getBoundingClientRect().top - from)) {
            row = element;
          }
        }
        // A folder carries its open chats and its empty line: the spots that name it.
        if (
          item.kind === "project" &&
          zone.folderId === item.id &&
          zone.row?.kind !== "project" &&
          !zone.header
        ) {
          moving.push(element);
        }
      }
      if (!row) return;
      const to = (row.querySelector(ROW_FACE_SELECTOR) ?? row).getBoundingClientRect().top;
      const delta = from - to;
      if (Math.abs(delta) < 2) return;
      // Only the outermost of nested spots move, or a spot would move twice.
      const all = [row, ...moving.filter((element) => element !== row)];
      const slide = all.filter(
        (element) => !all.some((other) => other !== element && other.contains(element)),
      );
      for (const element of slide) {
        element.style.transition = "none";
        element.style.transform = `translateY(${delta}px)`;
        element.style.zIndex = "1";
      }
      row.getBoundingClientRect();
      for (const element of slide) {
        element.style.transition = `transform ${SETTLE_MS}ms cubic-bezier(0.2, 0.8, 0.2, 1)`;
        element.style.transform = "";
      }
      window.setTimeout(() => {
        for (const element of slide) {
          element.style.transition = "";
          element.style.zIndex = "";
        }
      }, SETTLE_MS + 20);
    }),
  );
}

/** The list the row scrolls while it is carried: the nearest ancestor that actually scrolls. */
export function scrollerOf(element: Element | null): HTMLElement | null {
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
  /** Read on drop: whether the row may slide into place or should just appear there. */
  reducedMotion?: () => boolean;
}

export interface SidebarDragApi {
  /** The carried row, for painting it dimmed where it was. Null between drags. */
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
  const ghost = useRef<RowGhost | null>(null);
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
    ghost.current?.element.remove();
    ghost.current = null;
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
        // Touch scrolls the list. The right button opens the row menu, and a control with its
        // own press keeps it.
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
          if (ghost.current) placeGhost(ghost.current, at.y);
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
            ghost.current = liftRow(
              row,
              startY,
              scroller.current ?? row.closest('[data-sidebar="content"]') ?? document.documentElement,
            );
            if (ghost.current) placeGhost(ghost.current, moved.clientY);
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
          const from = ghost.current?.element.getBoundingClientRect().top ?? null;
          clear();
          if (dragged && aimed && aimed.outcome !== STAY) {
            optionsRef.current.onDrop(aimed.outcome, dragged);
            if (from !== null && !optionsRef.current.reducedMotion?.()) settleRow(dragged, from);
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
    (key: string): boolean => litRingKey(plan) === key,
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
