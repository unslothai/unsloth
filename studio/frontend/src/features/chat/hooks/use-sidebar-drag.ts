// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// React side of sidebar drag-and-drop; drop semantics live in lib/sidebar-drag.ts.
// Pointer events, not HTML5 drag: the desktop webview never forwards OS drags to the page.

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

export const SPRING_OPEN_DELAY_MS = 450;

/** Travel before a press becomes a drag, so clicks and double-click rename still work. */
export const DRAG_THRESHOLD_PX = 5;

/** Marks a drop zone; the hit test reads the zone back off the element under the pointer. */
export const DROP_ZONE_ATTR = "data-sidebar-drop";

/** Set on the drag box, never the body: restyling the body caused a visible hitch. */
export const DRAGGING_CLASS = "pointer-dragging";

function refuseSelection(event: Event) {
  event.preventDefault();
}

export function markDragging(box: Element | null, on: boolean) {
  if (on) {
    box?.classList.add(DRAGGING_CLASS);
    window.getSelection()?.removeAllRanges();
    document.addEventListener("selectstart", refuseSelection, true);
    return;
  }
  document.removeEventListener("selectstart", refuseSelection, true);
  for (const marked of document.querySelectorAll(`.${DRAGGING_CLASS}`)) {
    marked.classList.remove(DRAGGING_CLASS);
  }
}

/** Persistent drag overlay layer: removing body children restyles the whole page. */
export function dragLayer(): HTMLElement {
  let layer = document.getElementById(DRAG_LAYER_ID);
  if (!layer) {
    layer = document.createElement("div");
    layer.id = DRAG_LAYER_ID;
    layer.setAttribute("aria-hidden", "true");
    document.body.append(layer);
  }
  return layer;
}

const DRAG_LAYER_ID = "pointer-drag-layer";

export const sidebarOf = (element: Element): Element | null =>
  element.closest('[data-sidebar="sidebar"]');

const NO_DRAG_SELECTOR = ".sidebar-row-action";

export const ROW_GHOST_CLASS = "sidebar-row-ghost";

export const DROP_CUE_CLASS = "sidebar-drop-cue";

const CUE_OVERLAY_CLASS = "sidebar-drop-cue-overlay";

const SETTLE_MS = 180;

export const EDGE_PX = 48;
export const EDGE_STEP_PX = 12;

interface ZoneHit {
  zone: SidebarDropZone;
  closed: boolean;
  rect: DOMRect;
}

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

/** Recents is always last, so empty sidebar below it acts as its end strip. */
function zonePastRecents(x: number, y: number, list: Element | null): ZoneHit | null {
  if (typeof document === "undefined") return null;
  const tail = document.querySelector(
    `[${ROW_KEY_ATTR}="${CSS.escape(rowKey(SIDEBAR_TAIL_SCOPE, "recents"))}"]`,
  );
  if (!tail) return null;
  const rect = tail.getBoundingClientRect();
  if (rect.height === 0 || y < rect.bottom || x < rect.left || x > rect.right) return null;
  if (list && y > list.getBoundingClientRect().bottom) return null;
  try {
    const parsed = JSON.parse(tail.getAttribute(DROP_ZONE_ATTR) ?? "") as {
      zone: SidebarDropZone;
    };
    return { zone: parsed.zone, closed: false, rect };
  } catch {
    return null;
  }
}

const ADJACENT_PX = 3;

const ROW_KEY_ATTR = "data-sidebar-row-key";

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

const ROW_FACE_SELECTOR = '[data-sidebar="menu-button"]';

const GHOST_DROPPED_ATTRS = [
  "id",
  DROP_ZONE_ATTR,
  ROW_KEY_ATTR,
  "data-active",
  "data-testid",
  "data-thread-id",
  "data-thread-type",
] as const;

export interface RowGhost {
  element: HTMLElement;
  grab: number;
  view: Element;
  top: number;
  cues: HTMLElement[];
}

function liftRow(row: HTMLElement, pressY: number, view: Element): RowGhost | null {
  const face = row.querySelector<HTMLElement>(ROW_FACE_SELECTOR);
  if (!face) return null;
  const ghost = liftCopy(face, pressY, view, ROW_GHOST_CLASS, GHOST_DROPPED_ATTRS);
  const iconSize = getComputedStyle(face).getPropertyValue("--icon-size");
  if (iconSize) ghost.element.style.setProperty("--icon-size", iconSize);
  return ghost;
}

export function liftCopy(
  face: HTMLElement,
  pressY: number,
  view: Element,
  className: string,
  dropped: readonly string[],
): RowGhost {
  const rect = face.getBoundingClientRect();
  const style = getComputedStyle(face);
  const copy = face.cloneNode(true) as HTMLElement;
  for (const node of [copy, ...copy.querySelectorAll<HTMLElement>("*")]) {
    for (const name of dropped) node.removeAttribute(name);
  }
  copy.tabIndex = -1;
  const element = document.createElement("div");
  element.setAttribute("aria-hidden", "true");
  element.inert = true;
  element.className = className;
  element.append(copy);
  // Lifted where the row is, so a frame before the first transform is not at the top.
  Object.assign(element.style, {
    top: `${rect.top}px`,
    left: `${rect.left}px`,
    width: `${rect.width}px`,
    height: `${rect.height}px`,
    borderRadius: style.borderRadius,
    fontFamily: style.fontFamily,
    color: style.color,
  });
  dragLayer().append(element);
  return { element, grab: pressY - rect.top, view, top: rect.top, cues: [] };
}

/** Redraw cue borders over the copy (border only, or the tint doubles). */
export function placeCue(ghost: RowGhost) {
  const view = ghost.view.getBoundingClientRect();
  let shown = 0;
  for (const cue of document.querySelectorAll<HTMLElement>(`.${DROP_CUE_CLASS}`)) {
    const style = getComputedStyle(cue, "::before");
    if (style.content === "none" || cue.offsetHeight === 0) continue;
    let overlay = ghost.cues[shown];
    if (!overlay) {
      overlay = document.createElement("div");
      overlay.setAttribute("aria-hidden", "true");
      overlay.className = CUE_OVERLAY_CLASS;
      dragLayer().append(overlay);
      ghost.cues.push(overlay);
    }
    shown += 1;
    const box = cue.getBoundingClientRect();
    const left = box.left + (parseFloat(style.left) || 0);
    const top = box.top + (parseFloat(style.top) || 0);
    const height = parseFloat(style.height) || 0;
    Object.assign(overlay.style, {
      display: "block",
      width: style.width,
      height: `${height}px`,
      borderStyle: style.borderStyle,
      borderColor: style.borderColor,
      borderTopWidth: style.borderTopWidth,
      borderRightWidth: style.borderRightWidth,
      borderBottomWidth: style.borderBottomWidth,
      borderLeftWidth: style.borderLeftWidth,
      borderRadius: style.borderRadius,
      left: `${left}px`,
      top: `${top}px`,
      clipPath: `inset(${Math.max(0, view.top - top)}px 0 ${Math.max(0, top + height - view.bottom)}px 0)`,
    });
  }
  for (const overlay of ghost.cues.slice(shown)) overlay.style.display = "none";
}

export function placeGhost(ghost: RowGhost, y: number) {
  const view = ghost.view.getBoundingClientRect();
  const height = ghost.element.offsetHeight;
  const top = Math.min(Math.max(y - ghost.grab, view.top), view.bottom - height);
  ghost.element.style.transform = `translate3d(0, ${Math.round(top - ghost.top)}px, 0)`;
}

function zoneOf(element: Element): SidebarDropZone | null {
  try {
    return (JSON.parse(element.getAttribute(DROP_ZONE_ATTR) ?? "") as { zone: SidebarDropZone }).zone;
  } catch {
    return null;
  }
}

function settleRow(item: SidebarDragItem, from: number) {
  // Two frames: the drop re-renders the sidebar before the row can be measured.
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
          const face = element.querySelector(ROW_FACE_SELECTOR) ?? element;
          const top = face.getBoundingClientRect().top;
          const best = row?.querySelector(ROW_FACE_SELECTOR) ?? row;
          if (!best || Math.abs(top - from) < Math.abs(best.getBoundingClientRect().top - from)) {
            row = element;
          }
        }
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
  context: () => SidebarDropContext;
  onDrop: (plan: SidebarDropPlan, drag: SidebarDragItem) => void;
  onSpringOpen?: (zone: SidebarDropZone) => void;
  reducedMotion?: () => boolean;
}

export interface SidebarDragApi {
  drag: SidebarDragItem | null;
  plan: SidebarDropPlan | null;
  dragHandleProps: (item: SidebarDragItem) => {
    onPointerDown: (event: React.PointerEvent) => void;
  };
  dropZoneProps: (
    zone: SidebarDropZone,
    options?: { closed?: boolean },
  ) => Record<string, string>;
  lineAt: (scope: string, id: string) => { edge: DropEdge; inFolder: boolean } | undefined;
  ringLit: (key: string) => boolean;
}

export function useSidebarDrag(options: UseSidebarDragOptions): SidebarDragApi {
  const [drag, setDrag] = useState<SidebarDragItem | null>(null);
  const [plan, setPlan] = useState<SidebarDropPlan | null>(null);
  const planRef = useRef<{ key: string; plan: SidebarDropPlan | null }>({
    key: "",
    plan: null,
  });
  const optionsRef = useRef(options);
  useEffect(() => {
    optionsRef.current = options;
  });
  const spring = useRef<{ key: string; timer: number } | null>(null);
  const scroller = useRef<HTMLElement | null>(null);
  const ghost = useRef<RowGhost | null>(null);
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
    for (const overlay of ghost.current?.cues ?? []) overlay.remove();
    ghost.current = null;
    markDragging(null, false);
    setDrag(null);
    showPlan(null);
  }, [cancelSpring, showPlan]);

  useEffect(() => () => clear(), [clear]);

  const aim = useCallback(
    (
      x: number,
      y: number,
    ): { hit: ZoneHit; outcome: SidebarDropPlan | typeof STAY } | null => {
      const dragged = sidebarDragSource();
      if (!dragged) return null;
      const context = optionsRef.current.context();
      const under = zonesUnder(x, y);
      const past = under.length === 0 ? zonePastRecents(x, y, scroller.current) : null;
      for (const hit of past ? [past] : under) {
        const outcome = planSidebarDrop(
          dragged,
          hit.zone,
          dropEdgeAt(hit.rect, y),
          context,
        );
        if (!outcome) continue;
        // One line per gap when both neighbours land the drop identically.
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
            return { hit, outcome: { ...outcome, cue: alt.cue } };
          }
        }
        return { hit, outcome };
      }
      return null;
    },
    [],
  );

  /** Edge-scroll step from the frame loop: a resting pointer sends no moves. */
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
        if (event.button !== 0 || event.pointerType === "touch") return;
        if (!event.isPrimary) return;
        if ((event.target as Element | null)?.closest?.(NO_DRAG_SELECTOR)) {
          return;
        }
        // Abandon, not refuse: a release the window never saw would block every later drag.
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
          // Captured on the body, which no re-render can unmount mid-gesture.
          if (document.body.hasPointerCapture?.(pointerId)) {
            document.body.releasePointerCapture(pointerId);
          }
        };
        // Re-aim every frame: rows move under a resting pointer (edge scroll, spring open).
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
          if (ghost.current) placeCue(ghost.current);
        };
        const self = {
          end: () => {
            detach();
            if (started) clear();
          },
        };
        press.current = self;

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

        // Answer only the starting pointer; a second finger is not this drag.
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
            markDragging(sidebarOf(row), true);
            // Without capture a release outside the window never arrives and the row sticks.
            try {
              document.body.setPointerCapture(pointerId);
            } catch {
              // No capture: window listeners still carry the drag inside the window.
            }
            setSidebarDragSource(item);
            setDrag(item);
            frame = requestAnimationFrame(onFrame);
          }
          moved.preventDefault();
          track(moved.clientX, moved.clientY);
        }

        function onUp(released: PointerEvent) {
          if (released.pointerId !== pointerId) return;
          detach();
          if (!started) return;
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
            detach();
            return;
          }
          pressed.preventDefault();
          // Button still down: keep listeners so the eventual release is swallowed.
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

  const lineAt = useCallback(
    (scope: string, id: string) => {
      if (!plan || !("line" in plan.cue)) return undefined;
      const { rowKey: key, edge, folderId } = plan.cue.line;
      return key === rowKey(scope, id) ? { edge, inFolder: folderId !== undefined } : undefined;
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
    lineAt,
    ringLit,
  };
}
