// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Reordering the sidebar's sections by their headers. The grabbed header lifts and follows the
// pointer, the section it came from dims, and one line glides to the gap it would land in. All
// of that is drawn straight onto the DOM: the sidebar is one large component, and redrawing it
// on every pointer move is what made the drag lag. React only hears about the drop.
//
// Pointer events, for the same reason as the row drag: the desktop webview never forwards the
// HTML5 drag API to the page.

import { useCallback, useEffect, useRef } from "react";

import type { DropEdge } from "../lib/sidebar-drag.ts";
import { placeIdAt } from "../stores/sidebar-organization-store.ts";
import {
  DRAG_THRESHOLD_PX,
  DRAGGING_BODY_CLASS,
  EDGE_PX,
  EDGE_STEP_PX,
  scrollerOf,
} from "./use-sidebar-drag.ts";

/** Marks a draggable section's box with its key. */
export const SECTION_ATTR = "data-sidebar-section";
/** Set on the section being carried, which the stylesheet dims. */
export const SECTION_DRAGGING_ATTR = "data-section-dragging";
const HEADER_SELECTOR = '[data-sidebar="group-label"]';
/** The header's own buttons ("+", "..."), which keep their press. */
const HEADER_ACTION_SELECTOR = ".sidebar-header-action";
/** How long a dropped section takes to slide from where it was let go into its slot. */
const SETTLE_MS = 180;

export type SectionLanding = { target: string; edge: DropEdge };

/** A drawn section, top to bottom, as the landing reads it. */
export type SectionBlock = { key: string; top: number; bottom: number };

/**
 * Where a section carried to `y` lands among `blocks` (the carried one left out): against the
 * top of the first block whose middle is below the pointer, or under the last block. Null with
 * nowhere to go.
 */
export function sectionLandingAt(
  blocks: readonly SectionBlock[],
  y: number,
): SectionLanding | null {
  for (const block of blocks) {
    if (y < (block.top + block.bottom) / 2) return { target: block.key, edge: "top" };
  }
  const last = blocks.at(-1);
  return last ? { target: last.key, edge: "bottom" } : null;
}

/** Whether landing `key` there changes the order `drawn` shows. A landing that leaves it as it
 *  is draws no line, so a line always means the drop will do something. */
export function landingMoves(
  drawn: readonly string[],
  key: string,
  landing: SectionLanding,
): boolean {
  const next = placeIdAt([...drawn], key, landing.target, landing.edge);
  return next.some((id, index) => id !== drawn[index]);
}

/** The height of the gap above a header's text, where its line is drawn: the header carries the
 *  space between sections as its own top padding. */
function lineYAbove(header: Element): number {
  const box = header.getBoundingClientRect();
  const text = (header.querySelector("button") ?? header).getBoundingClientRect();
  return Math.round(box.top + (text.top - box.top) / 2);
}

/** The next drawn element after `element` that has a header of its own. */
function headedSiblingAfter(element: Element | undefined): Element | null {
  for (let next = element?.nextElementSibling; next; next = next.nextElementSibling) {
    if (next instanceof HTMLElement && next.offsetHeight > 0 && next.querySelector(HEADER_SELECTOR)) {
      return next;
    }
  }
  return null;
}

export interface UseSectionDragOptions {
  /** Commits a drop that changes the order. */
  onDrop: (key: string, landing: SectionLanding) => void;
  /** Read on drop: whether the section may slide into place or should just appear there. */
  reducedMotion?: () => boolean;
}

/** Returns the header's pointerdown handler; spread it on the box marked with SECTION_ATTR. */
export function useSectionDrag(
  options: UseSectionDragOptions,
): (event: React.PointerEvent<HTMLElement>, key: string) => void {
  const optionsRef = useRef(options);
  useEffect(() => {
    optionsRef.current = options;
  });
  /** The gesture in flight, so a second press cannot start another over it. */
  const gesture = useRef<{ end: () => void } | null>(null);
  useEffect(() => () => gesture.current?.end(), []);

  return useCallback((event: React.PointerEvent<HTMLElement>, key: string) => {
    // Touch scrolls the sidebar, the right button opens a menu, and "+" or "..." keep their press.
    if (event.button !== 0 || event.pointerType === "touch" || !event.isPrimary) return;
    const pressed = event.target as Element;
    const header = pressed.closest(HEADER_SELECTOR);
    if (!header || !event.currentTarget.contains(header)) return;
    if (pressed.closest(HEADER_ACTION_SELECTOR)) return;
    const block = event.currentTarget;
    const parent = block.parentElement;
    if (!parent) return;
    // Typed here, not narrowed: the handlers below are hoisted and would lose the narrowing.
    const list: HTMLElement = parent;
    gesture.current?.end();

    const pointerId = event.pointerId;
    const startX = event.clientX;
    const startY = event.clientY;
    let pointerY = startY;
    let started = false;
    let escaped = false;
    let frame = 0;
    let landing: SectionLanding | null = null;
    let scroller: HTMLElement = list;
    let ghost: HTMLElement | null = null;
    let line: HTMLElement | null = null;
    // Where on the header text the press landed, so the lifted copy does not jump.
    const text = header.querySelector("button") ?? header;
    const grab = startY - text.getBoundingClientRect().top;

    const drawnBlocks = () =>
      [...list.querySelectorAll<HTMLElement>(`:scope > [${SECTION_ATTR}]`)].filter(
        (element) => element.offsetHeight > 0,
      );

    /** A copy of the header's text on a raised pill, over the header it came from. */
    const lift = () => {
      const textRect = text.getBoundingClientRect();
      const headerRect = header.getBoundingClientRect();
      const style = getComputedStyle(header);
      ghost = document.createElement("div");
      ghost.setAttribute("aria-hidden", "true");
      ghost.className = "sidebar-section-ghost";
      const copy = text.cloneNode(true) as HTMLElement;
      copy.removeAttribute("id");
      copy.tabIndex = -1;
      ghost.append(copy);
      const inset = 10;
      Object.assign(ghost.style, {
        left: `${textRect.left - inset}px`,
        width: `${headerRect.right - 8 - (textRect.left - inset)}px`,
        height: `${textRect.height + 12}px`,
        paddingInline: `${inset}px`,
        fontFamily: style.fontFamily,
        fontSize: style.fontSize,
        fontWeight: style.fontWeight,
        letterSpacing: style.letterSpacing,
        color: style.color,
      });
      line = document.createElement("div");
      line.setAttribute("aria-hidden", "true");
      line.className = "sidebar-section-drop-line";
      document.body.append(ghost, line);
    };

    const place = () => {
      const view = scroller.getBoundingClientRect();
      if (ghost) {
        const height = ghost.offsetHeight;
        const top = Math.min(
          Math.max(pointerY - grab - 6, view.top),
          view.bottom - height,
        );
        ghost.style.transform = `translate3d(0, ${Math.round(top)}px, 0)`;
      }
      const blocks = drawnBlocks();
      const drawn = blocks.map((element) => element.getAttribute(SECTION_ATTR)!);
      const others = blocks.filter((element) => element !== block);
      const hit = sectionLandingAt(
        others.map((element) => {
          const rect = element.getBoundingClientRect();
          return { key: element.getAttribute(SECTION_ATTR)!, top: rect.top, bottom: rect.bottom };
        }),
        pointerY,
      );
      landing = hit && landingMoves(drawn, key, hit) ? hit : null;
      if (!line) return;
      let y: number | null = null;
      if (landing) {
        const { target: targetKey, edge } = landing;
        const target = others.find((element) => element.getAttribute(SECTION_ATTR) === targetKey);
        // Under the last section is the gap above whatever follows it: Recents' header.
        const below = edge === "top" ? target : headedSiblingAfter(target);
        const belowHeader = below?.querySelector(HEADER_SELECTOR);
        if (belowHeader) y = lineYAbove(belowHeader);
        else if (target) y = Math.round(target.getBoundingClientRect().bottom);
      }
      if (y === null || y < view.top || y > view.bottom) {
        line.style.opacity = "0";
        return;
      }
      const shown = line.style.opacity === "1";
      // Glides between gaps once it is up; the first placement is instant.
      line.style.transition = shown ? "" : "none";
      Object.assign(line.style, {
        left: `${view.left + 8}px`,
        width: `${view.width - 16}px`,
        transform: `translate3d(0, ${y}px, 0)`,
        opacity: "1",
      });
    };

    /** One step of the edge scroll, from the frame loop: a pointer resting at the edge sends no
     *  moves, and the list must keep going. */
    const edgeScroll = () => {
      const view = scroller.getBoundingClientRect();
      if (pointerY < view.top + EDGE_PX) scroller.scrollTop -= EDGE_STEP_PX;
      else if (pointerY > view.bottom - EDGE_PX) scroller.scrollTop += EDGE_STEP_PX;
    };

    const onFrame = () => {
      if (!started || escaped) {
        frame = 0;
        return;
      }
      frame = requestAnimationFrame(onFrame);
      edgeScroll();
      place();
    };

    /** Takes down everything the drag drew. Returns where the lifted copy was, for the settle. */
    const putDown = (): number | null => {
      const ghostTop = ghost ? ghost.getBoundingClientRect().top + 6 : null;
      ghost?.remove();
      line?.remove();
      ghost = null;
      line = null;
      block.removeAttribute(SECTION_DRAGGING_ATTR);
      document.body.classList.remove(DRAGGING_BODY_CLASS);
      return ghostTop;
    };

    const detach = () => {
      if (gesture.current === self) gesture.current = null;
      if (frame) cancelAnimationFrame(frame);
      frame = 0;
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onCancel);
      window.removeEventListener("keydown", onKey, true);
      if (document.body.hasPointerCapture?.(pointerId)) {
        document.body.releasePointerCapture(pointerId);
      }
    };

    // Abandons the gesture whole: an unmount, or a new press that never saw this one's release.
    const self = {
      end: () => {
        detach();
        if (started) putDown();
      },
    };
    gesture.current = self;

    /** The release clicks the header it lands on, which would fold the section just moved. */
    const swallowClick = () => {
      const stop = (clicked: MouseEvent) => {
        clicked.preventDefault();
        clicked.stopPropagation();
      };
      window.addEventListener("click", stop, { capture: true, once: true });
      window.setTimeout(() => window.removeEventListener("click", stop, { capture: true }), 0);
    };

    /** Slides the dropped section from where it was let go into its new slot. */
    const settle = (from: number) => {
      if (optionsRef.current.reducedMotion?.()) return;
      // Two frames: the drop re-renders the list, and the section is measured once it has moved.
      requestAnimationFrame(() =>
        requestAnimationFrame(() => {
          const moved = list.querySelector<HTMLElement>(
            `:scope > [${SECTION_ATTR}="${CSS.escape(key)}"]`,
          );
          const movedHeader = moved?.querySelector(HEADER_SELECTOR);
          if (!moved || !movedHeader) return;
          const to = (movedHeader.querySelector("button") ?? movedHeader).getBoundingClientRect().top;
          const delta = from - to;
          if (Math.abs(delta) < 2) return;
          moved.style.transition = "none";
          moved.style.transform = `translateY(${delta}px)`;
          moved.style.zIndex = "1";
          moved.getBoundingClientRect();
          moved.style.transition = `transform ${SETTLE_MS}ms cubic-bezier(0.2, 0.8, 0.2, 1)`;
          moved.style.transform = "";
          window.setTimeout(() => {
            moved.style.transition = "";
            moved.style.zIndex = "";
          }, SETTLE_MS + 20);
        }),
      );
    };

    function onMove(moved: PointerEvent) {
      if (moved.pointerId !== pointerId || escaped) return;
      pointerY = moved.clientY;
      if (!started) {
        if (
          Math.abs(moved.clientX - startX) < DRAG_THRESHOLD_PX &&
          Math.abs(moved.clientY - startY) < DRAG_THRESHOLD_PX
        ) {
          return;
        }
        started = true;
        scroller = scrollerOf(list) ?? list;
        document.body.classList.add(DRAGGING_BODY_CLASS);
        // Without capture a release outside the window never arrives and the drag sticks.
        try {
          document.body.setPointerCapture(pointerId);
        } catch {
          // Window listeners still carry the drag inside the window.
        }
        block.setAttribute(SECTION_DRAGGING_ATTR, "");
        lift();
        frame = requestAnimationFrame(onFrame);
      }
      moved.preventDefault();
      place();
    }

    function onUp(released: PointerEvent) {
      if (released.pointerId !== pointerId) return;
      detach();
      if (!started) return;
      swallowClick();
      if (escaped) return;
      pointerY = released.clientY;
      place();
      const dropped = landing;
      const from = putDown();
      if (!dropped) return;
      optionsRef.current.onDrop(key, dropped);
      if (from !== null) settle(from);
    }

    function onCancel(aborted: PointerEvent) {
      if (aborted.pointerId !== pointerId) return;
      detach();
      if (started) putDown();
    }

    function onKey(keyEvent: KeyboardEvent) {
      if (keyEvent.key !== "Escape" || escaped) return;
      if (!started) {
        // Nothing was lifted, so a click after this is still the header's.
        detach();
        return;
      }
      keyEvent.preventDefault();
      keyEvent.stopPropagation();
      // The button is still down: keep listening, so its release does not fold the section.
      escaped = true;
      putDown();
    }

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onCancel);
    window.addEventListener("keydown", onKey, true);
  }, []);
}
