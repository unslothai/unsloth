// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Section header drag: DOM-only updates during the move (re-rendering the sidebar lagged);
// pointer events because the desktop webview never forwards the HTML5 drag API.

import { useCallback, useEffect, useRef } from "react";

import type { DropEdge } from "../lib/sidebar-drag.ts";
import { placeIdAt } from "../stores/sidebar-organization-store.ts";
import {
  DRAG_THRESHOLD_PX,
  EDGE_PX,
  EDGE_STEP_PX,
  dragLayer,
  markDragging,
  scrollerOf,
  sidebarOf,
} from "./use-sidebar-drag.ts";

export const SECTION_ATTR = "data-sidebar-section";
export const SECTION_DRAGGING_ATTR = "data-section-dragging";
const HEADER_SELECTOR = '[data-sidebar="group-label"]';
const HEADER_ACTION_SELECTOR = ".sidebar-header-action";
const SETTLE_MS = 180;

export type SectionLanding = { target: string; edge: DropEdge };

export type SectionBlock = { key: string; top: number; bottom: number };

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

/** Whether landing `key` changes the drawn order; a no-op landing draws no line. */
export function landingMoves(
  drawn: readonly string[],
  key: string,
  landing: SectionLanding,
): boolean {
  const next = placeIdAt([...drawn], key, landing.target, landing.edge);
  return next.some((id, index) => id !== drawn[index]);
}

function lineYAbove(header: Element): number {
  const box = header.getBoundingClientRect();
  const text = (header.querySelector("button") ?? header).getBoundingClientRect();
  return Math.round(box.top + (text.top - box.top) / 2);
}

function headedSiblingAfter(element: Element | undefined): Element | null {
  for (let next = element?.nextElementSibling; next; next = next.nextElementSibling) {
    if (next instanceof HTMLElement && next.offsetHeight > 0 && next.querySelector(HEADER_SELECTOR)) {
      return next;
    }
  }
  return null;
}

export function sectionKeyLanding(
  event: React.KeyboardEvent<HTMLElement>,
  key: string,
): SectionLanding | null {
  if (!event.altKey || (event.key !== "ArrowUp" && event.key !== "ArrowDown")) return null;
  const pressed = event.target as Element;
  const header = pressed.closest(HEADER_SELECTOR);
  if (!header || !event.currentTarget.contains(header)) return null;
  if (pressed.closest(HEADER_ACTION_SELECTOR)) return null;
  const list = event.currentTarget.parentElement;
  if (!list) return null;
  const drawn = [...list.querySelectorAll<HTMLElement>(`:scope > [${SECTION_ATTR}]`)]
    .filter((element) => element.offsetHeight > 0)
    .map((element) => element.getAttribute(SECTION_ATTR));
  const up = event.key === "ArrowUp";
  const at = drawn.indexOf(key);
  const neighbour = at === -1 ? undefined : drawn[at + (up ? -1 : 1)];
  return neighbour ? { target: neighbour, edge: up ? "top" : "bottom" } : null;
}

export interface UseSectionDragOptions {
  onDrop: (key: string, landing: SectionLanding) => void;
  reducedMotion?: () => boolean;
}

export function useSectionDrag(
  options: UseSectionDragOptions,
): (event: React.PointerEvent<HTMLElement>, key: string) => void {
  const optionsRef = useRef(options);
  useEffect(() => {
    optionsRef.current = options;
  });
  const gesture = useRef<{ end: () => void } | null>(null);
  useEffect(() => () => gesture.current?.end(), []);

  return useCallback((event: React.PointerEvent<HTMLElement>, key: string) => {
    if (event.button !== 0 || event.pointerType === "touch" || !event.isPrimary) return;
    const pressed = event.target as Element;
    const header = pressed.closest(HEADER_SELECTOR);
    if (!header || !event.currentTarget.contains(header)) return;
    if (pressed.closest(HEADER_ACTION_SELECTOR)) return;
    const block = event.currentTarget;
    const parent = block.parentElement;
    if (!parent) return;
    // Typed here, not narrowed: the hoisted handlers below would lose the narrowing.
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
    let ghostTop = 0;
    let line: HTMLElement | null = null;
    const text = header.querySelector("button") ?? header;
    const grab = startY - text.getBoundingClientRect().top;

    const drawnBlocks = () =>
      [...list.querySelectorAll<HTMLElement>(`:scope > [${SECTION_ATTR}]`)].filter(
        (element) => element.offsetHeight > 0,
      );

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
      // Lifted where the header is, so a frame before the first transform is not at the top.
      ghostTop = textRect.top - 6;
      Object.assign(ghost.style, {
        top: `${ghostTop}px`,
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
      dragLayer().append(ghost, line);
    };

    const place = () => {
      const view = scroller.getBoundingClientRect();
      if (ghost) {
        const height = ghost.offsetHeight;
        const top = Math.min(
          Math.max(pointerY - grab - 6, view.top),
          view.bottom - height,
        );
        ghost.style.transform = `translate3d(0, ${Math.round(top - ghostTop)}px, 0)`;
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
      // Shown and placed in one change, so it is never drawn a frame early.
      line.style.transition = shown ? "" : "none";
      Object.assign(line.style, {
        left: `${view.left + 8}px`,
        width: `${view.width - 16}px`,
        top: `${y - 0.75}px`,
        opacity: "1",
      });
    };

    /** Edge-scroll step from the frame loop: a resting pointer sends no moves. */
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

    const putDown = (): number | null => {
      const ghostTop = ghost ? ghost.getBoundingClientRect().top + 6 : null;
      ghost?.remove();
      line?.remove();
      ghost = null;
      line = null;
      block.removeAttribute(SECTION_DRAGGING_ATTR);
      markDragging(null, false);
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

    const settle = (from: number) => {
      if (optionsRef.current.reducedMotion?.()) return;
      // Two frames: the drop re-renders the list before the section can be measured.
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
        markDragging(sidebarOf(list), true);
        // Without capture a release outside the window never arrives and the drag sticks.
        try {
          document.body.setPointerCapture(pointerId);
        } catch {
          // Window listeners still carry the drag inside the window.
        }
        block.setAttribute(SECTION_DRAGGING_ATTR, "");
        lift();
        place();
        frame = requestAnimationFrame(onFrame);
      }
      // Place once a frame: moves outpace frames, and each placement reads every box.
      moved.preventDefault();
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
        detach();
        return;
      }
      keyEvent.preventDefault();
      keyEvent.stopPropagation();
      // The button is still down: keep listening so its release does not fold the section.
      escaped = true;
      putDown();
    }

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onCancel);
    window.addEventListener("keydown", onKey, true);
  }, []);
}
