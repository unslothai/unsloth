// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import {
  type TitleStandIn,
  placeTitleTooltip,
  resolveTitleTarget,
  titleStandIn,
} from "@/lib/title-tooltip-placement";
import { Z_LAYER } from "@/lib/z-layers";

const SHOW_DELAY_MS = 120;

type Shown = { text: string; anchor: DOMRect };

/** Disabled buttons are `pointer-events: none` (shadcn), so the hit lands on their parent: find the
 *  titled one under the pointer, which the native tooltip never showed at all. */
function pointIn(el: Element, x: number, y: number): boolean {
  const r = el.getBoundingClientRect();
  return x >= r.left && x <= r.right && y >= r.top && y <= r.bottom;
}

function disabledUnder(hit: Element | null, x: number, y: number): HTMLElement | null {
  if (!hit) return null;
  for (const el of hit.querySelectorAll<HTMLElement>("[title]:disabled, [title][aria-disabled='true']")) {
    if (pointIn(el, x, y)) return el;
  }
  return null;
}

/** A collapsed composer pill already shows its own label tooltip from CSS. */
function hasOwnHoverLabel(el: Element): boolean {
  if (!el.hasAttribute("data-pill-label")) return false;
  const after = getComputedStyle(el, "::after").content;
  return Boolean(after) && after !== "none" && after !== "normal";
}

/**
 * Native `title` tooltips are drawn by the OS and ignore the theme. Hundreds of controls set one, so
 * rather than wrap each, this lifts the attribute off whatever the pointer is over (which stops the
 * native one) and draws the same text with `.tooltip-compact`. The attribute goes back on leave, so
 * the DOM, accessible names and tests see it as written. Radix tooltip triggers keep their own.
 */
export function TitleTooltipLayer() {
  const [shown, setShown] = useState<Shown | null>(null);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);
  const tipRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let current: HTMLElement | null = null;
    let lifted: string | null = null;
    let standIn: TitleStandIn = null;
    let showing = false;
    let timer: number | undefined;

    const lift = (el: HTMLElement) => {
      lifted = el.getAttribute("title") ?? "";
      el.removeAttribute("title");
      if (standIn) el.setAttribute(standIn, lifted);
    };
    // A title React rewrites while hovered (a download's rate and ETA) is lifted again, so the native
    // tooltip stays off and the drawn one stays current.
    const retitled = new MutationObserver(() => {
      if (!current?.hasAttribute("title")) return;
      lift(current);
      if (showing) setShown({ text: lifted ?? "", anchor: current.getBoundingClientRect() });
    });

    const release = () => {
      window.clearTimeout(timer);
      retitled.disconnect();
      if (current && lifted !== null && !current.hasAttribute("title")) {
        current.setAttribute("title", lifted);
      }
      if (current && standIn) current.removeAttribute(standIn);
      current = null;
      lifted = null;
      standIn = null;
      showing = false;
      setShown(null);
      setPos(null);
    };

    const track = (x: number, y: number) => {
      const hit = document.elementFromPoint(x, y);
      const found = hit?.closest<HTMLElement>("[title]") ?? disabledUnder(hit, x, y);
      const el = resolveTitleTarget(
        current,
        found,
        current !== null && (current.contains(hit) || pointIn(current, x, y)),
      );
      if (el === current) return;
      release();
      if (!el || el.closest("svg") || !el.getAttribute("title")?.trim()) return;
      current = el;
      // The title may be the control's only name (an icon button): keep it for assistive tech.
      standIn = titleStandIn({
        labelled: el.hasAttribute("aria-label") || el.hasAttribute("aria-labelledby"),
        named: Boolean(el.textContent?.trim()) || Boolean(el.getAttribute("alt")?.trim()),
      });
      if (standIn && el.hasAttribute(standIn)) standIn = null;
      lift(el);
      retitled.observe(el, { attributes: true, attributeFilter: ["title"] });
      if (el.closest('[data-slot="tooltip-trigger"]') || hasOwnHoverLabel(el)) return;
      timer = window.setTimeout(() => {
        if (current !== el || !el.isConnected) return;
        showing = true;
        setShown({ text: lifted ?? "", anchor: el.getBoundingClientRect() });
      }, SHOW_DELAY_MS);
    };

    const onMove = (event: PointerEvent) => {
      if (event.pointerType === "touch") return;
      track(event.clientX, event.clientY);
    };
    const onLeaveWindow = (event: PointerEvent) => {
      if (!event.relatedTarget) release();
    };
    document.addEventListener("pointermove", onMove, { capture: true, passive: true });
    document.addEventListener("pointerout", onLeaveWindow, true);
    document.addEventListener("pointerdown", release, true);
    document.addEventListener("keydown", release, true);
    window.addEventListener("scroll", release, true);
    window.addEventListener("blur", release);
    return () => {
      release();
      document.removeEventListener("pointermove", onMove, true);
      document.removeEventListener("pointerout", onLeaveWindow, true);
      document.removeEventListener("pointerdown", release, true);
      document.removeEventListener("keydown", release, true);
      window.removeEventListener("scroll", release, true);
      window.removeEventListener("blur", release);
    };
  }, []);

  useEffect(() => {
    const tip = tipRef.current;
    if (!shown || !tip) return;
    setPos(
      placeTitleTooltip(
        shown.anchor,
        { width: tip.offsetWidth, height: tip.offsetHeight },
        { width: window.innerWidth, height: window.innerHeight },
      ),
    );
  }, [shown]);

  if (!shown) return null;
  return createPortal(
    <div
      ref={tipRef}
      role="tooltip"
      data-title-tooltip=""
      className="tooltip-compact pointer-events-none fixed line-clamp-[12] w-fit max-w-xs break-words whitespace-pre-line"
      style={{
        left: pos?.left ?? 0,
        top: pos?.top ?? 0,
        visibility: pos ? "visible" : "hidden",
        zIndex: Z_LAYER.TOOLTIP,
      }}
    >
      {shown.text}
    </div>,
    document.body,
  );
}
