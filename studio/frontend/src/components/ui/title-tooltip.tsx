// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { placeTitleTooltip } from "@/lib/title-tooltip-placement";
import { Z_LAYER } from "@/lib/z-layers";

const SHOW_DELAY_MS = 120;

type Shown = { text: string; anchor: DOMRect };

/** Disabled buttons are `pointer-events: none` (shadcn), so the hit lands on their parent: find the
 *  titled one under the pointer, which the native tooltip never showed at all. */
function disabledUnder(hit: Element | null, x: number, y: number): HTMLElement | null {
  if (!hit) return null;
  for (const el of hit.querySelectorAll<HTMLElement>("[title]:disabled, [title][aria-disabled='true']")) {
    const r = el.getBoundingClientRect();
    if (x >= r.left && x <= r.right && y >= r.top && y <= r.bottom) return el;
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
    let timer: number | undefined;

    const release = () => {
      window.clearTimeout(timer);
      if (current && lifted !== null && !current.hasAttribute("title")) {
        current.setAttribute("title", lifted);
      }
      current = null;
      lifted = null;
      setShown(null);
      setPos(null);
    };

    const track = (x: number, y: number) => {
      const hit = document.elementFromPoint(x, y);
      const el = hit?.closest<HTMLElement>("[title]") ?? disabledUnder(hit, x, y);
      if (el === current) return;
      release();
      if (!el || el.closest("svg") || !el.getAttribute("title")?.trim()) return;
      current = el;
      lifted = el.getAttribute("title") ?? "";
      el.removeAttribute("title");
      if (el.closest('[data-slot="tooltip-trigger"]') || hasOwnHoverLabel(el)) return;
      const text = lifted;
      timer = window.setTimeout(() => {
        if (current === el && el.isConnected) setShown({ text, anchor: el.getBoundingClientRect() });
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
      className="tooltip-compact pointer-events-none fixed w-fit max-w-xs break-words whitespace-pre-line"
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
