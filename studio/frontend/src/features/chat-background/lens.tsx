// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useAppliedMaterial, useGlassStrength, useMaterialStore } from "./material";
import { GLASS_MOVING_ATTR, GLASS_SETTLING_ATTR } from "./glass-motion";
import { LENS_CLEAR_HOSTS, LENS_HOSTS, LENS_SCALE, lensDrawn } from "./model";

const HOST_FLAG = "data-glass-lens-host";

// The style drawing a surface's glass: its own, or its ::before (the Desktop sidebar frame, layered panels).
function glassLayer(el: HTMLElement): CSSStyleDeclaration | null {
  for (const pseudo of [null, "::before"]) {
    const style = getComputedStyle(el, pseudo);
    const filter = style.backdropFilter || style.getPropertyValue("-webkit-backdrop-filter");
    if (filter && filter !== "none") return style;
  }
  return null;
}

// React rewrites a text-only element with textContent, which would delete a portaled lens.
const textOnly = (el: HTMLElement) =>
  [...el.childNodes].every((n) => n.nodeType === Node.TEXT_NODE || (n as Element).classList?.contains("glass-lens"));

// Beside the pinned strip the panel's glass is the strip's, so its lens reads the strip.
const STRIP_PANEL =
  'html[data-glass][data-sidebar-glass-frame][data-chat-wallpaper-workspace-ready]:not(.webkit-legacy-compositor) aside[data-slot="chat-settings-panel"]';
const glassSource = (host: HTMLElement): HTMLElement =>
  (host.matches(STRIP_PANEL) &&
    document.querySelector<HTMLElement>('[data-slot="sidebar-container"][data-windows-glass-frame]')) ||
  host;

// Only surfaces drawing glass right now: Glass menus off, an opaque sheet or a split card has no backdrop filter.
// The composer keeps its lens without one, as on main (Transparent forces backdrop-filter off).
function drawsGlass(el: HTMLElement): boolean {
  const { width, height } = el.getBoundingClientRect();
  if (width < 8 || height < 8 || textOnly(el)) return false;
  if (el.matches(STRIP_PANEL)) return true;
  return el.matches(LENS_CLEAR_HOSTS) || el.hasAttribute(GLASS_MOVING_ATTR) || glassLayer(el) !== null;
}

const painted = (s: CSSStyleDeclaration) =>
  (s.backgroundColor !== "rgba(0, 0, 0, 0)" && s.backgroundColor !== "transparent") || s.backgroundImage !== "none";

// One lens per sidebar column: legacy WebKitGTK masks the frame's ::before (and its lens) to the caption and gives the
// sidebar its own glass, elsewhere the frame holds it. Keyed on the compositor, not the motion flag both carry while moving.
const FRAME = '[data-slot="sidebar-container"][data-windows-glass-frame]';
const FRAME_SIDEBAR = '[data-windows-glass-frame] > [data-slot="sidebar-inner"]';
const legacyCompositor = () => document.documentElement.classList.contains("webkit-legacy-compositor");
function oneSidebarLens(found: HTMLElement[]): HTMLElement[] {
  const legacy = legacyCompositor();
  const inner = found.filter((h) => h.matches(FRAME_SIDEBAR) && found.includes(h.parentElement as HTMLElement));
  const drop = new Set<Element | null>(legacy ? [] : inner);
  // The frame's lens spans the whole strip (index.css), docked Run settings included: a second lens there seams.
  const stripLens = !legacy && found.some((h) => h.matches(FRAME));
  return found.filter((h) => !drop.has(h) && !(stripLens && h.matches(STRIP_PANEL)));
}

// Legacy WebKitGTK draws the strip as three lenses (caption, sidebar, docked Run settings). Scaled about one origin,
// the strip's, their copies line up where they meet, so the three read as one sheet.
const LEGACY_STRIP_HOSTS = `${FRAME}, ${FRAME_SIDEBAR}, html[data-sidebar-glass-frame] aside[data-slot="chat-settings-panel"]`;
type Box = { left: number; top: number; width: number; height: number };
function legacyStripBox(host: HTMLElement): Box | null {
  if (!legacyCompositor() || !host.matches(LEGACY_STRIP_HOSTS)) return null;
  const frame = host.matches(FRAME) ? host : document.querySelector<HTMLElement>(FRAME);
  if (!frame) return null;
  // The frame's lens box: inset 0 auto 0 0, 100vw wide (index.css).
  const r = frame.getBoundingClientRect();
  return { left: r.left, top: r.top, width: window.innerWidth, height: r.height };
}

const PREVIEW_HOSTS = ".wallpaper-preview-composer";

// Keyed per host, not by index: a host joining earlier in the list would otherwise remount every later lens.
const hostKeys = new WeakMap<HTMLElement, number>();
let nextHostKey = 0;
function hostKey(host: HTMLElement): number {
  let key = hostKeys.get(host);
  if (key === undefined) hostKeys.set(host, (key = ++nextHostKey));
  return key;
}

/** A wallpaper copy in every glass surface, since backdrop-filter cannot scale and SVG backdrop filters are Chromium only (not WebKit).
 * `preview` keeps it to a settings preview's own sample composers, over that preview's wallpaper. */
export function GlassLenses({
  wallpaper,
  preview = false,
}: {
  wallpaper: HTMLElement | null;
  preview?: boolean;
}) {
  const material = useAppliedMaterial();
  const lens = useMaterialStore((s) => s.lens);
  const menus = useMaterialStore((s) => s.menus);
  const clear = useGlassStrength() === 0;
  const [hosts, setHosts] = useState<HTMLElement[]>([]);
  const workspace = wallpaper?.parentElement ?? null;
  const on = Boolean(workspace) && lensDrawn(material, lens);
  useEffect(() => {
    if (!workspace || !on) return;
    let frame = 0;
    const scan = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const found = oneSidebarLens([
          ...(preview ? workspace : document).querySelectorAll<HTMLElement>(preview ? PREVIEW_HOSTS : LENS_HOSTS),
        ].filter((h) => drawsGlass(h) && (!clear || h.matches(LENS_CLEAR_HOSTS))));
        setHosts((prev) =>
          prev.length === found.length && prev.every((h, i) => h === found[i])
            ? prev
            : found,
        );
      });
    };
    // Menus and sheets portal to <body>, and app-level toasts mount anywhere under the root, not only in the workspace.
    const observer = new MutationObserver(scan);
    observer.observe(preview ? workspace : document.body, { childList: true, subtree: true });
    // A host only moving (glass-motion) counts as glass; once the flag clears, one with no glass at rest drops out.
    const motion = new MutationObserver(scan);
    motion.observe(preview ? workspace : document.body, { attributes: true, subtree: true, attributeFilter: [GLASS_MOVING_ATTR] });
    // A panel opening or a menu's open animation settling changes which hosts draw glass.
    document.addEventListener("transitionend", scan, true);
    document.addEventListener("animationend", scan, true);
    scan();
    return () => {
      observer.disconnect();
      motion.disconnect();
      document.removeEventListener("transitionend", scan, true);
      document.removeEventListener("animationend", scan, true);
      cancelAnimationFrame(frame);
      setHosts([]);
    };
  }, [workspace, on, material, menus, preview, clear]);
  if (!wallpaper || !on) return null;
  return hosts.map((host) =>
    createPortal(
      <Lens host={host} wallpaper={wallpaper} />,
      host,
      `glass-lens-${hostKey(host)}`,
    ),
  );
}

function Lens({
  host,
  wallpaper,
}: {
  host: HTMLElement;
  wallpaper: HTMLElement;
}) {
  const lens = useMaterialStore((s) => s.lens);
  const scale = LENS_SCALE[lens];
  const root = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const el = root.current;
    if (!el) return;
    const copies = [...el.querySelectorAll<HTMLElement>("[data-lens-copy]")];
    const fillCopy = el.querySelector<HTMLElement>(".glass-lens-fill [data-lens-copy]");
    const wash = el.querySelector<HTMLElement>("[data-lens-wash]");
    const tint = el.querySelector<HTMLElement>("[data-lens-tint]");
    const workspace = wallpaper.parentElement;
    let last = "";
    const place = () => {
      // The translated lens counts as scrollable overflow: drop it first so a host whose content shrank can clamp its scroll.
      if (el.style.translate) el.style.translate = "";
      const sx = host.scrollLeft;
      const sy = host.scrollTop;
      el.style.translate = sx || sy ? `${sx}px ${sy}px` : "";
      const w = wallpaper.getBoundingClientRect();
      // The frame's lens is the strip's width, not the sidebar's.
      const h = (host.matches(FRAME) ? el : host).getBoundingClientRect();
      const o = legacyStripBox(host) ?? h;
      const opacity = getComputedStyle(wallpaper).opacity;
      const source = glassSource(host);
      const glass = glassLayer(source);
      const own = getComputedStyle(source);
      // Settings and layered panels blur on ::before but keep their fill on the host.
      const fill = !glass || painted(own) ? own : glass;
      // The copy takes the surface's own blur: menus and dialogs blur more than the shared --glass-blur.
      const blur = glass ? glass.backdropFilter || glass.getPropertyValue("-webkit-backdrop-filter") : "";
      const key = [
        w.left,
        w.top,
        w.width,
        w.height,
        h.left,
        h.top,
        h.width,
        h.height,
        o.left,
        o.top,
        o.width,
        o.height,
        opacity,
        fill.backgroundColor,
        fill.backgroundImage,
        blur,
      ].join();
      if (key === last) return;
      last = key;
      for (const copy of copies) {
        copy.style.left = `${w.left - h.left}px`;
        copy.style.top = `${w.top - h.top}px`;
        copy.style.width = `${w.width}px`;
        copy.style.height = `${w.height}px`;
        copy.style.opacity = opacity;
        copy.style.transformOrigin = `${o.left - w.left + o.width / 2}px ${o.top - w.top + o.height / 2}px`;
        copy.style.transform = `scale(${scale})`;
      }
      // A moving host drops its backdrop filter (glass-motion.ts); the copy drops its blur with it instead of the CSS fallback.
      if (fillCopy) fillCopy.style.filter = blur || (host.hasAttribute(GLASS_MOVING_ATTR) ? "none" : "");
      if (wash) {
        const fills: string[] = [];
        for (
          let node = host.parentElement;
          node && node !== workspace && node !== document.body;
          node = node.parentElement
        ) {
          const color = getComputedStyle(node).backgroundColor;
          if (color !== "rgba(0, 0, 0, 0)" && color !== "transparent")
            fills.push(`linear-gradient(${color}, ${color})`);
        }
        wash.style.backgroundImage = fills.join(", ");
      }
      if (tint) {
        tint.style.backgroundColor = fill.backgroundColor;
        tint.style.backgroundImage = fill.backgroundImage;
      }
    };
    const copy = () => {
      for (const target of copies)
        target.replaceChildren(wallpaper.cloneNode(true));
      last = "";
      place();
    };
    let frame = 0;
    const recopy = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(copy);
    };
    copy();
    // The lens paints at z-index -1, so the host becomes its stacking context (index.css).
    host.setAttribute(HOST_FLAG, getComputedStyle(host).position === "static" ? "static" : "");
    host.addEventListener("animationend", place);
    host.addEventListener("scroll", place, { passive: true });
    const resize = new ResizeObserver(place);
    resize.observe(host);
    resize.observe(wallpaper);
    const motion = new MutationObserver(place);
    // Dragged panels move by a transform on the host or a wrapper, which no resize or scroll event reports.
    // One observe() per target: a second call replaces its options.
    motion.observe(host, { attributes: true, attributeFilter: [GLASS_MOVING_ATTR, GLASS_SETTLING_ATTR, "style"] });
    for (let node = host.parentElement; node && node !== document.body; node = node.parentElement)
      motion.observe(node, { attributes: true, attributeFilter: ["style"] });
    const mutations = new MutationObserver(recopy);
    mutations.observe(wallpaper, {
      childList: true,
      subtree: true,
      attributes: true,
    });
    const poll = window.setInterval(place, 300);
    window.addEventListener("resize", place);
    return () => {
      host.removeAttribute(HOST_FLAG);
      host.removeEventListener("animationend", place);
      host.removeEventListener("scroll", place);
      resize.disconnect();
      motion.disconnect();
      mutations.disconnect();
      window.clearInterval(poll);
      window.removeEventListener("resize", place);
      cancelAnimationFrame(frame);
    };
  }, [host, wallpaper, scale]);
  return (
    <div ref={root} aria-hidden="true" className="glass-lens" data-lens={lens}>
      <div className="glass-lens-fill">
        <div data-lens-copy="" className="glass-lens-copy" />
        <div data-lens-wash="" className="glass-lens-layer" />
        <div data-lens-tint="" className="glass-lens-layer" />
      </div>
      <div className="glass-lens-rim">
        <div data-lens-copy="" className="glass-lens-copy" />
      </div>
    </div>
  );
}
