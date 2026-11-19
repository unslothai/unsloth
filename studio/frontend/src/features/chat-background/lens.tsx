// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useAppliedMaterial, useMaterialStore } from "./material";
import { LENS_HOSTS, LENS_SCALE, lensDrawn } from "./model";

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

// Only surfaces drawing glass right now: Glass menus off, an opaque sheet or a split card has no backdrop filter.
function drawsGlass(el: HTMLElement): boolean {
  const { width, height } = el.getBoundingClientRect();
  if (width < 8 || height < 8) return false;
  return el.hasAttribute("data-glass-moving") || glassLayer(el) !== null;
}

/** A wallpaper copy in every glass surface, since backdrop-filter cannot scale and SVG backdrop filters are Chromium only (not WebKit). */
export function GlassLenses({ wallpaper }: { wallpaper: HTMLElement | null }) {
  const material = useAppliedMaterial();
  const lens = useMaterialStore((s) => s.lens);
  const menus = useMaterialStore((s) => s.menus);
  const [hosts, setHosts] = useState<HTMLElement[]>([]);
  const workspace = wallpaper?.parentElement ?? null;
  const on = Boolean(workspace) && lensDrawn(material, lens);
  useEffect(() => {
    if (!workspace || !on) return;
    let frame = 0;
    const scan = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        // Menus, popovers and sheets are portaled to <body>, outside the workspace.
        const found = [...document.querySelectorAll<HTMLElement>(LENS_HOSTS)].filter(drawsGlass);
        setHosts((prev) =>
          prev.length === found.length && prev.every((h, i) => h === found[i])
            ? prev
            : found,
        );
      });
    };
    const observer = new MutationObserver(scan);
    observer.observe(workspace, { childList: true, subtree: true });
    observer.observe(document.body, { childList: true });
    // A panel opening or a menu's open animation settling changes which hosts draw glass.
    document.addEventListener("transitionend", scan, true);
    document.addEventListener("animationend", scan, true);
    scan();
    return () => {
      observer.disconnect();
      document.removeEventListener("transitionend", scan, true);
      document.removeEventListener("animationend", scan, true);
      cancelAnimationFrame(frame);
      setHosts([]);
    };
  }, [workspace, on, menus]);
  if (!wallpaper || !on) return null;
  return hosts.map((host, i) =>
    createPortal(
      <Lens host={host} wallpaper={wallpaper} />,
      host,
      `glass-lens-${i}`,
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
    const wash = el.querySelector<HTMLElement>("[data-lens-wash]");
    const tint = el.querySelector<HTMLElement>("[data-lens-tint]");
    const workspace = wallpaper.parentElement;
    let last = "";
    const place = () => {
      const w = wallpaper.getBoundingClientRect();
      const h = host.getBoundingClientRect();
      const opacity = getComputedStyle(wallpaper).opacity;
      const fill = glassLayer(host) ?? getComputedStyle(host);
      const key = [
        w.left,
        w.top,
        w.width,
        w.height,
        h.left,
        h.top,
        h.width,
        h.height,
        opacity,
        fill.backgroundColor,
        fill.backgroundImage,
      ].join();
      if (key === last) return;
      last = key;
      for (const copy of copies) {
        copy.style.left = `${w.left - h.left}px`;
        copy.style.top = `${w.top - h.top}px`;
        copy.style.width = `${w.width}px`;
        copy.style.height = `${w.height}px`;
        copy.style.opacity = opacity;
        copy.style.transformOrigin = `${h.left - w.left + h.width / 2}px ${h.top - w.top + h.height / 2}px`;
        copy.style.transform = `scale(${scale})`;
      }
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
    const resize = new ResizeObserver(place);
    resize.observe(host);
    resize.observe(wallpaper);
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
      resize.disconnect();
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
