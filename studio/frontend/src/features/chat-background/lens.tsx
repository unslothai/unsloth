// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { LiquidGlassLayer, isLiquidMaterial } from "./liquid-glass";
import { useMaterialStore } from "./material";
import { LENS_SCALE } from "./model";

const COMPOSER = ".unsloth-composer-surface";

/** A wallpaper copy, since backdrop-filter cannot scale and SVG backdrop filters are Chromium only (not WebKit). */
export function ComposerGlassLenses({
  wallpaper,
}: {
  wallpaper: HTMLElement | null;
}) {
  const material = useMaterialStore((s) => s.material);
  const lens = useMaterialStore((s) => s.lens);
  const [hosts, setHosts] = useState<HTMLElement[]>([]);
  const workspace = wallpaper?.parentElement ?? null;
  const on = Boolean(workspace) && material !== "solid";
  const liquid = isLiquidMaterial(material) && lens === "off";
  useEffect(() => {
    if (!workspace || !on) return;
    let frame = 0;
    const scan = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const found = [...workspace.querySelectorAll<HTMLElement>(COMPOSER)];
        setHosts((prev) =>
          prev.length === found.length && prev.every((h, i) => h === found[i])
            ? prev
            : found,
        );
      });
    };
    const observer = new MutationObserver(scan);
    observer.observe(workspace, { childList: true, subtree: true });
    scan();
    return () => {
      observer.disconnect();
      cancelAnimationFrame(frame);
      setHosts([]);
    };
  }, [workspace, on]);
  if (!wallpaper || !on) return null;
  return hosts.map((host, i) =>
    createPortal(
      <>
        {liquid && <LiquidGlassLayer host={host} kind="composer" />}
        <Lens host={host} wallpaper={wallpaper} />
      </>,
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
      const fill = getComputedStyle(host);
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
          node && node !== workspace;
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
      resize.disconnect();
      mutations.disconnect();
      window.clearInterval(poll);
      window.removeEventListener("resize", place);
      cancelAnimationFrame(frame);
    };
  }, [host, wallpaper, scale]);
  return (
    <div ref={root} aria-hidden="true" className="glass-lens" data-lens={lens}>
      {lens !== "off" && (
        <div className="glass-lens-fill">
          <div data-lens-copy="" className="glass-lens-copy" />
          <div data-lens-wash="" className="glass-lens-layer" />
          <div data-lens-tint="" className="glass-lens-layer" />
        </div>
      )}
      <div className="glass-lens-rim">
        <div data-lens-copy="" className="glass-lens-copy" />
      </div>
    </div>
  );
}
