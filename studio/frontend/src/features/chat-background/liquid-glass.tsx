// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { LiquiGlass } from "@liqui-design/glass";
import { useEffect, useLayoutEffect, useState } from "react";
import { createPortal } from "react-dom";
import { useMaterialStore } from "./material";

/**
 * Liquid Glass surfaces drawn by @liqui-design/glass: an SVG displacement filter that bends whatever is
 * behind the rim (Chromium; WebKit and Firefox fall back to its frosted tier) and a normal-lit specular
 * rim. Rendered as a layer behind the host's content, so the host keeps its own markup.
 */
// Narrow bezels: the bend and the lit rim stay within a few pixels of the edge, like real glass.
const OPTICS = {
  composer: {
    liquid: { refraction: 30, bezel: 6, specular: 0.15, dispersion: 0 },
    "liquid-3d": {
      refraction: 60,
      bezel: 10,
      specular: 0.25,
      dispersion: 0.08,
    },
  },
  menu: {
    liquid: { refraction: 24, bezel: 6, specular: 0.3, dispersion: 0 },
    "liquid-3d": { refraction: 44, bezel: 9, specular: 0.45, dispersion: 0 },
  },
} as const;
// Frost sets the library's tint opacity (0.25 + 0.75 * frost) and adds frost * 14px of blur.
const FROST = { composer: 0.35, menu: 1 } as const;
const BASE_BLUR = { composer: 10, menu: 18 } as const;
// The library inflates the radius this much where corner-shape: squircle renders; we draw round corners.
const SQUIRCLE_SCALE =
  typeof CSS !== "undefined" && CSS.supports?.("corner-shape", "squircle")
    ? 1.4
    : 1;

export function isLiquidMaterial(
  material: string,
): material is "liquid" | "liquid-3d" {
  return material === "liquid" || material === "liquid-3d";
}

function strengthFactor(strength: number): number {
  return strength <= 50
    ? 0.2 + (0.8 * strength) / 50
    : 1 + (strength - 50) / 50;
}

export function LiquidGlassLayer({
  host,
  kind,
}: {
  host: HTMLElement;
  kind: "composer" | "menu";
}) {
  const material = useMaterialStore((s) => s.material);
  const strength = useMaterialStore((s) => s.strength);
  const [radius, setRadius] = useState(0);
  useLayoutEffect(() => {
    setRadius(
      Number.parseFloat(getComputedStyle(host).borderTopLeftRadius) || 0,
    );
  }, [host]);
  if (!isLiquidMaterial(material)) return null;
  const optics = OPTICS[kind][material];
  const frost = FROST[kind];
  const blur = Math.max(
    0,
    BASE_BLUR[kind] * (kind === "menu" ? 1 : strengthFactor(strength)) -
      frost * 14,
  );
  return (
    <LiquiGlass
      aria-hidden="true"
      className="unsloth-liquid-layer"
      data-liquid-layer={kind}
      material="auto"
      radius={radius / SQUIRCLE_SCALE}
      refraction={optics.refraction}
      bezel={optics.bezel}
      specular={optics.specular}
      dispersion={optics.dispersion}
      frost={frost}
      blur={Math.round(blur * 10) / 10}
    />
  );
}

const MENUS =
  '[data-slot$="menu-content"], [data-slot$="menu-sub-content"], [data-slot="select-content"], [data-slot="combobox-content"], [data-slot="popover-content"]';

/** Liquid Glass layers for open menus, which Radix portals straight into <body>. */
export function GlassMenuLayers({ active }: { active: boolean }) {
  const [hosts, setHosts] = useState<HTMLElement[]>([]);
  useEffect(() => {
    if (!active) return;
    let frame = 0;
    const scan = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const found = [...document.querySelectorAll<HTMLElement>(MENUS)];
        setHosts((prev) =>
          prev.length === found.length && prev.every((h, i) => h === found[i])
            ? prev
            : found,
        );
      });
    };
    const observer = new MutationObserver(scan);
    observer.observe(document.body, { childList: true });
    scan();
    return () => {
      observer.disconnect();
      cancelAnimationFrame(frame);
      setHosts([]);
    };
  }, [active]);
  if (!active) return null;
  return hosts.map((host, i) =>
    createPortal(
      <LiquidGlassLayer host={host} kind="menu" />,
      host,
      `glass-menu-${i}`,
    ),
  );
}
