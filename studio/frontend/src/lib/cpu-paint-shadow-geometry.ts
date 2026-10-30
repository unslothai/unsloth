// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Parsing and 9-slice sizing for cpu-paint-shadows.ts. No DOM, so node tests import it.

export interface ShadowLayer {
  color: string;
  x: number;
  y: number;
  blur: number;
  spread: number;
  inset: boolean;
}

export interface CornerRadii {
  tl: number;
  tr: number;
  br: number;
  bl: number;
}

/** One axis of the 9-slice, in device pixels unless named in CSS px. */
export interface SliceAxis {
  /** border-image-slice for both ends of this axis, image px. */
  slice: number;
  /** border-image-width for both ends, CSS px. */
  width: number;
}

export interface NineSliceGeometry {
  /** border-image-outset, CSS px: how far the shadow reaches past the border box. */
  outset: number;
  /** Shadow margin in device px; the box edge sits this far into the image. */
  outsetPx: number;
  /** Top and bottom slices. */
  y: SliceAxis;
  /** Left and right slices. */
  x: SliceAxis;
  /** Device px of the stretched middle strip. */
  middle: number;
  /** Image size, device px. */
  imageWidth: number;
  imageHeight: number;
  /** The larger scratch canvas the shadow is drawn on, device px. */
  scratchWidth: number;
  scratchHeight: number;
  /** How far the scratch's corners are spread apart, device px. */
  gap: number;
}

/** Split on separators outside parentheses. */
function splitTopLevel(
  value: string,
  isSeparator: (ch: string) => boolean,
): string[] {
  const parts: string[] = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i < value.length; i++) {
    const ch = value[i];
    if (ch === "(") depth++;
    else if (ch === ")") depth = Math.max(0, depth - 1);
    else if (depth === 0 && isSeparator(ch)) {
      parts.push(value.slice(start, i));
      start = i + 1;
    }
  }
  parts.push(value.slice(start));
  return parts.map((part) => part.trim()).filter(Boolean);
}

const LENGTH = /^(-?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)(px)?$/i;

/**
 * Parse a box-shadow in computed ("rgba(0, 0, 0, 0.16) 0px 2px 8px -2px") or
 * specified form. Null for anything else (another unit, a var()), so the
 * caller keeps the CSS shadow.
 */
export function parseShadowLayers(value: string): ShadowLayer[] | null {
  const text = value.trim();
  if (!text || text === "none") return [];
  const layers: ShadowLayer[] = [];
  for (const raw of splitTopLevel(text, (ch) => ch === ",")) {
    const lengths: number[] = [];
    let color = "";
    let inset = false;
    for (const token of splitTopLevel(raw, (ch) => /\s/.test(ch))) {
      const length = LENGTH.exec(token);
      if (length) {
        // A bare number is only a length when it is 0.
        if (!length[2] && Number(length[1]) !== 0) return null;
        lengths.push(Number(length[1]));
      } else if (token.toLowerCase() === "inset") {
        inset = true;
      } else if (!color && !/^var\(/i.test(token)) {
        color = token;
      } else {
        return null;
      }
    }
    if (lengths.length < 2 || lengths.length > 4) return null;
    layers.push({
      color: color || "currentcolor",
      x: lengths[0],
      y: lengths[1],
      blur: Math.max(0, lengths[2] ?? 0),
      spread: lengths[3] ?? 0,
      inset,
    });
  }
  return layers;
}

/**
 * The layers drawn into the image. Only blurred layers are slow, but once one
 * is drawn every outer layer goes with it: border-image paints above
 * box-shadow, so a ring left behind would end up under the blur.
 */
export function imageLayers(layers: ShadowLayer[]): ShadowLayer[] {
  if (!layers.some((layer) => !layer.inset && layer.blur > 0)) return [];
  return layers.filter((layer) => !layer.inset);
}

function formatLayer(layer: ShadowLayer): string {
  const inset = layer.inset ? "inset " : "";
  return `${inset}${layer.x}px ${layer.y}px ${layer.blur}px ${layer.spread}px ${layer.color}`;
}

/** What stays a real box-shadow: the layers not drawn into the image. */
export function remainingShadow(layers: ShadowLayer[]): string {
  const drawn = imageLayers(layers);
  const kept = layers.filter((layer) => !drawn.includes(layer));
  return kept.length ? kept.map(formatLayer).join(", ") : "0 0 #0000";
}

/** css-backgrounds-3: how a corner radius follows the spread distance. */
export function spreadRadius(radius: number, spread: number): number {
  if (radius <= 0) return 0;
  if (spread < 0) return Math.max(0, radius + spread);
  if (spread > 0 && radius < spread) {
    return radius + spread * (1 + (radius / spread - 1) ** 3);
  }
  return radius + spread;
}

/**
 * Sizes for the 9-slice at `scale` device px per CSS px. Each corner tile runs
 * far enough into the box that the edge next to it no longer depends on the
 * corner, so stretching the middle strip is exact. It stops at half the
 * smallest box the image is used on: past that CSS shrinks every slice.
 */
export function nineSliceGeometry(
  layers: ShadowLayer[],
  radii: CornerRadii,
  scale: number,
  minWidth: number,
  minHeight: number,
): NineSliceGeometry | null {
  const drawn = imageLayers(layers);
  if (!drawn.length || !(scale > 0)) return null;
  let margin = 0;
  let reach = 0;
  for (const layer of drawn) {
    // Sigma is blur / 2; Skia and Gecko stop the kernel at 3 sigma.
    const tail =
      1.5 * layer.blur + Math.max(Math.abs(layer.x), Math.abs(layer.y));
    margin = Math.max(margin, tail + Math.max(layer.spread, 0));
    reach = Math.max(reach, tail);
  }
  const maxRadius = Math.max(radii.tl, radii.tr, radii.br, radii.bl, 0);
  // Whole device px, so the box edge in the image lands on the box's own pixel edge.
  const outsetPx = Math.ceil((margin + 2) * scale);
  const want = Math.ceil(maxRadius + reach) + 2;
  const axis = (min: number): SliceAxis => {
    const inside = Math.max(1, Math.min(want, Math.floor(min / 2) - 2));
    const slice = outsetPx + Math.ceil(inside * scale);
    return { slice, width: slice / scale };
  };
  const x = axis(minWidth);
  const y = axis(minHeight);
  const middle = 2;
  const gap = Math.ceil(2 * (reach + maxRadius) * scale) + middle;
  return {
    outset: outsetPx / scale,
    outsetPx,
    x,
    y,
    middle,
    imageWidth: 2 * x.slice + middle,
    imageHeight: 2 * y.slice + middle,
    scratchWidth: 2 * x.slice + gap,
    scratchHeight: 2 * y.slice + gap,
    gap,
  };
}

/** The border-image shorthand for a published image. No `fill`: the box keeps its own background. */
export function borderImageValue(
  url: string,
  geometry: NineSliceGeometry,
): string {
  const { x, y, outset } = geometry;
  return `url("${url}") ${y.slice} ${x.slice} / ${y.width}px ${x.width}px / ${outset}px stretch`;
}
