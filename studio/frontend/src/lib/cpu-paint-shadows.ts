// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// When WebKitGTK paints on the CPU, Skia re-blurs every blurred box-shadow over
// its whole area on each repaint (68%+ of the web process while scrolling
// Settings). Large surfaces draw the same shadow from an image instead: the
// blurred layers are drawn once onto a canvas and painted with border-image,
// which is a plain image copy. index.css switches surfaces over under
// html.webkit-cpu-paint and reads the images from the variables set here.

import {
  type CornerRadii,
  type NineSliceGeometry,
  type ShadowLayer,
  borderImageValue,
  imageLayers,
  nineSliceGeometry,
  parseShadowLayers,
  remainingShadow,
  spreadRadius,
} from "./cpu-paint-shadow-geometry";

export const CPU_PAINT_CLASS = "webkit-cpu-paint";

interface ShadowVariant {
  /** Publishes --cpu-<name> (border-image) and --cpu-<name>-rest (box-shadow left over). */
  name: string;
  /** The shadow, as the surface's own rule writes it. */
  shadow: string;
  radius: string;
  /** The smallest box the image is used on, CSS px. */
  minWidth: number;
  minHeight: number;
}

/** Sonner's --border-radius in components/ui/sonner.tsx. */
export const TOAST_RADIUS = "calc(var(--radius-lg) + 2px)";
const COMPOSER_FOCUS_SHADOW =
  "var(--composer-focus-shadow, var(--composer-shadow))";

// Composer first: it is on screen from the first frame.
export const CPU_PAINT_SHADOW_VARIANTS: readonly ShadowVariant[] = [
  {
    name: "composer",
    shadow: "var(--composer-shadow)",
    radius: "var(--radius-3xl)",
    minWidth: 200,
    minHeight: 48,
  },
  {
    name: "composer-focus",
    shadow: COMPOSER_FOCUS_SHADOW,
    radius: "var(--radius-3xl)",
    minWidth: 200,
    minHeight: 48,
  },
  {
    name: "toast",
    shadow: "var(--composer-shadow)",
    radius: TOAST_RADIUS,
    minWidth: 200,
    minHeight: 48,
  },
  {
    name: "popup",
    shadow: "var(--popup-shadow)",
    radius: "var(--radius-3xl)",
    minWidth: 200,
    minHeight: 120,
  },
  // Sheets, and square dialogs.
  {
    name: "sheet",
    shadow: "var(--popup-shadow)",
    radius: "0px",
    minWidth: 200,
    minHeight: 200,
  },
  // Dialogs that drop to --radius-2xl at phone width.
  {
    name: "popup-compact",
    shadow: "var(--popup-shadow)",
    radius: "var(--radius-2xl)",
    minWidth: 200,
    minHeight: 120,
  },
  {
    name: "files",
    shadow: "var(--composer-shadow)",
    radius: "var(--radius-2xl)",
    minWidth: 200,
    minHeight: 60,
  },
  // The composer's corner drops to --radius-2xl in a narrow composer-shell container.
  {
    name: "composer-compact",
    shadow: "var(--composer-shadow)",
    radius: "var(--radius-2xl)",
    minWidth: 120,
    minHeight: 48,
  },
  {
    name: "composer-focus-compact",
    shadow: COMPOSER_FOCUS_SHADOW,
    radius: "var(--radius-2xl)",
    minWidth: 120,
    minHeight: 48,
  },
];

const NO_SHADOW = "0 0 #0000";

interface Published {
  image: string;
  rest: string;
}

function makeCanvas(width: number, height: number): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

type Context = CanvasRenderingContext2D;

function roundedRect(
  ctx: Context,
  x: number,
  y: number,
  w: number,
  h: number,
  r: CornerRadii,
): void {
  ctx.beginPath();
  ctx.moveTo(x + r.tl, y);
  ctx.lineTo(x + w - r.tr, y);
  if (r.tr) ctx.arcTo(x + w, y, x + w, y + r.tr, r.tr);
  ctx.lineTo(x + w, y + h - r.br);
  if (r.br) ctx.arcTo(x + w, y + h, x + w - r.br, y + h, r.br);
  ctx.lineTo(x + r.bl, y + h);
  if (r.bl) ctx.arcTo(x, y + h, x, y + h - r.bl, r.bl);
  ctx.lineTo(x, y + r.tl);
  if (r.tl) ctx.arcTo(x, y, x + r.tl, y, r.tl);
  ctx.closePath();
}

function mapRadii(r: CornerRadii, f: (v: number) => number): CornerRadii {
  return { tl: f(r.tl), tr: f(r.tr), br: f(r.br), bl: f(r.bl) };
}

/** False when canvas cannot parse the color, which it ignores silently. */
function canvasAcceptsColor(ctx: Context, color: string): boolean {
  ctx.fillStyle = "#010203";
  ctx.fillStyle = color;
  return (
    ctx.fillStyle !== "#010203" || color.trim().toLowerCase() === "#010203"
  );
}

/**
 * Draws the shadow of a large box on a scratch canvas, then copies its corners
 * and one middle strip per side into the 9-slice image. Each shape is drawn far
 * off the canvas with only its shadow offset back on; the border box is then
 * cut out, as CSS clips an outer shadow under the box.
 */
export function renderNineSlice(
  layers: ShadowLayer[],
  radii: CornerRadii,
  scale: number,
  g: NineSliceGeometry,
): string | null {
  const scratch = makeCanvas(g.scratchWidth, g.scratchHeight);
  const ctx = scratch.getContext("2d");
  if (!ctx) return null;
  const drawn = imageLayers(layers);
  if (!drawn.every((layer) => canvasAcceptsColor(ctx, layer.color)))
    return null;
  const edge = g.outsetPx;
  const boxWidth = g.scratchWidth - 2 * edge;
  const boxHeight = g.scratchHeight - 2 * edge;
  const away = g.scratchWidth + 1000;
  // CSS paints the first layer on top.
  for (const layer of [...drawn].reverse()) {
    const spread = layer.spread * scale;
    const w = boxWidth + 2 * spread;
    const h = boxHeight + 2 * spread;
    if (w <= 0 || h <= 0) continue;
    const shape = mapRadii(radii, (r) => spreadRadius(r, layer.spread) * scale);
    ctx.save();
    if (layer.blur > 0) {
      // shadowBlur and shadowOffset ignore the transform, so they are device px already.
      ctx.shadowColor = layer.color;
      ctx.shadowBlur = layer.blur * scale;
      ctx.shadowOffsetX = layer.x * scale + away;
      ctx.shadowOffsetY = layer.y * scale;
      ctx.fillStyle = "#000";
      roundedRect(ctx, edge - spread - away, edge - spread, w, h, shape);
    } else {
      // A blur-free layer is just the shape; a canvas shadow can land half a pixel off.
      ctx.fillStyle = layer.color;
      roundedRect(
        ctx,
        edge - spread + layer.x * scale,
        edge - spread + layer.y * scale,
        w,
        h,
        shape,
      );
    }
    ctx.fill();
    ctx.restore();
  }
  ctx.globalCompositeOperation = "destination-out";
  ctx.fillStyle = "#000";
  roundedRect(
    ctx,
    edge,
    edge,
    boxWidth,
    boxHeight,
    mapRadii(radii, (r) => r * scale),
  );
  ctx.fill();

  const out = makeCanvas(g.imageWidth, g.imageHeight);
  const o = out.getContext("2d");
  if (!o) return null;
  const kx = g.x.slice;
  const ky = g.y.slice;
  const m = g.middle;
  const farX = g.scratchWidth - kx;
  const farY = g.scratchHeight - ky;
  const midX = Math.floor((g.scratchWidth - m) / 2);
  const midY = Math.floor((g.scratchHeight - m) / 2);
  const piece = (
    sx: number,
    sy: number,
    w: number,
    h: number,
    dx: number,
    dy: number,
  ) => o.drawImage(scratch, sx, sy, w, h, dx, dy, w, h);
  piece(0, 0, kx, ky, 0, 0);
  piece(farX, 0, kx, ky, kx + m, 0);
  piece(0, farY, kx, ky, 0, ky + m);
  piece(farX, farY, kx, ky, kx + m, ky + m);
  piece(midX, 0, m, ky, kx, 0);
  piece(midX, farY, m, ky, kx, ky + m);
  piece(0, midY, kx, m, 0, ky);
  piece(farX, midY, kx, m, kx + m, ky);
  return out.toDataURL("image/png");
}

function px(value: string): number {
  const n = Number.parseFloat(value);
  return Number.isFinite(n) ? n : 0;
}

function build(
  variant: ShadowVariant,
  shadow: string,
  radius: number,
  scale: number,
): Published {
  const fallback = {
    image: "none",
    rest: shadow && shadow !== "none" ? shadow : NO_SHADOW,
  };
  const layers = parseShadowLayers(shadow);
  if (!layers) return fallback;
  const radii = { tl: radius, tr: radius, br: radius, bl: radius };
  const geometry = nineSliceGeometry(
    layers,
    radii,
    scale,
    variant.minWidth,
    variant.minHeight,
  );
  if (!geometry) return { image: "none", rest: remainingShadow(layers) };
  let url: string | null = null;
  try {
    url = renderNineSlice(layers, radii, scale, geometry);
  } catch {
    // A canvas the engine will not allocate or export: keep the CSS shadow.
  }
  if (!url) return fallback;
  return {
    image: borderImageValue(url, geometry),
    rest: remainingShadow(layers),
  };
}

function resolutionQuery(dppx: number): string {
  return `(min-resolution: ${dppx - 0.005}dppx) and (max-resolution: ${dppx + 0.005}dppx)`;
}

/**
 * Device px per CSS px before page zoom. WebKitGTK 2.54 counts page zoom in
 * devicePixelRatio and 2.50 does not; the resolution media feature leaves it
 * out in both, so it tells them apart.
 */
function deviceScale(zoom: number): number {
  const dpr = window.devicePixelRatio || 1;
  if (zoom === 1 || window.matchMedia(resolutionQuery(dpr)).matches) return dpr;
  return dpr / zoom;
}

/** Device px per CSS px the page is painted at. */
function renderScale(zoom: number): number {
  return deviceScale(zoom) * zoom;
}

export interface CpuPaintShadowOptions {
  /** Page zoom handed to the webview. */
  interfaceZoom: () => number;
  subscribeInterfaceZoom: (listener: () => void) => () => void;
}

/**
 * Adds the class and keeps the image variables current. Images are drawn off
 * the event that changed them, one variant per task; until one is ready its
 * surface keeps the real CSS shadow. Returns a function that undoes it all.
 */
export function startCpuPaintShadows(
  options: CpuPaintShadowOptions,
): () => void {
  const root = document.documentElement;
  const sheet = document.createElement("style");
  sheet.textContent = ":root {}";
  document.head.append(sheet);
  const rule = sheet.sheet?.cssRules[0] as CSSStyleRule | undefined;
  // One hidden probe per variant resolves its shadow and radius in the current theme.
  const probes = document.createElement("div");
  probes.setAttribute("aria-hidden", "true");
  probes.style.cssText =
    "position:fixed;left:0;top:0;width:0;height:0;overflow:hidden;visibility:hidden;pointer-events:none";
  const probeFor = new Map<ShadowVariant, HTMLElement>();
  for (const variant of CPU_PAINT_SHADOW_VARIANTS) {
    const probe = document.createElement("div");
    probe.style.boxShadow = variant.shadow;
    probe.style.borderRadius = variant.radius;
    probes.append(probe);
    probeFor.set(variant, probe);
  }
  document.body.append(probes);

  const cache = new Map<string, Published>();
  const shown = new Map<string, string>();
  // Drawn one per task; a task whose variant changed again since it was queued does nothing.
  const queue: (() => void)[] = [];
  let pending: ReturnType<typeof setTimeout> | undefined;
  const pump = () => {
    pending = undefined;
    try {
      queue.shift()?.();
    } finally {
      if (queue.length) pending = setTimeout(pump, 0);
    }
  };

  const publish = (name: string, value: Published) => {
    rule?.style.setProperty(`--cpu-${name}`, value.image);
    rule?.style.setProperty(`--cpu-${name}-rest`, value.rest);
  };

  const refresh = () => {
    const scale = renderScale(options.interfaceZoom() || 1);
    for (const variant of CPU_PAINT_SHADOW_VARIANTS) {
      const style = getComputedStyle(probeFor.get(variant) as HTMLElement);
      const shadow = style.boxShadow;
      const radius = px(style.borderTopLeftRadius);
      const key = `${shadow}|${radius}|${scale}|${variant.minWidth}x${variant.minHeight}`;
      // Most <html> mutations change nothing here.
      if (shown.get(variant.name) === key) continue;
      shown.set(variant.name, key);
      const hit = cache.get(key);
      if (hit) {
        publish(variant.name, hit);
        continue;
      }
      // Until the image is drawn, the surface keeps its CSS shadow.
      publish(variant.name, {
        image: "none",
        rest: shadow && shadow !== "none" ? shadow : NO_SHADOW,
      });
      queue.push(() => {
        if (shown.get(variant.name) !== key) return;
        const value = cache.get(key) ?? build(variant, shadow, radius, scale);
        cache.set(key, value);
        publish(variant.name, value);
      });
    }
    if (queue.length && pending === undefined) pending = setTimeout(pump, 0);
  };

  root.classList.add(CPU_PAINT_CLASS);
  refresh();

  // Theme, palette, contrast and custom appearance colors all land on <html>.
  const observer = new MutationObserver(refresh);
  observer.observe(root, {
    attributes: true,
    attributeFilter: ["class", "data-palette", "data-contrast-adjust", "style"],
  });
  let resolution: MediaQueryList | undefined;
  const watchResolution = () => {
    resolution?.removeEventListener("change", onResolution);
    resolution = window.matchMedia(
      resolutionQuery(deviceScale(options.interfaceZoom() || 1)),
    );
    resolution.addEventListener("change", onResolution);
  };
  function onResolution() {
    watchResolution();
    refresh();
  }
  watchResolution();
  const unsubscribeZoom = options.subscribeInterfaceZoom(onResolution);

  return () => {
    queue.length = 0;
    if (pending !== undefined) clearTimeout(pending);
    observer.disconnect();
    unsubscribeZoom();
    resolution?.removeEventListener("change", onResolution);
    root.classList.remove(CPU_PAINT_CLASS);
    sheet.remove();
    probes.remove();
  };
}

/**
 * Linux desktop app only: asks the shell whether this launch paints on the CPU
 * and, unless it clearly does not, switches large shadows to images. An older
 * shell without the answer gets the images too; they look the same either way.
 */
export function armCpuPaintShadows(
  options: CpuPaintShadowOptions & {
    cpuPainting: () => Promise<boolean | undefined>;
  },
): void {
  const root = document.documentElement;
  if (
    !root.classList.contains("tauri") ||
    !root.classList.contains("render-linux")
  )
    return;
  options.cpuPainting().then(
    (cpuPainting) => {
      if (cpuPainting !== false) startCpuPaintShadows(options);
    },
    () => startCpuPaintShadows(options),
  );
}
