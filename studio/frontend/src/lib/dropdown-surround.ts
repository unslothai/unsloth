// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Dark dropdowns glow in the darkest surface they touch (--dropdown-surround-bg).
 * Menus that would blend into their surface get data-surround-blend.
 * Sub-menus copy their parent's fill. Model picker dropdowns glow in its colour.
 */

const DROPDOWN_SELECTOR = [
  '[data-slot="dropdown-menu-content"]',
  '[data-slot="dropdown-menu-sub-content"]',
  '[data-slot="context-menu-content"]',
  '[data-slot="context-menu-sub-content"]',
  '[data-slot="select-content"]',
  '[data-slot="combobox-content"]',
  '[data-slot="popover-content"]',
  // Bespoke dropdowns not built on ui/ primitives.
  ".dropdown-surface",
].join(",");

const SUB_MENU_SELECTOR =
  '[data-slot="dropdown-menu-sub-content"], [data-slot="context-menu-sub-content"]';

// Panels whose dropdowns glow in the panel's colour.
const GLOW_HOST_SELECTOR = ".unsloth-model-selector-menu";

const SURROUND_VAR = "--dropdown-surround-bg";
const BLEND_ATTR = "data-surround-blend";
// Max per-channel difference (0-255) that counts as the same surface.
const BLEND_TOLERANCE = 6;
// Probe distance outside the menu edge.
const EDGE_PROBE_PX = 8;
// Radix positions poppers after a few frames.
const MAX_POSITION_FRAMES = 6;

let canvasContext: CanvasRenderingContext2D | null | undefined;
// Sub-menus whose fill this module set inline, so a theme switch can clear it.
const inlineFills = new WeakSet<HTMLElement>();

function getCanvasContext(doc: Document): CanvasRenderingContext2D | null {
  if (canvasContext === undefined) {
    const canvas = doc.createElement("canvas");
    canvas.width = 1;
    canvas.height = 1;
    canvasContext = canvas.getContext("2d", { willReadFrequently: true });
  }
  return canvasContext;
}

type Rgb = [number, number, number];

function isTransparent(color: string): boolean {
  return color === "transparent" || color === "rgba(0, 0, 0, 0)";
}

/** Alpha of any CSS colour the canvas can parse (oklch, color-mix output, ...). */
function alphaOf(ctx: CanvasRenderingContext2D, color: string): number {
  ctx.clearRect(0, 0, 1, 1);
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, 1, 1);
  return ctx.getImageData(0, 0, 1, 1).data[3] / 255;
}

/**
 * Composited surface colour at (x, y) outside `dropdown`. Boxes smaller than
 * the menu are skipped so rows and chips do not tint the glow.
 */
function surfaceColorAt(
  doc: Document,
  ctx: CanvasRenderingContext2D,
  dropdown: Element,
  minArea: number,
  x: number,
  y: number,
): Rgb | null {
  const view = doc.defaultView;
  if (!view || x < 0 || y < 0 || x >= view.innerWidth || y >= view.innerHeight)
    return null;
  const layers: string[] = [];
  for (const el of doc.elementsFromPoint(x, y)) {
    if (dropdown.contains(el)) continue;
    const style = view.getComputedStyle(el);
    if (isTransparent(style.backgroundColor) || style.opacity === "0") continue;
    const box = el.getBoundingClientRect();
    if (box.width * box.height < minArea) continue;
    layers.push(style.backgroundColor);
    if (alphaOf(ctx, style.backgroundColor) >= 1) break;
  }
  if (layers.length === 0) return null;
  ctx.clearRect(0, 0, 1, 1);
  for (let i = layers.length - 1; i >= 0; i--) {
    ctx.fillStyle = layers[i];
    ctx.fillRect(0, 0, 1, 1);
  }
  const [r, g, b, a] = ctx.getImageData(0, 0, 1, 1).data;
  // No opaque floor; fall back to --background.
  if (a < 255) return null;
  return [r, g, b];
}

/** `dropdown`'s own fill as painted over `surround`. */
function paintedFill(
  ctx: CanvasRenderingContext2D,
  dropdown: Element,
  surround: Rgb,
): Rgb {
  const view = dropdown.ownerDocument.defaultView;
  ctx.clearRect(0, 0, 1, 1);
  ctx.fillStyle = `rgb(${surround.join(" ")})`;
  ctx.fillRect(0, 0, 1, 1);
  const fill = view?.getComputedStyle(dropdown).backgroundColor;
  if (fill && !isTransparent(fill)) {
    ctx.fillStyle = fill;
    ctx.fillRect(0, 0, 1, 1);
  }
  const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data;
  return [r, g, b];
}

function luminance([r, g, b]: Rgb): number {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

interface Surround {
  /** The surface the menu mostly sits on: decides whether it would vanish. */
  majority: Rgb;
  /** The deepest surface it touches: the glow's colour. */
  darkest: Rgb;
}

function measureSurround(
  doc: Document,
  ctx: CanvasRenderingContext2D,
  dropdown: HTMLElement,
): Surround | null {
  const rect = dropdown.getBoundingClientRect();
  const minArea = rect.width * rect.height;
  const cx = rect.left + rect.width / 2;
  const cy = rect.top + rect.height / 2;
  const top = rect.top - EDGE_PROBE_PX;
  const right = rect.right + EDGE_PROBE_PX;
  const bottom = rect.bottom + EDGE_PROBE_PX;
  const left = rect.left - EDGE_PROBE_PX;
  // Ring just outside the menu.
  const points: [number, number][] = [
    [cx, top],
    [right, cy],
    [cx, bottom],
    [left, cy],
    [left, top],
    [right, top],
    [right, bottom],
    [left, bottom],
  ];

  // Modal layers set inline pointer-events: none, hiding them from
  // hit-testing. Lift it for this synchronous probe only.
  const disabled = [
    ...doc.querySelectorAll<HTMLElement>('[style*="pointer-events: none"]'),
  ];
  for (const el of disabled) el.style.pointerEvents = "auto";
  const votes = new Map<string, number>();
  const colors = new Map<string, Rgb>();
  let best: string | null = null;
  let darkest: Rgb | null = null;
  try {
    for (const [x, y] of points) {
      const color = surfaceColorAt(doc, ctx, dropdown, minArea, x, y);
      if (!color) continue;
      const key = color.join(" ");
      colors.set(key, color);
      const count = (votes.get(key) ?? 0) + 1;
      votes.set(key, count);
      if (best === null || count > (votes.get(best) ?? 0)) best = key;
      if (!darkest || luminance(color) < luminance(darkest)) darkest = color;
    }
  } finally {
    for (const el of disabled) el.style.pointerEvents = "none";
  }
  const majority = best === null ? undefined : colors.get(best);
  return majority && darkest ? { majority, darkest } : null;
}

/** The menu a sub-menu opened from. */
function parentMenuOf(doc: Document, subMenu: HTMLElement): HTMLElement | null {
  return (
    triggerOf(doc, subMenu)?.closest<HTMLElement>(DROPDOWN_SELECTOR) ?? null
  );
}

/** The control that opened `dropdown`. */
function triggerOf(doc: Document, dropdown: HTMLElement): Element | null {
  if (dropdown.id) {
    const direct = doc.querySelector(
      `[aria-controls="${CSS.escape(dropdown.id)}"]`,
    );
    if (direct) return direct;
  }
  // Some pickers point aria-controls at an inner listbox.
  for (const el of doc.querySelectorAll(
    '[aria-expanded="true"][aria-controls]',
  )) {
    const controlled = doc.getElementById(
      el.getAttribute("aria-controls") ?? "",
    );
    if (controlled && dropdown.contains(controlled)) return el;
  }
  return null;
}

function applySurround(
  win: Window,
  dropdown: HTMLElement,
  framesLeft = MAX_POSITION_FRAMES,
): void {
  win.requestAnimationFrame(() => {
    if (!dropdown.isConnected) return;
    const rect = dropdown.getBoundingClientRect();
    const placed =
      rect.width > 0 &&
      rect.height > 0 &&
      rect.bottom > 0 &&
      rect.top < win.innerHeight;
    if (!placed) {
      if (framesLeft > 0) applySurround(win, dropdown, framesLeft - 1);
      return;
    }
    const ctx = getCanvasContext(win.document);
    const surround = ctx ? measureSurround(win.document, ctx, dropdown) : null;
    const host = triggerOf(win.document, dropdown)?.closest(GLOW_HOST_SELECTOR);
    if (host) {
      dropdown.style.setProperty(
        SURROUND_VAR,
        win.getComputedStyle(host).backgroundColor,
      );
    } else if (surround) {
      dropdown.style.setProperty(
        SURROUND_VAR,
        `rgb(${surround.darkest.join(" ")})`,
      );
    } else {
      dropdown.style.removeProperty(SURROUND_VAR);
    }

    // The parent was measured earlier, so its fill is final.
    const parent = dropdown.matches(SUB_MENU_SELECTOR)
      ? parentMenuOf(win.document, dropdown)
      : null;
    if (parent) {
      dropdown.toggleAttribute(BLEND_ATTR, parent.hasAttribute(BLEND_ATTR));
      // Important, so no menu class can override it.
      dropdown.style.setProperty(
        "background-color",
        win.getComputedStyle(parent).backgroundColor,
        "important",
      );
      inlineFills.add(dropdown);
      return;
    }

    if (!ctx || !surround) {
      dropdown.removeAttribute(BLEND_ATTR);
      return;
    }
    const fill = paintedFill(ctx, dropdown, surround.majority);
    const blends = fill.every(
      (channel, i) =>
        Math.abs(channel - surround.majority[i]) <= BLEND_TOLERANCE,
    );
    dropdown.toggleAttribute(BLEND_ATTR, blends);
  });
}

/** Drops everything applySurround set, for light mode. */
function clearSurround(dropdown: HTMLElement): void {
  dropdown.style.removeProperty(SURROUND_VAR);
  dropdown.removeAttribute(BLEND_ATTR);
  if (inlineFills.delete(dropdown)) {
    dropdown.style.removeProperty("background-color");
  }
}

export function watchDropdownSurround(win: Window): void {
  const doc = win.document;
  const root = doc.documentElement;
  const observer = new MutationObserver((records) => {
    // Only dark-theme rules use these.
    if (!root.classList.contains("dark")) return;
    for (const record of records) {
      for (const node of record.addedNodes) {
        if (!(node instanceof HTMLElement)) continue;
        if (node.matches(DROPDOWN_SELECTOR)) applySurround(win, node);
        for (const el of node.querySelectorAll<HTMLElement>(
          DROPDOWN_SELECTOR,
        )) {
          applySurround(win, el);
        }
      }
    }
  });
  observer.observe(doc.body, { childList: true, subtree: true });

  // Menus left open across a theme switch: clear them for light, re-measure them for dark.
  let dark = root.classList.contains("dark");
  new MutationObserver(() => {
    const next = root.classList.contains("dark");
    if (next === dark) return;
    dark = next;
    const open = [...doc.querySelectorAll<HTMLElement>(DROPDOWN_SELECTOR)];
    if (!dark) {
      for (const el of open) clearSurround(el);
      return;
    }
    // Wait out the theme's colour fades, or the probe reads them halfway.
    const fades = doc
      .getAnimations()
      .filter((animation) => animation instanceof CSSTransition)
      .map((animation) => animation.finished);
    void Promise.allSettled(fades).then(() => {
      if (!root.classList.contains("dark")) return;
      // Document order puts a parent menu before its sub-menu.
      for (const el of open) applySurround(win, el);
    });
  }).observe(root, { attributes: true, attributeFilter: ["class"] });
}
