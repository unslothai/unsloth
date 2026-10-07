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

/** Whether (x, y) falls in a cut-off rounded corner, which hit-testing skips. */
function inRoundedCorner(
  style: CSSStyleDeclaration,
  box: DOMRect,
  x: number,
  y: number,
): boolean {
  const length = (value: string, size: number) =>
    (value.endsWith("%")
      ? (Number.parseFloat(value) / 100) * size
      : Number.parseFloat(value)) || 0;
  const [tl, tr, br, bl] = [
    style.borderTopLeftRadius,
    style.borderTopRightRadius,
    style.borderBottomRightRadius,
    style.borderBottomLeftRadius,
  ].map((radius) => {
    const [h, v = h] = radius.split(" ");
    return [length(h, box.width), length(v, box.height)];
  });
  // Overlapping radii shrink together, as in CSS (rounded-full is ~infinite).
  const fit = (size: number, a: number, b: number) =>
    a + b > size ? size / (a + b) : 1;
  const scale = Math.min(
    fit(box.width, tl[0], tr[0]),
    fit(box.width, bl[0], br[0]),
    fit(box.height, tl[1], bl[1]),
    fit(box.height, tr[1], br[1]),
  );
  const corners: [number[], number, number][] = [
    [tl, box.left, box.top],
    [tr, box.right, box.top],
    [br, box.right, box.bottom],
    [bl, box.left, box.bottom],
  ];
  for (const [[h, v], cornerX, cornerY] of corners) {
    const rx = h * scale;
    const ry = v * scale;
    if (!(rx > 0 && ry > 0)) continue;
    // Distance from the corner's centre of curvature, in radii.
    const dx = Math.abs(x - cornerX) - rx;
    const dy = Math.abs(y - cornerY) - ry;
    if (dx < 0 && dy < 0 && (dx / rx) ** 2 + (dy / ry) ** 2 > 1) return true;
  }
  return false;
}

interface Hit {
  el: Element;
  /** z-index of the outermost z-indexed box around `el`. */
  layer: number;
  order: number;
}

/**
 * Boxes under each point outside `dropdown`, topmost first. Walks the DOM
 * instead of hit-testing: a modal layer's inline `pointer-events: none` on
 * <body> hides everything from elementsFromPoint, and lifting it restyles the
 * whole document twice per open. Paint order is approximated by the outermost
 * z-index (portaled dialogs, toasts), then document order.
 */
function boxesAt(
  view: Window,
  dropdown: Element,
  points: [number, number][],
): Element[][] {
  const doc = view.document;
  // The page background paints under every layer, negative z-index included.
  const hits = points.map((): Hit[] => [
    { el: doc.documentElement, layer: -Infinity, order: 0 },
    { el: doc.body, layer: -Infinity, order: 1 },
  ]);
  let order = 2;
  const visit = (parent: Element, inside: number[], layer: number | null) => {
    for (const el of parent.children) {
      if (el === dropdown) continue;
      const box = el.getBoundingClientRect();
      // A box with no size (display: contents, a wrapper of fixed children)
      // still has children that paint.
      const empty = box.width === 0 && box.height === 0;
      const under = empty
        ? inside
        : inside.filter((i) => {
            const [x, y] = points[i];
            return (
              x >= box.left && x < box.right && y >= box.top && y < box.bottom
            );
          });
      if (under.length === 0 || (empty && el.childElementCount === 0)) continue;
      const style = view.getComputedStyle(el);
      if (style.display === "none") continue;
      const own =
        layer ??
        (style.position !== "static" && style.zIndex !== "auto"
          ? Number(style.zIndex) || 0
          : null);
      if (!empty) {
        const hit = { el, layer: own ?? 0, order: order++ };
        for (const i of under) {
          if (!inRoundedCorner(style, box, ...points[i])) hits[i].push(hit);
        }
      }
      visit(el, under, own);
    }
  };
  const all = points.map((_, i) => i);
  visit(doc.body, all, null);
  return hits.map((stack) =>
    stack
      .sort((a, b) => b.layer - a.layer || b.order - a.order)
      .map((hit) => hit.el),
  );
}

/**
 * Composited surface colour of `stack`, a topmost-first box list. Boxes
 * smaller than the menu are skipped so rows and chips do not tint the glow.
 */
function surfaceColorOf(
  view: Window,
  ctx: CanvasRenderingContext2D,
  stack: Element[],
  minArea: number,
): Rgb | null {
  const layers: string[] = [];
  for (const el of stack) {
    const style = view.getComputedStyle(el);
    if (
      isTransparent(style.backgroundColor) ||
      style.opacity === "0" ||
      style.visibility !== "visible"
    )
      continue;
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
  view: Window,
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
  // Ring just outside the menu, inside the viewport.
  const points = (
    [
      [cx, top],
      [right, cy],
      [cx, bottom],
      [left, cy],
      [left, top],
      [right, top],
      [right, bottom],
      [left, bottom],
    ] as [number, number][]
  ).filter(
    ([x, y]) => x >= 0 && y >= 0 && x < view.innerWidth && y < view.innerHeight,
  );

  const votes = new Map<string, number>();
  const colors = new Map<string, Rgb>();
  let best: string | null = null;
  let darkest: Rgb | null = null;
  for (const stack of boxesAt(view, dropdown, points)) {
    const color = surfaceColorOf(view, ctx, stack, minArea);
    if (!color) continue;
    const key = color.join(" ");
    colors.set(key, color);
    const count = (votes.get(key) ?? 0) + 1;
    votes.set(key, count);
    if (best === null || count > (votes.get(best) ?? 0)) best = key;
    if (!darkest || luminance(color) < luminance(darkest)) darkest = color;
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
    const surround = ctx ? measureSurround(win, ctx, dropdown) : null;
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
