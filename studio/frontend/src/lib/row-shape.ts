// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Marks one-line hover rows with data-single-line so CSS can draw them as pills; rows that
 * wrap keep --radius-row. Also sets each menu's top and bottom corners to its edge row's
 * radius plus the gap, so the curves stay concentric.
 */

// ui/ menu primitives, each its own surface.
const MENU_SLOT_SELECTOR = [
  '[data-slot="dropdown-menu-content"]',
  '[data-slot="dropdown-menu-sub-content"]',
  '[data-slot="context-menu-content"]',
  '[data-slot="context-menu-sub-content"]',
  '[data-slot="menubar-content"]',
  '[data-slot="menubar-sub-content"]',
  '[data-slot="select-content"]',
  '[data-slot="combobox-content"]',
].join(",");

// Plus bespoke and nested lists.
const MENU_SELECTOR = `${MENU_SLOT_SELECTOR}, [role="menu"], [role="listbox"]`;

const MENU_ROW_SELECTOR =
  '[role="menuitem"], [role="menuitemcheckbox"], [role="menuitemradio"], [role="option"]';

const ROW_SELECTOR = `.rounded-row, :is(${MENU_SELECTOR}) :is(${MENU_ROW_SELECTOR})`;

const SINGLE_LINE_ATTR = "data-single-line";

// Max distance (beyond the side gap) for a row to count as a menu's edge row.
const EDGE_SLACK_PX = 4;
// Rows inset further than this are not menu rows.
const MAX_SIDE_GAP_PX = 16;

// Measures the text's line boxes, not the row box (padding or icons can stretch it).
// Ignores text laid out outside the row, like visually hidden labels.
function isSingleLine(el: HTMLElement): boolean {
  const box = el.getBoundingClientRect();
  const range = el.ownerDocument.createRange();
  const walker = el.ownerDocument.createTreeWalker(el, NodeFilter.SHOW_TEXT);
  let top = Number.POSITIVE_INFINITY;
  let bottom = Number.NEGATIVE_INFINITY;
  let lineHeight = 0;
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    if (!node.textContent?.trim()) continue;
    range.selectNodeContents(node);
    for (const rect of range.getClientRects()) {
      if (!rect.height || rect.width <= 1) continue;
      if (rect.top < box.top - 1 || rect.bottom > box.bottom + 1) continue;
      top = Math.min(top, rect.top);
      bottom = Math.max(bottom, rect.bottom);
      lineHeight = Math.max(lineHeight, rect.height);
    }
  }
  // No visible text counts as one line.
  return !lineHeight || bottom - top < lineHeight * 1.5;
}

function setCorners(
  menu: HTMLElement,
  edge: "top" | "bottom",
  radius: number | null,
): void {
  for (const side of ["left", "right"]) {
    const prop = `border-${edge}-${side}-radius`;
    if (radius === null) menu.style.removeProperty(prop);
    // !important to beat the menu radii in index.css.
    else menu.style.setProperty(prop, `${radius}px`, "important");
  }
}

const isPainted = (cs: CSSStyleDeclaration) =>
  cs.boxShadow !== "none" ||
  !/^(transparent|rgba\(0, 0, 0, 0\))$/.test(cs.backgroundColor);
const isFloating = (el: HTMLElement | null) =>
  !!el && /^(absolute|fixed)$/.test(getComputedStyle(el).position);

// The painted box a menu draws as; null when the list is not a popup.
function surfaceOf(menu: HTMLElement): HTMLElement | null {
  if (menu.matches(MENU_SLOT_SELECTOR)) return menu;
  let el: HTMLElement | null = menu;
  for (let depth = 0; el && depth < 4; depth++, el = el.parentElement) {
    if (!isPainted(getComputedStyle(el))) continue;
    return isFloating(el) || isFloating(el.parentElement) ? el : null;
  }
  return null;
}

// Edges without a row against them keep the stylesheet radius.
function shapeMenu(list: HTMLElement): void {
  const menu = surfaceOf(list);
  if (!menu) return;
  const outer = menu.getBoundingClientRect();
  if (!outer.width) return;
  const rowRadius =
    Number.parseFloat(
      getComputedStyle(menu).getPropertyValue("--radius-row"),
    ) || 14;
  // Undo the open animation's scale.
  const scale = menu.offsetWidth / outer.width;
  const ownRows = [
    ...list.querySelectorAll<HTMLElement>(MENU_ROW_SELECTOR),
  ].filter(
    (row) => row.offsetHeight > 0 && row.closest(MENU_SELECTOR) === list,
  );
  // A nested list shapes the surface itself.
  if (!ownRows.length) return;
  // Full-width rows only, so a small row in a heading is not an edge.
  const rows = ownRows
    .map((row) => {
      const rect = row.getBoundingClientRect();
      const left = (rect.left - outer.left) * scale;
      const right = (outer.right - rect.right) * scale;
      const height = rect.height * scale;
      // Line test, not computed radius, which can lag a render.
      const radius = isSingleLine(row)
        ? height / 2
        : Math.min(rowRadius, height / 2);
      return {
        rect,
        side: Math.max(left, right),
        radius,
        fullWidth: Math.abs(left - right) <= 2,
      };
    })
    .filter((row) => row.fullWidth && row.side <= MAX_SIDE_GAP_PX);
  if (!rows.length) {
    setCorners(menu, "top", null);
    setCorners(menu, "bottom", null);
    return;
  }
  const first = rows[0];
  const last = rows[rows.length - 1];
  const top = (first.rect.top - outer.top) * scale;
  const bottom = (outer.bottom - last.rect.bottom) * scale;
  setCorners(
    menu,
    "top",
    top >= -1 && top <= first.side + EDGE_SLACK_PX
      ? first.radius + first.side
      : null,
  );
  setCorners(
    menu,
    "bottom",
    bottom >= -1 && bottom <= last.side + EDGE_SLACK_PX
      ? last.radius + last.side
      : null,
  );
}

export function watchRowShape(win: Window): void {
  const doc = win.document;
  // Runs after layout, before paint.
  const sizes = new ResizeObserver((entries) => {
    const menus = new Set<HTMLElement>();
    for (const entry of entries) {
      const el = entry.target as HTMLElement;
      // Removed rows report a final zero size.
      if (!el.isConnected) {
        sizes.unobserve(el);
        continue;
      }
      if (el.matches(MENU_SELECTOR)) {
        menus.add(el);
        continue;
      }
      el.toggleAttribute(SINGLE_LINE_ATTR, isSingleLine(el));
      const menu = el.closest<HTMLElement>(MENU_SELECTOR);
      if (menu) menus.add(menu);
    }
    for (const menu of menus) shapeMenu(menu);
  });
  const track = (node: Node) => {
    if (!(node instanceof HTMLElement)) return;
    // Menus too, since their width can settle after opening.
    const watched = `${ROW_SELECTOR}, ${MENU_SELECTOR}`;
    if (node.matches(watched)) sizes.observe(node);
    for (const el of node.querySelectorAll<HTMLElement>(watched)) {
      sizes.observe(el);
    }
  };
  new MutationObserver((records) => {
    for (const record of records) {
      for (const node of record.addedNodes) track(node);
    }
  }).observe(doc.body, { childList: true, subtree: true });
  track(doc.body);
}
