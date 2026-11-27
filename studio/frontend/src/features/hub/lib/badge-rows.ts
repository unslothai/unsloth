// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/*
 * README badge rows (a paragraph of two or more linked images) lay out as one wrapped flex row.
 * They are marked here rather than styled through `:is(p, div):has(> a:nth-of-type(2) > img) img`:
 * a style after a :has() made Chromium test that :has() on every p/div above every image and link
 * in the app, so the chat thread's favicons and source chips paid for a rule meant for the Hub.
 */
export const BADGE_ROW_ATTRIBUTE = "data-badge-row";
const BADGE_ROW = ":scope > a:nth-of-type(2) > img";

/** Flag exactly the p/div elements under `root` the old :has() matched. Idempotent. */
export function markBadgeRows(root: ParentNode): void {
  for (const el of root.querySelectorAll("p, div")) {
    const on = el.querySelector(BADGE_ROW) !== null;
    if (on !== el.hasAttribute(BADGE_ROW_ATTRIBUTE))
      el.toggleAttribute(BADGE_ROW_ATTRIBUTE, on);
  }
}

/** Ref callback: mark now and again whenever the rendered README changes shape. */
export function badgeRowsRef(
  root: HTMLElement | null,
): (() => void) | undefined {
  if (!root) return undefined;
  markBadgeRows(root);
  const observer = new MutationObserver(() => markBadgeRows(root));
  observer.observe(root, { childList: true, subtree: true });
  return () => observer.disconnect();
}
