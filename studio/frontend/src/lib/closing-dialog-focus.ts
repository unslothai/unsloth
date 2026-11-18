// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A closing dialog keeps focus inside it until it unmounts after its exit animation. In WebKit that
 * exit, with the focused element still in the fading dialog, ends in one long restyle of the whole
 * page: about 2 s on a 100-turn chat when Settings, the command palette, search or rename closes.
 * Blurring as the exit animation starts avoids it. Radix still returns focus to the trigger when the
 * dialog unmounts, so where focus lands afterwards is unchanged. Chromium never paid this.
 */

type ClosingTarget = {
  getAttribute(name: string): string | null;
  matches(selector: string): boolean;
  contains(other: unknown): boolean;
};

type Focusable = { blur(): void };

const DIALOG = '[role="dialog"], [role="alertdialog"]';

/** True when `target` is a dialog in its closing state that holds `active`, which was then blurred. */
export function blurInsideClosingDialog(
  target: ClosingTarget | null,
  active: Focusable | null,
): boolean {
  if (!target || !active || target.getAttribute("data-state") !== "closed") return false;
  if (!target.matches(DIALOG) || !target.contains(active)) return false;
  active.blur();
  return true;
}

function onAnimationStart(event: Event): void {
  const target = event.target;
  if (!(target instanceof Element)) return;
  const active = target.ownerDocument.activeElement;
  blurInsideClosingDialog(target, active instanceof HTMLElement ? active : null);
}

/** Idempotent: one capture listener per document. */
export function watchClosingDialogFocus(doc: Document): void {
  doc.addEventListener("animationstart", onAnimationStart, true);
}
