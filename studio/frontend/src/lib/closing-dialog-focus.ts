// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Blur on exit start: a focused element in the fading dialog makes WebKit restyle the whole page
 * (about 2 s at 100 turns); Radix still returns focus to the trigger, so landing is unchanged
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
