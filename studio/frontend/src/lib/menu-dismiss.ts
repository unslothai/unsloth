// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type RefObject, useEffect } from "react";

/**
 * Swallow the outside click that dismisses a non-modal menu before it activates another control.
 * Capture on `document` precedes React and Radix; state survives content unmount until the click.
 */

const MENU_SURFACE =
  '[role="menu"],[role="menuitem"],[data-radix-popper-content-wrapper]';

/** Module-level so it survives menu-content unmount during dismissal. */
let armed = false;
let graceTimer: number | undefined;
/** Radix defers touch dismissal until the resulting click. */
let armedByTouch = false;
let pointerIsDown = false;
let armedPointerId: number | undefined;
/** Space activates on keyup; Enter activates on keydown. */
let activationKeyIsDown = false;
/** The pointer click was handled; a held Space keyup still owes one click. */
let keyboardOnly = false;
let armedPressTarget: Node | undefined;
let focusBeforePress: Element | null = null;
let armedTrigger: HTMLElement | null = null;

const CLICK_GRACE_MS = 500;

const isActivationKey = (event: KeyboardEvent): boolean =>
  event.key === " " || event.key === "Spacebar";

const disarmOnKey = (event: KeyboardEvent): void => {
  if (isActivationKey(event)) {
    activationKeyIsDown = true;
    return;
  }
  if (pointerIsDown) return;
  disarmAndReleaseFocus();
};

const disarmOnActivationKeyUp = (event: KeyboardEvent): void => {
  if (!isActivationKey(event)) return;
  activationKeyIsDown = false;
  if (pointerIsDown) return;
  if (graceTimer !== undefined) window.clearTimeout(graceTimer);
  graceTimer = window.setTimeout(disarmAndReleaseFocus, 0);
};

const disarm = (): void => {
  if (graceTimer !== undefined) {
    window.clearTimeout(graceTimer);
    graceTimer = undefined;
  }
  if (!armed) return;
  armed = false;
  pointerIsDown = false;
  armedPointerId = undefined;
  activationKeyIsDown = false;
  keyboardOnly = false;
  armedPressTarget = undefined;
  focusBeforePress = null;
  armedTrigger = null;
  document.removeEventListener("pointerdown", disarmOnNewPointerDown, true);
  document.removeEventListener("click", swallowClick, true);
  document.removeEventListener("pointerup", startGrace, true);
  document.removeEventListener("pointercancel", disarmOnPointerCancel, true);
  document.removeEventListener("keydown", disarmOnKey, true);
  document.removeEventListener("keyup", disarmOnActivationKeyUp, true);
  window.removeEventListener("blur", disarmAndReleaseFocus);
};

const isAnotherPointer = (event: PointerEvent): boolean =>
  armedPointerId !== undefined && event.pointerId !== armedPointerId;

const disarmOnPointerCancel = (event: PointerEvent): void => {
  if (isAnotherPointer(event)) return;
  disarmAndReleaseFocus();
};

function startGrace(event: PointerEvent): void {
  if (isAnotherPointer(event)) return;
  pointerIsDown = false;
  if (graceTimer !== undefined) window.clearTimeout(graceTimer);
  if (activationKeyIsDown) {
    graceTimer = undefined;
    return;
  }
  graceTimer = window.setTimeout(disarmAndReleaseFocus, CLICK_GRACE_MS);
}

/** Dismissing a menu by clicking into a text field must leave the caret there. */
const TEXT_ENTRY = "input,textarea,select";

function releaseFocusTakenByTheGuardedPress(): void {
  // Move focus off the control this press focused, so a later Space cannot activate it.
  const active = document.activeElement;
  if (!(active instanceof HTMLElement)) return;
  if (active === focusBeforePress) return;
  if (active.isContentEditable || active.matches(TEXT_ENTRY)) return;
  if (!(armedPressTarget instanceof Node)) return;
  if (!active.contains(armedPressTarget)) return;
  const trigger = armedTrigger;
  if (
    trigger?.isConnected &&
    !trigger.matches(":disabled") &&
    !trigger.closest("[inert]")
  ) {
    try {
      trigger.focus({ preventScroll: true });
      if (document.activeElement === trigger) return;
    } catch {
      // Fall through to the existing blur fallback for an unfocusable trigger.
    }
  }
  active.blur();
}

function disarmAndReleaseFocus(): void {
  releaseFocusTakenByTheGuardedPress();
  disarm();
}

function disarmOnNewPointerDown(event: PointerEvent): void {
  if (pointerIsDown && isAnotherPointer(event)) return;
  disarmAndReleaseFocus();
}

function swallowClick(event: Event): void {
  const keyboardGenerated = (event as MouseEvent).detail === 0;
  if (keyboardOnly) {
    disarm();
    if (!keyboardGenerated) return;
    event.stopPropagation();
    event.preventDefault();
    return;
  }
  if (pointerIsDown && keyboardGenerated) {
    event.stopPropagation();
    event.preventDefault();
    return;
  }
  if (activationKeyIsDown && !keyboardGenerated && !armedByTouch) {
    keyboardOnly = true;
    event.stopPropagation();
    event.preventDefault();
    releaseFocusTakenByTheGuardedPress();
    return;
  }
  const touch = armedByTouch;
  event.stopPropagation();
  event.preventDefault();
  releaseFocusTakenByTheGuardedPress();
  disarm();
  if (!touch) return;
  // Radix closes touch menus on a document click. Re-raise a non-bubbling one after removing the guard.
  document.dispatchEvent(new MouseEvent("click", { bubbles: false }));
}

const arm = (
  touch: boolean,
  pointerId: number,
  pressTarget: Node,
  trigger: HTMLElement | null,
): void => {
  if (armed) return;
  armed = true;
  armedByTouch = touch;
  pointerIsDown = true;
  armedPointerId = pointerId;
  activationKeyIsDown = false;
  keyboardOnly = false;
  armedPressTarget = pressTarget;
  focusBeforePress = document.activeElement;
  armedTrigger = trigger;
  document.addEventListener("pointerdown", disarmOnNewPointerDown, true);
  document.addEventListener("click", swallowClick, true);
  document.addEventListener("pointerup", startGrace, true);
  document.addEventListener("pointercancel", disarmOnPointerCancel, true);
  document.addEventListener("keydown", disarmOnKey, true);
  document.addEventListener("keyup", disarmOnActivationKeyUp, true);
  window.addEventListener("blur", disarmAndReleaseFocus);
};

export function installDismissingClickGuard(
  triggerRef?: RefObject<HTMLElement | null>,
): () => void {
  const onPointerDown = (event: PointerEvent): void => {
    if (armed && pointerIsDown && event.pointerId !== armedPointerId) return;
    disarmAndReleaseFocus();
    if (event.button !== 0) return;
    const target = event.target;
    if (!(target instanceof Element)) return;
    if (target.closest(MENU_SURFACE)) return;
    const trigger = triggerRef?.current;
    arm(
      event.pointerType === "touch",
      event.pointerId,
      target,
      trigger instanceof HTMLElement ? trigger : null,
    );
  };
  document.addEventListener("pointerdown", onPointerDown, true);
  return () => {
    document.removeEventListener("pointerdown", onPointerDown, true);
  };
}

export function useDismissingClickGuard(
  triggerRef: RefObject<HTMLElement | null>,
): void {
  useEffect(() => installDismissingClickGuard(triggerRef), [triggerRef]);
}
