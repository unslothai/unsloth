// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";

/**
 * Swallow the outside click that dismisses a non-modal menu before it activates an adjacent
 * control. Capture on `document` precedes React and Radix; module state survives content unmount
 * until the owed click, cancel, blur, new gesture, or release grace. Pointer identity, keyboard
 * activation, and focus are tracked, and touch dismissal is re-raised after the swallowed click.
 */

/** Menu and popper surfaces are selections, not dismissals. */
const MENU_SURFACE =
  '[role="menu"],[role="menuitem"],[data-radix-popper-content-wrapper]';

/** Shared state survives menu-content unmount during dismissal. */
let armed = false;
let graceTimer: number | undefined;
/** Touch dismissal is deferred by Radix until the resulting click. */
let armedByTouch = false;
/** Tracks whether the guarded pointer still owes a click. */
let pointerIsDown = false;
/** Only the armed pointer may end the guarded gesture. */
let armedPointerId: number | undefined;
/** Space activates on keyup; Enter activates on keydown. */
let activationKeyIsDown = false;
/** The pointer click was handled; a held Space keyup still owes one click. */
let keyboardOnly = false;
/** Used to undo focus taken by the swallowed press. */
let armedPressTarget: Node | undefined;
let focusBeforePress: Element | null = null;

/** Maximum post-release click delay. */
const CLICK_GRACE_MS = 500;

/** Keys that activate on keyup. */
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
  document.removeEventListener("pointerdown", disarmOnNewPointerDown, true);
  document.removeEventListener("click", swallowClick, true);
  document.removeEventListener("pointerup", startGrace, true);
  document.removeEventListener("pointercancel", disarmOnPointerCancel, true);
  document.removeEventListener("keydown", disarmOnKey, true);
  document.removeEventListener("keyup", disarmOnActivationKeyUp, true);
  window.removeEventListener("blur", disarmAndReleaseFocus);
};

/** The release or cancel of a pointer that is not the one the guard armed for. */
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

/** Anything the user types into. Dismissing a menu by clicking into it must leave the caret. */
const TEXT_ENTRY = "input,textarea,select";

function releaseFocusTakenByTheGuardedPress(): void {
  // Swallowing the click is not enough: blur only focus acquired by this guarded press, so a
  // later Space key cannot activate the dismissed control.
  const active = document.activeElement;
  if (!(active instanceof HTMLElement)) return;
  if (active === focusBeforePress) return;
  if (active.isContentEditable || active.matches(TEXT_ENTRY)) return;
  if (!(armedPressTarget instanceof Node)) return;
  if (!active.contains(armedPressTarget)) return;
  active.blur();
}

/** Retire a gesture that produced no click without leaving its pressed control focused. */
function disarmAndReleaseFocus(): void {
  releaseFocusTakenByTheGuardedPress();
  disarm();
}

/** A new pointer supersedes an uncompleted gesture. */
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
  // Radix defers touch dismissal to a document click. Re-raise a non-bubbling synthetic click
  // after removing this guard so Radix can close the menu without re-entering the swallower.
  document.dispatchEvent(new MouseEvent("click", { bubbles: false }));
}

const arm = (touch: boolean, pointerId: number, pressTarget: Node): void => {
  if (armed) return;
  armed = true;
  armedByTouch = touch;
  pointerIsDown = true;
  armedPointerId = pointerId;
  activationKeyIsDown = false;
  keyboardOnly = false;
  armedPressTarget = pressTarget;
  focusBeforePress = document.activeElement;
  document.addEventListener("pointerdown", disarmOnNewPointerDown, true);
  document.addEventListener("click", swallowClick, true);
  document.addEventListener("pointerup", startGrace, true);
  document.addEventListener("pointercancel", disarmOnPointerCancel, true);
  document.addEventListener("keydown", disarmOnKey, true);
  document.addEventListener("keyup", disarmOnActivationKeyUp, true);
  window.addEventListener("blur", disarmAndReleaseFocus);
};

/** Install the watcher for one open menu. */
export function installDismissingClickGuard(): () => void {
  const onPointerDown = (event: PointerEvent): void => {
    if (armed && pointerIsDown && event.pointerId !== armedPointerId) return;
    disarmAndReleaseFocus();
    if (event.button !== 0) return;
    const target = event.target;
    if (!(target instanceof Element)) return;
    if (target.closest(MENU_SURFACE)) return;
    arm(event.pointerType === "touch", event.pointerId, target);
  };
  document.addEventListener("pointerdown", onPointerDown, true);
  return () => {
    document.removeEventListener("pointerdown", onPointerDown, true);
  };
}

/** Mount the guard inside an open non-modal menu's content. */
export function useDismissingClickGuard(): void {
  useEffect(installDismissingClickGuard, []);
}
