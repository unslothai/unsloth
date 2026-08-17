// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";

/**
 * Swallow the single click that dismisses an open non-modal Radix menu.
 *
 * WHY THESE MENUS ARE `modal={false}` AT ALL
 *
 * A modal Radix layer parks `pointer-events: none` on `<body>` for as long as it is open.
 * `pointer-events` is an INHERITED property, so that one write invalidates computed style
 * for the entire mounted subtree underneath it. On a long chat thread that subtree is the
 * thread, and opening a menu turns into a full-document style recalculation whose cost
 * scales with the thread rather than with the menu.
 *
 * WHAT THE SHIELD WAS ALSO DOING
 *
 * Absorbing the click that dismisses the menu, so the first click outside only ever closed
 * it. Dropping the shield brings back a real footgun: Radix's outside handler dismisses but
 * never cancels the event, so one click on a control next to the menu both closes the menu
 * and fires that control. In the assistant action bar the neighbours are "Refresh" and an
 * unconfirmed "Delete message", two buttons from the trigger.
 *
 * WHY THIS IS ARMED FROM `pointerdown` AND NOT FROM RADIX'S CALLBACK
 *
 * The obvious place to restore the swallow is `onPointerDownOutside`, and it is wrong in two
 * ways that a quick press hides. Measured on the heavy-thread smoke page, chromium, by
 * pressing the adjacent unconfirmed "Delete message" button with the menu open:
 *
 *     press and release in the same tick   menu closed, nothing deleted
 *     press held 600 ms                    menu closed, MESSAGE DELETED
 *     press with the main thread busy      menu closed, MESSAGE DELETED
 *     touch tap                            menu closed, MESSAGE DELETED
 *
 * The first two are the timer: a listener armed by `onPointerDownOutside` and disarmed by a
 * fixed deadline is gone before the browser synthesises `click` on release, and a person can
 * hold a button down for as long as they like. Blocking the main thread reproduces the same
 * outcome from a normal-length press, which is the version a heavy thread produces on its own.
 *
 * The third is ordering. `usePointerDownOutside` in @radix-ui/react-dismissable-layer 1.1.11
 * defers to the resulting `click` when `pointerType === "touch"`, and listens on
 * `ownerDocument` in the BUBBLE phase. React 19 delegates to the root container, which is
 * inside document, so the control's own `onClick` has already run by the time Radix calls us.
 * A guard armed there cannot swallow the click that armed it, and no amount of fixing the
 * timer changes that.
 *
 * So the guard watches `pointerdown` itself, on `document`, in the CAPTURE phase, which is
 * ahead of React's delegation and ahead of Radix's own listener for both pointer types. It is
 * disarmed by the click it was armed for, or by the next gesture, never by a deadline anchored
 * at the press.
 *
 * THE UPPER BOUND, AND WHY IT IS ANCHORED AT RELEASE
 *
 * Plenty of dismissing gestures never become a click at all: a press that turns into a drag or a
 * scroll, a press on an element the menu's own close unmounts before release, a window that
 * loses focus mid-gesture. With no upper bound the swallower survives all of those and eats an
 * unrelated click an arbitrary time later, which the user experiences as the app ignoring a
 * click for no reason and cannot report usefully. So the window opens at `pointerup` rather than
 * at `pointerdown`: a press held for a minute is still covered, and a gesture that produces no
 * click is still bounded.
 *
 * A bound is not a substitute for not arming, though. A right click raises `contextmenu` and
 * never a `click`, and measured on all three engines it ate the user's next left click; the
 * bound only shortened that to 500 ms. The primary-button check below is the real answer, and
 * the bound is what catches the cases nobody has enumerated.
 *
 * WHY THE ARM STATE OUTLIVES THE COMPONENT
 *
 * The dismissing `pointerdown` unmounts the menu content synchronously, so anything torn down
 * in this effect's cleanup would be gone before the `click` arrives. The arm state is
 * therefore module-level and self-disarming; the effect owns only the `pointerdown` watcher.
 */

/**
 * Anything a pointer can land on that belongs to an open overlay: a menu, a menu item, or the
 * wrapper Radix positions any popper content in. A press in there is a selection rather than a
 * dismissal, so it must reach its handler untouched.
 */
const MENU_SURFACE =
  '[role="menu"],[role="menuitem"],[data-radix-popper-content-wrapper]';

/**
 * At most one dismissing gesture is ever in flight, so one module-level flag serves every menu
 * and two menus open at once cannot leave a second listener armed behind the first.
 */
let armed = false;
let graceTimer: number | undefined;
/**
 * Whether the armed gesture came from a finger. Radix dismisses on the `pointerdown` itself for
 * a mouse, but for touch it defers to the resulting `click`, which the swallow below denies it.
 */
let armedByTouch = false;
/**
 * Whether the gesture that armed the guard is still pressed. A keydown only means "the user has
 * moved on" once the pointer is up; while it is still down, releasing over the same element will
 * still synthesise the click the guard exists to eat.
 */
let pointerIsDown = false;

/** How long after the pointer is RELEASED a click may still arrive. */
const CLICK_GRACE_MS = 500;

const disarmOnKey = (): void => {
  // Shift, Ctrl and friends get pressed mid-gesture constantly. Disarming on one while the button
  // is still held meant a press on the unconfirmed "Delete message" button, then a modifier, then
  // a release, deleted the message: measured on chromium, and safe once this returns early.
  if (pointerIsDown) return;
  disarm();
};

const disarm = (): void => {
  if (graceTimer !== undefined) {
    window.clearTimeout(graceTimer);
    graceTimer = undefined;
  }
  if (!armed) return;
  armed = false;
  pointerIsDown = false;
  document.removeEventListener("click", swallowClick, true);
  document.removeEventListener("pointerup", startGrace, true);
  document.removeEventListener("pointercancel", disarm, true);
  document.removeEventListener("keydown", disarmOnKey, true);
  window.removeEventListener("blur", disarm);
};

function startGrace(): void {
  pointerIsDown = false;
  if (graceTimer !== undefined) window.clearTimeout(graceTimer);
  graceTimer = window.setTimeout(disarm, CLICK_GRACE_MS);
}

function swallowClick(event: Event): void {
  const touch = armedByTouch;
  disarm();
  event.stopPropagation();
  event.preventDefault();
  if (!touch) return;
  // On touch, Radix's dismissal is a `once` CLICK listener on `document` in the bubble phase,
  // and the `stopPropagation` above is what it would otherwise have been woken by. Measured:
  // without this, a tap on non-focusable thread background leaves the menu open on chromium and
  // webkit. A tap on a focusable control happened to still close it, via `useFocusOutside`,
  // which is why the first version of this looked fine.
  //
  // Radix's deferred handler takes no arguments and ignores the click entirely -- it only
  // re-raises the pointerdown it already captured -- so a bare click dispatched at `document`
  // is enough to release it. `bubbles: false` keeps it to listeners on `document` itself, and
  // `disarm()` above has already removed ours, so this cannot re-enter.
  //
  // Two properties of this worth knowing before reusing it. It wakes EVERY listener on
  // `document`, not just Radix's, so it releases every dismissable layer that is currently
  // deferring: that is bounded today only because two of these menus cannot be open at once,
  // and it would need revisiting if that changed. And `isTrusted` is false on a synthetic
  // event, so anything that gates on trusted input will ignore it -- Radix does not, which is
  // what makes this work at all.
  document.dispatchEvent(new MouseEvent("click", { bubbles: false }));
}

const arm = (touch: boolean): void => {
  if (armed) return;
  armed = true;
  armedByTouch = touch;
  pointerIsDown = true;
  // Capture, so this runs before React's root-container delegation reaches any control.
  document.addEventListener("click", swallowClick, true);
  // A gesture that never becomes a click must not leave the swallower waiting for an unrelated
  // one: bound it at release, and drop it outright on a cancel, a key, or losing the window.
  document.addEventListener("pointerup", startGrace, true);
  document.addEventListener("pointercancel", disarm, true);
  document.addEventListener("keydown", disarmOnKey, true);
  window.addEventListener("blur", disarm);
};

/**
 * Call from inside an open non-modal menu's content. Mount it via `<MenuDismissGuard />`
 * rather than calling it directly, so the guard's lifetime is exactly the content's.
 */
export function useDismissingClickGuard(): void {
  useEffect(() => {
    const onPointerDown = (event: PointerEvent): void => {
      // A new gesture always supersedes the last one, so a press that never produced a click
      // cannot leave the swallower armed.
      disarm();
      // Only the primary button ever synthesises the click this exists to eat. A right or
      // middle press raises `contextmenu` or `auxclick` instead, so arming for one can only eat
      // the user's NEXT left click: measured on all three engines, the click after a right-click
      // dismissal was suppressed. The release-anchored bound caps that at 500 ms rather than
      // forever, but not arming at all is the actual answer.
      // Only the primary button ever synthesises the click this exists to eat. A right or
      // middle press raises `contextmenu` or `auxclick` instead, so arming for one can only eat
      // the user's NEXT left click: measured on all three engines, the click after a right-click
      // dismissal was suppressed. The release-anchored bound caps that at 500 ms rather than
      // forever, but not arming at all is the actual answer.
      if (event.button !== 0) return;
      const target = event.target;
      if (!(target instanceof Element)) return;
      if (target.closest(MENU_SURFACE)) return;
      arm(event.pointerType === "touch");
    };
    document.addEventListener("pointerdown", onPointerDown, true);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown, true);
    };
  }, []);
}
