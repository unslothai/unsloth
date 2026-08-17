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
 * disarmed by the click it was armed for, or by the next gesture, never by a deadline.
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

const disarm = (): void => {
  if (!armed) return;
  armed = false;
  document.removeEventListener("click", swallowClick, true);
  document.removeEventListener("pointercancel", disarm, true);
  document.removeEventListener("keydown", disarm, true);
};

function swallowClick(event: Event): void {
  disarm();
  event.stopPropagation();
  event.preventDefault();
}

const arm = (): void => {
  if (armed) return;
  armed = true;
  // Capture, so this runs before React's root-container delegation reaches any control.
  document.addEventListener("click", swallowClick, true);
  // A gesture that never becomes a click -- a pointercancel, or a keyboard interaction that
  // synthesises one later -- must not leave the swallower waiting for an unrelated click.
  document.addEventListener("pointercancel", disarm, true);
  document.addEventListener("keydown", disarm, true);
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
      const target = event.target;
      if (!(target instanceof Element)) return;
      if (target.closest(MENU_SURFACE)) return;
      arm();
    };
    document.addEventListener("pointerdown", onPointerDown, true);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown, true);
    };
  }, []);
}
