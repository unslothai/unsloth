// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useSyncExternalStore } from "react";

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
/**
 * Whether an activation key is held down during the guarded gesture. Space is the whole set: a
 * button activates on Space's KEYUP and on Enter's keydown, so Enter has always fired before the
 * pointer's own click and is covered by the `pointerIsDown` branch in `swallowClick`, while Space
 * still owes a click after the pointer is long gone.
 *
 * The engines split on whether that late click survives an intervening mouse release, and the
 * split is in their source rather than in anything we do. Gecko tracks the pending activation on
 * its own `HTML_ELEMENT_ACTIVE_FOR_KEYBOARD` flag, set on keydown and unset only on keyup
 * (`nsGenericHTMLElement::HandleKeyboardActivation`), so the release does not touch it. Blink and
 * WebKit gate the same activation on the shared `:active` state (`HTMLElement::
 * HandleKeyboardActivation`: `if (IsActive()) DispatchSimulatedClick`), which the release clears,
 * so they never fire it. Measured on all three: the message is deleted on firefox and not on
 * chromium or webkit.
 */
let activationKeyIsDown = false;
/**
 * Armed for a KEYBOARD-generated click only. Entered when the pointer's own click was swallowed
 * with Space still held: the release is accounted for, the Space keyup click is not. Any real
 * click in this state is a new gesture and must land.
 */
let keyboardOnly = false;

/** How long after the pointer is RELEASED a click may still arrive. */
const CLICK_GRACE_MS = 500;

/** A key that fires the focused control's click on its KEYUP. Enter fires on keydown instead. */
const isActivationKey = (event: KeyboardEvent): boolean =>
  event.key === " " || event.key === "Spacebar";

const disarmOnKey = (event: KeyboardEvent): void => {
  // Space has not activated anything yet -- that happens on its keyup -- so it cannot mean "the
  // user has moved on", whatever the pointer does next. Auto-repeat re-sets this, which is what
  // recovers the flag if a disarm cleared it while the key was still physically down.
  if (isActivationKey(event)) {
    activationKeyIsDown = true;
    return;
  }
  // Shift, Ctrl and friends get pressed mid-gesture constantly. Disarming on one while the button
  // is still held meant a press on the unconfirmed "Delete message" button, then a modifier, then
  // a release, deleted the message: measured on chromium, and safe once this returns early.
  if (pointerIsDown) return;
  disarm();
};

const disarmOnActivationKeyUp = (event: KeyboardEvent): void => {
  if (!isActivationKey(event)) return;
  activationKeyIsDown = false;
  // The pointer is still down, so its release still owes a click and the bound belongs to that.
  if (pointerIsDown) return;
  // The activation click is dispatched by THIS keyup's own default action, so it lands before a
  // zero-delay timeout runs; anything later than that is not this gesture's. Capture phase runs
  // ahead of the default action, which is why the timeout is scheduled here rather than after.
  if (graceTimer !== undefined) window.clearTimeout(graceTimer);
  graceTimer = window.setTimeout(disarm, 0);
};

const disarm = (): void => {
  if (graceTimer !== undefined) {
    window.clearTimeout(graceTimer);
    graceTimer = undefined;
  }
  if (!armed) return;
  armed = false;
  pointerIsDown = false;
  activationKeyIsDown = false;
  keyboardOnly = false;
  document.removeEventListener("click", swallowClick, true);
  document.removeEventListener("pointerup", startGrace, true);
  document.removeEventListener("pointercancel", disarm, true);
  document.removeEventListener("keydown", disarmOnKey, true);
  document.removeEventListener("keyup", disarmOnActivationKeyUp, true);
  window.removeEventListener("blur", disarm);
};

function startGrace(): void {
  pointerIsDown = false;
  if (graceTimer !== undefined) window.clearTimeout(graceTimer);
  // A release-anchored bound cannot retire the guard while Space is still down: the click that
  // key fires on its own keyup is still to come, and on Gecko it comes however long the key is
  // held. `disarmOnActivationKeyUp` re-imposes the bound the moment it is released, and `blur`
  // and `pointercancel` still cover a gesture that never gets that far.
  if (activationKeyIsDown) {
    graceTimer = undefined;
    return;
  }
  graceTimer = window.setTimeout(disarm, CLICK_GRACE_MS);
}

/** Anything the user types into. Dismissing a menu by clicking into it must leave the caret. */
const TEXT_ENTRY = "input,textarea,select";

function releaseFocusTakenByTheSwallowedPress(event: Event): void {
  // Throwing the click away is only half of undoing the press. The press also FOCUSED what it
  // landed on, and a focused button is one Space away from firing: measured on chromium, firefox
  // and webkit, click the unconfirmed "Delete message" button to dismiss the menu, then press
  // Space -- the key a reader uses to scroll -- and the message is gone, with no click involved
  // for the guard to swallow. The modal shield never left this behind, because with
  // `pointer-events: none` on the body the press landed on `HTML` and the button was never
  // focused: measured on the pre-PR shape, `document.activeElement` stays `BODY` throughout.
  //
  // Blur rather than restore: what was focused when the guard armed is the menu content, and the
  // dismissal has already unmounted it. Body is where the modal shape left focus anyway.
  const active = document.activeElement;
  if (!(active instanceof HTMLElement)) return;
  if (active.isContentEditable || active.matches(TEXT_ENTRY)) return;
  // Only focus the swallowed press itself moved. Anything else is the app's own, and taking it
  // would be a second unasked-for effect in place of the first.
  if (!(event.target instanceof Node)) return;
  if (!active.contains(event.target)) return;
  active.blur();
}

function swallowClick(event: Event): void {
  const keyboardGenerated = (event as MouseEvent).detail === 0;
  if (keyboardOnly) {
    // Everything the pointer owed has been paid; the only click left to eat is the one a held
    // Space fires on its keyup. A real click here is a NEW gesture and must land -- eating it is
    // the "swallowed too much" failure `second_click` and `rightclick_then_click` exist to catch.
    disarm();
    if (!keyboardGenerated) return;
    event.stopPropagation();
    event.preventDefault();
    return;
  }
  // A KEYBOARD-generated click carries no click count, and one that arrives while the guarded
  // pointer is still down is not the click this guard was armed for. Pressing a control focuses
  // it, so Enter or Space mid-gesture activates it, and treating that as the awaited click
  // leaves nothing armed for the one the RELEASE still synthesises. Measured on chromium: press
  // and hold the unconfirmed "Delete message" button with the menu open, press Enter, release,
  // and the message is gone. Swallow it and stay armed.
  if (pointerIsDown && keyboardGenerated) {
    event.stopPropagation();
    event.preventDefault();
    return;
  }
  if (activationKeyIsDown && !keyboardGenerated && !armedByTouch) {
    // The pointer's own click, with Space still held. On Gecko the control's activation click is
    // still to come on that key's keyup, so disarming here leaves nothing to eat it: measured on
    // firefox, press and hold the unconfirmed "Delete message" button with the menu open, hold
    // Space, release the pointer, release Space, and the message is gone. Swallow this one and
    // stay armed for exactly one keyboard-generated click. Mouse only: a touch tap has no key in
    // flight, and the branch below owes Radix a re-raised click that this path does not send.
    keyboardOnly = true;
    event.stopPropagation();
    event.preventDefault();
    // Throwing the click away is only half of undoing the press here too. Without this the
    // button the press landed on keeps focus for the whole of the held key and after it, so an
    // ordinary Space later activates it and deletes the message this guard just saved. Measured
    // on chromium, firefox and webkit. Blurring also removes the pending activation at its
    // source on Gecko, whose keyup handler returns early when the element is no longer the
    // focused one (`nsGenericHTMLElement::HandleKeyboardActivation`), so the swallow below is
    // belt and braces on the engines that would still fire one.
    releaseFocusTakenByTheSwallowedPress(event);
    return;
  }
  const touch = armedByTouch;
  disarm();
  event.stopPropagation();
  event.preventDefault();
  releaseFocusTakenByTheSwallowedPress(event);
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
  activationKeyIsDown = false;
  keyboardOnly = false;
  // Capture, so this runs before React's root-container delegation reaches any control.
  document.addEventListener("click", swallowClick, true);
  // A gesture that never becomes a click must not leave the swallower waiting for an unrelated
  // one: bound it at release, and drop it outright on a cancel, a key, or losing the window.
  document.addEventListener("pointerup", startGrace, true);
  document.addEventListener("pointercancel", disarm, true);
  document.addEventListener("keydown", disarmOnKey, true);
  document.addEventListener("keyup", disarmOnActivationKeyUp, true);
  window.addEventListener("blur", disarm);
};

/**
 * Whether any non-modal menu is open, published so the controls that COMMIT ON POINTERDOWN can
 * take themselves out of the hit test for exactly that long.
 *
 * WHAT THIS IS FOR
 *
 * Swallowing the click is enough for every control that acts on `click`, which is all of them
 * bar two: Radix Slider commits in `onPointerDown` -> `onSlideStart` -> `updateValues` ->
 * `onValueChange`, and Radix Select opens in `onPointerDown`. Those two have already acted by
 * the time `swallowClick` runs, so on desktop a single press on a visible `ParamSlider` with a
 * composer, action-bar or project menu open both dismissed the menu and wrote a new inference
 * value: measured, temperature 0.6 -> 1.7 on chromium and 0.6 -> 1.69 on firefox and webkit,
 * reaching `chat_settings` in `studio.db` and surviving a reload. The merge base's modal shield
 * absorbed the same press (`elementFromPoint` was `HTML`, body `pointer-events: none`), so this
 * is a regression of this branch rather than something pre-existing.
 *
 * WHY A SUBSCRIPTION AND NOT A DOCUMENT-LEVEL FLAG
 *
 * The obvious shape is an attribute on `<html>` and a rule that reads it. It was built and
 * measured and it is not free, because a descendant rule hung off the ROOT makes the engine walk
 * the tree to find its matches whether or not any exist. Menu open+close on the heavy-thread
 * page, chromium, medians of 3, 25K -> 300K characters, on a page with no slider on it at all:
 *
 *     branch as it stands                    67.3 -> 66.3 ms    recalc style   4.7 ->    4.1 ms
 *     root attribute + `[data-slot=...]`     56.3 -> 187.5 ms   recalc style  10.6 ->  114.1 ms
 *     root attribute + a class               67.0 -> 104.5 ms   recalc style   7.2 ->   33.0 ms
 *     the modal shield this branch removed  782.6 -> 4113.3 ms  recalc style 625.6 -> 3691.2 ms
 *
 * The first of those is the `data-slot` trap: nearly every element in this tree carries one, so
 * a rule ending in `[data-slot="slider"]` pulls all of them into the attribute's invalidation
 * set. The class is much better and still walks. Publishing the flag instead means the only
 * elements that do anything are the controls that subscribed, so the cost is the number of
 * sliders on the page and nothing else.
 *
 * Cancelling the `pointerdown` was built and rejected earlier on this branch: it removes the
 * press from React's delegation entirely, so the settings panel's own resize handle stopped
 * dragging and another menu's trigger opened nothing. Taking only the committing controls out
 * of the hit test leaves every other press exactly where it was, the resize handle included.
 */

/**
 * Ref-counted: submenus mount a guard of their own inside the parent's content, so the last
 * one to unmount owns the removal. A plain set/clear pair drops the shield while the parent
 * menu is still open.
 */
let openGuards = 0;
const menuOpenListeners = new Set<() => void>();

const publishMenuOpen = (): void => {
  for (const listener of menuOpenListeners) listener();
};

/** `useSyncExternalStore` pair, so a control re-renders exactly when this flips. */
export function subscribeNonModalMenuOpen(listener: () => void): () => void {
  menuOpenListeners.add(listener);
  return () => {
    menuOpenListeners.delete(listener);
  };
}

export function isNonModalMenuOpen(): boolean {
  return openGuards > 0;
}

/**
 * Publish that a non-modal menu is open, and return the release. Exported so the ref-count can
 * be tested without a renderer; mount `<MenuDismissGuard />` in application code.
 */
export function markNonModalMenuOpen(): () => void {
  if (openGuards++ === 0) publishMenuOpen();
  let released = false;
  return () => {
    // Idempotent: React can run an effect's cleanup twice under StrictMode, and a second
    // decrement would drop the shield while another menu is still open.
    if (released) return;
    released = true;
    openGuards = Math.max(0, openGuards - 1);
    if (openGuards === 0) publishMenuOpen();
  };
}

/**
 * Call from a control that ACTS ON POINTERDOWN. Everything else is covered by the click
 * swallower and must not use this: taking a control out of the hit test also takes away the
 * drag that starts on it.
 */
export function useShieldedFromDismissingPress(): boolean {
  return useSyncExternalStore(
    subscribeNonModalMenuOpen,
    isNonModalMenuOpen,
    () => false,
  );
}

/**
 * Call from inside an open non-modal menu's content. Mount it via `<MenuDismissGuard />`
 * rather than calling it directly, so the guard's lifetime is exactly the content's.
 */
export function useDismissingClickGuard(): void {
  useEffect(() => {
    const releaseMenuMark = markNonModalMenuOpen();
    const onPointerDown = (event: PointerEvent): void => {
      // A new gesture always supersedes the last one, so a press that never produced a click
      // cannot leave the swallower armed.
      disarm();
      // Only the primary button ever synthesises the click this exists to eat. A right or
      // middle press raises `contextmenu` or `auxclick` instead, so arming for one can only eat
      // the user's NEXT left click: measured on all three engines, the click after a right-click
      // dismissal was suppressed. The release-anchored bound caps that at 500 ms rather than
      // forever, but not arming at all is the actual answer.
      //
      // `button` is the whole test on purpose. macOS spells its secondary click ctrl+left, which
      // arrives as button 0 with `ctrlKey`, and the engines disagree about what follows it:
      // Blink raises `contextmenu` and no `click`, WebKit sends both. Skipping those would drop
      // the swallow on WebKit, which is the engine Desktop ships on macOS, and on every ctrl+left
      // elsewhere, where it is an ordinary primary click. So that one is left to the bound.
      if (event.button !== 0) return;
      const target = event.target;
      if (!(target instanceof Element)) return;
      if (target.closest(MENU_SURFACE)) return;
      arm(event.pointerType === "touch");
    };
    document.addEventListener("pointerdown", onPointerDown, true);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown, true);
      releaseMenuMark();
    };
  }, []);
}
