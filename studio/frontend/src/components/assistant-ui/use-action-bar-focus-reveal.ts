// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useAui } from "@assistant-ui/react";
import {
  useCallback,
  useEffect,
  useRef,
  type FocusEvent as ReactFocusEvent,
  type MouseEvent as ReactMouseEvent,
} from "react";
import { isTouchClick } from "@/components/ui/touch-click";

/**
 * Mounts an autohidden action bar while focus is inside the message, the way hovering it does.
 *
 * `autohide="not-last"` UNMOUNTS every bar but the newest reply's, so Copy, Edit, Refresh,
 * Delete, Read aloud and More leave the tab order on older messages and a keyboard or screen
 * reader user has no way back: `:focus-within` in CSS cannot help, there is nothing to style.
 * The reveal has to be JS, and it drives `message.setIsHovering`, the same flag the library's
 * own `mouseenter`/`mouseleave` (MessagePrimitive.Root) writes and the only input to
 * `useActionBarFloatStatus` besides the More menu's interaction lock. Reusing it rather than
 * layering a second visibility source is what keeps the two from disagreeing.
 *
 * One flag, two writers, so the two clobber each other unless this hook covers both crossings:
 *   - pointer leaves while focus is inside (a Tab that scrolls the message under a parked
 *     cursor does exactly this): the library clears the flag, which would unmount the element
 *     that currently has focus. `reassert` below sets it back inside the same event.
 *   - focus leaves while the pointer is still over the message: clearing would unmount a bar
 *     the user is pointing at, and no second `mouseenter` is coming. The `:hover` test defers
 *     to the library's own `mouseleave` instead.
 */
export function useActionBarFocusReveal() {
  const aui = useAui();
  const rootRef = useRef<HTMLDivElement | null>(null);
  const focusWithinRef = useRef(false);
  const clearFrameRef = useRef<number | null>(null);
  const popupObserverRef = useRef<MutationObserver | null>(null);

  // The More menu is portaled OUTSIDE the message, so focus entering it looks like a blur.
  // Its own interaction lock keeps the bar mounted meanwhile, but the trigger this hook has to
  // hand focus back to lives in that bar, so a popup this message owns counts as engaged.
  // Scoped to the action bar, NOT to every expanded descendant. Reasoning and tool cards are
  // Radix CollapsibleTriggers and render aria-expanded="true" while open, which is the resting
  // state of a message whose tool output the reader has expanded. An unscoped lookup treated
  // those as an open popup, so `decide` rescheduled itself every frame for as long as the
  // disclosure stayed open, held focusWithinRef and the synthetic hover set, and left the bar
  // mounted: a per-frame DOM query per such message, which is the slowdown this branch removes.
  const openPopupTrigger = useCallback(
    () =>
      rootRef.current?.querySelector(
        ':is(.aui-assistant-action-bar-root, .aui-user-action-bar-root) [aria-expanded="true"]',
      ) ?? null,
    [],
  );

  const isEngaged = useCallback(() => {
    const el = rootRef.current;
    if (!el) return false;
    const active = document.activeElement;
    if (active && el.contains(active)) return true;
    return openPopupTrigger() !== null;
  }, [openPopupTrigger]);

  const cancelPendingClear = useCallback(() => {
    if (clearFrameRef.current !== null) {
      cancelAnimationFrame(clearFrameRef.current);
      clearFrameRef.current = null;
    }
    popupObserverRef.current?.disconnect();
    popupObserverRef.current = null;
  }, []);

  /** Defer clearing until focus settles; observe popup closure instead of polling. */
  const scheduleClear = useCallback(
    (restart: boolean) => {
      if (clearFrameRef.current !== null || popupObserverRef.current !== null) {
        if (!restart) return;
        cancelPendingClear();
      }
      const decide = () => {
        clearFrameRef.current = null;
        const el = rootRef.current;
        if (!el || !focusWithinRef.current) return;
        const active = document.activeElement;
        if (active && el.contains(active)) return;
        const trigger = openPopupTrigger();
        if (trigger) {
          // Watch closure or removal without waking for streamed content.
          const observer = new MutationObserver(() => {
            if (
              el.contains(trigger) &&
              trigger.getAttribute("aria-expanded") === "true"
            )
              return;
            observer.disconnect();
            popupObserverRef.current = null;
            clearFrameRef.current = requestAnimationFrame(decide);
          });
          popupObserverRef.current = observer;
          observer.observe(trigger, {
            attributes: true,
            attributeFilter: ["aria-expanded"],
          });
          for (
            let parent = trigger.parentElement;
            parent;
            parent = parent.parentElement
          ) {
            observer.observe(parent, { childList: true });
            if (parent === el) break;
          }
          return;
        }
        focusWithinRef.current = false;
        if (!el.matches(":hover")) {
          aui.message().setIsHovering(false);
        }
      };
      clearFrameRef.current = requestAnimationFrame(decide);
    },
    [aui, openPopupTrigger, cancelPendingClear],
  );

  // onFocus/onBlur on a container are focusin/focusout in React, so they give focus-within.
  const handleFocus = useCallback(
    (event: ReactFocusEvent<HTMLDivElement>) => {
      const el = rootRef.current;
      const target = event.target as Node | null;
      if (el && target && !el.contains(target)) {
        // Portaled focus must keep the pending popup-close check.
        scheduleClear(false);
        return;
      }
      cancelPendingClear();
      if (focusWithinRef.current) return;
      focusWithinRef.current = true;
      aui.message().setIsHovering(true);
    },
    [aui, cancelPendingClear, scheduleClear],
  );

  const handleBlur = useCallback(() => {
    if (!focusWithinRef.current) return;
    scheduleClear(true);
  }, [scheduleClear]);

  useEffect(() => {
    const el = rootRef.current;
    if (!el) return;
    // From an effect on purpose: MessagePrimitive.Root binds its own mouseleave from a ref
    // callback, which commits before effects run, so this listener is registered second and
    // runs second on the same element. Both writes land in one dispatch, the store settles on
    // `true`, and React never renders the intermediate `false` -- so the bar does not unmount
    // and the focused control is not destroyed under the user.
    const reassert = () => {
      if (focusWithinRef.current && isEngaged()) {
        aui.message().setIsHovering(true);
      }
    };
    el.addEventListener("mouseleave", reassert);
    return () => {
      el.removeEventListener("mouseleave", reassert);
      cancelPendingClear();
    };
  }, [aui, isEngaged, cancelPendingClear]);

  return {
    ref: rootRef,
    onFocus: handleFocus,
    onBlur: handleBlur,
    onClick: focusMessageOnTouch,
  };
}

function focusMessageOnTouch(event: ReactMouseEvent<HTMLDivElement>) {
  // Touch has no hover. Focus the existing message tab stop after a tap on
  // its prose, without stealing a link/button action or a text selection.
  if (event.defaultPrevented || !isTouchClick(event)) return;
  const target = event.target as Element;
  if (!event.currentTarget.contains(target)) return;
  if (
    target.closest(
      'button, a, input, textarea, select, [role="button"], [contenteditable="true"]',
    )
  )
    return;
  if (window.getSelection()?.isCollapsed === false) return;
  event.currentTarget.focus({ preventScroll: true });
}
