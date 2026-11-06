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
 * Mounts an autohidden action bar while focus is inside the message, via the same
 * `message.setIsHovering` flag the library's mouseenter/mouseleave writes. Two writers share one
 * flag, so this covers both crossings: pointer leaving with focus inside (`reassert`), and focus
 * leaving with the pointer still over (defers to the library's mouseleave).
 */
export function useActionBarFocusReveal() {
  const aui = useAui();
  const rootRef = useRef<HTMLDivElement | null>(null);
  const focusWithinRef = useRef(false);
  const clearFrameRef = useRef<number | null>(null);
  const popupObserverRef = useRef<MutationObserver | null>(null);

  // The More menu is portaled outside the message, so focus in it counts as engaged. Scoped to the
  // action bar: open disclosures also carry aria-expanded and would hold the bar every frame.
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
    // From an effect on purpose: it registers after the library's mouseleave, so both writes land in
    // one dispatch and React never renders the intermediate `false`.
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
  // Touch has no hover: focus the message tab stop on a prose tap, not stealing actions.
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
