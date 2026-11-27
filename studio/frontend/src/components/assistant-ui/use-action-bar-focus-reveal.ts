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

const TOOLTIP_CONTENT = '[data-slot="tooltip-content"]';

/**
 * Keeps the autohidden bar mounted while focus is inside, via message.setIsHovering shared with the
 * library's mouseenter/mouseleave; covers pointer-out with focus inside (reassert) and focus-out
 * with pointer still over (defers to library's mouseleave).
 */
export function useActionBarFocusReveal() {
  const aui = useAui();
  const rootRef = useRef<HTMLDivElement | null>(null);
  const focusWithinRef = useRef(false);
  const clearFrameRef = useRef<number | null>(null);
  const popupObserverRef = useRef<MutationObserver | null>(null);
  const onCardRef = useRef(false);

  // Portaled More menu counts as engaged; other aria-expanded disclosures would hold the bar.
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
        if (!el.matches(":hover") && !onCardRef.current) {
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
    // Registers after mouseleave so one dispatch batches both writes; no intermediate false.
    let release: ((event: PointerEvent) => void) | null = null;
    const stopRelease = () => {
      if (release) document.removeEventListener("pointermove", release, true);
      release = null;
      onCardRef.current = false;
    };
    const reassert = (event: MouseEvent) => {
      // Hovering a portaled tooltip card keeps the bar; clears once the pointer leaves both.
      const into = event.relatedTarget as Element | null;
      if (!into?.closest?.(TOOLTIP_CONTENT)) {
        if (focusWithinRef.current && isEngaged()) aui.message().setIsHovering(true);
        return;
      }
      aui.message().setIsHovering(true);
      stopRelease();
      onCardRef.current = true;
      release = (move) => {
        const target = move.target as Element | null;
        if (target?.closest?.(TOOLTIP_CONTENT)) return;
        stopRelease();
        if (target && el.contains(target)) return;
        if (!(focusWithinRef.current && isEngaged())) aui.message().setIsHovering(false);
      };
      document.addEventListener("pointermove", release, true);
    };
    el.addEventListener("mouseleave", reassert);
    return () => {
      el.removeEventListener("mouseleave", reassert);
      stopRelease();
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
