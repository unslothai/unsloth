// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { useAuiEvent } from "@assistant-ui/react";
import {
  type ReactNode,
  type RefCallback,
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useSyncExternalStore,
} from "react";

/**
 * Intent-aware autoscroll for a Thread viewport.
 *
 * Why we don't reuse assistant-ui's built-in autoscroll:
 *   `useThreadViewportAutoScroll` runs unconditionally whenever
 *   `ThreadPrimitive.Viewport` is mounted. Even with every
 *   `scrollToBottomOn*` prop disabled, it still installs observers
 *   that write `isAtBottom` to the shared viewport store on every
 *   layout change. On sidebar toggles, browser resizes, or
 *   mobile↔desktop breakpoint crossings, that write races our scroll
 *   correction — whoever writes last wins, and the scroll-to-bottom
 *   button flickers or sticks depending on observer ordering.
 *
 * Strategy:
 *   - Own `isAtBottom` as local state (exposed via
 *     `useIsThreadAtBottom`). Upstream can still write to its own
 *     store; nobody reads from it.
 *   - Drive the viewport with a single rAF loop governed by a follow
 *     deadline (`followUntilRef`). Any signal that can invalidate
 *     bottom alignment (resize, mutation, AUI event, programmatic
 *     scroll) extends the deadline; the loop pins to the bottom and
 *     reports `isAtBottom=true` until it expires, then settles on
 *     pure DOM observation.
 *   - Detect user intent (wheel up, touch swipe up, scroll direction)
 *     to detach. While detached, resize/mutation signals don't extend
 *     the deadline, so the button appears and stays. Re-attach when
 *     the user scrolls *down* within 24px of the bottom.
 */

// 2px, not 1: subpixel rounding on HiDPI displays can leave a fractional
// gap at the "true" bottom that a 1px threshold reports as not-at-bottom.
const AT_BOTTOM_THRESHOLD_PX = 2;
const RE_ATTACH_THRESHOLD_PX = 24;
const TOUCH_MOVE_THRESHOLD_PX = 4;
// Cumulative upward movement (summed across scroll events) that
// counts as a deliberate detach. Summed rather than per-event so that
// slow 1-px-per-event sources — middle-click autoscroll, scrollbar
// drags, some trackpads — accumulate instead of slipping under a
// per-event threshold forever.
const UPWARD_DETACH_THRESHOLD_PX = 2;
// Window during which the viewport pins to the bottom through
// layout/content races. Extends on every resize/mutation, so streaming
// keeps the viewport pinned as long as content keeps arriving; settles
// this long after the last change.
const FOLLOW_SETTLE_MS = 600;

export type ScrollToBottom = (behavior?: ScrollBehavior) => void;

type AutoScrollContextValue = {
  scrollToBottom: ScrollToBottom;
  getIsAtBottom: () => boolean;
  subscribe: (listener: () => void) => () => void;
};

const noopContext: AutoScrollContextValue = {
  scrollToBottom: () => {
    /* no viewport mounted */
  },
  getIsAtBottom: () => true,
  subscribe: () => () => {
    /* no-op */
  },
};

const AutoScrollContext = createContext<AutoScrollContextValue>(noopContext);

export function IntentAwareScrollProvider({
  value,
  children,
}: {
  value: AutoScrollContextValue;
  children: ReactNode;
}) {
  return (
    <AutoScrollContext.Provider value={value}>
      {children}
    </AutoScrollContext.Provider>
  );
}

export function useScrollThreadToBottom(): ScrollToBottom {
  return useContext(AutoScrollContext).scrollToBottom;
}

export function useIsThreadAtBottom(): boolean {
  const ctx = useContext(AutoScrollContext);
  return useSyncExternalStore(ctx.subscribe, ctx.getIsAtBottom, () => true);
}

export function useIntentAwareAutoScroll(): {
  ref: RefCallback<HTMLElement>;
  context: AutoScrollContextValue;
} {
  const cleanupRef = useRef<(() => void) | null>(null);

  const userDetachedRef = useRef(false);
  const followUntilRef = useRef(0);

  const isAtBottomRef = useRef(true);
  const listenersRef = useRef<Set<() => void>>(new Set());

  const scrollImplRef = useRef<ScrollToBottom>(() => {
    /* no viewport mounted */
  });

  const getIsAtBottom = useCallback(() => isAtBottomRef.current, []);

  const subscribe = useCallback((listener: () => void) => {
    listenersRef.current.add(listener);
    return () => {
      listenersRef.current.delete(listener);
    };
  }, []);

  const setIsAtBottom = useCallback((value: boolean) => {
    if (isAtBottomRef.current === value) {
      return;
    }
    isAtBottomRef.current = value;
    for (const listener of listenersRef.current) {
      listener();
    }
  }, []);

  const scrollToBottom = useCallback<ScrollToBottom>((behavior) => {
    scrollImplRef.current(behavior);
  }, []);

  const attach = useCallback(
    (el: HTMLElement) => {
      let rafId: number | null = null;
      let lastScrollTop = el.scrollTop;
      let lastClientWidth = el.clientWidth;
      let lastClientHeight = el.clientHeight;
      let upwardAccumulator = 0;
      let touchStartY = 0;

      const distanceFromBottom = (): number => {
        if (el.scrollHeight <= el.clientHeight) {
          return 0;
        }
        return el.scrollHeight - el.scrollTop - el.clientHeight;
      };
      let lastDistanceFromBottom = distanceFromBottom();

      const atBottomStrict = (): boolean =>
        distanceFromBottom() <= AT_BOTTOM_THRESHOLD_PX;

      // True only when there is room to scroll upward. Guards the
      // wheel/touch detach paths: a wheel-up or swipe-down gesture on
      // a viewport with nothing above (short thread, or already at
      // the top) can't express intent to leave the bottom and must
      // not flip userDetachedRef — otherwise later streaming updates
      // skip extendFollow and auto-follow stays dead for the session.
      const canScrollUp = (): boolean => el.scrollTop > 0;

      // True when a nested scrollable ancestor of the event target
      // (e.g. the reasoning panel's own overflow-y-auto region, or
      // any tool output with internal scroll) has room above and
      // will therefore consume the upward delta before it reaches
      // the viewport. Walking stops at the viewport element itself,
      // so only intermediate inner scrollers count.
      //
      // Wheel and touchmove events bubble, so without this check a
      // user reading back through a long reasoning pane mid-stream
      // would falsely detach the outer viewport.
      const innerScrollWillConsumeUpward = (
        target: EventTarget | null,
      ): boolean => {
        let node =
          target instanceof Element ? (target as Element | null) : null;
        while (node && node !== el) {
          if (node.scrollTop > 0) {
            const { overflowY } = window.getComputedStyle(node);
            if (
              overflowY === "auto" ||
              overflowY === "scroll" ||
              overflowY === "overlay"
            ) {
              return true;
            }
          }
          node = node.parentElement;
        }
        return false;
      };

      const extendFollow = (): void => {
        if (userDetachedRef.current) {
          return;
        }
        followUntilRef.current = performance.now() + FOLLOW_SETTLE_MS;
      };

      const detach = (): void => {
        userDetachedRef.current = true;
        followUntilRef.current = 0;
      };

      const requestTick = (): void => {
        if (rafId === null) {
          rafId = requestAnimationFrame(tick);
        }
      };

      // Single rAF loop. While within the follow window and not
      // detached, pin the viewport to the bottom and report
      // isAtBottom=true every frame. Otherwise settle on whatever the
      // DOM says. Scheduling is edge-triggered: scroll/resize/mutation
      // events call requestTick(), and the loop self-perpetuates only
      // as long as pinning is still active.
      const tick = (): void => {
        rafId = null;
        const following =
          !userDetachedRef.current &&
          performance.now() < followUntilRef.current;

        if (following) {
          if (!atBottomStrict() && el.scrollHeight > el.clientHeight) {
            el.scrollTo({ top: el.scrollHeight, behavior: "instant" });
          }
          setIsAtBottom(true);
          requestTick();
          return;
        }

        setIsAtBottom(atBottomStrict());
      };

      scrollImplRef.current = (behavior = "auto") => {
        userDetachedRef.current = false;
        followUntilRef.current = performance.now() + FOLLOW_SETTLE_MS;
        if (el.scrollHeight > el.clientHeight) {
          el.scrollTo({ top: el.scrollHeight, behavior });
        }
        setIsAtBottom(true);
        requestTick();
      };

      const onWheel = (e: WheelEvent) => {
        if (
          e.deltaY < 0 &&
          canScrollUp() &&
          !innerScrollWillConsumeUpward(e.target)
        ) {
          detach();
        }
      };

      const onTouchStart = (e: TouchEvent) => {
        touchStartY = e.touches[0]?.clientY ?? 0;
      };

      const onTouchMove = (e: TouchEvent) => {
        const y = e.touches[0]?.clientY ?? 0;
        // Finger moves DOWN on the screen = content scrolls UP.
        if (
          y - touchStartY > TOUCH_MOVE_THRESHOLD_PX &&
          canScrollUp() &&
          !innerScrollWillConsumeUpward(e.target)
        ) {
          detach();
        }
      };

      const onScroll = () => {
        const scrollTop = el.scrollTop;
        const clientWidth = el.clientWidth;
        const clientHeight = el.clientHeight;
        const sizeChanged =
          clientWidth !== lastClientWidth || clientHeight !== lastClientHeight;

        const delta = scrollTop - lastScrollTop;
        const distanceNow = distanceFromBottom();

        // Viewport resizes can clamp scrollTop and produce spurious
        // direction signals. Only flip intent on deliberate scrolls.
        if (sizeChanged) {
          upwardAccumulator = 0;
        } else if (delta > 0) {
          // Downward: reset the upward accumulator, and re-attach when
          // the user has scrolled back within range of the bottom.
          upwardAccumulator = 0;
          if (
            userDetachedRef.current &&
            distanceNow <= RE_ATTACH_THRESHOLD_PX
          ) {
            userDetachedRef.current = false;
            extendFollow();
          }
        } else if (delta < 0 && !userDetachedRef.current) {
          // Upward: sum across events. Middle-click autoscroll and
          // some trackpads deliver 1px-per-event scrolls that each
          // slip under a per-event threshold; summing catches them.
          //
          // Count distance-from-bottom growth, not raw scrollTop
          // delta. When content above collapses (reasoning panels
          // auto-closing after streaming, tool outputs auto-hiding),
          // browsers scroll-anchor to keep visible content stable:
          // scrollTop decreases but scrollHeight decreases by the
          // same amount, so distance is unchanged. Those layout-
          // induced deltas must not flip user intent.
          const distanceDelta = distanceNow - lastDistanceFromBottom;
          if (distanceDelta > 0) {
            upwardAccumulator += distanceDelta;
            if (upwardAccumulator >= UPWARD_DETACH_THRESHOLD_PX) {
              detach();
              upwardAccumulator = 0;
            }
          }
        }

        lastScrollTop = scrollTop;
        lastClientWidth = clientWidth;
        lastClientHeight = clientHeight;
        lastDistanceFromBottom = distanceNow;
        requestTick();
      };

      const resizeObserver = new ResizeObserver(() => {
        extendFollow();
        requestTick();
      });

      const mutationObserver = new MutationObserver(() => {
        extendFollow();
        requestTick();
      });

      const onViewportResize = () => {
        extendFollow();
        requestTick();
      };

      // Fresh attach always starts pinned. `userDetachedRef` survives
      // ref rebinds (it's hook-scoped), so if the viewport element is
      // ever unmounted and remounted without an AUI lifecycle event
      // (e.g. a parent layout refactor that remounts the viewport),
      // a prior detach would silently disable auto-follow for the
      // rest of the session.
      userDetachedRef.current = false;

      // Pin to bottom when the ref first attaches. Covers the case
      // where `thread.initialize` fires before the ref is bound.
      extendFollow();
      if (el.scrollHeight > el.clientHeight) {
        el.scrollTo({ top: el.scrollHeight, behavior: "instant" });
      }
      setIsAtBottom(true);
      requestTick();

      resizeObserver.observe(el);
      mutationObserver.observe(el, {
        childList: true,
        subtree: true,
        characterData: true,
        // Layout-affecting attributes only. Excludes `style`, which
        // elements may mutate in response to viewport state (feedback
        // loop). `class` catches Tailwind show/hide; `hidden` /
        // `aria-hidden` / `aria-expanded` / `data-state` catch Radix
        // and native collapsible toggles that change scrollHeight
        // without triggering the viewport ResizeObserver.
        attributes: true,
        attributeFilter: [
          "class",
          "hidden",
          "aria-hidden",
          "aria-expanded",
          "data-state",
        ],
      });
      el.addEventListener("wheel", onWheel, { passive: true });
      el.addEventListener("touchstart", onTouchStart, { passive: true });
      el.addEventListener("touchmove", onTouchMove, { passive: true });
      el.addEventListener("scroll", onScroll, { passive: true });
      // ResizeObserver above covers browser-window resizes (they resize
      // the viewport element). visualViewport.resize is the only signal
      // for iOS software-keyboard changes, where the visual viewport
      // shrinks without the viewport element's clientHeight changing.
      window.visualViewport?.addEventListener("resize", onViewportResize);

      return () => {
        if (rafId !== null) {
          cancelAnimationFrame(rafId);
          rafId = null;
        }
        resizeObserver.disconnect();
        mutationObserver.disconnect();
        el.removeEventListener("wheel", onWheel);
        el.removeEventListener("touchstart", onTouchStart);
        el.removeEventListener("touchmove", onTouchMove);
        el.removeEventListener("scroll", onScroll);
        window.visualViewport?.removeEventListener("resize", onViewportResize);
        scrollImplRef.current = () => {
          /* no viewport mounted */
        };
      };
    },
    [setIsAtBottom],
  );

  // Thread lifecycle moments that always pin to the bottom, regardless
  // of prior detach state. "auto" respects CSS smooth scrolling for
  // runStart so new turns glide in; "instant" snaps for load/switch
  // where any animation would just be wasted motion.
  const pinToBottom = useCallback((behavior: ScrollBehavior) => {
    userDetachedRef.current = false;
    scrollImplRef.current(behavior);
  }, []);

  useAuiEvent("thread.runStart", () => pinToBottom("auto"));
  useAuiEvent("thread.initialize", () => pinToBottom("instant"));
  useAuiEvent("threadListItem.switchedTo", () => pinToBottom("instant"));

  const ref = useCallback<RefCallback<HTMLElement>>(
    (el) => {
      if (cleanupRef.current) {
        cleanupRef.current();
        cleanupRef.current = null;
      }
      if (el) {
        cleanupRef.current = attach(el);
      }
    },
    [attach],
  );

  const context = useMemo<AutoScrollContextValue>(
    () => ({ scrollToBottom, getIsAtBottom, subscribe }),
    [scrollToBottom, getIsAtBottom, subscribe],
  );

  return { ref, context };
}
