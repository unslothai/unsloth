// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { useChatPreferencesStore } from "@/features/chat/stores/chat-preferences-store";
import { useAui, useAuiEvent } from "@assistant-ui/react";
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
 * Replaces assistant-ui's autoscroll, whose observers write `isAtBottom` on every layout change
 * and race our correction. A rAF loop pins until a follow deadline; upward intent detaches, and
 * scrolling back within 24px of the bottom re-attaches.
 */

// 2px, not 1: HiDPI subpixel rounding can leave a fractional gap.
const AT_BOTTOM_THRESHOLD_PX = 2;
const RE_ATTACH_THRESHOLD_PX = 24;
const TOUCH_MOVE_THRESHOLD_PX = 4;
// Summed, not per event, so 1px-per-event sources still detach.
const UPWARD_DETACH_THRESHOLD_PX = 2;
// Pin window; extends on every resize/mutation, settling this long after the last change.
const FOLLOW_SETTLE_MS = 600;
// Content can grow with no mutation or border-box resize (image decode, fonts, KaTeX), so a
// quiet pinned frame re-checks layout on this timer.
const SETTLE_CHECK_MS = 100;
// Absorbs sub-frame transients (shiki re-renders); larger shrinks are real removals and re-pin.
const STABILIZER_MAX_PX = 64;

export type ScrollToBottom = (behavior?: ScrollBehavior) => void;

type AutoScrollContextValue = {
  scrollToBottom: ScrollToBottom;
  getIsAtBottom: () => boolean;
  subscribe: (listener: () => void) => () => void;
  /** Detach as if scrolled up: composer growth must not let observer pins shove the chat up. */
  detachFromBottom: () => void;
  /** Shift by content inserted above; called on every widening commit, zero included. */
  adjustForContentInsertedAbove: (deltaPx: number) => void;
};

const noopContext: AutoScrollContextValue = {
  scrollToBottom: () => {
    /* no viewport mounted */
  },
  getIsAtBottom: () => true,
  subscribe: () => () => {
    /* no-op */
  },
  detachFromBottom: () => {
    /* no viewport mounted */
  },
  adjustForContentInsertedAbove: () => {
    /* no viewport mounted */
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

export function useAdjustForContentInsertedAbove(): (deltaPx: number) => void {
  return useContext(AutoScrollContext).adjustForContentInsertedAbove;
}

/** Opening a collapsible near the bottom must grow it downward, not pin and shove the header. */
export function useDetachThreadFromBottom(): () => void {
  return useContext(AutoScrollContext).detachFromBottom;
}

export function useIsThreadAtBottom(): boolean {
  const ctx = useContext(AutoScrollContext);
  return useSyncExternalStore(ctx.subscribe, ctx.getIsAtBottom, () => true);
}

export function useIntentAwareAutoScroll(): {
  ref: RefCallback<HTMLElement>;
  context: AutoScrollContextValue;
} {
  const aui = useAui();
  const cleanupRef = useRef<(() => void) | null>(null);

  const userDetachedRef = useRef(false);
  const followUntilRef = useRef(0);
  const runStartedHereRef = useRef(false);
  // runEnd fires before the last chunk commits, so the hold outlives it briefly.
  const runEndAtRef = useRef(Number.NEGATIVE_INFINITY);

  const isAtBottomRef = useRef(true);
  const listenersRef = useRef<Set<() => void>>(new Set());

  const scrollImplRef = useRef<ScrollToBottom>(() => {
    /* no viewport mounted */
  });
  const detachImplRef = useRef<() => void>(() => {
    /* no viewport mounted */
  });
  const adjustImplRef = useRef<(deltaPx: number) => void>(() => {
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
    runStartedHereRef.current = false;
    scrollImplRef.current(behavior);
  }, []);

  const detachFromBottom = useCallback(() => {
    detachImplRef.current();
  }, []);

  const adjustForContentInsertedAbove = useCallback((deltaPx: number) => {
    adjustImplRef.current(deltaPx);
  }, []);

  const attach = useCallback(
    (el: HTMLElement, isRebind: boolean) => {
      let rafId: number | null = null;
      let settleTimer: number | null = null;
      let settleCheckDue = false;
      let layoutChanged = true;
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

      // A gesture with nothing above must not detach, or auto-follow stays dead for the session.
      const canScrollUp = (): boolean => el.scrollTop > 0;

      // An inner scroller with room above consumes the upward delta, so it must not detach.
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

      // Stabilizer state lives in this closure so it resets when the viewport remounts.
      let stabilizerPx = 0;
      let maxContentHeight = 0;

      const releaseStabilizer = (): void => {
        if (stabilizerPx === 0) {
          return;
        }
        stabilizerPx = 0;
        el.style.removeProperty("--aui-scroll-stabilizer");
      };

      const extendFollow = (): void => {
        if (userDetachedRef.current) {
          return;
        }
        followUntilRef.current = performance.now() + FOLLOW_SETTLE_MS;
      };

      const holdStill = (): boolean =>
        runStartedHereRef.current &&
        !useChatPreferencesStore.getState().autoScrollWhileGenerating &&
        (aui.thread().getState().isRunning ||
          performance.now() - runEndAtRef.current < FOLLOW_SETTLE_MS);

      const holdCeiling = (): number | null => {
        const rows = el.querySelectorAll<HTMLElement>("[data-role]");
        let reply: HTMLElement | null = null;
        let user: HTMLElement | null = null;
        for (let i = rows.length - 1; i >= 0 && !user; i--) {
          if (rows[i].dataset.role === "user") {
            user = rows[i];
          } else if (rows[i].dataset.role === "assistant") {
            reply = rows[i];
          }
        }
        if (!user || !reply) {
          return null;
        }
        const origin =
          el.getBoundingClientRect().top -
          el.scrollTop +
          (Number.parseFloat(getComputedStyle(el).paddingTop) || 0);
        return Math.max(
          0,
          user.getBoundingClientRect().top - origin,
          reply.getBoundingClientRect().top - origin - el.clientHeight / 2,
        );
      };

      const clearSettleCheck = (): void => {
        if (settleTimer !== null) {
          clearTimeout(settleTimer);
          settleTimer = null;
        }
        settleCheckDue = false;
      };

      let parked = false;

      const detach = (): void => {
        parked = false;
        userDetachedRef.current = true;
        followUntilRef.current = 0;
        // Hygiene: `following` checks userDetached first, so a queued check cannot re-pin anyway.
        clearSettleCheck();
        // Drop residual stabilizer padding once the user scrolls up so the bottom stays flush.
        releaseStabilizer();
        maxContentHeight = el.scrollHeight;
      };

      const parkIfHeld = (): boolean => {
        if (userDetachedRef.current || !holdStill()) {
          return false;
        }
        const ceiling = holdCeiling();
        if (ceiling === null || el.scrollHeight - el.clientHeight < ceiling) {
          return false;
        }
        el.scrollTo({ top: ceiling, behavior: "instant" });
        detach();
        parked = true;
        return true;
      };

      const requestTick = (): void => {
        if (rafId === null) {
          rafId = requestAnimationFrame(tick);
        }
      };

      // A quiet pinned frame hands the window to this self-rearming timer. See SETTLE_CHECK_MS.
      const scheduleSettleCheck = (): void => {
        if (settleTimer !== null) {
          return;
        }
        const remaining = followUntilRef.current - performance.now();
        if (remaining <= 0) {
          return;
        }
        settleTimer = window.setTimeout(
          () => {
            settleTimer = null;
            // The window may have closed by then; grant that frame one last follow pass.
            settleCheckDue = true;
            requestTick();
          },
          Math.min(SETTLE_CHECK_MS, remaining),
        );
      };

      // Edge-triggered rAF loop: re-arms only while layout is still moving.
      const tick = (): void => {
        rafId = null;
        const settling = settleCheckDue;
        settleCheckDue = false;
        // Park first so a frame without an observer record can't overshoot.
        parkIfHeld();
        const following =
          !userDetachedRef.current &&
          (settling || performance.now() < followUntilRef.current);

        if (following) {
          const pinned = atBottomStrict();
          if (!pinned && el.scrollHeight > el.clientHeight) {
            el.scrollTo({ top: el.scrollHeight, behavior: "instant" });
          }
          setIsAtBottom(true);
          if (layoutChanged || !pinned) {
            layoutChanged = false;
            requestTick();
            return;
          }
          scheduleSettleCheck();
          return;
        }

        setIsAtBottom(atBottomStrict());
      };

      scrollImplRef.current = (behavior = "auto") => {
        parked = false;
        userDetachedRef.current = false;
        followUntilRef.current = performance.now() + FOLLOW_SETTLE_MS;
        if (el.scrollHeight > el.clientHeight) {
          el.scrollTo({ top: el.scrollHeight, behavior });
        }
        setIsAtBottom(true);
        requestTick();
      };

      detachImplRef.current = () => {
        detach();
        requestTick();
      };

      // Only matters while DETACHED: when following, pinIfFollowing already lands on the same pixel.
      // `behavior: "instant"` is required because the viewport has `scroll-smooth`.
      adjustImplRef.current = (deltaPx: number) => {
        if (!userDetachedRef.current) {
          return;
        }
        if (deltaPx !== 0) {
          el.scrollTo({ top: el.scrollTop + deltaPx, behavior: "instant" });
        }
        // Resync lastScrollTop even with no write: native anchoring may have moved scrollTop and its
        // scroll event would otherwise read as a downward scroll that re-attaches the user.
        lastScrollTop = el.scrollTop;
        lastDistanceFromBottom = distanceFromBottom();
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

        // Resizes can clamp scrollTop and fake direction; only deliberate scrolls flip intent.
        if (sizeChanged) {
          upwardAccumulator = 0;
        } else if (delta > 0) {
          upwardAccumulator = 0;
          if (
            userDetachedRef.current &&
            distanceNow <= RE_ATTACH_THRESHOLD_PX &&
            !holdStill()
          ) {
            userDetachedRef.current = false;
            extendFollow();
          }
        } else if (delta < 0 && !userDetachedRef.current) {
          // Count distance-from-bottom growth, not scrollTop: anchoring moves both when content collapses.
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

      // Keeps scrollHeight monotonic during the follow window: a finalizing code block briefly shrinks
      // and the browser caps scrollTop. The shortfall goes into `--aui-scroll-stabilizer` padding.
      const stabilize = (): number => {
        const sh = el.scrollHeight;
        const currentContent = sh - stabilizerPx;
        const followActive =
          !userDetachedRef.current &&
          performance.now() < followUntilRef.current;
        if (!followActive) {
          // Outside the follow window: stop adjusting but keep maxContentHeight current.
          maxContentHeight = currentContent;
          return sh;
        }
        if (currentContent > maxContentHeight) {
          maxContentHeight = currentContent;
        }
        const shrink = maxContentHeight - currentContent;
        // Large shrinks are intentional removals; release and re-anchor rather than pad a gap.
        if (shrink > STABILIZER_MAX_PX) {
          maxContentHeight = currentContent;
          if (stabilizerPx !== 0) {
            stabilizerPx = 0;
            el.style.removeProperty("--aui-scroll-stabilizer");
          }
          return currentContent;
        }
        const needed = Math.max(0, shrink);
        if (needed !== stabilizerPx) {
          stabilizerPx = needed;
          el.style.setProperty("--aui-scroll-stabilizer", `${stabilizerPx}px`);
        }
        return currentContent + stabilizerPx;
      };

      // Observer callbacks run after layout and before paint, so this lands in the same frame.
      const pinIfFollowing = (scrollHeight: number): void => {
        if (userDetachedRef.current) {
          return;
        }
        if (performance.now() >= followUntilRef.current) {
          return;
        }
        if (scrollHeight <= el.clientHeight) {
          return;
        }
        el.scrollTo({ top: scrollHeight, behavior: "instant" });
      };

      // Order matters: extend first so the stabilizer sees follow active, then stabilize, then pin.
      const onLayoutChange = (): void => {
        layoutChanged = true;
        if (!parkIfHeld()) {
          extendFollow();
        }
        const scrollHeight = stabilize();
        pinIfFollowing(scrollHeight);
        requestTick();
      };

      const resizeObserver = new ResizeObserver(onLayoutChange);
      const mutationObserver = new MutationObserver(onLayoutChange);
      const onViewportResize = onLayoutChange;

      // Only a fresh attach pins: React re-runs the composed ref on unrelated renders for the same element.
      if (!isRebind) {
        userDetachedRef.current = false;

        // Pin on first attach, covering thread.initialize firing before the ref is bound.
        extendFollow();
        if (el.scrollHeight > el.clientHeight) {
          el.scrollTo({ top: el.scrollHeight, behavior: "instant" });
        }
        setIsAtBottom(true);
      }
      requestTick();

      // Border box: the stabilizer's padding writes would otherwise echo back as resizes.
      resizeObserver.observe(el, { box: "border-box" });
      mutationObserver.observe(el, {
        childList: true,
        subtree: true,
        characterData: true,
        // Excludes `style` to avoid a feedback loop; `class` and the rest catch collapsibles.
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
      // visualViewport.resize is the only signal for iOS software-keyboard changes.
      window.visualViewport?.addEventListener("resize", onViewportResize);

      const unsubscribePreferences = useChatPreferencesStore.subscribe(
        (state, prev) => {
          if (
            !parked ||
            !state.autoScrollWhileGenerating ||
            prev.autoScrollWhileGenerating ||
            !aui.thread().getState().isRunning
          ) {
            return;
          }
          parked = false;
          userDetachedRef.current = false;
          extendFollow();
          requestTick();
        },
      );

      return () => {
        unsubscribePreferences();
        if (rafId !== null) {
          cancelAnimationFrame(rafId);
          rafId = null;
        }
        clearSettleCheck();
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
        detachImplRef.current = () => {
          /* no viewport mounted */
        };
        adjustImplRef.current = () => {
          /* no viewport mounted */
        };
      };
    },
    [aui, setIsAtBottom],
  );

  // Lifecycle pins ignore detach: "auto" glides on runStart, "instant" on load/switch.
  const pinToBottom = useCallback((behavior: ScrollBehavior) => {
    userDetachedRef.current = false;
    scrollImplRef.current(behavior);
  }, []);

  useAuiEvent("thread.runStart", () => {
    runStartedHereRef.current = true;
    runEndAtRef.current = Number.NEGATIVE_INFINITY;
    pinToBottom("auto");
  });
  useAuiEvent("thread.runEnd", () => {
    runEndAtRef.current = performance.now();
  });
  useAuiEvent("thread.initialize", () => {
    runStartedHereRef.current = false;
    pinToBottom("instant");
  });
  useAuiEvent("threadListItem.switchedTo", () => {
    runStartedHereRef.current = false;
    pinToBottom("instant");
  });

  const lastElRef = useRef<HTMLElement | null>(null);
  const ref = useCallback<RefCallback<HTMLElement>>(
    (el) => {
      if (cleanupRef.current) {
        cleanupRef.current();
        cleanupRef.current = null;
      }
      if (el) {
        const isRebind = lastElRef.current === el;
        lastElRef.current = el;
        cleanupRef.current = attach(el, isRebind);
      }
      // On null, keep lastElRef so a rebind to the same element is recognized.
    },
    [attach],
  );

  const context = useMemo<AutoScrollContextValue>(
    () => ({
      scrollToBottom,
      getIsAtBottom,
      subscribe,
      detachFromBottom,
      adjustForContentInsertedAbove,
    }),
    [
      scrollToBottom,
      getIsAtBottom,
      subscribe,
      detachFromBottom,
      adjustForContentInsertedAbove,
    ],
  );

  return { ref, context };
}
