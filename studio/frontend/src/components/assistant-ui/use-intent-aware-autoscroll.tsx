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
 * Why not assistant-ui's built-in: `useThreadViewportAutoScroll` always
 * installs observers that write `isAtBottom` to the shared store on every
 * layout change, even with `scrollToBottomOn*` disabled. On resizes /
 * breakpoint crossings that write races our correction and the
 * scroll-to-bottom button flickers depending on observer ordering.
 *
 * Strategy:
 *   - Own `isAtBottom` as local state (via `useIsThreadAtBottom`); nobody
 *     reads upstream's store.
 *   - Drive the viewport via a single rAF loop with a follow deadline
 *     (`followUntilRef`). Any signal that can invalidate bottom alignment
 *     extends the deadline; the loop pins and reports `isAtBottom=true`
 *     until it expires, then settles on DOM observation.
 *   - Detect user intent (wheel up, swipe up, scroll direction) to detach.
 *     While detached, resize/mutation don't extend the deadline. Re-attach
 *     when the user scrolls down within 24px of the bottom.
 */

// 2px, not 1: HiDPI subpixel rounding can leave a fractional gap that a
// 1px threshold reports as not-at-bottom.
const AT_BOTTOM_THRESHOLD_PX = 2;
const RE_ATTACH_THRESHOLD_PX = 24;
const TOUCH_MOVE_THRESHOLD_PX = 4;
// Cumulative upward movement counting as a deliberate detach. Summed (not
// per-event) so slow 1px-per-event sources (middle-click autoscroll,
// scrollbar drags, some trackpads) accumulate instead of slipping under.
const UPWARD_DETACH_THRESHOLD_PX = 2;
// Window the viewport stays pinned through layout/content races. Extends
// on every resize/mutation, so streaming keeps it pinned; settles this
// long after the last change.
const FOLLOW_SETTLE_MS = 600;
// Max stabilizer compensation. Absorbs sub-frame transients (~5-15px shiki
// re-renders, ~8px action-bar drift). Larger shrinks are intentional
// content removals (delete, regenerate clear, reasoning collapse); padding
// those over leaves empty space, so above this we release and let
// autoscroll re-pin to the new height.
const STABILIZER_MAX_PX = 64;

export type ScrollToBottom = (behavior?: ScrollBehavior) => void;

type AutoScrollContextValue = {
  scrollToBottom: ScrollToBottom;
  getIsAtBottom: () => boolean;
  subscribe: (listener: () => void) => () => void;
  /**
   * Mark the user as detached, as if they scrolled up. Called when the
   * composer (and bottom spacer) grow: the chat is then above the new
   * bottom and observer-driven pins must not shove it up. Scrolling back
   * re-attaches; explicit pins (run start, button) still work.
   */
  detachFromBottom: () => void;
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
  const detachImplRef = useRef<() => void>(() => {
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

  const detachFromBottom = useCallback(() => {
    detachImplRef.current();
  }, []);

  const attach = useCallback(
    (el: HTMLElement, isRebind: boolean) => {
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

      // Room to scroll upward. Guards wheel/touch detach: a gesture on a
      // viewport with nothing above can't express intent to leave the
      // bottom and must not flip userDetachedRef, else later streaming
      // skips extendFollow and auto-follow stays dead for the session.
      const canScrollUp = (): boolean => el.scrollTop > 0;

      // True when a nested scrollable ancestor of the target (reasoning
      // panel, tool output) has room above and will consume the upward
      // delta before it reaches the viewport. Walk stops at the viewport,
      // so only intermediate inner scrollers count. Without this, wheel/
      // touchmove bubbling would falsely detach while reading a long
      // reasoning pane mid-stream.
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

      // Stabilizer state (see `stabilize`). In this closure so it resets
      // when the viewport remounts (Compare-pane swap, thread switch).
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

      const detach = (): void => {
        userDetachedRef.current = true;
        followUntilRef.current = 0;
        // The stabilizer only matters while pinning. Once the user
        // scrolls up, drop residual padding so the bottom stays flush on
        // return. Safe mid-content: shrinking scrollHeight can't cap their
        // scrollTop.
        releaseStabilizer();
        maxContentHeight = el.scrollHeight;
      };

      const requestTick = (): void => {
        if (rafId === null) {
          rafId = requestAnimationFrame(tick);
        }
      };

      // Single rAF loop. Within the follow window and not detached, pin to
      // bottom and report isAtBottom=true each frame; otherwise settle on
      // the DOM. Edge-triggered: scroll/resize/mutation call requestTick(),
      // and the loop self-perpetuates only while pinning is active.
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

      // Programmatic detach (see detachFromBottom). Same as scrolling up;
      // the tick refresh updates isAtBottom.
      detachImplRef.current = () => {
        detach();
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
          // Upward: sum across events to catch 1px-per-event sources
          // (middle-click autoscroll, some trackpads) that slip under a
          // per-event threshold. Count distance-from-bottom growth, not
          // raw scrollTop delta: when content above collapses, browsers
          // scroll-anchor so scrollTop and scrollHeight drop together and
          // distance is unchanged. Those layout deltas must not flip intent.
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

      // Scroll stabilizer.
      //
      // Problem: when a trailing code block finalizes (shiki re-renders the
      // <pre> with highlight spans), its height briefly dips then recovers.
      // The dip shrinks `scrollHeight`, so the browser synchronously caps
      // `scrollTop` to the new max — a one-frame upward jump, then a
      // "snap back". Re-scrolling can't help: once scrollHeight drops the
      // cap has happened and scrollTop can't exceed the new max.
      //
      // Fix: keep `scrollHeight` monotonic across the follow window. Track
      // max content height (scrollHeight minus our padding) and write any
      // shortfall into CSS var `--aui-scroll-stabilizer`, read by the
      // viewport's padding-bottom. A 5px shrink grows padding 5px so the
      // browser sees no scrollHeight change. Padding shrinks back to zero
      // as content grows past its prior high-water mark.
      //
      // Self-contained: lives on the viewport element via a CSS variable,
      // touching no other UI.
      //
      // Returns the post-adjustment scrollHeight so one layout read per
      // observer callback feeds both stabilization and pinning.
      const stabilize = (): number => {
        const sh = el.scrollHeight;
        const currentContent = sh - stabilizerPx;
        const followActive =
          !userDetachedRef.current &&
          performance.now() < followUntilRef.current;
        if (!followActive) {
          // Outside the follow window: stop adjusting but keep
          // maxContentHeight current so the next session isn't stale.
          maxContentHeight = currentContent;
          return sh;
        }
        if (currentContent > maxContentHeight) {
          maxContentHeight = currentContent;
        }
        const shrink = maxContentHeight - currentContent;
        // Large shrinks (over STABILIZER_MAX_PX) are intentional removals
        // (delete, regenerate clear, reasoning collapse). Compensating
        // would leave an empty gap, so release the stabilizer and rebase
        // the high-water mark; the pinIfFollowing call below re-anchors to
        // the new bottom.
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
          el.style.setProperty(
            "--aui-scroll-stabilizer",
            `${stabilizerPx}px`,
          );
        }
        return currentContent + stabilizerPx;
      };

      // Synchronous pin-to-bottom. Observer callbacks run after layout,
      // before paint, so this scrollTo composites in the same frame as the
      // triggering mutation.
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

      // All three layout-change signals fan in here. Order matters: extend
      // first so the stabilizer sees follow as active; stabilize before
      // pinning so we scroll to the post-adjustment scrollHeight.
      const onLayoutChange = (): void => {
        extendFollow();
        const scrollHeight = stabilize();
        pinIfFollowing(scrollHeight);
        requestTick();
      };

      const resizeObserver = new ResizeObserver(onLayoutChange);
      const mutationObserver = new MutationObserver(onLayoutChange);
      const onViewportResize = onLayoutChange;

      // Fresh attach always starts pinned. Rebinds to the SAME element must
      // not pin or reset detach state: the Viewport's composed ref identity
      // changes on re-render, so React re-runs the ref (null, then same
      // element) on unrelated renders (composer resizes); pinning would
      // yank the chat to bottom each time. Observers are re-installed either
      // way.
      if (!isRebind) {
        userDetachedRef.current = false;

        // Pin on first attach, covering thread.initialize firing before the
        // ref is bound.
        extendFollow();
        if (el.scrollHeight > el.clientHeight) {
          el.scrollTo({ top: el.scrollHeight, behavior: "instant" });
        }
        setIsAtBottom(true);
      }
      requestTick();

      // Observe the border box, not content box. The stabilizer writes
      // padding-bottom (shrinks the content box); observing that would echo
      // every adjustment back as a resize into onLayoutChange. Border-box is
      // stable through padding changes but still tracks parent resizes.
      resizeObserver.observe(el, { box: "border-box" });
      mutationObserver.observe(el, {
        childList: true,
        subtree: true,
        characterData: true,
        // Layout-affecting attributes only. Excludes `style` (elements may
        // mutate it in response to viewport state → feedback loop). `class`
        // catches Tailwind show/hide; the rest catch Radix/native
        // collapsibles that change scrollHeight without a viewport resize.
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
      // ResizeObserver covers window resizes. visualViewport.resize is the
      // only signal for iOS software-keyboard changes, where the visual
      // viewport shrinks without the element's clientHeight changing.
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
        detachImplRef.current = () => {
          /* no viewport mounted */
        };
      };
    },
    [setIsAtBottom],
  );

  // Thread lifecycle moments that always pin, regardless of detach state.
  // "auto" respects CSS smooth scroll for runStart (new turns glide in);
  // "instant" snaps for load/switch where animation is wasted.
  const pinToBottom = useCallback((behavior: ScrollBehavior) => {
    userDetachedRef.current = false;
    scrollImplRef.current(behavior);
  }, []);

  useAuiEvent("thread.runStart", () => pinToBottom("auto"));
  useAuiEvent("thread.initialize", () => pinToBottom("instant"));
  useAuiEvent("threadListItem.switchedTo", () => pinToBottom("instant"));

  const lastElRef = useRef<HTMLElement | null>(null);
  const ref = useCallback<RefCallback<HTMLElement>>(
    (el) => {
      if (cleanupRef.current) {
        cleanupRef.current();
        cleanupRef.current = null;
      }
      if (el) {
        // Same-element rebind vs a genuinely new element, see attach().
        const isRebind = lastElRef.current === el;
        lastElRef.current = el;
        cleanupRef.current = attach(el, isRebind);
      }
      // On null, keep lastElRef so a rebind to the same element is
      // recognized; a real remount binds a different element anyway.
    },
    [attach],
  );

  const context = useMemo<AutoScrollContextValue>(
    () => ({ scrollToBottom, getIsAtBottom, subscribe, detachFromBottom }),
    [scrollToBottom, getIsAtBottom, subscribe, detachFromBottom],
  );

  return { ref, context };
}
