// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import { AuiProvider, useAui, useAuiState } from "@assistant-ui/react";
import { useT } from "@/i18n";
import { MessageByIndexProvider } from "./stale-safe-scopes";
import { type FC, type ReactElement, type RefObject, memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createRowNotificationGate } from "./row-notification-gate";
import { findAgentTextMatches, type AgentFindMatch } from "./agent-history-search";
import { notifyFindTargets, registerFindTarget, type FindTargetResult } from "@/features/find-in-page/lib/find-targets";
import { AGENT_DOM_TARGETS, AGENT_FIXED_VIEWPORT_DOM_TARGET, AGENT_FIXED_VIEWPORT_MAX_ROWS, AgentDomCostIndex, agentBudgetWindow, agentPhysicalNeighborWindow, agentWheelPrefetchTarget, needsAgentBudgetRebalance } from "./agent-history-dom-budget";
import { AGENT_DEFAULT_ROWS, AgentHeightIndex, agentIndexAtScrollPosition, agentViewportCoverageRecoveryTarget, agentWindowAtIndex, agentWindowOnAppend, latestAgentWindow, nextProgressiveAgentWindow, type AgentScrollWindow } from "./agent-history-scroll-window";
import { agentNativeThumbBottomSpacerPx, agentNativeThumbLiveTarget, beginAgentNativeThumbDrag, finishAgentNativeThumbDrag, resolveAgentNativeThumbRelease, type AgentNativeThumbDrag } from "./agent-history-native-thumb";
import { useAdjustForContentInsertedAbove, useDetachThreadFromBottom, useGetThreadBottomFollowIntent, useNavigateThreadViewport, useScrollThreadToBottom } from "./use-intent-aware-autoscroll";
import {
  allowsAgentReaderScrollIntent,
  allowsRecentAgentBoundaryReconcile,
  allowsPassiveAgentScroll,
  completeAgentHistoryReveal,
  consumeAgentReaderScrollIntent,
  initialAgentHistoryNavigation,
  initialAgentReaderScrollIntent,
  interruptAgentHistoryReveal,
  markAgentHistoryRevealPositioned,
  requestAgentHistoryReveal,
  requestAgentReaderScrollIntent,
} from "./agent-history-navigation";
import type { AgentHistoryNavigation } from "./agent-history-navigation";

/**
 * Experimental opt-in renderer. All messages remain in the assistant-ui store
 * and persisted transcript; only the mounted DOM is bounded.
 * Search / select-all parity MUST be addressed before release.
 */
export const AgentHistoryScrollMessages: FC<{
  renderMessage: () => ReactElement;
  resetKey: string | undefined;
  viewportRef: RefObject<HTMLElement | null>;
  windowRows?: number;
  domLevel?: "low" | "medium" | "high";
  fixedRowsStrict?: boolean;
  /** Fixed adapts short-turn viewport coverage but caps rows and estimated DOM. */
  fixedViewportAdaptive?: boolean;
}> = memo(function AgentHistoryScrollMessages({
  renderMessage, resetKey, viewportRef, windowRows = AGENT_DEFAULT_ROWS, domLevel, fixedRowsStrict = false, fixedViewportAdaptive = false,
}) {
  const t = useT();
  const count = useAuiState(({ thread }) => thread.messages.length);
  const aui = useAui();
  const gate = useMemo(() => createRowNotificationGate(aui), [aui]);
  const heights = useMemo(() => new AgentHeightIndex(count), [resetKey]);
  heights.resize(count);
  const costs = useMemo(() => new AgentDomCostIndex(count), [resetKey]);
  costs.resize(count);
  // Fixed-low no longer means a permanent five-message viewport. It can grow
  // until the viewport is filled, but never exceed 32 mounted messages or the
  // same conservative 8k-descendant budget as Dynamic.
  const budget = domLevel ? AGENT_DOM_TARGETS[domLevel]
    : fixedViewportAdaptive ? AGENT_FIXED_VIEWPORT_DOM_TARGET : null;
  const budgetRows = fixedViewportAdaptive ? AGENT_FIXED_VIEWPORT_MAX_ROWS
    : domLevel === "low" ? 5 : domLevel === "medium" ? 16 : 32;
  const jumpBottom = useScrollThreadToBottom();
  const getIsFollowingBottom = useGetThreadBottomFollowIntent();
  const correctAnchor = useAdjustForContentInsertedAbove();
  const detachFromBottom = useDetachThreadFromBottom();
  const navigateViewport = useNavigateThreadViewport();
  const following = useRef(true);
  const countRef = useRef(count);
  countRef.current = count;
  const animation = useRef<number | null>(null);
  const topSpacer = useRef<HTMLDivElement>(null);
  const bottomSpacer = useRef<HTMLDivElement>(null);
  const anchor = useRef<{ index: number; top: number } | null>(null);
  const jump = useRef<"top" | "bottom" | null>(null);
  // Native scrollbar gestures temporarily own the physical viewport position.
  const nativeThumbDrag = useRef<AgentNativeThumbDrag | null>(null);
  // Drag-scoped virtual document extent. The unmounted suffix compensates
  // ledger corrections so Chromium does not resize its native thumb mid-drag.
  const nativeThumbVirtualTotal = useRef<number | null>(null);
  const staged = useRef<{ destination: AgentScrollWindow; target: number } | null>(null);
  const pendingPassiveReveal = useRef<number | null>(null);
  // A real scroll received during staging must be sampled after the last
  // window commit, never allowed to abort staging with only one row mounted.
  const deferredScroll = useRef(false);
  const lastBlankRecoveryEpoch = useRef(-1);
  // A physical-wheel epoch may require limited further reconciliation after
  // height-ledger corrections. Never run an unbounded layout-recovery loop.
  const coverageRecovery = useRef({ epoch: -1, attempts: 0 });
  const wheelCoverageTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const wheelIdleReady = useRef(false);
  // At most one follow-up adjacent-row admission after each physical wheel
  // event; a second unknown/heavy tool row cannot bypass the cost index.
  const wheelPrefetchPending = useRef<{ at: number; deltaY: number } | null>(null);
  const [state, setState] = useState(() => ({
    key: resetKey, count, windowRows, domLevel, fixedRowsStrict, fixedViewportAdaptive,
    range: budget !== null ? agentBudgetWindow(count, count - 1, costs, budget, budgetRows)
      : latestAgentWindow(count, windowRows),
  }));
  const visibleIndex = useRef(Math.max(0, count - 1));
  const lastWindowRows = useRef(windowRows);
  const lastDomLevel = useRef(domLevel);
  const lastFixedRowsStrict = useRef(fixedRowsStrict);
  const lastFixedAdaptive = useRef(fixedViewportAdaptive);
  const [heightRevision, setHeightRevision] = useState(0);
  const [searchHit, setSearchHit] = useState(-1);
  const [revealRevision, setRevealRevision] = useState(0);
  const searchHitRef = useRef(-1);
  const navigation = useRef<AgentHistoryNavigation>(initialAgentHistoryNavigation());
  const readerIntent = useRef(initialAgentReaderScrollIntent());
  const pendingScrollEpoch = useRef<number | null>(null);
  const lastReaderGestureAt = useRef(Number.NEGATIVE_INFINITY);
  // Old reader gestures must not carry over when the user changes threads.
  useLayoutEffect(() => {
    // A previous chat's detached window must not survive a thread switch.
    following.current = true;
    visibleIndex.current = Math.max(0, countRef.current - 1);
    readerIntent.current = initialAgentReaderScrollIntent();
    lastReaderGestureAt.current = Number.NEGATIVE_INFINITY;
    pendingScrollEpoch.current = null;
    navigation.current = initialAgentHistoryNavigation();
    nativeThumbDrag.current = null;
    nativeThumbVirtualTotal.current = null;
    staged.current = null;
    pendingPassiveReveal.current = null;
    deferredScroll.current = false;
    lastBlankRecoveryEpoch.current = -1;
    coverageRecovery.current = { epoch: -1, attempts: 0 };
    wheelPrefetchPending.current = null;
    if (wheelCoverageTimer.current !== null) clearTimeout(wheelCoverageTimer.current);
    wheelCoverageTimer.current = null;
    wheelIdleReady.current = false;
  }, [resetKey]);
  const revealFrame = useRef<number | null>(null);
  useEffect(() => () => {
    if (revealFrame.current !== null) cancelAnimationFrame(revealFrame.current);
  }, []);
  const heightFrame = useRef<number | null>(null);
  const requestHeightRevision = useCallback(() => {
    if (heightFrame.current !== null) return;
    heightFrame.current = requestAnimationFrame(() => {
      heightFrame.current = null;
      setHeightRevision(v => v + 1);
    });
  }, []);
  useEffect(() => () => {
    if (heightFrame.current !== null) cancelAnimationFrame(heightFrame.current);
  }, []);

  // Preserve progressive staging and reader-intent ownership; only the
  // selected window policy changes under an explicit Low/High preference.
  const budgetWindow = (index: number) =>
    budget === null ? null : agentBudgetWindow(count, index, costs, budget, budgetRows);
  const latestWindow = () => budgetWindow(count - 1) ?? latestAgentWindow(count, windowRows);
  const visibleWindow = () => budgetWindow(visibleIndex.current) ?? agentWindowAtIndex(
    count, Math.min(visibleIndex.current, Math.max(0, count - 1)), undefined, windowRows, fixedRowsStrict,
  );
  // The shared viewport owner, not our gesture cache, decides whether a new
  // streamed message belongs in the visible tail window.
  const followingOnAppend = getIsFollowingBottom();
  const range = state.key !== resetKey ? latestWindow()
    : state.domLevel !== domLevel || state.windowRows !== windowRows ||
      state.fixedRowsStrict !== fixedRowsStrict || state.fixedViewportAdaptive !== fixedViewportAdaptive
      ? followingOnAppend ? latestWindow() : visibleWindow()
      : state.count !== count
        ? budget !== null
          ? followingOnAppend ? latestWindow() : state.range
          : agentWindowOnAppend(state.range, state.count, count, followingOnAppend, windowRows, fixedRowsStrict)
        : state.range;
  if (state.key !== resetKey || state.count !== count ||
      state.windowRows !== windowRows || state.domLevel !== domLevel ||
      state.fixedRowsStrict !== fixedRowsStrict || state.fixedViewportAdaptive !== fixedViewportAdaptive) {
    setState({ key: resetKey, count, windowRows, domLevel, fixedRowsStrict, fixedViewportAdaptive, range });
  }
  const previousCountRef = useRef(count);
  useLayoutEffect(() => {
    if (previousCountRef.current === count) return;
    previousCountRef.current = count;
    if (getIsFollowingBottom()) {
      following.current = true;
      visibleIndex.current = Math.max(0, count - 1);
    }
  }, [count, getIsFollowingBottom]);
  const rangeRef = useRef<AgentScrollWindow>(range);
  rangeRef.current = range;
  // Synchronously reconcile the spacer in layout/ResizeObserver callbacks:
  // waiting for a React frame could let Chromium recalculate the scrollbar
  // track from newly mounted actual DOM heights in the meantime.
  const reconcileNativeThumbSpacer = useCallback(() => {
    const frozenTotal = nativeThumbVirtualTotal.current;
    const spacer = bottomSpacer.current;
    if (frozenTotal === null || !spacer) return;
    spacer.style.height = agentNativeThumbBottomSpacerPx(
      frozenTotal, heights.offset(rangeRef.current.end),
    ) + "px";
  }, [heights]);

  const seek = useCallback((index: number, snap?: "top" | "bottom") => {
    const viewport = viewportRef.current;
    const next = budget !== null
      ? agentBudgetWindow(countRef.current, index, costs, budget, budgetRows)
      : agentWindowAtIndex(countRef.current, index, rangeRef.current, windowRows, fixedRowsStrict);
    if (next.start === rangeRef.current.start && next.end === rangeRef.current.end) {
      if (snap === "bottom") jumpBottom("auto");
      if (snap === "top" && viewport) navigateViewport(0);
      return false;
    }
    anchor.current = null;
    const disjoint = next.end <= rangeRef.current.start ||
      next.start >= rangeRef.current.end;
    const progressive = !snap && !following.current && disjoint &&
      next.end - next.start > 1;
    const target = Math.min(Math.max(index, next.start), next.end - 1);
    const candidate = progressive ? { start: target, end: target + 1 } : next;
    staged.current = progressive ? { destination: next, target } : null;
    pendingPassiveReveal.current = disjoint && !snap && nativeThumbDrag.current === null ? target : null;
    if (viewport && !snap && nativeThumbDrag.current === null) {
      const fold = viewport.getBoundingClientRect().top;
      for (const el of viewport.querySelectorAll<HTMLElement>("[data-agent-history-row]")) {
        if (el.getBoundingClientRect().bottom > fold) {
          const at = Number(el.dataset.agentHistoryRow);
          if (at >= next.start && at < next.end) anchor.current = { index: at, top: el.getBoundingClientRect().top };
          break;
        }
      }
    }
    jump.current = snap ?? null;
    rangeRef.current = candidate;
    setState(old => ({ ...old, range: candidate }));
    return true;
  }, [jumpBottom, navigateViewport, viewportRef, windowRows, costs, budget, budgetRows, fixedRowsStrict]);

  // The first prefetch mounts a single new row. After its actual DOM cost is
  // measured, we may add just ONE more neighbor if physical coverage still
  // needs it. No loops, scrollTop writes, timer-based continuous retries or
  // multiple unknown heavy tool rows are possible.
  useEffect(() => {
    const pending = wheelPrefetchPending.current;
    if (!pending) return;
    wheelPrefetchPending.current = null;
    if (budget === null || performance.now() - pending.at > 250 ||
        staged.current !== null || pendingPassiveReveal.current !== null ||
        navigation.current.kind !== "reader" || getIsFollowingBottom()) return;
    const viewport = viewportRef.current;
    const first = viewport?.querySelector<HTMLElement>("[data-agent-history-row]");
    if (!viewport || !first) return;
    const top = viewport.getBoundingClientRect();
    const target = agentWheelPrefetchTarget(
      Math.max(-240, pending.deltaY), first.getBoundingClientRect().top,
      top.top, viewport.clientHeight, rangeRef.current.start,
    );
    if (target === null) return;
    const next = agentPhysicalNeighborWindow(
      rangeRef.current, target, countRef.current, costs, budget, budgetRows,
    );
    if (!next) return;
    rangeRef.current = next;
    setState(old => ({ ...old, range: next }));
  }, [range.start, range.end, budget, budgetRows, costs, viewportRef, getIsFollowingBottom]);

  useEffect(() => {
    const task = staged.current;
    if (!task) return;
    const frame = requestAnimationFrame(() => {
      if (staged.current !== task) return;
      const current = rangeRef.current;
      const next = nextProgressiveAgentWindow(current, task.destination);
      if (next.start === current.start && next.end === current.end) {
        staged.current = null;
        return;
      }
      if (next.start < current.start) {
        const row = viewportRef.current?.querySelector<HTMLElement>(
          '[data-agent-history-row="' + task.target + '"]',
        );
        if (row) anchor.current = { index: task.target, top: row.getBoundingClientRect().top };
      }
      rangeRef.current = next;
      setState(old => ({ ...old, range: next }));
      if (next.start === task.destination.start && next.end === task.destination.end) {
        staged.current = null;
      }
    });
    return () => cancelAnimationFrame(frame);
  }, [range.start, range.end, viewportRef]);

  useLayoutEffect(() => {
    if (staged.current !== null || !deferredScroll.current) return;
    deferredScroll.current = false;
    const viewport = viewportRef.current;
    if (!viewport) return;
    // Read the final layout instead of using a stale coordinate from the
    // temporarily one-row document. React has now committed the full window.
    const frame = requestAnimationFrame(() => {
      viewport.dispatchEvent(new Event("scroll"));
    });
    return () => cancelAnimationFrame(frame);
  }, [range.start, range.end, viewportRef]);

  const revealSearchRow = useCallback((index: number) => {
    const nextIntent = requestAgentHistoryReveal(navigation.current, index, countRef.current);
    if (nextIntent === navigation.current) return;
    navigation.current = nextIntent;
    readerIntent.current = consumeAgentReaderScrollIntent(readerIntent.current, readerIntent.current.epoch);
    pendingScrollEpoch.current = null;
    staged.current = null;
    deferredScroll.current = false;
    // A search command is reader navigation, never permission to continue
    // following a streaming tail. The viewport owner also detaches BEFORE
    // writing scrollTop once the destination row has mounted.
    following.current = false;
    visibleIndex.current = index;
    anchor.current = null;
    jump.current = null;
    searchHitRef.current = index;
    setSearchHit(index);
    // A repeated hit may already be mounted, and setSearchHit(sameIndex)
    // does not produce a render. Force a fresh layout navigation transaction.
    setRevealRevision(v => v + 1);
    const next = budget !== null
      ? agentBudgetWindow(countRef.current, index, costs, budget, budgetRows)
      : agentWindowAtIndex(countRef.current, index, rangeRef.current, windowRows, fixedRowsStrict);
    if (next.start !== rangeRef.current.start || next.end !== rangeRef.current.end) {
      rangeRef.current = next;
      setState(old => ({ ...old, range: next }));
    }
  }, [windowRows, costs, budget, budgetRows, fixedRowsStrict]);

  // Changing the preference should not strand the reader at a stale scroll offset.
  // Keep height measurements; only the mounted window changes.
  useLayoutEffect(() => {
    if (lastWindowRows.current === windowRows && lastDomLevel.current === domLevel &&
        lastFixedRowsStrict.current === fixedRowsStrict &&
        lastFixedAdaptive.current === fixedViewportAdaptive) return;
    lastWindowRows.current = windowRows;
    lastDomLevel.current = domLevel;
    lastFixedRowsStrict.current = fixedRowsStrict;
    lastFixedAdaptive.current = fixedViewportAdaptive;
    // A pending progressive destination belongs to the PREVIOUS mode. Do
    // not let its next frame reintroduce rows outside the new DOM budget.
    staged.current = null;
    deferredScroll.current = false;
    readerIntent.current = consumeAgentReaderScrollIntent(readerIntent.current, readerIntent.current.epoch);
    pendingScrollEpoch.current = null;
    if (animation.current !== null) cancelAnimationFrame(animation.current);
    animation.current = null;
    if (getIsFollowingBottom()) jumpBottom("auto");
    else navigateViewport(heights.offset(visibleIndex.current));
  }, [windowRows, domLevel, fixedRowsStrict, fixedViewportAdaptive, range.start, range.end, heights, jumpBottom, navigateViewport, getIsFollowingBottom]);

  // Register a search target only for the opt-in virtualized thread. The normal
  // find engine can keep its existing DOM semantics without mounting the history.
  useEffect(() => {
    let query = "";
    let matches: AgentFindMatch[] = [];
    let active = -1;
    let result: FindTargetResult = { count: 0, active: -1 };
    let timer: ReturnType<typeof setTimeout> | null = null;
    const recalculate = (fresh: boolean) => {
      const state = aui.thread().getState();
      const found = findAgentTextMatches(state.messages, query);
      const former = !fresh && active >= 0 ? matches[active] : null;
      matches = found.matches;
      active = former ? matches.findIndex(hit =>
        hit.row === former.row && hit.offset === former.offset) : -1;
      if (active < 0 && matches.length > 0) {
        const firstVisible = matches.findIndex(hit => hit.row >= rangeRef.current.start);
        active = firstVisible >= 0 ? firstVisible : 0;
      }
      result = { count: matches.length, active, capped: found.capped };
      if (active >= 0 && fresh) revealSearchRow(matches[active].row);
      else if (active < 0) { searchHitRef.current = -1; setSearchHit(-1); }
      notifyFindTargets();
    };
    const target = {
      id: "agent-chat-history:" + (resetKey ?? "current"),
      kind: "chat" as const,
      available: () => viewportRef.current !== null,
      contains: (node: Node) => viewportRef.current?.contains(node) ?? false,
      search: (value: string) => {
        if (query === value) return;
        query = value;
        recalculate(true);
      },
      step: (delta: -1 | 1) => {
        if (!matches.length) return;
        active = (active + delta + matches.length) % matches.length;
        result = { ...result, active };
        revealSearchRow(matches[active].row);
        notifyFindTargets();
      },
      result: () => result,
    };
    const unregister = registerFindTarget(target);
    const unsubscribe = aui.subscribe(() => {
      if (!query || timer !== null) return;
      timer = setTimeout(() => {
        timer = null;
        recalculate(false);
      }, 300);
    });
    return () => {
      unregister();
      unsubscribe();
      if (timer !== null) clearTimeout(timer);
    };
  }, [aui, resetKey, viewportRef, revealSearchRow]);

  useLayoutEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    // React measurements/anchor repair must not compete with a native thumb drag.
    if (nativeThumbDrag.current !== null) {
      anchor.current = null;
      jump.current = null;
      pendingPassiveReveal.current = null;
      return;
    }
    const intent = navigation.current;
    if (intent.kind === "search") {
      const row = viewport.querySelector<HTMLElement>('[data-agent-history-row="' + intent.row + '"]');
      if (row) {
        if (revealFrame.current !== null) cancelAnimationFrame(revealFrame.current);
        // `instant` is required: the viewport has scroll-smooth styling.
        // The common intent-aware owner detaches follow and commits scrollTop.
        const desired = viewport.scrollTop +
          row.getBoundingClientRect().top - viewport.getBoundingClientRect().top - 72;
        navigateViewport(desired);
        navigation.current = markAgentHistoryRevealPositioned(navigation.current, intent.epoch);
        revealFrame.current = requestAnimationFrame(() => {
          revealFrame.current = null;
          const active = navigation.current;
          if (active.kind !== "search" || active.epoch !== intent.epoch) return;
          const rendered = viewport.querySelector<HTMLElement>('[data-agent-history-row="' + intent.row + '"]');
          const bounds = rendered?.getBoundingClientRect();
          const viewportBounds = viewport.getBoundingClientRect();
          const visible = !!bounds && bounds.bottom > viewportBounds.top &&
            bounds.top < viewportBounds.bottom;
          navigation.current = completeAgentHistoryReveal(active, intent.epoch, visible);
        });
      }
    }
    const action = jump.current;
    jump.current = null;
    const oldAnchor = anchor.current;
    anchor.current = null;
    if (action === "top") navigateViewport(0);
    else if (action === "bottom") jumpBottom("auto");
    else if (oldAnchor) {
      const row = viewport.querySelector<HTMLElement>('[data-agent-history-row="' + oldAnchor.index + '"]');
      if (row) correctAnchor(row.getBoundingClientRect().top - oldAnchor.top);
    }
    // A scrollbar thumb can jump millions of estimated pixels in one event.
    // Do not require a SECOND wheel gesture merely to reveal its destination:
    // after the staged window is fully committed, reconcile the mounted target
    // with the real viewport geometry once, before the browser paints.
    const pending = pendingPassiveReveal.current;
    if (pending !== null && staged.current === null) {
      const row = viewport.querySelector<HTMLElement>(
        '[data-agent-history-row="' + pending + '"]',
      );
      if (row) {
        pendingPassiveReveal.current = null;
        const viewportBox = viewport.getBoundingClientRect();
        const rowBox = row.getBoundingClientRect();
        if (rowBox.bottom <= viewportBox.top + 8 ||
            rowBox.top >= viewportBox.bottom - 8) {
          navigateViewport(
            viewport.scrollTop + rowBox.top - viewportBox.top -
            Math.min(72, viewport.clientHeight / 4),
          );
        }
      }
    }
    // Unseen rows have estimated heights. A measurement/scroll-anchor update
    // after a physical wheel can put ALL mounted rows outside the viewport,
    // and the reader epoch intentionally blocks layout-origin scroll echoes.
    // Recover once per real wheel epoch, only if not a single row intersects
    // the viewport; ordinary partial-height windows are budgeted separately.
    const epoch = readerIntent.current.epoch;
    const recentWheel = epoch > 0 &&
      performance.now() - lastReaderGestureAt.current <= 1500;
    // The total mounted content height is NOT the same as physical viewport
    // coverage. After rows shrink from the 720px estimate to their measured
    // heights, the physical top may lie in the virtual spacer immediately
    // above our 16/32-row window. In that state 16 mounted short messages can
    // coexist with a 600px blank area that never gets filled on idle.
    //
    // Re-target the existing budgeted window to the actual physical endpoint,
    // never write scrollTop here and never bypass search or staging ownership.
    // Retarget only AFTER a short period without a physical wheel event:
    // applying correction mid-burst creates additional 1680px blank frames.
    if (recentWheel && wheelIdleReady.current && budget !== null &&
        staged.current === null && pendingPassiveReveal.current === null &&
        navigation.current.kind === "reader" && !getIsFollowingBottom()) {
      const bounds = viewport.getBoundingClientRect();
      const viewportHeight = viewport.clientHeight;
      const remaining = viewport.scrollHeight - viewport.scrollTop - viewportHeight;
      const interior = viewport.scrollTop >= viewportHeight && remaining >= viewportHeight;
      if (interior && viewportHeight > 0) {
        const wrappers = [...viewport.querySelectorAll<HTMLElement>("[data-agent-history-row]")];
        const first = wrappers[0], last = wrappers[wrappers.length - 1];
        if (first && last) {
          const spacer = topSpacer.current;
          if (spacer) {
            const originTop = spacer.getBoundingClientRect().top;
            const physicalTopIndex = agentIndexAtScrollPosition(
              heights, originTop, bounds.top, viewport.scrollTop, 0,
            );
            const physicalBottomIndex = agentIndexAtScrollPosition(
              heights, originTop, bounds.top, viewport.scrollTop, viewportHeight * 3,
            );
            const candidate = agentViewportCoverageRecoveryTarget(
              rangeRef.current, physicalTopIndex, physicalBottomIndex,
              Math.max(0, first.getBoundingClientRect().top - bounds.top),
              Math.max(0, bounds.bottom - last.getBoundingClientRect().bottom),
              viewportHeight,
            );
            const previous = coverageRecovery.current;
            const attempts = previous.epoch === epoch ? previous.attempts : 0;
            if (candidate !== null && attempts < 3) {
              coverageRecovery.current = { epoch, attempts: attempts + 1 };
              visibleIndex.current = candidate;
              const adjacent = agentPhysicalNeighborWindow(
                rangeRef.current, candidate, countRef.current, costs,
                budget, budgetRows,
              );
              if (adjacent) {
                // Keep up to 15/31 ALREADY measured/mounted rows and replace
                // only the opposite edge. An adjacent unknown row must not
                // collapse an entire otherwise-filled window to one row.
                rangeRef.current = adjacent;
                setState(old => ({ ...old, range: adjacent }));
                return;
              }
              if (seek(candidate)) return;
            }
          }
        }
      }
    }
    if (pendingPassiveReveal.current === null && staged.current === null &&
        recentWheel && !getIsFollowingBottom() &&
        navigation.current.kind === "reader" &&
        lastBlankRecoveryEpoch.current !== epoch) {
      const bounds = viewport.getBoundingClientRect();
      const rows = [...viewport.querySelectorAll<HTMLElement>("[data-agent-history-row]")];
      const hasVisible = rows.some(row => {
        const rect = row.getBoundingClientRect();
        return rect.bottom > bounds.top + 1 && rect.top < bounds.bottom - 1;
      });
      if (!hasVisible && rows.length > 0) {
        const index = Math.min(Math.max(visibleIndex.current, range.start), range.end - 1);
        const row = rows.find(item => Number(item.dataset.agentHistoryRow) === index) ?? rows[0];
        lastBlankRecoveryEpoch.current = epoch;
        navigateViewport(viewport.scrollTop +
          row.getBoundingClientRect().top - bounds.top -
          Math.min(72, viewport.clientHeight / 4));
      }
    }
  }, [range.start, range.end, heightRevision, searchHit, revealRevision, viewportRef, jumpBottom, correctAnchor, navigateViewport, getIsFollowingBottom, budget, budgetRows, costs, heights, seek]);

  // A scroll event is not itself evidence of user navigation. Internal
  // anchor corrections, CSS/native scroll anchoring and remeasuring huge tool
  // rows all emit scroll events. Authorize ONE window change per fresh reader
  // gesture epoch, then ignore geometry-driven echoes until the next gesture.
  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    const physicalEdgeReconcile = () => {
      const atTop = viewport.scrollTop < 1;
      const atBottom = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight < 24;
      return allowsRecentAgentBoundaryReconcile(
        readerIntent.current, lastReaderGestureAt.current, performance.now(),
        rangeRef.current.start, rangeRef.current.end, countRef.current, atTop, atBottom,
      );
    };
    const onScroll = () => {
      const epoch = readerIntent.current.epoch;
      // Native Chromium captures the thumb outside the DOM pointermove stream.
      // Its actual scroll events are reader intent for the *entire* captured
      // gesture, not just its first window transition. Regular wheel/anchor
      // events retain their strict one-transition-per-epoch gate.
      if (nativeThumbDrag.current === null &&
          !allowsAgentReaderScrollIntent(readerIntent.current, epoch) &&
          !physicalEdgeReconcile()) return;
      pendingScrollEpoch.current = epoch;
      if (animation.current !== null) return;
      animation.current = requestAnimationFrame(() => {
        animation.current = null;
        const pendingEpoch = pendingScrollEpoch.current;
        pendingScrollEpoch.current = null;
        const activeThumb = nativeThumbDrag.current !== null;
        const freshGesture = pendingEpoch !== null &&
          allowsAgentReaderScrollIntent(readerIntent.current, pendingEpoch);
        const recentPhysicalEdge = !freshGesture && physicalEdgeReconcile();
        if (pendingEpoch === null || (!activeThumb && !freshGesture && !recentPhysicalEdge) ||
            !allowsPassiveAgentScroll(navigation.current)) return;
        if (activeThumb) {
          // The viewport is owned by the native thumb. Chase its actual
          // physical coordinate with DOM-window changes only: never write
          // scrollTop, synthesize pointermove or enable passive anchor repair.
          // Retarget a progressive window only after the thumb leaves its
          // eventual destination; otherwise let its budgeted staging finish.
          const spacer = topSpacer.current;
          if (!spacer || countRef.current === 0) return;
          const index = agentIndexAtScrollPosition(
            heights, spacer.getBoundingClientRect().top,
            viewport.getBoundingClientRect().top,
            viewport.scrollTop, viewport.clientHeight,
          );
          const nextTarget = agentNativeThumbLiveTarget(
            rangeRef.current, staged.current?.destination ?? null,
            index, countRef.current,
          );
          if (nextTarget !== null) {
            visibleIndex.current = nextTarget;
            seek(nextTarget);
          }
          return;
        }
        if (staged.current !== null) {
          // A second REAL wheel gesture may move by many screens before the
          // old one-row staging transaction completes. Retarget immediately
          // rather than waiting for the previous destination to fill first.
          // Each replacement still begins with ONE row, so heavy tool outputs
          // cannot all mount in a single event-loop turn.
          const spacer = topSpacer.current;
          if (freshGesture && spacer && countRef.current > 0) {
            const index = agentIndexAtScrollPosition(
              heights, spacer.getBoundingClientRect().top,
              viewport.getBoundingClientRect().top,
              viewport.scrollTop, viewport.clientHeight,
            );
            const destination = staged.current.destination;
            if (index < destination.start || index >= destination.end) {
              visibleIndex.current = index;
              if (seek(index)) {
                readerIntent.current = consumeAgentReaderScrollIntent(
                  readerIntent.current, pendingEpoch,
                );
              }
              deferredScroll.current = false;
              return;
            }
          }
          // If still within the staged window, let it complete, then sample
          // the last actual wheel position once (not every layout echo).
          deferredScroll.current = true;
          return;
        }
        const total = countRef.current;
        if (!total) return;
        // Automatic follow pins can fire scroll events after a click has armed
        // a gesture epoch. An echo is not an instruction to leave the tail.
        if (getIsFollowingBottom()) {
          following.current = true;
          visibleIndex.current = total - 1;
          if (rangeRef.current.end !== total) seek(total - 1, "bottom");
          return;
        }
        const atBottom = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight < 24;
        following.current = atBottom;
        let changed = false;
        if (atBottom) changed = seek(total - 1, recentPhysicalEdge ? "bottom" : undefined);
        else if (viewport.scrollTop < 1) changed = seek(0, "top");
        else {
          const spacer = topSpacer.current;
          if (!spacer) return;
          const index = agentIndexAtScrollPosition(
            heights,
            spacer.getBoundingClientRect().top,
            viewport.getBoundingClientRect().top,
            viewport.scrollTop,
            viewport.clientHeight,
          );
          visibleIndex.current = index;
          changed = seek(index);
        }
        if (changed && freshGesture) {
          // Consume BEFORE React commits rows and correction writes scrollTop.
          // Those native scroll events cannot reverse this chosen window.
          readerIntent.current = consumeAgentReaderScrollIntent(readerIntent.current, pendingEpoch);
        }
      });
    };
    viewport.addEventListener("scroll", onScroll, { passive: true });
    // Each physical gesture supersedes a pending search and authorizes a fresh
    // reader transaction. A pointer drag needs repeated epochs because scroll
    // bars can move continuously while the button remains depressed.
    const interrupt = () => {
      if (wheelCoverageTimer.current !== null) {
        clearTimeout(wheelCoverageTimer.current);
        wheelCoverageTimer.current = null;
      }
      wheelIdleReady.current = false;
      lastReaderGestureAt.current = performance.now();
      readerIntent.current = requestAgentReaderScrollIntent(readerIntent.current);
      navigation.current = interruptAgentHistoryReveal(navigation.current);
      if (revealFrame.current !== null) {
        cancelAnimationFrame(revealFrame.current);
        revealFrame.current = null;
      }
    };
    // Native scrollbar capture bypasses DOM pointermove. Under virtualization
    // the physical scrollbar range changes as unknown heights get measured,
    // so release needs a logical position independent of last physical scrollTop.
    const onPointerDown = (event: PointerEvent) => {
      const gutterWidth = viewport.offsetWidth - viewport.clientWidth;
      const rect = viewport.getBoundingClientRect();
      const inGutter = gutterWidth >= 6 && (
        (event.clientX >= rect.right - gutterWidth && event.clientX <= rect.right) ||
        (event.clientX >= rect.left && event.clientX <= rect.left + gutterWidth)
      );
      if (inGutter) interrupt();
      nativeThumbDrag.current = event.pointerType === "mouse" && event.button === 0 && inGutter
        ? beginAgentNativeThumbDrag({
            pointerId: event.pointerId, pointerY: event.clientY,
            viewportTop: rect.top, viewportHeight: viewport.clientHeight,
            scrollTop: viewport.scrollTop, scrollHeight: viewport.scrollHeight,
          })
        : null;
      nativeThumbVirtualTotal.current = nativeThumbDrag.current !== null
        ? heights.total : null;
      if (nativeThumbDrag.current !== null) {
        // Detach the shared autoscroll owner at gesture BEGIN, before any
        // row-height changes can make an initial upward scroll look like a clamp.
        detachFromBottom();
        following.current = false;
      }
    };
    const onPointerRelease = (event: PointerEvent) => {
      const drag = nativeThumbDrag.current;
      if (!drag || drag.pointerId !== event.pointerId) return;
      nativeThumbDrag.current = null;
      nativeThumbVirtualTotal.current = null;
      // Restore live document length after the native gesture ends, regardless
      // of whether the reader actually moved. The same physical scroll owner
      // will perform any necessary final logical navigation below.
      requestHeightRevision();
      const destination = finishAgentNativeThumbDrag(drag, {
        pointerId: event.pointerId, pointerY: event.clientY,
        scrollTop: viewport.scrollTop, viewportHeight: viewport.clientHeight, heights,
      });
      if (!destination) {
        // Click-only/track behavior stays native.
        if (Math.abs(viewport.scrollTop - drag.startScrollTop) >= 1) {
          interrupt();
          onScroll();
        }
        return;
      }
      // Chromium is the position authority during the entire native drag.
      // It may show a different thumb position than the OS pointer. Resolve
      // the *actual rendered transcript* before restoring the live suffix,
      // rather than making the chat jump back to the pointer's coordinate.
      const bounds = viewport.getBoundingClientRect();
      const visibleRow = [...viewport.querySelectorAll<HTMLElement>("[data-agent-history-row]")]
        .find(row => {
          const rect = row.getBoundingClientRect();
          return rect.bottom > bounds.top + 8 && rect.top < bounds.bottom - 8;
        });
      const spacer = topSpacer.current;
      const nativeIndex = spacer
        ? agentIndexAtScrollPosition(
            heights, spacer.getBoundingClientRect().top,
            bounds.top, viewport.scrollTop, viewport.clientHeight,
          )
        : Number(visibleRow?.dataset.agentHistoryRow ?? 0);
      const release = resolveAgentNativeThumbRelease(destination, {
        nativeScrollTop: viewport.scrollTop,
        nativeScrollHeight: viewport.scrollHeight,
        viewportHeight: viewport.clientHeight,
        nativeIndex,
        rowVisibleAtNativePosition: visibleRow !== undefined,
        count: countRef.current,
      });
      interrupt();
      readerIntent.current = consumeAgentReaderScrollIntent(
        readerIntent.current, readerIntent.current.epoch,
      );
      pendingScrollEpoch.current = null;
      deferredScroll.current = false;
      pendingPassiveReveal.current = null;
      anchor.current = null;
      visibleIndex.current = release.index;
      following.current = release.kind === "bottom";
      if (release.kind === "keep") {
        // Nothing to repair: the right message is already under the native
        // scrollbar. Leave any one/two-row staged DOM-window admission alive
        // so the normal budgeted renderer can finish without a second seek.
        return;
      }
      staged.current = null;
      if (release.kind === "top") {
        navigateViewport(0);
        seek(0, "top");
      } else if (release.kind === "bottom") {
        seek(countRef.current - 1, "bottom");
      } else {
        // Physical scroll landed in an as-yet unmounted part of the virtual
        // document. Chase THAT position, not the divergent pointer coordinate.
        // seek() owns only the DOM window; it never directly writes scrollTop.
        seek(release.index);
      }
    };
    const onPointerCancel = (event: PointerEvent) => {
      if (nativeThumbDrag.current?.pointerId !== event.pointerId) return;
      nativeThumbDrag.current = null;
      nativeThumbVirtualTotal.current = null;
      requestHeightRevision();
      // Cancellation is not a committed destination. Resume native reader
      // sampling; do not map a cancelled coordinate to a logical message.
      interrupt();
      onScroll();
    };
    const onPointerMove = (event: PointerEvent) => {
      if (event.buttons !== 0) interrupt();
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.altKey || event.metaKey) return;
      if (!["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " "].includes(event.key)) return;
      const target = event.target;
      if (target instanceof HTMLElement) {
        if (target.closest('input, textarea, select, [contenteditable="true"], [role="textbox"]')) return;
        if (!viewport.contains(target) && target !== document.body && target !== document.documentElement) return;
      }
      interrupt();
    };
    const onWheel = (event: WheelEvent) => {
      interrupt();
      wheelPrefetchPending.current = null;
      // Predict ONLY an adjacent older-history row for a native, moderate
      // wheel gesture. The existing measured-DOM policy rejects a second
      // unknown heavy row and drops opposite-edge rows before exceeding caps.
      // Large jumps still use conservative staged seeks as before.
      if (budget !== null && !event.ctrlKey && !event.defaultPrevented &&
          staged.current === null && pendingPassiveReveal.current === null &&
          navigation.current.kind === "reader" && !getIsFollowingBottom()) {
        const first = viewport.querySelector<HTMLElement>("[data-agent-history-row]");
        if (first) {
          const box = viewport.getBoundingClientRect();
          const target = agentWheelPrefetchTarget(
            event.deltaY, first.getBoundingClientRect().top,
            box.top, viewport.clientHeight, rangeRef.current.start,
          );
          if (target !== null) {
            const next = agentPhysicalNeighborWindow(
              rangeRef.current, target, countRef.current, costs, budget, budgetRows,
            );
            if (next) {
              wheelPrefetchPending.current = { at: performance.now(), deltaY: event.deltaY };
              rangeRef.current = next;
              setState(old => ({ ...old, range: next }));
            }
          }
        }
      }
      // This is a genuine trailing-wheel reconciliation, NOT a synthetic
      // scroll event or another scrollTop owner. The layout effect samples
      // the real physical viewport only after the reader stops spinning.
      wheelCoverageTimer.current = setTimeout(() => {
        wheelCoverageTimer.current = null;
        wheelIdleReady.current = true;
        requestHeightRevision();
      }, 100);
    };
    viewport.addEventListener("wheel", onWheel, { passive: true });
    viewport.addEventListener("touchmove", interrupt, { passive: true });
    viewport.addEventListener("pointerdown", onPointerDown, { passive: true });
    viewport.addEventListener("pointermove", onPointerMove, { passive: true });
    window.addEventListener("pointerup", onPointerRelease, { passive: true });
    window.addEventListener("pointercancel", onPointerCancel, { passive: true });
    window.addEventListener("keydown", onKeyDown);
    return () => {
      viewport.removeEventListener("scroll", onScroll);
      viewport.removeEventListener("wheel", onWheel);
      nativeThumbDrag.current = null;
      nativeThumbVirtualTotal.current = null;
      if (wheelCoverageTimer.current !== null) clearTimeout(wheelCoverageTimer.current);
      wheelCoverageTimer.current = null;
      wheelIdleReady.current = false;
      viewport.removeEventListener("touchmove", interrupt);
      viewport.removeEventListener("pointerdown", onPointerDown);
      viewport.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerRelease);
      window.removeEventListener("pointercancel", onPointerCancel);
      window.removeEventListener("keydown", onKeyDown);
      if (animation.current !== null) cancelAnimationFrame(animation.current);
      animation.current = null;
      pendingScrollEpoch.current = null;
    };
  }, [viewportRef, heights, seek, navigateViewport, detachFromBottom, getIsFollowingBottom, requestHeightRevision, budget, budgetRows, costs]);
  // Measure the committed window and settle towards the known DOM budget.
  // Never supersede an in-progress staged seek or an active search epoch.
  const budgetFrame = useRef<number | null>(null);
  const lastBudgetProbe = useRef(0);
  const budgetProbeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const scheduleBudgetRebalance = useCallback(() => {
    if (budget === null || budgetFrame.current !== null) return;
    budgetFrame.current = requestAnimationFrame(() => {
      budgetFrame.current = null;
      if (staged.current !== null) return;
      const intent = navigation.current;
      const followingTail = getIsFollowingBottom();
      const target = intent.kind === "search" ? intent.row
        : followingTail ? countRef.current - 1 : visibleIndex.current;
      const next = agentBudgetWindow(countRef.current, target, costs, budget, budgetRows);
      const viewport = viewportRef.current;
      if (!viewport) return;
      // The old policy greedily added one newly measured neighbor on every
      // animation frame, moving anchored text over and over in Low/High mode.
      // Never grow a window that already fills the reader's viewport.
      const visibleHeight = [...viewport.querySelectorAll<HTMLElement>("[data-agent-history-row]")]
        .reduce((sum, el) => sum + el.getBoundingClientRect().height, 0);
      if (!needsAgentBudgetRebalance(
        rangeRef.current, next, target, visibleHeight, viewport.clientHeight,
      )) return;
      const row = viewport.querySelector<HTMLElement>(
        '[data-agent-history-row="' + target + '"]',
      );
      anchor.current = row ? { index: target, top: row.getBoundingClientRect().top } : null;
      if (followingTail) jump.current = "bottom";
      rangeRef.current = next;
      setState(old => ({ ...old, range: next }));
    });
  }, [budget, budgetRows, costs, viewportRef, getIsFollowingBottom]);
  // Throttle costly descendant counting but ALWAYS keep one trailing probe:
  // a final resize during the throttle window must not leave an oversized
  // streaming tool result at its old, underestimated cost indefinitely.
  const scheduleBudgetProbe = useCallback(() => {
    if (budget === null || budgetProbeTimer.current !== null) return;
    const delay = Math.max(0, 1_000 - (performance.now() - lastBudgetProbe.current));
    budgetProbeTimer.current = setTimeout(() => {
      budgetProbeTimer.current = null;
      lastBudgetProbe.current = performance.now();
      if (staged.current !== null) return;
      const viewport = viewportRef.current;
      if (!viewport) return;
      let changed = false;
      for (const wrapper of viewport.querySelectorAll<HTMLElement>("[data-agent-history-row]")) {
        const index = Number(wrapper.dataset.agentHistoryRow);
        if (Number.isSafeInteger(index)) {
          changed = costs.measure(index, wrapper.querySelectorAll("*").length + 1) || changed;
        }
      }
      if (changed) scheduleBudgetRebalance();
    }, delay);
  }, [budget, costs, viewportRef, scheduleBudgetRebalance]);
  useEffect(() => () => {
    if (budgetFrame.current !== null) cancelAnimationFrame(budgetFrame.current);
    if (budgetProbeTimer.current !== null) clearTimeout(budgetProbeTimer.current);
    budgetFrame.current = null;
    budgetProbeTimer.current = null;
  }, [resetKey, budget]);
  // Content changes do not necessarily resize a row: clipped tool output can
  // add thousands of descendants without changing the observed box. Counting
  // only in ResizeObserver would leave the DOM budget stale indefinitely.
  // Observe mounted message content, not the entire page, and use the existing
  // trailing throttled probe rather than counting on each mutation.
  useEffect(() => {
    if (budget === null || typeof MutationObserver === "undefined") return;
    const viewport = viewportRef.current;
    if (!viewport) return;
    const observer = new MutationObserver(() => scheduleBudgetProbe());
    for (const wrapper of viewport.querySelectorAll<HTMLElement>("[data-agent-history-row]")) {
      const content = wrapper.querySelector<HTMLElement>("[data-role]");
      if (content) observer.observe(content, { childList: true, subtree: true });
    }
    return () => observer.disconnect();
  }, [budget, range.start, range.end, viewportRef, scheduleBudgetProbe]);

  useLayoutEffect(() => {
    if (budget === null || staged.current !== null) return;
    const viewport = viewportRef.current;
    if (!viewport) return;
    let changed = false;
    for (const el of viewport.querySelectorAll<HTMLElement>("[data-agent-history-row]")) {
      const index = Number(el.dataset.agentHistoryRow);
      if (Number.isSafeInteger(index)) {
        changed = costs.measure(index, el.querySelectorAll("*").length + 1) || changed;
      }
    }
    if (changed) {
      lastBudgetProbe.current = performance.now();
      scheduleBudgetRebalance();
    }
  }, [budget, costs, range.start, range.end, viewportRef, scheduleBudgetRebalance]);

  useLayoutEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(entries => {
      let changed = false;
      for (const entry of entries) {
        const row = entry.target as HTMLElement;
        const raw = row.closest<HTMLElement>("[data-agent-history-row]")?.dataset.agentHistoryRow;
        if (raw === undefined) continue;
        const index = Number(raw);
        if (Number.isSafeInteger(index)) {
          changed = heights.measure(index, row.getBoundingClientRect().height) || changed;
          if (budget !== null) scheduleBudgetProbe();
        }
      }
      if (changed) {
        reconcileNativeThumbSpacer();
        requestHeightRevision();
      }
    });
    for (const wrapper of viewport.querySelectorAll<HTMLElement>("[data-agent-history-row]")) {
      const row = wrapper.querySelector<HTMLElement>("[data-role]");
      if (!row) continue;
      const index = Number(wrapper.dataset.agentHistoryRow);
      row.setAttribute("aria-setsize", String(count));
      row.setAttribute("aria-posinset", String(index + 1));
      if (heights.measure(index, row.getBoundingClientRect().height)) requestHeightRevision();
      observer.observe(row);
    }
    reconcileNativeThumbSpacer();
    return () => observer.disconnect();
  }, [range.start, range.end, count, viewportRef, heights, requestHeightRevision, budget, scheduleBudgetProbe, reconcileNativeThumbSpacer]);

  // A ResizeObserver measurement must update both spacers, not merely the ledger.
  void heightRevision;
  const rows = useMemo(() => {
    const element = renderMessage();
    const result: ReactElement[] = [];
    for (let index = range.start; index < range.end; index++) {
      result.push(<div key={index} data-agent-history-row={index} data-agent-history-find-active={searchHit === index ? "true" : undefined} role="listitem" aria-current={searchHit === index ? "location" : undefined} aria-posinset={index + 1} aria-setsize={count} className={searchHit === index ? "min-w-0 rounded ring-2 ring-primary/70" : "min-w-0"}>
        <AuiProvider value={gate.row(index)}><MessageByIndexProvider index={index}>{element}</MessageByIndexProvider></AuiProvider>
      </div>);
    }
    return result;
  }, [range.start, range.end, renderMessage, gate, count, searchHit]);
  // Reclaim row clients from previously visited windows after React has
  // finished unmounting their subscriptions. Keep the mounted window stable.
  useEffect(() => {
    gate.pruneOutside(range.start, range.end);
  }, [gate, range.start, range.end]);
  const backToLatest = () => {
    navigation.current = interruptAgentHistoryReveal(navigation.current);
    staged.current = null;
    deferredScroll.current = false;
    following.current = true;
    seek(countRef.current - 1, "bottom");
  };
  const bottomSpacerPx = agentNativeThumbBottomSpacerPx(
    nativeThumbVirtualTotal.current ?? heights.total, heights.offset(range.end),
  );
  return <div data-agent-history-scroll-list="true" data-agent-history-dom-level={domLevel ?? "legacy"} data-agent-history-qa-stage={staged.current !== null ? "true" : "false"} data-agent-history-qa-stage-target={staged.current?.target ?? ""} data-agent-history-qa-visible-index={visibleIndex.current} data-agent-history-qa-window-start={range.start} data-agent-history-qa-window-end={range.end} role="list" className="flex min-w-0 flex-col">
    {range.end < count && <button type="button" className="sticky top-2 z-20 mx-auto rounded-full border bg-background px-3 py-1 text-xs shadow" onClick={backToLatest}>{t("settings.chat.agentHistoryJumpToLatest")}</button>}
    <div ref={topSpacer} data-agent-history-spacer="top" aria-hidden="true" style={{ height: heights.offset(range.start), flexShrink: 0 }} />
    {rows}
    <div ref={bottomSpacer} data-agent-history-spacer="bottom" aria-hidden="true" style={{ height: bottomSpacerPx, flexShrink: 0 }} />
  </div>;
});
