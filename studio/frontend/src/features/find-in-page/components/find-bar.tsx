// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { isImeComposing } from "@/features/settings";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { MessageCircleIcon } from "@/lib/hugeicons-derived";
import { Cancel01Icon, InternetIcon, Search01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { ArrowDownIcon, ArrowUpIcon } from "lucide-react";
import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import { useFindInPage } from "../hooks/use-find-in-page.ts";
import {
  EMPTY_FIND_RESULT,
  type FindTarget,
  availableFindTargets,
  findTarget,
  findTargetsVersion,
  subscribeFindTargets,
} from "../lib/find-targets.ts";
import { isFindScopeBackgrounded } from "../lib/find-backgrounded.ts";
import {
  resolveDismissiblePortalSurfaces,
  resolveFindScope,
} from "../lib/find-dom.ts";
export type FindBarProps = {
  query: string;
  setQuery: (query: string) => void;
  /** What it searches: the chat (null), or a find target's id. */
  scope: string | null;
  setScope: (scope: string | null) => void;
  close: () => void;
  focusToken: number;
  restoreSelection: (input: HTMLInputElement) => boolean;
  pendingSteps: { query: string; delta: -1 | 1 }[];
  clearPendingSteps: () => void;
};

function keepFocusInField(event: { preventDefault: () => void }): void {
  event.preventDefault();
}

function rewindToStart(event: { currentTarget: HTMLInputElement }): void {
  const input = event.currentTarget;
  input.setSelectionRange(0, 0);
  input.scrollLeft = 0;
}

const FIND_BUTTON_CLASS = "size-8 hover:bg-[rgb(0_0_0_/_calc(0.06*var(--contrast-wash-gain,1)))] dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))]";

export const FIND_QUERY_SETTLE_MS = 100;

function useSettledQuery(query: string): [string, () => void] {
  const [settled, setSettled] = useState(query);
  useEffect(() => {
    if (query.length === 0) {
      setSettled("");
      return;
    }
    const timer = setTimeout(() => setSettled(query), FIND_QUERY_SETTLE_MS);
    return () => clearTimeout(timer);
  }, [query]);
  return [settled, () => setSettled(query)];
}

function useTargetFind(target: FindTarget | undefined, query: string) {
  useEffect(() => {
    if (!target) return;
    target.search(query);
  }, [target, query]);
  useEffect(() => {
    if (!target) return;
    return () => target.search("");
  }, [target]);
  const result = useSyncExternalStore(subscribeFindTargets, () => target?.result() ?? EMPTY_FIND_RESULT);
  return {
    count: result.count,
    active: result.active,
    next: () => target?.step(1),
    previous: () => target?.step(-1),
  };
}

// biome-ignore lint/style/noDefaultExport: React.lazy requires the component as a default export.
export default function FindBar({
  query,
  setQuery,
  scope,
  setScope,
  close,
  focusToken,
  restoreSelection,
  pendingSteps,
  clearPendingSteps,
}: FindBarProps) {
  const t = useT();
  const [settledQuery, settleQuery] = useSettledQuery(query);

  const [completedQuery, setCompletedQuery] = useState<string | null>(null);
  const queryPending = query !== settledQuery;
  useSyncExternalStore(subscribeFindTargets, findTargetsVersion);
  const targets = availableFindTargets();
  const target = scope === null ? undefined : targets.find((candidate) => candidate.id === scope);
  useEffect(() => {
    if (scope !== null && !findTarget(scope)?.available()) setScope(null);
  });
  const chat = useFindInPage(target ? "" : settledQuery, queryPending);
  const page = useTargetFind(target, settledQuery);
  // A target that can only step (a native page) reports no count; its walk stays open.
  const pageCount = page.count ?? (settledQuery ? 1 : 0);
  const { count, active, capped, truncated, next, previous } = target
    ? { ...page, count: pageCount, capped: false, truncated: false }
    : chat;
  const uncounted = target !== undefined && page.count === null;

  useEffect(() => {
    if (!queryPending) {
      setCompletedQuery(settledQuery);
    }
  }, [queryPending, settledQuery]);
  const inputRef = useRef<HTMLInputElement>(null);
  const queuedStepsRef = useRef([...pendingSteps]);
  useEffect(() => {
    queuedStepsRef.current = [...pendingSteps];
  }, [pendingSteps]);
  const stepWhenSettled = (delta: -1 | 1) => {
    if (queryPending) {
      queuedStepsRef.current.push({ query, delta });
      settleQuery();
      return;
    }
    if (delta < 0) previous();
    else next();
  };
  useEffect(() => {
    const matchingSteps = queuedStepsRef.current.filter(
      (step) => step.query === query,
    );
    if (matchingSteps.length === queuedStepsRef.current.length) {
      return;
    }
    queuedStepsRef.current = matchingSteps;
    if (matchingSteps.length === 0) {
      clearPendingSteps();
    }
  }, [clearPendingSteps, query]);

  useEffect(() => {
    if (
      queryPending ||
      completedQuery !== settledQuery ||
      queuedStepsRef.current.length === 0
    )
      return;
    const steps = queuedStepsRef.current;
    queuedStepsRef.current = [];
    if (count > 0) {
      for (const step of steps) {
        if (step.delta < 0) previous();
        else next();
      }
    }
    clearPendingSteps();
  }, [
    clearPendingSteps,
    completedQuery,
    count,
    next,
    previous,
    queryPending,
    settledQuery,
  ]);
  // Declared above the focus effect so it reads `activeElement` before the field takes it.
  const barRef = useRef<HTMLDivElement>(null);
  const originRef = useRef<HTMLElement | null>(null);
  useEffect(() => {
    const active = document.activeElement as HTMLElement | null;
    // StrictMode replays this effect, so take the first answer and never the bar's own input.
    if (
      originRef.current === null &&
      active !== null &&
      barRef.current?.contains(active) !== true
    ) {
      originRef.current = active;
    }
    return () => {
      const origin = originRef.current;
      if (!origin?.isConnected || typeof origin.focus !== "function") return;
      const focused = document.activeElement;
      if (focused !== null && focused !== document.body) return;
      origin.focus();
    };
  }, []);

  // Capture on window so closing the bar cannot trigger another Escape action.
  useEffect(() => {
    const onEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || isImeComposing(event)) return;
      if (isFindScopeBackgrounded()) return;
      if (resolveDismissiblePortalSurfaces(resolveFindScope()).length > 0)
        return;
      event.preventDefault();
      event.stopPropagation();
      close();
    };
    window.addEventListener("keydown", onEscape, true);
    return () => window.removeEventListener("keydown", onEscape, true);
  }, [close]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: each token requests a fresh focus/select.
  useEffect(() => {
    const input = inputRef.current;
    if (!input) return;
    input.focus();
    if (!restoreSelection(input)) input.select();
  }, [focusToken, restoreSelection]);

  const searching = query.length > 0;
  // A pending query has no count yet; `stepWhenSettled` queues the press until it arrives.
  const canStep = searching && (count > 0 || queryPending);
  const counter =
    searching && !queryPending && !uncounted
      ? `${count === 0 ? 0 : active + 1}/${count}${capped ? "+" : ""}`
      : "";

  return (
    // `data-find-skip` keeps the bar out of its own index.
    <div
      ref={barRef}
      data-find-skip=""
      // biome-ignore lint/a11y/useSemanticElements: this landmark contains the field and its navigation controls.
      role="search"
      aria-label={t("shell.find.label")}
      data-scoped={targets.length > 0 ? "" : undefined}
      className="find-bar-surface fixed top-[calc(var(--studio-content-top-inset,0px)+3.5rem)] right-4 z-50 flex h-13 w-[calc(22.25rem*var(--ui-space-scale,1))] max-w-[calc(100vw-2rem)] items-center gap-1 rounded-full pr-4 pl-4.5 data-scoped:w-[calc(27.75rem*var(--ui-space-scale,1))] sm:w-[calc(28.25rem*var(--ui-space-scale,1))] sm:data-scoped:w-[calc(33.75rem*var(--ui-space-scale,1))]"
    >
      <HugeiconsIcon
        icon={Search01Icon}
        strokeWidth={1.75}
        aria-hidden={true}
        className="mr-1.5 size-[calc(18px*var(--ui-space-scale,1))] shrink-0 text-muted-foreground"
      />
      <input
        ref={inputRef}
        type="text"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            // An IME candidate commit Enter arrives here too and must not walk.
            if (isImeComposing(event.nativeEvent)) return;
            event.preventDefault();
            stepWhenSettled(event.shiftKey ? -1 : 1);
          }
        }}
        onBlur={rewindToStart}
        placeholder={t("shell.find.label")}
        aria-label={t("shell.find.label")}
        spellCheck={false}
        autoComplete="off"
        autoCorrect="off"
        className="min-w-0 flex-1 bg-transparent text-ui-15 outline-none placeholder:text-muted-foreground"
      />
      <span
        aria-live="polite"
        title={truncated ? t("shell.find.truncated") : undefined}
        className={cn(
          "min-w-12 shrink-0 pr-3 text-right text-muted-foreground text-sm tabular-nums",
          truncated && "cursor-help underline decoration-dotted",
        )}
      >
        {counter}
      </span>
      <Button
        variant="ghost"
        size="icon"
        className={FIND_BUTTON_CLASS}
        disabled={!canStep}
        onMouseDown={keepFocusInField}
        onClick={() => stepWhenSettled(-1)}
        aria-label={t("shell.find.previous")}
        title={t("shell.find.previous")}
      >
        <ArrowUpIcon strokeWidth={1.75} className="size-[calc(18px*var(--ui-space-scale,1))]" />
      </Button>
      <Button
        variant="ghost"
        size="icon"
        className={FIND_BUTTON_CLASS}
        disabled={!canStep}
        onMouseDown={keepFocusInField}
        onClick={() => stepWhenSettled(1)}
        aria-label={t("shell.find.next")}
        title={t("shell.find.next")}
      >
        <ArrowDownIcon strokeWidth={1.75} className="size-[calc(18px*var(--ui-space-scale,1))]" />
      </Button>
      {targets.length > 0
        ? [null, ...targets.map((candidate) => candidate.id)].map((id) => {
            const label = t(id === null ? "shell.find.searchChat" : "shell.find.searchBrowser");
            const selected = scope === id;
            return (
              <Button
                key={id ?? "chat"}
                variant="ghost"
                size="icon"
                className={cn(
                  FIND_BUTTON_CLASS,
                  "shrink-0",
                  selected ? "text-foreground" : "text-muted-foreground/70 hover:text-foreground",
                )}
                aria-pressed={selected}
                onMouseDown={keepFocusInField}
                onClick={() => setScope(id)}
                aria-label={label}
                title={label}
              >
                <HugeiconsIcon
                  icon={id === null ? MessageCircleIcon : InternetIcon}
                  strokeWidth={1.75}
                  className="size-[calc(18px*var(--ui-space-scale,1))]"
                />
              </Button>
            );
          })
        : null}
      {targets.length > 0 ? <span aria-hidden={true} className="mx-1 h-5 w-px shrink-0 bg-border" /> : null}
      <Button
        variant="ghost"
        size="icon"
        className={FIND_BUTTON_CLASS}
        onClick={close}
        aria-label={t("shell.find.close")}
        title={t("shell.find.close")}
      >
        <HugeiconsIcon icon={Cancel01Icon} className="size-[calc(18px*var(--ui-space-scale,1))]" />
      </Button>
    </div>
  );
}
