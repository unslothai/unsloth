// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { isImeComposing, isSurfaceBackgrounded } from "@/features/settings";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
// lucide supplies the directional arrows used throughout the app.
import { ArrowDownIcon, ArrowUpIcon } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useFindInPage } from "../hooks/use-find-in-page.ts";
import { FIND_SCOPE_ATTRIBUTE } from "../lib/find-attributes.ts";
import {
  resolveDismissiblePortalSurfaces,
  resolveFindScope,
} from "../lib/find-dom.ts";
export type FindBarProps = {
  query: string;
  setQuery: (query: string) => void;
  close: () => void;
  focusToken: number;
  restoreSelection: (input: HTMLInputElement) => boolean;
  pendingStep: -1 | 0 | 1;
  clearPendingStep: () => void;
};

/** Keep the caret in the field when a walk button is clicked. */
function keepFocusInField(event: { preventDefault: () => void }): void {
  event.preventDefault();
}

/** Show the start of a long query again once the field loses focus. */
function rewindToStart(event: { currentTarget: HTMLInputElement }): void {
  const input = event.currentTarget;
  input.setSelectionRange(0, 0);
  input.scrollLeft = 0;
}

/** The wash reads on both the light and dark find-bar surfaces. */
const FIND_BUTTON_CLASS = "size-8 hover:bg-black/[0.06] dark:hover:bg-white/10";

/** Coalesce a typing burst before the DOM search/highlight work runs. */
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

/** The on-demand UI and engine for an open find session. */
// biome-ignore lint/style/noDefaultExport: React.lazy requires the component as a default export.
export default function FindBar({
  query,
  setQuery,
  close,
  focusToken,
  restoreSelection,
  pendingStep,
  clearPendingStep,
}: FindBarProps) {
  const t = useT();
  const [settledQuery, settleQuery] = useSettledQuery(query);
  const queryPending = query !== settledQuery;
  const { count, active, capped, truncated, next, previous } = useFindInPage(
    settledQuery,
    queryPending,
  );
  const inputRef = useRef<HTMLInputElement>(null);
  const queuedStepRef = useRef<-1 | 0 | 1>(pendingStep);
  const stepWhenSettled = (delta: -1 | 1) => {
    if (queryPending) {
      queuedStepRef.current = delta;
      settleQuery();
      return;
    }
    if (delta < 0) previous();
    else next();
  };
  useEffect(() => {
    if (queryPending || count === 0 || queuedStepRef.current === 0) return;
    const delta = queuedStepRef.current;
    queuedStepRef.current = 0;
    if (delta < 0) previous();
    else next();
    clearPendingStep();
  }, [clearPendingStep, count, next, previous, queryPending]);
  // Hand focus back to whatever had it, usually the composer, so closing a search leaves the reader
  // typing. Declared above the focus effect so it reads `activeElement` before the field takes it.
  const barRef = useRef<HTMLDivElement>(null);
  const originRef = useRef<HTMLElement | null>(null);
  useEffect(() => {
    const active = document.activeElement as HTMLElement | null;
    // First answer only, and never anything in the bar: StrictMode replays this effect, and by the
    // second run the field has focus, so the bar would try to hand focus back to its own input.
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
      // Only when closing dropped focus on the floor. Anywhere else and the reader moved it.
      const focused = document.activeElement;
      if (focused !== null && focused !== document.body) return;
      origin.focus();
    };
  }, []);

  // Capture on the window so closing the bar cannot carry on to another bare-Escape action. A
  // modal owns Escape while it backgrounds the scope; a transient popover/menu/listbox owns its
  // first Escape. Persistent monitor panels deliberately do not trap the find bar.
  useEffect(() => {
    const onEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || isImeComposing(event)) return;
      if (isSurfaceBackgrounded(`[${FIND_SCOPE_ATTRIBUTE}]`)) return;
      if (resolveDismissiblePortalSurfaces(resolveFindScope()).length > 0)
        return;
      event.preventDefault();
      event.stopPropagation();
      close();
    };
    window.addEventListener("keydown", onEscape, true);
    return () => window.removeEventListener("keydown", onEscape, true);
  }, [close]);

  // Every press of the chord, not just the one that opened the bar, selects the current query.
  // biome-ignore lint/correctness/useExhaustiveDependencies: each token requests a fresh focus/select.
  useEffect(() => {
    const input = inputRef.current;
    if (!input) return;
    input.focus();
    if (!restoreSelection(input)) input.select();
  }, [focusToken, restoreSelection]);

  const searching = query.length > 0;
  const empty = !queryPending && searching && count === 0;
  // A pending query has no count of its own yet, so the settled one's zero must not disable the
  // walk: `stepWhenSettled` queues the press and runs it once the count arrives.
  const canStep = searching && (count > 0 || queryPending);
  const counter =
    searching && !queryPending
      ? `${count === 0 ? 0 : active + 1}/${count}${capped ? "+" : ""}`
      : "";

  return (
    // `data-find-skip` keeps the bar out of its own index: without it every keystroke finds itself.
    <div
      ref={barRef}
      data-find-skip=""
      // biome-ignore lint/a11y/useSemanticElements: this landmark contains the field and its navigation controls.
      role="search"
      aria-label={t("shell.find.label")}
      className="find-bar-surface fixed top-[calc(var(--studio-content-top-inset,0px)+3.5rem)] right-4 z-50 flex h-13 w-[22.25rem] max-w-[calc(100vw-2rem)] items-center gap-1 rounded-full pr-4 pl-5 sm:w-[28.25rem]"
    >
      <input
        ref={inputRef}
        type="text"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            // The Enter committing an IME candidate arrives here too; taken, it walks to the next
            // match and throws away the word.
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
        className={cn(
          "min-w-0 flex-1 bg-transparent text-ui-15 outline-none placeholder:text-muted-foreground",
          empty && "text-destructive",
        )}
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
        <ArrowUpIcon strokeWidth={1.75} className="size-[18px]" />
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
        <ArrowDownIcon strokeWidth={1.75} className="size-[18px]" />
      </Button>
      <Button
        variant="ghost"
        size="icon"
        className={FIND_BUTTON_CLASS}
        onClick={close}
        aria-label={t("shell.find.close")}
        title={t("shell.find.close")}
      >
        <HugeiconsIcon icon={Cancel01Icon} className="size-[18px]" />
      </Button>
    </div>
  );
}
