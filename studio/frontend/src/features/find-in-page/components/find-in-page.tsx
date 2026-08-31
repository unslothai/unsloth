// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useShortcut } from "@/features/settings";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
// The 02 arrows, not the 01 pair: 01 is a bare chevron, and these read as arrows everywhere else.
import {
  ArrowDown02Icon,
  ArrowUp02Icon,
  Cancel01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useRef } from "react";
import { useFindInPage } from "../hooks/use-find-in-page.ts";
import { MAX_MATCHES } from "../lib/find-text-index.ts";
import { useFindInPageStore } from "../stores/find-in-page-store.ts";

/**
 * The find bar, and the chord that raises it.
 *
 * Split in two: this component is always mounted and holds nothing but the shortcut, while
 * `FindBar` owns the index, the observer and the highlights and exists only while the bar is open.
 * A Studio nobody is searching runs one keydown listener and no engine.
 */
export function FindInPage({ enabled = true }: { enabled?: boolean }) {
  const open = useFindInPageStore((state) => state.open);
  const requestFocus = useFindInPageStore((state) => state.requestFocus);

  // Not `skipInTextFields`: the chord has to work from the composer, and pressing it inside the
  // find field is how a find bar is asked to start over.
  useShortcut("findInPage", () => requestFocus(), { enabled });

  if (!enabled || !open) return null;
  return <FindBar />;
}

/**
 * Keep the caret in the field when a walk button is clicked. Without this, one click on the down
 * arrow and Enter stops working, because the field no longer has the key.
 */
function keepFocusInField(event: { preventDefault: () => void }): void {
  event.preventDefault();
}

/**
 * Show the start of the query again once the field loses focus. A query longer than the field
 * scrolls as it is typed, which is right while typing and wrong once the caret is gone: what is
 * left is the tail of a word. The caret goes home too, so the scroll does not spring back.
 */
function rewindToStart(event: { currentTarget: HTMLInputElement }): void {
  const input = event.currentTarget;
  input.setSelectionRange(0, 0);
  input.scrollLeft = 0;
}

/**
 * The walk and close buttons. The ghost variant's `--muted/50` hover resolves to within a shade of
 * this bar's background, so in dark mode nothing visibly happens; a plain wash reads on either
 * surface. Same one the outline variant uses.
 */
const FIND_BUTTON_CLASS = "size-8 hover:bg-black/[0.06] dark:hover:bg-white/10";

function FindBar() {
  const t = useT();
  const query = useFindInPageStore((state) => state.query);
  const setQuery = useFindInPageStore((state) => state.setQuery);
  const close = useFindInPageStore((state) => state.close);
  const focusToken = useFindInPageStore((state) => state.focusToken);
  const { count, active, truncated, next, previous } = useFindInPage(query);
  const inputRef = useRef<HTMLInputElement>(null);

  // Every press of the chord, not just the one that opened the bar: pressing it again selects what
  // is in the field so the next keystroke replaces the last search.
  useEffect(() => {
    const input = inputRef.current;
    if (!input) return;
    input.focus();
    input.select();
  }, [focusToken]);

  const searching = query.length > 0;
  const empty = searching && count === 0;
  const counter = searching
    ? `${count === 0 ? 0 : active + 1}/${count}${count >= MAX_MATCHES ? "+" : ""}`
    : "";

  return (
    // `data-find-skip` keeps the bar out of its own index: without it every keystroke finds itself.
    <div
      data-find-skip=""
      role="search"
      aria-label={t("shell.find.label")}
      className="find-bar-surface absolute top-3 right-4 z-50 flex h-13 items-center gap-1 rounded-full pr-4 pl-5"
    >
      <input
        ref={inputRef}
        type="text"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            // Stopped as well as prevented: bare Escape also declines a tool request, and closing
            // a search must not answer a prompt still on screen.
            event.preventDefault();
            event.stopPropagation();
            close();
            return;
          }
          if (event.key === "Enter") {
            event.preventDefault();
            if (event.shiftKey) previous();
            else next();
          }
        }}
        onBlur={rewindToStart}
        // The same string the landmark is labelled with: one phrase, one key to translate.
        placeholder={t("shell.find.label")}
        aria-label={t("shell.find.label")}
        spellCheck={false}
        autoComplete="off"
        autoCorrect="off"
        className={cn(
          "w-64 bg-transparent text-[15px] outline-none placeholder:text-muted-foreground",
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
        disabled={count === 0}
        onMouseDown={keepFocusInField}
        onClick={previous}
        aria-label={t("shell.find.previous")}
        title={t("shell.find.previous")}
      >
        <HugeiconsIcon icon={ArrowUp02Icon} className="size-[18px]" />
      </Button>
      <Button
        variant="ghost"
        size="icon"
        className={FIND_BUTTON_CLASS}
        disabled={count === 0}
        onMouseDown={keepFocusInField}
        onClick={next}
        aria-label={t("shell.find.next")}
        title={t("shell.find.next")}
      >
        <HugeiconsIcon icon={ArrowDown02Icon} className="size-[18px]" />
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
