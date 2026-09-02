// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  isImeComposing,
  isSurfaceBackgrounded,
  useShortcut,
} from "@/features/settings";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
// lucide for the two arrows, which is where #10129 moved every directional arrow in the app: the
// hugeicons shaft is a bezier that bows out rather than meeting at a point, and after that change
// there is no other use of the pair left to match.
import { ArrowDownIcon, ArrowUpIcon } from "lucide-react";
import { useEffect, useRef } from "react";
import { useFindInPage } from "../hooks/use-find-in-page.ts";
import { resolveFindScope, resolvePortalSurfaces } from "../lib/find-dom.ts";
import { FIND_SCOPE_ATTRIBUTE } from "../lib/find-text-index.ts";
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
  const reset = useFindInPageStore((state) => state.reset);

  // Leaving the shell for good, which on the web means signing out: the store is module-global and
  // keeps the query across a close, so without this the next person to sign in in the same tab is
  // handed the last one's search. Unmount, not `enabled`, which a dialog also turns off.
  useEffect(() => reset, [reset]);

  // Not `skipInTextFields`: the chord has to work from the composer, and pressing it inside the
  // find field is how a find bar is asked to start over.
  useShortcut("findInPage", requestFocus, {
    enabled,
    // Every modal, not just Settings: Radix marks the shell `aria-hidden`/`inert` while one is up,
    // so a bar behind it is unreachable. As `claims`, not a return from the handler, which runs
    // after the event is prevented: declining there would leave the chord dead and native find
    // suppressed.
    claims: () => !isSurfaceBackgrounded(`[${FIND_SCOPE_ATTRIBUTE}]`),
  });

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
  const { count, active, capped, truncated, next, previous } =
    useFindInPage(query);
  const inputRef = useRef<HTMLInputElement>(null);

  // Hand focus back to whatever had it, usually the composer, so closing a search leaves the reader
  // typing. Declared above the focus effect so it reads `activeElement` before the field takes it.
  const barRef = useRef<HTMLDivElement>(null);
  const originRef = useRef<HTMLElement | null>(null);
  useEffect(() => {
    const active = document.activeElement as HTMLElement | null;
    // First answer only, and never anything in the bar: StrictMode replays this effect, and by the
    // second run the field has focus, so the bar would try to hand focus back to its own input.
    // The bar's own element, not `data-find-skip`, which the composer carries too.
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

  // Escape closes the bar for as long as it is open, wherever focus is.
  //
  // On the bar it only reached presses that started inside it, so clicking a message to read it
  // left the reader with a bar Escape would not close -- and with a tool request waiting, that
  // same unprevented Escape carried on to `declineToolRequest` (bare Escape, and
  // `isTextEntryFocused` is false on a message body) and denied the call. Closing a find bar must
  // not be able to answer a tool request.
  //
  // Capture on the window, so it runs before the registry's own keydown listener rather than
  // racing it, and two things are deliberately left alone: a modal above the bar backgrounds the
  // scope and owns Escape, and an open popover, menu or listbox is dismissed by its own Escape
  // first. Composition is left to the IME, as before.
  useEffect(() => {
    const onEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || isImeComposing(event)) return;
      if (isSurfaceBackgrounded(`[${FIND_SCOPE_ATTRIBUTE}]`)) return;
      if (resolvePortalSurfaces(resolveFindScope()).length > 0) return;
      event.preventDefault();
      event.stopPropagation();
      close();
    };
    window.addEventListener("keydown", onEscape, true);
    return () => window.removeEventListener("keydown", onEscape, true);
  }, [close]);

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
    ? `${count === 0 ? 0 : active + 1}/${count}${capped ? "+" : ""}`
    : "";

  return (
    // `data-find-skip` keeps the bar out of its own index: without it every keystroke finds itself.
    <div
      ref={barRef}
      data-find-skip=""
      role="search"
      aria-label={t("shell.find.label")}
      // Escape is handled on the window for the lifetime of the bar (see the effect above), which
      // covers the walk buttons and the field as well as everything outside it, so there is no
      // handler here to also reach the presses that start inside.
      className="find-bar-surface fixed top-[calc(var(--studio-content-top-inset,0px)+3.5rem)] right-4 z-50 flex h-13 max-w-[calc(100vw-2rem)] items-center gap-1 rounded-full pr-4 pl-5"
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
          // Narrow by default so the whole bar clears a small window, wide once there is room.
          // `min-w-0` lets the field, rather than the bar, give way if it still does not fit.
          // `text-ui-15`, not a raw pixel size, which ignores the UI font size preference.
          "w-40 min-w-0 bg-transparent text-ui-15 outline-none placeholder:text-muted-foreground sm:w-64",
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
        <ArrowUpIcon strokeWidth={1.75} className="size-[18px]" />
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
