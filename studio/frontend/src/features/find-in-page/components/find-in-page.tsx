// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  LazyImportBoundary,
  LazyImportFailure,
} from "@/components/lazy-import-boundary";

import { useT } from "@/i18n";

import {
  isImeComposing,
  isSurfaceBackgrounded,
  useShortcut,
} from "@/features/settings";
import {
  type ReactNode,
  Suspense,
  lazy,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { FIND_SCOPE_ATTRIBUTE } from "../lib/find-attributes.ts";

const DISMISSIBLE_SURFACE_SELECTOR =
  '[data-slot="popover-content"], [role="menu"], [role="listbox"]';

function hasOpenDismissibleSurface(): boolean {
  const scope = document.querySelector(`[${FIND_SCOPE_ATTRIBUTE}]`);
  return [...document.querySelectorAll(DISMISSIBLE_SURFACE_SELECTOR)].some(
    (surface) =>
      scope?.contains(surface) !== true &&
      surface.getAttribute("data-state") !== "closed",
  );
}

const FindBar = lazy(() => import("./find-bar-loader.tsx"));

function FindBarAfterComposition({
  handoffBlock,
  children,
}: {
  handoffBlock: Promise<void> | null;
  children: ReactNode;
}) {
  if (handoffBlock) {
    throw handoffBlock;
  }
  return children;
}

function FindBarLoading({
  query,
  setQuery,
  close,
  focusToken,
  rememberSelection,
  queueStep,
  beginComposition,
  endComposition,
}: {
  query: string;
  setQuery: (query: string) => void;
  close: () => void;
  focusToken: number;
  rememberSelection: (input: HTMLInputElement) => void;
  queueStep: (delta: -1 | 1) => void;
  beginComposition: () => void;
  endComposition: () => void;
}) {
  const t = useT();
  const inputRef = useRef<HTMLInputElement>(null);
  // biome-ignore lint/correctness/useExhaustiveDependencies: focus per token.
  useLayoutEffect(() => {
    const input = inputRef.current;
    if (!input) return;
    input.focus();
    input.select();
    return () => rememberSelection(input);
  }, [focusToken, rememberSelection]);

  useEffect(() => {
    const onEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || isImeComposing(event)) return;
      if (isSurfaceBackgrounded(`[${FIND_SCOPE_ATTRIBUTE}]`)) return;
      if (hasOpenDismissibleSurface()) return;
      event.preventDefault();
      event.stopPropagation();
      close();
    };
    window.addEventListener("keydown", onEscape, true);
    return () => window.removeEventListener("keydown", onEscape, true);
  }, [close]);

  return (
    <div
      data-find-skip=""
      data-testid="find-in-page-loading"
      // biome-ignore lint/a11y/useSemanticElements: loading shell is a search landmark.
      role="search"
      aria-busy="true"
      aria-label={t("shell.find.label")}
      className="find-bar-surface fixed top-[calc(var(--studio-content-top-inset,0px)+3.5rem)] right-4 z-50 flex h-13 max-w-[calc(100vw-2rem)] items-center rounded-full px-5"
    >
      <input
        ref={inputRef}
        type="text"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        onCompositionStart={beginComposition}
        onCompositionEnd={endComposition}
        onSelect={(event) => rememberSelection(event.currentTarget)}
        onKeyDown={(event) => {
          if (event.key !== "Enter" || isImeComposing(event.nativeEvent))
            return;
          event.preventDefault();
          queueStep(event.shiftKey ? -1 : 1);
        }}
        placeholder={t("shell.find.label")}
        aria-label={t("shell.find.label")}
        spellCheck={false}
        autoComplete="off"
        autoCorrect="off"
        className="w-40 min-w-0 bg-transparent text-ui-15 outline-none placeholder:text-muted-foreground sm:w-64"
      />
    </div>
  );
}

/**
 * Lightweight controller; the bar and search engine load only when opened.
 */
export function FindInPage({ enabled = true }: { enabled?: boolean }) {
  const t = useT();
  // Session state belongs to the mounted shell: closing keeps the query, while signing out and
  // unmounting the shell drops it without keeping a module-global user value.
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [focusToken, setFocusToken] = useState(0);
  const loadingSelectionRef = useRef<{
    start: number;
    end: number;
    direction: "forward" | "backward" | "none";
  } | null>(null);

  const [pendingStep, setPendingStep] = useState<{
    query: string;
    delta: -1 | 1;
  } | null>(null);

  const [handoffBlock, setHandoffBlock] = useState<Promise<void> | null>(null);
  const releaseHandoffRef = useRef<(() => void) | null>(null);

  const releaseHandoffTimerRef = useRef<number | null>(null);
  const beginLoadingComposition = useCallback(() => {
    if (releaseHandoffTimerRef.current !== null) {
      window.clearTimeout(releaseHandoffTimerRef.current);
      releaseHandoffTimerRef.current = null;
    }
    if (releaseHandoffRef.current) {
      return;
    }
    let release: () => void = () => undefined;
    const block = new Promise<void>((resolve) => {
      release = resolve;
    });
    releaseHandoffRef.current = release;
    setHandoffBlock(block);
  }, []);
  const endLoadingComposition = useCallback(() => {
    releaseHandoffTimerRef.current = window.setTimeout(() => {
      releaseHandoffTimerRef.current = null;
      releaseHandoffRef.current?.();
      releaseHandoffRef.current = null;
      setHandoffBlock(null);
    }, 0);
  }, []);
  useEffect(
    () => () => {
      if (releaseHandoffTimerRef.current !== null) {
        window.clearTimeout(releaseHandoffTimerRef.current);
      }
      releaseHandoffRef.current?.();
      releaseHandoffRef.current = null;
    },
    [],
  );
  const originRef = useRef<HTMLElement | null>(null);
  const requestFocus = useCallback(() => {
    const active = document.activeElement;
    if (
      originRef.current === null &&
      active instanceof HTMLElement &&
      active.closest('[role="search"]') === null
    ) {
      originRef.current = active;
    }
    loadingSelectionRef.current = null;

    setPendingStep(null);
    setOpen(true);
    setFocusToken((token) => token + 1);
  }, []);
  const rememberLoadingSelection = useCallback((input: HTMLInputElement) => {
    loadingSelectionRef.current = {
      start: input.selectionStart ?? input.value.length,
      end: input.selectionEnd ?? input.value.length,
      direction: input.selectionDirection ?? "none",
    };
  }, []);

  const queueLoadingStep = useCallback(
    (delta: -1 | 1) => setPendingStep({ query, delta }),
    [query],
  );
  const clearPendingStep = useCallback(() => setPendingStep(null), []);
  const restoreLoadingSelection = useCallback((input: HTMLInputElement) => {
    const selection = loadingSelectionRef.current;
    if (!selection) return false;
    const length = input.value.length;
    input.setSelectionRange(
      Math.min(selection.start, length),
      Math.min(selection.end, length),
      selection.direction,
    );
    return true;
  }, []);
  const close = useCallback(() => {
    const origin = originRef.current;
    originRef.current = null;
    setOpen(false);
    requestAnimationFrame(() => {
      const active = document.activeElement;
      if (
        origin?.isConnected &&
        typeof origin.focus === "function" &&
        (active === null || active === document.body)
      ) {
        origin.focus();
      }
    });
  }, []);

  // The chord works from text fields, including the composer and an already-open find input.
  useShortcut("findInPage", requestFocus, {
    enabled,
    // A modal backgrounds the shell and owns the chord while its surface is active.
    claims: () => !isSurfaceBackgrounded(`[${FIND_SCOPE_ATTRIBUTE}]`),
  });

  if (!enabled || !open) return null;
  return (
    <LazyImportBoundary
      fallback={
        <LazyImportFailure
          message={t("settings.dialog.panelFailed")}
          reloadLabel={t("settings.dialog.panelReload")}
          dismissLabel={t("common.close")}
          onDismiss={close}
          testId="find-in-page-load-failure"
          className="fixed top-3 right-3 z-[100] max-w-xs rounded-xl border border-border bg-popover p-4 text-popover-foreground shadow-lg"
        />
      }
    >
      <Suspense
        fallback={
          <FindBarLoading
            query={query}
            setQuery={setQuery}
            close={close}
            focusToken={focusToken}
            rememberSelection={rememberLoadingSelection}
            queueStep={queueLoadingStep}
            beginComposition={beginLoadingComposition}
            endComposition={endLoadingComposition}
          />
        }
      >
        <FindBarAfterComposition handoffBlock={handoffBlock}>
          <FindBar
            query={query}
            setQuery={setQuery}
            close={close}
            focusToken={focusToken}
            restoreSelection={restoreLoadingSelection}
            pendingStep={pendingStep}
            clearPendingStep={clearPendingStep}
          />
        </FindBarAfterComposition>
      </Suspense>
    </LazyImportBoundary>
  );
}
