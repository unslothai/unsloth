// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  useSyncExternalStore,
} from "react";
import type { TourStep } from "../types";

export const TOUR_OPEN_EVENT = "omx:tour:open";

export type TourOpenDetail = {
  id?: string;
};

// Tour ids with a live listener. A page can mount its controller behind a gate (Video) or disable
// it while it loads (Train), and the menu must not offer a tour nothing answers.
const listening = new Map<string, number>();
const listeningSubscribers = new Set<() => void>();

function registerTour(id: string): () => void {
  listening.set(id, (listening.get(id) ?? 0) + 1);
  for (const notify of listeningSubscribers) notify();
  return () => {
    const left = (listening.get(id) ?? 1) - 1;
    if (left > 0) listening.set(id, left);
    else listening.delete(id);
    for (const notify of listeningSubscribers) notify();
  };
}

function subscribeToListening(notify: () => void): () => void {
  listeningSubscribers.add(notify);
  return () => {
    listeningSubscribers.delete(notify);
  };
}

/** Whether opening this tour would reach anyone. False for a null id. */
export function useTourAvailable(id: string | null): boolean {
  return useSyncExternalStore(
    subscribeToListening,
    () => (id === null ? false : listening.has(id)),
    () => false,
  );
}

export function useGuidedTourController({
  id,
  steps,
  enabled = true,
  autoKey,
  autoWhen = false,
}: {
  id: string;
  steps: TourStep[];
  enabled?: boolean;
  autoKey?: string;
  autoWhen?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [hasRuntime, setHasRuntime] = useState(false);

  useEffect(() => setHasRuntime(true), []);

  useEffect(() => {
    if (!hasRuntime || !enabled) return;
    if (!autoKey || !autoWhen) return;
    if (steps.length === 0) return;
    if (localStorage.getItem(autoKey)) return;
    setOpen(true);
  }, [autoKey, autoWhen, enabled, hasRuntime, steps.length]);

  useEffect(() => {
    if (!hasRuntime || !enabled) return;
    if (steps.length === 0) return;
    function onOpen(e: Event) {
      const ce = e as CustomEvent<TourOpenDetail>;
      if (ce.detail?.id && ce.detail.id !== id) return;
      setOpen(true);
    }
    window.addEventListener(TOUR_OPEN_EVENT, onOpen);
    const unregister = registerTour(id);
    return () => {
      window.removeEventListener(TOUR_OPEN_EVENT, onOpen);
      unregister();
    };
  }, [enabled, hasRuntime, id, steps.length]);

  // Keyed on `enabled` alone, so a page going inactive mid-tour drops the tour rather than
  // silently restarting it at step one on return, while a changing step count leaves it alone.
  useEffect(() => {
    if (!enabled) return;
    return () => setOpen(false);
  }, [enabled]);

  const onSkip = useCallback(() => {
    if (!autoKey) return;
    localStorage.setItem(autoKey, "skipped");
  }, [autoKey]);

  const onComplete = useCallback(() => {
    if (!autoKey) return;
    localStorage.setItem(autoKey, "done");
  }, [autoKey]);

  const tourProps = useMemo(
    () => ({
      open,
      onOpenChange: setOpen,
      steps,
      onSkip,
      onComplete,
    }),
    [onComplete, onSkip, open, steps],
  );

  return { open, setOpen, onSkip, onComplete, tourProps };
}

