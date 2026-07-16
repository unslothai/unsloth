// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo, useState } from "react";
import type { TourStep } from "../types";

export const TOUR_OPEN_EVENT = "omx:tour:open";

export type TourOpenDetail = {
  id?: string;
};

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
    function onOpen(e: Event) {
      const ce = e as CustomEvent<TourOpenDetail>;
      if (ce.detail?.id && ce.detail.id !== id) return;
      if (steps.length === 0) return;
      setOpen(true);
    }
    window.addEventListener(TOUR_OPEN_EVENT, onOpen);
    return () => window.removeEventListener(TOUR_OPEN_EVENT, onOpen);
  }, [enabled, hasRuntime, id, steps.length]);

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

