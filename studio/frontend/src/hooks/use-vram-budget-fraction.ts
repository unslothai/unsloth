// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Cached module-wide because a Hub catalog mounts a card per repo and loadVramBudgetSettings only
 * coalesces overlapping requests. Kept live by the settings change event. `null` until answered,
 * and forever on a backend without the route; classifyGgufFit treats that as the shared default.
 */

import { useEffect, useState } from "react";

import {
  loadVramBudgetSettings,
  subscribeVramBudgetSettings,
} from "@/features/settings/api/vram-budget";

let cachedFraction: number | null = null;
let inFlight: Promise<void> | null = null;
/** 404 means no fraction ever; without this an older backend is retried per card. */
let routeAbsent = false;

export function __resetVramBudgetFractionCache(): void {
  cachedFraction = null;
  inFlight = null;
  routeAbsent = false;
}

function ensureLoaded(): Promise<void> {
  if (cachedFraction !== null || routeAbsent) return Promise.resolve();
  if (inFlight) return inFlight;
  inFlight = loadVramBudgetSettings()
    .then((settings) => {
      // Null is the route being absent, not a transient failure.
      if (settings) cachedFraction = settings.fraction;
      else routeAbsent = true;
    })
    .catch(() => {
      // Not marked absent: a transient failure must not pin every card to the default.
    })
    .finally(() => {
      inFlight = null;
    });
  return inFlight;
}

export function useVramBudgetFraction(): number | null {
  const [fraction, setFraction] = useState<number | null>(cachedFraction);

  useEffect(() => {
    let alive = true;
    // Subscribe BEFORE loading so a save during the GET is not missed.
    const unsubscribe = subscribeVramBudgetSettings((settings) => {
      cachedFraction = settings.fraction;
      routeAbsent = false;
      if (alive) setFraction(settings.fraction);
    });
    void ensureLoaded().then(() => {
      if (alive) setFraction(cachedFraction);
    });
    return () => {
      alive = false;
      unsubscribe();
    };
  }, []);

  return fraction;
}
