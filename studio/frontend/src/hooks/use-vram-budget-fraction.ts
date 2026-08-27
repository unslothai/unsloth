// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The user's saved VRAM Budget fraction, live, and read ONCE per session.
 *
 * Several places already did this by hand -- `use-model-memory.ts`, and three
 * effects in `model-config-page.tsx` -- each with its own `useState` plus a
 * subscribe-and-load effect. The Hub's fit badge is the next caller, and copying
 * the pattern again is how the surfaces drift apart: the badge and the memory
 * bar sit on the SAME ROW and must agree about the fraction.
 *
 * The caching here is not an optimization, it is a correctness constraint on a
 * LIST. `loadVramBudgetSettings` deliberately has no read-through cache --
 * `reloadRequired` describes the running process and goes stale on any load --
 * and it clears its shared promise once the request settles, so it only
 * coalesces callers whose requests overlap in time. A Hub catalog mounts a card
 * per repo progressively, through scrolling, filtering and pagination, so one
 * GET per card is what that adds up to, and on a backend predating the route it
 * is one 404 per card.
 *
 * So the FRACTION is cached module-wide, while the settings loader keeps its
 * always-refetch behaviour for the callers that need `reloadRequired`. The cache
 * is kept live by the same change event every other reader already subscribes
 * to, so a save still reaches every mounted card immediately.
 *
 * `null` until the first answer arrives, and `null` forever on a backend too old
 * to serve `/api/settings/vram-budget`. Callers pass it straight to
 * `classifyGgufFit`, whose `budgetFraction` treats absent as "use the shared
 * default" rather than as "the whole card".
 */

import { useEffect, useState } from "react";

import {
  loadVramBudgetSettings,
  subscribeVramBudgetSettings,
} from "@/features/settings/api/vram-budget";

/** The last fraction seen, from any reader. `null` = not answered yet. */
let cachedFraction: number | null = null;
/** In flight, so cards mounting in the same tick share one request. */
let inFlight: Promise<void> | null = null;
/** The route answered 404: there is no fraction to fetch, ever. Without this a
 *  catalog on an older backend retries per card forever. */
let routeAbsent = false;

/** Reset for tests; not used by the app. */
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
      // Null is the route being absent, not a transient failure, so record it
      // rather than asking the next card to find out again.
      if (settings) cachedFraction = settings.fraction;
      else routeAbsent = true;
    })
    .catch(() => {
      // Leaves the cache empty, which is the "use the shared default" answer.
      // Not marked absent: a transient failure should not permanently pin every
      // card to the default for the rest of the session.
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
    // Subscribe BEFORE loading. A save committed while the GET is in the air
    // would otherwise be missed entirely, leaving the badge scoring against a
    // fraction the user has already replaced. This is also what keeps the
    // module cache honest: every writer publishes on this event.
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
