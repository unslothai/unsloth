// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The user's saved VRAM Budget fraction, live.
 *
 * Four places already did this by hand -- `use-model-memory.ts`, and three
 * effects in `model-config-page.tsx` -- each with its own `useState` plus a
 * subscribe-and-load effect. This is the fifth caller, the Hub's fit badge, and
 * copying the pattern a fifth time is how the surfaces drift apart again: the
 * badge and the memory bar sit on the SAME ROW, and the whole point of reading
 * the fraction here is that they must agree about it.
 *
 * `null` until the first answer arrives, and `null` forever on a backend too old
 * to serve `/api/settings/vram-budget`. Callers pass it straight through to
 * `classifyGgufFit`, whose `budgetFraction` treats absent as "use the shared
 * default" rather than as "the whole card".
 */

import { useEffect, useState } from "react";

import {
  loadVramBudgetSettings,
  subscribeVramBudgetSettings,
} from "@/features/settings/api/vram-budget";

export function useVramBudgetFraction(): number | null {
  const [fraction, setFraction] = useState<number | null>(null);

  useEffect(() => {
    let alive = true;
    // Subscribe BEFORE loading. A save committed while the GET is in the air
    // would otherwise be missed entirely, leaving the badge scoring against a
    // fraction the user has already replaced.
    const unsubscribe = subscribeVramBudgetSettings((settings) => {
      if (alive) setFraction(settings.fraction);
    });
    void loadVramBudgetSettings()
      .then((settings) => {
        // Null on a backend predating the route; the default covers it.
        if (alive && settings) setFraction(settings.fraction);
      })
      .catch(() => {
        // Stays null, which is the "use the shared default" answer. Not a
        // failure worth surfacing: the badge is advisory and the bar beside it
        // reports the same fallback.
      });
    return () => {
      alive = false;
      unsubscribe();
    };
  }, []);

  return fraction;
}
