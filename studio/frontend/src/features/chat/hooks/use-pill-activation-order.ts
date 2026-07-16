// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";

// Tracks which keys are active and their activation order, so opt-in
// composer pills (Canvas, MCP) render in toggle-on order, not a fixed one.
export function usePillActivationOrder(states: Record<string, boolean>): string[] {
  const [order, setOrder] = useState<string[]>(() =>
    Object.keys(states).filter((key) => states[key]),
  );
  // Re-run only when the active/inactive set changes, not on every render.
  const signature = Object.keys(states)
    .map((key) => `${key}:${states[key] ? 1 : 0}`)
    .join(",");
  useEffect(() => {
    setOrder((prev) => {
      const next = prev.filter((key) => states[key]);
      for (const key of Object.keys(states)) {
        if (states[key] && !next.includes(key)) next.push(key);
      }
      const unchanged =
        next.length === prev.length && next.every((key, i) => key === prev[i]);
      return unchanged ? prev : next;
    });
    // states is read fresh inside; signature captures its boolean values.
  }, [signature]);
  return order;
}
