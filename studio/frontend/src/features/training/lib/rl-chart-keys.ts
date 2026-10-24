// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { RlMetricPoint } from "../types/runtime";

/** Metric keys worth a chart. KL stays out while every value is 0: with beta 0 TRL never
 * measures it, so a flat line would read as "no drift" when nothing was computed. */
export function rlChartKeys(history: readonly RlMetricPoint[]): Set<string> {
  const keys = new Set<string>();
  let klMeasured = false;
  for (const point of history) {
    for (const [key, value] of Object.entries(point.values)) {
      if (key === "kl") {
        klMeasured ||= value !== 0;
      } else {
        keys.add(key);
      }
    }
  }
  if (klMeasured) {
    keys.add("kl");
  }
  return keys;
}
