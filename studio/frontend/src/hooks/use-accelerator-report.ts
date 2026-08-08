// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { useEffect, useState } from "react";
import {
  type AcceleratorReport,
  parseAcceleratorReport,
} from "./accelerator-report";

// Deliberately NOT part of useHardwareInfo. That hook's `include_details=true` response is
// read by Export, Video and onboarding, and the accelerator block costs a child interpreter
// on the backend to import the native packages somewhere they cannot poison anything. Only
// the Settings surfaces ask for it, and only while the Settings dialog is mounted.
const ENDPOINT =
  "/api/system/hardware?include_details=true&include_accelerators=true";

// Module-level cache so the About section and the banner share one request.
let cached: AcceleratorReport | null = null;
let fetchPromise: Promise<AcceleratorReport | null> | null = null;

export function invalidateAcceleratorReport() {
  cached = null;
  fetchPromise = null;
}

async function fetchOnce(): Promise<AcceleratorReport | null> {
  if (cached) return cached;
  if (fetchPromise) return fetchPromise;

  fetchPromise = (async () => {
    try {
      const res = await authFetch(ENDPOINT);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      cached = parseAcceleratorReport(data?.accelerators);
      return cached;
    } catch {
      // Reset so a later mount retries; a failed read is "unknown", and null renders as
      // no section and no banner rather than as an all-clear.
      fetchPromise = null;
      return null;
    }
  })();

  return fetchPromise;
}

/**
 * Optimized-kernel health, or null while unknown.
 *
 * Null covers all three of "still loading", "the read failed" and "this backend predates
 * the field". None of them may render as healthy, and none of them may render a banner.
 */
export function useAcceleratorReport(): AcceleratorReport | null {
  const [report, setReport] = useState<AcceleratorReport | null>(cached);

  useEffect(() => {
    let cancelled = false;
    fetchOnce().then((next) => {
      if (!cancelled) setReport(next);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return report;
}
