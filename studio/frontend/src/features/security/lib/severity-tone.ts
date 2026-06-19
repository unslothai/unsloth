// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { RemoteCodeSeverity } from "../types";

/** Tailwind classes for a severity badge. */
export function severityTone(severity: RemoteCodeSeverity | string): string {
  switch (severity) {
    case "CRITICAL":
    case "HIGH":
      return "border-red-500/30 text-red-700 dark:text-red-300";
    case "MEDIUM":
      return "border-amber-500/30 text-amber-700 dark:text-amber-300";
    default:
      return "border-border text-muted-foreground";
  }
}
