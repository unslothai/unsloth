// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function voicePillIconClass(states: readonly (string | null)[]): string {
  const known = states.filter((s): s is string => s !== null);
  return known.length > 0 && known.every((s) => s === "loaded")
    ? "text-emerald-500"
    : "text-muted-foreground";
}
