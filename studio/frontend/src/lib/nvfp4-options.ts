// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function withNvfp4Option<T extends readonly [string, string]>(
  options: readonly T[],
  nvfp4Enabled: boolean,
): T[] {
  if (nvfp4Enabled) return [...options];
  return options.filter(([value]) => value.trim().toLowerCase() !== "nvfp4");
}

/** A held `nvfp4` falls back to `auto` once the backend says it is off; before `known`, nothing resets. */
export function nvfp4SelectionFallback<T extends string>(
  value: T,
  nvfp4Known: boolean,
  nvfp4Enabled: boolean,
): T | "auto" {
  if (!nvfp4Known || nvfp4Enabled) return value;
  return value.trim().toLowerCase() === "nvfp4" ? "auto" : value;
}
