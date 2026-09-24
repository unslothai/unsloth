// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The backend's NVFP4 switch (UNSLOTH_NVFP4_DIFFUSION, reported as `/api/system.nvfp4_diffusion`)
// decides whether the image and video pickers offer NVFP4 at all. While it is off the backend 400s
// any request naming it, so the select must not list an option that can only fail.

/** `options` as given when NVFP4 is enabled, otherwise without the `nvfp4` entry. */
export function withNvfp4Option<T extends readonly [string, string]>(
  options: readonly T[],
  nvfp4Enabled: boolean,
): T[] {
  if (nvfp4Enabled) return [...options];
  return options.filter(([value]) => value.trim().toLowerCase() !== "nvfp4");
}

/** The select value to keep once the backend has said whether it accepts NVFP4. A held `nvfp4`
 *  (reseeded from an earlier load, before a restart turned the switch off) would sit in a select
 *  that no longer lists it, blank, and the next load or Reapply would send it for a 400, so it
 *  falls back to `auto`. Until system info arrives (`known` false) nothing is reset: the switch
 *  reads off by default and would otherwise wipe a real NVFP4 pick on every page load. */
export function nvfp4SelectionFallback<T extends string>(
  value: T,
  nvfp4Known: boolean,
  nvfp4Enabled: boolean,
): T | "auto" {
  if (!nvfp4Known || nvfp4Enabled) return value;
  return value.trim().toLowerCase() === "nvfp4" ? "auto" : value;
}
