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
