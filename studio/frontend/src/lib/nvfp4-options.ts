// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Shown on the disabled NVFP4 option of a Blackwell host whose backend has NVFP4 switched off. */
export const NVFP4_DISABLED_REASON =
  "NVFP4 is off in this build. Start the Unsloth backend with UNSLOTH_NVFP4_DIFFUSION=1 to enable it.";

/** A select option; a third entry marks it disabled and is the reason shown on hover. */
export type PrecisionOption = [string, string] | [string, string, string];

function isNvfp4(value: string): boolean {
  return value.trim().toLowerCase() === "nvfp4";
}

/**
 * NVFP4 in place while the backend enables it. While it is off, a host that could run it
 * (``nvfp4Capable``: every card is Blackwell) keeps the option, disabled with the reason;
 * any other host does not list it.
 */
export function withNvfp4Option<T extends readonly [string, string]>(
  options: readonly T[],
  nvfp4Enabled: boolean,
  nvfp4Capable = false,
): PrecisionOption[] {
  if (nvfp4Enabled) return options.map(([value, label]) => [value, label]);
  if (nvfp4Capable) {
    return options.map(([value, label]) =>
      isNvfp4(value) ? [value, label, NVFP4_DISABLED_REASON] : [value, label],
    );
  }
  return options.filter(([value]) => !isNvfp4(value)).map(([value, label]) => [value, label]);
}

/** A held `nvfp4` falls back to `auto` once the backend says it is off; before `known`, nothing resets. */
export function nvfp4SelectionFallback<T extends string>(
  value: T,
  nvfp4Known: boolean,
  nvfp4Enabled: boolean,
): T | "auto" {
  if (!nvfp4Known || nvfp4Enabled) return value;
  return isNvfp4(value) ? "auto" : value;
}
