// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** A select option: value, label, and, when set, why it cannot be picked (shown disabled with that reason). */
export type SelectOption = readonly [value: string, label: string, disabledReason?: string | null];

/** Why NVFP4 is greyed out while the backend switch is off: it names the switch rather than hiding the option. */
export const NVFP4_DISABLED_HINT = "Off in this build. Start Studio with UNSLOTH_NVFP4_DIFFUSION=1 to enable it.";

function isNvfp4(value: string): boolean {
  return value.trim().toLowerCase() === "nvfp4";
}

/** NVFP4 stays listed while the backend switch is off, disabled with the flag that turns it on. */
export function withNvfp4Option(options: readonly SelectOption[], nvfp4Enabled: boolean): SelectOption[] {
  if (nvfp4Enabled) return [...options];
  return options.map((option) => (isNvfp4(option[0]) ? [option[0], option[1], NVFP4_DISABLED_HINT] : option));
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
