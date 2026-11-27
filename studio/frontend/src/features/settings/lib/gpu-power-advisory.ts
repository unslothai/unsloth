// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Returns whether a live GPU sample is close enough to its configured power
 * limit to warrant a stability reminder.
 *
 * Power telemetry is optional across vendors and drivers. Treat missing,
 * non-finite, or non-positive values as unknown rather than manufacturing a
 * warning from an incomplete sample.
 */
export function gpuPowerLimitAdvisory(
  powerDrawW: number | null | undefined,
  powerLimitW: number | null | undefined,
): boolean {
  if (
    typeof powerDrawW !== "number" ||
    typeof powerLimitW !== "number" ||
    !Number.isFinite(powerDrawW) ||
    !Number.isFinite(powerLimitW) ||
    powerDrawW < 0 ||
    powerLimitW <= 0
  ) {
    return false;
  }
  return powerDrawW / powerLimitW >= 0.95;
}
