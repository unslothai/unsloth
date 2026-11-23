// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The global panel's poll parks while hidden (null); pickers keep polling at the hidden pace. */
export function enginePollDelayMs(
  hidden: boolean,
  installing: boolean,
  background: boolean,
): number | null {
  if (hidden) return background ? null : 30000;
  return installing ? 1000 : background ? 30000 : 3000;
}
