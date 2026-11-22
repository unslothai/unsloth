// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** No `@/` alias imports: see ./format.ts. */

/** Verdict threshold for "will it fit", distinct from the pressure ramp below. */
export const MEMORY_FIT_TIGHT_RATIO = 0.85;

/** Live meters step at 70/90; a reservation holds the accent until 80. */
export const PRESSURE_HIGH_PCT = 80;

export const PRESSURE_CRITICAL_PCT = 90;

/**
 * Used until the VRAM Budget setting is read. Matches the loader's `_CTX_FIT_VRAM_FRACTION`.
 * Not 0.90: "0.90 dropped 91-94% fits to CPU offload, #5106".
 */
export const DEFAULT_VRAM_BUDGET_FRACTION = 0.97;
