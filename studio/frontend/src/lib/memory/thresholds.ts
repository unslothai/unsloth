// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Every number that decides how a memory figure is COLOURED or JUDGED.
 *
 * These were spread across two features that never referenced each other, so
 * the Load Model panel and the Hub memory bar could describe one load
 * differently while being fed identical bytes. Collecting them here does not
 * make them one number -- two of them measure genuinely different things -- but
 * it does mean the difference is visible in one place instead of being
 * discovered by comparing two screens.
 *
 * No `@/` alias imports: see the note in ./format.ts.
 */

/**
 * Above this share of the capacity, a fit is reported as tight rather than clean.
 *
 * This is a VERDICT threshold: it answers "will this load fit", and the answer
 * changes at 85% because the remaining headroom stops being enough for the
 * allocator's slack. Distinct from the pressure ramp below, which answers "how
 * full does this look" -- the two are different questions and it is correct for
 * them to have different numbers.
 */
export const MEMORY_FIT_TIGHT_RATIO = 0.85;

/**
 * Fill fraction (%) at which the bar leaves the accent colour.
 *
 * Studio's live meters step at 70/90 (`resources-tab.tsx`), but those show usage
 * as it happens, where a sustained 70% is worth flagging. This is a reservation
 * you can see coming, so it holds the accent until 80.
 */
export const PRESSURE_HIGH_PCT = 80;

/** Fill fraction (%) at which the bar turns destructive. */
export const PRESSURE_CRITICAL_PCT = 90;

/**
 * The share of a card a load may claim when the user's VRAM Budget setting has
 * not been read yet, or the backend is too old to serve it.
 *
 * 0.97, matching the loader's own `_CTX_FIT_VRAM_FRACTION`
 * (`core/inference/llama_cpp.py`), because a fit verdict drawn against a
 * different fraction than the one admission actually uses is wrong by
 * construction.
 *
 * Explicitly NOT 0.90. The Hub bar used that, and it is not merely a more
 * cautious guess -- it is a value this project already measured and reverted:
 *
 *   "0.90 dropped 91-94% fits to CPU offload, #5106"  (llama_cpp.py)
 *
 * A bar that warns at 0.90 tells users a load will not fit when the loader will
 * happily admit it, which is the same class of wrong answer as the false "fits"
 * the bar exists to prevent, pointing the other way.
 */
export const DEFAULT_VRAM_BUDGET_FRACTION = 0.97;
