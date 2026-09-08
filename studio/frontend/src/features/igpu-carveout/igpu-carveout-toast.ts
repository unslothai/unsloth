// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The integrated-GPU memory advice, as a toast rather than a modal.
//
// It was an AlertDialog first, which was wrong for what this is: the model has
// already loaded, nothing is being asked, and there is no decision to block on. A
// modal dims the app and takes the next click no matter what the user was doing,
// for a notice about a setting they cannot change from here anyway. Every other
// "worth knowing, carry on" notice in Studio is a toast (the download start, the
// Xet 0% explanation, the kept-sandbox offer), so this is one too.
//
// No store and no mounted component: the toast owns its own text for as long as it
// is up, so there is no state to keep or to invalidate, which is what the dialog's
// store existed for.

import { toast } from "@/lib/toast";

import { dismissCarveoutNotice } from "./api/igpu-carveout-notice";
import { parseCarveoutAdvice } from "./types";

/** One id for the notice, so a second load REPLACES rather than stacks.
 *
 * The dialog deliberately refused to swap its text under the user's cursor. A toast
 * is the other way round: it is transient and it describes the model that just
 * loaded, so the newest load is the one worth showing, and a stack of two would sit
 * over the composer describing a model that is no longer resident. */
export const IGPU_CARVEOUT_TOAST_ID = "igpu-carveout-notice";

/** Longer than the Toaster's 5s default, like the other explanatory toasts, and
 *  longer than those because this one carries an action worth reading first. */
export const IGPU_CARVEOUT_NOTICE_DURATION_MS = 12000;

export const IGPU_CARVEOUT_NOTICE_TITLE = "This model could run faster";

/** Hand a load response's advice field to the notice. Safe to call on every load.
 *
 * Absent on nearly every load, so callers pass the field through unconditionally
 * and anything malformed is treated as no advice at all: the notice quotes numbers,
 * and a partial payload must produce no toast rather than one reading "undefined
 * GB". */
export function showCarveoutAdvice(value: unknown): void {
  const advice = parseCarveoutAdvice(value);
  if (!advice) {
    // This load has nothing to advise, so the previous load's numbers have stopped
    // being true. Same moment the store used to clear its copy.
    toast.dismiss(IGPU_CARVEOUT_TOAST_ID);
    return;
  }
  toast.info(IGPU_CARVEOUT_NOTICE_TITLE, {
    id: IGPU_CARVEOUT_TOAST_ID,
    description: advice.message,
    duration: IGPU_CARVEOUT_NOTICE_DURATION_MS,
    action: {
      label: "Don't show again",
      // Fire and forget: the toast is gone by the time this resolves, and
      // dismissCarveoutNotice swallows its own failures. Worst case the notice
      // returns on a later load.
      onClick: () => {
        void dismissCarveoutNotice(advice.current_gb);
      },
    },
  });
}
