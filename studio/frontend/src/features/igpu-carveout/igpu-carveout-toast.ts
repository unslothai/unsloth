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

/** The action, outlined and right-aligned rather than sonner's filled default.
 *
 * Two things it fixes, both visible in a screenshot before anything else. A solid
 * button reads as the thing to do, and the thing to do here is nothing: the model
 * has loaded, and the setting is in someone's firmware. An outline says "a control"
 * without saying "act now".
 *
 * And the shared toast CSS puts an action at `justify-self: start`, which is right
 * when the description is one line and wrong under six: the button ends up floating
 * mid-toast, aligned to neither edge. `!justify-self-end` walks it out to the text
 * column's right edge. Overriding here rather than in that shared rule on purpose:
 * every other toast's action IS the thing to do. */
export const IGPU_CARVEOUT_ACTION_CLASS =
  "!justify-self-end !h-[26px] !border !border-border !bg-transparent !px-3 " +
  "!font-medium !text-foreground hover:!bg-accent";

/** The model the notice on screen is about, so an unload can take it down.
 *
 * A path rather than a flag: several models can be resident, and unloading one of
 * the others leaves this notice true. Null when the caller did not say which, and
 * an unload then clears it, because a notice that cannot be matched to a model is
 * worse left up. */
let advisedModelPath: string | null = null;

/** Take the notice down when the model it describes is unloaded.
 *
 * The toast lives 12 seconds and says "this model could run faster" beside an
 * offer to remember the dismissal for the current allocation. Both stop being
 * true the moment the model is gone, and the load path cannot clear it because no
 * load happened. */
export function dismissCarveoutAdviceForModel(modelPath?: string | null): void {
  if (advisedModelPath !== null && modelPath && modelPath !== advisedModelPath) return;
  advisedModelPath = null;
  toast.dismiss(IGPU_CARVEOUT_TOAST_ID);
}

/** Hand a load response's advice field to the notice. Safe to call on every load.
 *
 * Absent on nearly every load, so callers pass the field through unconditionally
 * and anything malformed is treated as no advice at all: the notice quotes numbers,
 * and a partial payload must produce no toast rather than one reading "undefined
 * GB". */
export function showCarveoutAdvice(value: unknown, modelPath?: string | null): void {
  const advice = parseCarveoutAdvice(value);
  if (!advice) {
    // This load has nothing to advise, so the previous load's numbers have stopped
    // being true. Same moment the store used to clear its copy.
    advisedModelPath = null;
    toast.dismiss(IGPU_CARVEOUT_TOAST_ID);
    return;
  }
  advisedModelPath = modelPath ?? null;
  toast.info(IGPU_CARVEOUT_NOTICE_TITLE, {
    id: IGPU_CARVEOUT_TOAST_ID,
    description: advice.message,
    duration: IGPU_CARVEOUT_NOTICE_DURATION_MS,
    classNames: { actionButton: IGPU_CARVEOUT_ACTION_CLASS },
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
