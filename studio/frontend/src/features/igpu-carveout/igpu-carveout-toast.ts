// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The integrated-GPU memory advice, as a toast rather than a modal: the model has
// already loaded, nothing is being asked, and there is no decision to block on, so a
// modal would dim the app and eat the next click for a setting the user cannot change
// from here. Every other "worth knowing, carry on" notice in Studio is a toast.
//
// No store and no mounted component either: the toast owns its text for as long as it
// is up, so there is no state to keep or invalidate.

import { toast } from "@/lib/toast";

import { dismissCarveoutNotice } from "./api/igpu-carveout-notice";
import { parseCarveoutAdvice } from "./types";

/** One id for the notice, so a second load REPLACES rather than stacks: it is
 *  transient and describes the model that just loaded, so a stack of two would sit
 *  over the composer describing a model that is no longer resident. */
export const IGPU_CARVEOUT_TOAST_ID = "igpu-carveout-notice";

/** Longer than the Toaster's 5s default, like the other explanatory toasts, and
 *  longer than those because this one carries an action worth reading first. */
export const IGPU_CARVEOUT_NOTICE_DURATION_MS = 12000;

export const IGPU_CARVEOUT_NOTICE_TITLE = "This model could run faster";

/** The action, outlined and right-aligned rather than sonner's filled default.
 *
 * A solid button reads as the thing to do, and the thing to do here is nothing. And
 * the shared toast CSS puts an action at `justify-self: start`, which under six lines
 * of description floats it mid-toast, aligned to neither edge. Overridden here rather
 * than in that shared rule because every other toast's action IS the thing to do. */
export const IGPU_CARVEOUT_ACTION_CLASS =
  "!justify-self-end !h-[26px] !border !border-border !bg-transparent !px-3 " +
  "!font-medium !text-foreground hover:!bg-accent";

/** The model the notice on screen is about, so an unload can take it down.
 *
 * Paths rather than a flag: several models can be resident, and unloading another
 * leaves this notice true. More than one path, because a cached Hub candidate is
 * requested by its `loadId` while the runtime stores the checkpoint the backend
 * echoes back, and the unload is issued with the second. Empty when the caller named
 * none, and an unload then clears it: an unmatchable notice is worse left up. */
let advisedModelPaths: string[] = [];

/** Take the notice down when the model it describes is unloaded: it says "this model
 *  could run faster" beside an offer to remember the dismissal, and both stop being
 *  true the moment the model is gone, with no load happening to clear them. */
export function dismissCarveoutAdviceForModel(modelPath?: string | null): void {
  if (advisedModelPaths.length > 0 && modelPath && !advisedModelPaths.includes(modelPath)) return;
  advisedModelPaths = [];
  toast.dismiss(IGPU_CARVEOUT_TOAST_ID);
}

/** Hand a load response's advice field to the notice. Safe to call on every load:
 *  the field is absent on nearly every one, and anything malformed is treated as no
 *  advice, since the notice quotes numbers and a partial payload must produce no
 *  toast rather than one reading "undefined GB". */
export function showCarveoutAdvice(
  value: unknown,
  ...modelPaths: (string | null | undefined)[]
): void {
  const advice = parseCarveoutAdvice(value);
  if (!advice) {
    // This load has nothing to advise, so the previous load's numbers are stale.
    advisedModelPaths = [];
    toast.dismiss(IGPU_CARVEOUT_TOAST_ID);
    return;
  }
  advisedModelPaths = [...new Set(modelPaths.filter((path): path is string => !!path))];
  toast.info(IGPU_CARVEOUT_NOTICE_TITLE, {
    id: IGPU_CARVEOUT_TOAST_ID,
    description: advice.message,
    duration: IGPU_CARVEOUT_NOTICE_DURATION_MS,
    classNames: { actionButton: IGPU_CARVEOUT_ACTION_CLASS },
    action: {
      label: "Don't show again",
      // Fire and forget: dismissCarveoutNotice swallows its own failures, and the
      // worst case is the notice returning on a later load.
      onClick: () => {
        void dismissCarveoutNotice(advice.current_gb);
      },
    },
  });
}
