// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "@/lib/toast";

import { dismissCarveoutNotice } from "./api/igpu-carveout-notice";
import { parseCarveoutAdvice } from "./types";

/** One id so a second load replaces rather than stacks the notice. */
export const IGPU_CARVEOUT_TOAST_ID = "igpu-carveout-notice";

export const IGPU_CARVEOUT_NOTICE_DURATION_MS = 12000;

export const IGPU_CARVEOUT_NOTICE_TITLE = "This model could run faster";

/** Outlined and right-aligned: the shared toast CSS floats actions mid-toast at justify-self: start. */
export const IGPU_CARVEOUT_ACTION_CLASS =
  "!justify-self-end !h-[calc(26px*var(--ui-space-scale,1))] !border !border-border !bg-transparent !px-3 " +
  "!font-medium !text-foreground hover:!bg-accent";

/**
 * Paths of the model the notice describes: a Hub pick is requested by `loadId` but unloaded by
 * the echoed checkpoint. Empty means any unload clears it.
 */
let advisedModelPaths: string[] = [];

export function dismissCarveoutAdviceForModel(modelPath?: string | null): void {
  if (advisedModelPaths.length > 0 && modelPath && !advisedModelPaths.includes(modelPath)) return;
  advisedModelPaths = [];
  toast.dismiss(IGPU_CARVEOUT_TOAST_ID);
}

/** Malformed advice yields no toast rather than one reading "undefined GB". */
export function showCarveoutAdvice(
  value: unknown,
  ...modelPaths: (string | null | undefined)[]
): void {
  const advice = parseCarveoutAdvice(value);
  if (!advice) {
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
      onClick: () => {
        void dismissCarveoutNotice(advice.current_gb);
      },
    },
  });
}
