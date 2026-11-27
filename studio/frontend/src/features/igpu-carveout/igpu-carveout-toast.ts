// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "@/lib/toast";

import { dismissCarveoutNotice } from "./api/igpu-carveout-notice";
import { parseCarveoutAdvice } from "./types";

/** One id so a second load replaces rather than stacks the notice. */
export const IGPU_CARVEOUT_TOAST_ID = "igpu-carveout-notice";

export const IGPU_CARVEOUT_NOTICE_DURATION_MS = 12000;

export const IGPU_CARVEOUT_NOTICE_TITLE = "This model could run faster";

/** Neutral wash, not sonner's filled default; shared toast CSS floats actions, so override here. */
export const IGPU_CARVEOUT_ACTION_CLASS =
  "!justify-self-end !h-[calc(26px*var(--ui-space-scale,1))] !border-0 !px-3 !font-medium !text-foreground " +
  "!bg-[rgb(0_0_0_/_calc(0.06*var(--contrast-wash-gain,1)))] hover:!bg-[rgb(0_0_0_/_calc(0.09*var(--contrast-wash-gain,1)))] dark:!bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))] dark:hover:!bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))]";

/**
 * Hub picks are requested by loadId, unloaded by the echoed checkpoint; empty clears on any unload.
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
