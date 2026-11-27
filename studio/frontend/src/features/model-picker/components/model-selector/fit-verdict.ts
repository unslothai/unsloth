// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { PickerFitStatus } from "./media-picker-fit";

export interface FitVerdict {
  label: string;
  tone: string;
  hint: string;
}

export const AMBER = "!text-yellow-600 dark:!text-yellow-400";
export const ORANGE = "!text-orange-600 dark:!text-orange-300";

/** Over the VRAM Budget but under the card: `_select_gpus` hands it to --fit every time. */
export const MIGHT_FIT: FitVerdict = {
  label: "Over budget",
  tone: AMBER,
  hint: "Larger than your VRAM Budget allows, so part of it offloads even on an idle GPU. It is still smaller than the card, so raising the budget can keep it resident.",
};
/** Past the card; llama-server never refuses a GGUF on size, --fit offloads. */
export const OFFLOADS: FitVerdict = {
  label: "Does not fit",
  tone: ORANGE,
  hint: "Model may not fit but still works with offloading. Expect slower inference.",
};
/** checkVramFit's 75-100% band, the only source of `tight`: fits on the card with no --fit. */
export const DEVICE_TIGHT: FitVerdict = {
  label: "Tight fit",
  tone: AMBER,
  hint: "Uses nearly all your VRAM, with little headroom for anything else.",
};
/** A torch load has no --fit: the pipeline goes wholly on the device. */
export const WONT_FIT: FitVerdict = {
  label: "Does not fit",
  tone: ORANGE,
  // "memory", not "VRAM": this also covers diffusion refusals on a shared pool.
  hint: "Estimated memory exceeds this device's usable capacity. Try a smaller model or free memory.",
};

/** Fit verdict marks keyed by the Hub's classes; `tight`/`exceeds` come only from torch or
 *  QLoRA estimates, which never offload. */
export const VRAM_VERDICT: Record<PickerFitStatus, FitVerdict | null> = {
  fits: null,
  offload: {
    label: "RAM offloading",
    tone: AMBER,
    hint: "Might be slower with RAM offloading.",
  },
  marginal: MIGHT_FIT,
  tight: DEVICE_TIGHT,
  partial: OFFLOADS,
  ram: null,
  streamed: {
    label: "Fits (lookup tables streamed from disk)",
    tone: AMBER,
    hint: "Fits with its lookup tables streamed from disk, llama.cpp's default for them. Prefill and decode can be slower and vary with disk speed.",
  },
  oom: OFFLOADS,
  exceeds: WONT_FIT,
};

/** `marginal` is not over budget: it is a full GPU load with little room to spare. */
export function isOverBudget(status?: PickerFitStatus | null): boolean {
  return (
    status === "partial" ||
    status === "ram" ||
    status === "oom" ||
    status === "exceeds"
  );
}
