// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { MmprojFallbackReason } from "../types/api";

const MMPROJ_FALLBACK_MESSAGES: Record<MmprojFallbackReason, string> = {
  // Covers both routes to the same placement: the fit estimate predicting the
  // projector will not fit in VRAM alongside the model, and a GPU startup failure
  // recovered by retrying on CPU. Worded for the outcome rather than the route,
  // since "could not start on the GPU" is untrue of the predicted case, which
  // never attempts a GPU load.
  cpu_offload:
    "Studio is running the vision projector in system memory, because it does not fit in VRAM alongside the model. Image input remains available, but image processing may be slower.",
  projector_incompatible:
    "The vision projector is incompatible with the installed llama.cpp build, so Studio reloaded this model in text-only mode. Update Studio, then reload the model to restore image input.",
  projector_startup_failure:
    "The vision projector could not start on the GPU or CPU, so Studio reloaded this model in text-only mode. Free memory or check the GPU logs, then reload the model to restore image input.",
};

export function isTextOnlyMmprojFallback(
  reason: MmprojFallbackReason | null | undefined,
): boolean {
  return (
    reason === "projector_incompatible" ||
    reason === "projector_startup_failure"
  );
}

export function mmprojFallbackMessage(reason: MmprojFallbackReason): string {
  return MMPROJ_FALLBACK_MESSAGES[reason];
}

export function mmprojLoadNotice(
  modelName: string,
  reason: MmprojFallbackReason,
): { title: string; description: string } {
  return {
    title:
      reason === "cpu_offload"
        ? `${modelName} loaded with vision on CPU`
        : `${modelName} loaded without vision`,
    description: mmprojFallbackMessage(reason),
  };
}
