// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ModelConfigHandoffRequest } from "@/features/model-picker";

/** The resident model as Chat's runtime holds it: the same fields Chat's own reload sends. */
export type ResidentRunTarget = {
  checkpoint: string;
  ggufVariant: string | null;
  isGguf: boolean;
  isMlx: boolean;
  isLora: boolean;
  isDiffusion: boolean;
  loadId: string | null;
  nativePathToken: string | null;
  nativePathExpiresAtMs: number | null;
};

/** Only GGUF and MLX loads have a Parallel Slots field; the diffusion runner gets no --parallel. */
export function residentHasSlotsSetting(
  resident: ResidentRunTarget | null,
  nowMs: number = Date.now(),
): resident is ResidentRunTarget {
  if (
    !resident ||
    resident.isDiffusion ||
    !(resident.isGguf || resident.isMlx)
  ) {
    return false;
  }
  // An expired file token cannot reload the model, so there is nothing to apply the change to.
  return !(
    resident.nativePathToken &&
    resident.nativePathExpiresAtMs != null &&
    nowMs >= resident.nativePathExpiresAtMs
  );
}

/** Opens the resident's Run settings on Parallel Slots, so Run saves and reloads as it does there. */
export function slotsSettingsRequest(
  resident: ResidentRunTarget | null,
  requestId: string,
  nowMs: number = Date.now(),
): ModelConfigHandoffRequest | null {
  if (!residentHasSlotsSetting(resident, nowMs)) {
    return null;
  }
  return {
    requestId,
    id: resident.checkpoint,
    focus: "parallelSlots",
    meta: {
      source: "local",
      isLora: resident.isLora,
      loadId: resident.loadId,
      ggufVariant: resident.ggufVariant ?? undefined,
      nativePathToken: resident.nativePathToken ?? undefined,
      nativePathExpiresAtMs: resident.nativePathExpiresAtMs,
      isGguf: resident.isGguf,
      isDiffusion: resident.isDiffusion,
      isDownloaded: true,
    },
  };
}
