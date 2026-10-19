// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// eslint-disable-next-line no-restricted-imports -- The barrel pulls in React components, which node tests cannot load.
import {
  type ModelConfigHandoffRequest,
  modelConfigTarget,
  modelConfigTargetIsResident,
} from "@/features/model-picker/model-config/model-config-handoff";

/** The resident model as Chat's runtime holds it: the same fields Chat's own reload sends. */
export type ResidentRunTarget = {
  checkpoint: string;
  ggufVariant: string | null;
  isGguf: boolean;
  isLora: boolean;
  isDiffusion: boolean;
  loadId: string | null;
  nativePathToken: string | null;
  nativePathExpiresAtMs: number | null;
};

/** Only a llama-server load reports slots to the monitor; the diffusion runner gets no --parallel. */
export function residentHasSlotsSetting(
  resident: ResidentRunTarget | null,
  nowMs: number = Date.now(),
): resident is ResidentRunTarget {
  if (!resident?.isGguf || resident.isDiffusion) {
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

/** The picker's residency test on the target the request opens. A file-picked GGUF (path vs the
 *  monitor's stem) or a quant Chat has not caught up with fails it and would open on stored settings. */
export function residentMatchesMonitor(
  resident: ResidentRunTarget | null,
  monitor: { id: string; ggufVariant: string | null },
  nowMs: number = Date.now(),
): boolean {
  const request = slotsSettingsRequest(resident, "", nowMs);
  return (
    request !== null &&
    modelConfigTargetIsResident({
      target: modelConfigTarget(request.id, request.meta),
      selectedId: monitor.id,
      activeGgufVariant: monitor.ggufVariant,
    })
  );
}
