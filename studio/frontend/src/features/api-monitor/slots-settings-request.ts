// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isStandaloneGgufPath } from "@/features/hub/lib/model-identity";
import { cachedRepoConfigId } from "@/features/model-picker/model-config/model-identity";
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

/** Run can only apply settings while access to the resident model is still valid. */
export function residentHasRunSettings(
  resident: ResidentRunTarget | null,
  nowMs: number = Date.now(),
): resident is ResidentRunTarget {
  if (!resident) {
    return false;
  }
  // An expired file token cannot reload the model, so there is nothing to apply the change to.
  return !(
    resident.nativePathToken &&
    resident.nativePathExpiresAtMs != null &&
    nowMs >= resident.nativePathExpiresAtMs
  );
}

/** Only a llama-server load reports slots to the monitor; the diffusion runner gets no --parallel. */
export function residentHasSlotsSetting(
  resident: ResidentRunTarget | null,
  nowMs: number = Date.now(),
): resident is ResidentRunTarget {
  return (
    residentHasRunSettings(resident, nowMs) &&
    resident.isGguf &&
    !resident.isDiffusion
  );
}

/** Open the loaded model's settings with Advanced visible for this visit. */
export function runSettingsRequest(
  resident: ResidentRunTarget | null,
  requestId: string,
  nowMs: number = Date.now(),
): ModelConfigHandoffRequest | null {
  if (!residentHasRunSettings(resident, nowMs)) {
    return null;
  }
  return {
    requestId,
    id: resident.checkpoint,
    focus: "advanced",
    standalone: true,
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

export function keptRunSettingsRequest(
  model: { checkpoint: string; quant: string | null },
  requestId: string,
): ModelConfigHandoffRequest {
  // A standalone .gguf's quant is a filename label; saved without it (residentModelConfigTarget), else a second record.
  const standalone = isStandaloneGgufPath(model.checkpoint);
  const ggufVariant = standalone ? undefined : (model.quant ?? undefined);
  // An HF cache snapshot keeps its settings under the repo id: open that record, load the snapshot.
  const repoId = cachedRepoConfigId(model.checkpoint, ggufVariant);
  return {
    requestId,
    id: repoId ?? model.checkpoint,
    focus: "advanced",
    standalone: true,
    meta: {
      source: repoId ? "hub" : "local",
      ...(repoId ? { loadId: model.checkpoint } : {}),
      isLora: false,
      ggufVariant,
      isGguf: standalone || model.quant != null,
      isDownloaded: true,
    },
  };
}

/** Opens the resident's Run settings on Parallel Slots, so Run saves and reloads as it does there. */
export function slotsSettingsRequest(
  resident: ResidentRunTarget | null,
  requestId: string,
  nowMs: number = Date.now(),
): ModelConfigHandoffRequest | null {
  const request = runSettingsRequest(resident, requestId, nowMs);
  return request && residentHasSlotsSetting(resident, nowMs)
    ? { ...request, focus: "parallelSlots" }
    : null;
}

/** The picker's residency test on the target the request opens. A file-picked GGUF (path vs the
 *  monitor's stem) or a quant Chat has not caught up with fails it and would open on stored settings. */
export function residentMatchesMonitor(
  resident: ResidentRunTarget | null,
  monitor: { id: string; ggufVariant: string | null },
  nowMs: number = Date.now(),
): boolean {
  const request = runSettingsRequest(resident, "", nowMs);
  return (
    request !== null &&
    modelConfigTargetIsResident({
      target: modelConfigTarget(request.id, request.meta),
      selectedId: monitor.id,
      activeGgufVariant: monitor.ggufVariant,
    })
  );
}
