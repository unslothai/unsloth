// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ChatModelSummary } from "../types/runtime";

import type { MmprojFallbackReason } from "../types/api";
import { isTextOnlyMmprojFallback } from "./mmproj-fallback.ts";

function textOnlyMmprojUnavailableReason(
  activeModel: ChatModelSummary | undefined,
  reason: MmprojFallbackReason | null | undefined,
): string | null {
  if (!isTextOnlyMmprojFallback(reason)) {
    return null;
  }
  const label = activeModel?.name || activeModel?.id || "This vision model";
  return `${label}'s vision projector failed to start, so Unsloth reloaded it in text-only mode. Free memory or update Unsloth, then reload the model before attaching images.`;
}

export function getImageInputUnavailableReason({
  activeModel,
  isExternalModel,
  externalSupportsVision,
  externalModelLabel,
  loadedIsMultimodal,
  modelLoaded,
  loadError,
  visionDisabledByUser,
  mmprojFallbackReason,
}: {
  activeModel?: ChatModelSummary;
  isExternalModel: boolean;
  // true/false = caller knows; null/undefined = unknown (default-allow). External selections are
  // not in runtime.models[], so callers resolve provider-type capability and pass it here.
  externalSupportsVision?: boolean | null;
  // Fallback toast label when activeModel is missing.
  externalModelLabel?: string | null;
  loadedIsMultimodal: boolean;
  modelLoaded: boolean;
  // Runtime lastModelLoadError; lets the no-model branch flag a failed load.
  loadError?: string | null;
  // Backend-reported: image input is off because Vision was switched off for this model, not
  // because no projector could be found.
  visionDisabledByUser?: boolean | null;
  mmprojFallbackReason?: MmprojFallbackReason | null;
}): string | null {
  if (isExternalModel) {
    const explicitlyNonVision =
      externalSupportsVision === false ||
      (activeModel &&
        activeModel.isVision === false &&
        !activeModel.isAudio &&
        !activeModel.hasAudioInput);
    if (explicitlyNonVision) {
      const label =
        activeModel?.name ||
        externalModelLabel ||
        activeModel?.id ||
        "Current model";
      return `${label} cannot accept images.`;
    }
    return null;
  }
  if (!modelLoaded) {
    // Distinguish a failed load from "no model picked yet".
    if (loadError) {
      return "The last model failed to load. Check the server logs, then load a model before adding images.";
    }
    return "Load a model before adding images.";
  }
  const fallbackReason = textOnlyMmprojUnavailableReason(
    activeModel,
    mmprojFallbackReason,
  );
  if (fallbackReason) {
    return fallbackReason;
  }

  // loadedIsMultimodal is true for vision OR audio and cannot tell them apart, so only block when
  // activeModel confirms audio-only (audio capability set AND isVision === false). Otherwise
  // trust the load response: the models-list entry may be stale or not yet injected.
  if (loadedIsMultimodal) {
    const isAudioOnly =
      Boolean(activeModel?.isAudio || activeModel?.hasAudioInput) &&
      activeModel?.isVision === false;
    if (!isAudioOnly) {
      return null;
    }
  }
  const label = activeModel?.name || activeModel?.id || "Current model";
  // Before the generic message below, which would otherwise send someone who switched Vision off
  // hunting for a vision model with a valid mmproj. The model is capable and the projector is
  // fine; the setting is what is in the way.
  if (visionDisabledByUser) {
    return `Vision is turned off for ${label}. Turn it back on in the model's Advanced Settings to attach images.`;
  }
  const suffix = activeModel?.isGguf
    ? " with a valid mmproj before attaching images."
    : " before attaching images.";
  return (
    fallbackReason ??
    `${label} cannot accept images. Load a vision-capable model${suffix}`
  );
}

/** The owners of the running-flag pulse the gate fires when it refuses a turn. `chat-adapter.ts`
 *  flips `runningByThreadId` on and straight back off before it throws, so compare mode's
 *  `waitForRunEnd` resolves instead of hanging on a run that never reached the streaming path.
 *  It is a settlement for waiters, not a run. A WeakSet rather than a flag on the function,
 *  because the store types an owner as a bare `() => void`. Fresh per pulse: siblings share
 *  the "__default" key and `setThreadRunning` clears by owner, so two gates firing on one key
 *  must not clear each other's entry. */
const imageGateRunOwners = new WeakSet<() => void>();

/** A run-owner token that marks its `setThreadRunning` pair as the gate's own pulse. */
export function createImageGateRunOwner(): () => void {
  const owner = () => {};
  imageGateRunOwners.add(owner);
  return owner;
}

/** Whether everything holding this thread's running flag is a gate pulse. Asked by readers that
 *  mean "is this thread generating", so a real run sharing the key still answers yes. An empty
 *  list is a run from before per-run tracking and is taken at face value. */
export function isImageGateRunOnly(
  owners: readonly { owner: () => void }[] | undefined,
): boolean {
  return (
    owners !== undefined &&
    owners.length > 0 &&
    owners.every((entry) => imageGateRunOwners.has(entry.owner))
  );
}
