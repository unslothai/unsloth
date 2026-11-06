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
  // null/undefined means unknown (default-allow); external models are not in runtime.models[].
  externalSupportsVision?: boolean | null;
  externalModelLabel?: string | null;
  loadedIsMultimodal: boolean;
  modelLoaded: boolean;
  loadError?: string | null;
  // Vision was switched off by the user, not missing a projector.
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

  // loadedIsMultimodal covers vision or audio; block only when activeModel confirms audio-only.
  if (loadedIsMultimodal) {
    const isAudioOnly =
      Boolean(activeModel?.isAudio || activeModel?.hasAudioInput) &&
      activeModel?.isVision === false;
    if (!isAudioOnly) {
      return null;
    }
  }
  const label = activeModel?.name || activeModel?.id || "Current model";
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

/** Owners of the gate's running-flag pulse, which settles compare-mode waiters on refusal.
 *  Fresh per pulse: siblings share the "__default" key and clearing is by owner. */
const imageGateRunOwners = new WeakSet<() => void>();

export function createImageGateRunOwner(): () => void {
  const owner = () => {};
  imageGateRunOwners.add(owner);
  return owner;
}

/** An empty owner list is a legacy run and is taken at face value. */
export function isImageGateRunOnly(
  owners: readonly { owner: () => void }[] | undefined,
): boolean {
  return (
    owners !== undefined &&
    owners.length > 0 &&
    owners.every((entry) => imageGateRunOwners.has(entry.owner))
  );
}
