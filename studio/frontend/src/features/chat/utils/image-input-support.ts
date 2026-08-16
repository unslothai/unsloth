// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ChatModelSummary } from "../types/runtime";

export function getImageInputUnavailableReason({
  activeModel,
  isExternalModel,
  externalSupportsVision,
  externalModelLabel,
  loadedIsMultimodal,
  modelLoaded,
  loadError,
  visionDisabledByUser,
}: {
  activeModel?: ChatModelSummary;
  isExternalModel: boolean;
  // true/false = caller knows; null/undefined = unknown (default-allow).
  // External selections aren't in runtime.models[], so callers resolve
  // provider-type capability and pass it here.
  externalSupportsVision?: boolean | null;
  // Fallback toast label when activeModel is missing.
  externalModelLabel?: string | null;
  loadedIsMultimodal: boolean;
  modelLoaded: boolean;
  // Runtime lastModelLoadError; lets the no-model branch flag a failed load.
  loadError?: string | null;
  // Backend-reported: image input is off because Vision was switched off for this
  // model, not because no projector could be found.
  visionDisabledByUser?: boolean | null;
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
  // loadedIsMultimodal is true for vision OR audio; that one flag can't tell
  // them apart, so only block when activeModel confirms audio-only (audio
  // capability set AND isVision === false). Otherwise trust the load
  // response: the models-list entry may be stale or not yet injected.
  if (loadedIsMultimodal) {
    const isAudioOnly =
      Boolean(activeModel?.isAudio || activeModel?.hasAudioInput) &&
      activeModel?.isVision === false;
    if (!isAudioOnly) return null;
  }

  const label = activeModel?.name || activeModel?.id || "Current model";
  // Checked before the generic message below, which would otherwise tell someone
  // who switched Vision off to go and find a vision model with a valid mmproj --
  // sending them after a problem they do not have. The model is capable and the
  // projector is fine; the setting is the only thing in the way.
  if (visionDisabledByUser) {
    return `Vision is turned off for ${label}. Turn it back on in the model's Advanced Settings to attach images.`;
  }
  const suffix = activeModel?.isGguf
    ? " with a valid mmproj before attaching images."
    : " before attaching images.";
  return `${label} cannot accept images. Load a vision-capable model${suffix}`;
}
