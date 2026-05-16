// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ChatModelSummary } from "../types/runtime";

export function getImageInputUnavailableReason({
  activeModel,
  isExternalModel,
  loadedIsMultimodal,
  modelLoaded,
}: {
  activeModel?: ChatModelSummary;
  isExternalModel: boolean;
  loadedIsMultimodal: boolean;
  modelLoaded: boolean;
}): string | null {
  // External providers: assume vision works. Only block if the catalog
  // says non-vision AND there are no audio flags either. The audio flags
  // signal the catalog is current, so an explicit isVision=false there
  // is real (Cohere chat being the usual case).
  if (isExternalModel) {
    if (
      activeModel &&
      activeModel.isVision === false &&
      !activeModel.isAudio &&
      !activeModel.hasAudioInput
    ) {
      const label = activeModel.name || activeModel.id || "Current model";
      return `${label} cannot accept images.`;
    }
    return null;
  }
  if (!modelLoaded) return "Load a model before adding images.";
  // loadedIsMultimodal is true for vision OR audio. Can't tell them apart
  // from that one flag, so only block when activeModel confirms
  // audio-only: audio capability set AND isVision === false. Otherwise
  // trust the load response. The models-list entry might be stale, or
  // not even there yet (gets auto-injected after load).
  if (loadedIsMultimodal) {
    const isAudioOnly =
      Boolean(activeModel?.isAudio || activeModel?.hasAudioInput) &&
      activeModel?.isVision === false;
    if (!isAudioOnly) return null;
  }

  const label = activeModel?.name || activeModel?.id || "Current model";
  const suffix = activeModel?.isGguf
    ? " with a valid mmproj before attaching images."
    : " before attaching images.";
  return `${label} cannot accept images. Load a vision-capable model${suffix}`;
}
