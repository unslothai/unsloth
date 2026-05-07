// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ChatModelSummary } from "../types/runtime";

export const IMAGE_UNSUPPORTED_MESSAGE =
  "Image provided but current model does not support vision.";
export const IMAGE_REQUIRES_VISION_MODEL_MESSAGE =
  "Select a vision model before adding images.";

export function modelSupportsImageInput(
  model: ChatModelSummary | undefined,
): boolean {
  return model?.isVision === true;
}

export function findChatModelById(
  models: ChatModelSummary[],
  id: string | null | undefined,
): ChatModelSummary | undefined {
  if (!id) return undefined;
  return models.find((model) => model.id === id);
}
