// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const DIFFUSION_TRAIN_GUIDE_URLS = {
  image: "https://unsloth.ai/docs/basics/fine-tune-diffusion-models-with-lora",
  video: "https://unsloth.ai/docs/basics/fine-tune-video-models-with-lora",
} as const;

export function diffusionTrainGuideUrl(
  familyName: string | null | undefined,
  families: ReadonlyArray<{ name: string; is_video?: boolean }> | null | undefined,
): string {
  const isVideo =
    Boolean(familyName) && families?.find((f) => f.name === familyName)?.is_video === true;
  return isVideo ? DIFFUSION_TRAIN_GUIDE_URLS.video : DIFFUSION_TRAIN_GUIDE_URLS.image;
}
