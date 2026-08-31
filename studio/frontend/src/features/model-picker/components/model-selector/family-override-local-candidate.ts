// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * An explicit diffusion-family override is the missing architecture evidence for an otherwise
 * unclassified local safetensors checkpoint. Keep this deliberately narrower than task=null:
 * arbitrary local folders and unknown chat formats must not leak into media pickers.
 */
export function isFamilyOverrideLocalCandidate(
  model: {
    model_format?: string | null;
    task?: string | null;
    capabilities?: {
      canChat?: boolean;
      canTrain?: boolean;
      supportsLora?: boolean;
    } | null;
  },
  allowUnknownLocalModels: boolean,
  familyOverride?: string | null,
): boolean {
  const family = familyOverride?.trim().toLowerCase();
  const supportsSingleFile = ![
    "ideogram-4",
    "minimax-h3",
    "h3",
    "wan2.2-t2v-a14b",
  ].includes(family ?? "");
  const recoverableDiffusionGguf =
    model.model_format === "gguf" &&
    model.task === "image-diffusion-unsupported";
  const inertSafetensors =
    model.task == null &&
    model.model_format === "safetensors" &&
    model.capabilities?.canChat === false &&
    model.capabilities.canTrain === false &&
    model.capabilities.supportsLora === false;
  return (
    allowUnknownLocalModels &&
    supportsSingleFile &&
    (recoverableDiffusionGguf || inertSafetensors)
  );
}
