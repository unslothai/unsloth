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
  },
  allowUnknownLocalModels: boolean,
): boolean {
  return (
    allowUnknownLocalModels &&
    model.task == null &&
    model.model_format === "safetensors"
  );
}
